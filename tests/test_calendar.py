"""Tests for data.calendar module.

Covers: abstract interface, MT5 backend (mocked), ForexFactory backend (mocked),
EconCalendar orchestrator, tag_dataframe, anti-lookahead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.calendar.base import (
    CalendarBackendUnavailableError,
    CalendarScrapingError,
    EconCalendarBackend,
    EconCalendarError,
)
from data.calendar.econ_calendar import EconCalendar
from data.calendar.forexfactory_backend import ForexFactoryBackend
from data.calendar.mt5_backend import MT5CalendarBackend, _matches_high_impact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events_df(
    timestamps: list[datetime],
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Create a synthetic events DataFrame."""
    n = len(timestamps)
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, utc=True),
        "country": ["US"] * n,
        "event_name": names or [f"Event_{i}" for i in range(n)],
        "impact": ["high"] * n,
        "actual": [None] * n,
        "forecast": [None] * n,
        "previous": [None] * n,
    })


def _make_ohlcv(
    start: str = "2024-03-04 08:00",
    periods: int = 96,
    freq: str = "15min",
) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": 2100 + rng.normal(0, 1, periods).cumsum(),
            "high": 2105 + rng.normal(0, 1, periods).cumsum(),
            "low": 2095 + rng.normal(0, 1, periods).cumsum(),
            "close": 2100 + rng.normal(0, 1, periods).cumsum(),
            "volume": rng.uniform(100, 1000, periods),
        },
        index=idx,
    )


class _MockBackend(EconCalendarBackend):
    """Concrete mock backend for testing."""

    def __init__(
        self,
        available: bool = True,
        events: pd.DataFrame | None = None,
        raise_on_fetch: Exception | None = None,
    ) -> None:
        self._available = available
        self._events = events if events is not None else _make_events_df([])
        self._raise_on_fetch = raise_on_fetch
        self.fetch_call_count = 0

    def is_available(self) -> bool:
        return self._available

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        self.fetch_call_count += 1
        if self._raise_on_fetch:
            raise self._raise_on_fetch
        return self._events


# ---------------------------------------------------------------------------
# Test abstract interface
# ---------------------------------------------------------------------------


class TestBaseBackend:
    """Tests for abstract base."""

    def test_base_backend_is_abstract(self) -> None:
        """Cannot instantiate EconCalendarBackend directly."""
        with pytest.raises(TypeError, match="abstract"):
            EconCalendarBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MT5 Backend (mocked)
# ---------------------------------------------------------------------------


class TestMT5Backend:
    """Tests for MT5CalendarBackend with mocked MT5."""

    def test_mt5_backend_is_available_when_mt5_initialized(self) -> None:
        """When MT5 responds, backend is available."""
        backend = MT5CalendarBackend()
        with patch.dict("sys.modules", {"MetaTrader5": MagicMock()}):
            import sys
            mt5_mock = sys.modules["MetaTrader5"]
            mt5_mock.terminal_info.return_value = True
            mt5_mock.calendar_event_get.return_value = []
            with patch("data.calendar.mt5_backend.mt5", mt5_mock, create=True):
                # Patch the import inside is_available
                with patch.dict("sys.modules", {"MetaTrader5": mt5_mock}):
                    assert backend.is_available() is True

    def test_mt5_backend_is_available_when_mt5_down(self) -> None:
        """When MT5 not responding, backend is unavailable."""
        backend = MT5CalendarBackend()
        with patch.dict("sys.modules", {"MetaTrader5": MagicMock()}):
            import sys
            mt5_mock = sys.modules["MetaTrader5"]
            mt5_mock.terminal_info.return_value = None
            with patch.dict("sys.modules", {"MetaTrader5": mt5_mock}):
                assert backend.is_available() is False

    def test_mt5_backend_fetch_events_basic(self) -> None:
        """Mock event_get + value_history_get → correct DataFrame."""
        backend = MT5CalendarBackend()

        # Mock MT5 event object
        event_obj = MagicMock()
        event_obj.name = "Non-Farm Payrolls"
        event_obj.id = 1

        # Mock value object
        val_obj = MagicMock()
        val_obj.time = int(datetime(2024, 3, 8, 13, 30, tzinfo=UTC).timestamp())
        val_obj.actual_value = 275.0
        val_obj.forecast_value = 200.0
        val_obj.previous_value = 229.0

        mt5_mock = MagicMock()
        mt5_mock.terminal_info.return_value = True
        mt5_mock.calendar_event_get.return_value = [event_obj]
        mt5_mock.calendar_value_history_get.return_value = [val_obj]

        with patch.dict("sys.modules", {"MetaTrader5": mt5_mock}):
            result = backend.fetch_events(
                datetime(2024, 3, 1, tzinfo=UTC),
                datetime(2024, 3, 31, tzinfo=UTC),
            )

        assert len(result) == 1
        assert result["event_name"].iloc[0] == "Non-Farm Payrolls"
        assert result["actual"].iloc[0] == 275.0

    def test_mt5_backend_filters_high_impact(self) -> None:
        """Only high-impact events are returned when impact=['high']."""
        # "Farm Equipment Index" should NOT match
        assert _matches_high_impact("Non-Farm Payrolls") is True
        assert _matches_high_impact("Nonfarm Payrolls") is True
        assert _matches_high_impact("Farm Equipment Index") is False
        assert _matches_high_impact("CPI m/m") is True
        assert _matches_high_impact("Consumer Price Index") is True

    def test_mt5_backend_handles_event_name_variations(self) -> None:
        """Various NFP name formats all match."""
        assert _matches_high_impact("Non-Farm Payrolls") is True
        assert _matches_high_impact("Nonfarm Payrolls") is True
        assert _matches_high_impact("Non Farm Employment Change") is True
        assert _matches_high_impact("non-farm employment change") is True

    def test_mt5_backend_returns_utc_timestamps(self) -> None:
        """Output timestamps must be tz-aware UTC."""
        backend = MT5CalendarBackend()

        event_obj = MagicMock()
        event_obj.name = "FOMC Statement"
        event_obj.id = 2

        val_obj = MagicMock()
        val_obj.time = int(datetime(2024, 3, 20, 18, 0, tzinfo=UTC).timestamp())
        val_obj.actual_value = 0
        val_obj.forecast_value = 0
        val_obj.previous_value = 0

        mt5_mock = MagicMock()
        mt5_mock.terminal_info.return_value = True
        mt5_mock.calendar_event_get.return_value = [event_obj]
        mt5_mock.calendar_value_history_get.return_value = [val_obj]

        with patch.dict("sys.modules", {"MetaTrader5": mt5_mock}):
            result = backend.fetch_events(
                datetime(2024, 3, 1, tzinfo=UTC),
                datetime(2024, 3, 31, tzinfo=UTC),
            )

        assert result["timestamp_utc"].dt.tz is not None


# ---------------------------------------------------------------------------
# ForexFactory Backend (mocked HTTP)
# ---------------------------------------------------------------------------


_SAMPLE_FF_HTML = """
<html><body>
<table class="calendar__table">
<tr class="calendar__row">
  <td class="calendar__date">Mon Mar 4</td>
  <td class="calendar__time">8:30am</td>
  <td class="calendar__impact"><span class="icon--ff-impact-red"></span></td>
  <td class="calendar__currency">USD</td>
  <td class="calendar__event">Non-Farm Payrolls</td>
  <td class="calendar__actual">275K</td>
  <td class="calendar__forecast">200K</td>
  <td class="calendar__previous">229K</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__date"></td>
  <td class="calendar__time">10:00am</td>
  <td class="calendar__impact"><span class="icon--ff-impact-ora"></span></td>
  <td class="calendar__currency">USD</td>
  <td class="calendar__event">ISM Manufacturing PMI</td>
  <td class="calendar__actual">52.3</td>
  <td class="calendar__forecast">51.5</td>
  <td class="calendar__previous">50.8</td>
</tr>
</table>
</body></html>
"""


class TestForexFactoryBackend:
    """Tests for ForexFactory scraping backend."""

    def test_ff_backend_is_available(self) -> None:
        """Mock GET OK → available."""
        import httpx as httpx_mod

        backend = ForexFactoryBackend()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(httpx_mod, "head", return_value=mock_resp):
            assert backend.is_available() is True

    def test_ff_backend_is_unavailable_on_timeout(self) -> None:
        """Mock timeout → unavailable."""
        import httpx as httpx_mod

        backend = ForexFactoryBackend()
        with patch.object(
            httpx_mod, "head", side_effect=httpx_mod.TimeoutException("timeout")
        ):
            assert backend.is_available() is False

    def test_ff_backend_scraping_parses_correctly(self) -> None:
        """Mock HTML fixture → events extracted correctly."""
        backend = ForexFactoryBackend(cache_dir=Path("/tmp/test_ff_cache"))

        result = backend._parse_html(_SAMPLE_FF_HTML, 2024, 10)

        assert len(result) == 2
        assert result["event_name"].iloc[0] == "Non-Farm Payrolls"
        assert result["country"].iloc[0] == "US"
        assert result["impact"].iloc[0] == "high"
        assert result["impact"].iloc[1] == "medium"

    def test_ff_backend_cache_write_and_read(self, tmp_path: Path) -> None:
        """Fetch then re-fetch → 2nd uses cache (1 HTTP call)."""
        backend = ForexFactoryBackend(cache_dir=tmp_path, cache_ttl_hours=24)

        with patch.object(backend, "_scrape_week") as mock_scrape:
            mock_scrape.return_value = _make_events_df(
                [datetime(2024, 3, 4, 13, 30, tzinfo=UTC)]
            )

            # First call: scrapes
            backend._fetch_week(2024, 10)
            assert mock_scrape.call_count == 1

            # Second call: cache hit
            backend._fetch_week(2024, 10)
            assert mock_scrape.call_count == 1  # No additional call

    def test_ff_backend_cache_ttl_expiry(self, tmp_path: Path) -> None:
        """Cache expired → re-fetch."""
        import os

        backend = ForexFactoryBackend(cache_dir=tmp_path, cache_ttl_hours=1)

        with patch.object(backend, "_scrape_week") as mock_scrape:
            events = _make_events_df([datetime(2024, 3, 4, 13, 30, tzinfo=UTC)])
            mock_scrape.return_value = events

            # First call: scrapes and caches
            backend._fetch_week(2024, 10)
            assert mock_scrape.call_count == 1

            # Backdate the cache file by 2 hours to simulate expiry
            cache_file = tmp_path / "ff_week_2024-W10.parquet"
            old_time = os.path.getmtime(cache_file) - 7200  # 2h ago
            os.utime(cache_file, (old_time, old_time))

            # Second call: cache expired (age > 1h TTL) → re-scrapes
            backend._fetch_week(2024, 10)
            assert mock_scrape.call_count == 2

    def test_ff_backend_handles_html_change(self) -> None:
        """Unexpected HTML → CalendarScrapingError with clear message."""
        backend = ForexFactoryBackend()
        bad_html = "<html><body><p>Page redesigned!</p></body></html>"
        with pytest.raises(CalendarScrapingError, match="Cannot find calendar table"):
            backend._parse_html(bad_html, 2024, 10)

    def test_ff_backend_timezone_conversion(self) -> None:
        """FF displays EST, output must be UTC."""
        backend = ForexFactoryBackend()
        result = backend._parse_html(_SAMPLE_FF_HTML, 2024, 10)
        # 8:30am EST on Mar 4 2024 = 13:30 UTC
        ts = result["timestamp_utc"].iloc[0]
        assert ts.hour == 13
        assert ts.minute == 30
        assert str(ts.tzinfo) == "UTC"


# ---------------------------------------------------------------------------
# EconCalendar Orchestrator
# ---------------------------------------------------------------------------


class TestEconCalendar:
    """Tests for the orchestrator."""

    def test_calendar_uses_primary_when_available(self) -> None:
        """Primary OK → primary called, not fallback."""
        events = _make_events_df([datetime(2024, 3, 8, 13, 30, tzinfo=UTC)])
        primary = _MockBackend(available=True, events=events)
        fallback = _MockBackend(available=True, events=_make_events_df([]))

        cal = EconCalendar(primary_backend=primary, fallback_backend=fallback)
        result = cal.fetch_events(
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 3, 31, tzinfo=UTC),
        )

        assert len(result) == 1
        assert primary.fetch_call_count == 1
        assert fallback.fetch_call_count == 0

    def test_calendar_falls_back_when_primary_fails(self) -> None:
        """Primary raises → fallback called."""
        fallback_events = _make_events_df([datetime(2024, 3, 8, 13, 30, tzinfo=UTC)])
        primary = _MockBackend(
            available=True,
            raise_on_fetch=CalendarBackendUnavailableError("MT5 down"),
        )
        fallback = _MockBackend(available=True, events=fallback_events)

        cal = EconCalendar(primary_backend=primary, fallback_backend=fallback)
        result = cal.fetch_events(
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 3, 31, tzinfo=UTC),
        )

        assert len(result) == 1
        assert primary.fetch_call_count == 1
        assert fallback.fetch_call_count == 1

    def test_calendar_raises_when_both_fail(self) -> None:
        """Both backends raise → EconCalendarError."""
        primary = _MockBackend(
            available=True,
            raise_on_fetch=CalendarBackendUnavailableError("MT5 down"),
        )
        fallback = _MockBackend(
            available=True,
            raise_on_fetch=CalendarScrapingError("HTML changed"),
        )

        cal = EconCalendar(primary_backend=primary, fallback_backend=fallback)
        with pytest.raises(EconCalendarError, match="Both calendar backends failed"):
            cal.fetch_events(
                datetime(2024, 3, 1, tzinfo=UTC),
                datetime(2024, 3, 31, tzinfo=UTC),
            )

    def test_calendar_disabled_mode_no_backends(self) -> None:
        """primary=None, fallback=None → all False + warning."""
        cal = EconCalendar(primary_backend=None, fallback_backend=None)
        df = _make_ohlcv(periods=50)
        result = cal.tag_dataframe(df)
        assert "is_news_window" in result.columns
        assert not result["is_news_window"].any()


# ---------------------------------------------------------------------------
# is_in_news_window
# ---------------------------------------------------------------------------


class TestIsInNewsWindow:
    """Tests for is_in_news_window."""

    def _make_calendar_with_event(self, event_time: datetime) -> EconCalendar:
        events = _make_events_df([event_time], ["NFP"])
        backend = _MockBackend(available=True, events=events)
        return EconCalendar(primary_backend=backend)

    def test_is_in_news_window_basic(self) -> None:
        """Event at 14:00, query at 13:50 with pre=15min → True."""
        event_time = datetime(2024, 3, 8, 14, 0, tzinfo=UTC)
        cal = self._make_calendar_with_event(event_time)
        assert cal.is_in_news_window(
            datetime(2024, 3, 8, 13, 50, tzinfo=UTC), pre_minutes=15
        ) is True

    def test_is_in_news_window_pre_boundary(self) -> None:
        """Query at exactly event_time - pre → True (inclusive)."""
        event_time = datetime(2024, 3, 8, 14, 0, tzinfo=UTC)
        cal = self._make_calendar_with_event(event_time)
        assert cal.is_in_news_window(
            datetime(2024, 3, 8, 13, 45, tzinfo=UTC), pre_minutes=15
        ) is True

    def test_is_in_news_window_post_boundary(self) -> None:
        """Query at exactly event_time + post → True (inclusive)."""
        event_time = datetime(2024, 3, 8, 14, 0, tzinfo=UTC)
        cal = self._make_calendar_with_event(event_time)
        assert cal.is_in_news_window(
            datetime(2024, 3, 8, 14, 30, tzinfo=UTC), post_minutes=30
        ) is True

    def test_is_in_news_window_outside(self) -> None:
        """Query 1 min before window → False."""
        event_time = datetime(2024, 3, 8, 14, 0, tzinfo=UTC)
        cal = self._make_calendar_with_event(event_time)
        assert cal.is_in_news_window(
            datetime(2024, 3, 8, 13, 44, tzinfo=UTC), pre_minutes=15
        ) is False

    def test_is_in_news_window_uses_provided_events(self) -> None:
        """Pass events_df → no internal fetch."""
        events = _make_events_df(
            [datetime(2024, 3, 8, 14, 0, tzinfo=UTC)], ["NFP"]
        )
        # Backend that would raise if called
        backend = _MockBackend(
            available=True,
            raise_on_fetch=RuntimeError("Should not be called"),
        )
        cal = EconCalendar(primary_backend=backend)
        result = cal.is_in_news_window(
            datetime(2024, 3, 8, 13, 50, tzinfo=UTC),
            pre_minutes=15,
            events_df=events,
        )
        assert result is True
        assert backend.fetch_call_count == 0


# ---------------------------------------------------------------------------
# tag_dataframe
# ---------------------------------------------------------------------------


class TestTagDataframe:
    """Tests for tag_dataframe."""

    def test_tag_dataframe_basic(self) -> None:
        """1 event in middle → bars in window tagged True."""
        # Event at 12:00 UTC, pre=15min, post=30min → window 11:45-12:30
        event_time = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)
        events = _make_events_df([event_time], ["NFP"])
        backend = _MockBackend(available=True, events=events)
        cal = EconCalendar(primary_backend=backend)

        # DataFrame from 08:00 to 16:00 (M15)
        df = _make_ohlcv(start="2024-03-04 08:00", periods=32, freq="15min")
        result = cal.tag_dataframe(df, pre_minutes=15, post_minutes=30)

        # Bars at 11:45, 12:00, 12:15, 12:30 should be True
        tagged_times = result.index[result["is_news_window"]]
        assert pd.Timestamp("2024-03-04 11:45", tz="UTC") in tagged_times
        assert pd.Timestamp("2024-03-04 12:00", tz="UTC") in tagged_times
        assert pd.Timestamp("2024-03-04 12:15", tz="UTC") in tagged_times
        # 12:30 is boundary inclusive
        assert pd.Timestamp("2024-03-04 12:30", tz="UTC") in tagged_times
        # 11:30 is outside (>15min before)
        assert pd.Timestamp("2024-03-04 11:30", tz="UTC") not in tagged_times

    def test_tag_dataframe_returns_copy(self) -> None:
        """Original df is not modified."""
        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        df = _make_ohlcv(periods=20)
        original_cols = list(df.columns)
        _ = cal.tag_dataframe(df)
        assert list(df.columns) == original_cols

    def test_tag_dataframe_no_events_in_range(self) -> None:
        """No events → all False, no error."""
        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        df = _make_ohlcv(periods=50)
        result = cal.tag_dataframe(df)
        assert not result["is_news_window"].any()

    def test_tag_dataframe_column_name_custom(self) -> None:
        """column_name='my_news' is respected."""
        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        df = _make_ohlcv(periods=20)
        result = cal.tag_dataframe(df, column_name="my_news")
        assert "my_news" in result.columns
        assert "is_news_window" not in result.columns

    def test_tag_dataframe_existing_column_raises(self) -> None:
        """Column already present → ValueError."""
        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        df = _make_ohlcv(periods=20)
        df["is_news_window"] = False
        with pytest.raises(ValueError, match="already contains column"):
            cal.tag_dataframe(df)

    def test_tag_dataframe_empty_df(self) -> None:
        """Empty df → empty df returned."""
        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        df = pd.DataFrame({"open": []}, index=idx)
        result = cal.tag_dataframe(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests."""

    def test_calendar_events_save_load_parquet(self, tmp_path: Path) -> None:
        """Events saved in Parquet reload identically."""
        events = _make_events_df(
            [datetime(2024, 3, 8, 13, 30, tzinfo=UTC)],
            ["Non-Farm Payrolls"],
        )
        path = tmp_path / "events.parquet"
        events.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(events, loaded)

    def test_tagged_df_passes_validation(self) -> None:
        """df + time features + news tag → validator passes."""
        from data.enrichment.time_features import enrich_time_features

        df = _make_ohlcv(periods=50)
        enriched = enrich_time_features(df)

        backend = _MockBackend(available=True, events=_make_events_df([]))
        cal = EconCalendar(primary_backend=backend)
        tagged = cal.tag_dataframe(enriched)

        # Verify column exists and is bool
        assert tagged["is_news_window"].dtype == bool


# ---------------------------------------------------------------------------
# Anti-lookahead
# ---------------------------------------------------------------------------


class TestAntiLookahead:
    """Causality tests for news tagging."""

    def test_tag_dataframe_is_causal(self) -> None:
        """Tag at time t depends only on known event schedule, not future bars."""
        from tests.test_no_lookahead import assert_function_is_causal

        # Create event at 12:00 on the same day as our data
        event_time = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)
        events = _make_events_df([event_time], ["NFP"])
        backend = _MockBackend(available=True, events=events)
        cal = EconCalendar(primary_backend=backend)

        df = _make_ohlcv(start="2024-03-04 08:00", periods=96, freq="15min")

        def tag_func(input_df: pd.DataFrame) -> pd.DataFrame:
            return cal.tag_dataframe(input_df)

        assert_function_is_causal(
            func=tag_func,
            df=df,
            added_columns=["is_news_window"],
        )
