"""Tests for MT5 data loader.

Since MT5 terminal is not available in CI/test environments, all tests use mocks.
Run `scripts/test_mt5_loader.py` manually on Windows with MT5 running for integration test.
"""

from datetime import UTC, datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from data.ingestion.mt5_loader import (
    MT5ConnectionError,
    MT5DataError,
    MT5Loader,
)

_RATES_DTYPE = np.dtype([
    ("time", "i8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("tick_volume", "i8"),
    ("spread", "i4"),
    ("real_volume", "i8"),
])


def _make_fake_rates(n: int, start_ts: int = 1700000000, interval: int = 900) -> np.ndarray:
    """Create a fake MT5 rates structured array (mimics mt5.copy_rates_range output).

    Args:
        n: Number of bars
        start_ts: Starting Unix timestamp
        interval: Seconds between bars (900 = M15)
    """
    rates = np.zeros(n, dtype=_RATES_DTYPE)
    for i in range(n):
        t = start_ts + i * interval
        o = 1950.0 + i * 0.1
        h = o + 0.5
        l = o - 0.3  # noqa: E741
        c = o + 0.2
        rates[i] = (t, o, h, l, c, 100 + i, 20, 0)
    return rates


def _make_fake_rates_weekdays_only(
    n: int, start_ts: int = 1700000000, interval: int = 900
) -> np.ndarray:
    """Create fake rates skipping Saturday and early Sunday (realistic for forex).

    Generates bars only on Mon-Fri and Sunday >= 21:00 UTC.
    """
    rates_list = []
    t = start_ts
    i = 0
    while len(rates_list) < n:
        dt = datetime.fromtimestamp(t, tz=UTC)
        weekday = dt.weekday()
        # Skip: Saturday (5) entirely, Sunday (6) before 21:00 UTC
        if weekday == 5 or (weekday == 6 and dt.hour < 21):
            t += interval
            continue
        o = 1950.0 + i * 0.1
        h = o + 0.5
        l = o - 0.3  # noqa: E741
        c = o + 0.2
        rates_list.append((t, o, h, l, c, 100 + i, 20, 0))
        t += interval
        i += 1
    return np.array(rates_list, dtype=_RATES_DTYPE)


class TestMT5LoaderConnection:
    """Test connection handling."""

    @patch("data.ingestion.mt5_loader.mt5")
    def test_connect_success(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="ICMarkets", build=4000)

        loader = MT5Loader(path="C:/MT5/terminal64.exe")
        loader.connect()

        assert loader.is_connected()
        mock_mt5.initialize.assert_called_once()

    @patch("data.ingestion.mt5_loader.mt5")
    def test_connect_failure(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (-1, "Terminal not found")

        loader = MT5Loader()
        with pytest.raises(MT5ConnectionError, match="initialization failed"):
            loader.connect()

        assert not loader.is_connected()

    @patch("data.ingestion.mt5_loader.mt5")
    def test_disconnect(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="Test", build=1)

        loader = MT5Loader()
        loader.connect()
        loader.disconnect()

        assert not loader.is_connected()
        mock_mt5.shutdown.assert_called_once()

    @patch("data.ingestion.mt5_loader.mt5")
    def test_context_manager(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="Test", build=1)

        with MT5Loader() as loader:
            assert loader.is_connected()
        assert not loader.is_connected()

    def test_connect_without_mt5_package(self) -> None:
        """Test graceful error when MetaTrader5 package is missing."""
        with patch("data.ingestion.mt5_loader.mt5", None):
            loader = MT5Loader()
            with pytest.raises(MT5ConnectionError, match="not available"):
                loader.connect()


class TestMT5LoaderDownload:
    """Test OHLCV download logic."""

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_basic(self, mock_mt5: MagicMock) -> None:
        """Test basic download returns correct DataFrame structure."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)

        fake_rates = _make_fake_rates(100)
        mock_mt5.copy_rates_range.return_value = fake_rates

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("XAUUSD", "M15", start, end)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "open", "high", "low", "close", "tick_volume", "spread", "real_volume"
        ]
        assert df.index.name == "timestamp_utc"
        assert df.index.tz is not None  # UTC-aware
        assert len(df) == 100

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_not_connected_raises(self, mock_mt5: MagicMock) -> None:
        loader = MT5Loader()
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(MT5ConnectionError, match="Not connected"):
            loader.download_ohlcv("XAUUSD", "M15", start, end)

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_unknown_symbol(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = None

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(MT5DataError, match="not found"):
            loader.download_ohlcv("FAKESYMBOL", "M15", start, end)

    @patch("data.ingestion.mt5_loader.mt5")
    def test_download_unknown_timeframe(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(MT5DataError, match="Unknown timeframe"):
            loader.download_ohlcv("XAUUSD", "M99", start, end)

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_no_data_raises(self, mock_mt5: MagicMock) -> None:
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_mt5.copy_rates_range.return_value = None
        mock_mt5.last_error.return_value = (0, "No data")

        loader = MT5Loader()
        loader.connect()

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 2, 1, tzinfo=UTC)

        with pytest.raises(MT5DataError, match="No data returned"):
            loader.download_ohlcv("XAUUSD", "M15", start, end)

    @patch("data.ingestion.mt5_loader.mt5")
    @patch(
        "data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15}
    )
    @patch("data.ingestion.mt5_loader._MAX_BARS_PER_REQUEST", 50)
    def test_download_pagination(self, mock_mt5: MagicMock) -> None:
        """Test that large downloads are paginated correctly."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)

        # First call returns 50 bars (max), second returns 30 (less than max = done)
        chunk1 = _make_fake_rates(50, start_ts=1700045000, interval=900)
        chunk2 = _make_fake_rates(30, start_ts=1700000000, interval=900)
        mock_mt5.copy_rates_range.side_effect = [chunk1, chunk2]

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("XAUUSD", "M15", start, end)

        # Should have 2 calls to copy_rates_range
        assert mock_mt5.copy_rates_range.call_count == 2
        # Total bars = deduplicated merge of 50 + 30
        assert len(df) <= 80

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_dataframe_ohlc_consistency(self, mock_mt5: MagicMock) -> None:
        """Verify high >= max(open,close) and low <= min(open,close)."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(50)

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("XAUUSD", "M15", start, end)

        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_no_duplicate_timestamps(self, mock_mt5: MagicMock) -> None:
        """Verify deduplication works."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(100)

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("XAUUSD", "M15", start, end)

        assert not df.index.duplicated().any()

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_partial_data_warns(self, mock_mt5: MagicMock) -> None:
        """Warn when broker returns less history than requested."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)

        # Return only ~2 weeks of weekday-only data starting Nov 6 (Monday),
        # even though we request a full year starting Jan 1
        fake_rates = _make_fake_rates_weekdays_only(
            n=1000,
            start_ts=int(datetime(2023, 11, 6, tzinfo=UTC).timestamp()),
            interval=900,
        )
        mock_mt5.copy_rates_range.return_value = fake_rates

        # WHY: loguru doesn't integrate with pytest caplog.
        # We add a temporary StringIO sink to capture warnings.
        log_output = StringIO()
        sink_id = logger.add(log_output, level="WARNING", format="{message}")

        try:
            loader = MT5Loader()
            loader.connect()

            start = datetime(2023, 1, 1, tzinfo=UTC)
            end = datetime(2023, 12, 1, tzinfo=UTC)
            df = loader.download_ohlcv("XAUUSD", "M15", start, end)

            assert len(df) == 1000

            log_text = log_output.getvalue()
            assert "PARTIAL DATA" in log_text
            assert "2023-01-01" in log_text
        finally:
            logger.remove(sink_id)

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_download_uses_symbol_mapping(self, mock_mt5: MagicMock) -> None:
        """Verify that mt5_symbol_map resolves canonical to broker name."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(50)

        # Map XAUUSD → GOLD
        loader = MT5Loader(symbol_map={"XAUUSD": "GOLD"})
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        loader.download_ohlcv("XAUUSD", "M15", start, end)

        # Verify that copy_rates_range was called with "GOLD", not "XAUUSD"
        call_args = mock_mt5.copy_rates_range.call_args
        assert call_args[0][0] == "GOLD"

        # Also verify symbol_info was called with "GOLD"
        mock_mt5.symbol_info.assert_called_with("GOLD")

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_timezone_sanity_detects_sunday_bars(self, mock_mt5: MagicMock) -> None:
        """Detect broker timezone leak via Sunday bars."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)

        # Create bars that include a Sunday at 12:00 UTC (impossible for XAU)
        # Sunday 2023-11-05 12:00 UTC = timestamp 1699185600
        sunday_noon_ts = int(datetime(2023, 11, 5, 12, 0, tzinfo=UTC).timestamp())
        fake_rates = _make_fake_rates(10, start_ts=sunday_noon_ts, interval=900)
        mock_mt5.copy_rates_range.return_value = fake_rates

        loader = MT5Loader()
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)

        with pytest.raises(MT5DataError, match="Timezone sanity check failed"):
            loader.download_ohlcv("XAUUSD", "M15", start, end)

    @patch("data.ingestion.mt5_loader.mt5")
    @patch("data.ingestion.mt5_loader.TIMEFRAME_MAP", {"M15": 15})
    def test_timezone_sanity_skips_crypto(self, mock_mt5: MagicMock) -> None:
        """Crypto symbols skip the Sunday check (24/7 market)."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(name="T", build=1)
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)

        # Sunday bars for BTCUSDT should NOT raise
        sunday_noon_ts = int(datetime(2023, 11, 5, 12, 0, tzinfo=UTC).timestamp())
        fake_rates = _make_fake_rates(10, start_ts=sunday_noon_ts, interval=900)
        mock_mt5.copy_rates_range.return_value = fake_rates

        loader = MT5Loader(crypto_symbols=["BTCUSDT"])
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)

        # Should not raise — crypto trades on Sundays
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)
        assert len(df) == 10
