"""Tests for data.enrichment.time_features module.

Covers: input validation, session/killzone detection, DST transitions,
XAU market hours, Friday management, meta features, edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.enrichment.time_features import _ADDED_COLUMNS, enrich_time_features

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_df(
    start: str = "2024-03-04 00:00",
    periods: int = 100,
    freq: str = "15min",
) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame with UTC DatetimeIndex."""
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "open": 2100 + rng.normal(0, 1, periods).cumsum(),
            "high": 2105 + rng.normal(0, 1, periods).cumsum(),
            "low": 2095 + rng.normal(0, 1, periods).cumsum(),
            "close": 2100 + rng.normal(0, 1, periods).cumsum(),
            "volume": rng.uniform(100, 1000, periods),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


def _make_single_bar(dt_utc: str) -> pd.DataFrame:
    """Create a single-bar DataFrame at given UTC datetime string."""
    idx = pd.DatetimeIndex([pd.Timestamp(dt_utc, tz="UTC")])
    return pd.DataFrame(
        {"open": [2100], "high": [2105], "low": [2095], "close": [2102], "volume": [500]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation in enrich_time_features."""

    def test_raises_type_error_if_not_datetime_index(self) -> None:
        df = pd.DataFrame({"open": [1, 2], "close": [1, 2]}, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            enrich_time_features(df)

    def test_raises_value_error_if_naive_tz(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame({"open": range(5)}, index=idx)
        with pytest.raises(ValueError, match="tz-aware UTC"):
            enrich_time_features(df)

    def test_raises_value_error_if_wrong_tz(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="US/Eastern")
        df = pd.DataFrame({"open": range(5)}, index=idx)
        with pytest.raises(ValueError, match="must be UTC"):
            enrich_time_features(df)

    def test_raises_value_error_on_column_conflict(self) -> None:
        df = _make_df(periods=10)
        df["paris_hour"] = 0  # conflict
        with pytest.raises(ValueError, match="already contains columns"):
            enrich_time_features(df)

    def test_empty_dataframe_returns_empty(self) -> None:
        idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        df = pd.DataFrame({"open": []}, index=idx)
        result = enrich_time_features(df)
        assert result.empty
        assert result is not df  # copy

    def test_does_not_modify_input(self) -> None:
        df = _make_df(periods=20)
        original_cols = list(df.columns)
        _ = enrich_time_features(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Session / Killzone Detection
# ---------------------------------------------------------------------------


class TestSessions:
    """Tests for session and killzone labeling."""

    def test_asia_session_midnight_paris(self) -> None:
        """Bar at 23:00 UTC in winter = 00:00 Paris (CET=UTC+1) → Asia."""
        df = _make_single_bar("2024-01-15 23:00")  # Mon, 00:00 Paris
        result = enrich_time_features(df)
        assert result["is_asia_session"].iloc[0] is np.True_

    def test_london_killzone_detection(self) -> None:
        """Bar at 07:00 UTC in winter = 08:00 Paris → London KZ."""
        df = _make_single_bar("2024-01-15 07:00")
        result = enrich_time_features(df)
        assert result["is_london_kz"].iloc[0] is np.True_
        assert result["is_killzone"].iloc[0] is np.True_

    def test_ny_killzone_detection(self) -> None:
        """Bar at 12:30 UTC in winter = 13:30 Paris → NY KZ."""
        df = _make_single_bar("2024-01-15 12:30")
        result = enrich_time_features(df)
        assert result["is_ny_kz"].iloc[0] is np.True_
        assert result["is_killzone"].iloc[0] is np.True_

    def test_overlap_london_ny(self) -> None:
        """Bar at 13:30 Paris is in both London session and NY session → overlap."""
        df = _make_single_bar("2024-01-15 12:30")  # 13:30 Paris
        result = enrich_time_features(df)
        assert result["is_overlap_london_ny"].iloc[0] is np.True_
        assert result["is_london_session"].iloc[0] is np.True_
        assert result["is_ny_session"].iloc[0] is np.True_

    def test_session_label_priority(self) -> None:
        """Overlap label takes priority over individual KZ labels."""
        # 13:30 Paris = overlap + london_session + ny_kz
        df = _make_single_bar("2024-01-15 12:30")
        result = enrich_time_features(df)
        assert result["session_label"].iloc[0] == "overlap"

    def test_off_session_late_night(self) -> None:
        """Bar at 22:00 Paris → off (after NY session ends)."""
        df = _make_single_bar("2024-01-15 21:00")  # 22:00 Paris (CET)
        result = enrich_time_features(df)
        assert result["session_label"].iloc[0] == "off"

    def test_session_label_is_categorical(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        assert isinstance(result["session_label"].dtype, pd.CategoricalDtype)
        expected_cats = ["overlap", "london_kz", "ny_kz", "london", "ny", "asia", "off"]
        assert list(result["session_label"].cat.categories) == expected_cats

    def test_no_killzone_outside_hours(self) -> None:
        """Bar at 05:00 Paris → Asia only, no killzone."""
        df = _make_single_bar("2024-01-15 04:00")  # 05:00 Paris
        result = enrich_time_features(df)
        assert result["is_killzone"].iloc[0] is np.False_
        assert result["is_asia_session"].iloc[0] is np.True_


# ---------------------------------------------------------------------------
# DST Transitions
# ---------------------------------------------------------------------------


class TestDST:
    """Tests for DST-aware timezone handling."""

    def test_winter_utc_plus_1(self) -> None:
        """In January, Paris = UTC+1. 07:00 UTC → 08:00 Paris."""
        df = _make_single_bar("2024-01-15 07:00")
        result = enrich_time_features(df)
        assert result["paris_hour"].iloc[0] == 8

    def test_summer_utc_plus_2(self) -> None:
        """In July, Paris = UTC+2. 07:00 UTC → 09:00 Paris."""
        df = _make_single_bar("2024-07-15 07:00")
        result = enrich_time_features(df)
        assert result["paris_hour"].iloc[0] == 9

    def test_spring_dst_transition(self) -> None:
        """2024-03-31 at 01:00 UTC, Paris jumps from CET to CEST (02:00→03:00).
        So 01:00 UTC = 03:00 Paris (CEST)."""
        df = _make_single_bar("2024-03-31 01:00")
        result = enrich_time_features(df)
        assert result["paris_hour"].iloc[0] == 3

    def test_autumn_dst_transition(self) -> None:
        """2024-10-27 at 01:00 UTC, Paris falls back (CEST→CET).
        01:00 UTC = 02:00 Paris (CET, after fallback)."""
        df = _make_single_bar("2024-10-27 01:00")
        result = enrich_time_features(df)
        assert result["paris_hour"].iloc[0] == 2

    def test_london_kz_shifts_with_dst(self) -> None:
        """London KZ is 08:00-11:00 Paris.
        In winter: 07:00-10:00 UTC. In summer: 06:00-09:00 UTC."""
        # Winter: 07:00 UTC = 08:00 Paris → London KZ
        df_winter = _make_single_bar("2024-01-15 07:00")
        assert enrich_time_features(df_winter)["is_london_kz"].iloc[0] is np.True_

        # Summer: 06:00 UTC = 08:00 Paris → London KZ
        df_summer = _make_single_bar("2024-07-15 06:00")
        assert enrich_time_features(df_summer)["is_london_kz"].iloc[0] is np.True_

        # Summer: 07:00 UTC = 09:00 Paris → still London KZ (08-11)
        df_summer2 = _make_single_bar("2024-07-15 07:00")
        assert enrich_time_features(df_summer2)["is_london_kz"].iloc[0] is np.True_


# ---------------------------------------------------------------------------
# XAU Market Hours
# ---------------------------------------------------------------------------


class TestXAUMarketHours:
    """Tests for XAU market open/close detection."""

    def test_xau_open_weekday(self) -> None:
        """Wednesday 12:00 UTC → market open."""
        df = _make_single_bar("2024-01-17 12:00")  # Wednesday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.True_

    def test_xau_closed_friday_after_21utc(self) -> None:
        """Friday 21:00 UTC → market closed."""
        df = _make_single_bar("2024-01-19 21:00")  # Friday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.False_

    def test_xau_closed_saturday(self) -> None:
        """Saturday any time → market closed."""
        df = _make_single_bar("2024-01-20 12:00")  # Saturday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.False_

    def test_xau_closed_sunday_before_22utc(self) -> None:
        """Sunday 21:00 UTC → market still closed."""
        df = _make_single_bar("2024-01-21 21:00")  # Sunday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.False_

    def test_xau_open_sunday_at_22utc(self) -> None:
        """Sunday 22:00 UTC → market opens."""
        df = _make_single_bar("2024-01-21 22:00")  # Sunday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.True_

    def test_xau_open_friday_before_21utc(self) -> None:
        """Friday 20:45 UTC → market still open."""
        df = _make_single_bar("2024-01-19 20:45")  # Friday
        result = enrich_time_features(df)
        assert result["is_xau_market_open"].iloc[0] is np.True_


# ---------------------------------------------------------------------------
# Friday Management
# ---------------------------------------------------------------------------


class TestFridayManagement:
    """Tests for XAU Friday pre-close and force-close flags."""

    def test_friday_pre_close_default_18h_paris(self) -> None:
        """Friday 17:00 UTC in winter = 18:00 Paris → pre-close flag."""
        df = _make_single_bar("2024-01-19 17:00")  # Friday, 18:00 Paris
        result = enrich_time_features(df)
        assert result["is_xau_friday_pre_close"].iloc[0] is np.True_

    def test_friday_force_close_default_2230_paris(self) -> None:
        """Friday 21:30 UTC in winter = 22:30 Paris → force-close flag."""
        df = _make_single_bar("2024-01-19 21:30")  # Friday, 22:30 Paris
        result = enrich_time_features(df)
        assert result["is_xau_friday_force_close"].iloc[0] is np.True_

    def test_friday_before_pre_close_not_flagged(self) -> None:
        """Friday 16:00 UTC in winter = 17:00 Paris → no pre-close."""
        df = _make_single_bar("2024-01-19 16:00")  # Friday, 17:00 Paris
        result = enrich_time_features(df)
        assert result["is_xau_friday_pre_close"].iloc[0] is np.False_

    def test_non_friday_not_flagged(self) -> None:
        """Wednesday at any time → no Friday flags."""
        df = _make_single_bar("2024-01-17 18:00")  # Wednesday
        result = enrich_time_features(df)
        assert result["is_xau_friday_pre_close"].iloc[0] is np.False_
        assert result["is_xau_friday_force_close"].iloc[0] is np.False_


# ---------------------------------------------------------------------------
# Meta Time Features
# ---------------------------------------------------------------------------


class TestMetaFeatures:
    """Tests for iso_week, month, quarter, year."""

    def test_iso_week_correct(self) -> None:
        df = _make_single_bar("2024-01-01 12:00")  # ISO week 1
        result = enrich_time_features(df)
        assert result["iso_week"].iloc[0] == 1

    def test_month_and_quarter(self) -> None:
        df = _make_single_bar("2024-04-15 12:00")  # April = Q2
        result = enrich_time_features(df)
        assert result["month"].iloc[0] == 4
        assert result["quarter"].iloc[0] == 2

    def test_year_correct(self) -> None:
        df = _make_single_bar("2024-12-31 23:00")
        result = enrich_time_features(df)
        # 23:00 UTC = 00:00 Paris Jan 1 2025 (CET)
        assert result["year"].iloc[0] == 2025

    def test_paris_date_reflects_timezone(self) -> None:
        """23:30 UTC on Dec 31 = 00:30 Paris on Jan 1."""
        df = _make_single_bar("2024-12-31 23:30")
        result = enrich_time_features(df)
        from datetime import date

        assert result["paris_date"].iloc[0] == date(2025, 1, 1)


# ---------------------------------------------------------------------------
# Dtype & Output Structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Tests for output DataFrame structure and dtypes."""

    def test_all_expected_columns_added(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        for col in _ADDED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        for col in df.columns:
            assert col in result.columns

    def test_int8_dtypes(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        for col in ["paris_hour", "paris_minute", "paris_weekday", "iso_week", "month", "quarter"]:
            assert result[col].dtype == np.int8, f"{col} dtype={result[col].dtype}"

    def test_year_int16(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        assert result["year"].dtype == np.int16

    def test_boolean_columns_are_bool(self) -> None:
        df = _make_df(periods=50)
        result = enrich_time_features(df)
        bool_cols = [
            "is_asia_session", "is_london_kz", "is_london_session",
            "is_ny_kz", "is_ny_session", "is_killzone", "is_overlap_london_ny",
            "is_xau_friday_pre_close", "is_xau_friday_force_close", "is_xau_market_open",
        ]
        for col in bool_cols:
            assert result[col].dtype == bool, f"{col} dtype={result[col].dtype}"
