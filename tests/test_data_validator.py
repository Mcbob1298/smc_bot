"""Tests for OHLCV data validator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.storage.data_validator import (
    DataValidationError,
    OHLCVValidator,
    ValidationSeverity,
)


def _make_clean_xau_m15(
    start: datetime | None = None,
    periods: int = 200,
) -> pd.DataFrame:
    """Generate clean XAU M15 data (no weekends, valid OHLC)."""
    if start is None:
        # Monday
        start = datetime(2024, 3, 4, 8, 0, tzinfo=UTC)

    # Generate timestamps, skip weekends
    timestamps = []
    current = start
    while len(timestamps) < periods:
        wd = current.weekday()
        h = current.hour
        # Skip: Saturday, Sunday before 22:00, Friday after 21:45
        if wd == 5 or (wd == 6 and h < 22) or (wd == 4 and h >= 22):
            current += timedelta(minutes=15)
            continue
        timestamps.append(current)
        current += timedelta(minutes=15)

    idx = pd.DatetimeIndex(timestamps, tz=UTC)
    rng = np.random.default_rng(42)
    base = 2000.0
    opens = base + np.cumsum(rng.normal(0, 0.3, periods))
    highs = opens + rng.uniform(0.5, 3.0, periods)
    lows = opens - rng.uniform(0.5, 3.0, periods)
    closes = opens + rng.normal(0, 0.5, periods)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(100, 1000, periods),
            "spread": rng.uniform(0.02, 0.04, periods),
            "real_volume": rng.uniform(50, 500, periods),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


class TestStructuralChecks:
    """Tests for structural validation checks."""

    def test_validate_clean_data_passes(self) -> None:
        df = _make_clean_xau_m15()
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert report.is_valid
        assert len(report.errors) == 0

    def test_naive_index_fails(self) -> None:
        idx = pd.date_range("2024-03-04 08:00", periods=50, freq="15min")
        df = pd.DataFrame(
            {
                col: np.ones(50)
                for col in ["open", "high", "low", "close", "volume", "spread", "real_volume"]
            },
            index=idx,
        )
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any(i.check_name == "check_index_is_datetime" for i in report.errors)

    def test_non_utc_index_fails(self) -> None:
        idx = pd.date_range("2024-03-04 08:00", periods=50, freq="15min", tz="Europe/Paris")
        df = pd.DataFrame(
            {
                col: np.ones(50)
                for col in ["open", "high", "low", "close", "volume", "spread", "real_volume"]
            },
            index=idx,
        )
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("UTC" in i.message for i in report.errors)

    def test_missing_column_fails(self) -> None:
        idx = pd.date_range("2024-03-04 08:00", periods=50, freq="15min", tz=UTC)
        df = pd.DataFrame(
            {"open": np.ones(50), "low": np.ones(50), "close": np.ones(50),
             "volume": np.ones(50), "spread": np.ones(50), "real_volume": np.ones(50)},
            index=idx,
        )  # missing 'high'
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("high" in i.message for i in report.errors)

    def test_wrong_dtype_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        df["close"] = ["bad"] * len(df)  # object dtype, clearly not numeric
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_dtypes" in i.check_name for i in report.errors)

    def test_nan_in_close_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_no_nan_in_ohlc" in i.check_name for i in report.errors)

    def test_non_monotonic_index_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        # Swap two timestamps to break monotonicity
        idx = df.index.tolist()
        idx[10], idx[11] = idx[11], idx[10]
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_index_monotonic" in i.check_name for i in report.errors)

    def test_duplicate_timestamps_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        idx = df.index.tolist()
        idx[5] = idx[4]  # duplicate
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("duplicate" in i.check_name for i in report.errors)


class TestOHLCChecks:
    """Tests for OHLC consistency checks."""

    def test_high_less_than_max_open_close_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        # Set high below open at bar 5
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["open"] - 1.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_high_is_max" in i.check_name for i in report.errors)

    def test_low_greater_than_min_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        # Set low above close at bar 5
        df.iloc[5, df.columns.get_loc("low")] = df.iloc[5]["close"] + 1.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_low_is_min" in i.check_name for i in report.errors)

    def test_high_less_than_low_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        # Swap high and low at bar 5
        h = df.iloc[5]["high"]
        lo = df.iloc[5]["low"]
        df.iloc[5, df.columns.get_loc("high")] = lo - 1.0
        df.iloc[5, df.columns.get_loc("low")] = h + 1.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_high_geq_low" in i.check_name for i in report.errors)

    def test_negative_price_fails(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        df.iloc[3, df.columns.get_loc("close")] = -1.0
        df.iloc[3, df.columns.get_loc("low")] = -2.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_prices_positive" in i.check_name for i in report.errors)


class TestVolumeChecks:
    """Tests for volume validation checks."""

    def test_negative_volume_warns(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        df.iloc[5, df.columns.get_loc("volume")] = -10.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert report.is_valid  # WARNING, not ERROR
        assert any(
            i.check_name == "check_volume_non_negative"
            and i.severity == ValidationSeverity.WARNING
            for i in report.issues
        )

    def test_many_zero_volumes_warns(self) -> None:
        df = _make_clean_xau_m15(periods=200)
        # Set 100 consecutive bars to 0 volume (>5% of total)
        df.iloc[50:150, df.columns.get_loc("volume")] = 0.0
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert report.is_valid
        assert any("check_volume_not_all_zero" in i.check_name for i in report.warnings)


class TestTemporalChecks:
    """Tests for temporal validation checks."""

    def test_correct_15min_spacing_passes(self) -> None:
        df = _make_clean_xau_m15()
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        # No timeframe consistency issues
        assert not any("check_timeframe_consistency" in i.check_name for i in report.errors)

    def test_irregular_spacing_warns(self) -> None:
        df = _make_clean_xau_m15(periods=50)
        # Shift one timestamp by -2 minutes (13min gap instead of 15)
        idx = df.index.tolist()
        idx[10] = idx[10] - timedelta(minutes=2)
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_timeframe_consistency" in i.check_name for i in report.warnings)

    def test_xau_weekend_gap_accepted(self) -> None:
        """A 48h gap over the weekend for XAU should NOT trigger warnings."""
        # Create data ending Friday 21:45 and resuming Sunday 22:00
        fri_end = datetime(2024, 3, 8, 21, 45, tzinfo=UTC)  # Friday
        sun_open = datetime(2024, 3, 10, 22, 0, tzinfo=UTC)  # Sunday

        idx_before = pd.date_range(
            end=fri_end, periods=20, freq="15min", tz=UTC
        )
        idx_after = pd.date_range(
            start=sun_open, periods=20, freq="15min", tz=UTC
        )
        idx = idx_before.append(idx_after)

        rng = np.random.default_rng(42)
        n = len(idx)
        df = pd.DataFrame(
            {
                "open": 2000 + rng.normal(0, 1, n),
                "high": 2002 + rng.uniform(0, 1, n),
                "low": 1998 + rng.uniform(0, 1, n),
                "close": 2000 + rng.normal(0, 1, n),
                "volume": rng.uniform(100, 500, n),
                "spread": rng.uniform(0.02, 0.04, n),
                "real_volume": rng.uniform(50, 200, n),
            },
            index=idx,
        )
        # Fix OHLC consistency
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df.index.name = "timestamp_utc"

        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        # Weekend gap should not appear as unexpected gap
        gap_issues = [i for i in report.issues if i.check_name == "check_unexpected_gaps"]
        assert len(gap_issues) == 0

    def test_xau_intraday_gap_warns(self) -> None:
        """A 2h gap during a weekday session should warn."""
        df = _make_clean_xau_m15(periods=50)
        # Insert a 2h gap in the middle (Tuesday)
        idx = df.index.tolist()
        # Shift everything after bar 25 by +2 hours
        for i in range(25, len(idx)):
            idx[i] = idx[i] + timedelta(hours=2)
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_unexpected_gaps" in i.check_name for i in report.warnings)

    def test_bar_during_xau_closed_market_fails(self) -> None:
        """A bar on Saturday 12:00 UTC for XAU is ERROR."""
        df = _make_clean_xau_m15(periods=50)
        # Replace a timestamp with Saturday 12:00
        idx = df.index.tolist()
        idx[10] = datetime(2024, 3, 9, 12, 0, tzinfo=UTC)  # Saturday
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_xau_market_hours" in i.check_name for i in report.errors)

    def test_future_timestamp_fails(self) -> None:
        """Timestamps in the future are ERROR."""
        df = _make_clean_xau_m15(periods=50)
        idx = df.index.tolist()
        idx[-1] = datetime.now(tz=UTC) + timedelta(hours=1)
        df.index = pd.DatetimeIndex(idx)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert not report.is_valid
        assert any("check_no_future_timestamps" in i.check_name for i in report.errors)


class TestPriceAnomalyChecks:
    """Tests for price anomaly detection."""

    def test_5pct_jump_warns(self) -> None:
        """A 5%+ price jump on M15 should warn."""
        df = _make_clean_xau_m15(periods=50)
        # Insert a 6% jump
        df.iloc[20, df.columns.get_loc("close")] = df.iloc[19]["close"] * 1.06
        df.iloc[20, df.columns.get_loc("high")] = df.iloc[20]["close"] + 1
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_price_jumps" in i.check_name for i in report.warnings)

    def test_sunday_open_gap_accepted(self) -> None:
        """A 3% gap at Sunday 22:00 UTC (market open) should NOT warn."""
        # Build data with Sunday 22:00 open
        sun_open = datetime(2024, 3, 10, 22, 0, tzinfo=UTC)
        idx = pd.date_range(start=sun_open, periods=50, freq="15min", tz=UTC)
        rng = np.random.default_rng(42)
        base = 2000.0
        closes = np.full(50, base)
        # First bar has a 3% gap from "previous close" (simulated via pct_change)
        closes[0] = base * 1.03
        closes[1:] = base + rng.normal(0, 0.5, 49)

        df = pd.DataFrame(
            {
                "open": closes - rng.uniform(0, 0.5, 50),
                "high": closes + rng.uniform(0.5, 2, 50),
                "low": closes - rng.uniform(0.5, 2, 50),
                "close": closes,
                "volume": rng.uniform(100, 500, 50),
                "spread": rng.uniform(0.02, 0.04, 50),
                "real_volume": rng.uniform(50, 200, 50),
            },
            index=idx,
        )
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df.index.name = "timestamp_utc"

        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        # The Sunday open gap should be excluded from price jump warnings
        jump_issues = [i for i in report.issues if i.check_name == "check_price_jumps"]
        assert len(jump_issues) == 0

    def test_zero_range_bar_warns_if_many(self) -> None:
        """More than 1% of bars with high==low should warn."""
        df = _make_clean_xau_m15(periods=200)
        # Set 3% of bars to high==low (zero range)
        for i in range(0, 6):
            df.iloc[i * 30, df.columns.get_loc("high")] = df.iloc[i * 30]["low"]
            df.iloc[i * 30, df.columns.get_loc("open")] = df.iloc[i * 30]["low"]
            df.iloc[i * 30, df.columns.get_loc("close")] = df.iloc[i * 30]["low"]
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_zero_range_bars" in i.check_name for i in report.warnings)

    def test_extreme_spread_warns(self) -> None:
        """Spread > 5x median should warn."""
        df = _make_clean_xau_m15(periods=100)
        # Set one bar's spread to 10x median
        median_spread = df["spread"].median()
        df.iloc[50, df.columns.get_loc("spread")] = median_spread * 10
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_extreme_spread" in i.check_name for i in report.warnings)

    def test_price_outlier_zscore(self) -> None:
        """A return at 8 sigma should be flagged as outlier."""
        df = _make_clean_xau_m15(periods=200)
        # Inject a massive outlier return (8+ sigma)
        df.iloc[100, df.columns.get_loc("close")] = df.iloc[99]["close"] * 1.15
        df.iloc[100, df.columns.get_loc("high")] = df.iloc[100]["close"] + 1
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        assert any("check_price_outliers_zscore" in i.check_name for i in report.warnings)


class TestModes:
    """Tests for strict vs non-strict modes."""

    def test_strict_mode_raises_on_first_error(self) -> None:
        """strict=True raises DataValidationError on first ERROR."""
        df = _make_clean_xau_m15(periods=50)
        df.iloc[5, df.columns.get_loc("close")] = np.nan  # ERROR

        validator = OHLCVValidator()
        with pytest.raises(DataValidationError, match="check_no_nan_in_ohlc"):
            validator.validate(df, "XAUUSD", "M15", strict=True)

    def test_non_strict_returns_report(self) -> None:
        """strict=False returns report with is_valid=False on errors."""
        df = _make_clean_xau_m15(periods=50)
        df.iloc[5, df.columns.get_loc("close")] = np.nan

        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15", strict=False)

        assert not report.is_valid
        assert len(report.errors) > 0


class TestParquetStoreIntegration:
    """Tests for validation integration with ParquetStore."""

    def test_parquet_save_with_validation_blocks_bad_data(self, tmp_path: Path) -> None:
        """strict_validation=True + invalid data → raise, nothing written."""
        from data.storage.parquet_store import ParquetStore

        store = ParquetStore(base_path=tmp_path)
        df = _make_clean_xau_m15(periods=50)
        df.iloc[5, df.columns.get_loc("close")] = np.nan

        with pytest.raises(DataValidationError):
            store.save(df, "XAUUSD", "M15", validate=True, strict_validation=True)

        # Nothing should have been written
        assert not store.exists("XAUUSD", "M15")

    def test_parquet_save_with_validation_logs_warnings(self, tmp_path: Path) -> None:
        """Warnings don't block save, data is written."""
        from data.storage.parquet_store import ParquetStore

        store = ParquetStore(base_path=tmp_path)
        df = _make_clean_xau_m15(periods=200)
        # Add volume warning (many zero volumes)
        df.iloc[50:150, df.columns.get_loc("volume")] = 0.0

        report = store.save(df, "XAUUSD", "M15", validate=True, strict_validation=False)

        assert store.exists("XAUUSD", "M15")
        assert report is not None
        assert report.is_valid  # warnings don't make it invalid

    def test_parquet_save_validate_false_skips_check(self, tmp_path: Path) -> None:
        """validate=False skips all validation — invalid data gets written."""
        from data.storage.parquet_store import ParquetStore

        store = ParquetStore(base_path=tmp_path)
        df = _make_clean_xau_m15(periods=50)
        df.iloc[5, df.columns.get_loc("close")] = np.nan

        report = store.save(df, "XAUUSD", "M15", validate=False)

        assert report is None
        assert store.exists("XAUUSD", "M15")


class TestReport:
    """Tests for ValidationReport format."""

    def test_validation_report_summary_format(self) -> None:
        """summary() produces readable text."""
        df = _make_clean_xau_m15(periods=50)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        summary = report.summary()
        assert "VALID" in summary
        assert "XAUUSD" in summary
        assert "M15" in summary

    def test_validation_report_to_dict_serializable(self) -> None:
        """to_dict() produces JSON-serializable dict."""
        import json

        df = _make_clean_xau_m15(periods=50)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")

        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert "XAUUSD" in serialized
        assert "is_valid" in serialized
