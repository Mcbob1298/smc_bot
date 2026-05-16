"""Tests for data.enrichment.atr module.

Covers: True Range, ATR Wilder/SMA, causality, edge cases, enrich_atr,
performance, integration with time features / validator / parquet.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from data.enrichment.atr import compute_atr, compute_true_range, enrich_atr
from tests.test_no_lookahead import assert_function_is_causal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic structure."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-15 00:00", periods=n, freq="15min", tz="UTC")
    base = 2100.0 + rng.normal(0, 0.5, n).cumsum()
    spread = rng.uniform(1.0, 5.0, n)
    opens = base
    highs = base + spread
    lows = base - spread
    closes = base + rng.normal(0, 1.0, n)
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(500, 5000, n),
        },
        index=idx,
    )


def _make_known_df() -> pd.DataFrame:
    """5-bar dataset with manually calculable values."""
    idx = pd.date_range("2024-01-15", periods=5, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "high": [110.0, 115.0, 112.0, 118.0, 114.0],
            "low": [100.0, 105.0, 102.0, 108.0, 104.0],
            "close": [105.0, 110.0, 107.0, 115.0, 109.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# True Range Tests
# ---------------------------------------------------------------------------


class TestTrueRange:
    """Tests for compute_true_range."""

    def test_true_range_simple_case(self) -> None:
        """Verify TR on 5-bar known dataset."""
        df = _make_known_df()
        tr = compute_true_range(df)

        # Bar 0: high-low = 110-100 = 10 (no prev_close)
        assert tr.iloc[0] == pytest.approx(10.0)

        # Bar 1: max(115-105, |115-105|, |105-105|) = max(10, 10, 0) = 10
        assert tr.iloc[1] == pytest.approx(10.0)

        # Bar 2: max(112-102, |112-110|, |102-110|) = max(10, 2, 8) = 10
        assert tr.iloc[2] == pytest.approx(10.0)

        # Bar 3: max(118-108, |118-107|, |108-107|) = max(10, 11, 1) = 11
        assert tr.iloc[3] == pytest.approx(11.0)

        # Bar 4: max(114-104, |114-115|, |104-115|) = max(10, 1, 11) = 11
        assert tr.iloc[4] == pytest.approx(11.0)

    def test_true_range_with_gaps(self) -> None:
        """Gap up: low[t] > close[t-1] → TR uses abs(high - prev_close)."""
        idx = pd.date_range("2024-01-15", periods=2, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {"high": [100.0, 120.0], "low": [90.0, 112.0], "close": [95.0, 115.0]},
            index=idx,
        )
        tr = compute_true_range(df)
        # Bar 1: max(120-112, |120-95|, |112-95|) = max(8, 25, 17) = 25
        assert tr.iloc[1] == pytest.approx(25.0)

    def test_true_range_first_bar(self) -> None:
        """TR[0] = high[0] - low[0], no prev_close dependency."""
        df = _make_known_df()
        tr = compute_true_range(df)
        assert tr.iloc[0] == pytest.approx(110.0 - 100.0)


# ---------------------------------------------------------------------------
# ATR Tests
# ---------------------------------------------------------------------------


class TestATR:
    """Tests for compute_atr."""

    def test_atr_period_14_known_values(self) -> None:
        """ATR on known data matches manual Wilder calculation."""
        df = _make_ohlcv(n=30, seed=1)
        tr = compute_true_range(df)
        atr = compute_atr(df, period=14)

        # ATR[13] = mean(TR[0:14])
        expected_init = tr.iloc[:14].mean()
        assert atr.iloc[13] == pytest.approx(expected_init, rel=1e-9)

        # ATR[14] = (ATR[13] * 13 + TR[14]) / 14
        expected_14 = (expected_init * 13 + tr.iloc[14]) / 14
        assert atr.iloc[14] == pytest.approx(expected_14, rel=1e-9)

        # ATR[15] = (ATR[14] * 13 + TR[15]) / 14
        expected_15 = (expected_14 * 13 + tr.iloc[15]) / 14
        assert atr.iloc[15] == pytest.approx(expected_15, rel=1e-9)

    def test_atr_first_n_minus_1_are_nan(self) -> None:
        """First period-1 values must be NaN."""
        df = _make_ohlcv(n=50)
        atr = compute_atr(df, period=14)
        assert atr.iloc[:13].isna().all()
        assert not np.isnan(atr.iloc[13])

    def test_atr_initialization_at_period_minus_1(self) -> None:
        """ATR[period-1] = mean(TR[0:period])."""
        df = _make_ohlcv(n=50)
        tr = compute_true_range(df)
        atr = compute_atr(df, period=14)
        expected = tr.iloc[:14].mean()
        assert atr.iloc[13] == pytest.approx(expected, rel=1e-9)

    def test_atr_wilder_recursive_formula(self) -> None:
        """Verify Wilder's formula: ATR[t] = (ATR[t-1]*(p-1) + TR[t]) / p."""
        df = _make_ohlcv(n=50)
        tr = compute_true_range(df)
        atr = compute_atr(df, period=14)

        # Check several bars after init
        for i in range(14, 20):
            expected = (atr.iloc[i - 1] * 13 + tr.iloc[i]) / 14
            assert atr.iloc[i] == pytest.approx(expected, rel=1e-9)

    def test_atr_sma_method(self) -> None:
        """method='sma' gives rolling mean of TR."""
        df = _make_ohlcv(n=50)
        tr = compute_true_range(df)
        atr_sma = compute_atr(df, period=14, method="sma")

        # Compare with explicit rolling mean
        expected = tr.rolling(14, min_periods=14).mean()
        pd.testing.assert_series_equal(atr_sma, expected, check_names=False)


# ---------------------------------------------------------------------------
# Causality Tests
# ---------------------------------------------------------------------------


class TestCausality:
    """Anti-lookahead tests for ATR."""

    def test_atr_is_causal(self) -> None:
        """ATR must produce same output regardless of future data."""
        df = _make_ohlcv(n=200)

        def atr_enricher(input_df: pd.DataFrame) -> pd.DataFrame:
            result = input_df.copy()
            result["atr_14"] = compute_atr(input_df, period=14)
            return result

        assert_function_is_causal(
            func=atr_enricher,
            df=df,
            added_columns=["atr_14"],
        )

    def test_true_range_is_causal(self) -> None:
        """TR must produce same output regardless of future data."""
        df = _make_ohlcv(n=200)

        def tr_enricher(input_df: pd.DataFrame) -> pd.DataFrame:
            result = input_df.copy()
            result["true_range"] = compute_true_range(input_df)
            return result

        assert_function_is_causal(
            func=tr_enricher,
            df=df,
            added_columns=["true_range"],
        )

    def test_atr_full_prefix_matches_truncated(self) -> None:
        """Compare ENTIRE prefix (not just last value) at k=25%, 50%, 75%."""
        df = _make_ohlcv(n=200)
        full_atr = compute_atr(df, period=14)

        for ratio in [0.25, 0.5, 0.75]:
            k = int(len(df) * ratio)
            truncated_atr = compute_atr(df.iloc[:k], period=14)
            # Compare ALL non-NaN values in the prefix
            np.testing.assert_allclose(
                np.asarray(full_atr.iloc[:k]),
                np.asarray(truncated_atr),
                rtol=1e-9,
                equal_nan=True,
                err_msg=f"Full prefix mismatch at ratio={ratio} (k={k})",
            )

    def test_atr_with_truncation_at_period_boundary(self) -> None:
        """At k=period exactly, ATR[period-1] is identical full vs truncated."""
        df = _make_ohlcv(n=100)
        period = 14

        full_atr = compute_atr(df, period=period)
        truncated_atr = compute_atr(df.iloc[:period], period=period)

        # Both should have the same value at index period-1
        assert full_atr.iloc[period - 1] == pytest.approx(
            truncated_atr.iloc[period - 1], rel=1e-9
        )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_df(self) -> None:
        """Empty df → empty Series."""
        idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        df = pd.DataFrame({"high": [], "low": [], "close": []}, index=idx)
        tr = compute_true_range(df)
        assert len(tr) == 0
        atr = compute_atr(df, period=14)
        assert len(atr) == 0

    def test_df_shorter_than_period(self) -> None:
        """len(df) < period → all NaN."""
        df = _make_ohlcv(n=10)
        atr = compute_atr(df, period=14)
        assert atr.isna().all()

    def test_df_exactly_period_length(self) -> None:
        """len(df) == period → one non-NaN value at the end."""
        df = _make_ohlcv(n=14)
        atr = compute_atr(df, period=14)
        assert atr.iloc[:13].isna().all()
        assert not np.isnan(atr.iloc[13])

    def test_missing_columns_raises(self) -> None:
        """df without 'high' → ValueError."""
        idx = pd.date_range("2024-01-15", periods=5, freq="15min", tz="UTC")
        df = pd.DataFrame({"low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5]}, index=idx)
        with pytest.raises(ValueError, match="missing required columns.*high"):
            compute_true_range(df)

    def test_invalid_period_raises(self) -> None:
        """period=0 or -1 → ValueError."""
        df = _make_ohlcv(n=50)
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_atr(df, period=0)
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_atr(df, period=-1)

    def test_nan_in_input_propagates(self) -> None:
        """NaN in close[5] → ATR contains NaN from that point onward."""
        df = _make_ohlcv(n=50)
        df.loc[df.index[5], "close"] = np.nan
        atr = compute_atr(df, period=14)
        # ATR[13] uses TR[0:14], TR[6] depends on close[5]=NaN → NaN propagates
        assert np.isnan(atr.iloc[13])

    def test_invalid_method_raises(self) -> None:
        """method='foo' → ValueError."""
        df = _make_ohlcv(n=50)
        with pytest.raises(ValueError, match="method must be"):
            compute_atr(df, period=14, method="foo")


# ---------------------------------------------------------------------------
# enrich_atr Tests
# ---------------------------------------------------------------------------


class TestEnrichATR:
    """Tests for enrich_atr wrapper."""

    def test_enrich_adds_column(self) -> None:
        """Column 'atr_14' is added."""
        df = _make_ohlcv(n=50)
        result = enrich_atr(df, period=14)
        assert "atr_14" in result.columns

    def test_enrich_custom_column_name(self) -> None:
        """column_name='my_atr' is respected."""
        df = _make_ohlcv(n=50)
        result = enrich_atr(df, period=14, column_name="my_atr")
        assert "my_atr" in result.columns
        assert "atr_14" not in result.columns

    def test_enrich_existing_column_raises(self) -> None:
        """If column already present → ValueError."""
        df = _make_ohlcv(n=50)
        df["atr_14"] = 0.0
        with pytest.raises(ValueError, match="already contains column"):
            enrich_atr(df, period=14)

    def test_enrich_returns_copy(self) -> None:
        """Original df is not modified."""
        df = _make_ohlcv(n=50)
        original_cols = list(df.columns)
        _ = enrich_atr(df, period=14)
        assert list(df.columns) == original_cols

    def test_enrich_preserves_other_columns(self) -> None:
        """All original columns are preserved."""
        df = _make_ohlcv(n=50)
        result = enrich_atr(df, period=14)
        for col in df.columns:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance benchmarks."""

    def test_performance_100k_bars(self) -> None:
        """ATR on 100k bars < 1s."""
        df = _make_ohlcv(n=100_000, seed=99)
        start = time.perf_counter()
        compute_atr(df, period=14)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100k bars took {elapsed:.2f}s (limit: 1s)"

    def test_performance_1m_bars(self) -> None:
        """ATR on 1M bars < 10s."""
        df = _make_ohlcv(n=1_000_000, seed=99)
        start = time.perf_counter()
        compute_atr(df, period=14)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"1M bars took {elapsed:.2f}s (limit: 10s)"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with other modules."""

    def test_atr_after_time_features(self) -> None:
        """enrich time + enrich atr → no column conflict."""
        from data.enrichment.time_features import enrich_time_features

        df = _make_ohlcv(n=50)
        enriched = enrich_time_features(df)
        result = enrich_atr(enriched, period=14)
        assert "atr_14" in result.columns
        assert "paris_hour" in result.columns

    def test_atr_passes_validation(self) -> None:
        """df with ATR passes OHLCVValidator (needs spread/real_volume)."""
        from data.storage.data_validator import OHLCVValidator

        rng = np.random.default_rng(42)
        n = 200
        df = _make_ohlcv(n=n)
        df["spread"] = rng.uniform(0.02, 0.04, n)
        df["real_volume"] = rng.uniform(200, 2000, n)
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")
        assert report.is_valid

    def test_atr_roundtrip_parquet(self, tmp_path: object) -> None:
        """Save/load preserves ATR column."""
        import tempfile
        from pathlib import Path

        from data.storage.parquet_store import ParquetStore

        tmp_dir = Path(tempfile.mkdtemp())
        store = ParquetStore(base_path=tmp_dir)
        rng = np.random.default_rng(42)
        df = _make_ohlcv(n=50)
        df["spread"] = rng.uniform(0.02, 0.04, 50)
        df["real_volume"] = rng.uniform(200, 2000, 50)
        enriched = enrich_atr(df, period=14)

        store.save(enriched, "XAUUSD", "M15", validate=False)
        loaded = store.load("XAUUSD", "M15")

        assert "atr_14" in loaded.columns
        # Compare non-NaN values
        mask = ~enriched["atr_14"].isna()
        np.testing.assert_allclose(
            np.asarray(loaded["atr_14"][mask]),
            np.asarray(enriched["atr_14"][mask]),
            rtol=1e-10,
        )
