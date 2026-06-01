"""Anti-lookahead testing utilities and tests.

Provides `assert_function_is_causal`: a generic helper that verifies
an enrichment function produces identical output for row k whether
computed on df[:k+1] or the full df. Any difference proves look-ahead.

Causality contract: enrichment functions must only use data at or before
each row's index to compute that row's features.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from config.strategy import (
    FVGConfig,
    LiquidityConfig,
    OrderBlockConfig,
    StructureConfig,
    SweepConfig,
    SwingConfig,
)
from data.enrichment.atr import compute_atr
from data.enrichment.time_features import enrich_time_features
from detectors._common import select_known
from detectors.fvg import detect_fvgs
from detectors.liquidity import detect_liquidity
from detectors.order_blocks import detect_order_blocks
from detectors.structure import detect_structure
from detectors.sweeps import detect_sweeps
from detectors.swings import detect_swings


def assert_function_is_causal(
    func: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    added_columns: list[str],
    truncation_ratios: list[float] | None = None,
) -> None:
    """Assert that `func` is causal (no look-ahead bias).

    Strategy: for each truncation point k, compute func on df[:k+1].
    The result at row k must equal what we get from func(full_df) at row k.

    Args:
        func: Enrichment function that takes a DataFrame and returns enriched DataFrame.
        df: Input DataFrame (must have enough rows for meaningful test).
        added_columns: Columns added by the function to check for equality.
        truncation_ratios: Fractions of len(df) to test at. Default [0.25, 0.5, 0.75].

    Raises:
        AssertionError: If any truncated computation differs from full computation.
    """
    if truncation_ratios is None:
        truncation_ratios = [0.25, 0.5, 0.75]

    # Full computation (reference)
    full_result = func(df)

    for ratio in truncation_ratios:
        k = int(len(df) * ratio)
        if k < 2:
            continue

        # Truncated computation
        truncated_df = df.iloc[:k].copy()
        truncated_result = func(truncated_df)

        # Compare row k-1 (last row of truncated) in both results
        for col in added_columns:
            full_val = full_result[col].iloc[k - 1]
            trunc_val = truncated_result[col].iloc[k - 1]

            # Handle NaN equality
            if isinstance(full_val, float) and np.isnan(full_val):
                assert isinstance(trunc_val, float) and np.isnan(trunc_val), (
                    f"Look-ahead detected in column '{col}' at ratio={ratio} (index {k-1}): "
                    f"truncated={trunc_val}, full={full_val}"
                )
            else:
                assert full_val == trunc_val, (
                    f"Look-ahead detected in column '{col}' at ratio={ratio} (index {k-1}): "
                    f"truncated={trunc_val}, full={full_val}"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_utc_df(periods: int = 200) -> pd.DataFrame:
    """Generate simple OHLCV DataFrame with UTC DatetimeIndex for testing."""
    idx = pd.date_range("2024-03-04 08:00", periods=periods, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
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


class TestAntiLookahead:
    """Tests that enrichment functions are causal."""

    def test_enrich_time_features_is_causal(self) -> None:
        """enrich_time_features must produce same output regardless of future data."""
        df = _make_utc_df(periods=200)

        added_columns = [
            "paris_hour",
            "paris_minute",
            "paris_weekday",
            "is_asia_session",
            "is_london_kz",
            "is_london_session",
            "is_ny_kz",
            "is_ny_session",
            "is_killzone",
            "is_overlap_london_ny",
            "session_label",
            "is_xau_friday_pre_close",
            "is_xau_friday_force_close",
            "is_xau_market_open",
            "iso_week",
            "month",
            "quarter",
            "year",
        ]

        assert_function_is_causal(
            func=enrich_time_features,
            df=df,
            added_columns=added_columns,
        )

    def test_causal_helper_detects_lookahead(self) -> None:
        """Verify the helper actually catches a function with look-ahead."""
        df = _make_utc_df(periods=100)

        def leaky_func(input_df: pd.DataFrame) -> pd.DataFrame:
            """A function that leaks future data into current row."""
            result = input_df.copy()
            # Future mean — clearly looks ahead
            result["future_mean"] = result["close"].shift(-5).rolling(5).mean()
            return result

        with pytest.raises(AssertionError, match="Look-ahead detected"):
            assert_function_is_causal(
                func=leaky_func,
                df=df,
                added_columns=["future_mean"],
            )


# ---------------------------------------------------------------------------
# Detector pipeline causality
# ---------------------------------------------------------------------------

_NF = SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False)


def _run_detectors(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full SMC detector pipeline on ``df`` and return its event frames."""
    atr = compute_atr(df, period=14)
    sw = detect_swings(df, _NF)
    st = detect_structure(df, sw, StructureConfig())
    fv = detect_fvgs(df, FVGConfig(), atr)
    liq = detect_liquidity(df, sw, LiquidityConfig(), atr)
    swp = detect_sweeps(df, sw, SweepConfig(), atr)
    ob = detect_order_blocks(
        df, st, fv, swp,
        OrderBlockConfig(require_fvg=False, require_prior_liquidity_sweep=False),
    )
    return {"swings": sw, "structure": st, "fvg": fv, "liquidity": liq,
            "sweeps": swp, "order_blocks": ob}


class TestDetectorCausality:
    """The strongest causality check: seeing the future must not change the past.

    For each truncation point k, the events a live system *could have known* by
    bar k (``select_known(full, ts)``) must be byte-for-byte identical to the
    events produced when the detector only ever saw ``df[:k]``.
    """

    def test_pipeline_is_causal(self) -> None:
        df = _make_utc_df(periods=180)
        full = _run_detectors(df)

        for ratio in (0.4, 0.6, 0.8):
            k = int(len(df) * ratio)
            ts = df.index[k - 1]
            trunc = _run_detectors(df.iloc[:k])

            for name, full_events in full.items():
                known = select_known(full_events, ts).sort_index()
                seen = trunc[name].sort_index()
                # Identity columns that must not depend on future bars
                # (expires_at is legitimately clamped to available data).
                assert list(known.index) == list(seen.index), (
                    f"{name}: event set differs at ratio={ratio}"
                )
                assert list(known["confirmed_at"]) == list(seen["confirmed_at"]), (
                    f"{name}: confirmed_at differs at ratio={ratio}"
                )

    def test_every_event_confirmed_at_or_after_its_bar(self) -> None:
        df = _make_utc_df(periods=180)
        for name, events in _run_detectors(df).items():
            if events.empty:
                continue
            assert (events["confirmed_at"] >= events.index).all(), (
                f"{name}: an event is confirmed before the bar it sits on"
            )
