"""Tests for liquidity pool detection (equal highs/lows, causal)."""

import pandas as pd

from config.strategy import LiquidityConfig, SwingConfig
from detectors.liquidity import detect_liquidity
from detectors.swings import detect_swings

NF = SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False)


def _frame(rows):
    idx = pd.date_range("2026-06-01 08:00", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, index=idx, columns=["high", "low", "close"])
    df["open"] = df["close"]
    return df


# Two near-equal swing highs (~10) at bars 2 and 6 → equal-highs pool.
ROWS = [
    (6, 4, 5), (8, 5, 7), (10, 7, 9), (8, 6, 7), (7, 5, 6),
    (8, 6, 7), (10.05, 7, 9), (8, 6, 7), (7, 5, 6),
]


def test_equal_highs_pool_formed_at_second_touch():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    liq = detect_liquidity(df, sw, LiquidityConfig(min_touches=2), atr)
    assert len(liq) == 1
    row = liq.iloc[0]
    assert row["kind"] == "equal_highs"
    assert row["touches"] == 2
    assert row["level"] == 10.05  # extreme of the touches
    assert row["confirmed_at"] >= liq.index[0]  # causal
    assert row["first_touch_time"] == df.index[2]


def test_tolerance_band_separates_distinct_levels():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    # Tiny tolerance (0.01*ATR=0.01) → 10.00 and 10.05 are NOT equal → no pool.
    liq = detect_liquidity(
        df, sw, LiquidityConfig(min_touches=2, equal_level_tolerance_atr_ratio=0.01), atr
    )
    assert liq.empty


def test_min_touches_three_not_met():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    liq = detect_liquidity(df, sw, LiquidityConfig(min_touches=3), atr)
    assert liq.empty  # only two equal highs


def test_atr_warmup_nan_skips_touch():
    df = _frame(ROWS)
    atr = pd.Series([float("nan")] * len(df), index=df.index)
    sw = detect_swings(df, NF)
    liq = detect_liquidity(df, sw, LiquidityConfig(min_touches=2), atr)
    assert liq.empty


def test_empty_swings_empty_pools():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    liq = detect_liquidity(df, df.iloc[:0], LiquidityConfig(), atr)
    assert liq.empty
