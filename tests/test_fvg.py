"""Tests for Fair Value Gap detection (causal)."""

import pandas as pd

from config.strategy import FVGConfig
from detectors.fvg import detect_fvgs


def _frame(rows):
    idx = pd.date_range("2026-06-01 08:00", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def test_bullish_fvg_zone_and_confirmation():
    # Bar0 high=10; bar2 low=12 > 10 → bullish gap [10, 12].
    df = _frame([
        [9, 10, 8, 9],     # 0 left
        [10, 18, 10, 17],  # 1 middle (impulse)
        [17, 20, 12, 19],  # 2 right
        [19, 19, 17, 18],  # 3
    ])
    atr = pd.Series(1.0, index=df.index)
    fvg = detect_fvgs(df, FVGConfig(min_size_atr_ratio=0.2), atr)
    assert len(fvg) == 1
    row = fvg.iloc[0]
    assert row["kind"] == "bullish"
    assert (row["bottom"], row["top"]) == (10.0, 12.0)
    assert row["mid"] == 11.0
    assert fvg.index[0] == df.index[1]            # indexed at middle bar
    assert row["confirmed_at"] == df.index[2]     # known only at right bar


def test_bearish_fvg_zone():
    # Bar0 low=18; bar2 high=12 < 18 → bearish gap [12, 18].
    df = _frame([
        [19, 20, 18, 19],  # 0 left
        [18, 18, 6, 7],    # 1 middle (impulse down)
        [7, 12, 5, 6],     # 2 right
    ])
    atr = pd.Series(1.0, index=df.index)
    fvg = detect_fvgs(df, FVGConfig(min_size_atr_ratio=0.2), atr)
    assert len(fvg) == 1
    assert fvg.iloc[0]["kind"] == "bearish"
    assert (fvg.iloc[0]["bottom"], fvg.iloc[0]["top"]) == (12.0, 18.0)


def test_min_size_filter_drops_small_gap():
    df = _frame([
        [9, 10, 8, 9],
        [10, 18, 10, 17],
        [17, 20, 12, 19],  # 2$ gap
    ])
    atr = pd.Series(20.0, index=df.index)  # 0.2*20 = 4$ required > 2$ gap
    fvg = detect_fvgs(df, FVGConfig(min_size_atr_ratio=0.2), atr)
    assert fvg.empty


def test_no_gap_when_candles_overlap():
    df = _frame([
        [9, 12, 8, 11],
        [11, 14, 10, 13],
        [13, 16, 11, 15],  # low 11 < left high 12 → no bullish gap
    ])
    atr = pd.Series(1.0, index=df.index)
    assert detect_fvgs(df, FVGConfig(), atr).empty


def test_atr_warmup_nan_drops_gap():
    df = _frame([
        [9, 10, 8, 9],
        [10, 18, 10, 17],
        [17, 20, 12, 19],
    ])
    atr = pd.Series([float("nan")] * 3, index=df.index)
    assert detect_fvgs(df, FVGConfig(), atr).empty
