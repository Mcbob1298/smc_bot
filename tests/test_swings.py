"""Tests for fractal swing detection (causal)."""

import pandas as pd
import pytest

from config.strategy import SwingConfig
from detectors.swings import detect_swings


def _frame(highs, lows):
    idx = pd.date_range("2026-06-01 08:00", periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame({"high": highs, "low": lows}, index=idx)


# Clean zigzag: highs at bars 2 & 6, lows at bars 4 & 8 (3-bar fractal).
HIGHS = [1, 3, 5, 3, 2, 4, 7, 5, 3, 5, 8]
LOWS = [0, 2, 4, 2, 1, 3, 6, 4, 1, 4, 7]


def _no_filter_cfg(**kw):
    return SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False, **kw)


def test_detects_known_pivots():
    df = _frame(HIGHS, LOWS)
    sw = detect_swings(df, _no_filter_cfg())
    highs = sw[sw["kind"] == "high"]
    lows = sw[sw["kind"] == "low"]
    assert list(highs.index) == [df.index[2], df.index[6]]
    assert list(lows.index) == [df.index[4], df.index[8]]
    assert list(highs["price"]) == [5, 7]
    assert list(lows["price"]) == [1, 1]


def test_confirmation_is_one_bar_after_centre_for_3bar():
    df = _frame(HIGHS, LOWS)
    sw = detect_swings(df, _no_filter_cfg())
    # centre bar 2 → confirmed at bar 3; strictly later than the centre.
    high2 = sw[(sw["kind"] == "high")].iloc[0]
    assert high2["confirmed_at"] == df.index[3]
    assert (sw["confirmed_at"] > sw.index).all()  # never knowable before it forms


def test_5bar_fractal_uses_two_bars_each_side():
    df = _frame(HIGHS, LOWS)
    sw = detect_swings(df, _no_filter_cfg(fractal_period=5))
    # Bar 6 high (7) dominates bars 4-8 → still a swing; confirmed 2 bars later.
    h = sw[(sw["kind"] == "high")]
    assert df.index[6] in list(h.index)
    assert h.loc[df.index[6], "confirmed_at"] == df.index[8]


def test_atr_filter_drops_small_swings():
    df = _frame(HIGHS, LOWS)
    # Big ATR makes the modest swings insignificant → all filtered out.
    atr = pd.Series(100.0, index=df.index)
    cfg = SwingConfig(atr_filter_enabled=True, atr_filter_ratio=0.3)
    sw = detect_swings(df, cfg, atr=atr)
    assert sw.empty


def test_atr_filter_keeps_significant_swings():
    df = _frame(HIGHS, LOWS)
    atr = pd.Series(1.0, index=df.index)
    cfg = SwingConfig(atr_filter_enabled=True, atr_filter_ratio=0.3)
    sw = detect_swings(df, cfg, atr=atr)
    assert not sw.empty


def test_atr_warmup_nan_is_refused_not_guessed():
    df = _frame(HIGHS, LOWS)
    atr = pd.Series([float("nan")] * len(df), index=df.index)
    cfg = SwingConfig(atr_filter_enabled=True, atr_filter_ratio=0.3)
    sw = detect_swings(df, cfg, atr=atr)
    assert sw.empty  # cannot establish significance → drop, no lookahead guess


def test_filter_enabled_without_atr_raises():
    df = _frame(HIGHS, LOWS)
    with pytest.raises(ValueError, match="ATR series is required"):
        detect_swings(df, SwingConfig(atr_filter_enabled=True))


def test_short_frame_returns_empty():
    df = _frame([1, 2], [0, 1])
    assert detect_swings(df, _no_filter_cfg()).empty
