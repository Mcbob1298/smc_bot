"""Tests for liquidity sweep (stop-hunt) detection (causal)."""

import pandas as pd

from config.strategy import SweepConfig, SwingConfig
from detectors.sweeps import detect_sweeps
from detectors.swings import detect_swings

NF = SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False)


def _frame(rows):
    idx = pd.date_range("2026-06-01 09:00", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, index=idx, columns=["high", "low", "close"])
    df["open"] = df["close"]
    return df


# Swing low = 5 at bar 2; bar 4 pierces to 4.5 then closes back at 6 → bullish sweep.
ROWS = [(9, 7, 8), (8, 6, 7), (8, 5, 6), (7, 6, 6.5), (7, 4.5, 6), (8, 6, 7)]


def test_bullish_sweep_of_swing_low():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    swp = detect_sweeps(df, sw, SweepConfig(min_penetration_atr_ratio=0.05), atr)
    assert len(swp) == 1
    row = swp.iloc[0]
    assert row["kind"] == "bullish"
    assert row["swept_level"] == 5.0
    assert row["penetration"] == 0.5
    assert row["confirmed_at"] == swp.index[0]  # known on the sweeping bar


def test_close_back_required_rejects_break_through():
    # Same pierce but the bar closes BELOW the level (4.8) → not a sweep when
    # require_close_back; it would be a plain break.
    rows = [(9, 7, 8), (8, 6, 7), (8, 5, 6), (7, 6, 6.5), (7, 4.5, 4.8), (5, 4, 4.5)]
    df = _frame(rows)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    assert detect_sweeps(df, sw, SweepConfig(require_close_back=True), atr).empty
    assert not detect_sweeps(df, sw, SweepConfig(require_close_back=False), atr).empty


def test_penetration_threshold_filters_shallow_poke():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    # Require 0.6$ penetration; actual is 0.5$ → filtered.
    swp = detect_sweeps(df, sw, SweepConfig(min_penetration_atr_ratio=0.6), atr)
    assert swp.empty


def test_lookback_window_expires_old_levels():
    df = _frame(ROWS)
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    # Lookback 1 bar: the swing low (bar 2) is too old when swept at bar 4.
    swp = detect_sweeps(df, sw, SweepConfig(lookback_bars=1), atr)
    assert swp.empty
