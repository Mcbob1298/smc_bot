"""Tests for order block detection (composite, causal)."""

import pandas as pd

from config.strategy import (
    FVGConfig,
    OrderBlockConfig,
    StructureConfig,
    SweepConfig,
    SwingConfig,
)
from detectors.fvg import detect_fvgs
from detectors.order_blocks import detect_order_blocks
from detectors.structure import detect_structure
from detectors.sweeps import detect_sweeps
from detectors.swings import detect_swings

NF = SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False)

ROWS = [
    (6, 4, 5), (8, 5, 7), (10, 7, 9), (8, 6, 7), (12, 7, 11),
    (14, 11, 13), (13, 11, 12), (12, 10, 11), (11, 10.5, 10.8),
    (10.5, 7, 8), (9, 6, 7),
]
OPENS = [6, 6, 8, 8.0, 7, 11, 13, 12, 11, 10.5, 9]  # bar 3 (open8>close7) = bearish OB


def _pipeline(ob_cfg):
    idx = pd.date_range("2026-06-01 08:00", periods=len(ROWS), freq="5min", tz="UTC")
    df = pd.DataFrame(ROWS, index=idx, columns=["high", "low", "close"])
    df["open"] = OPENS
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    st = detect_structure(df, sw, StructureConfig())
    fv = detect_fvgs(df, FVGConfig(min_size_atr_ratio=0.1), atr)
    swp = detect_sweeps(df, sw, SweepConfig(), atr)
    return df, detect_order_blocks(df, st, fv, swp, ob_cfg)


def test_bullish_ob_is_last_bearish_candle_before_break():
    df, ob = _pipeline(
        OrderBlockConfig(require_fvg=False, require_prior_liquidity_sweep=False)
    )
    bull = ob[ob["kind"] == "bullish"]
    assert len(bull) == 1
    row = bull.iloc[0]
    assert bull.index[0] == df.index[3]          # the bearish candle at bar 3
    assert (row["top"], row["bottom"]) == (8.0, 6.0)  # full range
    assert row["break_time"] == df.index[4]      # validated by the BOS at bar 4
    assert row["confirmed_at"] >= bull.index[0]  # causal


def test_zone_definition_body_only():
    df, ob = _pipeline(
        OrderBlockConfig(
            require_fvg=False, require_prior_liquidity_sweep=False,
            body_or_full_range="body_only",
        )
    )
    bull = ob[ob["kind"] == "bullish"].iloc[0]
    # bar 3: open 8, close 7 → body [7, 8].
    assert (bull["top"], bull["bottom"]) == (8.0, 7.0)


def test_require_fvg_filters_when_absent():
    # No FVGs supplied + require_fvg → nothing validates.
    idx = pd.date_range("2026-06-01 08:00", periods=len(ROWS), freq="5min", tz="UTC")
    df = pd.DataFrame(ROWS, index=idx, columns=["high", "low", "close"])
    df["open"] = OPENS
    atr = pd.Series(1.0, index=df.index)
    sw = detect_swings(df, NF)
    st = detect_structure(df, sw, StructureConfig())
    empty_fvg = detect_fvgs(df, FVGConfig(min_size_atr_ratio=99.0), atr)  # filtered out
    swp = detect_sweeps(df, sw, SweepConfig(), atr)
    ob = detect_order_blocks(
        df, st, empty_fvg, swp, OrderBlockConfig(require_fvg=True,
                                                 require_prior_liquidity_sweep=False)
    )
    assert ob.empty


def test_require_prior_sweep_filters_when_absent():
    df, ob = _pipeline(
        OrderBlockConfig(require_fvg=False, require_prior_liquidity_sweep=True)
    )
    # No prior same-direction sweep in this fixture → filtered out.
    assert ob.empty


def test_no_structure_no_obs():
    idx = pd.date_range("2026-06-01 08:00", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame(
        [[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]],
        index=idx, columns=["open", "high", "low", "close"],
    )
    empty = df.iloc[:0]
    ob = detect_order_blocks(df, empty, empty, empty, OrderBlockConfig())
    assert ob.empty
