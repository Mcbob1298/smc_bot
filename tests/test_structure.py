"""Tests for BOS / ChoCh structure detection (causal)."""

import pandas as pd

from config.strategy import StructureConfig, SwingConfig
from detectors.structure import detect_structure
from detectors.swings import detect_swings

# H, L, C per bar. Up-leg breaks a swing high (BOS bull), then a later close
# below the most recent swing low flips character (ChoCh bear).
ROWS = [
    (6, 4, 5),        # 0
    (8, 5, 7),        # 1
    (10, 7, 9),       # 2  swing high = 10
    (8, 6, 7),        # 3  swing low = 6
    (12, 7, 11),      # 4  close 11 > 10  -> BOS bull
    (14, 11, 13),     # 5  swing high = 14
    (13, 11, 12),     # 6
    (12, 10, 11),     # 7  swing low = 10
    (11, 10.5, 10.8), # 8
    (10.5, 7, 8),     # 9  close 8 < 10  -> ChoCh bear
    (9, 6, 7),        # 10
]


def _frame(rows=ROWS):
    idx = pd.date_range("2026-06-01 08:00", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, index=idx, columns=["high", "low", "close"])
    df["open"] = df["close"]
    return df


def _swings(df):
    return detect_swings(
        df, SwingConfig(atr_filter_enabled=False, atr_filter_enabled_ltf=False)
    )


def test_bos_then_choch_sequence():
    df = _frame()
    st = detect_structure(df, _swings(df), StructureConfig())
    assert list(zip(st["kind"], st["direction"])) == [
        ("BOS", "bullish"),
        ("ChoCh", "bearish"),
    ]
    assert st.iloc[0]["broken_level"] == 10.0
    assert st.iloc[1]["broken_level"] == 10.0


def test_break_confirmed_on_breaking_bar_and_after_swing():
    df = _frame()
    st = detect_structure(df, _swings(df), StructureConfig())
    assert (st["confirmed_at"] == st.index).all()
    assert (st["confirmed_at"] > st["broken_swing_time"]).all()


def test_close_vs_wick_break_definition():
    # A bar whose wick pokes above the swing high but closes back below it:
    # break_on_close=True ignores it, break_on_close=False (wick) takes it.
    rows = [
        (6, 4, 5),
        (8, 5, 7),
        (10, 7, 9),    # swing high 10
        (8, 6, 7),
        (11, 7, 9.5),  # wick to 11 > 10 but closes 9.5 < 10
    ]
    df = _frame(rows)
    sw = _swings(df)
    assert detect_structure(df, sw, StructureConfig(break_on_close=True)).empty
    wick = detect_structure(df, sw, StructureConfig(break_on_close=False))
    assert len(wick) == 1
    assert wick.iloc[0]["direction"] == "bullish"


def test_buffer_requires_atr_when_enabled():
    df = _frame()
    import pytest

    with pytest.raises(ValueError, match="ATR series is required"):
        detect_structure(df, _swings(df), StructureConfig(break_buffer_atr_ratio=0.1))


def test_buffer_blocks_marginal_break():
    # Close 11 is only 1.0 above the swing (10). A buffer of 0.2*ATR(10)=2.0
    # makes it insufficient → no break.
    df = _frame()
    atr = pd.Series(10.0, index=df.index)
    blocked = detect_structure(
        df, _swings(df), StructureConfig(break_buffer_atr_ratio=0.2), atr=atr
    )
    # The first (bullish) break at +1.0 is filtered; only later breaks (if any
    # clear the buffer) remain. The marginal BOS at bar 4 must be gone.
    assert df.index[4] not in list(blocked.index)


def test_no_swings_no_structure():
    df = _frame()
    empty = df.iloc[:0].copy()
    st = detect_structure(df, _swings(empty), StructureConfig())
    assert st.empty
