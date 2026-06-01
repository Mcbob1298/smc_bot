"""Tests for the Signal dataclass — the strategy→risk contract.

A Signal must guarantee a valid stop on the losing side and coherent,
monotonic take-profits. Construction is the first gate enforcing
"no idea without an invalidation level".
"""

from datetime import UTC, datetime

import pytest

from strategy.signal import Direction, Signal

TS = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


def _long(**kw) -> Signal:
    base = dict(
        symbol="XAUUSD",
        direction=Direction.LONG,
        entry=2000.0,
        sl=1995.0,
        timestamp=TS,
        tp1=2005.0,
        tp2=2010.0,
        tp3=2020.0,
    )
    base.update(kw)
    return Signal(**base)


def _short(**kw) -> Signal:
    base = dict(
        symbol="XAUUSD",
        direction=Direction.SHORT,
        entry=2000.0,
        sl=2005.0,
        timestamp=TS,
        tp1=1995.0,
        tp2=1990.0,
        tp3=1980.0,
    )
    base.update(kw)
    return Signal(**base)


def test_valid_long_signal_constructs():
    s = _long()
    assert s.is_long
    assert s.risk_distance == pytest.approx(5.0)
    assert s.tp1_rr == pytest.approx(1.0)  # TP1 at exactly 1R
    assert s.rr_to(2010.0) == pytest.approx(2.0)


def test_valid_short_signal_constructs():
    s = _short()
    assert not s.is_long
    assert s.risk_distance == pytest.approx(5.0)
    assert s.tp1_rr == pytest.approx(1.0)
    assert s.rr_to(1990.0) == pytest.approx(2.0)


def test_direction_sign():
    assert Direction.LONG.sign == 1
    assert Direction.SHORT.sign == -1


def test_long_sl_above_entry_rejected():
    with pytest.raises(ValueError, match="SL must be below"):
        _long(sl=2001.0)


def test_short_sl_below_entry_rejected():
    with pytest.raises(ValueError, match="SL must be above"):
        _short(sl=1999.0)


def test_sl_equal_to_entry_rejected():
    with pytest.raises(ValueError):
        _long(sl=2000.0)


def test_non_positive_prices_rejected():
    with pytest.raises(ValueError):
        _long(entry=-1.0)
    with pytest.raises(ValueError):
        _long(sl=0.0)


def test_long_non_monotonic_tps_rejected():
    # TP2 below TP1 for a long is incoherent.
    with pytest.raises(ValueError, match="tp2"):
        _long(tp2=2004.0)


def test_long_tp1_not_beyond_entry_rejected():
    with pytest.raises(ValueError, match="tp1"):
        _long(tp1=1999.0)


def test_optional_tps_none_allowed():
    s = _long(tp2=None, tp3=None)
    assert s.tp2 is None
    assert s.first_target == 2005.0


def test_first_target_falls_through_to_tp2():
    s = _long(tp1=None)
    assert s.first_target == 2010.0


def test_reasons_map_carried():
    s = _long(reasons={"entry": "M15 bullish OB retest", "sl": "below OB low + buffer"})
    assert s.reasons["sl"].startswith("below OB")


def test_signal_is_frozen():
    s = _long()
    with pytest.raises(Exception):
        s.entry = 2001.0  # type: ignore[misc]
