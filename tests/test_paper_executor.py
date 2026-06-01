"""Tests for the paper executor wired to the RiskManager.

Covers the veto wiring (no position on rejection), the TP1/2/3 ladder with
break-even, pessimistic stop-first resolution, kill-switch on floating loss,
and basic P&L accounting.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from config.strategy import RiskConfig
from journal.trade_logger import TradeLogger
from live.paper_executor import PaperExecutor
from risk.daily_guard import DailyLossGuard
from risk.exposure import ExposureTracker
from risk.position_sizer import SymbolSpec
from risk.risk_manager import RiskManager
from strategy.signal import Direction, Signal

START = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
KZ = "2026-06-01:london"
SPEC = SymbolSpec.xauusd_vantage()


def _signal(**kw) -> Signal:
    base = dict(
        symbol="XAUUSD",
        direction=Direction.LONG,
        entry=2000.0,
        sl=1995.0,
        timestamp=START,
        tp1=2005.0,
        tp2=2010.0,
        tp3=2020.0,
    )
    base.update(kw)
    return Signal(**base)


def _executor(equity=10_000, max_daily=3.0, balance=None, **risk_kw):
    config = RiskConfig(**risk_kw)
    guard = DailyLossGuard(equity, max_daily, date(2026, 6, 1))
    rm = RiskManager(config, SPEC, guard, ExposureTracker())
    return PaperExecutor(rm, SPEC, balance or equity)


def _later(minutes: int) -> datetime:
    return START + timedelta(minutes=minutes)


# -- veto wiring -----------------------------------------------------------


def test_rejected_signal_opens_nothing():
    ex = _executor()
    res = ex.submit(_signal(), price=2000.0, now=START, news_blocked=True, killzone_id=KZ)
    assert not res.accepted
    assert res.position is None
    assert ex.open_positions == {}
    assert any("news" in r for r in res.decision.reasons)


def test_accepted_signal_opens_position_and_registers_exposure():
    ex = _executor()
    res = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ)
    assert res.accepted
    assert res.position is not None
    assert res.position.lot == pytest.approx(0.20)  # 1% of 10k, 5$ stop
    assert ex.open_positions[res.position.ticket] is res.position
    assert ex.rm.exposure.open_count == 1


def test_second_trade_same_killzone_blocked_by_exposure():
    ex = _executor()  # default max_trades_per_killzone=1
    ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ)
    res2 = ex.submit(_signal(), price=2000.0, now=_later(5), killzone_id=KZ)
    assert not res2.accepted
    assert any("killzone" in r for r in res2.decision.reasons)


# -- TP ladder + break-even ------------------------------------------------


def test_tp1_partial_close_and_move_to_be():
    ex = _executor()
    pos = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2006.0, low=1999.0, close=2004.0)  # tags TP1 only
    assert pos.tp1_done
    assert pos.remaining_lot == pytest.approx(0.10)  # 50% closed
    assert pos.sl == pytest.approx(2000.0)  # moved to break-even
    # TP1 at 2005 with 0.10 lot: (5/0.01)*1*0.10 = 50$
    assert ex.balance == pytest.approx(10_050.0)


def test_full_ladder_tp1_tp2_tp3():
    ex = _executor()
    pos = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2005.5, low=1999.0, close=2005.0)  # TP1
    ex.on_bar(_later(10), high=2010.5, low=2004.0, close=2010.0)  # TP2
    ex.on_bar(_later(15), high=2020.5, low=2009.0, close=2020.0)  # TP3
    assert pos.closed
    # 0.20 lot: TP1 0.10@+5 =50; TP2 0.05@+10 =50; TP3 0.05@+20 =100 -> +200
    assert pos.realized_pnl == pytest.approx(200.0)
    assert ex.balance == pytest.approx(10_200.0)
    assert ex.rm.exposure.open_count == 0


def test_stop_loss_full_close_at_risk_amount():
    ex = _executor()
    pos = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2001.0, low=1994.0, close=1995.0)  # hits SL 1995
    assert pos.closed
    # 0.20 lot, -5$ -> -100 (the budgeted 1% risk)
    assert pos.realized_pnl == pytest.approx(-100.0)
    assert ex.balance == pytest.approx(9_900.0)


def test_breakeven_stop_after_tp1_is_scratch():
    ex = _executor()
    pos = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2005.5, low=1999.0, close=2004.0)  # TP1 -> BE
    ex.on_bar(_later(10), high=2002.0, low=1999.5, close=2000.0)  # back to BE stop
    assert pos.closed
    # Only the TP1 partial profit remains; BE exit on remainder = 0.
    assert pos.realized_pnl == pytest.approx(50.0)
    assert pos.fills[-1].kind == "be"


def test_pessimistic_stop_first_when_bar_spans_both():
    ex = _executor()
    pos = ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    # Bar spans both SL (1995) and TP1 (2005): stop assumed first.
    ex.on_bar(_later(5), high=2006.0, low=1994.0, close=2000.0)
    assert pos.closed
    assert pos.realized_pnl == pytest.approx(-100.0)


def test_tp1_only_signal_closes_fully_at_tp1():
    ex = _executor()
    sig = _signal(tp1=2005.0, tp2=None, tp3=None)
    pos = ex.submit(sig, price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2005.5, low=1999.0, close=2005.0)
    assert pos.closed
    assert pos.realized_pnl == pytest.approx(100.0)  # full 0.20 lot @ +5$


# -- short side ------------------------------------------------------------


def test_short_tp1_and_be():
    ex = _executor()
    sig = _signal(direction=Direction.SHORT, entry=2000.0, sl=2005.0,
                  tp1=1995.0, tp2=1990.0, tp3=1980.0)
    pos = ex.submit(sig, price=2000.0, now=START, killzone_id=KZ).position
    assert pos is not None
    ex.on_bar(_later(5), high=2001.0, low=1994.5, close=1996.0)  # TP1 short
    assert pos.tp1_done
    assert pos.sl == pytest.approx(2000.0)
    assert ex.balance == pytest.approx(10_050.0)


# -- kill switch -----------------------------------------------------------


def test_accumulated_losses_trip_kill_switch_and_block_next():
    # A single 1%-risk trade can't lose 3% (its stop caps it). The daily switch
    # trips on the *cumulative realized* drawdown of several losers.
    ex = _executor(equity=10_000, max_daily=3.0)
    i = 0
    while not ex.rm.guard.is_tripped and i < 10:
        t = _later(i * 30)
        res = ex.submit(_signal(), price=2000.0, now=t, killzone_id=f"kz-{i}")
        if res.accepted:
            ex.on_bar(t + timedelta(minutes=5), high=2001.0, low=1994.0, close=1995.0)  # SL
        i += 1
    assert ex.rm.guard.is_tripped
    assert ex.balance <= 10_000 - ex.rm.guard.loss_limit_amount  # >= 3% down
    # Next trade is vetoed by the kill switch.
    res = ex.submit(_signal(), price=1995.0, now=_later(400), killzone_id="kz-last")
    assert not res.accepted
    assert any("kill switch" in r for r in res.decision.reasons)


# -- accounting / utilities ------------------------------------------------


def test_equity_reflects_floating_pnl():
    ex = _executor()
    ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ)
    assert ex.equity(2003.0) == pytest.approx(10_060.0)  # +3$ * 0.20 lot *100
    assert ex.equity(1998.0) == pytest.approx(9_960.0)


def test_close_all_force_closes():
    ex = _executor()
    ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ)
    ex.close_all(_later(30), price=2002.0)
    assert ex.open_positions == {}
    assert ex.balance == pytest.approx(10_040.0)


def test_logger_writes_jsonl(tmp_path):
    logf = tmp_path / "trades.jsonl"
    config = RiskConfig()
    guard = DailyLossGuard(10_000, 3.0, date(2026, 6, 1))
    rm = RiskManager(config, SPEC, guard, ExposureTracker())
    ex = PaperExecutor(rm, SPEC, 10_000, logger=TradeLogger(logf))
    ex.submit(_signal(), price=2000.0, now=START, killzone_id=KZ)
    ex.on_bar(_later(5), high=2005.5, low=1999.0, close=2004.0)  # TP1 partial
    lines = logf.read_text().strip().splitlines()
    events = [line.split('"event": "')[1].split('"')[0] for line in lines]
    assert "open" in events
    assert "fill" in events
