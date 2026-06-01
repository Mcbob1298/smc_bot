"""Tests for the RiskManager veto core — the gate every order passes through."""

from datetime import UTC, date, datetime

import pytest

from config.strategy import RiskConfig
from risk.daily_guard import DailyLossGuard
from risk.exposure import ExposureTracker
from risk.position_sizer import SymbolSpec
from risk.risk_manager import RiskManager
from strategy.signal import Direction, Signal

NOW = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
KZ = "2026-06-01:london"


def _signal(**kw) -> Signal:
    base = dict(
        symbol="XAUUSD",
        direction=Direction.LONG,
        entry=2000.0,
        sl=1995.0,
        timestamp=NOW,
        tp1=2005.0,
        tp2=2010.0,
    )
    base.update(kw)
    return Signal(**base)


def _manager(equity=10_000, max_daily=3.0, **risk_kw):
    config = RiskConfig(**risk_kw)
    spec = SymbolSpec.xauusd_vantage()
    guard = DailyLossGuard(equity, max_daily, date(2026, 6, 1))
    exposure = ExposureTracker()
    return RiskManager(config, spec, guard, exposure), guard, exposure


def test_clean_signal_approved():
    rm, _, _ = _manager()
    d = rm.evaluate(_signal(), 10_000, now=NOW, news_blocked=False, killzone_id=KZ)
    assert d.approved
    assert d.lot == pytest.approx(0.20)  # 1% of 10k, 5$ stop
    assert d.sizing is not None
    assert any("sized" in r for r in d.reasons)


def test_daily_loss_veto():
    rm, guard, _ = _manager()
    guard.update(9_600)  # trip the switch
    d = rm.evaluate(_signal(), 9_600, now=NOW, news_blocked=False, killzone_id=KZ)
    assert not d.approved
    assert "kill switch" in d.reasons[0]


def test_news_blackout_veto():
    rm, _, _ = _manager()
    d = rm.evaluate(_signal(), 10_000, now=NOW, news_blocked=True, killzone_id=KZ)
    assert not d.approved
    assert "news" in d.reasons[0]


def test_exposure_veto():
    rm, _, exposure = _manager(max_concurrent_trades=1)
    exposure.register_open(1, "XAUUSD", "other-kz")
    d = rm.evaluate(_signal(), 10_000, now=NOW, news_blocked=False, killzone_id=KZ)
    assert not d.approved
    assert "concurrent" in d.reasons[0]


def test_rr_below_min_veto():
    # rr_min default 1.0; TP1 only 0.5R away → rejected.
    rm, _, _ = _manager()
    sig = _signal(tp1=2002.5, tp2=2010.0)  # 2.5$/5$ = 0.5R
    d = rm.evaluate(sig, 10_000, now=NOW, news_blocked=False, killzone_id=KZ)
    assert not d.approved
    assert "reward:risk" in d.reasons[0]


def test_sizing_refused_veto():
    # Small account + wide stop: min lot over-risks → rejected at sizing.
    rm, _, _ = _manager(equity=200, max_daily=3.0)
    sig = _signal(entry=2000.0, sl=1990.0, tp1=2010.0, tp2=None)  # 10$ stop
    d = rm.evaluate(sig, 200, now=NOW, news_blocked=False, killzone_id=KZ)
    assert not d.approved
    assert "stop too wide" in d.reasons[0]


def test_no_target_veto():
    rm, _, _ = _manager()
    sig = _signal(tp1=None, tp2=None, tp3=None)
    d = rm.evaluate(sig, 10_000, now=NOW, news_blocked=False, killzone_id=KZ)
    assert not d.approved
    assert "take-profit" in d.reasons[0]


def test_veto_order_daily_loss_first():
    # Both daily-loss tripped AND news blocked: daily loss reported first.
    rm, guard, _ = _manager()
    guard.update(9_600)
    d = rm.evaluate(_signal(), 9_600, now=NOW, news_blocked=True, killzone_id=KZ)
    assert "kill switch" in d.reasons[0]


def test_half_percent_risk_config():
    rm, _, _ = _manager(risk_per_trade_pct=0.5)
    d = rm.evaluate(_signal(), 10_000, now=NOW, news_blocked=False, killzone_id=KZ)
    assert d.approved
    assert d.lot == pytest.approx(0.10)
