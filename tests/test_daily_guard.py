"""Tests for the daily-loss kill switch."""

from datetime import date, datetime

import pytest

from risk.daily_guard import DailyLossGuard

DAY = date(2026, 6, 1)


def _guard(equity=10_000, pct=3.0):
    return DailyLossGuard(day_start_equity=equity, max_daily_loss_pct=pct, current_day=DAY)


def test_not_tripped_within_budget():
    g = _guard()
    assert g.loss_limit_amount == pytest.approx(300.0)
    assert not g.update(9_800)  # -200, within 300 budget
    assert not g.is_tripped
    assert g.remaining_budget(9_800) == pytest.approx(100.0)


def test_trips_at_limit():
    g = _guard()
    assert g.update(9_700)  # -300 exactly
    assert g.is_tripped
    assert g.remaining_budget(9_700) == 0.0


def test_trips_on_floating_loss_beyond_limit():
    g = _guard()
    assert g.update(9_500)  # -500, past limit
    assert g.is_tripped


def test_latches_after_recovery():
    g = _guard()
    g.update(9_600)  # trips
    assert g.is_tripped
    g.update(10_100)  # recovers above start
    assert g.is_tripped  # stays tripped — no re-entry on recovery


def test_low_watermark_tracked():
    g = _guard()
    g.update(9_900)
    g.update(9_400)
    g.update(9_950)
    assert g.max_drawdown_today == pytest.approx(600.0)


def test_roll_day_resets():
    g = _guard()
    g.update(9_500)  # trips
    g.roll_day(date(2026, 6, 2), 9_500)
    assert not g.is_tripped
    assert g.day_start_equity == 9_500
    assert g.loss_limit_amount == pytest.approx(285.0)


def test_roll_day_backwards_rejected():
    g = _guard()
    with pytest.raises(ValueError, match="backwards"):
        g.roll_day(date(2026, 5, 31), 10_000)


def test_maybe_roll_auto_rolls_on_new_day():
    g = _guard()
    g.update(9_500)  # trips today
    g.maybe_roll(datetime(2026, 6, 2, 0, 5), 9_500)
    assert not g.is_tripped
    assert g.current_day == date(2026, 6, 2)


def test_maybe_roll_noop_same_day():
    g = _guard()
    g.update(9_500)
    g.maybe_roll(datetime(2026, 6, 1, 23, 0), 9_500)
    assert g.is_tripped  # same day, no reset


def test_invalid_construction():
    with pytest.raises(ValueError):
        DailyLossGuard(day_start_equity=0, max_daily_loss_pct=3.0, current_day=DAY)
    with pytest.raises(ValueError):
        DailyLossGuard(day_start_equity=10_000, max_daily_loss_pct=0, current_day=DAY)
