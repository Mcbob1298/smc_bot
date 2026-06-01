"""Tests for position sizing — derived lot from risk %, stop distance, contract."""

import pytest

from risk.position_sizer import SymbolSpec, compute_lot

XAU = SymbolSpec.xauusd_vantage()


def test_basic_xau_sizing():
    # 10_000 equity, 1% = 100$ risk. SL distance 5.00$ → 5.00/0.01 * 1.0 = 500$/lot.
    # raw = 100/500 = 0.20 lot.
    r = compute_lot(equity=10_000, risk_pct=1.0, sl_distance=5.0, spec=XAU)
    assert r.lot == pytest.approx(0.20)
    assert r.risk_amount == pytest.approx(100.0)
    assert r.risk_per_lot == pytest.approx(500.0)
    assert r.projected_risk == pytest.approx(100.0)
    assert r.is_tradeable


def test_half_percent_risk():
    r = compute_lot(equity=10_000, risk_pct=0.5, sl_distance=5.0, spec=XAU)
    assert r.lot == pytest.approx(0.10)
    assert r.projected_risk == pytest.approx(50.0)


def test_rounds_down_never_over_risk():
    # raw lot would be 0.234..., must floor to 0.23 (never 0.24) so projected
    # risk never exceeds budget.
    r = compute_lot(equity=10_000, risk_pct=1.0, sl_distance=4.27, spec=XAU)
    assert r.lot == pytest.approx(0.23)
    assert r.projected_risk <= r.risk_amount + 1e-9


def test_wide_stop_small_account_refused():
    # Tiny account, wide stop: even 0.01 lot over-risks → refuse with reason.
    r = compute_lot(equity=200, risk_pct=1.0, sl_distance=10.0, spec=XAU)
    assert r.lot == 0.0
    assert not r.is_tradeable
    assert r.reason is not None and "stop too wide" in r.reason


def test_min_lot_exactly_affordable():
    # Budget exactly equals min-lot risk: 0.01 lot risks 0.01*1000 = 10$.
    # equity*risk% = 10 → equity 1000 at 1%. SL distance 10$ → risk_per_lot 1000.
    r = compute_lot(equity=1_000, risk_pct=1.0, sl_distance=10.0, spec=XAU)
    assert r.lot == pytest.approx(0.01)


def test_clamp_to_volume_max():
    spec = SymbolSpec("XAUUSD", 0.01, 1.0, 0.01, 0.05, 0.01)
    r = compute_lot(equity=10_000_000, risk_pct=1.0, sl_distance=1.0, spec=spec)
    assert r.lot == pytest.approx(0.05)
    assert r.reason is not None and "volume_max" in r.reason


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_lot(equity=0, risk_pct=1.0, sl_distance=5.0, spec=XAU)
    with pytest.raises(ValueError):
        compute_lot(equity=10_000, risk_pct=0, sl_distance=5.0, spec=XAU)
    with pytest.raises(ValueError):
        compute_lot(equity=10_000, risk_pct=101, sl_distance=5.0, spec=XAU)
    with pytest.raises(ValueError):
        compute_lot(equity=10_000, risk_pct=1.0, sl_distance=0.0, spec=XAU)
