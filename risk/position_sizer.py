"""Position sizing from stop distance and a fixed risk percentage.

The lot size is *derived*, never guessed: given the account equity, a fixed
risk fraction (0.5–1%), and the price distance to the stop-loss, there is
exactly one lot size that risks the intended amount. This module computes it
in broker-native terms (tick size / tick value, MT5 semantics) so the result
matches what MT5 will actually book.

Protective rounding: the raw lot is always rounded *down* to the broker's
volume step. Rounding up would silently risk more than the budget. If even the
broker minimum lot would exceed the risk budget (stop too wide for the
account), sizing returns ``lot == 0`` with a reason — the trade must be
skipped, not forced through at over-risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    """Broker contract specification needed to convert price risk into money.

    These mirror MT5 ``symbol_info`` fields. ``tick_value`` is the
    account-currency P&L of a one-tick move on a 1.0-lot position.
    """

    symbol: str
    tick_size: float  # minimum price increment (0.01 for XAUUSD on most brokers)
    tick_value: float  # account-ccy value of one tick per 1.0 lot
    volume_min: float  # smallest tradable lot
    volume_max: float  # largest tradable lot
    volume_step: float  # lot granularity
    contract_size: float = 100.0  # informational (oz per lot for XAU)

    @classmethod
    def xauusd_vantage(cls) -> SymbolSpec:
        """Sensible defaults for XAUUSD on Vantage (USD account).

        1 lot = 100 oz, tick 0.01 → a $1.00 move = $100 per lot, i.e. $1 per
        0.01 tick per lot. Override from live ``symbol_info`` in production —
        never hard-code these for real sizing.
        """
        return cls(
            symbol="XAUUSD",
            tick_size=0.01,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )


@dataclass(frozen=True)
class SizingResult:
    """Outcome of a sizing computation.

    ``lot == 0`` means the trade cannot be taken within the risk budget; the
    reason explains why. A positive lot is safe to submit (still subject to the
    risk manager's other vetoes).
    """

    lot: float
    risk_amount: float  # money budgeted for this trade
    risk_per_lot: float  # money lost per 1.0 lot if SL is hit
    sl_distance: float  # price distance entry→SL
    projected_risk: float  # money actually at risk at the chosen lot
    reason: str | None = None  # set when lot was clamped or refused

    @property
    def is_tradeable(self) -> bool:
        return self.lot > 0


def _round_down_to_step(value: float, step: float) -> float:
    """Round ``value`` down to the nearest multiple of ``step``.

    Uses a tiny epsilon so values landing on a step boundary through float
    error (e.g. 0.30000000004) aren't pushed down a whole step.
    """
    if step <= 0:
        raise ValueError(f"volume_step must be positive, got {step}")
    steps = math.floor(value / step + 1e-9)
    return round(steps * step, 10)


def compute_lot(
    equity: float,
    risk_pct: float,
    sl_distance: float,
    spec: SymbolSpec,
) -> SizingResult:
    """Compute the lot size that risks ``risk_pct``% of ``equity``.

    Args:
        equity: current account equity in account currency.
        risk_pct: fraction of equity to risk, in percent (0.5 → 0.5%).
        sl_distance: absolute price distance from entry to stop (> 0).
        spec: broker contract specification.

    Raises:
        ValueError: on non-positive equity, risk_pct outside (0, 100], or a
            non-positive stop distance — all programmer errors, surfaced loudly.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    if not 0 < risk_pct <= 100:
        raise ValueError(f"risk_pct must be in (0, 100], got {risk_pct}")
    if sl_distance <= 0:
        raise ValueError(f"sl_distance must be positive, got {sl_distance}")

    risk_amount = equity * risk_pct / 100.0
    # Money lost per 1.0 lot if the stop is hit.
    risk_per_lot = (sl_distance / spec.tick_size) * spec.tick_value
    raw_lot = risk_amount / risk_per_lot

    lot = _round_down_to_step(raw_lot, spec.volume_step)

    reason: str | None = None
    if lot < spec.volume_min:
        # Minimum lot would over-risk the budget → refuse, don't force it.
        return SizingResult(
            lot=0.0,
            risk_amount=risk_amount,
            risk_per_lot=risk_per_lot,
            sl_distance=sl_distance,
            projected_risk=0.0,
            reason=(
                f"min lot {spec.volume_min} risks "
                f"{spec.volume_min * risk_per_lot:.2f} > budget {risk_amount:.2f} "
                "(stop too wide for account)"
            ),
        )

    if lot > spec.volume_max:
        lot = spec.volume_max
        reason = f"clamped to volume_max {spec.volume_max}"

    projected_risk = lot * risk_per_lot
    return SizingResult(
        lot=lot,
        risk_amount=risk_amount,
        risk_per_lot=risk_per_lot,
        sl_distance=sl_distance,
        projected_risk=projected_risk,
        reason=reason,
    )
