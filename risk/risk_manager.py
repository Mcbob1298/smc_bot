"""Risk manager — the protective core with veto power over every order.

Nothing reaches the broker without passing through ``RiskManager.evaluate``.
It runs every non-negotiable check in order and, on the first failure, returns
a *rejected* decision. The executor must treat a rejection as final: no order,
no override. This is the layer the whole bot is built around.

Checks, in order (cheapest / most decisive first):
  1. Daily-loss kill switch is tripped               → reject (halt for the day)
  2. Signal has a stop-loss on the correct side      → reject if not
  3. Inside a macro-news blackout window             → reject
  4. Exposure caps (concurrent / per-killzone)       → reject
  5. Reward:risk below the configured minimum        → reject
  6. Position sizing produces a tradeable lot        → reject if min lot over-risks

A passing decision carries the computed lot and the full reason trail so the
journal can record *why* the trade was allowed, not just that it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from config.strategy import RiskConfig

from .daily_guard import DailyLossGuard
from .exposure import ExposureTracker
from .position_sizer import SizingResult, SymbolSpec, compute_lot

if TYPE_CHECKING:
    # Imported for typing only: risk stays decoupled from strategy at runtime.
    # RiskManager duck-types on .direction/.entry/.sl/.risk_distance/.tp1.
    from strategy.signal import Signal


@dataclass(frozen=True)
class RiskDecision:
    """Verdict for a single proposed trade.

    ``approved`` is the only thing the executor may key on. ``reasons`` always
    explains the verdict (rejections list the failed check; approvals list what
    was checked and the sizing summary).
    """

    approved: bool
    lot: float
    reasons: list[str] = field(default_factory=list)
    sizing: SizingResult | None = None

    @classmethod
    def reject(cls, reason: str) -> RiskDecision:
        return cls(approved=False, lot=0.0, reasons=[reason], sizing=None)


class RiskManager:
    """Aggregates the risk sub-systems and exercises the veto.

    The manager owns no market data and makes no directional call — it only
    decides whether a given Signal may be executed and at what size.
    """

    def __init__(
        self,
        config: RiskConfig,
        spec: SymbolSpec,
        guard: DailyLossGuard,
        exposure: ExposureTracker,
    ) -> None:
        self.config = config
        self.spec = spec
        self.guard = guard
        self.exposure = exposure

    def evaluate(
        self,
        signal: Signal,
        account_equity: float,
        *,
        now: datetime,
        news_blocked: bool,
        killzone_id: str | None,
    ) -> RiskDecision:
        """Run every veto check and, if all pass, size the position.

        Args:
            signal: the proposed trade idea.
            account_equity: current account equity (account currency).
            now: current time (caller's reference tz, typically Europe/Paris).
            news_blocked: True if inside a macro-news blackout window. The
                strategy/data layer owns the calendar; the risk layer only
                honours the flag.
            killzone_id: identifier of the active killzone, or None if outside one.

        Returns:
            A ``RiskDecision``. ``approved=True`` carries a positive ``lot``.
        """
        # Keep the kill switch current with the latest equity (floating P&L
        # included) before deciding anything.
        self.guard.update(account_equity)

        # 1. Daily-loss kill switch — the master veto.
        if self.guard.is_tripped:
            return RiskDecision.reject(
                f"daily-loss kill switch tripped "
                f"(drawdown {self.guard.max_drawdown_today:.2f} >= "
                f"limit {self.guard.loss_limit_amount:.2f})"
            )

        # 2. No order without a stop. Signal guarantees this at construction,
        #    but the risk layer re-asserts it rather than trusting the producer.
        if signal.sl <= 0 or signal.risk_distance <= 0:
            return RiskDecision.reject("signal has no valid stop-loss")
        sign = signal.direction.sign
        if sign * (signal.entry - signal.sl) <= 0:
            return RiskDecision.reject("stop-loss is on the wrong side of entry")

        # 3. Macro-news blackout.
        if news_blocked:
            return RiskDecision.reject("inside macro-news blackout window")

        # 4. Exposure caps (concurrency + anti-revenge per killzone).
        allowed, why = self.exposure.can_open(
            max_concurrent=self.config.max_concurrent_trades,
            max_per_killzone=self.config.max_trades_per_killzone,
            killzone_id=killzone_id,
        )
        if not allowed:
            return RiskDecision.reject(why or "exposure cap reached")

        # 5. Reward:risk floor — checked against TP1 (the first target that must
        #    at least cover risk). Signals without any TP are rejected: an idea
        #    with no target is not executable.
        if signal.first_target is None:
            return RiskDecision.reject("signal has no take-profit target")
        first_rr = signal.rr_to(signal.first_target)
        if first_rr < self.config.rr_min:
            return RiskDecision.reject(
                f"reward:risk {first_rr:.2f} below minimum {self.config.rr_min:.2f}"
            )

        # 6. Position sizing — last because it is the most expensive check and
        #    only meaningful once the trade is otherwise allowed.
        sizing = compute_lot(
            equity=account_equity,
            risk_pct=self.config.risk_per_trade_pct,
            sl_distance=signal.risk_distance,
            spec=self.spec,
        )
        if not sizing.is_tradeable:
            return RiskDecision.reject(sizing.reason or "sizing produced zero lot")

        reasons = [
            "daily-loss switch OK",
            "stop-loss present and valid",
            "no news blackout",
            "within exposure caps",
            f"R:R {first_rr:.2f} >= min {self.config.rr_min:.2f}",
            f"sized {sizing.lot} lot risking {sizing.projected_risk:.2f} "
            f"({self.config.risk_per_trade_pct}% of {account_equity:.2f})",
        ]
        if sizing.reason:
            reasons.append(sizing.reason)
        return RiskDecision(approved=True, lot=sizing.lot, reasons=reasons, sizing=sizing)
