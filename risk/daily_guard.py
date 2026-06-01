"""Daily-loss kill switch.

The single rule that the previous bot lacked, and the reason it gave back a
week of gains in one move: once the account is down a fixed percentage on the
day, the bot stops trading until the next session. No averaging in, no
"making it back".

The guard latches: once tripped it stays tripped for the rest of the day even
if equity recovers, because a recovering-then-re-dumping market is exactly the
trap that blows accounts. It resets only on an explicit new-day roll.

Drawdown is measured from the *day-start* equity and includes floating P&L
(via ``update(current_equity)``), so a large open loss trips the switch before
it is even realized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class DailyLossGuard:
    """Stateful per-day drawdown limiter with a latching trip.

    Args:
        day_start_equity: account equity at the start of the trading day.
        max_daily_loss_pct: drawdown from day-start that halts trading (e.g. 3.0).
        current_day: the trading day this guard is tracking.
    """

    day_start_equity: float
    max_daily_loss_pct: float
    current_day: date
    _tripped: bool = field(default=False, init=False)
    _low_watermark: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.day_start_equity <= 0:
            raise ValueError(f"day_start_equity must be positive, got {self.day_start_equity}")
        if not 0 < self.max_daily_loss_pct <= 100:
            raise ValueError(
                f"max_daily_loss_pct must be in (0, 100], got {self.max_daily_loss_pct}"
            )
        self._low_watermark = self.day_start_equity

    @property
    def loss_limit_amount(self) -> float:
        """Money the account may lose today before the switch trips."""
        return self.day_start_equity * self.max_daily_loss_pct / 100.0

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def max_drawdown_today(self) -> float:
        """Largest day-start→trough equity drop seen so far (>= 0)."""
        return self.day_start_equity - self._low_watermark

    def remaining_budget(self, current_equity: float) -> float:
        """Money left before the switch trips at ``current_equity`` (>= 0)."""
        used = self.day_start_equity - current_equity
        return max(0.0, self.loss_limit_amount - used)

    def update(self, current_equity: float) -> bool:
        """Feed the latest equity (including floating P&L). Returns ``is_tripped``.

        Call this on every tick/bar before considering a new entry. Once the
        drawdown from day-start reaches the limit, the guard latches tripped.
        """
        if current_equity < self._low_watermark:
            self._low_watermark = current_equity
        if self.day_start_equity - current_equity >= self.loss_limit_amount:
            self._tripped = True
        return self._tripped

    def roll_day(self, new_day: date, new_day_start_equity: float) -> None:
        """Start a fresh trading day: reset the latch and the watermark.

        No-op-guard: refuses to roll backwards in time to avoid accidentally
        clearing a trip with a stale timestamp.
        """
        if new_day < self.current_day:
            raise ValueError(
                f"cannot roll day backwards: {new_day} < current {self.current_day}"
            )
        if new_day_start_equity <= 0:
            raise ValueError(f"new_day_start_equity must be positive, got {new_day_start_equity}")
        self.current_day = new_day
        self.day_start_equity = new_day_start_equity
        self._low_watermark = new_day_start_equity
        self._tripped = False

    def maybe_roll(self, now: datetime, current_equity: float) -> None:
        """Roll the day automatically if ``now`` is past ``current_day``.

        Convenience for the live loop: the day boundary uses ``now.date()`` in
        whatever timezone the caller passes (the bot uses Europe/Paris).
        """
        if now.date() > self.current_day:
            self.roll_day(now.date(), current_equity)
