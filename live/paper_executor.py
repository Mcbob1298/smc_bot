"""Paper trading executor — wired to the RiskManager veto.

Simulates order execution against in-memory account state so the full pipeline
(strategy -> risk -> execution -> journal) can run end-to-end with no broker.
Crucially, **every** order request is routed through ``RiskManager.evaluate``
first: a rejected decision means no position is opened, full stop. This is the
same gate the live MT5 executor must use, so paper and live share one risk path.

Simplifications (honest paper, risk-first):
  - An accepted entry fills immediately at ``Signal.entry`` (SMC entries are
    limit orders at the OB/retracement; we assume price is there). Optional
    slippage can be applied to fills and exits.
  - Bar-based exit detection is *pessimistic*: within a bar that spans both the
    stop and a target, the stop is assumed hit first. Better to understate the
    edge than overstate it.
  - Partial closes use fractional lots (the simulation P&L is linear, so this
    is exact); the real sizer still rounds to broker steps at entry.

KNOWN LIMITATION (must fix before live — see docs/TODO_V2.md "realistic friction
model"): fills use the exact stop/target price with a single fixed ``slippage``.
This does NOT model XAU spread widening / slippage around news and session
opens, so measured demo expectancy sits slightly above reality. Fine for
proving an edge; not the number that clears the demo->real gate. Consume the
existing ``CostsConfig`` (variable spread + ATR-scaled slippage) before wiring
real MT5.

TP ladder: TP1 closes ``tp1_partial_pct`` and moves the stop to break-even;
with three targets the remainder is split 50/50 across TP2/TP3; with only TP2
the remainder closes there; with only TP1 the remainder rides to BE/stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from risk.position_sizer import SymbolSpec
from risk.risk_manager import RiskDecision, RiskManager
from strategy.signal import Direction, Signal

if TYPE_CHECKING:
    from journal.trade_logger import TradeLogger


@dataclass
class Fill:
    """A single (partial or full) execution against a position."""

    when: datetime
    price: float
    lot: float
    kind: str  # "tp1" | "tp2" | "tp3" | "sl" | "be" | "manual"
    pnl: float


@dataclass
class PaperPosition:
    """An open (or closed) simulated position and its lifecycle state."""

    ticket: int
    signal: Signal
    lot: float  # original size
    entry_price: float
    opened_at: datetime
    sl: float  # current stop (moves to BE after TP1)
    killzone_id: str | None
    remaining_lot: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False
    closed: bool = False
    realized_pnl: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.remaining_lot == 0.0 and not self.closed:
            self.remaining_lot = self.lot

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    @property
    def at_breakeven(self) -> bool:
        return self.tp1_done and self.sl == self.entry_price


@dataclass
class ExecutionResult:
    """Outcome of a ``submit`` call."""

    accepted: bool
    decision: RiskDecision
    position: PaperPosition | None = None


class PaperExecutor:
    """In-memory broker simulation gated by the RiskManager.

    Single-symbol by design (V1 = XAUUSD). ``balance`` is realized cash;
    ``equity`` adds floating P&L at a given mark price.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        spec: SymbolSpec,
        starting_balance: float,
        *,
        slippage: float = 0.0,
        logger: TradeLogger | None = None,
        first_ticket: int = 1000,
    ) -> None:
        if starting_balance <= 0:
            raise ValueError(f"starting_balance must be positive, got {starting_balance}")
        self.rm = risk_manager
        self.spec = spec
        self.balance = starting_balance
        self.slippage = slippage
        self.logger = logger
        self.open_positions: dict[int, PaperPosition] = {}
        self.closed_positions: list[PaperPosition] = []
        self._next_ticket = first_ticket

    # -- accounting ---------------------------------------------------------

    def _money(self, entry: float, exit_price: float, lot: float, sign: int) -> float:
        """Account-currency P&L of closing ``lot`` from ``entry`` to ``exit``."""
        ticks = (exit_price - entry) / self.spec.tick_size
        return sign * ticks * self.spec.tick_value * lot

    def floating_pnl(self, price: float) -> float:
        """Unrealized P&L of all open positions marked at ``price``."""
        return sum(
            self._money(p.entry_price, price, p.remaining_lot, p.direction.sign)
            for p in self.open_positions.values()
        )

    def equity(self, price: float) -> float:
        """Balance plus floating P&L at ``price``."""
        return self.balance + self.floating_pnl(price)

    # -- order entry --------------------------------------------------------

    def submit(
        self,
        signal: Signal,
        *,
        price: float,
        now: datetime,
        news_blocked: bool = False,
        killzone_id: str | None = None,
    ) -> ExecutionResult:
        """Route a signal through the risk veto and open it if approved.

        Args:
            signal: the trade idea.
            price: current market/mark price (for equity + fill reference).
            now: current time (Europe/Paris in the live loop).
            news_blocked: macro-news blackout flag from the data layer.
            killzone_id: active killzone identifier, or None.

        Returns:
            ``ExecutionResult``. On rejection, ``position`` is None and the
            decision carries the reasons.
        """
        equity = self.equity(price)
        decision = self.rm.evaluate(
            signal,
            equity,
            now=now,
            news_blocked=news_blocked,
            killzone_id=killzone_id,
        )
        if not decision.approved:
            if self.logger is not None:
                self.logger.log_rejection(signal, decision, now)
            return ExecutionResult(accepted=False, decision=decision)

        fill_price = self._apply_slippage(signal.entry, signal.direction, entering=True)
        ticket = self._next_ticket
        self._next_ticket += 1
        position = PaperPosition(
            ticket=ticket,
            signal=signal,
            lot=decision.lot,
            entry_price=fill_price,
            opened_at=now,
            sl=signal.sl,
            killzone_id=killzone_id,
        )
        self.open_positions[ticket] = position
        self.rm.exposure.register_open(ticket, signal.symbol, killzone_id)
        if self.logger is not None:
            self.logger.log_open(position, decision, now)
        return ExecutionResult(accepted=True, decision=decision, position=position)

    # -- market simulation --------------------------------------------------

    def on_bar(self, now: datetime, high: float, low: float, close: float) -> None:
        """Advance the simulation by one bar, resolving stops and targets.

        Processes every open position: stop first (pessimistic), then the TP
        ladder in order. Updates the daily-loss guard with floating equity so
        the kill switch can trip on open drawdown, and rolls the trading day.
        """
        self.rm.guard.maybe_roll(now, self.equity(close))

        for position in list(self.open_positions.values()):
            self._resolve_position(position, now, high, low)

        # Keep the kill switch current with end-of-bar floating equity.
        self.rm.guard.update(self.equity(close))

    def _resolve_position(
        self, pos: PaperPosition, now: datetime, high: float, low: float
    ) -> None:
        sign = pos.direction.sign

        # 1. Stop first (pessimistic). Long: low <= sl. Short: high >= sl.
        stop_hit = low <= pos.sl if sign > 0 else high >= pos.sl
        if stop_hit:
            kind = "be" if pos.at_breakeven else "sl"
            self._close_portion(pos, pos.remaining_lot, pos.sl, kind, now)
            return

        # 2. Take-profit ladder, in order, against the favourable bar extreme.
        extreme = high if sign > 0 else low
        sig = pos.signal

        if not pos.tp1_done and sig.tp1 is not None and _reached(extreme, sig.tp1, sign):
            self._take_tp1(pos, now)
            if pos.closed:
                return

        if not pos.tp2_done and sig.tp2 is not None and _reached(extreme, sig.tp2, sign):
            self._take_tp2(pos, now)
            if pos.closed:
                return

        if sig.tp3 is not None and _reached(extreme, sig.tp3, sign):
            self._close_portion(pos, pos.remaining_lot, sig.tp3, "tp3", now)

    # -- ladder steps -------------------------------------------------------

    def _take_tp1(self, pos: PaperPosition, now: datetime) -> None:
        cfg = self.rm.config
        sig = pos.signal
        # If TP1 is the only target, close the whole position there.
        if sig.tp2 is None and sig.tp3 is None:
            close_lot = pos.remaining_lot
        else:
            close_lot = min(pos.lot * cfg.tp1_partial_pct, pos.remaining_lot)
        self._close_portion(pos, close_lot, sig.tp1, "tp1", now)
        pos.tp1_done = True
        if not pos.closed and cfg.move_to_be_after_tp1:
            pos.sl = pos.entry_price  # move stop to break-even

    def _take_tp2(self, pos: PaperPosition, now: datetime) -> None:
        # With a TP3 defined, leave half the remainder to run; else close all.
        close_lot = pos.remaining_lot / 2.0 if pos.signal.tp3 is not None else pos.remaining_lot
        self._close_portion(pos, close_lot, pos.signal.tp2, "tp2", now)
        pos.tp2_done = True

    # -- closing ------------------------------------------------------------

    def _close_portion(
        self, pos: PaperPosition, lot: float, price: float, kind: str, now: datetime
    ) -> None:
        if lot <= 0:
            return
        lot = min(lot, pos.remaining_lot)
        exit_price = self._apply_slippage(price, pos.direction, entering=False)
        pnl = self._money(pos.entry_price, exit_price, lot, pos.direction.sign)
        self.balance += pnl
        pos.realized_pnl += pnl
        pos.remaining_lot = round(pos.remaining_lot - lot, 10)
        fill = Fill(when=now, price=exit_price, lot=lot, kind=kind, pnl=pnl)
        pos.fills.append(fill)
        if self.logger is not None:
            self.logger.log_fill(pos, fill, now)
        if pos.remaining_lot <= 0:
            self._finalize(pos, now)

    def _finalize(self, pos: PaperPosition, now: datetime) -> None:
        pos.closed = True
        pos.remaining_lot = 0.0
        self.open_positions.pop(pos.ticket, None)
        self.closed_positions.append(pos)
        self.rm.exposure.register_close(pos.ticket)
        if self.logger is not None:
            self.logger.log_close(pos, now)

    def close_all(self, now: datetime, price: float, kind: str = "manual") -> None:
        """Force-close every open position at ``price`` (e.g. Friday close)."""
        for pos in list(self.open_positions.values()):
            self._close_portion(pos, pos.remaining_lot, price, kind, now)

    # -- helpers ------------------------------------------------------------

    def _apply_slippage(self, price: float, direction: Direction, *, entering: bool) -> float:
        """Shift price against us by ``slippage`` (worse fills both ways)."""
        if self.slippage == 0.0:
            return price
        sign = direction.sign
        # Entering long: pay up (+); exiting long: get less (-). Mirror for short.
        adverse = sign if entering else -sign
        return price + adverse * self.slippage


def _reached(extreme: float, target: float, sign: int) -> bool:
    """True if the bar extreme reached ``target`` in the profit direction."""
    return extreme >= target if sign > 0 else extreme <= target
