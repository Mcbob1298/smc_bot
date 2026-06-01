"""Trade Signal — the contract between strategy and risk/execution layers.

A Signal is an *idea*: a direction, an entry, the price that invalidates the
idea (SL), and one or more take-profit targets. It carries a ``reasons`` map
explaining *why* each level sits where it does (SMC justification), so the
journal can annotate charts and so a human can audit every decision.

Hard invariant: a Signal cannot exist without a stop-loss placed on the
*losing* side of entry. An idea with no invalidation level is not a trade —
constructing such a Signal raises ``ValueError``. This is the first line of
the "no order without a stop" rule; the risk layer enforces the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 for LONG, -1 for SHORT. Profit moves in this direction."""
        return 1 if self is Direction.LONG else -1


# Canonical keys for the ``reasons`` map. Free-form values, fixed keys so the
# journal/chart annotator can rely on them being present when relevant.
REASON_KEYS = ("bias", "entry", "sl", "tp1", "tp2", "tp3")


@dataclass(frozen=True)
class Signal:
    """An immutable trade idea produced by the strategy layer.

    Prices are in the instrument's quote currency (USD for XAUUSD). ``tp2`` and
    ``tp3`` are optional (a scalp may only define TP1), but if present they must
    extend further in profit than the previous target.

    Validation runs in ``__post_init__`` and fails fast on any structurally
    impossible idea (SL on the wrong side, inverted TPs, non-positive prices).
    """

    symbol: str
    direction: Direction
    entry: float
    sl: float
    timestamp: datetime
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    atr: float | None = None
    reasons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entry <= 0 or self.sl <= 0:
            raise ValueError(f"entry and sl must be positive, got entry={self.entry} sl={self.sl}")

        sign = self.direction.sign

        # SL must invalidate the idea: strictly on the losing side of entry.
        if sign * (self.entry - self.sl) <= 0:
            side = "below" if self.direction is Direction.LONG else "above"
            raise ValueError(
                f"{self.direction.value} SL must be {side} entry "
                f"(entry={self.entry}, sl={self.sl})"
            )

        # TPs, when present, must each be further in profit than the previous,
        # and the first must be beyond entry in the profit direction.
        prev = self.entry
        for name, tp in (("tp1", self.tp1), ("tp2", self.tp2), ("tp3", self.tp3)):
            if tp is None:
                continue
            if tp <= 0:
                raise ValueError(f"{name} must be positive, got {tp}")
            if sign * (tp - prev) <= 0:
                raise ValueError(
                    f"{name}={tp} is not further in profit than previous level {prev} "
                    f"for a {self.direction.value} signal"
                )
            prev = tp

    @property
    def risk_distance(self) -> float:
        """Absolute price distance from entry to stop (the 1R unit)."""
        return abs(self.entry - self.sl)

    def rr_to(self, price: float) -> float:
        """Reward-to-risk ratio if the trade exits at ``price``.

        Positive in the profit direction, negative against. Returns ``inf`` only
        in the degenerate zero-risk case, which ``__post_init__`` already
        prevents — kept defensive.
        """
        if self.risk_distance == 0:
            return float("inf")
        return self.direction.sign * (price - self.entry) / self.risk_distance

    @property
    def tp1_rr(self) -> float | None:
        return None if self.tp1 is None else self.rr_to(self.tp1)

    @property
    def first_target(self) -> float | None:
        """First defined take-profit (TP1, else TP2, else TP3, else None)."""
        for tp in (self.tp1, self.tp2, self.tp3):
            if tp is not None:
                return tp
        return None

    @property
    def is_long(self) -> bool:
        return self.direction is Direction.LONG
