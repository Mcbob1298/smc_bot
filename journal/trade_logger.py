"""Structured JSON trade logging.

Appends one JSON object per event to a JSON Lines file (``.jsonl``), capturing
*why* each decision was made — the Signal's ``reasons`` map and the
RiskManager's decision reasons — so a trade can be audited months later and so
the journal analyzer / chart annotator can reconstruct it.

JSONL (one event per line) is chosen over a single JSON array so the live loop
can append cheaply without rewriting the file, and so a crash never corrupts
prior records.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from live.paper_executor import Fill, PaperPosition
    from risk.risk_manager import RiskDecision
    from strategy.signal import Signal


class TradeLogger:
    """Append-only JSONL logger for trade lifecycle events.

    Events: ``rejection``, ``open``, ``fill``, ``close``. Each line is a
    self-contained record with an ISO timestamp and an ``event`` tag.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _signal_dict(signal: Signal) -> dict[str, Any]:
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "entry": signal.entry,
            "sl": signal.sl,
            "tp1": signal.tp1,
            "tp2": signal.tp2,
            "tp3": signal.tp3,
            "atr": signal.atr,
            "risk_distance": signal.risk_distance,
            "reasons": signal.reasons,
        }

    def log_rejection(self, signal: Signal, decision: RiskDecision, now: datetime) -> None:
        self._write(
            {
                "event": "rejection",
                "ts": now.isoformat(),
                "signal": self._signal_dict(signal),
                "reasons": decision.reasons,
            }
        )

    def log_open(self, position: PaperPosition, decision: RiskDecision, now: datetime) -> None:
        self._write(
            {
                "event": "open",
                "ts": now.isoformat(),
                "ticket": position.ticket,
                "lot": position.lot,
                "entry_price": position.entry_price,
                "killzone_id": position.killzone_id,
                "signal": self._signal_dict(position.signal),
                "risk_reasons": decision.reasons,
            }
        )

    def log_fill(self, position: PaperPosition, fill: Fill, now: datetime) -> None:
        self._write(
            {
                "event": "fill",
                "ts": now.isoformat(),
                "ticket": position.ticket,
                "kind": fill.kind,
                "price": fill.price,
                "lot": fill.lot,
                "pnl": fill.pnl,
                "remaining_lot": position.remaining_lot,
            }
        )

    def log_close(self, position: PaperPosition, now: datetime) -> None:
        self._write(
            {
                "event": "close",
                "ts": now.isoformat(),
                "ticket": position.ticket,
                "realized_pnl": position.realized_pnl,
                "fills": [
                    {"kind": f.kind, "price": f.price, "lot": f.lot, "pnl": f.pnl}
                    for f in position.fills
                ],
            }
        )
