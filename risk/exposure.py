"""Concurrent trade exposure limits.

Caps how many positions can be open at once and how many entries a single
killzone may produce. The per-killzone cap is the anti-revenge rule: after a
loss you do not get to immediately re-enter the same session chasing it back.

State is tracked by broker ticket so the live layer can reconcile against MT5
``positions_get``. ``CLOSED`` trades are forgotten for concurrency but their
killzone is remembered for the session cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExposureTracker:
    """Tracks open positions and per-killzone entry counts.

    A "killzone id" is any caller-chosen string that uniquely identifies a
    session instance, e.g. ``"2026-06-01:london"``. The tracker is agnostic to
    its format.
    """

    open_tickets: dict[int, str] = field(default_factory=dict)  # ticket -> symbol
    killzone_entries: dict[str, int] = field(default_factory=dict)  # kz_id -> count

    @property
    def open_count(self) -> int:
        return len(self.open_tickets)

    def entries_in_killzone(self, killzone_id: str) -> int:
        return self.killzone_entries.get(killzone_id, 0)

    def can_open(
        self,
        max_concurrent: int,
        max_per_killzone: int,
        killzone_id: str | None,
    ) -> tuple[bool, str | None]:
        """Check whether a new position may be opened.

        Returns ``(allowed, reason)``. ``reason`` is ``None`` when allowed, else
        a short human-readable rejection. A ``killzone_id`` of ``None`` (entry
        outside any tracked session) skips the per-killzone cap but still
        respects ``max_concurrent``.
        """
        if self.open_count >= max_concurrent:
            return False, f"max concurrent trades reached ({self.open_count}/{max_concurrent})"
        if killzone_id is not None:
            used = self.entries_in_killzone(killzone_id)
            if used >= max_per_killzone:
                return False, (
                    f"max trades per killzone reached for {killzone_id} "
                    f"({used}/{max_per_killzone})"
                )
        return True, None

    def register_open(self, ticket: int, symbol: str, killzone_id: str | None) -> None:
        """Record a newly opened position and count it against its killzone."""
        if ticket in self.open_tickets:
            raise ValueError(f"ticket {ticket} already open")
        self.open_tickets[ticket] = symbol
        if killzone_id is not None:
            self.killzone_entries[killzone_id] = self.entries_in_killzone(killzone_id) + 1

    def register_close(self, ticket: int) -> None:
        """Record a position closing. Idempotent on unknown tickets."""
        self.open_tickets.pop(ticket, None)

    def reset_killzone_counts(self) -> None:
        """Clear per-killzone tallies (e.g. at the start of a new day)."""
        self.killzone_entries.clear()
