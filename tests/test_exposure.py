"""Tests for the exposure tracker (concurrency + per-killzone caps)."""

import pytest

from risk.exposure import ExposureTracker

KZ = "2026-06-01:london"


def test_allows_first_open():
    ex = ExposureTracker()
    ok, why = ex.can_open(max_concurrent=2, max_per_killzone=1, killzone_id=KZ)
    assert ok and why is None


def test_concurrency_cap():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", KZ)
    ex.register_open(2, "XAUUSD", "2026-06-01:ny")
    ok, why = ex.can_open(max_concurrent=2, max_per_killzone=5, killzone_id="2026-06-01:ny")
    assert not ok
    assert why is not None and "concurrent" in why


def test_per_killzone_cap_anti_revenge():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", KZ)
    ex.register_close(1)  # trade closed (e.g. a loss)
    # Concurrency is free again, but the killzone already spent its entry.
    ok, why = ex.can_open(max_concurrent=2, max_per_killzone=1, killzone_id=KZ)
    assert not ok
    assert why is not None and "killzone" in why


def test_close_frees_concurrency():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", KZ)
    ex.register_close(1)
    assert ex.open_count == 0
    ok, _ = ex.can_open(max_concurrent=1, max_per_killzone=5, killzone_id="other-kz")
    assert ok


def test_none_killzone_skips_session_cap():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", None)
    ok, _ = ex.can_open(max_concurrent=2, max_per_killzone=1, killzone_id=None)
    assert ok


def test_duplicate_ticket_rejected():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", KZ)
    with pytest.raises(ValueError, match="already open"):
        ex.register_open(1, "XAUUSD", KZ)


def test_close_unknown_ticket_idempotent():
    ex = ExposureTracker()
    ex.register_close(999)  # no raise
    assert ex.open_count == 0


def test_reset_killzone_counts():
    ex = ExposureTracker()
    ex.register_open(1, "XAUUSD", KZ)
    ex.reset_killzone_counts()
    assert ex.entries_in_killzone(KZ) == 0
