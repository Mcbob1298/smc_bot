"""Shared helpers and conventions for SMC detectors.

Event model & causality
------------------------
Every detector emits a pandas DataFrame of *events* (swings, FVGs, order
blocks, ...). Two timestamps matter and are kept distinct on purpose:

- the DataFrame **index** is the bar where the pattern physically *is*
  (e.g. the candle that forms the swing high);
- the ``confirmed_at`` column is the bar at which the pattern first becomes
  *causally knowable* — i.e. the earliest bar a live system could have reacted
  to it without seeing the future.

A 3-bar swing high, for instance, sits at its centre bar but is only confirmed
one bar later (the right shoulder must print first). Strict causality therefore
means: **a consumer at time T may only use events whose ``confirmed_at <= T``.**
``select_known`` enforces exactly that, and ``test_no_lookahead`` checks that
``confirmed_at >= index`` for every event of every detector.
"""

from __future__ import annotations

import pandas as pd

OHLC_COLUMNS = ("open", "high", "low", "close")


def validate_ohlc(df: pd.DataFrame, *, need: tuple[str, ...] = OHLC_COLUMNS) -> None:
    """Raise ValueError if ``df`` is missing any required OHLC column."""
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing} (need {list(need)})")


def select_known(events: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Return only the events causally known at ``now`` (``confirmed_at <= now``).

    The single chokepoint every consumer should use to read detector output, so
    that no lookahead can sneak in. Events with NaT ``confirmed_at`` (never
    confirmed within the data) are excluded.
    """
    if events.empty:
        return events
    if "confirmed_at" not in events.columns:
        raise ValueError("events frame has no 'confirmed_at' column")
    return events[events["confirmed_at"].notna() & (events["confirmed_at"] <= now)]


def empty_events(columns: list[str]) -> pd.DataFrame:
    """An empty, correctly-typed events frame (so callers never special-case)."""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    return df
