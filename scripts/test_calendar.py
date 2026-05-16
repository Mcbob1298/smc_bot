"""Demo script: economic calendar with news window tagging.

Run:
    uv run python scripts/test_calendar.py

Demonstrates calendar backends, news tagging, and disabled mode fallback.
Since MT5 and ForexFactory may not be available, uses mock data for demo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from data.calendar.base import EconCalendarBackend
from data.calendar.econ_calendar import EconCalendar
from data.calendar.mt5_backend import MT5CalendarBackend


class _DemoBackend(EconCalendarBackend):
    """Demo backend with synthetic high-impact events."""

    def __init__(self, events: pd.DataFrame) -> None:
        self._events = events

    def is_available(self) -> bool:
        return True

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        mask = (
            (self._events["timestamp_utc"] >= pd.Timestamp(start))
            & (self._events["timestamp_utc"] <= pd.Timestamp(end))
        )
        return self._events[mask].reset_index(drop=True)


def _generate_week_events() -> pd.DataFrame:
    """Generate realistic high-impact events for one week."""
    events = [
        (datetime(2024, 3, 4, 15, 0, tzinfo=UTC), "US", "ISM Manufacturing PMI"),
        (datetime(2024, 3, 5, 13, 30, tzinfo=UTC), "US", "JOLTS Job Openings"),
        (datetime(2024, 3, 6, 13, 15, tzinfo=UTC), "US", "ADP Non-Farm Employment"),
        (datetime(2024, 3, 7, 13, 30, tzinfo=UTC), "US", "Initial Jobless Claims"),
        (datetime(2024, 3, 7, 14, 45, tzinfo=UTC), "EU", "ECB Rate Decision"),
        (datetime(2024, 3, 8, 13, 30, tzinfo=UTC), "US", "Non-Farm Payrolls"),
        (datetime(2024, 3, 8, 13, 30, tzinfo=UTC), "US", "Unemployment Rate"),
    ]
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime([e[0] for e in events], utc=True),
        "country": [e[1] for e in events],
        "event_name": [e[2] for e in events],
        "impact": ["high"] * len(events),
        "actual": [None] * len(events),
        "forecast": [None] * len(events),
        "previous": [None] * len(events),
    })


def _generate_xau_m15_week() -> pd.DataFrame:
    """Generate 1 week of M15 XAU data (Mon-Fri)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-03-04 00:00", periods=480, freq="15min", tz="UTC")
    n = len(idx)
    base = 2100 + rng.normal(0, 1.0, n).cumsum()
    return pd.DataFrame(
        {
            "open": base,
            "high": base + rng.uniform(1, 5, n),
            "low": base - rng.uniform(1, 5, n),
            "close": base + rng.normal(0, 1, n),
            "volume": rng.uniform(500, 3000, n),
        },
        index=idx,
    )


def main() -> None:
    logger.info("=== Economic Calendar Demo ===")

    # 1. Check MT5 availability
    logger.info("\n--- Backend Availability ---")
    mt5_backend = MT5CalendarBackend()
    mt5_available = mt5_backend.is_available()
    logger.info("MT5 Calendar: {}", "AVAILABLE" if mt5_available else "NOT AVAILABLE")

    # 2. Use demo backend with synthetic events
    events_df = _generate_week_events()
    logger.info("\n--- Synthetic Events (1 week) ---")
    for _, row in events_df.iterrows():
        logger.info(
            "  {} | {} | {} | {}",
            row["timestamp_utc"].strftime("%a %H:%M UTC"),
            row["country"],
            row["event_name"],
            row["impact"],
        )

    # 3. Create calendar with demo backend
    demo_backend = _DemoBackend(events_df)
    cal = EconCalendar(primary_backend=demo_backend)

    # 4. Tag a DataFrame
    df = _generate_xau_m15_week()
    logger.info("\n--- Tagging M15 DataFrame ({} bars) ---", len(df))
    tagged = cal.tag_dataframe(df, pre_minutes=15, post_minutes=30)

    n_tagged = tagged["is_news_window"].sum()
    n_total = len(tagged)
    logger.info(
        "Tagged bars: {} / {} ({:.1f}%)",
        n_tagged, n_total, n_tagged / n_total * 100,
    )

    # Show which bars are tagged around NFP
    nfp_time = pd.Timestamp("2024-03-08 13:30", tz="UTC")
    window = tagged[
        (tagged.index >= nfp_time - timedelta(minutes=30))
        & (tagged.index <= nfp_time + timedelta(minutes=45))
    ]
    logger.info("\n--- Bars around NFP (13:30 UTC, Fri Mar 8) ---")
    logger.info("  {:>20}  is_news?", "timestamp")
    for ts, row in window.iterrows():
        marker = ">>>>" if row["is_news_window"] else "    "
        logger.info("  {} {:>5}  {}", marker, str(ts)[11:16], row["is_news_window"])

    # 5. is_in_news_window single check
    logger.info("\n--- Single Timestamp Checks ---")
    test_times = [
        datetime(2024, 3, 8, 13, 20, tzinfo=UTC),  # 10min before NFP
        datetime(2024, 3, 8, 13, 30, tzinfo=UTC),  # NFP exact
        datetime(2024, 3, 8, 14, 5, tzinfo=UTC),   # 35min after NFP (outside post=30)
        datetime(2024, 3, 8, 10, 0, tzinfo=UTC),   # No event nearby
    ]
    for ts in test_times:
        in_window = cal.is_in_news_window(ts, pre_minutes=15, post_minutes=30)
        logger.info("  {} → {}", ts.strftime("%H:%M UTC"), in_window)

    # 6. Disabled mode
    logger.info("\n--- Disabled Mode ---")
    cal_disabled = EconCalendar(primary_backend=None, fallback_backend=None)
    disabled_result = cal_disabled.tag_dataframe(df)
    assert not disabled_result["is_news_window"].any()
    logger.info("All bars False (as expected in disabled mode)")

    logger.info("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
