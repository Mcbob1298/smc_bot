"""MT5 economic calendar backend.

Uses MetaTrader 5's built-in economic calendar API for reliable,
rate-limit-free event data. Requires MT5 terminal running and connected.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from data.calendar.base import (
    CalendarBackendUnavailableError,
    CalendarMT5Error,
    EconCalendarBackend,
)

# High-impact event patterns for matching with tolerance to name variations.
# Each pattern is compiled as case-insensitive regex.
_HIGH_IMPACT_PATTERNS: list[re.Pattern[str]] = [
    # US
    re.compile(r"non[\s-]?farm\s*(payrolls|employment)", re.IGNORECASE),
    re.compile(r"\bCPI\b", re.IGNORECASE),
    re.compile(r"consumer\s+price\s+index", re.IGNORECASE),
    re.compile(r"core\s+CPI", re.IGNORECASE),
    re.compile(r"FOMC\s+(statement|press|meeting|minutes)", re.IGNORECASE),
    re.compile(r"federal\s+funds\s+rate", re.IGNORECASE),
    re.compile(r"\bGDP\b", re.IGNORECASE),
    re.compile(r"\bPPI\b", re.IGNORECASE),
    re.compile(r"core\s+PPI", re.IGNORECASE),
    re.compile(r"unemployment\s+rate", re.IGNORECASE),
    re.compile(r"ISM\s+(manufacturing|services|non[\s-]?manufacturing)\s+PMI", re.IGNORECASE),
    re.compile(r"retail\s+sales", re.IGNORECASE),
    re.compile(r"initial\s+jobless\s+claims", re.IGNORECASE),
    # EU / ECB
    re.compile(r"ECB\s+(press|rate|main\s+refinancing|monetary\s+policy)", re.IGNORECASE),
    re.compile(r"eurozone\s+(CPI|GDP|unemployment)", re.IGNORECASE),
    re.compile(r"main\s+refinancing\s+rate", re.IGNORECASE),
    # BOE
    re.compile(r"BOE\s+(rate|monetary\s+policy)", re.IGNORECASE),
]

# Default countries for XAU (USD-dominated)
_DEFAULT_COUNTRIES = ["US", "EU"]


def _matches_high_impact(event_name: str) -> bool:
    """Check if event name matches any high-impact pattern."""
    return any(p.search(event_name) for p in _HIGH_IMPACT_PATTERNS)


class MT5CalendarBackend(EconCalendarBackend):
    """MetaTrader 5 economic calendar backend.

    Uses mt5.calendar_event_get() and mt5.calendar_value_history_get()
    for reliable, integrated calendar data.
    """

    def is_available(self) -> bool:
        """Check if MT5 is initialized and calendar API responds."""
        try:
            import MetaTrader5 as mt5  # noqa: N813
        except ImportError:
            return False

        if not mt5.terminal_info():
            return False

        # Try fetching US events as availability check
        try:
            result = mt5.calendar_event_get(country="US")
            return result is not None
        except Exception:
            return False

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch events from MT5 calendar.

        Args:
            start, end: UTC datetime range.
            impact: Impact filter (default ["high"]).
            countries: Country filter (default ["US", "EU"]).

        Returns:
            DataFrame with standard calendar columns.

        Raises:
            CalendarBackendUnavailableError: If MT5 not running.
            CalendarMT5Error: If MT5 API fails.
        """
        try:
            import MetaTrader5 as mt5  # noqa: N813
        except ImportError as e:
            raise CalendarBackendUnavailableError(
                "MetaTrader5 package not installed"
            ) from e

        if not mt5.terminal_info():
            # Try to initialize
            if not mt5.initialize():
                raise CalendarBackendUnavailableError(
                    "MT5 terminal not running or cannot initialize"
                )

        impact_filter = impact or ["high"]
        country_list = countries or _DEFAULT_COUNTRIES

        all_events: list[dict[str, object]] = []

        for country in country_list:
            try:
                events_meta = mt5.calendar_event_get(country=country)
            except Exception as e:
                raise CalendarMT5Error(
                    f"Failed to get calendar events for {country}: {e}"
                ) from e

            if events_meta is None:
                logger.warning("MT5 calendar returned None for country={}", country)
                continue

            for event in events_meta:
                event_name = event.name if hasattr(event, "name") else str(event)

                # Filter by impact: use pattern matching for "high"
                if "high" in impact_filter and not _matches_high_impact(event_name):
                    # If we only want high impact, skip non-matching
                    if impact_filter == ["high"]:
                        continue

                # Fetch historical values for this event
                try:
                    values = mt5.calendar_value_history_get(
                        event.id, start, end
                    )
                except Exception:
                    continue

                if values is None:
                    continue

                for val in values:
                    event_time = datetime.fromtimestamp(val.time, tz=UTC)
                    all_events.append({
                        "timestamp_utc": event_time,
                        "country": country,
                        "event_name": event_name,
                        "impact": "high",
                        "actual": val.actual_value if val.actual_value != 0 else None,
                        "forecast": val.forecast_value if val.forecast_value != 0 else None,
                        "previous": val.previous_value if val.previous_value != 0 else None,
                    })

        df = self._build_dataframe(all_events)
        logger.info(
            "MT5 calendar: fetched {} events for {} countries in [{}, {}]",
            len(df), len(country_list), start.date(), end.date(),
        )
        return df

    def _build_dataframe(self, events: list[dict[str, object]]) -> pd.DataFrame:
        """Build standardized DataFrame from event dicts."""
        if not events:
            return pd.DataFrame(
                columns=[
                    "timestamp_utc", "country", "event_name",
                    "impact", "actual", "forecast", "previous",
                ]
            )

        df = pd.DataFrame(events)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.drop_duplicates(subset=["timestamp_utc", "event_name"])
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        return df
