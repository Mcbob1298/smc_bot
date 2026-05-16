"""Abstract base for economic calendar backends + typed exceptions.

Defines the EconCalendarBackend interface that all calendar data sources
must implement, plus exception hierarchy for calendar errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EconCalendarError(Exception):
    """Base exception for all calendar errors."""


class CalendarBackendUnavailableError(EconCalendarError):
    """A specific backend is not available (MT5 not running, network down)."""


class CalendarScrapingError(EconCalendarError):
    """Scraping failed — HTML changed, parsing error, etc."""


class CalendarMT5Error(EconCalendarError):
    """MT5 calendar-specific error."""


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class EconCalendarBackend(ABC):
    """Abstract base for economic calendar data sources."""

    @abstractmethod
    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch economic events in the given range.

        Args:
            start, end: UTC datetimes (tz-aware).
            impact: Filter by impact level. Default ["high"].
            countries: Filter by country ISO codes. Default ["US", "EU"].

        Returns:
            DataFrame with columns:
                - timestamp_utc: datetime64[ns, UTC]
                - country: str (ISO code)
                - event_name: str
                - impact: str ("high", "medium", "low")
                - actual: float | None
                - forecast: float | None
                - previous: float | None
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is functional right now."""
