"""Economic calendar orchestrator with fallback strategy.

High-level facade that tries the primary backend (MT5), falls back to
secondary (ForexFactory), and provides the `tag_dataframe` utility for
marking OHLCV bars within news windows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from data.calendar.base import (
    EconCalendarBackend,
    EconCalendarError,
)


class EconCalendar:
    """High-level calendar facade with backend fallback.

    Orchestrates one or two backends (primary + fallback). If both are None,
    operates in "disabled mode" where tag_dataframe returns all-False.
    """

    def __init__(
        self,
        primary_backend: EconCalendarBackend | None = None,
        fallback_backend: EconCalendarBackend | None = None,
    ) -> None:
        self._primary = primary_backend
        self._fallback = fallback_backend

        if primary_backend is None and fallback_backend is None:
            logger.warning(
                "EconCalendar: no backends configured. "
                "News filter DISABLED — may take trades during news events."
            )
        else:
            primary_name = type(primary_backend).__name__ if primary_backend else "None"
            fallback_name = type(fallback_backend).__name__ if fallback_backend else "None"
            logger.info(
                "EconCalendar: primary={}, fallback={}",
                primary_name, fallback_name,
            )

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch events using primary backend, fallback if primary fails.

        Args:
            start, end: UTC datetime range.
            impact: Impact filter (default ["high"]).
            countries: Country filter (default ["US", "EU"]).

        Returns:
            DataFrame with standard calendar columns.

        Raises:
            EconCalendarError: If both backends fail.
        """
        if self._primary is None and self._fallback is None:
            logger.warning("EconCalendar: both backends None, returning empty events")
            return self._empty_df()

        # Try primary
        if self._primary is not None:
            try:
                if self._primary.is_available():
                    return self._primary.fetch_events(start, end, impact, countries)
                else:
                    logger.warning(
                        "Primary backend ({}) not available, trying fallback",
                        type(self._primary).__name__,
                    )
            except Exception as e:
                logger.warning(
                    "Primary backend ({}) failed: {}. Trying fallback.",
                    type(self._primary).__name__, e,
                )

        # Try fallback
        if self._fallback is not None:
            try:
                if self._fallback.is_available():
                    return self._fallback.fetch_events(start, end, impact, countries)
                else:
                    logger.error("Fallback backend ({}) not available either",
                                 type(self._fallback).__name__)
            except Exception as e:
                logger.error(
                    "Fallback backend ({}) also failed: {}",
                    type(self._fallback).__name__, e,
                )

        raise EconCalendarError(
            "Both calendar backends failed. News filter cannot operate."
        )

    def is_in_news_window(
        self,
        timestamp: datetime,
        pre_minutes: int = 15,
        post_minutes: int = 30,
        impact: list[str] | None = None,
        events_df: pd.DataFrame | None = None,
    ) -> bool:
        """Check if a single timestamp falls within a news window.

        A news window is [event_time - pre_minutes, event_time + post_minutes].

        Args:
            timestamp: UTC datetime to check.
            pre_minutes: Minutes before event to start window.
            post_minutes: Minutes after event to end window.
            impact: Impact filter for fetching events.
            events_df: Pre-fetched events (avoids re-fetching).

        Returns:
            True if timestamp is within any event's news window.
        """
        if events_df is None:
            # Fetch events around the timestamp (±2 days for safety)
            fetch_start = timestamp - timedelta(days=2)
            fetch_end = timestamp + timedelta(days=2)
            try:
                events_df = self.fetch_events(fetch_start, fetch_end, impact=impact)
            except EconCalendarError:
                return False

        if events_df.empty:
            return False

        raw_ts = pd.Timestamp(timestamp)
        ts = raw_ts if raw_ts.tzinfo is not None else raw_ts.tz_localize(UTC)
        pre_delta = pd.Timedelta(minutes=pre_minutes)
        post_delta = pd.Timedelta(minutes=post_minutes)

        for event_time in events_df["timestamp_utc"]:
            event_ts = pd.Timestamp(event_time)
            if (event_ts - pre_delta) <= ts <= (event_ts + post_delta):
                return True

        return False

    def tag_dataframe(
        self,
        df: pd.DataFrame,
        pre_minutes: int = 15,
        post_minutes: int = 30,
        impact: list[str] | None = None,
        column_name: str = "is_news_window",
    ) -> pd.DataFrame:
        """Add boolean news window column to OHLCV DataFrame.

        For each bar, marks True if it falls within [event - pre, event + post]
        for any high-impact event in the range.

        Args:
            df: DataFrame with DatetimeIndex (UTC).
            pre_minutes: Minutes before event.
            post_minutes: Minutes after event.
            impact: Impact filter.
            column_name: Name for the new column.

        Returns:
            Copy of df with boolean column added.

        Raises:
            ValueError: If column already exists.

        Causality note:
            This is strictly causal because event *timings* are published
            days/weeks in advance. The tag at time t depends only on
            known-in-advance scheduled event times, not on future price data.
            The post_minutes window is also causal: we know at time t whether
            an event occurred at (t - post) because that's in the past.
        """
        if column_name in df.columns:
            raise ValueError(
                f"DataFrame already contains column '{column_name}'. "
                "Remove it first or use a different column_name."
            )

        if df.empty:
            result = df.copy()
            result[column_name] = pd.Series(dtype="bool")
            return result

        result = df.copy()

        # Disabled mode
        if self._primary is None and self._fallback is None:
            logger.warning(
                "News filter disabled: tagging all bars as is_news_window=False"
            )
            result[column_name] = False
            return result

        # Fetch events for the full range of the DataFrame
        pre_td = timedelta(minutes=pre_minutes)
        post_td = timedelta(minutes=post_minutes)
        fetch_start = df.index.min().to_pydatetime() - pre_td  # type: ignore[union-attr]
        fetch_end = df.index.max().to_pydatetime() + post_td  # type: ignore[union-attr]

        try:
            events_df = self.fetch_events(
                fetch_start, fetch_end, impact=impact  # type: ignore[arg-type]
            )
        except EconCalendarError:
            logger.warning(
                "Cannot fetch events for news tagging. "
                "Marking all bars as is_news_window=False."
            )
            result[column_name] = False
            return result

        if events_df.empty:
            result[column_name] = False
            return result

        # Vectorized tagging: for each event, mark bars in window
        mask = np.zeros(len(df), dtype=bool)
        pre_delta = pd.Timedelta(minutes=pre_minutes)
        post_delta = pd.Timedelta(minutes=post_minutes)

        for event_time in events_df["timestamp_utc"]:
            event_ts = pd.Timestamp(event_time)
            window_start = event_ts - pre_delta
            window_end = event_ts + post_delta
            mask |= (df.index >= window_start) & (df.index <= window_end)

        result[column_name] = mask
        n_tagged = int(mask.sum())
        logger.debug(
            "News tagging: {} / {} bars tagged ({} events)",
            n_tagged, len(df), len(events_df),
        )
        return result

    def _empty_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(
            columns=[
                "timestamp_utc", "country", "event_name",
                "impact", "actual", "forecast", "previous",
            ]
        )
