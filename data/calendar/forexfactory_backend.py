"""ForexFactory scraping backend for economic calendar.

Fallback when MT5 calendar is unavailable. Scrapes the ForexFactory weekly
calendar page and caches results in Parquet for offline/backtest use.

WARNING: Scraping is fragile. ForexFactory can change HTML without notice.
The parser is designed to fail loudly rather than return incorrect data.
"""

from __future__ import annotations

import time as time_mod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from data.calendar.base import (
    CalendarBackendUnavailableError,
    CalendarScrapingError,
    EconCalendarBackend,
)

# ForexFactory displays times in US/Eastern by default
_FF_TIMEZONE = ZoneInfo("US/Eastern")

# High-impact CSS marker in FF HTML
_FF_HIGH_IMPACT_CLASSES = ("high", "red", "icon--ff-impact-red")

# Default cache location relative to project
_DEFAULT_CACHE_DIR = Path("data_store/calendar/forexfactory")


class ForexFactoryBackend(EconCalendarBackend):
    """ForexFactory scraping backend with local Parquet cache.

    Attributes:
        cache_dir: Directory for weekly Parquet cache files.
        cache_ttl_hours: Hours before cache is considered stale.
    """

    BASE_URL = "https://www.forexfactory.com/calendar"

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 12,
    ) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_ttl_hours = cache_ttl_hours

    def is_available(self) -> bool:
        """Check if ForexFactory is reachable."""
        try:
            import httpx

            resp = httpx.head(
                "https://www.forexfactory.com",
                timeout=5.0,
                follow_redirects=True,
            )
            return resp.status_code < 500
        except Exception:
            return False

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        impact: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch events by scraping ForexFactory or reading cache.

        Fetches all weeks that overlap [start, end]. Uses cache if fresh,
        otherwise scrapes and updates cache.

        Args:
            start, end: UTC datetime range.
            impact: Impact filter (default ["high"]).
            countries: Country filter (default ["US", "EU"]).

        Returns:
            DataFrame with standard calendar columns.

        Raises:
            CalendarBackendUnavailableError: If network unreachable.
            CalendarScrapingError: If HTML parsing fails.
        """
        impact_filter = impact or ["high"]
        country_filter = countries or ["US", "EU"]

        # Determine which ISO weeks we need
        weeks = self._weeks_in_range(start, end)
        all_events: list[pd.DataFrame] = []

        for year, week in weeks:
            df_week = self._fetch_week(year, week)
            if df_week is not None and not df_week.empty:
                all_events.append(df_week)

        if not all_events:
            return self._empty_df()

        df = pd.concat(all_events, ignore_index=True)

        # Apply filters
        if impact_filter:
            df = df[df["impact"].isin(impact_filter)]
        if country_filter:
            df = df[df["country"].isin(country_filter)]

        # Filter to requested date range
        df = df[
            (df["timestamp_utc"] >= pd.Timestamp(start, tz=UTC))
            & (df["timestamp_utc"] <= pd.Timestamp(end, tz=UTC))
        ]

        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        logger.info(
            "ForexFactory: {} events in [{}, {}]",
            len(df), start.date(), end.date(),
        )
        return df

    def _fetch_week(self, year: int, week: int) -> pd.DataFrame | None:
        """Fetch a single ISO week — from cache or by scraping."""
        cache_path = self._cache_path(year, week)

        # Check cache freshness
        if cache_path.exists():
            age_hours = (
                time_mod.time() - cache_path.stat().st_mtime
            ) / 3600
            if age_hours < self.cache_ttl_hours:
                logger.debug("Cache hit: {}", cache_path.name)
                return pd.read_parquet(cache_path)
            else:
                logger.debug("Cache expired ({:.1f}h): {}", age_hours, cache_path.name)

        # Scrape
        try:
            df = self._scrape_week(year, week)
        except (CalendarScrapingError, CalendarBackendUnavailableError):
            # If scraping fails but we have stale cache, use it with warning
            if cache_path.exists():
                logger.warning(
                    "Scraping failed, using stale cache: {}", cache_path.name
                )
                return pd.read_parquet(cache_path)
            raise

        # Save to cache
        if df is not None and not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.debug("Cache written: {}", cache_path.name)

        return df

    def _scrape_week(self, year: int, week: int) -> pd.DataFrame:
        """Scrape ForexFactory for a specific ISO week.

        Raises:
            CalendarBackendUnavailableError: Network error.
            CalendarScrapingError: HTML parsing failed.
        """
        import httpx

        url = self._build_url(year, week)
        try:
            resp = httpx.get(
                url,
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SMCBot/1.0)"},
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise CalendarBackendUnavailableError(
                f"ForexFactory timeout for week {year}-W{week:02d}"
            ) from e
        except httpx.HTTPError as e:
            raise CalendarBackendUnavailableError(
                f"ForexFactory HTTP error for week {year}-W{week:02d}: {e}"
            ) from e

        return self._parse_html(resp.text, year, week)

    def _parse_html(self, html: str, year: int, week: int) -> pd.DataFrame:
        """Parse ForexFactory calendar HTML into structured DataFrame.

        Raises:
            CalendarScrapingError: If expected HTML structure not found.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Look for the calendar table
        table = soup.find("table", class_="calendar__table")
        if table is None:
            raise CalendarScrapingError(
                f"Cannot find calendar table in HTML for week {year}-W{week:02d}. "
                "ForexFactory may have changed their page structure."
            )

        events: list[dict[str, object]] = []
        current_date: datetime | None = None

        rows = table.find_all("tr", class_="calendar__row")  # type: ignore[union-attr]
        if not rows:
            raise CalendarScrapingError(
                f"No calendar rows found for week {year}-W{week:02d}. "
                "HTML structure may have changed."
            )

        for row in rows:
            # Date cell (spans multiple rows)
            date_cell = row.find("td", class_="calendar__date")
            if date_cell and date_cell.get_text(strip=True):
                date_text = date_cell.get_text(strip=True)
                parsed_date = self._parse_ff_date(date_text, year)
                if parsed_date:
                    current_date = parsed_date

            if current_date is None:
                continue

            # Time cell
            time_cell = row.find("td", class_="calendar__time")
            if not time_cell:
                continue
            time_text = time_cell.get_text(strip=True)
            if not time_text or time_text in ("", "All Day", "Tentative"):
                continue

            # Impact cell
            impact_cell = row.find("td", class_="calendar__impact")
            impact_level = self._parse_impact(impact_cell)

            # Country/currency cell
            currency_cell = row.find("td", class_="calendar__currency")
            country = currency_cell.get_text(strip=True) if currency_cell else ""

            # Event name
            event_cell = row.find("td", class_="calendar__event")
            event_name = event_cell.get_text(strip=True) if event_cell else ""

            # Actual/Forecast/Previous
            actual_cell = row.find("td", class_="calendar__actual")
            forecast_cell = row.find("td", class_="calendar__forecast")
            previous_cell = row.find("td", class_="calendar__previous")

            actual = self._parse_numeric(actual_cell)
            forecast = self._parse_numeric(forecast_cell)
            previous = self._parse_numeric(previous_cell)

            # Parse time and combine with date
            event_time = self._parse_ff_time(time_text, current_date)
            if event_time is None:
                continue

            # Convert from FF timezone (US/Eastern) to UTC
            event_time_utc = event_time.astimezone(UTC)

            events.append({
                "timestamp_utc": event_time_utc,
                "country": self._currency_to_country(country),
                "event_name": event_name,
                "impact": impact_level,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
            })

        if not events:
            return self._empty_df()

        df = pd.DataFrame(events)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df

    def _parse_impact(self, cell: object) -> str:
        """Determine impact level from cell's icon/class."""
        if cell is None:
            return "low"
        # Check for impact icon span
        cell_html = str(cell)
        for marker in _FF_HIGH_IMPACT_CLASSES:
            if marker in cell_html:
                return "high"
        if "medium" in cell_html or "orange" in cell_html or "ff-impact-ora" in cell_html:
            return "medium"
        return "low"

    def _parse_ff_date(self, text: str, year: int) -> datetime | None:
        """Parse FF date like 'Mon May 5' into datetime."""

        # FF format: "MonMay 5" or "Mon May 5" or "May 5"
        text = text.strip()
        # Try common formats
        for fmt in ("%a%b %d", "%a %b %d", "%b %d"):
            try:
                parsed = datetime.strptime(text, fmt).replace(year=year)
                return parsed
            except ValueError:
                continue
        return None

    def _parse_ff_time(
        self, time_text: str, date: datetime
    ) -> datetime | None:
        """Parse FF time like '8:30am' and combine with date in FF timezone."""
        time_text = time_text.strip().lower()
        try:
            # Handle formats: "8:30am", "2:00pm", "12:30pm"
            t = datetime.strptime(time_text, "%I:%M%p")
            combined = date.replace(
                hour=t.hour, minute=t.minute, second=0, microsecond=0
            )
            return combined.replace(tzinfo=_FF_TIMEZONE)
        except ValueError:
            return None

    def _parse_numeric(self, cell: object) -> float | None:
        """Extract numeric value from cell, handling '%', 'K', 'M' suffixes."""
        if cell is None:
            return None
        text = cell.get_text(strip=True) if hasattr(cell, "get_text") else str(cell)  # type: ignore[union-attr]
        if not text or text == "":
            return None
        # Remove common suffixes
        text = text.replace("%", "").replace("K", "e3").replace("M", "e6")
        text = text.replace("B", "e9").replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    def _currency_to_country(self, currency: str) -> str:
        """Map FF currency code to country code."""
        mapping = {
            "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
            "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
        }
        return mapping.get(currency.upper(), currency.upper())

    def _build_url(self, year: int, week: int) -> str:
        """Build ForexFactory URL for a given ISO week."""
        # FF uses format: calendar?week=may2.2025
        # We need to find the Monday of the ISO week
        from datetime import date

        # Monday of ISO week
        monday = date.fromisocalendar(year, week, 1)
        month_name = monday.strftime("%b").lower()
        # Week number within the month (1-based)
        day = monday.day
        week_in_month = (day - 1) // 7 + 1
        return f"{self.BASE_URL}?week={month_name}{week_in_month}.{year}"

    def _cache_path(self, year: int, week: int) -> Path:
        """Get cache file path for a given ISO week."""
        return self.cache_dir / f"ff_week_{year}-W{week:02d}.parquet"

    def _weeks_in_range(
        self, start: datetime, end: datetime
    ) -> list[tuple[int, int]]:
        """Get all ISO (year, week) tuples that overlap [start, end]."""

        start_date = start.date() if isinstance(start, datetime) else start
        end_date = end.date() if isinstance(end, datetime) else end

        weeks: list[tuple[int, int]] = []
        current = start_date
        while current <= end_date:
            iso = current.isocalendar()
            key = (iso[0], iso[1])
            if key not in weeks:
                weeks.append(key)
            current += timedelta(days=7)

        # Ensure end week is included
        iso_end = end_date.isocalendar()
        key_end = (iso_end[0], iso_end[1])
        if key_end not in weeks:
            weeks.append(key_end)

        return weeks

    def _empty_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(
            columns=[
                "timestamp_utc", "country", "event_name",
                "impact", "actual", "forecast", "previous",
            ]
        )
