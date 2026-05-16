"""Time-based features: killzones, sessions, DST-aware Paris timezone.

Enriches OHLCV DataFrames with temporal features essential to the SMC strategy:
killzone detection, session labeling, XAU market hours, weekday/date info.

Causality: All features are derived purely from each row's timestamp.
No look-ahead or look-back. Strictly causal by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config.strategy import StrategyConfig

# XAU market hours (UTC):
# Close: Friday 21:00 UTC (some brokers 22:00, we use 21:00 conservative)
# Open: Sunday 22:00 UTC
_XAU_CLOSE_WEEKDAY = 4  # Friday
_XAU_CLOSE_HOUR_UTC = 21
_XAU_OPEN_WEEKDAY = 6  # Sunday
_XAU_OPEN_HOUR_UTC = 22

# Columns added by this module
_ADDED_COLUMNS = [
    "paris_hour",
    "paris_minute",
    "paris_weekday",
    "paris_date",
    "is_asia_session",
    "is_london_kz",
    "is_london_session",
    "is_ny_kz",
    "is_ny_session",
    "is_killzone",
    "is_overlap_london_ny",
    "session_label",
    "is_xau_friday_pre_close",
    "is_xau_friday_force_close",
    "is_xau_market_open",
    "iso_week",
    "month",
    "quarter",
    "year",
]


def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string to (hour, minute) tuple."""
    parts = time_str.split(":")
    return int(parts[0]), int(parts[1])


def _time_in_range(
    hour: np.ndarray | pd.Index,  # type: ignore[type-arg]
    minute: np.ndarray | pd.Index,  # type: ignore[type-arg]
    start_h: int,
    start_m: int,
    end_h: int,
    end_m: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Vectorized check: is (hour, minute) in [start, end) range.

    Returns boolean ndarray. Works for ranges that don't cross midnight.
    """
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    current_total = np.asarray(hour) * 60 + np.asarray(minute)
    result: np.ndarray = (current_total >= start_total) & (current_total < end_total)  # type: ignore[type-arg]
    return result


def enrich_time_features(
    df: pd.DataFrame,
    tz: str = "Europe/Paris",
    strategy_config: StrategyConfig | None = None,
) -> pd.DataFrame:
    """Add time-based features to OHLCV DataFrame.

    Args:
        df: DataFrame with DatetimeIndex tz-aware UTC, named 'timestamp_utc'.
        tz: Reference timezone for session detection (default: Europe/Paris).
        strategy_config: Optional config to override default killzone times.

    Returns:
        New DataFrame with original columns + time feature columns added.

    Raises:
        TypeError: If index is not DatetimeIndex.
        ValueError: If index is not tz-aware UTC, or if a column conflict exists.

    Causality: All features are derived purely from each row's timestamp.
    No look-ahead or look-back. Strictly causal by construction.
    """
    if df.empty:
        return df.copy()

    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Index must be DatetimeIndex, got {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise ValueError("Index must be tz-aware UTC. Got naive (tz=None).")
    if str(df.index.tz) != "UTC":
        raise ValueError(
            f"Index must be UTC, got tz={df.index.tz}. Convert with .tz_convert('UTC') first."
        )

    # Check for column conflicts
    conflicts = set(_ADDED_COLUMNS) & set(df.columns)
    if conflicts:
        raise ValueError(
            f"DataFrame already contains columns that would be overwritten: {sorted(conflicts)}"
        )

    # Get killzone times from config or defaults
    if strategy_config is not None:
        kz = strategy_config.killzones
        london_start_h, london_start_m = _parse_time(kz.london_start)
        london_end_h, london_end_m = _parse_time(kz.london_end)
        ny_start_h, ny_start_m = _parse_time(kz.ny_start)
        ny_end_h, ny_end_m = _parse_time(kz.ny_end)
    else:
        london_start_h, london_start_m = 8, 0
        london_end_h, london_end_m = 11, 0
        ny_start_h, ny_start_m = 13, 30
        ny_end_h, ny_end_m = 16, 0

    # Friday pre-close / force-close times (Paris)
    if strategy_config is not None:
        fri_no_trade_h, fri_no_trade_m = _parse_time(
            strategy_config.risk.xau_friday_no_new_trade_after
        )
        fri_close_h, fri_close_m = _parse_time(
            strategy_config.risk.xau_friday_close_time
        )
    else:
        fri_no_trade_h, fri_no_trade_m = 18, 0
        fri_close_h, fri_close_m = 22, 30

    # Work on a copy
    result = df.copy()

    # Convert index to Paris timezone (vectorized, DST-aware)
    paris_index = df.index.tz_convert(ZoneInfo(tz))

    # Basic time components in Paris tz
    result["paris_hour"] = paris_index.hour.astype("int8")
    result["paris_minute"] = paris_index.minute.astype("int8")
    result["paris_weekday"] = paris_index.weekday.astype("int8")
    result["paris_date"] = paris_index.date

    # Extract numpy arrays for fast vectorized comparisons
    p_hour = np.asarray(paris_index.hour)
    p_minute = np.asarray(paris_index.minute)
    p_weekday = np.asarray(paris_index.weekday)

    # Sessions (all times in Paris)
    # Asia: 00:00 - 08:00 Paris
    is_asia = _time_in_range(p_hour, p_minute, 0, 0, 8, 0)

    # London KZ: default 08:00 - 11:00 Paris
    is_london_kz = _time_in_range(
        p_hour, p_minute, london_start_h, london_start_m, london_end_h, london_end_m
    )

    # London session: 08:00 - 16:30 Paris
    is_london_session = _time_in_range(p_hour, p_minute, 8, 0, 16, 30)

    # NY KZ: default 13:30 - 16:00 Paris
    is_ny_kz = _time_in_range(
        p_hour, p_minute, ny_start_h, ny_start_m, ny_end_h, ny_end_m
    )

    # NY session: 13:30 - 22:00 Paris
    is_ny_session = _time_in_range(p_hour, p_minute, 13, 30, 22, 0)

    # Overlap London-NY: 13:30 - 16:30 Paris
    is_overlap = _time_in_range(p_hour, p_minute, 13, 30, 16, 30)

    # Combined killzone flag
    is_killzone = is_london_kz | is_ny_kz

    result["is_asia_session"] = is_asia
    result["is_london_kz"] = is_london_kz
    result["is_london_session"] = is_london_session
    result["is_ny_kz"] = is_ny_kz
    result["is_ny_session"] = is_ny_session
    result["is_killzone"] = is_killzone
    result["is_overlap_london_ny"] = is_overlap

    # Session label with priority: overlap > kz > session > asia > off
    conditions = [
        is_overlap,
        is_london_kz,
        is_ny_kz,
        is_london_session,
        is_ny_session,
        is_asia,
    ]
    choices = ["overlap", "london_kz", "ny_kz", "london", "ny", "asia"]
    session_labels = np.select(conditions, choices, default="off")
    result["session_label"] = pd.Categorical(session_labels, categories=[
        "overlap", "london_kz", "ny_kz", "london", "ny", "asia", "off"
    ])

    # XAU Friday management (Paris times)
    is_friday_paris = p_weekday == 4
    fri_pre_close = is_friday_paris & (
        (p_hour * 60 + p_minute) >= (fri_no_trade_h * 60 + fri_no_trade_m)
    )
    fri_force_close = is_friday_paris & (
        (p_hour * 60 + p_minute) >= (fri_close_h * 60 + fri_close_m)
    )
    result["is_xau_friday_pre_close"] = fri_pre_close
    result["is_xau_friday_force_close"] = fri_force_close

    # XAU market open (UTC-based)
    # Closed: Friday >= 21:00 UTC, Saturday all day, Sunday < 22:00 UTC
    utc_hour = np.asarray(df.index.hour)
    utc_weekday = np.asarray(df.index.weekday)

    market_closed = (
        ((utc_weekday == _XAU_CLOSE_WEEKDAY) & (utc_hour >= _XAU_CLOSE_HOUR_UTC))
        | (utc_weekday == 5)  # Saturday
        | ((utc_weekday == _XAU_OPEN_WEEKDAY) & (utc_hour < _XAU_OPEN_HOUR_UTC))
    )
    result["is_xau_market_open"] = ~market_closed

    # Meta time features
    result["iso_week"] = np.asarray(paris_index.isocalendar().week).astype("int8")
    result["month"] = paris_index.month.astype("int8")
    result["quarter"] = paris_index.quarter.astype("int8")
    result["year"] = paris_index.year.astype("int16")

    return result
