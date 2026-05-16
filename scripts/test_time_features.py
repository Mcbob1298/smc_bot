"""Demo script: time features enrichment on synthetic M15 XAU data.

Run:
    uv run python scripts/test_time_features.py

Demonstrates DST-aware session detection across winter/summer periods.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from data.enrichment.time_features import enrich_time_features


def _generate_week_m15() -> pd.DataFrame:
    """Generate one week of M15 bars (Mon-Fri) starting in winter."""
    start = datetime(2024, 1, 15, 0, 0, tzinfo=UTC)  # Monday
    timestamps = []
    current = start
    while len(timestamps) < 480:  # ~5 days * 96 bars/day
        timestamps.append(current)
        current += timedelta(minutes=15)

    idx = pd.DatetimeIndex(timestamps, tz=UTC)
    rng = np.random.default_rng(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "open": 2100 + rng.normal(0, 0.5, n).cumsum(),
            "high": 2105 + rng.normal(0, 0.5, n).cumsum(),
            "low": 2095 + rng.normal(0, 0.5, n).cumsum(),
            "close": 2100 + rng.normal(0, 0.5, n).cumsum(),
            "volume": rng.uniform(500, 3000, n),
        },
        index=idx,
    )


def main() -> None:
    logger.info("=== Time Features Enrichment Demo ===")

    # Generate data
    df = _generate_week_m15()
    logger.info("Input: {} bars from {} to {}", len(df), df.index[0], df.index[-1])

    # Enrich
    result = enrich_time_features(df)
    added = len(result.columns) - len(df.columns)
    logger.info("Output columns: {} ({} added)", len(result.columns), added)

    # Session distribution
    logger.info("\n--- Session Distribution ---")
    session_counts = result["session_label"].value_counts()
    for label, count in session_counts.items():
        pct = count / len(result) * 100
        logger.info("  {:<12} {:>4} bars ({:.1f}%)", label, count, pct)

    # Killzone bars
    kz_bars = result["is_killzone"].sum()
    kz_pct = kz_bars / len(result) * 100
    logger.info("\nKillzone bars: {} / {} ({:.1f}%)", kz_bars, len(result), kz_pct)

    # XAU market hours
    open_bars = result["is_xau_market_open"].sum()
    open_pct = open_bars / len(result) * 100
    logger.info("XAU market open: {} / {} ({:.1f}%)", open_bars, len(result), open_pct)

    # Friday management
    fri_pre = result["is_xau_friday_pre_close"].sum()
    fri_force = result["is_xau_friday_force_close"].sum()
    logger.info("Friday pre-close bars: {}, force-close bars: {}", fri_pre, fri_force)

    # DST demo: compare winter vs summer
    logger.info("\n--- DST Comparison ---")
    summer_start = datetime(2024, 7, 15, 0, 0, tzinfo=UTC)
    summer_ts = [summer_start + timedelta(minutes=15 * i) for i in range(96)]
    summer_idx = pd.DatetimeIndex(summer_ts, tz=UTC)
    rng = np.random.default_rng(99)
    n_s = len(summer_idx)
    summer_df = pd.DataFrame(
        {
            "open": 2300 + rng.normal(0, 0.5, n_s).cumsum(),
            "high": 2305 + rng.normal(0, 0.5, n_s).cumsum(),
            "low": 2295 + rng.normal(0, 0.5, n_s).cumsum(),
            "close": 2300 + rng.normal(0, 0.5, n_s).cumsum(),
            "volume": rng.uniform(500, 3000, n_s),
        },
        index=summer_idx,
    )
    summer_result = enrich_time_features(summer_df)

    # Show first London KZ bar in winter vs summer
    winter_lkz = result[result["is_london_kz"]].index[0] if result["is_london_kz"].any() else None
    summer_lkz = (
        summer_result[summer_result["is_london_kz"]].index[0]
        if summer_result["is_london_kz"].any()
        else None
    )
    logger.info("First London KZ bar (winter): {} UTC", winter_lkz)
    logger.info("First London KZ bar (summer): {} UTC", summer_lkz)
    logger.info("→ 1h difference due to DST (CET→CEST)")

    # Memory info
    logger.info("\n--- Memory Usage ---")
    mem_bytes = result.memory_usage(deep=True).sum()
    logger.info("Enriched DataFrame memory: {:.1f} KB for {} bars", mem_bytes / 1024, len(result))
    logger.info("Per bar: {:.1f} bytes", mem_bytes / len(result))

    logger.info("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
