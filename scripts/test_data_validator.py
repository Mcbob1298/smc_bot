"""Demo script: OHLCV data validation with clean and corrupted datasets.

Run:
    uv run python scripts/test_data_validator.py

Demonstrates validation on clean data and various corruptions.
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data.storage.data_validator import OHLCVValidator


def _generate_clean_xau_m15(periods: int = 2000) -> pd.DataFrame:
    """Generate clean XAU M15 data (weekdays only, valid OHLC)."""
    start = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)  # Monday

    timestamps = []
    current = start
    while len(timestamps) < periods:
        wd = current.weekday()
        h = current.hour
        if wd == 5 or (wd == 6 and h < 22) or (wd == 4 and h >= 22):
            current += timedelta(minutes=15)
            continue
        timestamps.append(current)
        current += timedelta(minutes=15)

    idx = pd.DatetimeIndex(timestamps, tz=UTC)
    rng = np.random.default_rng(42)
    base = 2100.0
    opens = base + np.cumsum(rng.normal(0, 0.3, periods))
    highs = opens + rng.uniform(0.5, 3.0, periods)
    lows = opens - rng.uniform(0.5, 3.0, periods)
    closes = opens + rng.normal(0, 0.5, periods)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(500, 5000, periods),
            "spread": rng.uniform(0.02, 0.04, periods),
            "real_volume": rng.uniform(200, 2000, periods),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


def main() -> None:
    logger.info("=== OHLCV Data Validator Demo ===")
    validator = OHLCVValidator()

    # 1. Clean dataset
    logger.info("--- 1. Validating clean dataset (1 month M15 XAU) ---")
    clean_df = _generate_clean_xau_m15(periods=2000)
    report_clean = validator.validate(clean_df, "XAUUSD", "M15")
    logger.info("\n{}", report_clean.summary())

    # 2. Corrupted dataset
    logger.info("\n--- 2. Validating corrupted dataset ---")
    bad_df = clean_df.copy()

    # Inject corruptions
    bad_df.iloc[50, bad_df.columns.get_loc("close")] = np.nan  # NaN in OHLC
    bad_df.iloc[100, bad_df.columns.get_loc("close")] = bad_df.iloc[99]["close"] * 1.08  # 8% jump
    bad_df.iloc[100, bad_df.columns.get_loc("high")] = bad_df.iloc[100]["close"] + 1

    # Inject a Saturday bar
    idx = bad_df.index.tolist()
    idx[200] = datetime(2024, 3, 9, 12, 0, tzinfo=UTC)  # Saturday
    bad_df.index = pd.DatetimeIndex(idx)

    # Set extreme spread on one bar
    median_s = bad_df["spread"].median()
    bad_df.iloc[150, bad_df.columns.get_loc("spread")] = median_s * 12

    report_bad = validator.validate(bad_df, "XAUUSD", "M15")
    logger.info("\n{}", report_bad.summary())

    # 3. Save report as JSON
    tmp_dir = Path(tempfile.mkdtemp(prefix="validator_demo_"))
    report_path = tmp_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_bad.to_dict(), f, indent=2)
    logger.info("\nReport saved to: {}", report_path)

    # Show JSON preview
    report_dict = report_bad.to_dict()
    logger.info("JSON keys: {}", list(report_dict.keys()))
    logger.info("Issues count: {}", len(report_dict["issues"]))

    logger.info("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
