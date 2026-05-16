"""Demo script: ParquetStore functionality with synthetic XAU data.

Run:
    uv run python scripts/test_parquet_store.py

No MT5 connection needed — uses generated data.
"""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data.storage.parquet_store import INDEX_NAME, ParquetStore


def _generate_xau_m15(
    start: datetime,
    end: datetime,
    base_price: float = 2000.0,
) -> pd.DataFrame:
    """Generate realistic XAU M15 data (weekends excluded).

    Simulates price movement with random walk, realistic spreads.
    """
    # Generate all 15-min timestamps, then remove weekends
    all_timestamps = pd.date_range(start=start, end=end, freq="15min", tz=UTC)

    # Remove weekend: Friday 22:00 UTC → Sunday 22:00 UTC
    mask = ~(
        ((all_timestamps.weekday == 4) & (all_timestamps.hour >= 22))
        | (all_timestamps.weekday == 5)
        | ((all_timestamps.weekday == 6) & (all_timestamps.hour < 22))
    )
    timestamps = all_timestamps[mask]

    n = len(timestamps)
    rng = np.random.default_rng(42)

    # Random walk for realistic price movement
    returns = rng.normal(0, 0.0003, n)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * (1 + rng.uniform(0.0001, 0.002, n)),
            "low": prices * (1 - rng.uniform(0.0001, 0.002, n)),
            "close": prices * (1 + rng.normal(0, 0.0005, n)),
            "volume": rng.uniform(500, 5000, n),
            "spread": rng.uniform(0.02, 0.05, n),
            "real_volume": rng.uniform(200, 2000, n),
        },
        index=timestamps,
    )
    df.index.name = INDEX_NAME
    return df


def main() -> None:
    logger.info("=== ParquetStore Demo ===")

    # Use temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="parquet_store_demo_"))
    store = ParquetStore(base_path=tmp_dir)

    try:
        # 1. Generate 1 year of M15 XAU data
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, 23, 59, tzinfo=UTC)
        logger.info("Generating 1 year of XAUUSD M15 data...")
        df = _generate_xau_m15(start, end)
        logger.info("Generated {} bars [{} → {}]", len(df), df.index[0], df.index[-1])

        # 2. Save and check structure
        logger.info("Saving to ParquetStore...")
        store.save(df, "XAUUSD", "M15")

        # Show directory tree (first 20 entries)
        logger.info("--- Partition structure ---")
        parquet_files = sorted(tmp_dir.rglob("data.parquet"))
        for f in parquet_files[:20]:
            rel = f.relative_to(tmp_dir)
            size_kb = f.stat().st_size / 1024
            logger.info("  {} ({:.1f} KB)", rel, size_kb)
        if len(parquet_files) > 20:
            logger.info("  ... and {} more partitions", len(parquet_files) - 20)

        # 3. Load all and verify roundtrip
        logger.info("--- Load all ---")
        loaded = store.load("XAUUSD", "M15")
        assert len(loaded) == len(df), f"Mismatch: {len(loaded)} vs {len(df)}"
        assert str(loaded.index.tz) == "UTC"
        logger.info("Roundtrip OK: {} bars, tz={}", len(loaded), loaded.index.tz)

        # 4. Load partial (March only)
        logger.info("--- Load partial (March 2024) ---")
        march_start = datetime(2024, 3, 1, tzinfo=UTC)
        march_end = datetime(2024, 3, 31, 23, 59, tzinfo=UTC)
        march_df = store.load("XAUUSD", "M15", start_date=march_start, end_date=march_end)
        logger.info(
            "March: {} bars [{} → {}]",
            len(march_df),
            march_df.index[0],
            march_df.index[-1],
        )

        # 5. Stats
        logger.info("--- Stats ---")
        stats = store.stats("XAUUSD", "M15")
        for k, v in stats.items():
            logger.info("  {}: {}", k, v)

        # 6. Incremental update simulation
        logger.info("--- Incremental update (mock) ---")

        # Generate 1 extra month (Jan 2025)
        extra_start = datetime(2025, 1, 1, tzinfo=UTC)
        extra_end = datetime(2025, 1, 31, 23, 59, tzinfo=UTC)
        extra_df = _generate_xau_m15(extra_start, extra_end, base_price=2400.0)

        class MockLoader:
            def __enter__(self):
                self.connect()
                return self

            def __exit__(self, *_):
                self.disconnect()

            def connect(self):
                pass

            def disconnect(self):
                pass

            def download_ohlcv(self, symbol, timeframe, start_date, end_date):
                mask = (extra_df.index >= start_date) & (extra_df.index <= end_date)
                return extra_df[mask].copy()

        nb_new = store.incremental_update("XAUUSD", "M15", MockLoader())
        logger.info("Incremental update: {} new bars added", nb_new)

        # 7. Final stats
        logger.info("--- Final stats ---")
        final_stats = store.stats("XAUUSD", "M15")
        logger.info("  Total bars: {}", final_stats["nb_bars"])
        logger.info("  Span: {:.0f} days", final_stats["span_days"])
        logger.info("  Partitions: {}", final_stats["nb_partitions"])
        logger.info("  Size: {:.2f} MB", final_stats["total_size_mb"])
        logger.info("  Coverage: {:.1%}", final_stats["coverage_ratio"])

        # 8. List available
        logger.info("--- Available datasets ---")
        available = store.list_available()
        for sym, tfs in available.items():
            logger.info("  {}: {}", sym, tfs)

        logger.info("=== Demo complete ===")

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir)
        logger.info("Cleaned up {}", tmp_dir)


if __name__ == "__main__":
    main()
