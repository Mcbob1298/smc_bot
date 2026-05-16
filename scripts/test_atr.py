"""Demo script: ATR computation on synthetic M15 XAU data.

Run:
    uv run python scripts/test_atr.py

Demonstrates ATR calculation, causality verification, and performance.
"""

import time

import numpy as np
import pandas as pd
from loguru import logger

from data.enrichment.atr import compute_atr, compute_true_range


def _generate_xau_m15(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic XAU M15 data with ~4-8$ ATR."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-15 00:00", periods=n, freq="15min", tz="UTC")
    # XAU typically moves 3-8$ per 15min bar
    base = 2100.0 + rng.normal(0, 2.0, n).cumsum()
    spread = rng.uniform(2.0, 8.0, n)
    opens = base
    highs = base + spread
    lows = base - spread
    closes = base + rng.normal(0, 2.0, n)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(500, 5000, n),
        },
        index=idx,
    )


def main() -> None:
    logger.info("=== ATR Computation Demo ===")

    # 1. Generate data
    df = _generate_xau_m15(n=2000)
    logger.info("Input: {} bars", len(df))

    # 2. Compute TR and ATR
    tr = compute_true_range(df)
    atr = compute_atr(df, period=14)

    # 3. Stats
    logger.info("\n--- True Range Stats ---")
    logger.info("  Mean:   {:.2f}", tr.mean())
    logger.info("  Median: {:.2f}", tr.median())
    logger.info("  Min:    {:.2f}", tr.min())
    logger.info("  Max:    {:.2f}", tr.max())

    logger.info("\n--- ATR(14) Stats (after warm-up) ---")
    atr_valid = atr.dropna()
    logger.info("  Mean:   {:.2f}", atr_valid.mean())
    logger.info("  Median: {:.2f}", atr_valid.median())
    logger.info("  Min:    {:.2f}", atr_valid.min())
    logger.info("  Max:    {:.2f}", atr_valid.max())
    logger.info("  NaN:    {} (first {} bars)", atr.isna().sum(), 13)

    # 4. Compare TR vs ATR smoothness
    logger.info("\n--- Smoothing Effect (last 10 bars) ---")
    logger.info("  {:>8}  {:>8}", "TR", "ATR(14)")
    for i in range(-10, 0):
        logger.info("  {:>8.2f}  {:>8.2f}", tr.iloc[i], atr.iloc[i])

    # 5. Causality check
    logger.info("\n--- Causality Verification ---")
    full_atr = compute_atr(df, period=14)
    for ratio in [0.25, 0.5, 0.75]:
        k = int(len(df) * ratio)
        trunc_atr = compute_atr(df.iloc[:k], period=14)
        match = full_atr.iloc[k - 1] == trunc_atr.iloc[k - 1]
        status = "PASS" if match else "FAIL"
        logger.info(
            "  Truncation {:.0%} (k={}): {} (full={:.4f}, trunc={:.4f})",
            ratio, k, status, full_atr.iloc[k - 1], trunc_atr.iloc[k - 1],
        )

    # 6. Performance benchmark
    logger.info("\n--- Performance Benchmark ---")
    for n_bars in [10_000, 100_000, 1_000_000]:
        bench_df = _generate_xau_m15(n=n_bars, seed=99)
        start = time.perf_counter()
        compute_atr(bench_df, period=14)
        elapsed = time.perf_counter() - start
        logger.info("  {:>10,} bars: {:.3f}s", n_bars, elapsed)

    # 7. Wilder vs SMA comparison
    logger.info("\n--- Wilder vs SMA (last 5 bars) ---")
    atr_sma = compute_atr(df, period=14, method="sma")
    logger.info("  {:>12}  {:>12}", "Wilder", "SMA")
    for i in range(-5, 0):
        logger.info("  {:>12.4f}  {:>12.4f}", atr.iloc[i], atr_sma.iloc[i])

    logger.info("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
