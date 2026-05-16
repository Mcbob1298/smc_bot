"""Demo script: download 30 days of BTCUSDT M15 via Binance and display stats.

Run:
    uv run python scripts/test_ccxt_loader.py

No API key needed — uses public OHLCV endpoints.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from data.ingestion.ccxt_loader import CCXTLoader


def main() -> None:
    logger.info("=== CCXT Loader Integration Test ===")
    logger.info("Symbol: BTCUSDT | Timeframe: M15 | Range: last 30 days")

    loader = CCXTLoader(
        exchange_id="binance",
        symbol_map={"BTCUSDT": "BTC/USDT"},
    )

    try:
        loader.connect()

        end = datetime.now(tz=UTC)
        start = end - timedelta(days=30)

        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        # Stats
        logger.info("--- Results ---")
        logger.info("Bars downloaded: {}", len(df))
        logger.info("Date range: {} → {}", df.index[0], df.index[-1])
        logger.info(
            "Price range: {:.2f} → {:.2f}",
            df["close"].min(),
            df["close"].max(),
        )
        logger.info("Total volume: {:,.2f} BTC", df["volume"].sum())

        # Check for gaps
        idx = df.index
        diffs = idx[1:] - idx[:-1]
        expected = timedelta(minutes=15)
        gaps = [d for d in diffs if d > expected * 5]
        logger.info("Gaps > 75min detected: {}", len(gaps))

        # Save to temp Parquet
        out_path = Path("data/parquet/test_btcusdt_m15.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        logger.info(
            "Saved to: {} ({:.1f} KB)", out_path, out_path.stat().st_size / 1024
        )

    except Exception as e:
        logger.error("Failed: {}", e)
        raise
    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
