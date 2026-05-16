"""Demo script: download 1 month of XAUUSD M15 via MT5 and display stats.

Run manually on Windows with MT5 terminal running:
    uv run python scripts/test_mt5_loader.py

This is NOT a pytest test — it's an integration test requiring a live MT5 connection.
"""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from config.settings import Settings
from data.ingestion.mt5_loader import MT5Loader


def main() -> None:
    settings = Settings()

    logger.info("=== MT5 Loader Integration Test ===")
    logger.info("Symbol: XAUUSD | Timeframe: M15 | Range: last 30 days")

    loader = MT5Loader(
        path=settings.mt5_path,
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
    )

    try:
        loader.connect()

        # Show available XAU symbols (helps debug broker naming)
        xau_symbols = loader.get_available_symbols("*XAU*")
        logger.info("Available XAU symbols: {}", xau_symbols)

        # Download last 30 days of M15
        end = datetime.now(tz=UTC)
        start = datetime(end.year, end.month - 1 if end.month > 1 else 12, end.day, tzinfo=UTC)

        df = loader.download_ohlcv("XAUUSD", "M15", start, end)

        # Stats
        logger.info("--- Results ---")
        logger.info("Bars downloaded: {}", len(df))
        logger.info("Date range: {} → {}", df.index[0], df.index[-1])
        logger.info("Price range: {:.2f} → {:.2f}", df["close"].min(), df["close"].max())
        logger.info("Mean spread: {:.1f}", df["spread"].mean())
        logger.info("Total tick volume: {:,}", df["tick_volume"].sum())

        # Save to temp Parquet
        out_path = Path("data/parquet/test_xauusd_m15.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        logger.info("Saved to: {} ({:.1f} KB)", out_path, out_path.stat().st_size / 1024)

    except Exception as e:
        logger.error("Failed: {}", e)
        raise
    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
