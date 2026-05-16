"""Partitioned Parquet storage for OHLCV data.

Stores data as:
    {base_path}/ohlcv/{symbol}/{timeframe}/year={YYYY}/month={MM}/data.parquet

Supports atomic writes, incremental updates, and efficient partial-range reads
via PyArrow dataset filtering on year/month partitions.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:
    from data.ingestion.mt5_loader import MT5Loader

# Expected column schema (index is separate)
PARQUET_SCHEMA: dict[str, str] = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "spread": "float64",
    "real_volume": "float64",
}

REQUIRED_COLUMNS = set(PARQUET_SCHEMA.keys())

INDEX_NAME = "timestamp_utc"

# Timeframe → bar duration in minutes (for expected_bars calculation)
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


class ParquetStoreError(Exception):
    """Generic Parquet store error."""


class ParquetStoreSchemaError(ParquetStoreError):
    """Schema validation failed (wrong columns, non-UTC index, etc.)."""


class ParquetStoreIOError(ParquetStoreError):
    """I/O error (permissions, disk full, etc.)."""


class ParquetStore:
    """Partitioned Parquet storage for OHLCV data.

    Usage:
        store = ParquetStore(base_path=Path("data/parquet"))
        store.save(df, "XAUUSD", "M15")
        df = store.load("XAUUSD", "M15", start_date, end_date)
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize store with base directory.

        Args:
            base_path: Root directory for parquet files. Defaults to 'data/parquet'.
        """
        self._base_path = base_path or Path("data/parquet")

    def _symbol_tf_path(self, symbol: str, timeframe: str) -> Path:
        """Get the directory path for a symbol/timeframe combination."""
        return self._base_path / "ohlcv" / symbol / timeframe

    def _partition_path(self, symbol: str, timeframe: str, year: int, month: int) -> Path:
        """Get path to a specific year/month partition."""
        return self._symbol_tf_path(symbol, timeframe) / f"year={year}" / f"month={month:02d}"

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema before saving.

        Raises:
            ParquetStoreSchemaError: If schema is invalid.
        """
        # Check index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ParquetStoreSchemaError(
                f"Index must be DatetimeIndex, got {type(df.index).__name__}"
            )

        # Check index is tz-aware UTC
        if df.index.tz is None:
            raise ParquetStoreSchemaError(
                "Index must be tz-aware UTC. Got naive (tz=None). "
                "Use df.index = df.index.tz_localize('UTC')"
            )

        if str(df.index.tz) != "UTC":
            raise ParquetStoreSchemaError(
                f"Index must be UTC, got tz={df.index.tz}. "
                "Use df.index = df.index.tz_convert('UTC')"
            )

        # Check required columns exist
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ParquetStoreSchemaError(
                f"Missing required columns: {sorted(missing)}. "
                f"Expected: {sorted(REQUIRED_COLUMNS)}"
            )

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save DataFrame to partitioned Parquet files.

        Merges with existing data if partitions already exist.
        Extra columns beyond the standard schema are preserved.

        Args:
            df: OHLCV DataFrame with UTC DatetimeIndex named 'timestamp_utc'.
            symbol: Symbol name (e.g. "XAUUSD").
            timeframe: Timeframe (e.g. "M15").

        Raises:
            ParquetStoreSchemaError: If DataFrame schema is invalid.
            ParquetStoreIOError: If write fails.
        """
        if df.empty:
            logger.debug("Empty DataFrame, nothing to save for {}/{}", symbol, timeframe)
            return

        self._validate_schema(df)

        # Ensure index name is correct
        df = df.copy()
        df.index.name = INDEX_NAME

        # Split by (year, month)
        dt_index = pd.DatetimeIndex(df.index)
        df["_year"] = dt_index.year
        df["_month"] = dt_index.month
        groups = df.groupby(["_year", "_month"])

        nb_partitions = 0
        total_bars = 0

        for group_key, partition_df in groups:
            y = int(str(group_key[0]))
            m = int(str(group_key[1]))
            partition_df = partition_df.drop(columns=["_year", "_month"])
            self._write_partition(symbol, timeframe, y, m, partition_df)
            nb_partitions += 1
            total_bars += len(partition_df)
            logger.debug(
                "  Partition year={}/month={:02d}: {} bars", y, m, len(partition_df)
            )

        logger.info(
            "Saved {} bars to {} partitions for {}/{}",
            total_bars,
            nb_partitions,
            symbol,
            timeframe,
        )

    def _write_partition(
        self,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
        new_df: pd.DataFrame,
    ) -> None:
        """Write a single partition with merge and atomic replace.

        If partition exists, merges (dedup on index, keep='last').
        Uses tmp file + os.replace for atomicity.
        """
        partition_dir = self._partition_path(symbol, timeframe, year, month)
        partition_dir.mkdir(parents=True, exist_ok=True)

        final_path = partition_dir / "data.parquet"
        tmp_path = partition_dir / "data.parquet.tmp"

        # Clean orphan .tmp files
        if tmp_path.exists():
            tmp_path.unlink()

        # Merge with existing data if present
        if final_path.exists():
            try:
                existing_df = pd.read_parquet(final_path, engine="pyarrow")
                # Ensure existing has correct index name
                if existing_df.index.name != INDEX_NAME:
                    existing_df.index.name = INDEX_NAME
                merged = pd.concat([existing_df, new_df])
                merged = merged[~merged.index.duplicated(keep="last")]
                merged = merged.sort_index()
                new_df = merged
            except Exception as e:
                raise ParquetStoreIOError(
                    f"Failed to read existing partition {final_path}: {e}"
                ) from e

        # Write to tmp file
        try:
            table = pa.Table.from_pandas(new_df, preserve_index=True)
            pq.write_table(table, tmp_path, compression="snappy")
        except Exception as e:
            # Clean up tmp on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise ParquetStoreIOError(f"Failed to write {tmp_path}: {e}") from e

        # Atomic replace
        try:
            os.replace(tmp_path, final_path)
        except OSError as e:
            raise ParquetStoreIOError(
                f"Failed atomic replace {tmp_path} → {final_path}: {e}"
            ) from e

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet store.

        Args:
            symbol: Symbol name.
            timeframe: Timeframe.
            start_date: Optional start filter (inclusive).
            end_date: Optional end filter (inclusive).

        Returns:
            DataFrame with UTC DatetimeIndex, sorted, deduplicated.
            Empty DataFrame with correct schema if no data in range.

        Raises:
            ParquetStoreError: If symbol/timeframe has never been saved.
        """
        base_dir = self._symbol_tf_path(symbol, timeframe)

        if not base_dir.exists():
            raise ParquetStoreError(f"No data for {symbol}/{timeframe}")

        # Determine which partitions to read
        partitions = self._find_partitions(base_dir, start_date, end_date)

        if not partitions:
            return self._empty_dataframe()

        # Read and concatenate
        frames: list[pd.DataFrame] = []
        for ppath in partitions:
            parquet_file = ppath / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                if df.index.name != INDEX_NAME:
                    df.index.name = INDEX_NAME
                frames.append(df)

        if not frames:
            return self._empty_dataframe()

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        # Apply date filter (partition-level filtering is coarse)
        if start_date is not None:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=UTC)
            result = result[result.index >= start_date]
        if end_date is not None:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=UTC)
            result = result[result.index <= end_date]

        if result.empty:
            return self._empty_dataframe()

        # Coverage warning
        if start_date and end_date and not result.empty:
            expected = self._expected_bars_market_hours(start_date, end_date, timeframe)
            if expected > 0:
                coverage = len(result) / expected
                if coverage < 0.95:
                    logger.warning(
                        "Low coverage for {}/{}: {:.1%} ({} bars vs {} expected)",
                        symbol,
                        timeframe,
                        coverage,
                        len(result),
                        expected,
                    )

        logger.info(
            "Loaded {} bars from {}/{} [{} → {}]",
            len(result),
            symbol,
            timeframe,
            result.index[0],
            result.index[-1],
        )

        return result

    def _find_partitions(
        self,
        base_dir: Path,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        """Find partition directories matching the date range."""
        all_partitions: list[tuple[int, int, Path]] = []

        for year_dir in base_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.split("=")[1])
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.split("=")[1])
                all_partitions.append((year, month, month_dir))

        # Filter by date range
        if start_date or end_date:
            filtered = []
            for year, month, path in all_partitions:
                # Partition covers the entire month
                part_start = datetime(year, month, 1, tzinfo=UTC)
                if month == 12:
                    part_end = datetime(year + 1, 1, 1, tzinfo=UTC) - timedelta(seconds=1)
                else:
                    part_end = datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(seconds=1)

                if start_date and start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=UTC)
                if end_date and end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=UTC)

                # Include if partition overlaps with requested range
                if end_date and part_start > end_date:
                    continue
                if start_date and part_end < start_date:
                    continue
                filtered.append(path)
            return filtered

        return [path for _, _, path in all_partitions]

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with correct schema."""
        df = pd.DataFrame(
            columns=list(PARQUET_SCHEMA.keys()),
            dtype="float64",
        )
        df.index = pd.DatetimeIndex([], tz=UTC, name=INDEX_NAME)
        return df

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if any data exists for this symbol/timeframe."""
        base_dir = self._symbol_tf_path(symbol, timeframe)
        if not base_dir.exists():
            return False
        # Check if any data.parquet files exist
        return any(base_dir.rglob("data.parquet"))

    def last_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Get the last (most recent) timestamp in the store."""
        base_dir = self._symbol_tf_path(symbol, timeframe)
        if not base_dir.exists():
            return None

        # Find the latest partition
        partitions: list[tuple[int, int, Path]] = []
        for year_dir in base_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.split("=")[1])
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.split("=")[1])
                partitions.append((year, month, month_dir))

        if not partitions:
            return None

        # Sort descending and read last partition
        partitions.sort(reverse=True)
        for _, _, path in partitions:
            parquet_file = path / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                if not df.empty:
                    ts: datetime = df.index[-1].to_pydatetime()
                    return ts
        return None

    def first_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Get the first (oldest) timestamp in the store."""
        base_dir = self._symbol_tf_path(symbol, timeframe)
        if not base_dir.exists():
            return None

        partitions: list[tuple[int, int, Path]] = []
        for year_dir in base_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.split("=")[1])
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.split("=")[1])
                partitions.append((year, month, month_dir))

        if not partitions:
            return None

        # Sort ascending and read first partition
        partitions.sort()
        for _, _, path in partitions:
            parquet_file = path / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                if not df.empty:
                    ts: datetime = df.index[0].to_pydatetime()
                    return ts
        return None

    def incremental_update(
        self,
        symbol: str,
        timeframe: str,
        loader: MT5Loader,
        end_date: datetime | None = None,
    ) -> int:
        """Download and save new data since last stored timestamp.

        Args:
            symbol: Symbol to update.
            timeframe: Timeframe to update.
            loader: Data loader (MT5Loader or compatible interface).
            end_date: End date for download. Defaults to now UTC.

        Returns:
            Number of new bars added.
        """
        if end_date is None:
            end_date = datetime.now(tz=UTC)
        elif end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        last_ts = self.last_timestamp(symbol, timeframe)

        if last_ts is None:
            # First run — download from configured start date
            from config.settings import Settings

            settings = Settings()
            start_date = datetime.fromisoformat(settings.data_start_date).replace(tzinfo=UTC)
        else:
            # Advance past last bar
            tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
            start_date = last_ts + timedelta(minutes=tf_minutes)

        if start_date >= end_date:
            logger.info(
                "No new data needed for {}/{} — already up to date", symbol, timeframe
            )
            return 0

        # Download via loader
        with loader:
            df = loader.download_ohlcv(symbol, timeframe, start_date, end_date)

        if df.empty:
            logger.info("No new bars returned for {}/{}", symbol, timeframe)
            return 0

        nb_new = len(df)
        self.save(df, symbol, timeframe)

        logger.info(
            "Incremental update: {} new bars for {}/{}", nb_new, symbol, timeframe
        )
        return nb_new

    def delete(self, symbol: str, timeframe: str) -> None:
        """Delete all data for a symbol/timeframe."""
        base_dir = self._symbol_tf_path(symbol, timeframe)
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info("Deleted all data for {}/{}", symbol, timeframe)

    def stats(self, symbol: str, timeframe: str) -> dict:
        """Compute statistics for stored data.

        Returns:
            Dict with: nb_bars, first_timestamp, last_timestamp, span_days,
            nb_partitions, total_size_mb, expected_bars_market_hours, coverage_ratio.
        """
        base_dir = self._symbol_tf_path(symbol, timeframe)
        if not base_dir.exists():
            raise ParquetStoreError(f"No data for {symbol}/{timeframe}")

        # Count partitions and size
        parquet_files = list(base_dir.rglob("data.parquet"))
        nb_partitions = len(parquet_files)
        total_size_bytes = sum(f.stat().st_size for f in parquet_files)

        # Load all to get stats
        first_ts = self.first_timestamp(symbol, timeframe)
        last_ts = self.last_timestamp(symbol, timeframe)

        if first_ts is None or last_ts is None:
            return {
                "nb_bars": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "span_days": 0.0,
                "nb_partitions": nb_partitions,
                "total_size_mb": total_size_bytes / (1024 * 1024),
                "expected_bars_market_hours": 0,
                "coverage_ratio": 0.0,
            }

        # Count total bars
        nb_bars = 0
        for f in parquet_files:
            table = pq.read_table(f)
            nb_bars += table.num_rows

        span = last_ts - first_ts
        span_days = span.total_seconds() / 86400

        expected = self._expected_bars_market_hours(first_ts, last_ts, timeframe)
        coverage = nb_bars / expected if expected > 0 else 0.0

        return {
            "nb_bars": nb_bars,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "span_days": span_days,
            "nb_partitions": nb_partitions,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "expected_bars_market_hours": expected,
            "coverage_ratio": coverage,
        }

    def list_available(self) -> dict[str, list[str]]:
        """List all available symbol/timeframe combinations.

        Returns:
            Dict mapping symbol → list of timeframes with data.
        """
        ohlcv_dir = self._base_path / "ohlcv"
        if not ohlcv_dir.exists():
            return {}

        result: dict[str, list[str]] = {}
        for symbol_dir in ohlcv_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            timeframes = []
            for tf_dir in symbol_dir.iterdir():
                if tf_dir.is_dir() and any(tf_dir.rglob("data.parquet")):
                    timeframes.append(tf_dir.name)
            if timeframes:
                result[symbol_dir.name] = sorted(timeframes)
        return result

    @staticmethod
    def _expected_bars_market_hours(
        start: datetime, end: datetime, timeframe: str
    ) -> int:
        """Calculate expected number of bars for XAU excluding weekends.

        XAU market hours: Sunday 22:00 UTC → Friday 22:00 UTC (continuous).
        Weekend close: Friday 22:00 UTC → Sunday 22:00 UTC.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)

        # Count market-open minutes between start and end
        # Iterate day by day and exclude weekend hours
        total_minutes = 0.0
        current = start

        while current < end:
            # Determine next boundary
            weekday = current.weekday()  # 0=Mon ... 6=Sun

            if weekday == 4:  # Friday
                # Market closes at 22:00 UTC Friday
                friday_close = current.replace(hour=22, minute=0, second=0, microsecond=0)
                if current < friday_close:
                    # Market open until 22:00
                    boundary = min(friday_close, end)
                    total_minutes += (boundary - current).total_seconds() / 60
                # Skip to Sunday 22:00
                current = friday_close + timedelta(hours=48)
            elif weekday == 5:  # Saturday — market closed
                # Jump to Sunday 22:00
                sunday_open = current.replace(hour=22, minute=0, second=0, microsecond=0)
                sunday_open += timedelta(days=1)
                current = sunday_open
            elif weekday == 6:  # Sunday
                sunday_open = current.replace(hour=22, minute=0, second=0, microsecond=0)
                if current < sunday_open:
                    # Market closed until 22:00 Sunday
                    current = sunday_open
                else:
                    # Market open from 22:00 Sunday
                    next_day = (current + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    boundary = min(next_day, end)
                    total_minutes += (boundary - current).total_seconds() / 60
                    current = next_day
            else:
                # Mon-Thu: market open all day
                next_day = (current + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                boundary = min(next_day, end)
                total_minutes += (boundary - current).total_seconds() / 60
                current = next_day

        if tf_minutes == 0:
            return 0
        return int(total_minutes / tf_minutes)
