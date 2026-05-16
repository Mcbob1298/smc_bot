"""Tests for ParquetStore — partitioned Parquet storage for OHLCV data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.storage.parquet_store import (
    INDEX_NAME,
    PARQUET_SCHEMA,
    ParquetStore,
    ParquetStoreError,
    ParquetStoreSchemaError,
)


def _make_ohlcv_df(
    start: datetime,
    periods: int,
    freq_minutes: int = 15,
    base_price: float = 2000.0,
) -> pd.DataFrame:
    """Create a valid OHLCV DataFrame for testing."""
    idx = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min", tz=UTC)
    rng = np.random.default_rng(42)
    opens = base_price + np.cumsum(rng.normal(0, 0.5, periods))
    df = pd.DataFrame(
        {
            "open": opens,
            "high": opens + rng.uniform(0.5, 2.0, periods),
            "low": opens - rng.uniform(0.5, 2.0, periods),
            "close": opens + rng.normal(0, 0.3, periods),
            "volume": rng.uniform(100, 1000, periods),
            "spread": rng.uniform(0.02, 0.05, periods),
            "real_volume": rng.uniform(50, 500, periods),
        },
        index=idx,
    )
    df.index.name = INDEX_NAME
    return df


class TestSaveAndLoad:
    """Tests for basic save/load functionality."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload produces identical data."""
        store = ParquetStore(base_path=tmp_path)
        original = _make_ohlcv_df(datetime(2024, 3, 15, 10, 0, tzinfo=UTC), periods=100)

        store.save(original, "XAUUSD", "M15")
        loaded = store.load("XAUUSD", "M15")

        pd.testing.assert_frame_equal(loaded, original, check_freq=False)
        assert loaded.index.name == INDEX_NAME
        assert str(loaded.index.tz) == "UTC"
        # Check dtypes match
        for col in PARQUET_SCHEMA:
            assert loaded[col].dtype == np.float64

    def test_save_partitioning_correct(self, tmp_path: Path) -> None:
        """Verify partition directory structure."""
        store = ParquetStore(base_path=tmp_path)
        # Data spanning Jan-Feb 2024 (1500 bars × 15min = ~15 days, from Jan 20 → Feb 4)
        df = _make_ohlcv_df(datetime(2024, 1, 20, 10, 0, tzinfo=UTC), periods=1500)

        store.save(df, "XAUUSD", "M15")

        # Check directory structure
        jan_path = tmp_path / "ohlcv/XAUUSD/M15/year=2024/month=01/data.parquet"
        feb_path = tmp_path / "ohlcv/XAUUSD/M15/year=2024/month=02/data.parquet"
        assert jan_path.exists()
        assert feb_path.exists()

    def test_save_rejects_naive_index(self, tmp_path: Path) -> None:
        """Naive datetime index raises ParquetStoreSchemaError."""
        store = ParquetStore(base_path=tmp_path)
        idx = pd.date_range("2024-01-01", periods=10, freq="15min")  # naive
        df = pd.DataFrame(
            {col: np.ones(10) for col in PARQUET_SCHEMA},
            index=idx,
        )

        with pytest.raises(ParquetStoreSchemaError, match="tz-aware UTC"):
            store.save(df, "XAUUSD", "M15")

    def test_save_rejects_non_utc_index(self, tmp_path: Path) -> None:
        """Non-UTC timezone raises ParquetStoreSchemaError."""
        store = ParquetStore(base_path=tmp_path)
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="Europe/Paris")
        df = pd.DataFrame(
            {col: np.ones(10) for col in PARQUET_SCHEMA},
            index=idx,
        )

        with pytest.raises(ParquetStoreSchemaError, match="must be UTC"):
            store.save(df, "XAUUSD", "M15")

    def test_save_rejects_missing_columns(self, tmp_path: Path) -> None:
        """Missing required column raises ParquetStoreSchemaError."""
        store = ParquetStore(base_path=tmp_path)
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz=UTC)
        df = pd.DataFrame(
            {"open": np.ones(10), "close": np.ones(10)},  # missing high, low, etc.
            index=idx,
        )

        with pytest.raises(ParquetStoreSchemaError, match="Missing required columns"):
            store.save(df, "XAUUSD", "M15")

    def test_save_preserves_extra_columns(self, tmp_path: Path) -> None:
        """Extra columns beyond schema are preserved through save/load."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 3, 1, tzinfo=UTC), periods=50)
        df["is_killzone"] = True
        df["custom_indicator"] = 42.0

        store.save(df, "XAUUSD", "M15")
        loaded = store.load("XAUUSD", "M15")

        assert "is_killzone" in loaded.columns
        assert "custom_indicator" in loaded.columns
        assert loaded["is_killzone"].all()
        assert (loaded["custom_indicator"] == 42.0).all()


class TestLoadFiltering:
    """Tests for load with date range filtering."""

    def test_load_partial_range(self, tmp_path: Path) -> None:
        """Load a subset of stored data by date range."""
        store = ParquetStore(base_path=tmp_path)
        # Save 6 months of data
        df = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=6000)
        store.save(df, "XAUUSD", "M15")

        # Load only March
        start = datetime(2024, 3, 1, tzinfo=UTC)
        end = datetime(2024, 3, 31, 23, 59, tzinfo=UTC)
        loaded = store.load("XAUUSD", "M15", start_date=start, end_date=end)

        assert not loaded.empty
        assert loaded.index[0] >= start
        assert loaded.index[-1] <= end

    def test_load_range_outside_data(self, tmp_path: Path) -> None:
        """Load a range where no data exists returns empty DataFrame."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 3, 1, tzinfo=UTC), periods=100)
        store.save(df, "XAUUSD", "M15")

        # Load a range in June (no data there)
        start = datetime(2024, 6, 1, tzinfo=UTC)
        end = datetime(2024, 6, 30, tzinfo=UTC)
        loaded = store.load("XAUUSD", "M15", start_date=start, end_date=end)

        assert loaded.empty
        assert list(loaded.columns) == list(PARQUET_SCHEMA.keys())
        assert loaded.index.name == INDEX_NAME
        assert str(loaded.index.tz) == "UTC"

    def test_load_unknown_symbol(self, tmp_path: Path) -> None:
        """Load non-existent symbol raises ParquetStoreError."""
        store = ParquetStore(base_path=tmp_path)

        with pytest.raises(ParquetStoreError, match="No data for"):
            store.load("FAKESYM", "M15")


class TestMerge:
    """Tests for merge behavior on overlapping saves."""

    def test_merge_on_resave_overlapping(self, tmp_path: Path) -> None:
        """Overlapping save merges correctly, new values overwrite old."""
        store = ParquetStore(base_path=tmp_path)

        # Save days 1-10
        df1 = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=100)
        store.save(df1, "XAUUSD", "M15")

        # Save days 5-15 with modified close prices
        df2 = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=200)
        # Modify a value that overlaps with df1 to verify keep='last'
        overlap_idx = df1.index[50]
        df2_overlap = df2[df2.index == overlap_idx]
        if not df2_overlap.empty:
            df2.loc[overlap_idx, "close"] = 9999.0

        store.save(df2, "XAUUSD", "M15")
        loaded = store.load("XAUUSD", "M15")

        # New data should have overwritten
        if overlap_idx in loaded.index:
            assert loaded.loc[overlap_idx, "close"] == 9999.0

        # No duplicates
        assert not loaded.index.duplicated().any()

    def test_merge_on_resave_disjoint(self, tmp_path: Path) -> None:
        """Disjoint saves result in union of both datasets."""
        store = ParquetStore(base_path=tmp_path)

        df1 = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=50)
        df2 = _make_ohlcv_df(datetime(2024, 3, 1, tzinfo=UTC), periods=50)

        store.save(df1, "XAUUSD", "M15")
        store.save(df2, "XAUUSD", "M15")

        loaded = store.load("XAUUSD", "M15")
        assert len(loaded) == 100
        assert loaded.index[0] == df1.index[0]
        assert loaded.index[-1] == df2.index[-1]


class TestIncrementalUpdate:
    """Tests for incremental_update method."""

    def _make_mock_loader(self, df: pd.DataFrame):
        """Create a minimal mock loader that returns the given DataFrame."""

        class MockLoader:
            def __init__(self, data: pd.DataFrame):
                self._data = data

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
                # Filter data to requested range
                mask = (self._data.index >= start_date) & (self._data.index <= end_date)
                return self._data[mask].copy()

        return MockLoader(df)

    def test_incremental_update_first_run(self, tmp_path: Path, monkeypatch) -> None:
        """First run downloads from data_start_date."""
        store = ParquetStore(base_path=tmp_path)

        # Mock settings
        monkeypatch.setenv("DATA_START_DATE", "2024-01-01")

        fake_data = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=500)
        loader = self._make_mock_loader(fake_data)

        nb_new = store.incremental_update("XAUUSD", "M15", loader)

        assert nb_new == 500
        assert store.exists("XAUUSD", "M15")

    def test_incremental_update_second_run_idempotent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Second call with same data returns 0 new bars."""
        store = ParquetStore(base_path=tmp_path)
        monkeypatch.setenv("DATA_START_DATE", "2024-01-01")

        fake_data = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=100)
        loader = self._make_mock_loader(fake_data)

        nb1 = store.incremental_update("XAUUSD", "M15", loader)
        assert nb1 == 100

        # Second call — loader has same data, start will be past last bar
        nb2 = store.incremental_update("XAUUSD", "M15", loader)
        assert nb2 == 0

    def test_incremental_update_appends_correctly(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Incremental update appends new data after existing."""
        store = ParquetStore(base_path=tmp_path)
        monkeypatch.setenv("DATA_START_DATE", "2024-01-01")

        # First batch: January
        jan_data = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=200)
        loader1 = self._make_mock_loader(jan_data)
        store.incremental_update("XAUUSD", "M15", loader1)

        # Second batch: February (continuing from where jan left off)
        all_data = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=400)
        loader2 = self._make_mock_loader(all_data)
        nb_new = store.incremental_update("XAUUSD", "M15", loader2)

        assert nb_new == 200  # only new bars
        loaded = store.load("XAUUSD", "M15")
        assert len(loaded) == 400


class TestMetadata:
    """Tests for metadata methods (exists, timestamps, stats, etc.)."""

    def test_last_timestamp(self, tmp_path: Path) -> None:
        """last_timestamp returns correct value across partitions."""
        store = ParquetStore(base_path=tmp_path)
        # Spans Jan-Feb
        df = _make_ohlcv_df(datetime(2024, 1, 15, tzinfo=UTC), periods=500)
        store.save(df, "XAUUSD", "M15")

        last_ts = store.last_timestamp("XAUUSD", "M15")
        assert last_ts == df.index[-1].to_pydatetime()

    def test_first_timestamp(self, tmp_path: Path) -> None:
        """first_timestamp returns correct value."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 1, 15, 8, 30, tzinfo=UTC), periods=500)
        store.save(df, "XAUUSD", "M15")

        first_ts = store.first_timestamp("XAUUSD", "M15")
        assert first_ts == df.index[0].to_pydatetime()

    def test_exists_true_false(self, tmp_path: Path) -> None:
        """exists returns True/False correctly."""
        store = ParquetStore(base_path=tmp_path)
        assert not store.exists("XAUUSD", "M15")

        df = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=10)
        store.save(df, "XAUUSD", "M15")

        assert store.exists("XAUUSD", "M15")
        assert not store.exists("XAUUSD", "H4")

    def test_delete(self, tmp_path: Path) -> None:
        """delete removes all data for symbol/timeframe."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=100)
        store.save(df, "XAUUSD", "M15")

        assert store.exists("XAUUSD", "M15")
        store.delete("XAUUSD", "M15")

        assert not store.exists("XAUUSD", "M15")
        assert not (tmp_path / "ohlcv/XAUUSD/M15").exists()

    def test_stats_calculation(self, tmp_path: Path) -> None:
        """stats returns correct metrics."""
        store = ParquetStore(base_path=tmp_path)
        # 30 days of M15 data (Mon-Fri only to be realistic)
        df = _make_ohlcv_df(datetime(2024, 2, 1, tzinfo=UTC), periods=2000)
        store.save(df, "XAUUSD", "M15")

        s = store.stats("XAUUSD", "M15")

        assert s["nb_bars"] == 2000
        assert s["first_timestamp"] == df.index[0].to_pydatetime()
        assert s["last_timestamp"] == df.index[-1].to_pydatetime()
        assert s["span_days"] > 0
        assert s["nb_partitions"] >= 1
        assert s["total_size_mb"] > 0
        assert s["expected_bars_market_hours"] > 0
        assert 0 < s["coverage_ratio"] <= 2.0  # can exceed 1 if weekends in data

    def test_list_available(self, tmp_path: Path) -> None:
        """list_available returns correct mapping."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=50)

        store.save(df, "XAUUSD", "M15")
        store.save(df, "XAUUSD", "H4")

        available = store.list_available()
        assert "XAUUSD" in available
        assert sorted(available["XAUUSD"]) == ["H4", "M15"]


class TestAtomicityAndPartitions:
    """Tests for atomic save and partition boundary handling."""

    def test_save_preserves_tz_through_partition_boundary(self, tmp_path: Path) -> None:
        """Data spanning Dec/Jan preserves tz, no duplication or loss."""
        store = ParquetStore(base_path=tmp_path)
        # Spans Dec 2023 → Jan 2024
        df = _make_ohlcv_df(datetime(2023, 12, 28, tzinfo=UTC), periods=500)
        store.save(df, "XAUUSD", "M15")

        loaded = store.load("XAUUSD", "M15")

        assert len(loaded) == len(df)
        assert str(loaded.index.tz) == "UTC"
        assert not loaded.index.duplicated().any()
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    def test_atomic_save_no_tmp_files_after_success(self, tmp_path: Path) -> None:
        """No .tmp files remain after successful save."""
        store = ParquetStore(base_path=tmp_path)
        df = _make_ohlcv_df(datetime(2024, 1, 1, tzinfo=UTC), periods=100)
        store.save(df, "XAUUSD", "M15")

        # Check no .tmp files anywhere
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []

    def test_atomic_save_cleans_orphan_tmp(self, tmp_path: Path) -> None:
        """Orphan .tmp files from previous crashes are cleaned on save."""
        store = ParquetStore(base_path=tmp_path)

        # Create an orphan .tmp file manually
        partition_dir = tmp_path / "ohlcv/XAUUSD/M15/year=2024/month=01"
        partition_dir.mkdir(parents=True)
        orphan = partition_dir / "data.parquet.tmp"
        orphan.write_text("corrupt orphan data")

        assert orphan.exists()

        # Save new data to the same partition
        df = _make_ohlcv_df(datetime(2024, 1, 15, tzinfo=UTC), periods=50)
        store.save(df, "XAUUSD", "M15")

        # Orphan should be gone, replaced by valid data
        assert not orphan.exists()
        assert (partition_dir / "data.parquet").exists()
