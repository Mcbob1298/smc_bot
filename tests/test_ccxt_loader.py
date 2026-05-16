"""Tests for CCXT data loader.

All tests use mocks — no live exchange connection needed.
Run `scripts/test_ccxt_loader.py` for integration test with real Binance data.
"""

from datetime import UTC, datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest
from loguru import logger

from data.ingestion.ccxt_loader import (
    CCXTConnectionError,
    CCXTDataError,
    CCXTLoader,
)


def _make_fake_ohlcv(
    n: int,
    start_ts_ms: int = 1700000000000,
    interval_ms: int = 900_000,
) -> list[list]:
    """Create fake ccxt OHLCV data.

    Returns list of [timestamp_ms, open, high, low, close, volume].
    """
    ohlcv = []
    for i in range(n):
        ts = start_ts_ms + i * interval_ms
        o = 35000.0 + i * 10.0
        h = o + 50.0
        l = o - 30.0  # noqa: E741
        c = o + 20.0
        v = 100.0 + i
        ohlcv.append([ts, o, h, l, c, v])
    return ohlcv


def _make_fake_ohlcv_with_gap(
    n: int,
    gap_at: int = 5,
    gap_duration_ms: int = 600_000,
    start_ts_ms: int = 1700000000000,
    interval_ms: int = 900_000,
) -> list[list]:
    """Create fake OHLCV with a gap at position `gap_at`."""
    ohlcv = []
    ts = start_ts_ms
    for i in range(n):
        if i == gap_at:
            ts += gap_duration_ms  # inject gap
        o = 35000.0 + i * 10.0
        h = o + 50.0
        l = o - 30.0  # noqa: E741
        c = o + 20.0
        v = 100.0 + i
        ohlcv.append([ts, o, h, l, c, v])
        ts += interval_ms
    return ohlcv


class TestCCXTLoaderConnection:
    """Test connection handling."""

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_connect_success(self, mock_ccxt: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        mock_ccxt.binance.return_value = mock_exchange

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()

        assert loader.is_connected()
        mock_exchange.load_markets.assert_called_once()

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_connect_failure_network(self, mock_ccxt: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.load_markets.side_effect = ccxt.NetworkError("timeout")
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.NetworkError = ccxt.NetworkError
        mock_ccxt.ExchangeNotAvailable = ccxt.ExchangeNotAvailable
        mock_ccxt.AuthenticationError = ccxt.AuthenticationError

        loader = CCXTLoader(exchange_id="binance")
        with pytest.raises(CCXTConnectionError, match="Failed to connect"):
            loader.connect()

        assert not loader.is_connected()

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_disconnect(self, mock_ccxt: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_ccxt.binance.return_value = mock_exchange

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()
        loader.disconnect()

        assert not loader.is_connected()
        mock_exchange.close.assert_called_once()

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_context_manager(self, mock_ccxt: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_ccxt.binance.return_value = mock_exchange

        with CCXTLoader(exchange_id="binance") as loader:
            assert loader.is_connected()
        assert not loader.is_connected()


class TestCCXTLoaderDownload:
    """Test OHLCV download logic."""

    def _setup_connected_loader(
        self, mock_ccxt: MagicMock, ohlcv_data: list[list] | None = None
    ) -> tuple[CCXTLoader, MagicMock]:
        """Helper to create a connected loader with mocked exchange."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        if ohlcv_data is not None:
            mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_ccxt.binance.return_value = mock_exchange

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()
        return loader, mock_exchange

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_basic(self, mock_ccxt: MagicMock) -> None:
        """Test basic download returns correct DataFrame structure."""
        fake_data = _make_fake_ohlcv(100)
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "open", "high", "low", "close", "volume", "spread", "real_volume"
        ]
        assert df.index.name == "timestamp_utc"
        assert len(df) <= 100

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_not_connected_raises(self, mock_ccxt: MagicMock) -> None:
        loader = CCXTLoader()
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(CCXTConnectionError, match="Not connected"):
            loader.download_ohlcv("BTCUSDT", "M15", start, end)

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_unknown_symbol(self, mock_ccxt: MagicMock) -> None:
        loader, _ = self._setup_connected_loader(mock_ccxt, [])

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(CCXTDataError, match="not found"):
            loader.download_ohlcv("FAKECOIN", "M15", start, end)

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_unknown_timeframe(self, mock_ccxt: MagicMock) -> None:
        loader, _ = self._setup_connected_loader(mock_ccxt, [])

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2023, 2, 1, tzinfo=UTC)

        with pytest.raises(CCXTDataError, match="Unknown timeframe"):
            loader.download_ohlcv("BTCUSDT", "M99", start, end)

    @patch("data.ingestion.ccxt_loader.ccxt")
    @patch("data.ingestion.ccxt_loader._MAX_BARS_PER_REQUEST", 50)
    def test_download_pagination(self, mock_ccxt: MagicMock) -> None:
        """Test pagination with multiple chunks."""
        chunk1 = _make_fake_ohlcv(50, start_ts_ms=1700000000000, interval_ms=900_000)
        chunk2 = _make_fake_ohlcv(30, start_ts_ms=1700045000001, interval_ms=900_000)

        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_exchange.fetch_ohlcv.side_effect = [chunk1, chunk2]
        mock_ccxt.binance.return_value = mock_exchange

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert mock_exchange.fetch_ohlcv.call_count == 2
        # Total deduplicated bars
        assert len(df) <= 80

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_partial_data_warns(self, mock_ccxt: MagicMock) -> None:
        """Warn when exchange returns less history than requested."""
        # Data starting Nov 6, but we request from Jan 1
        fake_data = _make_fake_ohlcv(
            100,
            start_ts_ms=int(datetime(2023, 11, 6, tzinfo=UTC).timestamp() * 1000),
            interval_ms=900_000,
        )
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        log_output = StringIO()
        sink_id = logger.add(log_output, level="WARNING", format="{message}")

        try:
            start = datetime(2023, 1, 1, tzinfo=UTC)
            end = datetime(2023, 12, 1, tzinfo=UTC)
            df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

            assert len(df) == 100
            log_text = log_output.getvalue()
            assert "PARTIAL DATA" in log_text
            assert "2023-01-01" in log_text
        finally:
            logger.remove(sink_id)

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_download_uses_symbol_mapping(self, mock_ccxt: MagicMock) -> None:
        """Verify symbol_map resolves canonical to ccxt format."""
        fake_data = _make_fake_ohlcv(50)
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_exchange.fetch_ohlcv.return_value = fake_data
        mock_ccxt.binance.return_value = mock_exchange

        # Map BTCUSDT → BTC/USDT (standard ccxt format)
        loader = CCXTLoader(
            exchange_id="binance",
            symbol_map={"BTCUSDT": "BTC/USDT"},
        )
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        loader.download_ohlcv("BTCUSDT", "M15", start, end)

        # Verify fetch_ohlcv was called with "BTC/USDT"
        call_args = mock_exchange.fetch_ohlcv.call_args
        assert call_args[0][0] == "BTC/USDT"

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_retry_on_network_error(self, mock_ccxt: MagicMock) -> None:
        """First attempt fails with NetworkError, second succeeds."""
        fake_data = _make_fake_ohlcv(50)
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        # First call raises, second succeeds
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("connection reset"),
            fake_data,
        ]
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.NetworkError = ccxt.NetworkError
        mock_ccxt.RequestTimeout = ccxt.RequestTimeout
        mock_ccxt.ExchangeNotAvailable = ccxt.ExchangeNotAvailable
        mock_ccxt.BadSymbol = ccxt.BadSymbol

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert len(df) <= 50
        # fetch_ohlcv called twice (1 retry)
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_no_retry_on_bad_symbol(self, mock_ccxt: MagicMock) -> None:
        """BadSymbol raises immediately without retry."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_exchange.fetch_ohlcv.side_effect = ccxt.BadSymbol("INVALID/PAIR")
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.BadSymbol = ccxt.BadSymbol

        loader = CCXTLoader(exchange_id="binance")
        loader.connect()

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)

        with pytest.raises(CCXTDataError, match="Invalid symbol"):
            loader.download_ohlcv("BTCUSDT", "M15", start, end)

        # Only 1 call — no retry for BadSymbol
        assert mock_exchange.fetch_ohlcv.call_count == 1

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_no_duplicate_timestamps(self, mock_ccxt: MagicMock) -> None:
        """Verify deduplication after concat."""
        fake_data = _make_fake_ohlcv(100)
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert not df.index.duplicated().any()

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_dataframe_ohlc_consistency(self, mock_ccxt: MagicMock) -> None:
        """Verify high >= max(o,c) and low <= min(o,c)."""
        fake_data = _make_fake_ohlcv(50)
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_btc_gap_detection_warning(self, mock_ccxt: MagicMock) -> None:
        """Detect gaps > 5x expected interval in crypto data."""
        # Create data with a 10-minute gap at bar 5 (M15 expects 15min intervals,
        # 5x = 75min, so inject a gap of 2 hours = 7_200_000ms)
        fake_data = _make_fake_ohlcv_with_gap(
            n=20,
            gap_at=5,
            gap_duration_ms=7_200_000,  # 2 hours
            interval_ms=900_000,
        )
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        log_output = StringIO()
        sink_id = logger.add(log_output, level="WARNING", format="{message}")

        try:
            start = datetime(2023, 11, 1, tzinfo=UTC)
            end = datetime(2024, 1, 1, tzinfo=UTC)
            loader.download_ohlcv("BTCUSDT", "M15", start, end)

            log_text = log_output.getvalue()
            assert "GAP DETECTED" in log_text
        finally:
            logger.remove(sink_id)

    @patch("data.ingestion.ccxt_loader.ccxt")
    def test_index_is_utc_tz_aware(self, mock_ccxt: MagicMock) -> None:
        """Explicitly verify the index timezone is UTC."""
        fake_data = _make_fake_ohlcv(10)
        loader, _ = self._setup_connected_loader(mock_ccxt, fake_data)

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 12, 1, tzinfo=UTC)
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)

        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"
