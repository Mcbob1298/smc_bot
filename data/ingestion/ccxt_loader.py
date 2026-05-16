"""CCXT-based historical data loader for BTCUSDT (Binance).

Downloads OHLCV data from crypto exchanges via ccxt, with automatic pagination,
retry on network errors, and output format consistent with MT5Loader.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
from ccxt.base.errors import BadSymbol as _BadSymbol
from ccxt.base.errors import ExchangeNotAvailable as _ExchangeNotAvailable
from ccxt.base.errors import NetworkError as _NetworkError
from ccxt.base.errors import RequestTimeout as _RequestTimeout
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# WHY: Import exception classes directly so they don't get mocked in tests.
# Using direct imports from ccxt.base.errors avoids the issue where
# @patch("...ccxt") replaces ccxt.NetworkError with a MagicMock.
_RETRYABLE_ERRORS = (
    _NetworkError,
    _RequestTimeout,
    _ExchangeNotAvailable,
)

# WHY: Binance returns max 1000 candles per fetch_ohlcv call.
_MAX_BARS_PER_REQUEST = 1000

# Mapping of our canonical timeframe names to ccxt format
TIMEFRAME_MAP: dict[str, str] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}

# Milliseconds per timeframe (for gap detection and pagination)
TIMEFRAME_MS: dict[str, int] = {
    "M1": 60_000,
    "M5": 300_000,
    "M15": 900_000,
    "M30": 1_800_000,
    "H1": 3_600_000,
    "H4": 14_400_000,
    "D1": 86_400_000,
    "W1": 604_800_000,
}


class CCXTConnectionError(Exception):
    """Raised when exchange connection fails."""


class CCXTDataError(Exception):
    """Raised when data retrieval fails."""


class CCXTLoader:
    """Downloads OHLCV data from crypto exchanges via ccxt.

    Usage:
        loader = CCXTLoader(exchange_id="binance")
        loader.connect()
        df = loader.download_ohlcv("BTCUSDT", "M15", start, end)
        loader.disconnect()

    Or as a context manager:
        with CCXTLoader() as loader:
            df = loader.download_ohlcv(...)

    API-compatible with MT5Loader for downstream code consistency.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        symbol_map: dict[str, str] | None = None,
    ):
        """Initialize CCXT loader.

        Args:
            exchange_id: ccxt exchange ID (e.g. "binance", "bybit", "okx")
            api_key: Optional API key (higher rate limits if provided)
            api_secret: Optional API secret
            symbol_map: Mapping of canonical names to ccxt format
                        (e.g. {"BTCUSDT": "BTC/USDT"})
        """
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbol_map = symbol_map or {"BTCUSDT": "BTC/USDT"}
        self._exchange: ccxt.Exchange | None = None
        self._connected = False

    def __enter__(self) -> CCXTLoader:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Connect to the exchange (load markets).

        If API key is provided but invalid, logs a warning and continues
        without authentication (public endpoints still work for OHLCV).
        """
        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise CCXTConnectionError(
                f"Exchange '{self._exchange_id}' not found in ccxt. "
                f"Available: binance, bybit, okx, etc."
            )

        config: dict[str, object] = {
            "enableRateLimit": True,  # WHY: built-in ccxt throttling, no custom needed
        }
        if self._api_key:
            config["apiKey"] = self._api_key
            config["secret"] = self._api_secret

        self._exchange = exchange_class(config)

        logger.info("Connecting to {} exchange...", self._exchange_id)
        try:
            self._exchange.load_markets()
        except ccxt.AuthenticationError as e:
            # WHY: Invalid API key should not block public OHLCV access.
            # Log warning and retry without auth.
            logger.warning(
                "Authentication failed for {}: {}. "
                "Continuing without auth (public endpoints only).",
                self._exchange_id,
                e,
            )
            config.pop("apiKey", None)
            config.pop("secret", None)
            self._exchange = exchange_class(config)
            self._exchange.load_markets()
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            raise CCXTConnectionError(
                f"Failed to connect to {self._exchange_id}: {e}"
            ) from e

        self._connected = True
        n_markets = len(self._exchange.markets)
        logger.info(
            "Connected to {} — {} markets loaded", self._exchange_id, n_markets
        )

    def disconnect(self) -> None:
        """Close the exchange connection."""
        if self._exchange is not None and self._connected:
            self._exchange.close()
            self._connected = False
            logger.info("Disconnected from {}", self._exchange_id)

    def is_connected(self) -> bool:
        """Check if connected to exchange (passive flag, not a health check)."""
        return self._connected

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download OHLCV data for a symbol and timeframe.

        Args:
            symbol: Canonical symbol name (e.g. "BTCUSDT")
            timeframe: Timeframe string (M1, M5, M15, H1, H4, D1)
            start_date: Start of range (UTC)
            end_date: End of range (UTC)

        Returns:
            DataFrame with columns: open, high, low, close, volume,
            spread, real_volume. Index = timestamp_utc (UTC DatetimeIndex).

        Raises:
            CCXTConnectionError: If not connected
            CCXTDataError: If symbol/timeframe invalid or no data
        """
        if not self._connected or self._exchange is None:
            raise CCXTConnectionError(
                "Not connected to exchange. Call connect() first."
            )

        if timeframe not in TIMEFRAME_MAP:
            raise CCXTDataError(
                f"Unknown timeframe '{timeframe}'. "
                f"Available: {list(TIMEFRAME_MAP.keys())}"
            )

        # Resolve canonical → ccxt symbol format
        ccxt_symbol = self._resolve_symbol(symbol)
        ccxt_tf = TIMEFRAME_MAP[timeframe]
        tf_ms = TIMEFRAME_MS[timeframe]

        # Ensure dates are UTC-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        logger.info(
            "Downloading {} {} from {} to {}",
            symbol,
            timeframe,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Paginate forward in time
        all_ohlcv: list[list] = []
        since_ms = start_ms
        chunk_num = 0
        estimated_chunks = max(1, (end_ms - start_ms) // (tf_ms * _MAX_BARS_PER_REQUEST))

        while since_ms < end_ms:
            chunk_num += 1
            ohlcv = self._fetch_with_retry(ccxt_symbol, ccxt_tf, since_ms)

            if not ohlcv:
                if not all_ohlcv:
                    raise CCXTDataError(
                        f"No data returned for {ccxt_symbol} {ccxt_tf} "
                        f"starting from {datetime.fromtimestamp(since_ms / 1000, tz=UTC)}"
                    )
                # Exhausted available data
                logger.debug("No more data available after chunk {}", chunk_num)
                break

            all_ohlcv.extend(ohlcv)

            logger.debug(
                "Chunk {}/~{}: {} bars fetched",
                chunk_num,
                estimated_chunks,
                len(ohlcv),
            )

            # Move forward: last bar timestamp + 1ms to avoid overlap
            last_ts = ohlcv[-1][0]
            since_ms = last_ts + 1

            # If we got fewer bars than the limit, we've reached the end
            if len(ohlcv) < _MAX_BARS_PER_REQUEST:
                break

        if not all_ohlcv:
            raise CCXTDataError(
                f"No data available for {symbol} {timeframe} in requested range"
            )

        # Convert to DataFrame
        df = self._ohlcv_to_dataframe(all_ohlcv)

        # Filter to exact requested range and deduplicate
        df = df[~df.index.duplicated(keep="last")]
        df = df.loc[start_date:end_date]

        # WHY: Filter out the last bar if it's still forming (incomplete).
        # A bar is incomplete if its timestamp + timeframe_ms > now.
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        if not df.empty:
            last_bar_ms = int(df.index[-1].timestamp() * 1000)
            if last_bar_ms + tf_ms > now_ms:
                df = df.iloc[:-1]
                logger.debug("Removed last incomplete (still forming) bar")

        # Warn if partial data
        self._warn_if_partial(df, start_date, end_date, symbol, timeframe)

        # Check for suspicious gaps (crypto should be continuous)
        self._check_gaps(df, timeframe, symbol)

        logger.info(
            "Downloaded {} bars for {} {} [{} → {}]",
            len(df),
            symbol,
            timeframe,
            df.index[0] if not df.empty else "N/A",
            df.index[-1] if not df.empty else "N/A",
        )

        return df

    def _resolve_symbol(self, canonical: str) -> str:
        """Resolve canonical symbol to ccxt exchange format."""
        ccxt_symbol = self._symbol_map.get(canonical, canonical)
        if ccxt_symbol != canonical:
            logger.debug("Resolved {} → {} via ccxt_symbol_map", canonical, ccxt_symbol)

        # Verify symbol exists on exchange
        if self._exchange and ccxt_symbol not in self._exchange.markets:
            raise CCXTDataError(
                f"Symbol '{ccxt_symbol}' (mapped from '{canonical}') not found on "
                f"{self._exchange_id}. Available crypto pairs include: "
                f"{list(self._exchange.markets.keys())[:10]}..."
            )
        return ccxt_symbol

    def _fetch_with_retry(
        self, symbol: str, timeframe: str, since_ms: int
    ) -> list[list]:
        """Fetch OHLCV with retry on transient network errors.

        Uses tenacity for exponential backoff: 1s, 2s, 4s (3 attempts max).
        Non-retryable errors (BadSymbol, AuthenticationError) raise immediately.
        """
        assert self._exchange is not None

        @retry(
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            reraise=True,
        )
        def _do_fetch() -> list[list]:
            assert self._exchange is not None
            result: list[list] = self._exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since_ms,
                limit=_MAX_BARS_PER_REQUEST,
            )
            return result

        try:
            return _do_fetch()
        except _BadSymbol as e:
            raise CCXTDataError(f"Invalid symbol '{symbol}': {e}") from e
        except _RETRYABLE_ERRORS as e:
            raise CCXTConnectionError(
                f"Failed to fetch data after 3 retries: {e}"
            ) from e

    @staticmethod
    def _ohlcv_to_dataframe(ohlcv: list[list]) -> pd.DataFrame:
        """Convert ccxt OHLCV list to a DataFrame with UTC timestamps.

        ccxt returns: [[timestamp_ms, open, high, low, close, volume], ...]
        """
        df = pd.DataFrame(
            ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
        )

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp_utc").drop(columns=["timestamp_ms"])

        # WHY: Add spread and real_volume for API consistency with MT5Loader.
        # Downstream code expects these columns to exist regardless of source.
        df["spread"] = np.nan
        df["real_volume"] = df["volume"]

        # Reorder to match MT5Loader column order (volume where tick_volume would be)
        df = df[["open", "high", "low", "close", "volume", "spread", "real_volume"]]

        return df.sort_index()

    @staticmethod
    def _warn_if_partial(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Log a warning if received data doesn't cover the full requested range."""
        if df.empty:
            return

        actual_start = df.index[0].to_pydatetime()
        actual_end = df.index[-1].to_pydatetime()
        actual_span = actual_end - actual_start

        if actual_start - start_date > timedelta(days=7):
            logger.warning(
                "PARTIAL DATA: Requested {} {} from {} but only got data from {}. "
                "Exchange may not have history that far back. "
                "Got {} bars covering {:.0f} days.",
                symbol,
                timeframe,
                start_date.isoformat(),
                actual_start.isoformat(),
                len(df),
                actual_span.total_seconds() / 86400,
            )

        if end_date - actual_end > timedelta(days=1):
            logger.warning(
                "PARTIAL DATA: Requested {} {} until {} but last bar is {}. "
                "Missing recent data.",
                symbol,
                timeframe,
                end_date.isoformat(),
                actual_end.isoformat(),
            )

    @staticmethod
    def _check_gaps(df: pd.DataFrame, timeframe: str, symbol: str) -> None:
        """Warn if suspicious gaps exist in crypto data.

        WHY: Crypto trades 24/7, so gaps > 5 minutes are suspicious
        (usually Binance maintenance). We warn but don't crash.
        """
        if df.empty or len(df) < 2:
            return

        tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
        # Allow up to 5x the expected interval before flagging as a gap
        max_gap = timedelta(milliseconds=tf_ms * 5)

        idx = pd.DatetimeIndex(df.index)
        diffs = idx[1:] - idx[:-1]
        gaps = [(i, d) for i, d in enumerate(diffs) if d > max_gap]

        if gaps:
            logger.warning(
                "GAP DETECTED: {} {} has {} gaps exceeding {}. "
                "Largest gap: {} at index {}. "
                "Likely exchange maintenance.",
                symbol,
                timeframe,
                len(gaps),
                max_gap,
                max(d for _, d in gaps),
                gaps[0][0],
            )
