"""MetaTrader 5 historical data loader for XAUUSD.

Connects to a local MT5 terminal, downloads OHLCV data by chunks,
converts timestamps to UTC, and returns clean DataFrames.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5  # noqa: N813
except ImportError:
    mt5 = None  # type: ignore[assignment]

# WHY: MT5 returns at most ~100k bars per copy_rates_range call.
# We paginate backwards from end_date to start_date in chunks.
_MAX_BARS_PER_REQUEST = 99_000

# Mapping of string timeframe names to MT5 constants
TIMEFRAME_MAP: dict[str, int] = {}

# WHY: We defer populating TIMEFRAME_MAP until MT5 is actually imported
# because mt5.TIMEFRAME_* constants only exist after import.
if mt5 is not None:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


class MT5ConnectionError(Exception):
    """Raised when MT5 terminal connection fails."""


class MT5DataError(Exception):
    """Raised when MT5 data retrieval fails."""


class MT5Loader:
    """Downloads OHLCV data from a local MetaTrader 5 terminal.

    Usage:
        loader = MT5Loader()
        loader.connect()
        df = loader.download_ohlcv("XAUUSD", "M15", start, end)
        loader.disconnect()

    Or as a context manager:
        with MT5Loader() as loader:
            df = loader.download_ohlcv(...)
    """

    def __init__(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        symbol_map: dict[str, str] | None = None,
        crypto_symbols: list[str] | None = None,
    ):
        """Initialize with optional MT5 credentials.

        Args:
            path: Path to MT5 terminal executable
            login: MT5 account login number
            password: MT5 account password
            server: MT5 broker server name
            symbol_map: Mapping of canonical names to broker names
                        (e.g. {"XAUUSD": "GOLD"})
            crypto_symbols: Symbols to skip weekend sanity check for
                           (default: ["BTCUSDT", "ETHUSDT"])
        """
        self._path = path
        self._login = login
        self._password = password
        self._server = server
        self._symbol_map = symbol_map or {}
        self._crypto_symbols = crypto_symbols or ["BTCUSDT", "ETHUSDT"]
        self._connected = False

    def __enter__(self) -> MT5Loader:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Connect to the MT5 terminal."""
        if mt5 is None:
            raise MT5ConnectionError(
                "MetaTrader5 package not available. "
                "Install it on Windows with: pip install MetaTrader5"
            )

        # Build init kwargs, only passing non-None values
        kwargs: dict[str, str | int] = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login:
            kwargs["login"] = self._login
        if self._password:
            kwargs["password"] = self._password
        if self._server:
            kwargs["server"] = self._server

        logger.info("Connecting to MT5 terminal...")
        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            raise MT5ConnectionError(
                f"MT5 initialization failed: {error}. "
                "Ensure MetaTrader 5 is running."
            )

        self._connected = True
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(
                "Connected to MT5: {} (build {})",
                terminal_info.name,
                terminal_info.build,
            )

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if mt5 is not None and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def is_connected(self) -> bool:
        """Check if currently connected to MT5.

        WARNING: This is a passive flag check, not a live health check.
        If MT5 terminal crashes after connect(), this still returns True.
        TODO V2: implement live health check via mt5.terminal_info() with caching.
        """
        return self._connected

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download OHLCV data from MT5 for a given symbol and timeframe.

        Args:
            symbol: MT5 symbol name (e.g. "XAUUSD", "GOLD", "XAUUSD.m")
            timeframe: Timeframe string (M1, M5, M15, H1, H4, D1, etc.)
            start_date: Start of the data range (UTC)
            end_date: End of the data range (UTC)

        Returns:
            DataFrame with columns: open, high, low, close,
            tick_volume, spread, real_volume. Index = timestamp_utc (UTC).

        Raises:
            MT5ConnectionError: If not connected to MT5
            MT5DataError: If symbol is unavailable or data retrieval fails
        """
        if not self._connected:
            raise MT5ConnectionError("Not connected to MT5. Call connect() first.")

        if timeframe not in TIMEFRAME_MAP:
            raise MT5DataError(
                f"Unknown timeframe '{timeframe}'. "
                f"Available: {list(TIMEFRAME_MAP.keys())}"
            )

        mt5_tf = TIMEFRAME_MAP[timeframe]

        # Resolve broker-specific symbol name via symbol_map
        broker_symbol = self._symbol_map.get(symbol, symbol)
        if broker_symbol != symbol:
            logger.info("Resolved {} → {} via mt5_symbol_map", symbol, broker_symbol)

        # Verify symbol exists
        symbol_info = mt5.symbol_info(broker_symbol)
        if symbol_info is None:
            raise MT5DataError(
                f"Symbol '{broker_symbol}' (mapped from '{symbol}') not found. "
                "Check broker naming (XAUUSD, GOLD, XAUUSDm, etc.)"
            )
        if not symbol_info.visible:
            mt5.symbol_select(broker_symbol, True)

        # Ensure dates are UTC-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        logger.info(
            "Downloading {} {} from {} to {}",
            symbol,
            timeframe,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # WHY: MT5 copy_rates_range can return at most ~100k bars.
        # For large ranges (e.g. 5 years M1 = ~1.8M bars), we paginate.
        all_chunks: list[pd.DataFrame] = []
        chunk_end = end_date
        total_bars = 0

        while chunk_end > start_date:
            rates = mt5.copy_rates_range(broker_symbol, mt5_tf, start_date, chunk_end)

            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                if total_bars == 0:
                    raise MT5DataError(
                        f"No data returned for {symbol} {timeframe} "
                        f"[{start_date} → {chunk_end}]. MT5 error: {error}"
                    )
                # We've exhausted available data
                logger.debug(
                    "No more data for {} {} before {}",
                    symbol,
                    timeframe,
                    chunk_end,
                )
                break

            chunk_df = self._rates_to_dataframe(rates)
            chunk_bars = len(chunk_df)
            total_bars += chunk_bars
            all_chunks.append(chunk_df)

            logger.debug(
                "Chunk: {} bars, range {} → {}",
                chunk_bars,
                chunk_df.index[0],
                chunk_df.index[-1],
            )

            # If we got fewer bars than the limit, we've reached the start
            if chunk_bars < _MAX_BARS_PER_REQUEST:
                break

            # Move chunk_end back to just before the earliest bar we received
            # WHY: subtract 1 second to avoid overlapping the boundary bar
            chunk_end = chunk_df.index[0].to_pydatetime() - pd.Timedelta(seconds=1)

        if not all_chunks:
            raise MT5DataError(
                f"No data available for {symbol} {timeframe} in the requested range"
            )

        # Concatenate, sort, and deduplicate
        df = pd.concat(all_chunks).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Filter to exact requested range
        df = df.loc[start_date:end_date]

        # Fix 1: Warn if received data doesn't cover the requested range
        self._warn_if_partial(df, start_date, end_date, symbol, timeframe)

        # Fix 3: Validate timezone sanity (no Sunday bars for forex/metals)
        self._validate_timezone_sanity(df, symbol)

        logger.info(
            "Downloaded {} bars for {} {} [{} → {}]",
            len(df),
            symbol,
            timeframe,
            df.index[0],
            df.index[-1],
        )

        return df

    @staticmethod
    def _rates_to_dataframe(rates: np.ndarray) -> pd.DataFrame:
        """Convert MT5 rates array to a clean DataFrame with UTC timestamps.

        MT5 Python package returns timestamps as Unix epoch seconds already in UTC.
        """
        df = pd.DataFrame(rates)

        # WHY: MT5 returns 'time' as int64 Unix timestamp (seconds since epoch, UTC).
        df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("timestamp_utc")

        # Select and order columns
        df = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

        return df

    @staticmethod
    def _warn_if_partial(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Log a warning if the data doesn't cover the full requested range."""
        if df.empty:
            return

        actual_start = df.index[0].to_pydatetime()
        actual_end = df.index[-1].to_pydatetime()
        actual_span = actual_end - actual_start

        # Check start gap (> 7 days missing at the beginning)
        if actual_start - start_date > timedelta(days=7):
            logger.warning(
                "PARTIAL DATA: Requested {} {} from {} but only got data from {}. "
                "Broker likely doesn't store deeper history for this timeframe. "
                "Got {} bars covering {:.0f} days.",
                symbol,
                timeframe,
                start_date.isoformat(),
                actual_start.isoformat(),
                len(df),
                actual_span.total_seconds() / 86400,
            )

        # Check end gap (> 1 day missing at the end)
        if end_date - actual_end > timedelta(days=1):
            logger.warning(
                "PARTIAL DATA: Requested {} {} until {} but last bar is {}. "
                "Missing recent data (possible connection issue or market closed).",
                symbol,
                timeframe,
                end_date.isoformat(),
                actual_end.isoformat(),
            )

    def _validate_timezone_sanity(self, df: pd.DataFrame, symbol: str) -> None:
        """Verify no Sunday bars exist for forex/metals (market closed).

        If Sunday bars are found between 00:00-21:00 UTC, it indicates the
        timestamps are in broker time (UTC+2/+3) instead of UTC.

        Raises:
            MT5DataError: If Sunday bars detected (timezone leak)
        """
        if self._is_crypto(symbol):
            return  # Crypto trades 24/7, skip check

        if df.empty:
            return

        # Sunday = weekday 6 in pandas (Monday=0)
        idx = pd.DatetimeIndex(df.index)
        sunday_mask = idx.weekday == 6
        if not sunday_mask.any():
            return

        # XAU/Forex market opens Sunday ~21:00-22:00 UTC.
        # Bars before 21:00 UTC on Sunday are impossible.
        sunday_bars = df[sunday_mask]
        sunday_idx = pd.DatetimeIndex(sunday_bars.index)
        early_sunday = sunday_bars[sunday_idx.hour < 21]

        if not early_sunday.empty:
            raise MT5DataError(
                f"Timezone sanity check failed for {symbol}: found "
                f"{len(early_sunday)} bars on Sunday before 21:00 UTC. "
                f"First offending bar: {early_sunday.index[0]}. "
                "This suggests timestamps are in broker time (UTC+2/+3) "
                "instead of UTC. Check MT5 terminal timezone settings."
            )

    def _is_crypto(self, symbol: str) -> bool:
        """Check if a symbol is a crypto pair (skip weekend checks)."""
        return symbol.upper() in [s.upper() for s in self._crypto_symbols]

    def get_available_symbols(self, pattern: str = "*XAU*") -> list[str]:
        """List available symbols matching a pattern (useful to find XAU naming)."""
        if not self._connected:
            raise MT5ConnectionError("Not connected to MT5. Call connect() first.")

        symbols = mt5.symbols_get(pattern)
        if symbols is None:
            return []
        return [s.name for s in symbols]
