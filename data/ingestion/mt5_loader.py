"""MetaTrader 5 historical data loader for XAUUSD.

Connects to a local MT5 terminal, downloads OHLCV data by chunks,
converts timestamps to UTC, and returns clean DataFrames.
"""

from datetime import UTC, datetime

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
    ):
        """Initialize with optional MT5 credentials.

        If not provided, reads from config/settings.
        """
        self._path = path
        self._login = login
        self._password = password
        self._server = server
        self._connected = False

    def __enter__(self) -> "MT5Loader":
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
        """Check if currently connected to MT5."""
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

        # Verify symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise MT5DataError(
                f"Symbol '{symbol}' not found. "
                "Check broker naming (XAUUSD, GOLD, XAUUSDm, etc.)"
            )
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

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
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, chunk_end)

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

    def get_available_symbols(self, pattern: str = "*XAU*") -> list[str]:
        """List available symbols matching a pattern (useful to find XAU naming)."""
        if not self._connected:
            raise MT5ConnectionError("Not connected to MT5. Call connect() first.")

        symbols = mt5.symbols_get(pattern)
        if symbols is None:
            return []
        return [s.name for s in symbols]
