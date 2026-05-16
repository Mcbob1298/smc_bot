"""Global application settings loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # MetaTrader 5 connection
    mt5_path: str = Field(
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="Path to MetaTrader 5 terminal executable",
    )
    mt5_login: int = Field(default=0, description="MT5 account login")
    mt5_password: str = Field(default="", description="MT5 account password")
    mt5_server: str = Field(default="", description="MT5 broker server name")

    # Symbols
    symbols: list[str] = Field(
        default=["XAUUSD", "BTCUSDT"],
        description="Trading symbols",
    )
    mt5_symbol_map: dict[str, str] = Field(
        default={"XAUUSD": "XAUUSD"},
        description=(
            "Mapping of canonical symbol names to broker-specific MT5 names. "
            "E.g. {'XAUUSD': 'GOLD'} if your broker uses 'GOLD' instead."
        ),
    )

    # Timeframes hierarchy
    htf_timeframes: list[str] = Field(
        default=["H4", "D1"],
        description="Higher timeframes for bias determination",
    )
    mtf_timeframes: list[str] = Field(
        default=["M15"],
        description="Mid timeframes for zone identification",
    )
    ltf_timeframes: dict[str, str] = Field(
        default={"XAUUSD": "M1", "BTCUSDT": "M5"},
        description="Lower timeframes for entry confirmation (per symbol)",
    )

    # Timezone
    timezone: str = Field(
        default="Europe/Paris",
        description="Reference timezone for killzones and sessions",
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data/parquet"),
        description="Directory for Parquet data storage",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    # Binance (optional, for BTCUSDT)
    binance_api_key: str = Field(default="", description="Binance API key (optional)")
    binance_api_secret: str = Field(default="", description="Binance API secret (optional)")

    # Logging
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    # Data history
    data_start_date: str = Field(
        default="2021-01-01",
        description="Start date for historical data download (ISO format)",
    )
    history_years: int = Field(
        default=5,
        description="Number of years of historical data to download",
    )
