"""Strategy parameters for the SMC trading methodology (Kasper Trading).

All tunable parameters are centralized here. No magic numbers in logic code.
"""

from pydantic import BaseModel, Field


class SwingConfig(BaseModel):
    """Fractal swing detection parameters."""

    fractal_period: int = Field(
        default=3,
        description="Number of bars for fractal pattern (3 = classic 3-bar fractal)",
    )
    atr_filter_enabled: bool = Field(
        default=False,
        description="Filter out swings smaller than ATR threshold",
    )
    atr_filter_ratio: float = Field(
        default=0.3,
        description="Minimum swing size as ratio of ATR to be considered significant",
    )


class FVGConfig(BaseModel):
    """Fair Value Gap detection parameters."""

    min_size_atr_ratio: float = Field(
        default=0.2,
        description="Minimum FVG size as ratio of ATR(14) to be valid",
    )


class OrderBlockConfig(BaseModel):
    """Order Block detection and validity parameters."""

    max_age_bars: int = Field(
        default=50,
        description="Maximum age in bars before OB expires (per timeframe)",
    )
    require_fvg: bool = Field(
        default=True,
        description="OB must be followed by an FVG to be valid",
    )
    require_structure_break: bool = Field(
        default=True,
        description="OB must precede a BOS or ChoCh to be valid",
    )
    first_retest_only: bool = Field(
        default=True,
        description="Only trade the first retest (mitigation) of an OB",
    )


class LiquidityConfig(BaseModel):
    """Liquidity level detection parameters."""

    equal_level_tolerance_atr_ratio: float = Field(
        default=0.1,
        description="Tolerance for equal highs/lows as ratio of ATR",
    )
    min_touches: int = Field(
        default=2,
        description="Minimum number of swings at same level to form liquidity",
    )
    trendline_min_touches: int = Field(
        default=3,
        description="Minimum touches to form trendline liquidity",
    )


class KillzoneConfig(BaseModel):
    """Trading session killzones (times in Europe/Paris)."""

    london_start: str = Field(default="08:00", description="London KZ start")
    london_end: str = Field(default="11:00", description="London KZ end")
    ny_start: str = Field(default="13:30", description="New York KZ start")
    ny_end: str = Field(default="16:00", description="New York KZ end")


class RiskConfig(BaseModel):
    """Risk management parameters."""

    risk_per_trade_pct: float = Field(
        default=1.0,
        description="Percentage of account risked per trade",
    )
    rr_min: float = Field(
        default=1.0,
        description="Minimum Risk:Reward ratio to take a trade",
    )
    rr_target: float = Field(
        default=2.0,
        description="Target Risk:Reward ratio",
    )
    tp1_rr: float = Field(
        default=1.0,
        description="RR level for TP1 (partial close)",
    )
    tp1_partial_pct: float = Field(
        default=0.5,
        description="Fraction of position closed at TP1",
    )
    move_to_be_after_tp1: bool = Field(
        default=True,
        description="Move SL to breakeven after TP1 hit",
    )
    max_concurrent_trades: int = Field(
        default=2,
        description="Maximum number of simultaneous open trades",
    )


class StopLossConfig(BaseModel):
    """Stop loss buffer parameters."""

    buffer_xau_pips: float = Field(
        default=2.0,
        description="SL buffer in pips for XAUUSD (1 pip = 0.10$)",
    )
    buffer_btc_pct: float = Field(
        default=0.05,
        description="SL buffer as percentage of price for BTCUSDT",
    )


class NewsFilterConfig(BaseModel):
    """Economic news filter parameters."""

    enabled: bool = Field(default=True, description="Enable news filter")
    pre_window_minutes: int = Field(
        default=15,
        description="Minutes before news event to stop trading",
    )
    post_window_minutes: int = Field(
        default=30,
        description="Minutes after news event to resume trading",
    )
    impact_filter: list[str] = Field(
        default=["high"],
        description="News impact levels to filter (high = red on ForexFactory)",
    )


class EntryConfig(BaseModel):
    """LTF entry confirmation parameters."""

    require_choch_ltf: bool = Field(
        default=True,
        description="Require ChoCh on LTF before entry",
    )
    entry_mode: str = Field(
        default="ob_or_50pct",
        description="Entry mode: 'ob_or_50pct' (OB LTF or 50% retracement)",
    )


class ATRConfig(BaseModel):
    """ATR calculation parameters."""

    period: int = Field(default=14, description="ATR lookback period")


class StrategyConfig(BaseModel):
    """Complete strategy configuration aggregating all sub-configs."""

    swings: SwingConfig = Field(default_factory=SwingConfig)
    fvg: FVGConfig = Field(default_factory=FVGConfig)
    order_block: OrderBlockConfig = Field(default_factory=OrderBlockConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    killzones: KillzoneConfig = Field(default_factory=KillzoneConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    atr: ATRConfig = Field(default_factory=ATRConfig)
