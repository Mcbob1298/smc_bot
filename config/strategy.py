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
        default=True,
        description="Filter out swings smaller than ATR threshold (HTF/MTF)",
    )
    atr_filter_ratio: float = Field(
        default=0.3,
        description="Minimum swing size as ratio of ATR for HTF/MTF significance",
    )
    atr_filter_enabled_ltf: bool = Field(
        default=True,
        description="Filter out swings smaller than ATR threshold (LTF)",
    )
    atr_filter_ratio_ltf: float = Field(
        default=0.15,
        description="Minimum swing size as ratio of ATR for LTF (lower than HTF to stay reactive)",
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
    fvg_association_window_bars: int = Field(
        default=50,
        description="Max bars after OB to look for an associated FVG",
    )
    require_structure_break: bool = Field(
        default=True,
        description="OB must precede a BOS or ChoCh to be valid",
    )
    require_prior_liquidity_sweep: bool = Field(
        default=True,
        description="OB must have swept liquidity beforehand (toggle for A/B test)",
    )
    prior_liquidity_lookback_bars: int = Field(
        default=20,
        description="How many bars back to look for a liquidity sweep before the OB",
    )
    first_retest_only: bool = Field(
        default=True,
        description="Only trade the first retest (mitigation) of an OB",
    )
    wick_touch_counts_as_mitigation: bool = Field(
        default=False,
        description=(
            "If False, only a candle closing inside OB zone counts as mitigation "
            "(not a wick-only touch)"
        ),
    )
    body_or_full_range: str = Field(
        default="full_range",
        description=(
            "OB zone definition: 'full_range' (high-low), "
            "'body_only' (open-close), or 'body_plus_half_wick'"
        ),
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


class BiasConfig(BaseModel):
    """HTF bias determination parameters."""

    daily_alignment_mode: str = Field(
        default="optional_filter",
        description=(
            "How Daily TF interacts with H4 bias: "
            "'ignored' (H4 only), "
            "'optional_filter' (no trade if Daily contradicts H4), "
            "'required' (must align to trade)"
        ),
    )


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
    tp2_target_mode: str = Field(
        default="next_htf_liquidity",
        description=(
            "TP2 targeting mode: "
            "'next_htf_liquidity' (next opposing liquidity on HTF), "
            "'fixed_rr' (fixed RR multiple from entry), "
            "'structure' (next HTF structure level)"
        ),
    )
    max_concurrent_trades: int = Field(
        default=2,
        description="Maximum number of simultaneous open trades",
    )
    max_trades_per_killzone: int = Field(
        default=1,
        description="Max trades allowed per killzone session (anti-revenge)",
    )
    xau_friday_close_time: str = Field(
        default="22:30",
        description="Time (Europe/Paris) to force-close XAU positions on Friday",
    )
    xau_friday_no_new_trade_after: str = Field(
        default="18:00",
        description="Time (Europe/Paris) after which no new XAU trade on Friday",
    )


class StopLossConfig(BaseModel):
    """Stop loss buffer parameters."""

    buffer_xau_usd: float = Field(
        default=0.20,
        description=(
            "SL buffer for XAUUSD in absolute USD "
            "(e.g. 0.20 = 20 cents beyond OB edge)"
        ),
    )


class CostsConfig(BaseModel):
    """Trading costs for realistic backtesting.

    WHY separate from RiskConfig: costs affect PnL calculation, not trade decisions.
    Slippage is often more impactful than spread on SMC strategies that enter
    during fast moves (structure breaks, sweeps).
    """

    # Spread (in pips)
    spread_xau_london_pips: float = Field(
        default=2.5,
        description="XAU spread during London KZ (in pips, 1 pip = 0.01$)",
    )
    spread_xau_ny_pips: float = Field(
        default=2.0,
        description="XAU spread during NY KZ (in pips)",
    )
    spread_xau_off_session_pips: float = Field(
        default=4.0,
        description="XAU spread outside killzones (in pips)",
    )
    # Commission
    commission_xau_per_lot: float = Field(
        default=0.0,
        description="XAU commission per lot (most CFD brokers = 0, spread only)",
    )

    # Slippage (as multiplier of ATR(M1))
    slippage_kz_atr_multiplier: float = Field(
        default=0.5,
        description="Slippage during killzones as fraction of ATR(M1)",
    )
    slippage_off_kz_atr_multiplier: float = Field(
        default=1.0,
        description="Slippage outside killzones as fraction of ATR(M1)",
    )
    slippage_apply_to_exit: bool = Field(
        default=True,
        description="Apply slippage to exits too (not just entries)",
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
        description=(
            "Entry mode: 'ob_or_50pct' (OB LTF or 50% retracement of micro-impulse)"
        ),
    )
    retracement_pct: float = Field(
        default=0.5,
        description="Retracement level of the micro-impulse after LTF ChoCh (0.5 = 50%)",
    )
    ltf_timeframe: str = Field(
        default="M1",
        description="LTF timeframe for XAUUSD entry confirmation",
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
    bias: BiasConfig = Field(default_factory=BiasConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    atr: ATRConfig = Field(default_factory=ATRConfig)
