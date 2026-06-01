"""Order Block detection with validity criteria (causal, composite).

An order block (OB) is the last opposing candle before an impulsive move that
breaks structure — the footprint of institutional orders. This detector is a
*composition*: it consumes structure breaks, FVGs and sweeps (themselves causal)
and applies the validity toggles in ``OrderBlockConfig``:

- **last opposing candle**: for a bullish break, the most recent *bearish*
  candle within the impulse leg preceding the break (mirror for bearish).
- ``require_fvg``: an FVG of the same direction must appear within
  ``fvg_association_window_bars`` after the OB candle.
- ``require_structure_break``: satisfied by construction (we iterate breaks);
  the toggle is honoured for completeness.
- ``require_prior_liquidity_sweep``: a same-direction sweep must have occurred
  within ``prior_liquidity_lookback_bars`` before the OB candle.
- ``body_or_full_range``: how the OB zone is drawn (full range / body /
  body + half wick).

Causality: an OB only becomes valid when its break confirms, so
``confirmed_at = break.confirmed_at`` and the OB candle precedes it. Mitigation
/ first-retest logic is deliberately left to the strategy layer (it depends on
how the zone is *used*), so it is not computed here.
"""

from __future__ import annotations

import pandas as pd

from config.strategy import OrderBlockConfig

from ._common import validate_ohlc

OB_COLUMNS = [
    "kind",  # "bullish" | "bearish"
    "top",
    "bottom",
    "break_time",
    "has_fvg",
    "has_prior_sweep",
    "expires_at",
    "confirmed_at",
]


def detect_order_blocks(
    df: pd.DataFrame,
    structure: pd.DataFrame,
    fvgs: pd.DataFrame,
    sweeps: pd.DataFrame,
    config: OrderBlockConfig,
) -> pd.DataFrame:
    """Detect valid order blocks from structure breaks + FVGs + sweeps.

    Args:
        df: OHLC DataFrame with a DatetimeIndex.
        structure: output of ``detect_structure``.
        fvgs: output of ``detect_fvgs`` (may be empty if ``require_fvg`` False).
        sweeps: output of ``detect_sweeps`` (may be empty unless required).
        config: order-block validity parameters.

    Returns:
        DataFrame indexed by the OB candle timestamp, columns per ``OB_COLUMNS``.
    """
    validate_ohlc(df, need=("open", "high", "low", "close"))
    if structure.empty:
        return pd.DataFrame(columns=OB_COLUMNS)

    index = df.index
    n = len(df)
    pos_of = {ts: i for i, ts in enumerate(index)}
    open_ = df["open"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")

    rows: list[dict] = []
    for break_time, br in structure.iterrows():
        b = pos_of[break_time]
        direction = br["direction"]  # "bullish"/"bearish"
        ob_pos = _find_ob_candle(direction, b, open_, close, config.fvg_association_window_bars)
        if ob_pos is None:
            continue
        ob_time = index[ob_pos]

        has_fvg = _has_assoc_fvg(fvgs, direction, ob_time, index, ob_pos, config)
        if config.require_fvg and not has_fvg:
            continue
        has_sweep = _has_prior_sweep(sweeps, direction, ob_time, index, ob_pos, config)
        if config.require_prior_liquidity_sweep and not has_sweep:
            continue

        top, bottom = _zone(config.body_or_full_range, ob_pos, open_, high, low, close)
        expires_pos = min(ob_pos + config.max_age_bars, n - 1)
        rows.append(
            {
                "_time": ob_time,
                "kind": direction,
                "top": top,
                "bottom": bottom,
                "break_time": break_time,
                "has_fvg": has_fvg,
                "has_prior_sweep": has_sweep,
                "expires_at": index[expires_pos],
                "confirmed_at": br["confirmed_at"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=OB_COLUMNS)
    out = pd.DataFrame(rows).set_index("_time").sort_index(kind="stable")
    out.index.name = df.index.name
    # A break leg can surface the same OB candle twice; keep the first (earliest
    # break that validated it).
    out = out[~out.index.duplicated(keep="first")]
    return out[OB_COLUMNS]


def _find_ob_candle(direction, b, open_, close, window) -> int | None:
    """Most recent opposing candle in [b-window, b-1] before the break."""
    lo = max(0, b - window)
    for j in range(b - 1, lo - 1, -1):
        is_bear = close[j] < open_[j]
        is_bull = close[j] > open_[j]
        if direction == "bullish" and is_bear:
            return j
        if direction == "bearish" and is_bull:
            return j
    return None


def _has_assoc_fvg(fvgs, direction, ob_time, index, ob_pos, config) -> bool:
    if fvgs.empty:
        return False
    want = "bullish" if direction == "bullish" else "bearish"
    hi_pos = min(ob_pos + config.fvg_association_window_bars, len(index) - 1)
    hi_time = index[hi_pos]
    sub = fvgs[(fvgs["kind"] == want) & (fvgs.index >= ob_time) & (fvgs.index <= hi_time)]
    return not sub.empty


def _has_prior_sweep(sweeps, direction, ob_time, index, ob_pos, config) -> bool:
    if sweeps.empty:
        return False
    lo_pos = max(0, ob_pos - config.prior_liquidity_lookback_bars)
    lo_time = index[lo_pos]
    sub = sweeps[
        (sweeps["kind"] == direction)
        & (sweeps.index >= lo_time)
        & (sweeps.index <= ob_time)
    ]
    return not sub.empty


def _zone(mode, j, open_, high, low, close) -> tuple[float, float]:
    """Return (top, bottom) of the OB zone per the configured definition."""
    body_top = max(open_[j], close[j])
    body_bottom = min(open_[j], close[j])
    if mode == "full_range":
        return high[j], low[j]
    if mode == "body_only":
        return body_top, body_bottom
    if mode == "body_plus_half_wick":
        return (body_top + (high[j] - body_top) / 2.0, body_bottom - (body_bottom - low[j]) / 2.0)
    raise ValueError(f"unknown body_or_full_range mode: {mode!r}")
