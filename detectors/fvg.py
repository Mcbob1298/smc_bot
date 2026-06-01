"""Fair Value Gap (FVG / 3-candle imbalance) detection (causal).

An FVG is a price gap left by a fast move, spanning three consecutive candles
(left = i-1, middle = i, right = i+1):

- **Bullish FVG** when ``low[i+1] > high[i-1]``: the move up was so quick the
  body of candle i skipped a price range. Zone = ``[high[i-1], low[i+1]]``.
- **Bearish FVG** when ``high[i+1] < low[i-1]``. Zone = ``[high[i+1], low[i-1]]``.

The gap only exists once the right candle prints, so it is indexed at the
middle bar ``i`` but ``confirmed_at = index[i+1]`` — strictly causal.

A minimum-size filter (``size >= min_size_atr_ratio * ATR[i]``) discards trivial
imbalances; ATR is causal so no lookahead is introduced. During ATR warm-up the
gap cannot be sized against volatility and is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy import FVGConfig

from ._common import validate_ohlc

FVG_COLUMNS = ["kind", "top", "bottom", "size", "mid", "confirmed_at"]


def detect_fvgs(
    df: pd.DataFrame,
    config: FVGConfig,
    atr: pd.Series,
) -> pd.DataFrame:
    """Detect bullish/bearish fair value gaps.

    Args:
        df: OHLC DataFrame with a DatetimeIndex.
        config: FVG parameters (minimum size as ATR ratio).
        atr: causal ATR series aligned to ``df`` (used for the size filter).

    Returns:
        DataFrame indexed by the middle-bar timestamp, columns:
        ``kind`` ("bullish"/"bearish"), ``top``, ``bottom``, ``size``,
        ``mid`` (the 50% level), ``confirmed_at``.
    """
    validate_ohlc(df, need=("high", "low"))
    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=FVG_COLUMNS)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    index = df.index
    atr_arr = atr.to_numpy(dtype="float64")
    min_ratio = config.min_size_atr_ratio

    kinds: list[str] = []
    times: list[pd.Timestamp] = []
    tops: list[float] = []
    bottoms: list[float] = []
    confirmed: list[pd.Timestamp] = []

    for i in range(1, n - 1):
        # Bullish: gap between left high and right low.
        if low[i + 1] > high[i - 1]:
            bottom, top = high[i - 1], low[i + 1]
            if _passes(top - bottom, min_ratio, atr_arr, i):
                kinds.append("bullish")
                times.append(index[i])
                tops.append(top)
                bottoms.append(bottom)
                confirmed.append(index[i + 1])
        # Bearish: gap between right high and left low.
        elif high[i + 1] < low[i - 1]:
            bottom, top = high[i + 1], low[i - 1]
            if _passes(top - bottom, min_ratio, atr_arr, i):
                kinds.append("bearish")
                times.append(index[i])
                tops.append(top)
                bottoms.append(bottom)
                confirmed.append(index[i + 1])

    top_arr = np.array(tops, dtype="float64")
    bot_arr = np.array(bottoms, dtype="float64")
    out = pd.DataFrame(
        {
            "kind": kinds,
            "top": top_arr,
            "bottom": bot_arr,
            "size": top_arr - bot_arr,
            "mid": (top_arr + bot_arr) / 2.0,
            "confirmed_at": confirmed,
        },
        index=pd.DatetimeIndex(times, name=df.index.name),
    )
    return out


def _passes(size: float, min_ratio: float, atr_arr: np.ndarray, i: int) -> bool:
    """Size filter: drop gaps below ``min_ratio * ATR`` (and during warm-up)."""
    if np.isnan(atr_arr[i]):
        return False
    return size >= min_ratio * atr_arr[i]
