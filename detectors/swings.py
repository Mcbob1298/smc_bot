"""Fractal swing high/low detection (causal).

A swing high is a bar whose high strictly exceeds the highs of the ``side``
bars on its left *and* the ``side`` bars on its right; a swing low mirrors it
on lows. With ``fractal_period = 3`` this is the classic 3-bar fractal
(one bar each side); ``fractal_period = 5`` uses two bars each side, etc.

Causality
---------
Because a swing needs ``side`` bars to its right to be validated, the centre
bar's swing is only *confirmed* ``side`` bars later. The event is indexed at
its centre bar but carries ``confirmed_at = index[centre + side]``; a live
system could not have known about it any earlier.

ATR significance filter
-----------------------
Micro-pivots in noise are filtered out. A swing's ``magnitude`` is the vertical
span of its window — for a high, ``high[c] - min(low over window)``; for a low,
``max(high over window) - low[c]``. When the filter is enabled the swing is
kept only if ``magnitude >= ratio * ATR[centre]``. ATR is causal, so this adds
no lookahead. During ATR warm-up (NaN) a filtered swing cannot be validated and
is dropped — refusing rather than guessing significance. Separate ratios apply
to LTF (lower, to stay reactive) vs HTF/MTF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy import SwingConfig

from ._common import validate_ohlc

SWING_COLUMNS = ["kind", "price", "magnitude", "confirmed_at"]


def detect_swings(
    df: pd.DataFrame,
    config: SwingConfig,
    atr: pd.Series | None = None,
    *,
    is_ltf: bool = False,
) -> pd.DataFrame:
    """Detect fractal swing highs and lows.

    Args:
        df: OHLC DataFrame with a DatetimeIndex.
        config: swing parameters (fractal width, ATR filter ratios).
        atr: causal ATR series aligned to ``df`` (required iff the relevant ATR
            filter is enabled).
        is_ltf: select the LTF filter toggle/ratio instead of the HTF/MTF ones.

    Returns:
        DataFrame indexed by the swing's centre-bar timestamp, columns:
        ``kind`` ("high"/"low"), ``price``, ``magnitude``, ``confirmed_at``.
        Sorted by index; may contain both a high and a low at the same bar.
    """
    validate_ohlc(df, need=("high", "low"))
    side = (config.fractal_period - 1) // 2
    if side < 1:
        raise ValueError(f"fractal_period must be >= 3 (odd), got {config.fractal_period}")

    use_filter = config.atr_filter_enabled_ltf if is_ltf else config.atr_filter_enabled
    ratio = config.atr_filter_ratio_ltf if is_ltf else config.atr_filter_ratio
    if use_filter and atr is None:
        raise ValueError("ATR series is required when the swing ATR filter is enabled")

    n = len(df)
    if n < 2 * side + 1:
        return pd.DataFrame(columns=SWING_COLUMNS)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    index = df.index
    atr_arr = atr.to_numpy(dtype="float64") if atr is not None else None

    kinds: list[str] = []
    times: list[pd.Timestamp] = []
    prices: list[float] = []
    mags: list[float] = []
    confirmed: list[pd.Timestamp] = []

    for c in range(side, n - side):
        lo_w, hi_w = c - side, c + side
        left_high = high[lo_w:c].max()
        right_high = high[c + 1 : hi_w + 1].max()
        left_low = low[lo_w:c].min()
        right_low = low[c + 1 : hi_w + 1].min()

        if high[c] > left_high and high[c] > right_high:
            mag = high[c] - low[lo_w : hi_w + 1].min()
            if _passes(use_filter, mag, ratio, atr_arr, c):
                kinds.append("high")
                times.append(index[c])
                prices.append(high[c])
                mags.append(mag)
                confirmed.append(index[c + side])

        if low[c] < left_low and low[c] < right_low:
            mag = high[lo_w : hi_w + 1].max() - low[c]
            if _passes(use_filter, mag, ratio, atr_arr, c):
                kinds.append("low")
                times.append(index[c])
                prices.append(low[c])
                mags.append(mag)
                confirmed.append(index[c + side])

    out = pd.DataFrame(
        {"kind": kinds, "price": prices, "magnitude": mags, "confirmed_at": confirmed},
        index=pd.DatetimeIndex(times, name=df.index.name),
    )
    return out.sort_index(kind="stable")


def _passes(
    use_filter: bool,
    magnitude: float,
    ratio: float,
    atr_arr: np.ndarray | None,
    c: int,
) -> bool:
    """Apply the ATR significance filter; drop swings that can't be validated."""
    if not use_filter:
        return True
    if atr_arr is None or np.isnan(atr_arr[c]):
        return False  # can't establish significance during warm-up → refuse
    return magnitude >= ratio * atr_arr[c]
