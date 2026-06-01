"""Liquidity sweep (stop-hunt) detection (causal).

A sweep is the classic stop-hunt: price spikes *beyond* a recent swing level to
grab the resting liquidity, then rejects and closes back on the origin side. It
is the footprint that often precedes a reversal and validates an order block.

- **Bullish sweep** (grabs sell-side below): a bar whose low pierces a prior
  swing **low** by at least ``min_penetration_atr_ratio * ATR`` and (optionally)
  closes back *above* that low.
- **Bearish sweep** (grabs buy-side above): mirror on a prior swing **high**.

Causality: the swept swing must already be confirmed and sit within
``lookback_bars`` of the sweeping bar; the sweep is detected on the bar itself,
so ``confirmed_at == index``. Each swing is swept at most once until a new swing
of that side forms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy import SweepConfig

from ._common import validate_ohlc

SWEEP_COLUMNS = ["kind", "swept_level", "swept_swing_time", "penetration", "confirmed_at"]


def detect_sweeps(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    config: SweepConfig,
    atr: pd.Series,
) -> pd.DataFrame:
    """Detect bullish/bearish liquidity sweeps of recent swing levels.

    Args:
        df: OHLC DataFrame with a DatetimeIndex.
        swings: output of ``detect_swings``.
        config: sweep parameters (penetration, close-back, lookback).
        atr: causal ATR series aligned to ``df`` (sets the penetration band).

    Returns:
        DataFrame indexed by the sweeping-bar timestamp, columns:
        ``kind`` ("bullish"/"bearish"), ``swept_level``, ``swept_swing_time``,
        ``penetration`` (actual, in price), ``confirmed_at`` (== index).
    """
    validate_ohlc(df, need=("high", "low", "close"))
    n = len(df)
    index = df.index
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    atr_arr = atr.to_numpy(dtype="float64")
    pos_of = {ts: i for i, ts in enumerate(index)}

    sw = swings.sort_values("confirmed_at", kind="stable") if not swings.empty else swings
    has_sw = not sw.empty
    sw_conf = list(sw["confirmed_at"]) if has_sw else []
    sw_kind = sw["kind"].tolist() if has_sw else []
    sw_price = sw["price"].tolist() if has_sw else []
    sw_time = list(sw.index) if has_sw else []

    last_high: float | None = None
    last_high_time = None
    high_swept = True
    last_low: float | None = None
    last_low_time = None
    low_swept = True

    ptr = 0
    rows: list[dict] = []

    for i in range(n):
        now = index[i]
        while ptr < len(sw_conf) and sw_conf[ptr] <= now:
            if sw_kind[ptr] == "high":
                last_high, last_high_time, high_swept = sw_price[ptr], sw_time[ptr], False
            else:
                last_low, last_low_time, low_swept = sw_price[ptr], sw_time[ptr], False
            ptr += 1

        band = atr_arr[i] * config.min_penetration_atr_ratio
        if np.isnan(band):
            continue

        # Bearish sweep of a swing high.
        if (
            last_high is not None
            and not high_swept
            and i - pos_of[last_high_time] <= config.lookback_bars
            and high[i] >= last_high + band
            and (not config.require_close_back or close[i] < last_high)
        ):
            rows.append(_row("bearish", last_high, last_high_time, high[i] - last_high, now))
            high_swept = True

        # Bullish sweep of a swing low.
        if (
            last_low is not None
            and not low_swept
            and i - pos_of[last_low_time] <= config.lookback_bars
            and low[i] <= last_low - band
            and (not config.require_close_back or close[i] > last_low)
        ):
            rows.append(_row("bullish", last_low, last_low_time, last_low - low[i], now))
            low_swept = True

    if not rows:
        return pd.DataFrame(columns=SWEEP_COLUMNS)
    out = pd.DataFrame(rows).set_index("_time")
    out.index.name = df.index.name
    return out[SWEEP_COLUMNS]


def _row(kind, level, swing_time, penetration, now) -> dict:
    return {
        "_time": now,
        "kind": kind,
        "swept_level": level,
        "swept_swing_time": swing_time,
        "penetration": penetration,
        "confirmed_at": now,
    }
