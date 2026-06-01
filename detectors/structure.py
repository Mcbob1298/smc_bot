"""Break of Structure (BOS) and Change of Character (ChoCh) detection (causal).

Market structure is read from the sequence of confirmed swing pivots. The
detector walks the bars forward, tracking the most recent *unbroken* swing high
and swing low, and the prevailing ``trend``:

- Price breaking the tracked swing **high** is a *bullish* break; breaking the
  tracked swing **low** is a *bearish* break.
- The break is a **BOS** (continuation) when it agrees with the current trend
  (or sets the first trend), and a **ChoCh** (change of character) when it goes
  *against* it — the first counter-trend break that flips the regime.

Causality
---------
A break is detected on the breaking bar itself using that bar's price, and the
swing being broken must already be confirmed (``swing.confirmed_at <= break
bar``). Swings are ingested exactly when they become known, so no future
information is used; ``confirmed_at`` of each event equals the breaking bar.

Modelling choices (all configurable in ``StructureConfig``)
-----------------------------------------------------------
- ``break_on_close``: a break needs a candle *close* beyond the level (robust)
  rather than a wick poke (reactive).
- ``break_buffer_atr_ratio``: optional ATR-scaled cushion beyond the level to
  ignore marginal pokes.
A given swing produces at most one break; after it breaks, a *new* swing of the
same side must form before another same-side break can fire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy import StructureConfig

from ._common import validate_ohlc

STRUCTURE_COLUMNS = [
    "kind",  # "BOS" | "ChoCh"
    "direction",  # "bullish" | "bearish"
    "broken_level",
    "broken_swing_time",
    "confirmed_at",
]


def detect_structure(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    config: StructureConfig,
    atr: pd.Series | None = None,
) -> pd.DataFrame:
    """Detect BOS / ChoCh events from confirmed swings.

    Args:
        df: OHLC DataFrame with a DatetimeIndex.
        swings: output of ``detect_swings`` (needs ``kind``, ``price``,
            ``confirmed_at``).
        config: structure parameters (break definition + buffer).
        atr: causal ATR series, required iff ``break_buffer_atr_ratio > 0``.

    Returns:
        DataFrame indexed by the breaking-bar timestamp, columns:
        ``kind``, ``direction``, ``broken_level``, ``broken_swing_time``,
        ``confirmed_at`` (== index).
    """
    validate_ohlc(df, need=("high", "low", "close"))
    use_buffer = config.break_buffer_atr_ratio > 0
    if use_buffer and atr is None:
        raise ValueError("ATR series is required when break_buffer_atr_ratio > 0")

    n = len(df)
    index = df.index
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    atr_arr = atr.to_numpy(dtype="float64") if atr is not None else None

    # Swings ordered by the moment they become known, for a forward walk.
    sw = swings.sort_values("confirmed_at", kind="stable") if not swings.empty else swings
    has_sw = not sw.empty
    # Keep timestamps as (tz-aware) pandas objects to compare cleanly with the
    # bar index; numpy datetime64 would drop the timezone.
    sw_conf = list(sw["confirmed_at"]) if has_sw else []
    sw_kind = sw["kind"].tolist() if has_sw else []
    sw_price = sw["price"].to_numpy(dtype="float64") if has_sw else np.array([])
    sw_time = list(sw.index) if has_sw else []

    trend: str | None = None
    last_high: float | None = None
    last_high_time = None
    high_broken = True  # nothing to break until a swing high confirms
    last_low: float | None = None
    last_low_time = None
    low_broken = True

    ptr = 0
    out_rows: list[dict] = []

    for i in range(n):
        now = index[i]
        # Ingest every swing confirmed by this bar (clears the broken flag).
        while ptr < len(sw_conf) and sw_conf[ptr] <= now:
            if sw_kind[ptr] == "high":
                last_high, last_high_time, high_broken = sw_price[ptr], sw_time[ptr], False
            else:
                last_low, last_low_time, low_broken = sw_price[ptr], sw_time[ptr], False
            ptr += 1

        buffer = atr_arr[i] * config.break_buffer_atr_ratio if use_buffer else 0.0
        if use_buffer and np.isnan(buffer):
            continue  # cannot size the buffer yet → no break this bar

        up_price = close[i] if config.break_on_close else high[i]
        dn_price = close[i] if config.break_on_close else low[i]

        if last_high is not None and not high_broken and up_price > last_high + buffer:
            kind = "BOS" if trend in (None, "bull") else "ChoCh"
            out_rows.append(_row(kind, "bullish", last_high, last_high_time, now))
            trend, high_broken = "bull", True
        elif last_low is not None and not low_broken and dn_price < last_low - buffer:
            kind = "BOS" if trend in (None, "bear") else "ChoCh"
            out_rows.append(_row(kind, "bearish", last_low, last_low_time, now))
            trend, low_broken = "bear", True

    if not out_rows:
        return pd.DataFrame(columns=STRUCTURE_COLUMNS)
    out = pd.DataFrame(out_rows).set_index("_time")
    out.index.name = df.index.name
    return out[STRUCTURE_COLUMNS]


def _row(kind, direction, level, swing_time, now) -> dict:
    return {
        "_time": now,
        "kind": kind,
        "direction": direction,
        "broken_level": level,
        "broken_swing_time": swing_time,
        "confirmed_at": now,
    }
