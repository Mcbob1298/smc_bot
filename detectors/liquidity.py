"""Liquidity pool detection — equal highs / equal lows (causal).

Resting liquidity forms where price prints several swing pivots at roughly the
same level: equal **highs** are buy-side liquidity (stops of shorts sit above),
equal **lows** are sell-side liquidity (stops of longs sit below). These are the
levels Smart Money tends to sweep before reversing.

Detection clusters confirmed swing pivots whose prices fall within
``equal_level_tolerance_atr_ratio * ATR`` of one another. A pool is emitted once
a cluster reaches ``min_touches`` pivots; it becomes known — ``confirmed_at`` —
at the confirmation of that last pivot, never before.

Trendline (diagonal) liquidity is deferred to V2; only horizontal pools are
detected here.
"""

from __future__ import annotations

import pandas as pd

from config.strategy import LiquidityConfig

LIQUIDITY_COLUMNS = ["kind", "level", "touches", "first_touch_time", "confirmed_at"]


def detect_liquidity(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    config: LiquidityConfig,
    atr: pd.Series,
) -> pd.DataFrame:
    """Detect equal-highs / equal-lows liquidity pools.

    Args:
        df: OHLC DataFrame (used only for its index / ATR alignment).
        swings: output of ``detect_swings``.
        config: liquidity parameters (tolerance, min touches).
        atr: causal ATR series aligned to ``df`` (sets the equality tolerance).

    Returns:
        DataFrame indexed by the last-touch swing timestamp, columns:
        ``kind`` ("equal_highs"/"equal_lows"), ``level`` (the extreme of the
        touches), ``touches`` (count at emission), ``first_touch_time``,
        ``confirmed_at``.
    """
    if swings.empty:
        return pd.DataFrame(columns=LIQUIDITY_COLUMNS)

    rows: list[dict] = []
    rows += _pools_for_side(swings[swings["kind"] == "high"], "equal_highs", config, atr)
    rows += _pools_for_side(swings[swings["kind"] == "low"], "equal_lows", config, atr)
    if not rows:
        return pd.DataFrame(columns=LIQUIDITY_COLUMNS)

    out = pd.DataFrame(rows).set_index("_time").sort_index(kind="stable")
    out.index.name = df.index.name
    return out[LIQUIDITY_COLUMNS]


def _pools_for_side(
    side_swings: pd.DataFrame,
    kind: str,
    config: LiquidityConfig,
    atr: pd.Series,
) -> list[dict]:
    """Greedy causal clustering of one side's swings into equal-level pools."""
    if side_swings.empty:
        return []

    is_high = kind == "equal_highs"
    # Confirmation order = the order a live system learns the pivots.
    ordered = side_swings.sort_values("confirmed_at", kind="stable")

    # Each open cluster: list of (time, price), running mean, emitted flag.
    clusters: list[dict] = []
    rows: list[dict] = []

    for ts, row in ordered.iterrows():
        price = float(row["price"])
        try:
            atr_val = float(atr.loc[ts])
        except KeyError:
            continue
        if pd.isna(atr_val):
            continue  # no tolerance band yet → skip (no guessing)
        tol = config.equal_level_tolerance_atr_ratio * atr_val

        joined = None
        for cl in clusters:
            if abs(price - cl["mean"]) <= tol:
                joined = cl
                break
        if joined is None:
            clusters.append(
                {"times": [ts], "prices": [price], "mean": price, "emitted": False}
            )
            continue

        joined["times"].append(ts)
        joined["prices"].append(price)
        joined["mean"] = sum(joined["prices"]) / len(joined["prices"])
        if not joined["emitted"] and len(joined["prices"]) >= config.min_touches:
            level = max(joined["prices"]) if is_high else min(joined["prices"])
            rows.append(
                {
                    "_time": ts,
                    "kind": kind,
                    "level": level,
                    "touches": len(joined["prices"]),
                    "first_touch_time": joined["times"][0],
                    "confirmed_at": row["confirmed_at"],
                }
            )
            joined["emitted"] = True

    return rows
