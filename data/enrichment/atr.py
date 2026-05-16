"""Causal ATR (Average True Range) computation with Wilder's smoothing.

Provides True Range and ATR calculations with strict causality guarantees.
ATR is used throughout the SMC strategy as a volatility normalizer for:
swings filtering, FVG minimum size, OB validation, SL buffers, etc.

Causality: ATR[t] uses only OHLC[0:t+1]. No future data leakage.
Implementation uses explicit Wilder's recursive formula for full control
over initialization and numerical behavior (no pandas.ewm ambiguity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

_REQUIRED_COLUMNS = ["high", "low", "close"]


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range for each bar.

    TR[t] = max(
        high[t] - low[t],
        abs(high[t] - close[t-1]),
        abs(low[t] - close[t-1])
    )

    For t=0, prev_close doesn't exist: TR[0] = high[0] - low[0].

    Args:
        df: DataFrame with columns 'high', 'low', 'close'.

    Returns:
        Series of True Range values, same index as df.

    Raises:
        ValueError: If required columns are missing.

    Causality: TR[t] uses only high[t], low[t], close[t-1]. Strictly causal.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index, name="true_range")

    _validate_columns(df)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")

    # Shift close by 1 (prev_close)
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    # Three components of TR
    hl = high - low
    hpc = np.abs(high - prev_close)
    lpc = np.abs(low - prev_close)

    tr = np.maximum(hl, np.maximum(hpc, lpc))

    # First bar: no prev_close, TR = high - low
    tr[0] = high[0] - low[0]

    return pd.Series(tr, index=df.index, name="true_range")


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing.

    Wilder's smoothing (equivalent to EMA with alpha=1/period):
        ATR[t] = (ATR[t-1] * (period - 1) + TR[t]) / period

    Initialization:
        - First `period - 1` values: NaN (insufficient data)
        - At t=period-1: ATR = simple mean of TR[0:period]
        - At t>=period: Wilder's recursive formula

    Args:
        df: DataFrame with columns 'high', 'low', 'close'.
        period: ATR lookback period (default 14).
        method: 'wilder' (standard) or 'sma' (simple moving average of TR).

    Returns:
        Series of ATR values, same index as df.
        First `period - 1` values are NaN.

    Raises:
        ValueError: If required columns missing, period < 1, or method invalid.

    Causality: Each ATR[t] uses only TR[0:t+1], which uses only OHLC[0:t+1].
    No future data leakage.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if method not in ("wilder", "sma"):
        raise ValueError(f"method must be 'wilder' or 'sma', got '{method}'")

    if df.empty:
        return pd.Series(dtype="float64", index=df.index, name=f"atr_{period}")

    _validate_columns(df)

    tr = compute_true_range(df)
    n = len(df)

    if method == "sma":
        atr = tr.rolling(period, min_periods=period).mean()
        atr.name = f"atr_{period}"
        _log_result(n, period, method, atr)
        return atr

    # Wilder's smoothing
    tr_arr = tr.to_numpy(dtype="float64")
    atr_arr = np.full(n, np.nan)

    if n >= period:
        # Initialization: SMA of first `period` TR values
        atr_arr[period - 1] = np.mean(tr_arr[:period])

        # Recursive Wilder's formula
        for i in range(period, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr_arr[i]) / period

    atr = pd.Series(atr_arr, index=df.index, name=f"atr_{period}")
    _log_result(n, period, method, atr)
    return atr


def enrich_atr(
    df: pd.DataFrame,
    period: int = 14,
    column_name: str | None = None,
) -> pd.DataFrame:
    """Add ATR column to DataFrame.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        period: ATR period (default 14).
        column_name: Custom column name (default: f"atr_{period}").

    Returns:
        Copy of df with new ATR column added.

    Raises:
        ValueError: If column already exists or required columns missing.
    """
    col = column_name if column_name is not None else f"atr_{period}"

    if col in df.columns:
        raise ValueError(
            f"DataFrame already contains column '{col}'. "
            "Remove it first or use a different column_name."
        )

    result = df.copy()
    result[col] = compute_atr(df, period=period, method="wilder")
    return result


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}. "
            f"Required: {_REQUIRED_COLUMNS}"
        )


def _log_result(n: int, period: int, method: str, atr: pd.Series) -> None:
    """Log computation summary at DEBUG level."""
    nan_count = int(atr.isna().sum())
    nan_pct = nan_count / n * 100 if n > 0 else 0.0
    logger.debug(
        "ATR computed: n={}, period={}, method={}, NaN={} ({:.1f}%)",
        n, period, method, nan_count, nan_pct,
    )
    if nan_pct > 5.0 and nan_count > period:
        logger.warning(
            "ATR has {:.1f}% NaN values ({} / {}). "
            "This suggests data quality issues beyond the initial period warm-up.",
            nan_pct, nan_count, n,
        )
