"""OHLCV data validation — detect anomalies before they pollute detectors/backtest.

Validates structural integrity, OHLC consistency, temporal correctness,
volume sanity, and statistical anomalies. Better to crash explicitly at
bootstrap than produce false stats for 6 months.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # data unusable, blocks
    WARNING = "warning"  # data suspicious but usable
    INFO = "info"  # observation, not a problem


@dataclass
class ValidationIssue:
    """A single validation issue found during checks."""

    severity: ValidationSeverity
    check_name: str
    message: str
    timestamps: list[datetime] | None = None
    stats: dict | None = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""

    symbol: str
    timeframe: str
    is_valid: bool
    nb_bars: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Human-readable summary of the validation report."""
        lines = [
            f"=== Validation Report: {self.symbol} {self.timeframe} ===",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"Bars: {self.nb_bars}",
            f"Range: {self.first_timestamp} → {self.last_timestamp}",
            f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}",
        ]
        for issue in self.issues:
            prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARN"
            if issue.severity == ValidationSeverity.INFO:
                prefix = "INFO"
            lines.append(f"  [{prefix}] {issue.check_name}: {issue.message}")
        if self.stats:
            lines.append("--- Stats ---")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize report to dict (JSON-compatible)."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "is_valid": self.is_valid,
            "nb_bars": self.nb_bars,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "issues": [
                {
                    "severity": i.severity.value,
                    "check_name": i.check_name,
                    "message": i.message,
                    "timestamps": (
                        [t.isoformat() for t in i.timestamps] if i.timestamps else None
                    ),
                    "stats": i.stats,
                }
                for i in self.issues
            ],
            "stats": self.stats,
        }


class DataValidationError(Exception):
    """Raised in strict mode when a critical check fails."""

    def __init__(self, issue: ValidationIssue):
        self.issue = issue
        super().__init__(f"{issue.check_name}: {issue.message}")


# Required columns for OHLCV data
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "spread", "real_volume"}

# Timeframe durations in seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}

# Price jump thresholds per timeframe (percent)
PRICE_JUMP_THRESHOLDS: dict[str, float] = {
    "M1": 2.0,
    "M5": 3.0,
    "M15": 5.0,
    "M30": 6.0,
    "H1": 8.0,
    "H4": 10.0,
    "D1": 15.0,
    "W1": 20.0,
}


class OHLCVValidator:
    """Validates OHLCV DataFrames for structural integrity and data quality.

    Usage:
        validator = OHLCVValidator()
        report = validator.validate(df, "XAUUSD", "M15")
        if not report.is_valid:
            print(report.summary())
    """

    def __init__(self) -> None:
        pass

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strict: bool = False,
    ) -> ValidationReport:
        """Run all validation checks on a DataFrame.

        Args:
            df: OHLCV DataFrame to validate.
            symbol: Symbol name (e.g. "XAUUSD").
            timeframe: Timeframe (e.g. "M15").
            strict: If True, raise DataValidationError on first ERROR.

        Returns:
            ValidationReport with all issues found.
        """
        issues: list[ValidationIssue] = []

        # Handle empty DataFrame
        if df.empty:
            report = ValidationReport(
                symbol=symbol,
                timeframe=timeframe,
                is_valid=True,
                nb_bars=0,
                first_timestamp=None,
                last_timestamp=None,
                issues=[],
                stats={},
            )
            return report

        # Run all checks in order
        checks = [
            # 1. Structural checks (ERROR)
            self._check_index_is_datetime,
            self._check_required_columns,
            self._check_dtypes,
            self._check_no_nan_in_ohlc,
            self._check_index_monotonic,
            self._check_no_duplicate_timestamps,
            # 2. OHLC consistency (ERROR)
            self._check_high_is_max,
            self._check_low_is_min,
            self._check_high_geq_low,
            self._check_prices_positive,
            # 3. Volume (WARNING)
            self._check_volume_non_negative,
            self._check_volume_not_all_zero,
            # 4. Temporal (WARNING/ERROR)
            self._check_timeframe_consistency,
            self._check_unexpected_gaps,
            self._check_xau_market_hours,
            self._check_no_future_timestamps,
            # 5. Price anomalies (WARNING)
            self._check_price_jumps,
            self._check_zero_range_bars,
            self._check_extreme_spread,
            self._check_price_outliers_zscore,
        ]

        for check_fn in checks:
            try:
                new_issues = check_fn(df, symbol, timeframe)
                issues.extend(new_issues)

                if strict:
                    for issue in new_issues:
                        if issue.severity == ValidationSeverity.ERROR:
                            raise DataValidationError(issue)
            except DataValidationError:
                raise
            except Exception as e:
                # Internal check error — log and continue
                logger.error(
                    "Validation check {} raised an internal error: {}",
                    check_fn.__name__,
                    e,
                )
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        check_name=check_fn.__name__,
                        message=f"Check raised internal error: {e}",
                    )
                )

        # Compute stats (may fail if data is severely corrupted)
        try:
            stats = self._compute_stats(df, symbol, timeframe)
        except Exception as e:
            logger.error("Stats computation failed: {}", e)
            stats = {}

        # Build report
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        first_ts = df.index[0].to_pydatetime() if not df.empty else None
        last_ts = df.index[-1].to_pydatetime() if not df.empty else None

        report = ValidationReport(
            symbol=symbol,
            timeframe=timeframe,
            is_valid=not has_errors,
            nb_bars=len(df),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            issues=issues,
            stats=stats,
        )

        if report.is_valid:
            logger.info(
                "Validation PASSED for {}/{}: {} bars, {} warnings",
                symbol,
                timeframe,
                len(df),
                len(report.warnings),
            )
        else:
            logger.error(
                "Validation FAILED for {}/{}: {} errors, {} warnings",
                symbol,
                timeframe,
                len(report.errors),
                len(report.warnings),
            )

        return report

    # ──────────────────────────────────────────────────────────────────────
    # 1. Structural checks
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_index_is_datetime(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if not isinstance(df.index, pd.DatetimeIndex):
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_index_is_datetime",
                    message=f"Index must be DatetimeIndex, got {type(df.index).__name__}",
                )
            ]
        if df.index.tz is None:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_index_is_datetime",
                    message="Index must be tz-aware UTC, got naive (tz=None)",
                )
            ]
        if str(df.index.tz) != "UTC":
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_index_is_datetime",
                    message=f"Index must be UTC, got tz={df.index.tz}",
                )
            ]
        return []

    @staticmethod
    def _check_required_columns(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_required_columns",
                    message=f"Missing required columns: {sorted(missing)}",
                )
            ]
        return []

    @staticmethod
    def _check_dtypes(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        price_cols = ["open", "high", "low", "close"]
        bad_cols = []
        for col in price_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                bad_cols.append(f"{col}={df[col].dtype}")
        if bad_cols:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_dtypes",
                    message=f"Price columns must be numeric (float64): {bad_cols}",
                )
            ]
        return []

    @staticmethod
    def _check_no_nan_in_ohlc(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        ohlc_cols = ["open", "high", "low", "close"]
        cols_present = [c for c in ohlc_cols if c in df.columns]
        nan_mask = df[cols_present].isna().any(axis=1)
        if nan_mask.any():
            nan_timestamps = df.index[nan_mask].tolist()
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_no_nan_in_ohlc",
                    message=f"Found {nan_mask.sum()} bars with NaN in OHLC columns",
                    timestamps=[t.to_pydatetime() for t in nan_timestamps[:10]],
                )
            ]
        return []

    @staticmethod
    def _check_index_monotonic(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if not df.index.is_monotonic_increasing:
            # Find where it breaks
            diffs = df.index[1:] - df.index[:-1]
            bad_idx = [i for i, d in enumerate(diffs) if d <= timedelta(0)]
            timestamps = [df.index[i + 1].to_pydatetime() for i in bad_idx[:10]]
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_index_monotonic",
                    message=(
                        f"Index not strictly monotonic increasing. "
                        f"{len(bad_idx)} violations found."
                    ),
                    timestamps=timestamps,
                )
            ]
        return []

    @staticmethod
    def _check_no_duplicate_timestamps(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        dupes = df.index.duplicated()
        if dupes.any():
            dupe_timestamps = df.index[dupes].tolist()
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_no_duplicate_timestamps",
                    message=f"Found {dupes.sum()} duplicate timestamps",
                    timestamps=[t.to_pydatetime() for t in dupe_timestamps[:10]],
                )
            ]
        return []

    # ──────────────────────────────────────────────────────────────────────
    # 2. OHLC consistency checks
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_high_is_max(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "high" not in df.columns or "open" not in df.columns or "close" not in df.columns:
            return []
        max_oc = df[["open", "close"]].max(axis=1)
        violations = df["high"] < max_oc
        if violations.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_high_is_max",
                    message=f"{violations.sum()} bars where high < max(open, close)",
                    timestamps=[t.to_pydatetime() for t in df.index[violations][:10]],
                )
            ]
        return []

    @staticmethod
    def _check_low_is_min(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "low" not in df.columns or "open" not in df.columns or "close" not in df.columns:
            return []
        min_oc = df[["open", "close"]].min(axis=1)
        violations = df["low"] > min_oc
        if violations.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_low_is_min",
                    message=f"{violations.sum()} bars where low > min(open, close)",
                    timestamps=[t.to_pydatetime() for t in df.index[violations][:10]],
                )
            ]
        return []

    @staticmethod
    def _check_high_geq_low(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "high" not in df.columns or "low" not in df.columns:
            return []
        violations = df["high"] < df["low"]
        if violations.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_high_geq_low",
                    message=f"{violations.sum()} bars where high < low",
                    timestamps=[t.to_pydatetime() for t in df.index[violations][:10]],
                )
            ]
        return []

    @staticmethod
    def _check_prices_positive(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        violations = (df[price_cols] <= 0).any(axis=1)
        if violations.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_prices_positive",
                    message=f"{violations.sum()} bars with non-positive prices",
                    timestamps=[t.to_pydatetime() for t in df.index[violations][:10]],
                )
            ]
        return []

    # ──────────────────────────────────────────────────────────────────────
    # 3. Volume checks
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_volume_non_negative(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "volume" not in df.columns:
            return []
        violations = df["volume"] < 0
        if violations.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_volume_non_negative",
                    message=f"{violations.sum()} bars with negative volume",
                    timestamps=[t.to_pydatetime() for t in df.index[violations][:10]],
                )
            ]
        return []

    @staticmethod
    def _check_volume_not_all_zero(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "volume" not in df.columns:
            return []
        zero_vol = df["volume"] == 0
        if not zero_vol.any():
            return []
        # Check for consecutive zero-volume sequences > 5% of total bars
        max_streak = 0
        current_streak = 0
        for is_zero in zero_vol:
            if is_zero:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        threshold = max(10, int(len(df) * 0.05))
        if max_streak > threshold:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_volume_not_all_zero",
                    message=(
                        f"Longest consecutive zero-volume streak: {max_streak} bars "
                        f"(threshold: {threshold})"
                    ),
                    stats={"max_zero_streak": max_streak, "threshold": threshold},
                )
            ]
        return []

    # ──────────────────────────────────────────────────────────────────────
    # 4. Temporal checks
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_timeframe_consistency(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if len(df) < 2:
            return []
        expected_seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
        expected_td = timedelta(seconds=expected_seconds)
        tolerance = timedelta(seconds=10)

        idx = pd.DatetimeIndex(df.index)
        diffs = idx[1:] - idx[:-1]

        irregular = []
        for i, d in enumerate(diffs):
            # Skip if it's a valid XAU weekend gap
            if _is_xau_weekend_gap(idx[i], idx[i + 1], symbol):
                continue
            if abs(d - expected_td) > tolerance:
                irregular.append(idx[i + 1].to_pydatetime())

        if irregular:
            pct = len(irregular) / len(df) * 100
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_timeframe_consistency",
                    message=(
                        f"{len(irregular)} intervals ({pct:.1f}%) don't match "
                        f"expected {timeframe} ({expected_seconds}s ±10s)"
                    ),
                    timestamps=irregular[:10],
                )
            ]
        return []

    @staticmethod
    def _check_unexpected_gaps(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if len(df) < 2:
            return []
        expected_seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
        expected_td = timedelta(seconds=expected_seconds)

        idx = pd.DatetimeIndex(df.index)
        diffs = idx[1:] - idx[:-1]

        gaps = []
        for i, d in enumerate(diffs):
            if d <= expected_td + timedelta(seconds=10):
                continue
            # Skip valid XAU weekend gaps
            if _is_xau_weekend_gap(idx[i], idx[i + 1], symbol):
                continue
            gaps.append(
                (idx[i + 1].to_pydatetime(), d.total_seconds() / expected_seconds)
            )

        if gaps:
            pct = len(gaps) / len(df) * 100
            severity = (
                ValidationSeverity.WARNING if pct <= 1 else ValidationSeverity.WARNING
            )
            return [
                ValidationIssue(
                    severity=severity,
                    check_name="check_unexpected_gaps",
                    message=(
                        f"{len(gaps)} unexpected gaps found ({pct:.2f}% of bars). "
                        f"Largest: {max(g[1] for g in gaps):.1f}x timeframe"
                    ),
                    timestamps=[g[0] for g in gaps[:10]],
                    stats={"nb_gaps": len(gaps), "pct_of_bars": pct},
                )
            ]
        return []

    @staticmethod
    def _check_xau_market_hours(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        # Only check for XAU-like symbols
        if "XAU" not in symbol.upper():
            return []

        idx = pd.DatetimeIndex(df.index)
        weekdays = idx.weekday
        hours = idx.hour

        # XAU closed: Saturday all day, Sunday before 22:00, Friday after 22:00
        closed_mask = (
            (weekdays == 5)  # Saturday
            | ((weekdays == 6) & (hours < 22))  # Sunday before 22:00
            | ((weekdays == 4) & (hours >= 22))  # Friday after 22:00
        )

        if closed_mask.any():
            bad_timestamps = idx[closed_mask].tolist()
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_xau_market_hours",
                    message=(
                        f"{closed_mask.sum()} bars during XAU market close "
                        f"(Sat, Sun<22h, Fri>=22h UTC)"
                    ),
                    timestamps=[t.to_pydatetime() for t in bad_timestamps[:10]],
                )
            ]
        return []

    @staticmethod
    def _check_no_future_timestamps(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        now = pd.Timestamp.now(tz=UTC)
        future_mask = pd.DatetimeIndex(df.index) > now
        if future_mask.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="check_no_future_timestamps",
                    message=f"{future_mask.sum()} bars with timestamps in the future",
                    timestamps=[
                        t.to_pydatetime()
                        for t in pd.DatetimeIndex(df.index)[future_mask][:10]
                    ],
                )
            ]
        return []

    # ──────────────────────────────────────────────────────────────────────
    # 5. Price anomaly checks
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_price_jumps(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "close" not in df.columns or len(df) < 2:
            return []
        threshold_pct = PRICE_JUMP_THRESHOLDS.get(timeframe, 5.0)

        pct_changes = df["close"].pct_change().abs() * 100
        # Exclude the first bar (NaN) and XAU Sunday open gaps
        idx = pd.DatetimeIndex(df.index)

        jumps = []
        for i in range(1, len(df)):
            if pct_changes.iloc[i] > threshold_pct:
                # Allow Sunday open gap (Sunday 22:00 UTC)
                if _is_xau_market_open(idx[i], symbol):
                    continue
                jumps.append(
                    (idx[i].to_pydatetime(), float(pct_changes.iloc[i]))
                )

        if jumps:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_price_jumps",
                    message=(
                        f"{len(jumps)} price jumps > {threshold_pct}% detected "
                        f"(max: {max(j[1] for j in jumps):.2f}%)"
                    ),
                    timestamps=[j[0] for j in jumps[:10]],
                    stats={"threshold_pct": threshold_pct, "nb_jumps": len(jumps)},
                )
            ]
        return []

    @staticmethod
    def _check_zero_range_bars(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "high" not in df.columns or "low" not in df.columns:
            return []
        zero_range = df["high"] == df["low"]
        pct = zero_range.sum() / len(df) * 100
        if pct > 1.0:
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_zero_range_bars",
                    message=(
                        f"{zero_range.sum()} bars ({pct:.1f}%) with high==low (zero range)"
                    ),
                    timestamps=[
                        t.to_pydatetime() for t in df.index[zero_range][:10]
                    ],
                    stats={"pct_zero_range": pct},
                )
            ]
        return []

    @staticmethod
    def _check_extreme_spread(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "spread" not in df.columns:
            return []
        spread = df["spread"].dropna()
        if spread.empty or (spread == 0).all():
            return []
        median_spread = spread.median()
        if median_spread <= 0:
            return []
        extreme_mask = spread > 5 * median_spread
        if extreme_mask.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_extreme_spread",
                    message=(
                        f"{extreme_mask.sum()} bars with spread > 5× median "
                        f"(median={median_spread:.4f})"
                    ),
                    timestamps=[
                        t.to_pydatetime()
                        for t in spread.index[extreme_mask][:10]
                    ],
                    stats={"median_spread": float(median_spread)},
                )
            ]
        return []

    @staticmethod
    def _check_price_outliers_zscore(
        df: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[ValidationIssue]:
        if "close" not in df.columns or len(df) < 30:
            return []

        # Log returns for stationarity
        ratio = df["close"] / df["close"].shift(1)
        log_returns = pd.Series(np.log(ratio), index=ratio.index).dropna()
        if log_returns.empty:
            return []

        # Robust z-score using MAD (Median Absolute Deviation)
        median_ret = log_returns.median()
        mad = (log_returns - median_ret).abs().median()

        if mad == 0:
            # All returns identical (or near-identical) — no outliers detectable
            return []

        robust_z = (log_returns - median_ret) / (1.4826 * mad)
        outlier_mask = robust_z.abs() > 6

        if outlier_mask.any():
            return [
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="check_price_outliers_zscore",
                    message=(
                        f"{outlier_mask.sum()} price outliers detected "
                        f"(robust z-score > 6 sigma)"
                    ),
                    timestamps=[
                        t.to_pydatetime()
                        for t in log_returns.index[outlier_mask][:10]
                    ],
                    stats={"nb_outliers": int(outlier_mask.sum()), "mad": float(mad)},
                )
            ]
        return []

    # ──────────────────────────────────────────────────────────────────────
    # 6. Global stats computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stats(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Compute global statistics for the dataset."""
        if df.empty:
            return {}

        first_ts = df.index[0].to_pydatetime()
        last_ts = df.index[-1].to_pydatetime()
        span = last_ts - first_ts
        span_days = span.total_seconds() / 86400

        # Price stats
        close = df["close"]
        returns = close.pct_change().dropna()
        avg_range = ((df["high"] - df["low"]) / df["close"]).mean() if "high" in df.columns else 0

        # Bullish/bearish ratio
        bullish = (close > df["open"]).sum() if "open" in df.columns else 0
        bearish = (close < df["open"]).sum() if "open" in df.columns else 0

        stats = {
            "nb_bars": len(df),
            "span_days": round(span_days, 1),
            "avg_volatility_pct": round(float(returns.std() * 100), 4) if not returns.empty else 0,
            "avg_range_pct": round(float(avg_range * 100), 4),
            "avg_volume": round(float(df["volume"].mean()), 2) if "volume" in df.columns else 0,
            "pct_bullish": round(bullish / len(df) * 100, 1) if len(df) > 0 else 0,
            "pct_bearish": round(bearish / len(df) * 100, 1) if len(df) > 0 else 0,
        }
        return stats


def _is_xau_weekend_gap(ts_before: pd.Timestamp, ts_after: pd.Timestamp, symbol: str) -> bool:
    """Check if a gap between two timestamps is a valid XAU weekend gap.

    Valid weekend gap: Friday after 21:50 UTC → Sunday/Monday around 22:00+ UTC.
    We use a generous window to account for broker variations.
    """
    if "XAU" not in symbol.upper():
        return False

    # The bar before the gap should be Friday (weekday=4) around 21:45-22:00
    if ts_before.weekday() == 4 and ts_before.hour >= 21:
        # The bar after should be Sunday 22:xx or Monday 00:xx
        if (ts_after.weekday() == 6 and ts_after.hour >= 22) or ts_after.weekday() == 0:
            return True

    return False


def _is_xau_market_open(ts: pd.Timestamp, symbol: str) -> bool:
    """Check if a timestamp is the XAU Sunday open (gap acceptable)."""
    if "XAU" not in symbol.upper():
        return False
    # Sunday 22:00-23:00 UTC = market open, gaps expected
    return ts.weekday() == 6 and ts.hour >= 22
