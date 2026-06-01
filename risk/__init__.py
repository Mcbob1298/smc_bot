"""Risk management — the protective foundation every other layer consumes.

Public surface:
    - RiskManager / RiskDecision : veto core over all execution
    - DailyLossGuard             : per-day drawdown kill switch
    - ExposureTracker            : concurrent + per-killzone caps
    - position sizing helpers    : compute_lot, SymbolSpec, SizingResult
"""

from .daily_guard import DailyLossGuard
from .exposure import ExposureTracker
from .position_sizer import SizingResult, SymbolSpec, compute_lot
from .risk_manager import RiskDecision, RiskManager

__all__ = [
    "RiskManager",
    "RiskDecision",
    "DailyLossGuard",
    "ExposureTracker",
    "compute_lot",
    "SymbolSpec",
    "SizingResult",
]
