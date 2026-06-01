"""Strategy layer — SMC signal assembly.

Public surface (so far):
    - Signal / Direction : the trade-idea contract consumed by risk/execution
"""

from .signal import Direction, Signal

__all__ = ["Signal", "Direction"]
