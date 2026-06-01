"""SMC pattern detectors.

All detectors are strictly causal: every emitted event carries a
``confirmed_at`` timestamp — the bar at which it first becomes knowable — and
consumers must read through ``select_known`` so no future data leaks in.

Public surface:
    - detect_swings        : fractal swing highs/lows
    - detect_fvgs          : 3-candle fair value gaps
    - detect_structure     : BOS / ChoCh
    - detect_liquidity     : equal-highs / equal-lows pools
    - detect_sweeps        : liquidity sweeps (stop-hunts)
    - detect_order_blocks  : validated order blocks (composite)
    - select_known         : causality chokepoint for reading events
"""

from ._common import select_known
from .fvg import detect_fvgs
from .liquidity import detect_liquidity
from .order_blocks import detect_order_blocks
from .structure import detect_structure
from .sweeps import detect_sweeps
from .swings import detect_swings

__all__ = [
    "detect_swings",
    "detect_fvgs",
    "detect_structure",
    "detect_liquidity",
    "detect_sweeps",
    "detect_order_blocks",
    "select_known",
]
