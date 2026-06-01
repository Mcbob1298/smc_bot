"""Trade journaling layer.

Public surface:
    - TradeLogger : append-only JSONL logger of trade lifecycle + reasons
"""

from .trade_logger import TradeLogger

__all__ = ["TradeLogger"]
