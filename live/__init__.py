"""Live execution layer.

Public surface:
    - PaperExecutor : in-memory broker simulation gated by the RiskManager
    - PaperPosition / Fill / ExecutionResult : simulation state + outcomes
"""

from .paper_executor import ExecutionResult, Fill, PaperExecutor, PaperPosition

__all__ = ["PaperExecutor", "PaperPosition", "Fill", "ExecutionResult"]
