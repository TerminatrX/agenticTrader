"""Order construction and simulated fills.

Note what is absent: nothing here places an order. Python builds the payload;
the orchestrating agent submits it through the MCP tool. See `executor` for why.
"""

from agentic_trader.execution.executor import (
    ExecutionPlan,
    OrderPayload,
    PreflightError,
    build_order_payload,
)
from agentic_trader.execution.shadow_executor import ShadowExecutor, ShadowFill

__all__ = [
    "ExecutionPlan",
    "OrderPayload",
    "PreflightError",
    "ShadowExecutor",
    "ShadowFill",
    "build_order_payload",
]
