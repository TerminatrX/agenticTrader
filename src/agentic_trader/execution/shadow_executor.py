"""Simulated execution: the default mode until the system has earned live capital.

Shadow mode runs the entire pipeline — snapshot, strategy, risk, payload
construction — and then, instead of submitting, records what would have
happened. The value is that the recorded decisions accumulate into a real track
record built by the same code path that would have traded, so switching to live
changes one flag rather than one architecture.

Fills are modelled pessimistically. An optimistic simulator produces a
flattering track record and a nasty surprise, so slippage is always charged
against the position: buys fill above the reference, sells below it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal

from agentic_trader.execution.executor import ExecutionPlan
from agentic_trader.models import Side

CENTS = Decimal("0.01")
SHARE_DP = Decimal("0.000001")


@dataclass(frozen=True)
class ShadowFill:
    """A simulated fill, shaped to match what a real fill would record."""

    client_key: str
    symbol: str
    side: Side
    quantity: Decimal
    fill_price: Decimal
    notional: Decimal
    slippage_pct: Decimal
    submitted_at: datetime
    strategy: str
    managed_stop: Decimal | None = None
    managed_target: Decimal | None = None

    def to_row(self) -> dict[str, object]:
        return {
            "client_key": self.client_key,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "fill_price": str(self.fill_price),
            "notional": str(self.notional),
            "slippage_pct": str(self.slippage_pct),
            "submitted_at": self.submitted_at.isoformat(),
            "strategy": self.strategy,
            "managed_stop": str(self.managed_stop) if self.managed_stop is not None else None,
            "managed_target": str(self.managed_target) if self.managed_target is not None else None,
        }


class ShadowExecutor:
    """Simulates submission of an `ExecutionPlan`.

    `slippage_pct` defaults to 10 basis points, which is a plausible cost for a
    market order in a liquid large-cap. Widen it for anything thinner — and
    note that a shadow record is only as honest as this number.
    """

    def __init__(self, slippage_pct: Decimal = Decimal("0.001")) -> None:
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        self.slippage_pct = slippage_pct

    def submit(self, plan: ExecutionPlan, *, now: datetime | None = None) -> ShadowFill:
        intent = plan.intent
        reference = intent.reference_price

        # Slippage always works against the trade, in both directions.
        direction = Decimal("1") if intent.side is Side.BUY else Decimal("-1")
        fill_price = (reference * (Decimal("1") + direction * self.slippage_pct)).quantize(CENTS)

        if intent.side is Side.BUY:
            # A dollar order buys fewer shares once slippage lifts the price.
            quantity = (intent.notional / fill_price).quantize(SHARE_DP, rounding=ROUND_DOWN)
            notional = intent.notional
        else:
            quantity = intent.estimated_quantity.quantize(SHARE_DP, rounding=ROUND_DOWN)
            notional = (quantity * fill_price).quantize(CENTS)

        return ShadowFill(
            client_key=intent.client_key,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            fill_price=fill_price,
            notional=notional,
            slippage_pct=self.slippage_pct,
            submitted_at=now or datetime.now(UTC),
            strategy=intent.strategy,
            managed_stop=plan.managed_stop,
            managed_target=plan.managed_target,
        )
