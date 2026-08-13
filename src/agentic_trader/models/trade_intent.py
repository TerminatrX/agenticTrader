"""Types describing what the system wants to do, and what risk allows it to do.

The flow is deliberately one-directional:

    Signal  ->  TradeIntent  ->  RiskDecision  ->  (execution)

A Signal is a strategy's opinion. A TradeIntent is a concrete proposed order.
A RiskDecision is the risk engine's ruling on that order, and it is the only
object execution will act on. Nothing downstream may reconstruct an intent that
risk did not approve.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SignalStrength(StrEnum):
    NONE = "none"
    WATCH = "watch"       # Setup forming; conditions not all met.
    ENTER = "enter"       # Full entry conditions met.
    EXIT = "exit"         # Exit conditions met on an open position.


class Decision(StrEnum):
    APPROVED = "approved"
    RESIZED = "resized"   # Approved, but at a smaller notional than requested.
    REJECTED = "rejected"


class Signal(BaseModel):
    """A strategy's read on one symbol.

    `reasons` and `failed_conditions` are not decoration — the critic agent and
    the performance review read them directly, so they must be specific enough
    to reconstruct the logic months later.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    strategy: str
    strength: SignalStrength
    side: Side | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    reference_price: Decimal | None = None
    stop_price: Decimal | None = None
    target_price: Decimal | None = None

    reasons: list[str] = Field(default_factory=list)
    failed_conditions: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.strength in (SignalStrength.ENTER, SignalStrength.EXIT)

    @property
    def stop_distance_pct(self) -> Decimal | None:
        if self.stop_price is None or not self.reference_price:
            return None
        return (self.reference_price - self.stop_price) / self.reference_price


class Position(BaseModel):
    """An open position, as reconciled from the broker rather than assumed."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: Decimal
    average_cost: Decimal
    market_value: Decimal | None = None

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.average_cost


class AccountState(BaseModel):
    """Broker-sourced account truth at decision time.

    `unsettled_funds` matters more than it looks. On a cash account, proceeds
    from a sale are unavailable until settlement, and buying with them before
    then produces a good-faith violation. Buying power from the broker already
    excludes unsettled cash, but the risk engine tracks it separately so it can
    explain *why* an order was blocked instead of silently under-sizing.
    """

    model_config = ConfigDict(frozen=True)

    account_number: str
    is_cash_account: bool = True

    total_value: Decimal
    cash: Decimal
    buying_power: Decimal
    unsettled_funds: Decimal = Decimal("0")

    positions: list[Position] = Field(default_factory=list)
    open_order_symbols: list[str] = Field(default_factory=list)

    realized_pnl_today: Decimal = Decimal("0")

    def position_in(self, symbol: str) -> Position | None:
        return next((p for p in self.positions if p.symbol == symbol), None)

    @property
    def open_position_count(self) -> int:
        return len([p for p in self.positions if p.quantity > 0])


class TradeIntent(BaseModel):
    """A concrete proposed order, before risk review.

    `client_key` is the idempotency guard. An agentic loop can re-fire for many
    reasons — a retry, a crashed cycle, a duplicated schedule tick — and without
    a stable key derived from (symbol, side, strategy, session) a re-fire places
    a second order. Execution must refuse to submit a key already recorded.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    strategy: str
    client_key: str

    notional: Decimal = Field(gt=0, description="Order size in dollars, not shares.")
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    target_price: Decimal | None = None

    reference_price: Decimal
    confidence: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    created_at: datetime | None = None

    @property
    def estimated_quantity(self) -> Decimal:
        """Fractional share count. Robinhood supports fractional equity orders,
        which is the only way a small account can take a position in a name
        trading in the hundreds of dollars."""
        price = self.limit_price or self.reference_price
        return self.notional / price


class RiskDecision(BaseModel):
    """The risk engine's ruling. Execution acts on this and nothing else."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    intent: TradeIntent | None = None
    approved_notional: Decimal | None = None

    breached_limits: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def is_executable(self) -> bool:
        return self.decision in (Decision.APPROVED, Decision.RESIZED)
