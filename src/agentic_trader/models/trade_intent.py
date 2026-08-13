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

    # Human-readable intent. `reasons` records which conditions fired;
    # these two record what the trade actually believes and what would
    # prove it wrong. The critic and any later post-mortem read these —
    # without them a review is reconstruction rather than recall.
    thesis: str | None = None
    invalidation_reason: str | None = None

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

    @property
    def risk_reward_ratio(self) -> Decimal | None:
        """Reward per unit of risk, from entry to target against entry to stop.

        None when either level is missing, which callers must treat as
        "unknown" and refuse to trade on rather than as "acceptable".
        """
        if self.stop_price is None or self.target_price is None or not self.reference_price:
            return None
        risk = self.reference_price - self.stop_price
        if risk <= 0:
            return None
        return (self.target_price - self.reference_price) / risk


class Position(BaseModel):
    """An open position, as reconciled from the broker rather than assumed."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: Decimal
    average_cost: Decimal
    market_value: Decimal | None = None

    # Populated from fundamentals when available. `None` means unknown, which
    # the sector-exposure gate reports rather than silently treating as zero
    # exposure — an unknown sector under-counts concentration.
    sector: str | None = None

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.average_cost

    @property
    def exposure(self) -> Decimal:
        """Current value, falling back to cost when the mark is unavailable."""
        return self.market_value if self.market_value is not None else self.cost_basis


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

    def sector_exposure(self, sector: str) -> Decimal:
        """Total value held in one sector. Case-insensitive match."""
        target = sector.strip().casefold()
        return sum(
            (p.exposure for p in self.positions if (p.sector or "").strip().casefold() == target),
            Decimal("0"),
        )

    def positions_missing_sector(self) -> list[str]:
        """Held symbols whose sector is unknown.

        Surfaced by the exposure gate: each one is concentration the gate
        cannot see, so the check is weaker than it appears.
        """
        return [p.symbol for p in self.positions if p.quantity > 0 and not p.sector]


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

    # Carried through from the signal. See `Signal.thesis`.
    thesis: str | None = None
    invalidation_reason: str | None = None
    sector: str | None = None

    @property
    def estimated_quantity(self) -> Decimal:
        """Fractional share count. Robinhood supports fractional equity orders,
        which is the only way a small account can take a position in a name
        trading in the hundreds of dollars."""
        price = self.limit_price or self.reference_price
        return self.notional / price

    @property
    def estimated_max_loss(self) -> Decimal | None:
        """Modeled loss if the stop is hit — **not** a floor on what can be lost.

        Three things can make the real loss larger, and all three are live here:

        - The stop is *managed*, not broker-native. This order does not carry
          it, so between cycles there is nothing enforcing the level at all.
        - Entries are market orders (fractional orders cannot be limit orders),
          so the fill can be worse than the reference price.
        - An overnight gap can open straight through the level.

        Treat this as the intended risk of the setup, never as a guarantee.
        """
        if self.stop_price is None or not self.reference_price:
            return None
        distance = (self.reference_price - self.stop_price) / self.reference_price
        if distance <= 0:
            return None
        return (self.notional * distance).quantize(Decimal("0.01"))

    @property
    def risk_reward_ratio(self) -> Decimal | None:
        """Reward per unit of risk. None when either level is missing."""
        if self.stop_price is None or self.target_price is None or not self.reference_price:
            return None
        risk = self.reference_price - self.stop_price
        if risk <= 0:
            return None
        return ((self.target_price - self.reference_price) / risk).quantize(Decimal("0.01"))


class RiskDecision(BaseModel):
    """The risk engine's ruling. Execution acts on this and nothing else."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    intent: TradeIntent | None = None
    approved_notional: Decimal | None = None

    breached_limits: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    # Set when losses breached the sticky kill-switch threshold. The engine
    # only reports it — writing the HALT file is done by the caller, so this
    # module stays free of side effects. Unlike the daily-loss limit, which
    # resets tomorrow, tripping this requires a human to clear HALT.
    trip_kill_switch: bool = False

    @property
    def is_executable(self) -> bool:
        return self.decision in (Decision.APPROVED, Decision.RESIZED)
