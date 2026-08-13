"""Records the journal persists.

Two distinct streams, deliberately not merged:

*Audit* captures every cycle, including the overwhelming majority that decide
to do nothing. Rejections are the more valuable half of the record — a system
that only logs its trades cannot tell you whether its filters are working or
whether it simply never sees a setup.

*Trades* capture positions actually taken and how they resolved.

Both stay append-only. A journal that can be revised is a journal that will
eventually be revised into agreeing with whatever you hoped happened.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CycleOutcome(StrEnum):
    NO_SIGNAL = "no_signal"
    WATCH = "watch"
    REJECTED_BY_RISK = "rejected_by_risk"
    REJECTED_BY_CRITIC = "rejected_by_critic"
    SHADOW_FILLED = "shadow_filled"
    LIVE_FILLED = "live_filled"
    ERROR = "error"


class AuditEntry(BaseModel):
    """One symbol's evaluation in one cycle, whatever the result."""

    model_config = ConfigDict(frozen=True)

    cycle_id: str
    occurred_at: datetime
    symbol: str
    strategy: str
    outcome: CycleOutcome

    reference_price: Decimal | None = None
    confidence: float = 0.0
    signal_strength: str | None = None

    reasons: list[str] = Field(default_factory=list)
    failed_conditions: list[str] = Field(default_factory=list)
    risk_breaches: list[str] = Field(default_factory=list)
    critic_notes: list[str] = Field(default_factory=list)

    # The full snapshot, so a decision can be replayed exactly as it was made.
    snapshot_json: dict[str, Any] | None = None


class TradeRecord(BaseModel):
    """An entry and, once closed, its outcome.

    Deliberately not a pair of rows. Keeping entry and exit on one record makes
    the natural questions — hold time, realized R, whether the stop was
    respected — a read rather than a join.
    """

    model_config = ConfigDict(frozen=True)

    client_key: str
    symbol: str
    strategy: str
    mode: str  # "shadow" | "live"

    opened_at: datetime
    entry_price: Decimal
    quantity: Decimal
    notional: Decimal
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    entry_rationale: list[str] = Field(default_factory=list)

    closed_at: datetime | None = None
    exit_price: Decimal | None = None
    exit_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def realized_pnl(self) -> Decimal | None:
        if self.exit_price is None:
            return None
        return ((self.exit_price - self.entry_price) * self.quantity).quantize(Decimal("0.01"))

    @property
    def realized_r(self) -> Decimal | None:
        """P&L in units of initial risk — the only comparable measure across
        positions of different sizes and stop widths."""
        if self.exit_price is None or self.stop_price is None:
            return None
        risk_per_share = self.entry_price - self.stop_price
        if risk_per_share <= 0:
            return None
        return ((self.exit_price - self.entry_price) / risk_per_share).quantize(Decimal("0.01"))

    @property
    def holding_days(self) -> int | None:
        if self.closed_at is None:
            return None
        return (self.closed_at - self.opened_at).days


class JournalEntry(BaseModel):
    """Envelope pairing a cycle's audit trail with any trade it produced."""

    model_config = ConfigDict(frozen=True)

    audit: AuditEntry
    trade: TradeRecord | None = None
