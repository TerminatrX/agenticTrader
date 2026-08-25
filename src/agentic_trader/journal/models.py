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

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_trader.models import ExecutionMode, ProtectionState


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

    # Which execution context produced this. Required, with no default: a
    # default would be an inference dressed as a fact, and the whole reason
    # this field exists is that every available inference -- a TradeRecord
    # appearing, an ExecutionPlan existing, an outcome value -- reads the
    # consequence and guesses the context backwards. Five of the seven outcomes
    # produce no trade and no plan in *any* mode, so for those the guess has
    # nothing to go on at all.
    mode: ExecutionMode

    # The date the acquisition contract's ranged requests were built from.
    # Every `start_time` is `trading_date - lookback`, so this is the input
    # that decided which bars the broker computed over.
    #
    # Not `occurred_at`, not `snapshot.captured_at`, and certainly not today.
    # `captured_at` says when the snapshot was assembled; it answers a
    # different question and coincides with this only by habit. The exact
    # input is known at the boundary, so the input is what gets stored.
    trading_date: date

    # Which pinned request contract produced the inputs behind this decision.
    # The snapshot records what came back; these record what was asked for, and
    # without them a replay cannot tell a decision made on 30 bars of RSI
    # warm-up from the same decision made on 300.
    #
    # Required, with no defaults. The *columns* are nullable because rows
    # written before the contract existed genuinely have none, but this model
    # describes a NEW write -- and a default here would let a caller omit the
    # provenance and have the record claim the current contract anyway, which
    # is precisely the substitution the boundary check exists to prevent.
    acquisition_profile_ref: str
    acquisition_config_fingerprint: str

    reference_price: Decimal | None = None
    confidence: float = 0.0
    signal_strength: str | None = None

    # The critic's effect on sizing, recorded as a pair so the gap is
    # measurable. Equal values mean the critic did not move the position.
    original_confidence: float | None = None
    adjusted_confidence: float | None = None

    thesis: str | None = None
    invalidation_reason: str | None = None

    # Whether the position this cycle would open could be covered by a resting
    # broker stop, and the capability profile that judgement was made against.
    # `None` when the cycle produced no order. The profile ref matters because
    # capability understanding changes: without it you cannot later tell an
    # unprotectable position from one the system did not yet know how to protect.
    protection_state: ProtectionState | None = None
    capability_profile: str | None = None

    # The market backdrop this decision was made against. Recorded, never
    # enforced — the point is to accumulate expectancy per regime and let any
    # future gating rule follow the evidence rather than precede it.
    market_regime: str | None = None
    market_context: dict[str, Any] | None = None

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

    # Same type as `AuditEntry.mode`, so the two cannot describe one cycle in
    # two vocabularies. Previously a bare `str`, which admitted "Shadow",
    # "SHADOW", and typos as distinct modes.
    mode: ExecutionMode

    opened_at: datetime
    entry_price: Decimal
    quantity: Decimal
    notional: Decimal
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    entry_rationale: list[str] = Field(default_factory=list)

    # What the trade believed, and what would prove it wrong. Recorded at entry
    # so a post-mortem reads the reasoning as it stood, not as it is remembered.
    thesis: str | None = None
    invalidation_reason: str | None = None
    sector: str | None = None

    # Was this position ever actually covered by a resting broker stop? Kept on
    # the trade rather than only in `protective_orders` so the question is a
    # read on the record that already exists, and so a position that never had
    # a protective order at all is still answerable.
    protection_state: ProtectionState = ProtectionState.NOT_REQUIRED
    capability_profile: str | None = None

    # Denormalized onto the trade so "expectancy by regime" is a group-by on
    # closed trades rather than a join back through the audit stream.
    market_regime: str | None = None
    stop_basis: str | None = None

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
    def planned_risk_reward(self) -> Decimal | None:
        """Reward-to-risk as planned at entry, for comparison against realized R."""
        if self.stop_price is None or self.target_price is None:
            return None
        risk = self.entry_price - self.stop_price
        if risk <= 0:
            return None
        return ((self.target_price - self.entry_price) / risk).quantize(Decimal("0.01"))

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
