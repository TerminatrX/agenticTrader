"""Turn an approved `RiskDecision` into an exact broker payload — and stop there.

This module builds the argument dict for the `place_equity_order` MCP tool and
returns it. It does not submit. Submission is the agent's job, because the MCP
tools are exposed to the agent rather than to this process, and keeping the
only irreversible action outside the library means no unit test, import, or
stray function call can ever put money at risk.

Three broker constraints drive the payload shape, all of them discovered from
the tool schema rather than assumed:

1. **Dollar-denominated and fractional orders must be market orders**, in
   regular hours only. A fractional *limit* order is rejected outright. Since a
   small account can only take a position in a high-priced name fractionally,
   the default entry is a market order — which means no price protection, so
   `preflight` checks the spread and quote freshness instead. That check is
   load-bearing, not decorative.

2. **`ref_id` must be a UUID** and is the broker's own idempotency key. We pass
   the deterministic key from the risk engine, so a retry deduplicates on both
   sides.

3. **The entry order cannot carry its own stop.** `stop_price` on this endpoint
   selects a stop *order type*; it does not attach a protective stop to a
   market buy. The stop in a `TradeIntent` is therefore a managed level: after
   a fill, the system must either place a separate stop order or enforce the
   level on subsequent cycles. Treating it as broker-native is the mistake this
   docstring exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from agentic_trader.execution.capabilities import ROBINHOOD_MCP, BrokerCapabilities
from agentic_trader.models import (
    LIVE_PERMITTED_PROTECTION,
    MarketSnapshot,
    ProtectionState,
    RiskDecision,
    Side,
    TradeIntent,
)

MAX_FRACTIONAL_DP = Decimal("0.000001")  # Broker allows 6 decimal places.


class PreflightError(RuntimeError):
    """Raised when a decision must not become an order."""


OrderPayload = dict[str, Any]


@dataclass
class ExecutionPlan:
    """A ready-to-submit order plus everything a human needs to approve it."""

    payload: OrderPayload
    intent: TradeIntent
    mode: Literal["live", "shadow"]
    estimated_cost: Decimal
    managed_stop: Decimal | None = None
    managed_target: Decimal | None = None
    warnings: list[str] = field(default_factory=list)
    preflight_notes: list[str] = field(default_factory=list)

    # Whether this position will actually be covered by a resting broker order,
    # and which capability profile that judgement was made against. The profile
    # ref is stored so a decision stays readable after the profile moves on.
    protection: ProtectionState = ProtectionState.NOT_REQUIRED
    protection_note: str = ""
    capability_profile: str = ""

    def summary(self) -> str:
        verb = self.intent.side.value.upper()
        return (
            f"{verb} {self.intent.symbol} ${self.intent.notional:.2f} "
            f"(~{self.intent.estimated_quantity:.6f} sh @ ~{self.intent.reference_price:.2f}) "
            f"[{self.mode}]"
        )


def assess_protection(
    intent: TradeIntent,
    *,
    capabilities: BrokerCapabilities = ROBINHOOD_MCP,
) -> tuple[ProtectionState, str]:
    """Decide whether this order's position can be covered by a resting stop.

    Returns the state and the reason, never raising — the caller decides what
    the state is permitted to mean, which differs between shadow and live.
    """
    if intent.side is not Side.BUY or intent.stop_price is None:
        return (
            ProtectionState.NOT_REQUIRED,
            "exit order" if intent.side is not Side.BUY else "intent carries no stop",
        )

    feasibility = capabilities.protection_feasibility(intent.estimated_quantity)

    if feasibility.protectable is True:
        # Reachable, but nothing submits it yet: the stop lifecycle is not
        # implemented. PENDING rather than PROTECTED, because a position whose
        # stop was never placed is not protected however placeable it was.
        return (
            ProtectionState.PENDING,
            f"{feasibility.reason}; stop placement is not implemented, so the "
            "position is not yet covered",
        )

    # `None` (capability unknown) lands here with FALSE, deliberately. Unknown
    # protection is not protection, and it must gate exactly as a known-absent
    # capability does.
    return ProtectionState.UNAVAILABLE, feasibility.reason


def preflight(
    decision: RiskDecision,
    snapshot: MarketSnapshot,
    *,
    max_spread_pct: Decimal,
    max_price_drift_pct: Decimal,
    max_quote_age_seconds: int = 120,
    now: datetime | None = None,
    known_client_keys: set[str] | None = None,
) -> list[str]:
    """Last line of defence before an order exists. Raises rather than warns.

    Everything checked here is specific to the moment of submission — staleness,
    spread, duplicate keys — which is exactly what the risk engine, evaluated
    earlier in the cycle, cannot know.
    """
    if not decision.is_executable or decision.intent is None:
        raise PreflightError(
            f"decision is {decision.decision.value}; there is nothing to execute"
        )

    intent = decision.intent
    notes: list[str] = []

    if known_client_keys and intent.client_key in known_client_keys:
        raise PreflightError(
            f"client_key {intent.client_key} was already submitted — refusing to "
            "place a duplicate order"
        )

    if not snapshot.tradable:
        raise PreflightError(f"symbol not tradable: {snapshot.staleness_note}")

    current = now or datetime.now(UTC)

    # 1. Quote age, measured from the venue's own print time. Deliberately not
    # `captured_at`, which is set to now() when the snapshot is built and so
    # reports every snapshot as fresh — including one replayed from a bundle
    # months later.
    quote_age = snapshot.quote_age_seconds(current)
    if quote_age is None:
        raise PreflightError(
            "quote carries no venue timestamp, so its age cannot be verified — "
            "refusing to submit a market order against an unknown-age price"
        )
    if quote_age > max_quote_age_seconds:
        raise PreflightError(
            f"quote is {quote_age:.0f}s old (limit {max_quote_age_seconds}s) — "
            "re-fetch the quote before submitting a market order"
        )
    notes.append(f"quote age {quote_age:.0f}s")

    # 2. Spread. Entries are market orders by broker constraint, so this is paid
    # in full on every fill — a wide spread is how a small account donates money.
    spread = snapshot.spread_pct
    if spread is None:
        raise PreflightError(
            "no usable bid/ask (missing, zero, or crossed book) — the spread on "
            "a market order cannot be bounded, so the order is refused"
        )
    if spread > max_spread_pct:
        raise PreflightError(
            f"spread is {spread:.2%} (bid {snapshot.bid}, ask {snapshot.ask}), "
            f"exceeding {max_spread_pct:.2%} — the fill would give back more "
            "than the setup is worth"
        )
    notes.append(f"spread {spread:.2%} within tolerance")

    # 3. Drift between the price the decision was made at and the live price.
    # A separate question from the spread: this one asks whether the setup still
    # exists, not what crossing the book costs.
    drift = abs(snapshot.last_price - intent.reference_price) / intent.reference_price
    if drift > max_price_drift_pct:
        raise PreflightError(
            f"price moved {drift:.2%} from the decision reference "
            f"({intent.reference_price:.2f} -> {snapshot.last_price:.2f}), "
            f"exceeding {max_price_drift_pct:.2%} — re-evaluate rather than chase"
        )
    notes.append(f"price drift {drift:.2%} within tolerance")

    # Reported, never gated: how long this process took between assembling the
    # snapshot and reaching here. Useful for spotting a slow cycle; it says
    # nothing about whether the market data is current.
    notes.append(f"pipeline latency {(current - snapshot.captured_at).total_seconds():.0f}s")

    return notes


def build_order_payload(
    decision: RiskDecision,
    snapshot: MarketSnapshot,
    account_number: str,
    *,
    max_spread_pct: Decimal = Decimal("0.005"),
    max_price_drift_pct: Decimal = Decimal("0.005"),
    mode: Literal["live", "shadow"] = "shadow",
    use_dollar_amount: bool = True,
    known_client_keys: set[str] | None = None,
    allow_unprotected_shadow_entries: bool = True,
    capabilities: BrokerCapabilities = ROBINHOOD_MCP,
    now: datetime | None = None,
) -> ExecutionPlan:
    """Build the `place_equity_order` arguments for an approved decision.

    `use_dollar_amount=True` sends a notional and lets the broker compute
    shares. That is the right default for entries. For exits it is wrong — the
    goal there is to close a known share count exactly, so the quantity form is
    used regardless.
    """
    notes = preflight(
        decision,
        snapshot,
        max_spread_pct=max_spread_pct,
        max_price_drift_pct=max_price_drift_pct,
        known_client_keys=known_client_keys,
        now=now,
    )
    intent = decision.intent
    assert intent is not None  # preflight guarantees this

    protection, protection_note = assess_protection(intent, capabilities=capabilities)

    # Two gates, and the order matters. The structural one comes first and no
    # configuration reaches it: outside shadow, a position that is not provably
    # covered is refused, full stop. The config key can only tighten shadow
    # behaviour — it can never admit an unprotected live entry.
    if protection not in LIVE_PERMITTED_PROTECTION:
        if mode != "shadow":
            raise PreflightError(
                f"protection state is {protection.value} and mode is {mode!r} — "
                f"refusing to open a position that is not provably protected. "
                f"{protection_note}"
            )
        if not allow_unprotected_shadow_entries:
            raise PreflightError(
                f"protection state is {protection.value} and "
                f"allow_unprotected_shadow_entries is disabled. {protection_note}"
            )
    notes.append(f"protection {protection.value}: {protection_note}")

    payload: OrderPayload = {
        "account_number": account_number,
        "symbol": intent.symbol,
        "side": intent.side.value,
        # Fractional and dollar-denominated orders are market-only, regular
        # hours only. Both are enforced by the broker; setting them explicitly
        # keeps the rejection reason obvious if that ever changes.
        "type": "market",
        "market_hours": "regular_hours",
        "time_in_force": "gfd",
        "ref_id": intent.client_key,
    }

    if intent.side is Side.SELL or not use_dollar_amount:
        quantity = intent.estimated_quantity.quantize(MAX_FRACTIONAL_DP)
        payload["quantity"] = f"{quantity:f}"
    else:
        payload["dollar_amount"] = f"{intent.notional:.2f}"

    warnings = list(decision.warnings)
    warnings.append(
        "market order: no price protection. Preflight bounded the drift, but the "
        "fill may still differ from the reference price."
    )
    if intent.stop_price is not None:
        warnings.append(
            f"stop {intent.stop_price:.2f} is MANAGED, not broker-native — this order "
            "does not carry it. Place a separate stop order after the fill, or the "
            "position is unprotected between cycles."
        )
    if protection is ProtectionState.UNAVAILABLE:
        warnings.append(
            f"POSITION CANNOT BE PROTECTED: {protection_note}. The managed stop is "
            "checked once per cycle and nothing enforces it in between."
        )

    return ExecutionPlan(
        payload=payload,
        intent=intent,
        mode=mode,
        estimated_cost=intent.notional,
        managed_stop=intent.stop_price,
        managed_target=intent.target_price,
        warnings=warnings,
        preflight_notes=notes,
        protection=protection,
        protection_note=protection_note,
        capability_profile=capabilities.profile_ref,
    )


def build_review_payload(plan: ExecutionPlan) -> OrderPayload:
    """Arguments for `review_equity_order`, which mirrors place minus `ref_id`.

    Reviewing before placing is the default workflow the broker expects, and it
    surfaces cost and alerts the agent should show a human before committing.
    """
    return {k: v for k, v in plan.payload.items() if k != "ref_id"}


def describe_plan(plan: ExecutionPlan) -> str:
    """Human-readable block for the approval prompt."""
    lines = [
        plan.summary(),
        f"  strategy   : {plan.intent.strategy}",
        f"  confidence : {plan.intent.confidence:.2f}",
        f"  notional   : ${plan.estimated_cost:.2f}",
    ]
    if plan.managed_stop is not None:
        risk = plan.estimated_cost * (
            (plan.intent.reference_price - plan.managed_stop) / plan.intent.reference_price
        )
        lines.append(f"  stop       : {plan.managed_stop:.2f} (managed, risking ~${risk:.2f})")
    if plan.managed_target is not None:
        lines.append(f"  target     : {plan.managed_target:.2f}")
    for reason in plan.intent.rationale:
        lines.append(f"  + {reason}")
    for warning in plan.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
