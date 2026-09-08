"""The risk engine: the only path from a strategy opinion to an executable order.

Order of operations is fixed and deliberate:

    1. Gates      — is trading permitted at all? (risk.limits)
    2. Sizing     — how much, working back from the stop? (risk.sizing)
    3. Intent     — build the concrete order with an idempotency key
    4. Ruling     — APPROVED / RESIZED / REJECTED

Gates run first because sizing an order that was never permitted wastes work
and, worse, produces a plausible-looking notional that a careless caller might
act on. A rejected `RiskDecision` carries no intent at all, so there is nothing
downstream to accidentally execute.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from agentic_trader.config import RiskConfig
from agentic_trader.market.earnings_capabilities import (
    ROBINHOOD_MCP_EARNINGS,
    EarningsCapabilities,
)
from agentic_trader.models import (
    AccountState,
    Decision,
    MarketSnapshot,
    RiskDecision,
    Side,
    Signal,
    TradeIntent,
)
from agentic_trader.risk.limits import LimitCheck, check_limits
from agentic_trader.risk.sizing import exit_notional, size_position

# Stable namespace for deterministic order keys. Never change this value:
# doing so would make every historical key unreproducible and silently defeat
# deduplication against orders already recorded in the journal.
_ORDER_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def build_client_key(
    account_number: str,
    symbol: str,
    side: Side,
    strategy: str,
    session: date,
) -> str:
    """Deterministic idempotency key for one intent within one session.

    The same setup evaluated twice in a day produces the same key, so a
    re-fired cycle — a retry, a duplicated schedule tick, a crashed run
    restarted — collides with the already-recorded order instead of placing a
    second one. Execution is responsible for refusing a known key; this
    function only guarantees the key is stable.

    A UUIDv5 rather than a plain hash, because this value is passed straight to
    the broker as `ref_id`, which must be a UUID. Deriving it deterministically
    means the broker's own deduplication and ours key off the same string.
    """
    material = f"{account_number}|{symbol}|{side.value}|{strategy}|{session.isoformat()}"
    return str(uuid.uuid5(_ORDER_NAMESPACE, material))


def build_protective_client_key(
    entry_client_key: str,
    stop_price: Decimal,
    quantity: Decimal,
) -> str:
    """Deterministic `ref_id` for the resting stop that covers one entry.

    Derived from the entry's key so the protective order is traceable to the
    position it covers, and so a retry of the *same* stop — a crashed run
    resumed, a duplicated confirmation — dedupes at the broker instead of
    stacking a second stop on one position. Two resting stops on a position
    that can only be sold once means the second becomes a naked short the
    moment the first fills.

    Price and quantity are in the material because this broker cannot modify a
    resting order: `replace_orders` is unsupported, so moving a stop is a
    cancel followed by a new placement. A moved stop is genuinely a different
    order and must not collide with the one it replaces, while an identical
    re-submission must.
    """
    material = f"{entry_client_key}|protective_stop|{stop_price:f}|{quantity:f}"
    return str(uuid.uuid5(_ORDER_NAMESPACE, material))


def build_flatten_client_key(entry_client_key: str, quantity: Decimal) -> str:
    """Deterministic `ref_id` for the order that unwinds an unprotectable fill.

    Quantity is in the material because a partial flatten followed by a second
    attempt at the remainder is two genuinely different orders; repeating the
    *same* attempt is one, and must dedupe rather than sell the position twice.
    Selling twice is not a duplicate here — the second sale is a short.
    """
    material = f"{entry_client_key}|flatten|{quantity:f}"
    return str(uuid.uuid5(_ORDER_NAMESPACE, material))


class RiskEngine:
    """Stateless evaluator. Construct per cycle; hold no memory between calls."""

    def __init__(
        self,
        config: RiskConfig,
        *,
        is_halted: bool = False,
        earnings_capabilities: EarningsCapabilities = ROBINHOOD_MCP_EARNINGS,
    ) -> None:
        self.config = config
        self.is_halted = is_halted
        # Held explicitly rather than reached for inside the gate. The earnings
        # blackout's correctness depends on which source produced the evidence,
        # so that dependency belongs in the engine's signature where it can be
        # substituted in a test and seen in a review.
        self.earnings_capabilities = earnings_capabilities

    def evaluate(
        self,
        signal: Signal,
        snapshot: MarketSnapshot,
        account: AccountState,
        *,
        as_of: date | None = None,
        last_loss_exit: date | None = None,
    ) -> RiskDecision:
        today = as_of or datetime.now(UTC).date()

        if not signal.is_actionable:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=[f"signal strength is {signal.strength.value}; nothing to execute"],
            )
        if signal.side is None:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=["actionable signal carries no side"],
            )

        gates = check_limits(
            signal,
            snapshot,
            account,
            self.config,
            is_halted=self.is_halted,
            as_of=today,
            last_loss_exit=last_loss_exit,
            earnings_capabilities=self.earnings_capabilities,
        )
        if not gates.passed:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=gates.breaches,
                warnings=gates.warnings,
                notes=gates.notes,
                trip_kill_switch=gates.trip_kill_switch,
            )

        if signal.side is Side.SELL:
            return self._build_exit(signal, account, gates, today)
        return self._build_entry(
            signal, account, gates, today, sector=snapshot.sector, confidence_override=None
        )

    def resize(
        self,
        decision: RiskDecision,
        signal: Signal,
        snapshot: MarketSnapshot,
        account: AccountState,
        adjusted_confidence: float,
        *,
        as_of: date | None = None,
    ) -> RiskDecision:
        """Re-size an already-approved entry at a lower confidence.

        Used once per cycle after critic review. The gates are not re-run: they
        already passed, and a smaller order cannot breach a limit a larger one
        satisfied. The caller must guarantee `adjusted_confidence` is not above
        the original — this method does not police that, and
        `agents.orchestrator` asserts the resulting notional did not grow.
        """
        if decision.intent is None or decision.intent.side is not Side.BUY:
            return decision

        gates = LimitCheck(
            warnings=list(decision.warnings),
            notes=list(decision.notes),
            trip_kill_switch=decision.trip_kill_switch,
        )
        return self._build_entry(
            signal,
            account,
            gates,
            as_of or datetime.now(UTC).date(),
            sector=snapshot.sector,
            confidence_override=adjusted_confidence,
        )

    # ---------------------------------------------------------------- entries

    def _build_entry(
        self, signal, account, gates, today, *, sector, confidence_override
    ) -> RiskDecision:
        sizing = size_position(
            signal,
            account,
            self.config,
            sector=sector,
            confidence_override=confidence_override,
        )
        if not sizing.approved:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=[sizing.rejection_reason or "sizing rejected the order"],
                warnings=gates.warnings,
                notes=[*gates.notes, *sizing.caps_applied],
                trip_kill_switch=gates.trip_kill_switch,
            )

        intent = TradeIntent(
            symbol=signal.symbol,
            side=Side.BUY,
            strategy=signal.strategy,
            client_key=build_client_key(
                account.account_number, signal.symbol, Side.BUY, signal.strategy, today
            ),
            notional=sizing.notional,
            limit_price=signal.reference_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            reference_price=signal.reference_price,
            confidence=(
                signal.confidence if confidence_override is None else confidence_override
            ),
            rationale=signal.reasons,
            created_at=datetime.now(UTC),
            thesis=signal.thesis,
            invalidation_reason=signal.invalidation_reason,
            sector=sector,
        )

        notes = [
            *gates.notes,
            *sizing.caps_applied,
            f"risk budget {sizing.risk_budget:.2f} at {self.config.risk_per_trade_pct:.1%} "
            f"of {account.total_value:.2f}",
            f"stop distance {sizing.stop_distance_pct:.2%}; "
            f"binding constraint: {sizing.binding_constraint}",
            f"implied quantity {intent.estimated_quantity:.6f} shares "
            f"at {intent.reference_price:.2f} (fractional order)",
        ]
        if intent.estimated_max_loss is not None:
            notes.append(
                f"modeled loss to stop {intent.estimated_max_loss:.2f} "
                f"(not a floor — managed stop, market entry, gap risk)"
            )
        if intent.risk_reward_ratio is not None:
            notes.append(f"reward-to-risk {intent.risk_reward_ratio:.2f}")

        # "Resized" is reserved for the case where something other than the
        # risk formula itself shrank the order, since that is the signal a
        # human wants to notice.
        resized = sizing.binding_constraint not in (None, "risk_budget")
        return RiskDecision(
            decision=Decision.RESIZED if resized else Decision.APPROVED,
            intent=intent,
            approved_notional=sizing.notional,
            warnings=gates.warnings,
            notes=notes,
            trip_kill_switch=gates.trip_kill_switch,
        )

    # ----------------------------------------------------------------- exits

    def _build_exit(self, signal, account, gates, today) -> RiskDecision:
        price = signal.reference_price
        if price is None:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=["exit signal carries no reference price"],
                notes=gates.notes,
            )

        notional = exit_notional(account, signal.symbol, price)
        if notional <= 0:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=[f"computed exit notional for {signal.symbol} is {notional}"],
                notes=gates.notes,
            )

        intent = TradeIntent(
            symbol=signal.symbol,
            side=Side.SELL,
            strategy=signal.strategy,
            client_key=build_client_key(
                account.account_number, signal.symbol, Side.SELL, signal.strategy, today
            ),
            notional=notional,
            limit_price=price,
            reference_price=price,
            confidence=signal.confidence,
            rationale=signal.reasons,
            created_at=datetime.now(UTC),
            thesis=signal.thesis,
            invalidation_reason=signal.invalidation_reason,
        )

        notes = [*gates.notes]
        if account.is_cash_account:
            notes.append(
                f"cash account: ~{notional:.2f} in proceeds will be unsettled until T+1 "
                "and unavailable for new entries until then"
            )

        return RiskDecision(
            decision=Decision.APPROVED,
            intent=intent,
            approved_notional=notional,
            warnings=gates.warnings,
            notes=notes,
            trip_kill_switch=gates.trip_kill_switch,
        )
