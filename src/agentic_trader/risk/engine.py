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

from agentic_trader.config import RiskConfig
from agentic_trader.models import (
    AccountState,
    Decision,
    MarketSnapshot,
    RiskDecision,
    Side,
    Signal,
    TradeIntent,
)
from agentic_trader.risk.limits import check_limits
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


class RiskEngine:
    """Stateless evaluator. Construct per cycle; hold no memory between calls."""

    def __init__(self, config: RiskConfig, *, is_halted: bool = False) -> None:
        self.config = config
        self.is_halted = is_halted

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
        )
        if not gates.passed:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=gates.breaches,
                warnings=gates.warnings,
                notes=gates.notes,
            )

        if signal.side is Side.SELL:
            return self._build_exit(signal, account, gates, today)
        return self._build_entry(signal, account, gates, today)

    # ---------------------------------------------------------------- entries

    def _build_entry(self, signal, account, gates, today) -> RiskDecision:
        sizing = size_position(signal, account, self.config)
        if not sizing.approved:
            return RiskDecision(
                decision=Decision.REJECTED,
                breached_limits=[sizing.rejection_reason or "sizing rejected the order"],
                warnings=gates.warnings,
                notes=[*gates.notes, *sizing.caps_applied],
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
            confidence=signal.confidence,
            rationale=signal.reasons,
            created_at=datetime.now(UTC),
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
        )
