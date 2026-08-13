"""Hard gates evaluated before any order is sized.

These are pass/fail questions about whether trading is permitted at all —
distinct from sizing, which asks how much. Splitting them matters: a limit
breach is a refusal with a reason, while a sizing constraint quietly shrinks
the order. Conflating the two produces orders that are technically allowed but
were never actually approved.

Every check returns a reason string on failure. Those strings are what the
critic agent reads and what a human reads three months later wondering why a
trade did not happen, so they carry the actual numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from agentic_trader.config import RiskConfig
from agentic_trader.models import AccountState, MarketSnapshot, Side, Signal


@dataclass
class LimitCheck:
    """Outcome of the gate battery."""

    passed: bool = True
    breaches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def breach(self, reason: str) -> None:
        self.passed = False
        self.breaches.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)

    def note(self, reason: str) -> None:
        self.notes.append(reason)


def check_limits(
    signal: Signal,
    snapshot: MarketSnapshot,
    account: AccountState,
    config: RiskConfig,
    *,
    is_halted: bool = False,
    as_of: date | None = None,
    last_loss_exit: date | None = None,
) -> LimitCheck:
    """Run every gate. Ordered cheapest-and-most-absolute first."""
    result = LimitCheck()
    today = as_of or date.today()
    is_entry = signal.side is Side.BUY

    # --- Absolute stops ---------------------------------------------------

    if is_halted:
        result.breach("HALT file present — trading disabled")
        return result  # Nothing else is worth evaluating.

    if not snapshot.tradable:
        result.breach(f"symbol not tradable: {snapshot.staleness_note or 'unknown reason'}")

    # Exits are allowed to proceed past the remaining gates. A risk control
    # that blocks closing a position increases risk rather than reducing it.
    if not is_entry:
        _check_exit_specific(signal, account, result)
        return result

    # --- Portfolio-level gates -------------------------------------------

    daily_loss_limit = -(account.total_value * config.max_daily_loss_pct)
    if account.realized_pnl_today <= daily_loss_limit:
        result.breach(
            f"daily loss {account.realized_pnl_today:.2f} breached limit "
            f"{daily_loss_limit:.2f} ({config.max_daily_loss_pct:.1%} of account)"
        )

    if account.open_position_count >= config.max_open_positions:
        result.breach(
            f"already holding {account.open_position_count} positions "
            f"(max {config.max_open_positions})"
        )

    exposure = sum((p.market_value or p.cost_basis) for p in account.positions)
    max_exposure = account.total_value * config.max_portfolio_exposure_pct
    if exposure >= max_exposure:
        result.breach(
            f"portfolio exposure {exposure:.2f} at or above cap {max_exposure:.2f} "
            f"({config.max_portfolio_exposure_pct:.0%})"
        )

    # --- Symbol-level gates ----------------------------------------------

    if account.position_in(signal.symbol) is not None:
        result.breach(f"already holding {signal.symbol}; this strategy does not add to winners")

    if signal.symbol in account.open_order_symbols:
        result.breach(
            f"an order for {signal.symbol} is already open — "
            "resolve it before submitting another"
        )

    if last_loss_exit is not None and config.symbol_cooldown_days > 0:
        elapsed = (today - last_loss_exit).days
        if elapsed < config.symbol_cooldown_days:
            result.breach(
                f"{signal.symbol} in cooldown: {elapsed}d since losing exit "
                f"(requires {config.symbol_cooldown_days}d)"
            )

    # --- Event risk -------------------------------------------------------

    if snapshot.earnings is not None:
        days_out = snapshot.earnings.days_until(today)
        if 0 <= days_out <= config.earnings_blackout_days:
            confidence = "confirmed" if snapshot.earnings.verified else "tentative"
            result.breach(
                f"earnings in {days_out}d ({snapshot.earnings.report_date}, {confidence}) "
                f"— inside {config.earnings_blackout_days}d blackout"
            )
        elif days_out <= config.earnings_blackout_days + 5:
            result.warn(
                f"earnings in {days_out}d ({snapshot.earnings.report_date}) — "
                "position may need closing before the report"
            )

    # --- Liquidity --------------------------------------------------------

    if snapshot.average_volume_30d is None:
        result.warn("no volume data — liquidity unverified")
    elif snapshot.average_volume_30d < config.min_avg_volume_30d:
        result.breach(
            f"30d average volume {snapshot.average_volume_30d:,.0f} below "
            f"minimum {config.min_avg_volume_30d:,.0f}"
        )

    # --- Stop sanity ------------------------------------------------------

    stop_distance = signal.stop_distance_pct
    if stop_distance is None:
        result.breach("signal carries no stop; position cannot be sized")
    elif stop_distance <= 0:
        result.breach(f"stop {signal.stop_price} is at or above entry {signal.reference_price}")
    elif stop_distance > config.max_stop_pct:
        result.breach(
            f"stop distance {stop_distance:.2%} exceeds maximum {config.max_stop_pct:.2%} "
            "— setup too loose to size responsibly"
        )

    # --- Cash and settlement ---------------------------------------------

    if account.buying_power <= 0:
        result.breach(f"no buying power ({account.buying_power:.2f})")

    if config.respect_unsettled_funds and account.is_cash_account and account.unsettled_funds > 0:
        # Broker buying power already excludes unsettled proceeds. The note
        # exists so a smaller-than-expected order has a stated cause rather
        # than looking like a sizing bug.
        result.note(
            f"cash account: {account.unsettled_funds:.2f} unsettled and excluded from "
            f"buying power ({account.buying_power:.2f}); spending it would risk a "
            "good-faith violation"
        )

    return result


def _check_exit_specific(signal: Signal, account: AccountState, result: LimitCheck) -> None:
    """Exits have exactly one precondition: something to sell."""
    position = account.position_in(signal.symbol)
    if position is None or position.quantity <= 0:
        result.breach(f"no open position in {signal.symbol} to exit")
        return
    if signal.symbol in account.open_order_symbols:
        result.warn(f"an order for {signal.symbol} is already open; exit may conflict")
    result.note(
        f"exiting {position.quantity} shares of {signal.symbol} "
        f"(avg cost {position.average_cost:.2f})"
    )


def unsettled_after_sale(account: AccountState, proceeds: Decimal) -> Decimal:
    """Projected unsettled balance once a sale settles into the account.

    Useful for the review skill: sizing the *next* entry against what will
    actually be spendable rather than against total cash.
    """
    return account.unsettled_funds + proceeds
