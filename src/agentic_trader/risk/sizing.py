"""Position sizing, denominated in dollars rather than shares.

Sizing works backward from the loss you are willing to take, not forward from
the cash you happen to have:

    risk budget  = account value x risk_per_trade_pct
    notional     = risk budget / stop distance as a fraction of entry

A 1% risk budget with a 5% stop yields a 20% position; the same budget with a
10% stop yields 10%. The stop, not the conviction, is what decides the size.

Dollar denomination is not a stylistic choice. A $100 account cannot buy a
single share of a $300 stock, so a share-based sizer returns zero and the
system never trades. Robinhood supports fractional equity orders, so the
notional is the real quantity and shares are derived from it.

Every cap that binds is recorded. A silently shrunken order is indistinguishable
from a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

from agentic_trader.config import RiskConfig
from agentic_trader.models import AccountState, Signal

CENTS = Decimal("0.01")


@dataclass
class SizingResult:
    """The computed order size and the full record of how it got there."""

    notional: Decimal
    approved: bool
    risk_budget: Decimal
    stop_distance_pct: Decimal | None
    binding_constraint: str | None = None
    caps_applied: list[str] = field(default_factory=list)
    rejection_reason: str | None = None

    @property
    def quantity_at(self) -> object:
        """Convenience for callers: shares implied at a given price."""
        return lambda price: (self.notional / price) if price else Decimal("0")


def max_whole_share_price(
    account: AccountState,
    config: RiskConfig,
    stop_distance: Decimal,
    *,
    confidence: Decimal = Decimal("1"),
) -> Decimal:
    """Highest share price at which this account can hold a *whole* share.

    The notional a position would be sized to is also the most expensive single
    share it can buy outright, so this returns that notional. Its purpose is to
    make a constraint visible that is otherwise discovered one symbol at a time:
    above this price every position is fractional, and a fractional position
    cannot carry a resting stop, so it can never be held live.

    Deliberately an **upper bound**. Sector headroom and any confidence below
    1.0 only shrink the result, so a symbol priced under this number *may* be
    takeable while one priced above it definitely is not. Reporting the
    optimistic figure is the honest direction for a ceiling — it never claims a
    symbol is out of reach when it is not.

    Not a gate. Nothing calls this to refuse a trade: `assess_protection` still
    decides that, from the quantity actually sized against a real quote. This
    answers the operator's question — what can this account trade at all? —
    without becoming a second, drifting copy of the sizing rules.
    """
    if stop_distance <= 0:
        return Decimal("0")

    risk_budget = (account.total_value * config.risk_per_trade_pct).quantize(CENTS)
    ceiling = risk_budget / stop_distance

    if confidence > 0:
        ceiling = min(ceiling, ceiling * confidence)

    ceiling = min(ceiling, account.total_value * config.max_position_pct)
    if config.max_order_notional is not None:
        ceiling = min(ceiling, config.max_order_notional)
    ceiling = min(ceiling, account.buying_power)

    return ceiling.quantize(CENTS, rounding=ROUND_DOWN)


def size_position(
    signal: Signal,
    account: AccountState,
    config: RiskConfig,
    *,
    scale_by_confidence: bool = True,
    sector: str | None = None,
    confidence_override: float | None = None,
) -> SizingResult:
    """Compute the dollar notional for an entry signal.

    `sector` enables the sector-headroom cap; without it that limit cannot be
    enforced here and the gate in `risk.limits` only warns.

    `confidence_override` re-sizes with an adjusted confidence after critic
    review. It may only ever *lower* the effective confidence — the caller is
    responsible for clamping, and `agents.orchestrator` asserts the resulting
    notional did not grow.
    """
    stop_distance = signal.stop_distance_pct

    if stop_distance is None or stop_distance <= 0:
        return SizingResult(
            notional=Decimal("0"),
            approved=False,
            risk_budget=Decimal("0"),
            stop_distance_pct=stop_distance,
            rejection_reason="cannot size without a valid stop below entry",
        )

    risk_budget = (account.total_value * config.risk_per_trade_pct).quantize(CENTS)
    if risk_budget <= 0:
        return SizingResult(
            notional=Decimal("0"),
            approved=False,
            risk_budget=risk_budget,
            stop_distance_pct=stop_distance,
            rejection_reason=f"risk budget is {risk_budget} — account value too small to size",
        )

    # The core relationship: risking `risk_budget` across a `stop_distance`
    # move implies this much exposure.
    notional = risk_budget / stop_distance
    caps: list[str] = []
    binding: str | None = "risk_budget"

    def apply_cap(limit: Decimal, label: str) -> None:
        nonlocal notional, binding
        if limit < notional:
            caps.append(f"{label} capped {notional:.2f} -> {limit:.2f}")
            notional = limit
            binding = label

    # Conviction may shrink a position but never grow it beyond what the stop
    # already justified.
    effective_confidence = (
        signal.confidence if confidence_override is None else confidence_override
    )
    if scale_by_confidence and effective_confidence > 0:
        scaled = notional * Decimal(str(effective_confidence))
        if scaled < notional:
            caps.append(
                f"confidence {effective_confidence:.2f} scaled {notional:.2f} -> {scaled:.2f}"
            )
            notional = scaled
            binding = "confidence"

    apply_cap(account.total_value * config.max_position_pct, "max_position_pct")

    if config.max_order_notional is not None:
        apply_cap(config.max_order_notional, "max_order_notional")

    # Remaining room in this sector before the concentration cap. Enforced here
    # rather than only warned about in `risk.limits`, since the gate runs
    # before a notional exists.
    if sector:
        headroom = (
            account.total_value * config.max_sector_exposure_pct
            - account.sector_exposure(sector)
        )
        apply_cap(max(headroom, Decimal("0")), f"sector_exposure[{sector}]")

    # Buying power is the hard wall. On a cash account it already excludes
    # unsettled proceeds, so no separate subtraction is needed here — see the
    # settlement note emitted by `risk.limits`.
    apply_cap(account.buying_power, "buying_power")

    notional = notional.quantize(CENTS, rounding=ROUND_DOWN)

    if notional < config.min_order_notional:
        return SizingResult(
            notional=notional,
            approved=False,
            risk_budget=risk_budget,
            stop_distance_pct=stop_distance,
            binding_constraint=binding,
            caps_applied=caps,
            rejection_reason=(
                f"sized notional {notional:.2f} below minimum "
                f"{config.min_order_notional:.2f} (binding constraint: {binding})"
            ),
        )

    return SizingResult(
        notional=notional,
        approved=True,
        risk_budget=risk_budget,
        stop_distance_pct=stop_distance,
        binding_constraint=binding,
        caps_applied=caps,
    )


def exit_notional(account: AccountState, symbol: str, price: Decimal) -> Decimal:
    """Full value of an open position, for a complete exit.

    Partial exits are intentionally unsupported for now: scaling out needs a
    policy for how the remainder's stop moves, and a half-defined version of
    that is worse than none.
    """
    position = account.position_in(symbol)
    if position is None:
        return Decimal("0")
    return (position.quantity * price).quantize(CENTS, rounding=ROUND_DOWN)
