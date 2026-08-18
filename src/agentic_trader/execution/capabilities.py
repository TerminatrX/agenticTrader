"""What a broker can actually do at order time, and how confidently we know it.

The shared vocabulary — `Capability`, `Evidence`, `CapabilityProfile` — lives in
`models.capabilities`; this module holds only the order-execution profile. The
scanner has its own, in `universe.scanner_capabilities`, because filter
vocabularies and row shapes evolve independently of order semantics and neither
should force a version bump on the other.

The support/evidence split is load-bearing here. The capability driving this
whole module —
whether a fractional position can carry a resting protective stop — is
documented as unsupported in the MCP tool schema, but could not be confirmed
empirically. `review_equity_order` accepted a fractional `stop_market` sell with
no alerts, *and* accepted a short sale in an account holding none of the symbol,
which suggests it does not validate order-parameter legality at all rather than
that the restriction is absent. The only definitive test is placing a real
order, which this system will not do to satisfy curiosity.

So the restriction is honoured (`supported=False`) while the uncertainty is
recorded where it belongs (`evidence=SCHEMA_DOCUMENTED`) rather than being
flattened into a comment or, worse, into the protection state.

Nothing in the trading path may branch on a broker's *name*. Callers ask a
`BrokerCapabilities` instance, which is what makes a second brokerage a new
profile rather than a search for `if robinhood`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from agentic_trader.models.capabilities import (
    Capability,
    CapabilityProfile,
    Evidence,
    capability_items,
)


@dataclass(frozen=True)
class ProtectionFeasibility:
    """Whether a resting protective stop could be placed for a given quantity."""

    protectable: bool | None
    reason: str

    @property
    def certainly_impossible(self) -> bool:
        return self.protectable is False


@dataclass(frozen=True)
class BrokerCapabilities(CapabilityProfile):
    """A broker's order-execution capabilities, as understood at `as_of`.

    Versioned on purpose. A decision made months ago was made against whatever
    was believed then, and a profile that silently mutates would make past
    journal entries unreadable — you could no longer tell whether a position
    went unprotected because protection was impossible or because the system
    did not yet know it was possible.
    """

    market_orders: Capability
    limit_orders: Capability
    stop_market_orders: Capability
    stop_limit_orders: Capability

    fractional_market_orders: Capability
    fractional_stop_orders: Capability

    gtc_orders: Capability
    cancel_orders: Capability
    replace_orders: Capability
    bracket_orders: Capability

    stops_regular_hours_only: Capability

    def fingerprint_items(self) -> tuple[str, ...]:
        """Execution claims are all booleans, so support and evidence are all of it.

        Contrast the scanner profile, which also carries observed vocabularies
        and must fingerprint those too.
        """
        return capability_items(self)

    def protection_feasibility(self, quantity: Decimal) -> ProtectionFeasibility:
        """Can a resting stop be placed for exactly this quantity?

        The test is **integrality, not size**. A stop order carries a quantity
        like any other order, so 1.5 shares is a fractional order just as much
        as 0.5 is, and a `quantity >= 1` check would wave it through. Whole
        shares are the requirement, not "at least one share".
        """
        # Division produces full precision (0.03878660813732555854...), which is
        # unreadable in a warning a human is meant to act on. Six places is the
        # broker's own limit, so nothing meaningful is hidden.
        shown = quantity.quantize(Decimal("0.000001")) if quantity.is_finite() else quantity

        if quantity <= 0:
            return ProtectionFeasibility(
                protectable=False, reason=f"quantity {shown} is not positive"
            )

        if not self.stop_market_orders.usable:
            return ProtectionFeasibility(
                protectable=None if self.stop_market_orders.supported is None else False,
                reason=f"broker stop_market orders {self.stop_market_orders.describe()}",
            )

        if quantity == quantity.to_integral_value():
            return ProtectionFeasibility(
                protectable=True,
                reason=f"{shown} is a whole-share quantity; a resting stop can be placed",
            )

        if self.fractional_stop_orders.supported is None:
            return ProtectionFeasibility(
                protectable=None,
                reason=(
                    f"{shown} is fractional and fractional stop support is "
                    f"{self.fractional_stop_orders.describe()}"
                ),
            )

        if self.fractional_stop_orders.usable:
            return ProtectionFeasibility(
                protectable=True,
                reason=(
                    "broker accepts fractional stops "
                    f"({self.fractional_stop_orders.describe()})"
                ),
            )

        return ProtectionFeasibility(
            protectable=False,
            reason=(
                f"{shown} shares is fractional, and fractional quantities are "
                f"accepted only on type=market — a stop order cannot carry one "
                f"[{self.fractional_stop_orders.evidence.value}]"
            ),
        )


# The Robinhood MCP profile, read from the `place_equity_order` and
# `review_equity_order` tool schemas on 2026-08-14. Bump `version` and `as_of`
# whenever a claim here changes, so journalled decisions stay attributable to
# what was known at the time.
#
# Quoting the schema for the two that matter most:
#   "Fractional shares: only on type=market with market_hours=regular_hours,
#    eligible accounts, up to 6 decimal places, no short sells."
#   "Market and stop orders are regular_hours-only"
ROBINHOOD_MCP = BrokerCapabilities(
    profile_id="robinhood-mcp",
    version="2026-08-14",
    as_of=date(2026, 8, 14),
    market_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "type='market'"),
    limit_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "type='limit'"),
    stop_market_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "type='stop_market'"),
    stop_limit_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "type='stop_limit'"),
    fractional_market_orders=Capability(
        True,
        Evidence.SCHEMA_DOCUMENTED,
        "market + regular_hours only, up to 6dp, no short sells",
    ),
    fractional_stop_orders=Capability(
        False,
        Evidence.SCHEMA_DOCUMENTED,
        "fractional quantities are documented as market-only; review_equity_order "
        "neither confirmed nor denied this and appears not to validate order "
        "parameters, so the documented restriction stands unverified but honoured",
    ),
    gtc_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "time_in_force='gtc'"),
    cancel_orders=Capability(True, Evidence.SCHEMA_DOCUMENTED, "cancel_equity_order"),
    replace_orders=Capability(
        False,
        Evidence.SCHEMA_DOCUMENTED,
        "no replace/modify tool exists; moving a stop means cancel then re-place, "
        "which leaves the position unprotected in between",
    ),
    bracket_orders=Capability(
        False,
        Evidence.INFERRED,
        "no advanced-order placement tool is exposed and get_advanced_orders is "
        "absent from this MCP, so OCO/bracket orders cannot be placed or read",
    ),
    stops_regular_hours_only=Capability(
        True,
        Evidence.SCHEMA_DOCUMENTED,
        "stops placed outside regular hours queue for the next regular open",
    ),
)
