"""What a broker can actually do, and how confidently we know it.

Two facts are tracked separately for every capability, because conflating them
is how a system ends up trusting a guess:

- **supported** — can the broker do this? `None` means unknown.
- **evidence** — how that was established, from a verified round trip down to
  an inference nobody has checked.

The distinction is load-bearing here. The capability driving this whole module —
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

import hashlib
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal
from enum import StrEnum


class Evidence(StrEnum):
    """How a capability claim was established, strongest first."""

    EMPIRICALLY_VERIFIED = "empirically_verified"
    """Observed directly against the broker. The only category that proves
    behaviour rather than describing it."""

    SCHEMA_DOCUMENTED = "schema_documented"
    """Stated in the MCP tool schema. Authoritative about intent, but the
    schema and the running system can disagree."""

    PUBLIC_DOCUMENTED = "public_documented"
    """Stated in the broker's public documentation, which may lag the API."""

    INFERRED = "inferred"
    """Deduced from adjacent facts. Reasoning, not a source."""

    UNKNOWN = "unknown"
    """No basis. Must be treated as unsupported wherever failing closed matters."""


@dataclass(frozen=True)
class Capability:
    """One thing a broker can or cannot do, plus the basis for saying so."""

    supported: bool | None
    evidence: Evidence
    note: str = ""

    @property
    def is_certain(self) -> bool:
        """True only for a claim actually observed against the broker."""
        return self.supported is not None and self.evidence is Evidence.EMPIRICALLY_VERIFIED

    @property
    def usable(self) -> bool:
        """Fail closed: unknown support is not permission.

        Deliberately not `supported is not False` — `None` must never read as a
        yes, and a capability nobody has established is exactly the case where
        an optimistic default does the most damage.
        """
        return self.supported is True

    def describe(self) -> str:
        state = {True: "supported", False: "not supported", None: "unknown"}[self.supported]
        return f"{state} ({self.evidence.value})" + (f": {self.note}" if self.note else "")


@dataclass(frozen=True)
class ProtectionFeasibility:
    """Whether a resting protective stop could be placed for a given quantity."""

    protectable: bool | None
    reason: str

    @property
    def certainly_impossible(self) -> bool:
        return self.protectable is False


@dataclass(frozen=True)
class BrokerCapabilities:
    """A broker's capability profile, as understood at `as_of`.

    Versioned on purpose. A decision made months ago was made against whatever
    was believed then, and a profile that silently mutates would make past
    journal entries unreadable — you could no longer tell whether a position
    went unprotected because protection was impossible or because the system
    did not yet know it was possible.
    """

    profile_id: str
    version: str
    as_of: date

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

    @property
    def profile_ref(self) -> str:
        """Stable identifier to store alongside a decision."""
        return f"{self.profile_id}@{self.version}"

    @property
    def content_fingerprint(self) -> str:
        """SHA-256 over every capability's support and evidence.

        A stored `profile_ref` is only worth keeping if it identifies exactly
        one set of claims. Nothing stops someone editing a capability in place
        and leaving `version` alone, which would silently repoint every
        historical decision at claims that were never used to make it — the
        journal would say `robinhood-mcp@2026-08-14` and mean something else.

        This is the same trick `config/risk.lock` plays on the risk values:
        hash the semantic content, pin it in a test, and a change that skips the
        version bump fails loudly instead of quietly rewriting history. `note`
        is excluded — prose may be improved without invalidating a claim.
        """
        parts = [
            f"{f.name}={cap.supported}:{cap.evidence.value}"
            for f in fields(self)
            if isinstance(cap := getattr(self, f.name), Capability)
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

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
