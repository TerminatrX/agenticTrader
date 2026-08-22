"""What the broker can tell us about a symbol's earnings, and how we know.

Kept separate from `BrokerCapabilities` because the two describe different
surfaces and move for different reasons: one is about placing orders, this one
is about a market-data lookup that feeds a hard gate. Folding an earnings claim
into the execution profile would bump its version every time a data endpoint
changed, and vice versa.

Every claim here was established by direct observation on 2026-08-21, recorded
in the docstrings below rather than remembered. The probe made no order call.

The finding that drove this module into existence:

    `get_earnings_calendar` accepts no symbol argument.

It is a market-wide window scan. Feeding its response to a per-symbol parser
produced a *cross-symbol attribution*: a snapshot for NVO carrying NVZMY's
earnings date, from a payload in which NVO did not appear at all. That is worse
than a missing gate, because the journal recorded confident evidence that was
about a different company.

`get_earnings_results` takes exactly one symbol and returns the trailing eight
quarters *including scheduled future ones*, which is what a blackout needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from agentic_trader.models.capabilities import (
    Capability,
    CapabilityProfile,
    Evidence,
    capability_items,
)

E = Evidence


@dataclass(frozen=True)
class EarningsCapabilities(CapabilityProfile):
    """The earnings source contract the blackout gate is allowed to rely on.

    `symbol_scoped` is the load-bearing one. A source that cannot be asked
    about a single symbol cannot answer the only question the gate asks, and
    the gate must refuse rather than approximate.
    """

    #: The tool a per-symbol assessment may be built from.
    source_tool: str

    symbol_scoped: Capability
    """Accepts one symbol and returns only that symbol's events."""

    returns_future_events: Capability
    """Scheduled, not-yet-reported reports appear alongside historical ones."""

    event_date_available: Capability
    """Every entry carries `report.date`."""

    event_session_available: Capability
    """`report.timing` distinguishes before-open from after-close."""

    event_timestamp_available: Capability
    """A true point in time. It is NOT available — see `date_semantics`."""

    verification_flag_available: Capability
    """`report.verified` marks a date as confirmed rather than penciled in."""

    unresolved_symbol_is_explicit: Capability
    """An unknown ticker is reported in `not_found`, not as an empty result."""

    future_event_absence_authoritative: Capability
    """Does "no future row" mean "no report is coming"?

    **Not established, and deliberately not assumed.** The endpoint demonstrably
    *can* return scheduled future events, but that is a different claim from
    "it always does when one exists". Proving the second needs a symbol with a
    known imminent report that the endpoint omits — evidence nobody has, since
    all twelve symbols probed on 2026-08-21 returned a future row and none
    exhibited the absent-future state at all.

    Until it is established, a symbol whose rows are all historical resolves to
    UNKNOWN rather than NONE_SCHEDULED. Absence of evidence is not evidence of
    absence, and this is a hard entry gate.
    """

    reported_marker_is_reliable: Capability
    """Whether `eps.actual` reliably distinguishes reported from upcoming.

    It does **not**, in both directions, which is why the normalizer ignores it
    when deciding whether an event is still ahead. See `EARNINGS_ACTUAL_NOTE`.
    """

    #: How to read `report.date`. Observed: bare `YYYY-MM-DD`, no time, no zone.
    date_semantics: str

    #: Observed vocabulary of `report.timing`. `None` occurs and means unknown.
    timing_vocabulary: tuple[str, ...]

    #: Trailing quarters returned. Bounds how far back evidence reaches.
    max_quarters_returned: int

    def fingerprint_items(self) -> tuple[str, ...]:
        return (
            *capability_items(self),
            f"source_tool={self.source_tool}",
            f"date_semantics={self.date_semantics}",
            f"timing_vocabulary={','.join(self.timing_vocabulary)}",
            f"max_quarters_returned={self.max_quarters_returned}",
        )

    @property
    def usable_for_blackout(self) -> bool:
        """May the hard gate rely on this source at all?

        Requires the three claims without which a blackout decision is not a
        decision: we can ask about one symbol, we can see events that have not
        happened yet, and each one carries a date. Anything less and the honest
        answer to "is this symbol inside its window?" is *unknown*.
        """
        return (
            self.symbol_scoped.usable
            and self.returns_future_events.usable
            and self.event_date_available.usable
        )


EARNINGS_ACTUAL_NOTE = (
    "eps.actual is unreliable as a reported/upcoming marker in both directions. "
    "FL returned three PAST-dated entries (2025-12-03, 2026-03-04, 2026-05-28) "
    "whose actual was never backfilled, and the market-wide calendar returned "
    "IDCBY dated 2026-08-28 -- a future date -- with actual populated and a "
    "year field of 2018. Pendingness is therefore derived from report.date "
    "alone, which can only over-block, never under-block."
)


ROBINHOOD_MCP_EARNINGS = EarningsCapabilities(
    profile_id="robinhood-mcp-earnings",
    version="2026-08-21.1",
    as_of=date(2026, 8, 21),
    source_tool="get_earnings_results",
    symbol_scoped=Capability(
        True, E.EMPIRICALLY_VERIFIED,
        "One symbol per call. NVO/PDD/NVDA/FL each returned only their own rows.",
    ),
    returns_future_events=Capability(
        True, E.EMPIRICALLY_VERIFIED,
        "All four probes carried a future-dated entry: NVO 2026-11-04, "
        "PDD 2026-08-24, NVDA 2026-08-26, FL 2026-08-26, against a probe date "
        "of 2026-08-21.",
    ),
    event_date_available=Capability(
        True, E.EMPIRICALLY_VERIFIED, "report.date present on every entry observed.",
    ),
    event_session_available=Capability(
        True, E.EMPIRICALLY_VERIFIED,
        "report.timing observed as am/pm, and null on some ADR entries.",
    ),
    event_timestamp_available=Capability(
        False, E.EMPIRICALLY_VERIFIED,
        "No time component and no timezone marker anywhere in 549 calendar "
        "entries or any per-symbol response. Only a calendar date exists.",
    ),
    verification_flag_available=Capability(
        True, E.EMPIRICALLY_VERIFIED,
        "report.verified is a real boolean. FL's upcoming date is false.",
    ),
    unresolved_symbol_is_explicit=Capability(
        True, E.EMPIRICALLY_VERIFIED,
        'ZZZZQQ returned {"results": [], "not_found": ["ZZZZQQ"]}.',
    ),
    future_event_absence_authoritative=Capability(
        None, E.UNKNOWN,
        "No basis either way. 12/12 symbols probed on 2026-08-21 returned a "
        "future-dated row, so the absent-future case was never observed and "
        "cannot be characterised. GOF returned results=[] with no not_found -- "
        "resolvable but carrying no earnings at all -- which is a further "
        "reminder that an empty answer has more than one cause.",
    ),
    reported_marker_is_reliable=Capability(
        False, E.EMPIRICALLY_VERIFIED, EARNINGS_ACTUAL_NOTE,
    ),
    date_semantics="bare_calendar_date_no_timezone",
    timing_vocabulary=("am", "pm"),
    max_quarters_returned=8,
)
"""The earnings contract in force.

`get_earnings_calendar` is deliberately absent. It has no symbol parameter, so
it cannot back a per-symbol gate at all — recording it here as an unusable
alternative would invite someone to reach for it.
"""
