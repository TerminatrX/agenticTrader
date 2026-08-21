"""The earnings blackout, end to end: normalization, the gate, and replay.

Written after a live shadow run recorded confident earnings evidence that
belonged to a different company. `get_earnings_calendar` accepts no symbol
argument, and the old parser never read the `symbol` field on a row, so a
market-wide payload produced whichever report was nearest in the entire market.
NVO's snapshot carried NVZMY's date, from a payload NVO did not appear in.

Two properties are therefore load-bearing here, and each has tests that fail if
it regresses:

    identity   an event must be provably about the symbol being evaluated
    ignorance  not knowing must block an entry, never permit one
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.market.earnings_capabilities import ROBINHOOD_MCP_EARNINGS
from agentic_trader.market.snapshot import assess_earnings
from agentic_trader.models import EarningsAssessment, EarningsEvent, EarningsStatus
from agentic_trader.risk.limits import EARNINGS_UNKNOWN, check_limits

TODAY = date(2026, 8, 13)
PROFILE = ROBINHOOD_MCP_EARNINGS.profile_ref


def _row(symbol, report_date, *, timing="am", verified=True, actual=None, estimate="1.00"):
    return {
        "symbol": symbol,
        "year": 2026,
        "quarter": 3,
        "eps": {"estimate": estimate, "actual": actual},
        "report": {"date": report_date, "timing": timing, "verified": verified},
    }


def _payload(*rows, not_found=None):
    data = {"results": list(rows)}
    if not_found is not None:
        data["not_found"] = not_found
    return {"data": data}


def _assess(*rows, symbol="AAPL", as_of=TODAY, not_found=None):
    return assess_earnings(_payload(*rows, not_found=not_found), symbol, as_of)


def _gate(snapshot, signal, account, config, assessment, *, as_of=TODAY):
    return check_limits(
        signal,
        snapshot.model_copy(update={"earnings": assessment}),
        account,
        config,
        as_of=as_of,
    )


def _breached_on_earnings(result):
    return [b for b in result.breaches if "earnings" in b.lower()]


# --------------------------------------------------------------- A, C, F: window


@pytest.mark.parametrize(
    ("report_date", "blocked", "label"),
    [
        (date(2026, 8, 13), True, "same day"),
        (date(2026, 8, 14), True, "1 day out"),
        (date(2026, 8, 16), True, "exactly the 3-day boundary"),
        (date(2026, 8, 17), False, "one day past the window"),
        (date(2026, 9, 30), False, "well outside"),
    ],
)
def test_the_blackout_window_boundary_is_exact(
    entry_signal, bullish_pullback_snapshot, account, risk_config,
    report_date, blocked, label,
):
    """`earnings_blackout_days` is 3 and the comparison is inclusive on both
    ends: 0 <= days_out <= 3 blocks. The boundary day itself is inside."""
    assert risk_config.earnings_blackout_days == 3
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", report_date.isoformat())),
    )
    assert bool(_breached_on_earnings(result)) is blocked, label


# ------------------------------------------------------------- D, E: session timing


@pytest.mark.parametrize("timing", ["am", "pm", None])
def test_earnings_today_blocks_regardless_of_session(
    entry_signal, bullish_pullback_snapshot, account, risk_config, timing
):
    """Before-open, after-close, and unknown all block on the day.

    The broker publishes a bare calendar date with no time, so the system
    cannot place itself before or after an `am` report on the day it lands.
    Blocking is the only answer that is correct under both readings.
    """
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", "2026-08-13", timing=timing)),
    )
    assert _breached_on_earnings(result)


def test_the_session_is_recorded_in_the_breach_when_known(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", "2026-08-14", timing="pm")),
    )
    assert any("pm" in b for b in _breached_on_earnings(result))


# ------------------------------------------------------- G, H, M: ignorance blocks


def test_a_snapshot_with_no_assessment_blocks_the_entry(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The original bug. `if snapshot.earnings is not None` skipped the gate,
    so a missing payload read exactly like a clean calendar."""
    result = _gate(bullish_pullback_snapshot, entry_signal, account, risk_config, None)

    assert not result.passed
    assert any(EARNINGS_UNKNOWN in b for b in result.breaches)


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        (None, "missing or not an object"),
        ("not-json", "missing or not an object"),
        ({"data": {}}, "no results list"),
        ({"data": {"results": "nonsense"}}, "no results list"),
        ({"data": {"results": ["not-a-dict"]}}, "malformed entry"),
    ],
)
def test_missing_or_malformed_payloads_are_unknown(payload, fragment):
    assessment = assess_earnings(payload, "AAPL", TODAY)

    assert assessment.status is EarningsStatus.UNKNOWN
    assert fragment in assessment.reason


def test_an_unparseable_date_invalidates_the_whole_assessment():
    """Skipping the bad row and trusting the rest would be the dangerous
    choice: the unreadable row could be the one inside the window."""
    assessment = _assess(
        _row("AAPL", "2026-12-01"),
        _row("AAPL", "not-a-date"),
    )
    assert assessment.status is EarningsStatus.UNKNOWN
    assert "unparseable report date" in assessment.reason


def test_stale_evidence_cannot_clear_a_blackout(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """An assessment made for a different trading date says nothing about
    today — earnings dates move, and a stale all-clear is not an all-clear."""
    yesterday = _assess(_row("AAPL", "2026-09-30"), as_of=date(2026, 8, 12))
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config, yesterday
    )

    assert not result.passed
    assert any("stale evidence" in b for b in result.breaches)


# ------------------------------------------------- I: absence is not an all-clear


def test_a_symbol_absent_from_a_market_wide_response_is_unknown():
    """The exact live failure. NVO's payload held 233 rows and none were NVO's;
    the old parser returned NVZMY's date as NVO's."""
    assessment = _assess(
        _row("NVZMY", "2026-08-25"),
        _row("NBP", "2026-08-20"),
        symbol="NVO",
    )
    assert assessment.status is EarningsStatus.UNKNOWN
    assert "NVO absent" in assessment.reason
    assert assessment.event is None


def test_another_symbols_event_can_never_reach_the_gate(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Defence in depth: even a well-formed assessment about the wrong company
    is refused at the gate rather than silently applied."""
    wrong = EarningsAssessment(
        symbol="MSFT", status=EarningsStatus.NONE_SCHEDULED, as_of=TODAY,
        source="get_earnings_results", profile_ref=PROFILE,
    )
    result = _gate(bullish_pullback_snapshot, entry_signal, account, risk_config, wrong)

    assert not result.passed
    assert any("is for MSFT, not AAPL" in b for b in result.breaches)


def test_an_unresolved_symbol_is_unknown_not_clear():
    assessment = _assess(symbol="ZZZZQQ", not_found=["ZZZZQQ"])

    assert assessment.status is EarningsStatus.UNKNOWN
    assert "could not resolve" in assessment.reason


# ------------------------------------------- J: authoritative absence may pass


def test_resolved_symbol_with_only_past_reports_is_authoritatively_clear(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The one case where silence means something: the source returned this
    symbol's rows and every one is behind us."""
    assessment = _assess(
        _row("AAPL", "2026-05-01", actual="1.50"),
        _row("AAPL", "2026-08-01", actual="1.60"),
    )
    assert assessment.status is EarningsStatus.NONE_SCHEDULED
    assert assessment.event is None

    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config, assessment
    )
    assert not _breached_on_earnings(result)


# ----------------------------------------------------------- K: multiple events


def test_the_nearest_future_event_wins():
    assessment = _assess(
        _row("AAPL", "2026-11-04"),
        _row("AAPL", "2026-08-20"),
        _row("AAPL", "2026-09-15"),
    )
    assert assessment.event.report_date == date(2026, 8, 20)


def test_duplicate_dates_resolve_deterministically():
    """A symbol can carry two rows for one date. Whichever is chosen, it must
    be the same one on every replay."""
    rows = [
        _row("AAPL", "2026-08-20", timing="pm", verified=False),
        _row("AAPL", "2026-08-20", timing="am", verified=True),
    ]
    first = _assess(*rows)
    second = _assess(*reversed(rows))

    assert first.event == second.event


def test_past_dated_rows_never_become_the_nearest_event():
    assessment = _assess(
        _row("AAPL", "2026-08-01", actual=None),   # past, actual never backfilled
        _row("AAPL", "2026-09-15"),
    )
    assert assessment.event.report_date == date(2026, 9, 15)


def test_pendingness_ignores_eps_actual(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Observed live: the calendar returned IDCBY dated a week ahead with
    `actual` already populated, and FL carried three past dates whose `actual`
    was never filled in. The field is unreliable in both directions, so the
    date decides — which can only over-block."""
    assessment = _assess(_row("AAPL", "2026-08-14", actual="1.23"))

    assert assessment.status is EarningsStatus.UPCOMING
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config, assessment
    )
    assert _breached_on_earnings(result)


# ------------------------------------------------------------------ L: exits


@pytest.mark.parametrize("assessment", [None, "unknown_payload"])
def test_the_blackout_never_blocks_a_risk_reducing_exit(
    bullish_pullback_snapshot, account, risk_config, held_position, assessment
):
    """A gate that prevents closing a position increases risk. Exits return
    before the event-risk block entirely, so neither a missing assessment nor
    an UNKNOWN one can strand a position."""
    from agentic_trader.models import Side, Signal, SignalStrength

    earnings = None if assessment is None else assess_earnings(None, "AAPL", TODAY)
    if earnings is not None:
        assert earnings.status is EarningsStatus.UNKNOWN

    holding = account.model_copy(update={"positions": [held_position]})
    exit_signal = Signal(
        symbol="AAPL",
        strategy="trend_pullback",
        strength=SignalStrength.EXIT,
        side=Side.SELL,
        confidence=0.9,
        reference_price=Decimal("302.25"),
        reasons=["lost the 50-day"],
    )
    result = check_limits(
        exit_signal,
        bullish_pullback_snapshot.model_copy(update={"earnings": earnings}),
        holding,
        risk_config,
        as_of=TODAY,
    )
    assert not any(EARNINGS_UNKNOWN in b for b in result.breaches)
    assert not any("earnings" in b.lower() for b in result.breaches)


# ------------------------------------------------------------------ N: replay


def test_identical_inputs_replay_to_an_identical_assessment():
    payload = _payload(_row("AAPL", "2026-08-20"), _row("AAPL", "2026-11-04"))

    first = assess_earnings(payload, "AAPL", TODAY)
    second = assess_earnings(payload, "AAPL", TODAY)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_the_assessment_carries_its_own_provenance():
    """A reviewer must be able to answer "why was this blocked?" from the
    stored record alone, without calling the broker again."""
    assessment = _assess(_row("AAPL", "2026-08-14", timing="pm", verified=False))

    assert assessment.symbol == "AAPL"
    assert assessment.as_of == TODAY
    assert assessment.source == "get_earnings_results"
    assert assessment.profile_ref == PROFILE
    assert assessment.event.timing == "pm"
    assert assessment.event.verified is False


# ---------------------------------------------------- O, P: nothing may override


def test_the_critic_cannot_clear_an_earnings_breach(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The critic returns a non-positive confidence adjustment and never
    touches `LimitCheck`. There is no path from a model's opinion to a cleared
    breach, and this asserts the breach survives the harshest verdict."""
    from agentic_trader.agents.critic import CriticReport

    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", "2026-08-14")),
    )
    assert not result.passed

    report = CriticReport()
    report.concern("earnings imminent", 0.9)
    assert report.confidence_adjustment <= 0
    # The breach is a property of LimitCheck; the critic has no handle on it.
    assert not hasattr(report, "breaches")
    assert not result.passed


def test_scanner_source_values_cannot_influence_the_assessment():
    """Scanner columns are discovery diagnostics. Even a row shaped like an
    earnings hint must not reach the gate — only the authoritative payload's
    `symbol`/`report` fields are read."""
    contaminated = {"data": {"results": [
        {**_row("AAPL", "2026-11-04"),
         "columns": {"Symbol": "AAPL", "RSI": "40", "Earnings": "2026-08-14"},
         "source_values": {"earnings_date": "2026-08-14"}},
    ]}}
    assessment = assess_earnings(contaminated, "AAPL", TODAY)

    assert assessment.event.report_date == date(2026, 11, 4)


# ------------------------------------------------------------- capability pin


def test_the_earnings_capability_fingerprint_is_pinned():
    """Same contract as the other profiles: claims cannot move without the
    version moving with them."""
    assert ROBINHOOD_MCP_EARNINGS.profile_ref == "robinhood-mcp-earnings@2026-08-21"
    assert ROBINHOOD_MCP_EARNINGS.content_fingerprint == (
        "965ccb88631313ef7a6d2f7378aceb042f13bd950e6c5d3bf03899ec73f4ebc1"
    )


def test_the_calendar_endpoint_is_not_an_accepted_source():
    """`get_earnings_calendar` has no symbol parameter, so it can never back a
    per-symbol gate. The profile names only the tool that can."""
    assert ROBINHOOD_MCP_EARNINGS.source_tool == "get_earnings_results"
    assert ROBINHOOD_MCP_EARNINGS.symbol_scoped.usable
    assert ROBINHOOD_MCP_EARNINGS.usable_for_blackout


def test_a_source_that_cannot_answer_per_symbol_yields_unknown():
    """If the capability is ever downgraded, the gate must go dark rather than
    fall back to something approximate."""
    import dataclasses

    from agentic_trader.models.capabilities import Capability, Evidence

    crippled = dataclasses.replace(
        ROBINHOOD_MCP_EARNINGS,
        symbol_scoped=Capability(False, Evidence.EMPIRICALLY_VERIFIED, "market-wide only"),
    )
    assert not crippled.usable_for_blackout

    assessment = assess_earnings(
        _payload(_row("AAPL", "2026-11-04")), "AAPL", TODAY, capabilities=crippled
    )
    assert assessment.status is EarningsStatus.UNKNOWN
    assert "not capable" in assessment.reason


def test_an_unverified_date_just_outside_the_window_is_warned_not_blocked(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """A tentative date near the window can still move into it. Recording that
    is a bug fix; widening the window for it would be a policy change."""
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", "2026-08-18", verified=False)),  # 5d out
    )
    assert not _breached_on_earnings(result)
    assert any("unverified" in w for w in result.warnings)


def test_a_distant_unverified_date_does_not_warn(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Observed live: the quarter after next is unverified for nearly every
    symbol -- NVO, SPXC and EVRG were all 69-76 days out and unconfirmed. A
    warning that fires on almost every candidate is one nobody reads."""
    result = _gate(
        bullish_pullback_snapshot, entry_signal, account, risk_config,
        _assess(_row("AAPL", "2026-10-29", verified=False)),
    )
    assert not _breached_on_earnings(result)
    assert not any("unverified" in w for w in result.warnings)


def test_earnings_event_requires_a_symbol():
    """The field that would have prevented the whole class of bug."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EarningsEvent(report_date=date(2026, 8, 20))  # type: ignore[call-arg]


def test_eps_estimate_is_preserved_as_a_decimal():
    assessment = _assess(_row("AAPL", "2026-11-04", estimate="2.07"))
    assert assessment.event.eps_estimate == Decimal("2.07")


def test_captured_snapshots_serialise_the_assessment(bullish_pullback_snapshot):
    """Evidence has to survive into `snapshot_json` for replay to mean
    anything."""
    dumped = bullish_pullback_snapshot.model_dump(mode="json")

    assert dumped["earnings"]["symbol"] == "AAPL"
    assert dumped["earnings"]["status"] == "upcoming"
    assert dumped["earnings"]["source"] == "get_earnings_results"
    assert dumped["earnings"]["event"]["report_date"] == "2026-10-29"
    assert datetime.now(UTC) is not None  # import guard
