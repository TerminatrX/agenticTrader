"""Can a stored decision be understood without the session that made it?

The question this milestone answers is narrow and testable: given only
committed code, the pinned contract identities, and the journal, can a reviewer
reconstruct *the trading date, the execution mode, and the exact acquisition
contract* behind a past run? Three things stood between us and yes:

1. `scan_runs` recorded when a run started, not the date selection was seeded
   with. Those differ whenever a run crosses UTC midnight.
2. Audit rows recorded outcomes, from which mode was supposedly inferable. It
   is not: five of seven outcomes produce neither a trade nor a plan in *any*
   mode.
3. Nothing pinned what to request. Three workers reading one prose lookback
   fetched 30, 57, and 265 points for the same indicator.

The tests below are grouped by which of those they hold shut. Everything is
synthetic: the shared fixtures carry the suite's fake `TEST0000` account, and
no test here reads a payload or a balance.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.agents.orchestrator import run_cycle
from agentic_trader.config import AppConfig, StrategyConfig
from agentic_trader.journal import JournalRepository
from agentic_trader.journal.models import AuditEntry, CycleOutcome
from agentic_trader.market.acquisition import CURRENT_ACQUISITION
from agentic_trader.market.earnings_capabilities import ROBINHOOD_MCP_EARNINGS
from agentic_trader.models import (
    SELECTABLE_EXECUTION_MODES,
    ExecutionMode,
    ProtectionState,
)

DAY = date(2026, 8, 18)

# Pinned identity. A change to any request semantics must land here as a
# deliberate edit, in the same commit as the change that caused it.
ACQUISITION_REF = "agentic-acquisition@v4-2026-08-25"
ACQUISITION_FINGERPRINT = (
    "eceedacc620fe4e94e63fbf536d08e0eca28294c1de82d14a989f3bc780467c6"
)


def _columns(path, table: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _legacy_db(path, table: str, drop: set[str]) -> None:
    """Build a journal as it stood *before* a column existed.

    Derived from the repository's own SCHEMA with the named columns filtered
    out of one table, rather than hand-written. A hand-written fixture drifts
    the moment an unrelated column is added, and then proves nothing about the
    migration it claims to test.
    """
    from agentic_trader.journal.repository import SCHEMA

    out, inside = [], False
    for line in SCHEMA.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(f"CREATE TABLE IF NOT EXISTS {table.upper()} ("):
            inside = True
        elif inside and stripped.startswith(")"):
            inside = False
        elif inside and stripped.split(" ")[0] in drop:
            continue
        out.append(line)

    conn = sqlite3.connect(path)
    conn.executescript("\n".join(out))
    conn.commit()
    conn.close()


def _app_config(tmp_path, risk_config) -> AppConfig:
    return AppConfig(risk=risk_config, strategies=StrategyConfig(), project_root=tmp_path)


def _audit(**overrides) -> AuditEntry:
    base: dict = {
        "cycle_id": "cycle-1",
        "occurred_at": datetime(2026, 8, 18, 14, 0, tzinfo=UTC),
        "symbol": "AAPL",
        "strategy": "trend_pullback",
        "outcome": CycleOutcome.NO_SIGNAL,
        "mode": ExecutionMode.SHADOW,
        "trading_date": DAY,
        "acquisition_profile_ref": CURRENT_ACQUISITION.profile_ref,
        "acquisition_config_fingerprint": CURRENT_ACQUISITION.content_fingerprint,
    }
    return AuditEntry(**{**base, **overrides})


# ==========================================================================
# 1. The trading date, stored because it is an input
# ==========================================================================


def test_the_trading_date_survives_a_journal_round_trip(tmp_path):
    """(A) Stored exactly, not re-derived on the way out."""
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run("run-a", None, trading_date=DAY, coverage_status="complete")

    (run,) = repo.scan_runs()
    assert run["trading_date"] == "2026-08-18"
    assert date.fromisoformat(run["trading_date"]) == DAY


def test_an_aborted_run_still_records_its_trading_date(tmp_path):
    """(B) A run stopped by drift produces no batch at all, and that is exactly
    the run worth keeping. "We refused to discover, on this date" has to
    survive; a NULL here would make a refusal indistinguishable from a row
    written before the column existed."""
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run(
        "run-drift", None, trading_date=DAY,
        coverage_status="incomplete", aborted_reason="scan_definition_drift",
    )

    (run,) = repo.scan_runs()
    assert run["aborted_reason"] == "scan_definition_drift"
    assert run["trading_date"] == "2026-08-18"


def test_the_trading_date_cannot_be_omitted(tmp_path):
    """(R, for discovery) No default, so no caller can fall back to the wall
    clock without saying so."""
    repo = JournalRepository(tmp_path / "j.db")
    with pytest.raises(TypeError, match="trading_date"):
        repo.record_scan_run("run-x", None, coverage_status="complete")


def test_the_stored_date_is_not_the_start_timestamp(tmp_path):
    """The reason this column exists rather than being inferred.

    A discovery run launched at 01:30 UTC is still trading 18 August in New
    York. Deriving the date from `started_at` would file it under the 19th and
    silently mismatch the rotation seed that actually chose the candidates.
    """
    repo = JournalRepository(tmp_path / "j.db")
    after_utc_midnight = datetime(2026, 8, 19, 1, 30, tzinfo=UTC)
    repo.record_scan_run(
        "run-late", None, trading_date=DAY,
        started_at=after_utc_midnight, coverage_status="complete",
    )

    (run,) = repo.scan_runs()
    assert run["trading_date"] == "2026-08-18"
    assert run["started_at"].startswith("2026-08-19")
    assert run["trading_date"] != run["started_at"][:10]


def test_a_journal_predating_the_column_migrates_additively(tmp_path):
    """(C) The table shipped without this column. Recreating it would discard
    live history, so the column is added in place and old rows keep their
    honest NULL."""
    db = tmp_path / "old.db"
    _legacy_db(db, "scan_runs", {"trading_date"})
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO scan_runs (run_id, source, started_at, coverage_status)
           VALUES ('legacy-1', 'robinhood_scanner',
                   '2026-07-01T12:00:00+00:00', 'complete')"""
    )
    conn.commit()
    conn.close()
    assert "trading_date" not in _columns(db, "scan_runs"), "fixture must predate it"

    repo = JournalRepository(db)
    repo.record_scan_run("run-new", None, trading_date=DAY, coverage_status="complete")

    rows = {r["run_id"]: r for r in repo.scan_runs()}
    assert rows["legacy-1"]["trading_date"] is None, "must not invent a date"
    assert rows["run-new"]["trading_date"] == "2026-08-18"


# ==========================================================================
# 2. Execution mode, stored because it cannot be inferred
# ==========================================================================


@pytest.mark.parametrize(
    "outcome",
    [
        CycleOutcome.NO_SIGNAL,
        CycleOutcome.WATCH,
        CycleOutcome.REJECTED_BY_RISK,
        CycleOutcome.REJECTED_BY_CRITIC,
        CycleOutcome.SHADOW_FILLED,
        CycleOutcome.ERROR,
    ],
)
def test_every_outcome_persists_its_mode(tmp_path, outcome):
    """(D-H) Including the five that produce no trade and no plan — those are
    precisely the rows an inference cannot reach."""
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_audit(_audit(cycle_id=f"c-{outcome.value}", outcome=outcome))

    (row,) = repo.audit_for_cycle(f"c-{outcome.value}")
    assert row["outcome"] == outcome.value
    assert row["mode"] == "shadow"


def test_mode_cannot_be_omitted_from_an_audit_entry():
    """(R) Required with no default. A default would be an inference wearing a
    fact's clothes — and the inference is exactly what is unreliable."""
    with pytest.raises(Exception) as exc:
        AuditEntry(
            cycle_id="c", occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
            symbol="AAPL", strategy="trend_pullback", outcome=CycleOutcome.NO_SIGNAL,
        )
    assert "mode" in str(exc.value)


def test_an_exit_cycle_records_its_mode(tmp_path, bullish_pullback_snapshot,
                                        account, risk_config, held_position):
    """Exits are decisions too, and a shadow exit that read as live would
    misrepresent whether a real position was closed."""
    holding = account.model_copy(update={"positions": [held_position]})
    broken = bullish_pullback_snapshot.model_copy(
        update={
            "indicators": bullish_pullback_snapshot.indicators.model_copy(
                update={"sma_50": Decimal("999.00")}
            )
        }
    )
    result = run_cycle(
        broken, holding, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=broken.captured_at,
    )

    assert result.audit is not None
    assert result.audit.mode is ExecutionMode.SHADOW


def test_a_shadow_cycle_can_never_produce_a_live_audit(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(S) The invariant, asserted end to end rather than at the model."""
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=bullish_pullback_snapshot.captured_at,
    )

    assert result.outcome is CycleOutcome.SHADOW_FILLED
    assert result.audit is not None
    assert result.audit.mode is ExecutionMode.SHADOW
    assert result.trade is not None
    assert result.trade.mode is ExecutionMode.SHADOW
    assert result.should_submit is False


def test_audit_and_trade_agree_on_mode(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """One cycle must not be describable in two vocabularies. Both are typed
    `ExecutionMode` now; previously `TradeRecord.mode` was a bare str."""
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=bullish_pullback_snapshot.captured_at,
    )
    assert result.audit.mode == result.trade.mode
    assert isinstance(result.trade.mode, ExecutionMode)


def test_an_unimplemented_mode_is_refused_rather_than_downgraded(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """APPROVAL is declared in the domain so journalled history stays readable
    when it lands. Declaring it must not make it usable."""
    assert ExecutionMode.APPROVAL not in SELECTABLE_EXECUTION_MODES

    with pytest.raises(ValueError, match="declared but not"):
        run_cycle(
            bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
            mode=ExecutionMode.APPROVAL, trading_date=DAY,
            now=bullish_pullback_snapshot.captured_at,
        )


def test_an_unrecognized_mode_string_is_refused(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """A typo must not resolve to something safe-sounding. Silently reading
    'shaddow' as shadow would put a mode nobody chose into the journal."""
    with pytest.raises(ValueError):
        run_cycle(
            bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
            mode="shaddow", trading_date=DAY, now=bullish_pullback_snapshot.captured_at,
        )


def test_the_plan_mode_mapping_is_total_and_explicit():
    """Every selectable mode has a declared executor vocabulary, and nothing
    is chosen by elimination.

    The earlier form -- `"shadow" if mode is SHADOW else "live"` -- was
    fail-closed against the protection floor but fail-*open* against
    enablement: any mode added to SELECTABLE_EXECUTION_MODES would have
    received a live plan with no further edit.
    """
    from agentic_trader.agents.orchestrator import _PLAN_MODE

    assert _PLAN_MODE == {
        ExecutionMode.SHADOW: "shadow",
        ExecutionMode.LIVE: "live",
    }
    assert ExecutionMode.APPROVAL not in _PLAN_MODE
    assert set(SELECTABLE_EXECUTION_MODES) <= set(_PLAN_MODE), (
        "a mode may not be selectable without a declared plan vocabulary"
    )


def test_shadow_produces_a_shadow_plan_that_cannot_submit(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY,
        now=bullish_pullback_snapshot.captured_at,
    )
    assert result.plan.mode == "shadow"
    assert result.should_submit is False


def test_live_still_reaches_the_protection_floor(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """LIVE behaviour is unchanged by the new mapping: it takes the non-shadow
    branch, where an unprotectable fractional position is refused structurally.
    At this account size every position is fractional, so this is the ordinary
    outcome rather than an edge case."""
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.LIVE, trading_date=DAY,
        now=bullish_pullback_snapshot.captured_at,
    )
    assert result.outcome is CycleOutcome.REJECTED_BY_RISK
    assert result.plan is None
    assert result.should_submit is False
    assert any("protection" in e.lower() for e in result.errors)


def test_making_approval_selectable_alone_cannot_produce_a_live_plan(
    tmp_path, bullish_pullback_snapshot, account, risk_config, monkeypatch
):
    """The regression this mapping exists to prevent, simulated exactly.

    A future developer adds APPROVAL to the selectable set and changes nothing
    else. Under the old expression that produced a `mode="live"` ExecutionPlan
    immediately. Now it stops at the mapping, which is a second, independent
    decision -- enabling a mode and teaching the executor what it means must
    not be the same edit.
    """
    from agentic_trader.agents import orchestrator

    monkeypatch.setattr(
        orchestrator, "SELECTABLE_EXECUTION_MODES",
        frozenset({ExecutionMode.SHADOW, ExecutionMode.LIVE, ExecutionMode.APPROVAL}),
    )

    with pytest.raises(NotImplementedError, match="no executor vocabulary"):
        run_cycle(
            bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
            mode=ExecutionMode.APPROVAL, trading_date=DAY,
            now=bullish_pullback_snapshot.captured_at,
        )


def test_should_submit_requires_both_mode_readings_to_agree(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """`self.mode` is the context asked for; `plan.mode` is what the payload
    was built under. A disagreement between them -- a mapping bug, a
    hand-built result -- must resolve to "do not submit" rather than to
    whichever field a reader happens to check."""
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY,
        now=bullish_pullback_snapshot.captured_at,
    )
    assert result.critic is not None and result.critic.approved

    # A live-looking plan smuggled onto a shadow cycle.
    result.plan.mode = "live"
    assert result.mode is ExecutionMode.SHADOW
    assert result.should_submit is False, "plan.mode alone must not authorize"

    # And the converse: a live context whose plan was built as shadow.
    result.mode = ExecutionMode.LIVE
    result.plan.mode = "shadow"
    assert result.should_submit is False


def test_a_journal_predating_the_mode_column_migrates(tmp_path):
    """(C, for audit) Old rows read NULL rather than being back-filled with a
    mode nobody verified."""
    db = tmp_path / "old.db"
    dropped = {"mode", "acquisition_profile_ref", "acquisition_config_fingerprint"}
    _legacy_db(db, "audit", dropped)
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO audit (cycle_id, occurred_at, symbol, strategy, outcome)
           VALUES ('legacy', '2026-07-01T12:00:00+00:00', 'AAPL',
                   'trend_pullback', 'no_signal')"""
    )
    conn.commit()
    conn.close()
    assert not dropped & set(_columns(db, "audit")), "fixture must predate them"
    # The same column name exists on `trades` and must be untouched there.
    assert "mode" in _columns(db, "trades")

    repo = JournalRepository(db)
    repo.record_audit(_audit(cycle_id="modern"))

    (legacy,) = repo.audit_for_cycle("legacy")
    (modern,) = repo.audit_for_cycle("modern")
    assert legacy["mode"] is None
    assert legacy["acquisition_profile_ref"] is None
    assert modern["mode"] == "shadow"


# ==========================================================================
# 3. The acquisition contract
# ==========================================================================


def test_the_acquisition_profile_identity_is_pinned():
    """(I) The same trick risk.lock plays: hash the meaning, pin it in a test,
    and a change that skips the version bump fails loudly."""
    assert CURRENT_ACQUISITION.profile_ref == ACQUISITION_REF
    assert CURRENT_ACQUISITION.content_fingerprint == ACQUISITION_FINGERPRINT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval", "week"),
        ("bounds", "extended"),
        ("adjustment_type", "none"),
        ("lookback_calendar_days", 400),
        ("required_bars", 50),
        ("derivation_bars", 300),
    ],
)
def test_changing_historical_request_semantics_changes_the_fingerprint(field, value):
    """(L) Historicals matter doubly: with no `output` parameter the full
    series returns, and an omitted interval would let the server choose one
    from the range length."""
    altered = dataclasses.replace(
        CURRENT_ACQUISITION,
        historicals=dataclasses.replace(CURRENT_ACQUISITION.historicals, **{field: value}),
    )
    assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


@pytest.mark.parametrize("label", ["quote", "fundamentals", "earnings"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool", "some_other_tool"),
        ("symbol_param", "ticker"),
        # Differs from every current value, including fundamentals' own
        # ("bounds", "regular") -- so no case skips itself into passing.
        ("extra_params", (("bounds", "extended"),)),
    ],
)
def test_changing_a_single_call_spec_changes_the_fingerprint(label, field, value):
    """The un-ranged endpoints are part of the contract too. `symbol_param` in
    particular: the three do not agree on it, and getting it wrong produces a
    rejected call rather than a wrong number -- but it is still a request
    semantic, so it is pinned."""
    original = getattr(CURRENT_ACQUISITION, label)
    assert getattr(original, field) != value, "test would prove nothing"

    altered = dataclasses.replace(
        CURRENT_ACQUISITION, **{label: dataclasses.replace(original, **{field: value})}
    )
    assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


def test_fundamentals_is_part_of_the_contract():
    """It feeds two hard gates -- average_volume_30d for liquidity, sector for
    the exposure cap -- so specifying it only in skill prose left a decision
    input outside the contract the audit row names."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    fundamentals = plan["calls"]["fundamentals"]

    assert fundamentals["tool"] == "get_equity_fundamentals"
    assert fundamentals["params"] == {"symbols": ["AAPL"], "bounds": "regular"}

    joined = "|".join(CURRENT_ACQUISITION.fingerprint_items())
    assert "fundamentals.tool=get_equity_fundamentals" in joined
    assert "fundamentals.bounds=regular" in joined


def test_the_symbol_parameter_shape_matches_each_endpoint():
    """Two take a `symbols` array, one takes a scalar. Declared per endpoint
    rather than inferred from the label, which is what an earlier version did
    -- making the request shape depend on a string chosen for display."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)

    assert plan["calls"]["quote"]["params"]["symbols"] == ["AAPL"]
    assert plan["calls"]["fundamentals"]["params"]["symbols"] == ["AAPL"]
    assert plan["calls"]["earnings"]["params"]["symbol"] == "AAPL"
    assert "symbols" not in plan["calls"]["earnings"]["params"]


def test_the_end_time_policy_is_fingerprinted():
    altered = dataclasses.replace(CURRENT_ACQUISITION, end_time_policy="pinned_to_date")
    assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


def test_prose_notes_do_not_affect_the_fingerprint():
    """Explanations may be improved without invalidating a contract, the same
    rule `capability_items` applies to Capability.note."""
    altered = dataclasses.replace(
        CURRENT_ACQUISITION,
        earnings=dataclasses.replace(CURRENT_ACQUISITION.earnings, note="reworded"),
    )
    assert altered.content_fingerprint == CURRENT_ACQUISITION.content_fingerprint


def test_returned_market_values_cannot_reach_the_fingerprint():
    """(the negative half of I) The profile describes the request, so its
    identity must not move with the market. A fingerprint that changed with a
    price or a returned count would stop identifying a contract at all.

    Stated as the property rather than as a word search: the fingerprint is
    invariant under symbol and date, and carries no concrete ticker or
    response field. A blunt token list is tempting and wrong -- `symbol_param`
    is a *parameter name*, which is exactly the kind of request semantic that
    belongs in the hash.
    """
    from datetime import timedelta

    items = CURRENT_ACQUISITION.fingerprint_items()
    assert items, "a profile that hashes nothing pins nothing"

    # Invariance is the real claim. Nothing about a particular evaluation --
    # which symbol, which day -- may reach the contract identity.
    baseline = CURRENT_ACQUISITION.content_fingerprint
    for symbol in ("AAPL", "MSFT", "ZZZZ"):
        for day in (DAY, DAY + timedelta(days=97), date(2019, 1, 2)):
            CURRENT_ACQUISITION.request_plan(symbol, day)
            assert CURRENT_ACQUISITION.content_fingerprint == baseline

    joined = "|".join(items).lower()
    for ticker in ("aapl", "msft", "spy", "qqq"):
        assert ticker not in joined, f"a concrete ticker reached the contract: {ticker}"

    # Response fields, as opposed to request parameters. `value=` and
    # `captured` have no business here; `bounds=regular` and `period=14` do.
    for token in ("price", "value=", "close=", "last_trade", "captured", "returned_at"):
        assert token not in joined, f"{token!r} leaks response data into the contract"


# ------------------------------------------------------- the request itself


def test_the_request_spec_is_deterministic(tmp_path):
    """(M, N) Same symbol, same date, same profile — byte-identical requests.

    This is the property the milestone exists for. It held by accident before,
    right up until two workers read the same prose differently.
    """
    first = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    second = CURRENT_ACQUISITION.request_plan("aapl", DAY)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_different_date_moves_every_range():
    """The window follows the trading date, so replaying an old date rebuilds
    that day's request rather than today's."""
    old = CURRENT_ACQUISITION.request_plan("AAPL", date(2026, 1, 5))
    new = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    assert old["calls"]["historicals"]["params"]["start_time"] != (
        new["calls"]["historicals"]["params"]["start_time"]
    )


def test_the_generated_request_does_not_read_the_wall_clock():
    """(P) The *request* is wall-clock-independent: `start_time` derives from
    the trading date, and `end_time` is absent rather than an explicit "now"
    that would differ on every call.

    Scoped deliberately to the request. The broker's effective upper bound is
    request-time dependent by construction, so this says nothing about the
    response -- see the companion test below.
    """
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    for call in [plan["calls"]["historicals"], *plan["calls"]["indicators"].values()]:
        assert "end_time" not in call["params"]
        assert call["params"]["start_time"].endswith("T00:00:00Z")
    assert CURRENT_ACQUISITION.end_time_policy.startswith("omitted")

    # The un-ranged endpoints carry no time parameter at all, invented or
    # otherwise -- they have no window to choose.
    for label in ("quote", "fundamentals", "earnings"):
        params = plan["calls"][label]["params"]
        assert "start_time" not in params
        assert "end_time" not in params


def test_replay_rests_on_the_snapshot_not_on_the_request_being_reissuable():
    """The boundary of what the acquisition contract proves.

    Omitting `end_time` makes the request reproducible, not the response: the
    same call issued on two days can return different bars. Historical replay
    therefore reads the persisted snapshot, and the contract identity explains
    what that snapshot *is*. Neither substitutes for the other, and the policy
    string names request time rather than implying the stronger claim.
    """
    assert "request_time" in CURRENT_ACQUISITION.end_time_policy

    from agentic_trader.journal.models import AuditEntry as _AE

    required = {n for n, f in _AE.model_fields.items() if f.is_required()}
    assert {
        "acquisition_profile_ref",
        "acquisition_config_fingerprint",
        "trading_date",
    } <= required
    assert "snapshot_json" in _AE.model_fields


def test_the_interval_is_always_explicit():
    """Omitting it is not neutral: for historicals the server then picks an
    interval targeting ~2,500 bars across whatever range was asked for, so the
    lookback would silently decide the granularity."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    assert plan["calls"]["historicals"]["params"]["interval"] == "day"
    assert plan["calls"]["indicators"] == {}, "v4 requests no indicators"


def test_no_acquisition_call_can_touch_an_order():
    """(S) Every tool named is a market-data read. Worth asserting rather than
    assuming, because this output is handed to a worker as instructions."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    single = ("quote", "historicals", "fundamentals", "earnings")
    tools = {plan["calls"][k]["tool"] for k in single}
    tools |= {c["tool"] for c in plan["calls"]["indicators"].values()}

    assert tools == {
        "get_equity_quotes",
        "get_equity_historicals",
        "get_equity_fundamentals",
        "get_earnings_results",
    }, "v4 no longer calls the indicator endpoint"
    for tool in tools:
        assert tool.startswith("get_")
        for forbidden in ("order", "place", "cancel", "replace", "review"):
            assert forbidden not in tool


def test_the_earnings_call_is_the_symbol_scoped_one():
    """`get_earnings_calendar` takes no symbol argument. Substituting it once
    attributed one company's report date to another, live."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    earnings = plan["calls"]["earnings"]
    assert earnings["tool"] == ROBINHOOD_MCP_EARNINGS.source_tool == "get_earnings_results"
    assert earnings["params"] == {"symbol": "AAPL"}


def test_the_acquisition_profile_does_not_restate_earnings_semantics():
    """Which tool to call is an acquisition decision. What its answer *means*
    belongs to EarningsCapabilities and is versioned separately — folding its
    ref in here would force an acquisition bump every time an earnings claim
    was refined, and the snapshot already records that provenance."""
    joined = "|".join(CURRENT_ACQUISITION.fingerprint_items())
    assert ROBINHOOD_MCP_EARNINGS.profile_ref not in joined
    assert ROBINHOOD_MCP_EARNINGS.source_tool in joined


# --------------------------------------------------------- sufficiency (Phase 5)


# ==========================================================================
# 4. Reconstruction — the success criterion, end to end
# ==========================================================================


def test_a_stored_cycle_independently_identifies_its_own_context(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(O, and the milestone's stated criterion.)

    Persist a full shadow cycle, reload it from a *fresh* repository handle,
    and assert the row alone answers: which symbol, which date, which execution
    mode, which acquisition contract, which earnings contract, what was
    decided, and from what inputs.
    """
    db = tmp_path / "j.db"
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, cycle_id="reconstruct-1",
        now=bullish_pullback_snapshot.captured_at,
    )
    assert result.outcome is CycleOutcome.SHADOW_FILLED

    JournalRepository(db).record_audit(result.audit)
    JournalRepository(db).record_trade(result.trade)

    # A new handle: nothing in memory, only what SQLite holds.
    (row,) = JournalRepository(db).audit_for_cycle("reconstruct-1")

    assert row["symbol"] == "AAPL"
    assert row["mode"] == "shadow"
    assert row["outcome"] == "shadow_filled"
    assert row["acquisition_profile_ref"] == ACQUISITION_REF
    assert row["acquisition_config_fingerprint"] == ACQUISITION_FINGERPRINT

    snapshot = row["snapshot_json"]
    assert snapshot is not None
    assert snapshot["symbol"] == "AAPL"
    # Compared as an instant, not as text: the stored form uses `Z` where
    # `isoformat()` writes `+00:00`, and the reconstruction needs the moment.
    assert datetime.fromisoformat(snapshot["captured_at"]) == (
        bullish_pullback_snapshot.captured_at
    )
    # Earnings provenance travels with the snapshot rather than being restated
    # in the acquisition contract.
    assert snapshot["earnings"]["profile_ref"] == ROBINHOOD_MCP_EARNINGS.profile_ref
    assert snapshot["earnings"]["source"] == "get_earnings_results"

    assert row["risk_breaches"] == []
    assert row["reasons"]
    assert row["protection_state"] == ProtectionState.UNAVAILABLE.value


def test_a_reloaded_snapshot_replays_using_the_recorded_decision_time(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(Q, P) Replay the market-data side of a decision -- no broker, no wall
    clock -- given the same normalized account and config inputs.

    Scoped deliberately. The journal preserves the snapshot, trading date,
    acquisition identity, execution mode and decision timestamp; `AccountState`
    and `AppConfig` are still supplied from outside the row, and both are
    decision-significant. This is not a standalone event store, and calling it
    one would overstate what a stored row can answer.

    The clock the replay uses is `audit.occurred_at`, because that is the one
    the original cycle used. `run_cycle` threads `now` into the risk gate's
    `as_of`, the critic, and preflight quote-age and drift, so replaying under
    a different instant re-decides rather than reproduces.

    `snapshot.captured_at` is deliberately *not* substituted for it. The two
    answer different questions -- when the inputs were assembled versus when
    the decision was evaluated -- and this fixture holds them 45 seconds apart
    precisely so that using the wrong one could not silently pass.
    """
    from datetime import timedelta

    from agentic_trader.models import MarketSnapshot

    db = tmp_path / "j.db"
    config = _app_config(tmp_path, risk_config)

    # Decided slightly after the snapshot was assembled, as a real cycle is:
    # inside preflight's 120s quote-age bound, and on the same calendar day so
    # the earnings assessment stays current.
    decided_at = bullish_pullback_snapshot.captured_at + timedelta(seconds=45)
    assert decided_at != bullish_pullback_snapshot.captured_at

    original = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=DAY,
        cycle_id="replay-1", now=decided_at,
    )
    JournalRepository(db).record_audit(original.audit)

    # Everything below comes off the row. Nothing is carried in memory.
    (row,) = JournalRepository(db).audit_for_cycle("replay-1")
    rehydrated = MarketSnapshot.model_validate(row["snapshot_json"])
    occurred_at = datetime.fromisoformat(row["occurred_at"])

    assert occurred_at == decided_at
    assert occurred_at != rehydrated.captured_at, (
        "the row must distinguish decision time from capture time, or this "
        "test cannot tell which one the replay used"
    )

    replayed = run_cycle(
        rehydrated, account, config,
        mode=ExecutionMode(row["mode"]),
        trading_date=date.fromisoformat(row["trading_date"]),
        cycle_id="replay-2",
        now=occurred_at,
    )

    assert replayed.outcome is original.outcome
    assert replayed.signal.strength is original.signal.strength
    assert replayed.signal.confidence == original.signal.confidence
    assert replayed.signal.stop_price == original.signal.stop_price
    assert replayed.signal.target_price == original.signal.target_price
    assert replayed.risk_decision.approved_notional == (
        original.risk_decision.approved_notional
    )
    assert replayed.risk_decision.breached_limits == (
        original.risk_decision.breached_limits
    )
    assert replayed.plan.payload["ref_id"] == original.plan.payload["ref_id"]
    assert replayed.audit.mode is original.audit.mode
    assert replayed.audit.trading_date == original.audit.trading_date
    assert replayed.audit.acquisition_config_fingerprint == (
        original.audit.acquisition_config_fingerprint
    )


def test_the_recorded_decision_time_is_not_the_capture_time(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """Guards the fixture above. If a future change made `occurred_at` default
    to `captured_at`, the replay proof would still pass while silently no
    longer proving anything."""
    from datetime import timedelta

    decided_at = bullish_pullback_snapshot.captured_at + timedelta(seconds=45)
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=decided_at,
    )

    assert result.audit.occurred_at == decided_at
    assert result.audit.occurred_at != bullish_pullback_snapshot.captured_at
    # And neither is the trading date, which is a third distinct thing.
    assert result.audit.trading_date == DAY
    assert result.audit.trading_date != result.audit.occurred_at.date()


def test_replay_does_not_consult_the_current_default_profile(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(Q) A stored row names the contract *it* used. Loading it must not
    silently re-stamp it with whatever the current default happens to be —
    otherwise every historical decision would claim to have been made under
    today's contract."""
    db = tmp_path / "j.db"
    superseded = "agentic-acquisition@v0-2026-01-01"
    assert superseded != CURRENT_ACQUISITION.profile_ref

    JournalRepository(db).record_audit(
        _audit(
            cycle_id="old-contract",
            acquisition_profile_ref=superseded,
            acquisition_config_fingerprint="0" * 64,
        )
    )

    (row,) = JournalRepository(db).audit_for_cycle("old-contract")
    assert row["acquisition_profile_ref"] == superseded
    assert row["acquisition_config_fingerprint"] == "0" * 64


def test_two_contracts_are_distinguishable_in_the_journal(tmp_path):
    """The point of storing the fingerprint rather than only the ref: a
    contract edited without a version bump is still detectable."""
    db = tmp_path / "j.db"
    repo = JournalRepository(db)
    repo.record_audit(_audit(cycle_id="a"))
    repo.record_audit(
        _audit(cycle_id="b", acquisition_config_fingerprint="deadbeef" * 8)
    )

    (a,) = repo.audit_for_cycle("a")
    (b,) = repo.audit_for_cycle("b")
    assert a["acquisition_profile_ref"] == b["acquisition_profile_ref"]
    assert a["acquisition_config_fingerprint"] != b["acquisition_config_fingerprint"]


def test_the_cli_emits_the_pinned_contract(capsys):
    """(M) The seam a worker actually reads. Same identity as the module, so
    prose and code cannot drift apart."""
    from agentic_trader.cli import main

    assert main(["acquisition-spec", "AAPL", "--date", "2026-08-18"]) == 0
    out = json.loads(capsys.readouterr().out)

    assert out["acquisition_profile_ref"] == ACQUISITION_REF
    assert out["acquisition_config_fingerprint"] == ACQUISITION_FINGERPRINT
    assert out["symbols"]["AAPL"] == CURRENT_ACQUISITION.request_plan("AAPL", DAY)["calls"]


@pytest.mark.parametrize(
    "outcome", [CycleOutcome.NO_SIGNAL, CycleOutcome.REJECTED_BY_RISK]
)
def test_the_evaluate_output_reports_mode_without_a_plan(
    tmp_path, bullish_pullback_snapshot, account, risk_config, outcome
):
    """The reported result carries the context at cycle level, not only under
    `plan` -- the outcomes that produce no plan are precisely the ones whose
    mode could not otherwise be read off the output."""
    from agentic_trader.cli import _render_result

    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=bullish_pullback_snapshot.captured_at,
    )
    result.outcome = outcome
    result.plan = None

    rendered = _render_result(result, symbol_regime="uptrend")
    assert "plan" not in rendered
    assert rendered["mode"] == "shadow"
    assert rendered["acquisition_profile_ref"] == ACQUISITION_REF
    assert rendered["acquisition_config_fingerprint"] == ACQUISITION_FINGERPRINT


def test_the_cli_requires_an_explicit_date(capsys):
    """No ambient `date.today()` in the deterministic core, here either."""
    from agentic_trader.cli import main

    with pytest.raises(SystemExit):
        main(["acquisition-spec", "AAPL"])


# ==========================================================================
# 5. Nothing here enables live execution
# ==========================================================================


def test_this_milestone_enables_no_execution_mode(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(S) Stated as a test so it is checked rather than remembered."""
    assert set(SELECTABLE_EXECUTION_MODES) == {ExecutionMode.SHADOW, ExecutionMode.LIVE}

    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, trading_date=DAY, now=bullish_pullback_snapshot.captured_at,
    )
    assert result.should_submit is False
    assert result.plan.mode == "shadow"
    # Still unprotectable at this account size, and still allowed only in shadow.
    assert result.plan.protection is ProtectionState.UNAVAILABLE


def test_the_acquisition_module_performs_no_broker_io():
    """`src/` never touches the network. The acquisition profile *describes*
    calls; it must not be able to make one."""
    import agentic_trader.market.acquisition as module

    source = __import__("inspect").getsource(module)
    for banned in ("import requests", "import httpx", "urllib.request", "socket"):
        assert banned not in source


# ==========================================================================
# 6. The agent -> CLI boundary
#
# Everything above proves the core records provenance correctly. None of it
# proves the boundary *demands* provenance -- and the boundary is where a
# bundle fetched under an older contract would otherwise be evaluated and
# journalled as though it came from the current one. That is worse than no
# record at all: read back later, a false provenance claim is indistinguishable
# from a true one.
# ==========================================================================

from tests.test_pipeline import HISTORICALS, QUOTE, TEST_ACCOUNT  # noqa: E402

BUNDLE_DATE = "2026-08-13"

_OMIT = object()


def _bundle(**overrides) -> dict:
    """A minimal, complete evaluate bundle. Entirely synthetic."""
    bundle = {
        "symbol": "AAPL",
        "mode": "shadow",
        "strategy": "trend_pullback",
        "trading_date": BUNDLE_DATE,
        "acquisition_profile_ref": CURRENT_ACQUISITION.profile_ref,
        "acquisition_config_fingerprint": CURRENT_ACQUISITION.content_fingerprint,
        "account": {
            "account_number": TEST_ACCOUNT,
            "is_cash_account": True,
            "total_value": "100.00",
            "cash": "100.00",
            "buying_power": "100.00",
            "unsettled_funds": "0",
            "positions": [],
            "open_order_symbols": [],
            "realized_pnl_today": "0",
        },
        "payloads": {"quote": QUOTE, "historicals": HISTORICALS, "indicators": {}},
    }
    for key, value in overrides.items():
        if value is _OMIT:
            bundle.pop(key, None)
        else:
            bundle[key] = value
    return bundle


def _run_evaluate(tmp_path, bundle) -> int:
    from agentic_trader.cli import main

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    # No --project-root override: config lives in the repo, and the other CLI
    # tests resolve it the same way. The journal is redirected to tmp_path, and
    # these fixtures cannot trip the kill switch (realized_pnl_today is 0), so
    # nothing is written into the project tree.
    return main(["--db", str(tmp_path / "j.db"), "evaluate", "--input", str(path)])


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"acquisition_profile_ref": _OMIT}, "acquisition_profile_ref"),
        ({"acquisition_config_fingerprint": _OMIT}, "acquisition_config_fingerprint"),
        ({"acquisition_profile_ref": None}, "acquisition_profile_ref"),
        ({"acquisition_config_fingerprint": ""}, "acquisition_config_fingerprint"),
        (
            {"acquisition_profile_ref": "agentic-acquisition@v0-2020-01-01"},
            "contract in force",
        ),
        ({"acquisition_config_fingerprint": "0" * 64}, "without a version bump"),
        ({"trading_date": _OMIT}, "trading_date"),
        ({"trading_date": "not-a-date"}, "YYYY-MM-DD"),
    ],
)
def test_a_bundle_without_matching_provenance_is_refused(
    tmp_path, capsys, overrides, fragment
):
    """Missing, blank, stale, or wrong -- every one fails closed."""
    code = _run_evaluate(tmp_path, _bundle(**overrides))
    assert code != 0
    assert fragment in capsys.readouterr().out


def test_a_refused_bundle_writes_no_audit(tmp_path, capsys):
    """The half that matters. Refusing to evaluate is only useful if nothing
    is recorded: an audit row claiming the current contract for payloads
    fetched under another one is the exact falsehood being prevented."""
    db = tmp_path / "j.db"
    code = _run_evaluate(
        tmp_path, _bundle(acquisition_profile_ref="agentic-acquisition@v0-2020-01-01")
    )
    capsys.readouterr()
    assert code != 0
    assert not db.exists() or JournalRepository(db).recent_audit() == []


def test_a_matching_bundle_evaluates_and_records_the_supplied_identity(tmp_path, capsys):
    """The positive case, and the round trip: what the bundle declared is what
    the row carries."""
    db = tmp_path / "j.db"
    code = _run_evaluate(tmp_path, _bundle())
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert out["ok"] is True
    assert out["mode"] == "shadow"
    assert out["acquisition_profile_ref"] == ACQUISITION_REF
    assert out["acquisition_config_fingerprint"] == ACQUISITION_FINGERPRINT

    (row,) = JournalRepository(db).audit_for_cycle(out["cycle_id"])
    assert row["mode"] == "shadow"
    assert row["trading_date"] == BUNDLE_DATE
    assert row["acquisition_profile_ref"] == ACQUISITION_REF
    assert row["acquisition_config_fingerprint"] == ACQUISITION_FINGERPRINT


def test_the_stored_trading_date_is_the_bundle_value_not_the_snapshot_stamp(
    tmp_path, capsys
):
    """`captured_at` says when the snapshot was assembled; `trading_date` says
    what the ranged requests were built from. They answer different questions
    and coincide only by habit, so the declared value is stored verbatim."""
    db = tmp_path / "j.db"
    code = _run_evaluate(tmp_path, _bundle(trading_date="2026-03-02"))
    out = json.loads(capsys.readouterr().out)
    assert code == 0

    (row,) = JournalRepository(db).audit_for_cycle(out["cycle_id"])
    assert row["trading_date"] == "2026-03-02"
    assert row["snapshot_json"]["captured_at"][:10] != "2026-03-02", (
        "fixture must make the two differ, or this proves nothing"
    )


# ------------------------------------------------- required at construction


@pytest.mark.parametrize(
    "missing",
    ["mode", "trading_date", "acquisition_profile_ref", "acquisition_config_fingerprint"],
)
def test_a_new_audit_entry_cannot_omit_its_provenance(missing):
    """The columns are nullable because legacy rows genuinely lack these. The
    model describes a NEW write, where a default would let a caller skip the
    provenance and still have the record assert the current contract."""
    fields = {
        "cycle_id": "c",
        "occurred_at": datetime(2026, 8, 18, tzinfo=UTC),
        "symbol": "AAPL",
        "strategy": "trend_pullback",
        "outcome": CycleOutcome.NO_SIGNAL,
        "mode": ExecutionMode.SHADOW,
        "trading_date": DAY,
        "acquisition_profile_ref": CURRENT_ACQUISITION.profile_ref,
        "acquisition_config_fingerprint": CURRENT_ACQUISITION.content_fingerprint,
    }
    del fields[missing]

    with pytest.raises(Exception) as exc:
        AuditEntry(**fields)
    assert missing in str(exc.value)


def test_a_legacy_row_reads_back_without_fabricated_provenance(tmp_path):
    """Reading is not writing. Old rows return NULL through the repository
    rather than being handed today's contract on the way out."""
    db = tmp_path / "old.db"
    dropped = {
        "mode",
        "trading_date",
        "acquisition_profile_ref",
        "acquisition_config_fingerprint",
    }
    _legacy_db(db, "audit", dropped)
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO audit (cycle_id, occurred_at, symbol, strategy, outcome)
           VALUES ('legacy', '2026-07-01T12:00:00+00:00', 'AAPL',
                   'trend_pullback', 'no_signal')"""
    )
    conn.commit()
    conn.close()

    (row,) = JournalRepository(db).audit_for_cycle("legacy")
    assert row["mode"] is None
    assert row["trading_date"] is None
    assert row["acquisition_profile_ref"] is None
    assert row["acquisition_config_fingerprint"] is None


# ------------------------------------------------------ regenerating the plan


def test_a_stored_row_regenerates_the_request_plan_that_produced_it(tmp_path, capsys):
    """The success criterion, as an executable check.

    Profile plus symbol plus trading_date, all read back from the row, must
    rebuild the exact requests that fetched the inputs -- every `start_time`
    included.
    """
    from datetime import timedelta

    db = tmp_path / "j.db"
    code = _run_evaluate(tmp_path, _bundle())
    out = json.loads(capsys.readouterr().out)
    assert code == 0

    (row,) = JournalRepository(db).audit_for_cycle(out["cycle_id"])
    assert row["acquisition_profile_ref"] == CURRENT_ACQUISITION.profile_ref
    assert row["acquisition_config_fingerprint"] == (
        CURRENT_ACQUISITION.content_fingerprint
    )

    trading_date = date.fromisoformat(row["trading_date"])
    regenerated = CURRENT_ACQUISITION.request_plan(row["symbol"], trading_date)
    assert regenerated == CURRENT_ACQUISITION.request_plan(
        "AAPL", date.fromisoformat(BUNDLE_DATE)
    )

    # Spot-check the derived ranges rather than trusting equality alone.
    for spec in CURRENT_ACQUISITION.indicators:
        start = regenerated["calls"]["indicators"][spec.key]["params"]["start_time"]
        due = trading_date - timedelta(days=spec.lookback_calendar_days)
        assert start == f"{due.isoformat()}T00:00:00Z"


def test_regeneration_uses_the_stored_date_not_today(tmp_path, capsys):
    """If the row's date were ignored in favour of the wall clock, every
    historical decision would regenerate today's windows and the check above
    would pass vacuously."""
    db = tmp_path / "j.db"
    code = _run_evaluate(tmp_path, _bundle(trading_date="2026-03-02"))
    out = json.loads(capsys.readouterr().out)
    assert code == 0

    (row,) = JournalRepository(db).audit_for_cycle(out["cycle_id"])
    stored = CURRENT_ACQUISITION.request_plan(
        row["symbol"], date.fromisoformat(row["trading_date"])
    )
    today = CURRENT_ACQUISITION.request_plan(row["symbol"], date.today())

    assert row["trading_date"] == "2026-03-02"
    assert (
        stored["calls"]["historicals"]["params"]["start_time"]
        != today["calls"]["historicals"]["params"]["start_time"]
    )
