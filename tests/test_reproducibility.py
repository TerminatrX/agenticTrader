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
ACQUISITION_REF = "agentic-acquisition@v1-2026-08-22"
ACQUISITION_FINGERPRINT = (
    "de87e98f6bc5db437130e2fb7f7a82d8ba614700a66ba80dc47717b518734579"
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
        mode=ExecutionMode.SHADOW, now=broken.captured_at,
    )

    assert result.audit is not None
    assert result.audit.mode is ExecutionMode.SHADOW


def test_a_shadow_cycle_can_never_produce_a_live_audit(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(S) The invariant, asserted end to end rather than at the model."""
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode=ExecutionMode.SHADOW, now=bullish_pullback_snapshot.captured_at,
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
        mode=ExecutionMode.SHADOW, now=bullish_pullback_snapshot.captured_at,
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
            mode=ExecutionMode.APPROVAL, now=bullish_pullback_snapshot.captured_at,
        )


def test_an_unrecognized_mode_string_is_refused(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """A typo must not resolve to something safe-sounding. Silently reading
    'shaddow' as shadow would put a mode nobody chose into the journal."""
    with pytest.raises(ValueError):
        run_cycle(
            bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
            mode="shaddow", now=bullish_pullback_snapshot.captured_at,
        )


def test_a_non_shadow_mode_takes_the_structurally_gated_branch():
    """The mapping from ExecutionMode to the executor's own vocabulary is
    written as "shadow, or else live" rather than "live, or else shadow".

    The difference is not cosmetic: a mode added later and matched by name
    would fall through to "shadow" and past the protection floor, which is the
    one check no config can reach.
    """
    import inspect

    from agentic_trader.agents import orchestrator

    source = inspect.getsource(orchestrator.run_cycle)
    assert '"shadow" if execution_mode is ExecutionMode.SHADOW else "live"' in source


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
        ("period", 21),                 # (J) RSI length
        ("bounds", "extended"),         # (K) session semantics
        ("interval", "week"),           # (L) request granularity
        ("lookback_calendar_days", 60), # (L) request span
        ("output", "latest"),           # (K) how many points come back
        ("adjustment_type", "all"),     # corporate-action convention
        ("key", "rsi_alt"),             # which snapshot slot it fills
    ],
)
def test_changing_any_request_semantic_changes_the_fingerprint(field, value):
    """(J, K, L) Every field that alters what is asked for, or what the answer
    means, is inside the hash."""
    original = CURRENT_ACQUISITION.spec_for("rsi")
    assert getattr(original, field) != value, "test would prove nothing"

    mutated = dataclasses.replace(original, **{field: value})
    altered = dataclasses.replace(
        CURRENT_ACQUISITION,
        indicators=tuple(
            mutated if s.key == "rsi" else s for s in CURRENT_ACQUISITION.indicators
        ),
    )
    assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


def test_changing_the_derived_point_requirement_changes_the_fingerprint():
    """The endpoint accepts no point count, so these three are a *derivation*
    rather than a parameter — but they decide the pinned lookback, so a change
    to them is a change to the contract."""
    for field in ("warmup_bars", "required_output_bars", "convergence_bars"):
        mutated = dataclasses.replace(
            CURRENT_ACQUISITION.spec_for("macd"),
            **{field: getattr(CURRENT_ACQUISITION.spec_for("macd"), field) + 1},
        )
        altered = dataclasses.replace(
            CURRENT_ACQUISITION,
            indicators=tuple(
                mutated if s.key == "macd" else s for s in CURRENT_ACQUISITION.indicators
            ),
        )
        assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint, field


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval", "week"),
        ("bounds", "extended"),
        ("adjustment_type", "none"),
        ("lookback_calendar_days", 400),
        ("required_bars", 50),
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


def test_the_end_time_policy_is_fingerprinted():
    altered = dataclasses.replace(CURRENT_ACQUISITION, end_time_policy="pinned_to_date")
    assert altered.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


def test_adding_or_removing_an_indicator_changes_the_fingerprint():
    fewer = dataclasses.replace(
        CURRENT_ACQUISITION,
        indicators=tuple(s for s in CURRENT_ACQUISITION.indicators if s.key != "atr"),
    )
    assert fewer.content_fingerprint != CURRENT_ACQUISITION.content_fingerprint


def test_reordering_the_indicator_declarations_does_not_change_the_fingerprint():
    """Declaration order is not contract. Sorting before hashing keeps a
    cosmetic edit from reading as a retune — and a retune from hiding as one."""
    shuffled = dataclasses.replace(
        CURRENT_ACQUISITION, indicators=tuple(reversed(CURRENT_ACQUISITION.indicators))
    )
    assert shuffled.content_fingerprint == CURRENT_ACQUISITION.content_fingerprint


def test_prose_notes_do_not_affect_the_fingerprint():
    """Explanations may be improved without invalidating a contract, the same
    rule `capability_items` applies to Capability.note."""
    altered = dataclasses.replace(
        CURRENT_ACQUISITION,
        earnings=dataclasses.replace(CURRENT_ACQUISITION.earnings, note="reworded"),
    )
    assert altered.content_fingerprint == CURRENT_ACQUISITION.content_fingerprint


def test_returned_market_values_cannot_reach_the_fingerprint():
    """(the negative half of I) The profile describes the request. Nothing it
    hashes is a price, a timestamp, a symbol, or a count of what came back —
    otherwise a stored fingerprint would change with the market and stop
    identifying a contract at all."""
    items = CURRENT_ACQUISITION.fingerprint_items()
    assert items, "a profile that hashes nothing pins nothing"

    forbidden = ("price", "value=", "close", "volume", "as_of", "captured", "symbol")
    for item in items:
        lowered = item.lower()
        for token in forbidden:
            assert token not in lowered, f"{item!r} leaks response data into the contract"

    # And it is stable across evaluations, which a market-dependent hash is not.
    assert CURRENT_ACQUISITION.content_fingerprint == CURRENT_ACQUISITION.content_fingerprint


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
    for key in CURRENT_ACQUISITION.indicator_keys:
        assert (
            old["calls"]["indicators"][key]["params"]["start_time"]
            != new["calls"]["indicators"][key]["params"]["start_time"]
        )


def test_the_request_spec_does_not_read_the_wall_clock():
    """(P) `start_time` derives from the trading date; `end_time` is omitted
    entirely. Absent is more deterministic than an explicit "now" — an explicit
    now would make the same inputs produce a different request every call."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    for call in [plan["calls"]["historicals"], *plan["calls"]["indicators"].values()]:
        assert "end_time" not in call["params"]
        assert call["params"]["start_time"].endswith("T00:00:00Z")
    assert CURRENT_ACQUISITION.end_time_policy.startswith("omitted")


def test_start_times_are_the_pinned_lookback_before_the_trading_date():
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    for spec in CURRENT_ACQUISITION.indicators:
        stamp = plan["calls"]["indicators"][spec.key]["params"]["start_time"]
        expected = DAY - __import__("datetime").timedelta(days=spec.lookback_calendar_days)
        assert stamp == f"{expected.isoformat()}T00:00:00Z"


def test_every_request_names_only_parameters_its_endpoint_accepts():
    """The schema rejects a parameter the chosen indicator type does not take,
    so `period` must not ride along on a MACD request and vice versa."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)

    macd = plan["calls"]["indicators"]["macd"]["params"]
    assert "period" not in macd
    assert {"fast_period", "slow_period", "signal_period"} <= macd.keys()

    for key in ("rsi", "sma_20", "sma_50", "sma_200", "atr"):
        params = plan["calls"]["indicators"][key]["params"]
        assert "period" in params
        assert not {"fast_period", "slow_period", "signal_period"} & params.keys()
        assert {"symbol", "type", "interval", "start_time"} <= params.keys()


def test_the_interval_is_always_explicit():
    """Omitting it is not neutral: for historicals the server then picks an
    interval targeting ~2,500 bars across whatever range was asked for, so the
    lookback would silently decide the granularity."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    assert plan["calls"]["historicals"]["params"]["interval"] == "day"
    for key in CURRENT_ACQUISITION.indicator_keys:
        assert plan["calls"]["indicators"][key]["params"]["interval"] == "day"


def test_no_acquisition_call_can_touch_an_order():
    """(S) Every tool named is a market-data read. Worth asserting rather than
    assuming, because this output is handed to a worker as instructions."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", DAY)
    tools = {plan["calls"][k]["tool"] for k in ("quote", "historicals", "earnings")}
    tools |= {c["tool"] for c in plan["calls"]["indicators"].values()}

    assert tools == {
        "get_equity_quotes",
        "get_equity_historicals",
        "get_earnings_results",
        "get_equity_technical_indicators",
    }
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


def test_every_pinned_lookback_covers_its_derived_requirement():
    """The numbers are derived, not chosen. Each lookback must contain the
    indicator's warm-up, its convergence allowance, and the trailing points the
    strategy actually reads — with the trading-day conversion and a holiday
    buffer on top."""
    hist = CURRENT_ACQUISITION.historicals
    assert hist.lookback_calendar_days >= hist.minimum_calendar_days()

    for spec in CURRENT_ACQUISITION.indicators:
        assert spec.lookback_calendar_days >= spec.minimum_calendar_days(), spec.key


def test_the_strategy_reads_two_points_from_exactly_rsi_and_macd():
    """Tied to what the core actually consumes, so a strategy change that needs
    a third point fails here instead of silently reading a value that was never
    fetched."""
    two_point = {s.key for s in CURRENT_ACQUISITION.indicators if s.required_output_bars >= 2}
    assert two_point == {"rsi", "macd"}
    for key in two_point:
        assert CURRENT_ACQUISITION.spec_for(key).output == "last:2"
    for key in ("sma_20", "sma_50", "sma_200", "atr"):
        assert CURRENT_ACQUISITION.spec_for(key).output == "latest"


def test_recursive_smoothers_carry_a_convergence_allowance_and_sma_does_not():
    """RSI, ATR and MACD each depend on the previous value back to a seed, so a
    short range returns a genuinely different number. A simple moving average
    is a finite window and is unaffected — which is why an unpinned range looks
    harmless until it isn't."""
    for key in ("rsi", "atr", "macd"):
        assert CURRENT_ACQUISITION.spec_for(key).convergence_bars > 0, key
    for key in ("sma_20", "sma_50", "sma_200"):
        assert CURRENT_ACQUISITION.spec_for(key).convergence_bars == 0, key


def test_the_sma_200_window_is_the_longest():
    """200 bars of warm-up before a first value exists — the binding
    constraint, and the one a 90-day range would silently fail to satisfy."""
    longest = max(CURRENT_ACQUISITION.indicators, key=lambda s: s.lookback_calendar_days)
    assert longest.key == "sma_200"
    assert CURRENT_ACQUISITION.spec_for("sma_200").warmup_bars == 200


def test_the_indicator_keys_match_what_the_snapshot_parser_expects():
    """A response filed under the wrong key is a silent misattribution: an
    sma_50 landing in the sma_200 slot would still parse."""
    from agentic_trader.market.snapshot import parse_indicators

    payloads = dict.fromkeys(CURRENT_ACQUISITION.indicator_keys)
    parse_indicators(payloads)  # must not raise on the exact key set

    assert set(CURRENT_ACQUISITION.indicator_keys) == {
        "rsi", "macd", "sma_20", "sma_50", "sma_200", "atr",
    }


def test_an_unknown_indicator_key_is_refused():
    with pytest.raises(KeyError, match="sma_100"):
        CURRENT_ACQUISITION.spec_for("sma_100")


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
        mode=ExecutionMode.SHADOW, cycle_id="reconstruct-1",
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


def test_a_reloaded_snapshot_replays_to_the_same_decision(
    tmp_path, bullish_pullback_snapshot, account, risk_config
):
    """(Q, P) Replay from the persisted snapshot alone — no broker, no wall
    clock. The stored row plus committed code has to be enough."""
    db = tmp_path / "j.db"
    config = _app_config(tmp_path, risk_config)
    captured = bullish_pullback_snapshot.captured_at

    original = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, cycle_id="replay-1", now=captured,
    )
    JournalRepository(db).record_audit(original.audit)

    (row,) = JournalRepository(db).audit_for_cycle("replay-1")

    from agentic_trader.models import MarketSnapshot

    rehydrated = MarketSnapshot.model_validate(row["snapshot_json"])
    replayed = run_cycle(
        rehydrated, account, config,
        mode=ExecutionMode(row["mode"]), cycle_id="replay-2",
        now=rehydrated.captured_at,
    )

    assert replayed.outcome is original.outcome
    assert replayed.signal.strength is original.signal.strength
    assert replayed.signal.confidence == original.signal.confidence
    assert replayed.signal.stop_price == original.signal.stop_price
    assert replayed.signal.target_price == original.signal.target_price
    assert replayed.risk_decision.approved_notional == (
        original.risk_decision.approved_notional
    )
    assert replayed.audit.mode is original.audit.mode
    assert replayed.audit.acquisition_config_fingerprint == (
        original.audit.acquisition_config_fingerprint
    )


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
        mode=ExecutionMode.SHADOW, now=bullish_pullback_snapshot.captured_at,
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
        mode=ExecutionMode.SHADOW, now=bullish_pullback_snapshot.captured_at,
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
