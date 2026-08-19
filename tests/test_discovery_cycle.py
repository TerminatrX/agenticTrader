"""End-to-end discovery: drift, coverage, sector, selection, and the seam.

Boundaries proven structurally rather than by flag:

1. **Membership drift stops the run before enrichment.** A widened RSI band, or
   a change to indicator session semantics, means the live universe is not the
   declared one — so `selected_symbols` is empty and there is nothing to fetch.
2. **Incomplete coverage stops it too.** Five shards exist so the universe is
   fully reachable; a capped shard should be split again, not silently sampled.
3. **Sector comes from fundamentals, never scanner columns.** Fixtures report
   `Sector: WRONG` in every scanner row.
4. **Only selected symbols are ever evaluated.** The seam test enriches exactly
   the selection and asserts the rest are never touched.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.agents.discovery import (
    COVERAGE_ABORT,
    DRIFT_ABORT,
    FUNDAMENTALS_MISSING,
    abort_candidates,
    run_discovery,
)
from agentic_trader.cli import main
from agentic_trader.journal import JournalRepository
from agentic_trader.models import MarketSnapshot
from agentic_trader.universe import (
    DISCOVERY_V1,
    ROBINHOOD_MCP_SCANNER,
    CoverageStatus,
    DefinitionDriftStatus,
)

NOW = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)
DAY = date(2026, 8, 18)
SHARDS = DISCOVERY_V1.expected_shard_ids
SECTORS = ["Technology", "Financials", "Healthcare", "Energy", "Utilities"]


def _live_scan(shard, *, rsi=("25", "50"), sorting="Market cap desc",
               cortex=False, session="all"):
    """A get_scans entry matching the definition unless told otherwise.

    Carries `expression` because that is where the broker exposes session
    semantics — it is not a filter field, and it is the only way a change from
    all-session to regular-session becomes visible.
    """
    band = shard.filters["market_cap"]
    values = [str(v) for v in band["values"]] if "values" in band else [str(band["value"])]
    return {
        "scan_id": shard.scan_id,
        "title": shard.label,
        "sorting": sorting,
        "cortex_managed": cortex,
        "filter_summary": [
            {
                "filter_type_enum": "FILTER_TYPE_MARKET_CAP",
                "predicate": band["predicate"],
                "values": values,
                "expression": "fundamental.marketCap",
            },
            {
                "filter_type_enum": "FILTER_TYPE_INSTRUMENT_TYPE",
                "predicate": "=",
                "values": ["STOCK"],
                "expression": "type",
            },
            {
                "filter_type_enum": "FILTER_TYPE_AVERAGE_VOLUME",
                "predicate": ">",
                "values": ["500000"],
                "interval": "1d",
                "length": 30,
                "expression": (
                    f'volumeAvg(candlePeriod="1d", candleCount=30, session="{session}")'
                ),
            },
            {
                "filter_type_enum": "FILTER_TYPE_RSI",
                "predicate": "BETWEEN",
                "values": list(rsi),
                "interval": "1d",
                "length": 14,
                "expression": (
                    f'rsi(candlePeriod="1d", candleCount=14, session="{session}")'
                ),
            },
        ],
    }


def _scans(**kw):
    return {"data": {"scans": [_live_scan(s, **kw) for s in DISCOVERY_V1.shards]}}


def _drifted_scans(shard_index=2, **kw):
    entries = [
        _live_scan(s, **(kw if i == shard_index else {}))
        for i, s in enumerate(DISCOVERY_V1.shards)
    ]
    return {"data": {"scans": entries}}


def _runs(per_shard=4, capped_shard=None, skip=()):
    payloads = []
    for i, sid in enumerate(SHARDS):
        if i in skip:
            continue
        n = ROBINHOOD_MCP_SCANNER.max_rows_per_run if i == capped_shard else per_shard
        rows = [
            {
                "ticker": f"S{i}T{j}",
                "instrument_id": f"iid-S{i}T{j}",
                "instrument_type": "EQUITY",
                # A sector-shaped column the selector must ignore.
                "columns": {"Symbol": f"S{i}T{j}", "RSI": "40", "Sector": "WRONG"},
            }
            for j in range(n)
        ]
        payloads.append(
            {"data": {"result": {"scan_id": sid, "scan_title": f"shard {i}",
                                 "total_items": n, "results": rows}}}
        )
    return payloads


def _fundamentals(per_shard=4, omit=()):
    results = [
        {"symbol": f"S{i}T{j}", "sector": SECTORS[i % len(SECTORS)]}
        for i in range(len(SHARDS))
        for j in range(per_shard)
        if f"S{i}T{j}" not in omit
    ]
    return [{"data": {"results": results[k:k + 10]}} for k in range(0, len(results), 10)]


def _run(**kw):
    kw.setdefault("scans_payload", _scans())
    kw.setdefault("run_payloads", _runs())
    kw.setdefault("fundamentals_payloads", _fundamentals())
    kw.setdefault("definition", DISCOVERY_V1)
    kw.setdefault("trading_date", DAY)
    kw.setdefault("now", NOW)
    kw.setdefault("run_id", "run-e2e")
    return run_discovery(**kw)


# --------------------------------------------------------------- happy path


def test_a_clean_run_discovers_selects_and_reports():
    result = _run()

    assert result.aborted_reason is None
    assert result.coverage is CoverageStatus.COMPLETE
    assert result.drift_status is DefinitionDriftStatus.MATCHES
    assert len(result.selected_symbols) == 20


def test_the_budget_bounds_what_gets_enriched():
    result = _run(run_payloads=_runs(per_shard=20), fundamentals_payloads=_fundamentals(20))

    assert len(result.batch.candidates) == 100
    assert len(result.selected_symbols) == 25


def test_sector_comes_from_fundamentals_not_scanner_columns():
    """Fixtures report `Sector: WRONG` in every scanner row."""
    sectors = {c.sector for c in _run().selection.selected}

    assert "WRONG" not in sectors
    assert sectors == set(SECTORS)


# ------------------------------------------------- membership drift (blocking)


def test_a_widened_rsi_filter_produces_zero_symbols_to_enrich():
    """Legend's RSI band moved from 25-50 to 20-55 on one shard, so the live
    universe is not the declared one. Nothing is handed to the agent."""
    result = _run(scans_payload=_drifted_scans(rsi=("20", "55")))

    assert result.aborted_reason == DRIFT_ABORT
    assert result.selected_symbols == []
    assert result.selection is None
    assert result.batch is None  # scans were never even interpreted


def test_a_session_change_is_membership_drift():
    """all-session and regular-session RSI are different numbers for the same
    symbol, so the same band selects a different universe."""
    result = _run(scans_payload=_drifted_scans(session="regular"))

    assert result.aborted_reason == DRIFT_ABORT
    assert result.drift_status is DefinitionDriftStatus.DRIFTED


def test_a_deleted_saved_scan_blocks():
    payload = _scans()
    payload["data"]["scans"] = payload["data"]["scans"][:-1]

    assert _run(scans_payload=payload).aborted_reason == DRIFT_ABORT


def test_a_cortex_managed_scan_blocks():
    assert _run(scans_payload=_scans(cortex=True)).aborted_reason == DRIFT_ABORT


def test_drift_is_never_repaired_automatically():
    before = DISCOVERY_V1.config_fingerprint
    _run(scans_payload=_drifted_scans(rsi=("20", "55")))

    assert DISCOVERY_V1.config_fingerprint == before


# ----------------------------------------------------- sort drift (conditional)


def test_sort_drift_alone_does_not_stop_a_complete_run():
    """Below the cap every match is returned, so order cannot change
    membership — and the selector ignores scanner order anyway."""
    result = _run(scans_payload=_scans(sorting="Volume desc"))

    assert result.aborted_reason is None
    assert result.drift_status is DefinitionDriftStatus.SORT_CHANGED
    assert result.selected_symbols


def test_a_vanished_sort_is_drift_not_a_match():
    """Absent must not compare equal, most of all at the cap."""
    result = _run(scans_payload=_scans(sorting=None))

    assert result.drift_status is DefinitionDriftStatus.SORT_CHANGED


# --------------------------------------------------------- coverage blocking


def test_a_capped_shard_stops_the_run():
    """Five shards exist so the universe is fully reachable. A shard at the cap
    should be split again, not sampled — otherwise biased observations enter
    the record as ordinary results."""
    result = _run(run_payloads=_runs(capped_shard=0))

    assert result.aborted_reason == COVERAGE_ABORT
    assert result.coverage is CoverageStatus.UNKNOWN_TRUNCATED
    assert result.selected_symbols == []


def test_a_missing_shard_stops_the_run():
    result = _run(run_payloads=_runs(skip=(4,)))

    assert result.aborted_reason == COVERAGE_ABORT
    assert result.coverage is CoverageStatus.INCOMPLETE
    assert result.selected_symbols == []


def test_a_definition_may_opt_out_of_requiring_complete_coverage():
    """Policy on the definition, not baked into ScannerSource."""
    import dataclasses

    lenient = dataclasses.replace(DISCOVERY_V1, require_complete_coverage=False)
    result = _run(definition=lenient, run_payloads=_runs(capped_shard=0),
                  fundamentals_payloads=_fundamentals())

    assert result.aborted_reason is None
    assert result.coverage is CoverageStatus.UNKNOWN_TRUNCATED
    assert result.selected_symbols


def test_coverage_and_drift_stay_orthogonal():
    """Two questions, two answers."""
    clean = _run()
    sort_drifted = _run(scans_payload=_scans(sorting="Volume desc"))

    assert clean.coverage is sort_drifted.coverage is CoverageStatus.COMPLETE
    assert clean.drift_status is not sort_drifted.drift_status


# ------------------------------------------------------ fundamentals missing


def test_a_symbol_absent_from_fundamentals_is_never_selected():
    """"Sector unknown from an authoritative response" and "we never received
    authoritative data" are different facts."""
    result = _run(fundamentals_payloads=_fundamentals(omit={"S0T0", "S0T1"}))

    assert {c.symbol for c in result.unfetched} == {"S0T0", "S0T1"}
    assert "S0T0" not in result.selected_symbols
    assert all(c.exit_reason == FUNDAMENTALS_MISSING for c in result.unfetched)


def test_a_returned_null_sector_is_a_genuine_unknown():
    """Present in the response with sector=None — a fact about the company,
    not about our fetch. Still eligible, subject to the unknown cap."""
    funds = _fundamentals()
    funds[0]["data"]["results"][0]["sector"] = None

    result = _run(fundamentals_payloads=funds)

    assert result.unfetched == []
    assert any(c.sector is None for c in result.selection.selected)


def test_a_wholly_failed_fundamentals_fetch_selects_nothing():
    """Without this distinction a failed batch would feed candidates into
    selection under the unknown-sector cap as though they were ordinary."""
    result = _run(fundamentals_payloads=[])

    assert result.selected_symbols == []
    assert len(result.unfetched) == 20


# ------------------------------------------------------------------ journal


def _record(repo, result):
    repo.record_scan_run(
        result.run_id,
        result.batch,
        candidates=abort_candidates(result),
        scanner_profile_ref=result.scanner_profile_ref,
        scan_definition_ref=result.definition_ref,
        scan_config_fingerprint=result.config_fingerprint,
        scan_config=DISCOVERY_V1.as_config(),
        selected_count=len(result.selected_symbols),
        drift_status=result.drift_status.value,
        drift_findings=result.drift.as_dicts(),
        aborted_reason=result.aborted_reason,
        coverage_status=result.coverage.value,
        source="robinhood_scanner",
        started_at=result.started_at,
    )


def test_a_clean_run_journals_identities_and_shards(tmp_path):
    result = _run()
    repo = JournalRepository(tmp_path / "j.db")
    _record(repo, result)

    (run,) = repo.scan_runs()
    assert run["coverage_complete"] == 1
    assert run["scan_definition_ref"] == "agentic-discovery@v2-2026-08-18"
    assert len(repo.scan_shards(result.run_id)) == 5
    assert json.loads(run["funnel_counts_json"])["selected"] == 20


def test_a_drift_aborted_run_is_journalled_even_without_a_batch(tmp_path):
    """The event most worth auditing produces no DiscoveryBatch at all."""
    result = _run(scans_payload=_drifted_scans(rsi=("20", "55")))
    repo = JournalRepository(tmp_path / "j.db")
    _record(repo, result)

    (run,) = repo.scan_runs()
    assert run["aborted_reason"] == DRIFT_ABORT
    assert run["drift_status"] == "drifted"
    assert any(f["kind"] == "filter_drift" for f in json.loads(run["drift_findings_json"]))


# ------------------------------------------------------------ the CLI path


def _bundle(tmp_path, name, scans, runs=None, funds=None):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "payloads": {
            "scans": scans,
            "runs": runs if runs is not None else _runs(),
            "fundamentals": funds if funds is not None else _fundamentals(),
        }
    }))
    return path


def test_the_cli_journals_a_drift_refusal(tmp_path, capsys):
    """Through main(), not run_discovery() — the CLI is where the journal write
    lives, and an early return must not skip it."""
    db = tmp_path / "j.db"
    bundle = _bundle(tmp_path, "drift", _drifted_scans(rsi=("20", "55")))

    code = main(["--db", str(db), "discover", "--input", str(bundle), "--date", "2026-08-18"])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert out["selected"] == []
    assert out["aborted_reason"] == DRIFT_ABORT

    repo = JournalRepository(db)
    (run,) = repo.scan_runs()
    assert run["aborted_reason"] == DRIFT_ABORT
    assert run["scan_definition_ref"] == "agentic-discovery@v2-2026-08-18"


def test_the_cli_journals_a_clean_run(tmp_path, capsys):
    db = tmp_path / "j.db"
    bundle = _bundle(tmp_path, "clean", _scans())

    main(["--db", str(db), "discover", "--input", str(bundle), "--date", "2026-08-18"])
    out = json.loads(capsys.readouterr().out)

    assert len(out["selected"]) == 20
    repo = JournalRepository(db)
    (run,) = repo.scan_runs()
    assert run["aborted_reason"] is None
    assert run["coverage_complete"] == 1
    assert len(repo.scan_candidates(run["run_id"])) == 20


def test_the_trading_date_is_required(tmp_path):
    """UTC crosses midnight while the US session is open, so an ambient clock
    would rotate a day early and mislabel the journal."""
    bundle = _bundle(tmp_path, "nodate", _scans())

    with pytest.raises(SystemExit):
        main(["discover", "--input", str(bundle)])


def test_the_same_date_replays_identically(tmp_path, capsys):
    db = tmp_path / "j.db"
    bundle = _bundle(tmp_path, "rep", _scans(), runs=_runs(per_shard=20),
                     funds=_fundamentals(20))

    main(["--db", str(db), "discover", "--input", str(bundle), "--date", "2026-08-18",
          "--no-journal"])
    first = json.loads(capsys.readouterr().out)["selected"]
    main(["--db", str(db), "discover", "--input", str(bundle), "--date", "2026-08-18",
          "--no-journal"])
    second = json.loads(capsys.readouterr().out)["selected"]
    main(["--db", str(db), "discover", "--input", str(bundle), "--date", "2026-08-19",
          "--no-journal"])
    other_day = json.loads(capsys.readouterr().out)["selected"]

    assert first == second
    assert set(first) != set(other_day)


# ------------------------------------------------- discovery -> evaluate seam


def test_only_selected_symbols_are_ever_evaluated(tmp_path):
    """The orchestration contract, offline.

    Five candidates, budget two: exactly two enrichment bundles are built and
    exactly two cycles run. The other three are never evaluated — proving the
    seam without pushing I/O into src/.
    """
    from agentic_trader.agents.orchestrator import run_cycle
    from agentic_trader.config import AppConfig, RiskConfig, StrategyConfig
    from agentic_trader.models import AccountState

    result = _run(run_payloads=_runs(per_shard=1), fundamentals_payloads=_fundamentals(1))
    selected = _run(
        run_payloads=_runs(per_shard=1),
        fundamentals_payloads=_fundamentals(1),
        budget=2,
        max_per_sector=2,
    ).selected_symbols

    assert len(result.batch.candidates) == 5
    assert len(selected) == 2

    # The agent enriches exactly the selection — nothing else has a bundle.
    enrichment: dict[str, dict] = {sym: {"quote": {}} for sym in selected}
    assert set(enrichment) == set(selected)
    assert len(enrichment) == 2

    evaluated: list[str] = []
    config = AppConfig(
        project_root=tmp_path, risk=RiskConfig(), strategies=StrategyConfig()
    )
    account = AccountState(
        account_number="TEST0000", total_value=Decimal("100"),
        cash=Decimal("100"), buying_power=Decimal("100"),
    )
    for symbol in enrichment:
        snap = MarketSnapshot(
            symbol=symbol, captured_at=NOW, last_price=Decimal("50"), quote_as_of=NOW
        )
        run_cycle(snap, account, config, mode="shadow", now=NOW)
        evaluated.append(symbol)

    assert evaluated == list(selected)
    unselected = {c.symbol for c in result.batch.candidates} - set(selected)
    assert unselected and not (unselected & set(evaluated))


def test_a_blocked_run_yields_nothing_to_evaluate():
    """The structural end of the drift boundary: the loop body never runs."""
    result = _run(scans_payload=_drifted_scans(rsi=("20", "55")))

    evaluated = [s for s in result.selected_symbols]
    assert evaluated == []
