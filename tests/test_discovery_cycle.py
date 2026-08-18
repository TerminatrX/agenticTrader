"""End-to-end discovery: drift, coverage, sector, selection.

Two boundaries are structural here rather than conventional, and each has a
test that would fail loudly if the structure were weakened:

1. **Membership drift stops the run before enrichment.** A widened RSI filter
   means the live universe is not the declared one, so `selected_symbols` is
   empty and there is nothing for the agent to fetch. The proof is not that
   some flag was set — it is that the output the agent acts on contains no
   symbols at all.
2. **Sector comes from fundamentals, never from scanner columns.** The scanner
   reports a sector-shaped column in these fixtures; the selector must ignore
   it entirely.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from agentic_trader.agents.discovery import DRIFT_ABORT, abort_candidates, run_discovery
from agentic_trader.journal import JournalRepository
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


def _live_scan(shard, *, rsi=("25", "50"), sorting="Market cap desc", cortex=False):
    """A get_scans entry matching the definition unless told otherwise."""
    band = shard.filters["market_cap"]
    values = (
        [str(v) for v in band["values"]] if "values" in band else [str(band["value"])]
    )
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
            },
            {
                "filter_type_enum": "FILTER_TYPE_INSTRUMENT_TYPE",
                "predicate": "=",
                "values": ["STOCK"],
            },
            {
                "filter_type_enum": "FILTER_TYPE_AVERAGE_VOLUME",
                "predicate": ">",
                "values": ["500000"],
                "interval": "1d",
                "length": 30,
            },
            {
                "filter_type_enum": "FILTER_TYPE_RSI",
                "predicate": "BETWEEN",
                "values": list(rsi),
                "interval": "1d",
                "length": 14,
            },
        ],
    }


def _scans(**kw):
    return {"data": {"scans": [_live_scan(s, **kw) for s in DISCOVERY_V1.shards]}}


def _drifted_scans(shard_index=2, rsi=("20", "55")):
    """One shard's RSI band widened in Legend — the membership-drift case."""
    entries = []
    for i, shard in enumerate(DISCOVERY_V1.shards):
        entries.append(_live_scan(shard, rsi=rsi if i == shard_index else ("25", "50")))
    return {"data": {"scans": entries}}


def _runs(per_shard=4, capped_shard=None):
    payloads = []
    for i, sid in enumerate(SHARDS):
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


def _fundamentals(per_shard=4):
    """Batched get_equity_fundamentals — ten symbols per call in reality."""
    results = [
        {"symbol": f"S{i}T{j}", "sector": SECTORS[i % len(SECTORS)]}
        for i in range(len(SHARDS))
        for j in range(per_shard)
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
    assert len(result.batch.candidates) == 20
    assert len(result.selected_symbols) == 20  # under the 25 budget


def test_the_budget_bounds_what_gets_enriched():
    result = _run(run_payloads=_runs(per_shard=20), fundamentals_payloads=_fundamentals(20))

    assert len(result.batch.candidates) == 100
    assert len(result.selected_symbols) == 25
    assert result.selection.deferred_count == 75


def test_sector_comes_from_fundamentals_not_scanner_columns():
    """Fixtures report `Sector: WRONG` in every scanner row."""
    result = _run()

    sectors = {c.sector for c in result.selection.selected}
    assert "WRONG" not in sectors
    assert sectors <= set(SECTORS)
    assert len(sectors) == 5  # spread across all five, not concentrated


# ------------------------------------------------- membership drift (blocking)


def test_a_widened_rsi_filter_produces_zero_symbols_to_enrich():
    """The structural proof, per the review requirement.

    Legend's RSI band moved from 25-50 to 20-55 on one shard. The live universe
    is no longer the declared one, so the run aborts before reading any scan
    result and hands the agent nothing to fetch. Enrichment cannot happen
    because there is no symbol to enrich.
    """
    result = _run(scans_payload=_drifted_scans())

    assert result.aborted_reason == DRIFT_ABORT
    assert result.selected_symbols == []
    assert result.selection is None
    assert result.batch is None  # scans were never even interpreted
    assert result.drift_status is DefinitionDriftStatus.DRIFTED


def test_drift_reports_what_changed():
    result = _run(scans_payload=_drifted_scans())
    (finding,) = [f for f in result.drift.findings if f.kind.value == "filter_drift"]

    assert finding.severity.value == "blocking"
    assert finding.scan_id == SHARDS[2]
    assert any("20" in o for o in finding.observed)


def test_a_deleted_saved_scan_blocks():
    payload = _scans()
    payload["data"]["scans"] = payload["data"]["scans"][:-1]

    result = _run(scans_payload=payload)

    assert result.aborted_reason == DRIFT_ABORT
    assert result.selected_symbols == []


def test_a_cortex_managed_scan_blocks():
    """No longer ours to rely on."""
    result = _run(scans_payload=_scans(cortex=True))

    assert result.aborted_reason == DRIFT_ABORT


def test_drift_is_never_repaired_automatically():
    """A mismatch means either Legend or the definition should change, and
    which one is a human decision."""
    before = DISCOVERY_V1.config_fingerprint
    _run(scans_payload=_drifted_scans())

    assert DISCOVERY_V1.config_fingerprint == before


# ----------------------------------------------------- sort drift (conditional)


def test_sort_drift_alone_does_not_stop_a_complete_run():
    """Below the cap every match is returned, so order cannot change membership
    — and the selector ignores scanner order anyway."""
    result = _run(scans_payload=_scans(sorting="Volume desc"))

    assert result.aborted_reason is None
    assert result.drift_status is DefinitionDriftStatus.SORT_CHANGED
    assert result.selected_symbols


def test_sort_drift_blocks_once_a_shard_caps():
    """At the cap, sorting decides which 200 survive — membership-affecting."""
    result = _run(
        scans_payload=_scans(sorting="Volume desc"),
        run_payloads=_runs(capped_shard=0),
        fundamentals_payloads=_fundamentals(),
    )

    assert result.aborted_reason == DRIFT_ABORT
    assert result.selected_symbols == []
    assert result.batch is not None  # coverage was computed before aborting


def test_a_capped_shard_without_sort_drift_still_runs():
    """Truncation is recorded as UNKNOWN_TRUNCATED, not treated as drift."""
    result = _run(run_payloads=_runs(capped_shard=0), fundamentals_payloads=_fundamentals())

    assert result.aborted_reason is None
    assert result.coverage is CoverageStatus.UNKNOWN_TRUNCATED
    assert result.selected_symbols


# ------------------------------------------------------ coverage independence


def test_coverage_and_drift_are_orthogonal():
    """Two questions, two answers. Folding drift into INCOMPLETE would make
    coverage stop meaning coverage."""
    clean = _run()
    sorted_drift = _run(scans_payload=_scans(sorting="Volume desc"))

    assert clean.coverage is sorted_drift.coverage is CoverageStatus.COMPLETE
    assert clean.drift_status is not sorted_drift.drift_status


def test_a_missing_shard_run_is_incomplete_but_not_drifted():
    result = _run(run_payloads=_runs()[:4])

    assert result.coverage is CoverageStatus.INCOMPLETE
    assert result.drift_status is DefinitionDriftStatus.MATCHES


# ------------------------------------------------------------------ journal


def test_an_aborted_run_still_journals_its_funnel(tmp_path):
    """"We refused to look" and "we looked and found nothing" must never be
    confusable."""
    result = _run(scans_payload=_scans(sorting="Volume desc"),
                  run_payloads=_runs(capped_shard=0),
                  fundamentals_payloads=_fundamentals())
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run(
        result.run_id,
        result.batch,
        candidates=abort_candidates(result),
        scanner_profile_ref=result.scanner_profile_ref,
        scan_definition_ref=result.definition_ref,
        scan_config_fingerprint=result.config_fingerprint,
        scan_config=DISCOVERY_V1.as_config(),
    )

    stored = repo.scan_candidates(result.run_id)
    assert stored
    assert {c["exit_reason"] for c in stored} == {DRIFT_ABORT}


def test_a_clean_run_journals_identities_and_shards(tmp_path):
    result = _run()
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run(
        result.run_id,
        result.batch,
        candidates=result.all_candidates,
        scanner_profile_ref=result.scanner_profile_ref,
        scan_definition_ref=result.definition_ref,
        scan_config_fingerprint=result.config_fingerprint,
        scan_config=DISCOVERY_V1.as_config(),
        selected_count=len(result.selected_symbols),
    )

    (run,) = repo.scan_runs()
    assert run["coverage_complete"] == 1
    assert run["scan_definition_ref"] == "agentic-discovery@v1-2026-08-18"
    assert len(repo.scan_shards(result.run_id)) == 5
    assert json.loads(run["funnel_counts_json"])["selected"] == 20


def test_the_summary_is_agent_readable():
    summary = _run().summary()

    assert set(summary) >= {
        "run_id", "trading_date", "definition_ref", "coverage",
        "drift_status", "aborted_reason", "funnel", "selected",
    }
    assert summary["aborted_reason"] is None
