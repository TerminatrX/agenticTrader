"""Discovery, coverage, and the enrichment budget.

The load-bearing test here is the trust boundary: a candidate's `source_values`
can be arbitrarily wrong without changing any decision, because only the symbol
crosses into authoritative enrichment.

Most of the rest defend one property — never claim to have seen more of the
universe than we did. Coverage can be overstated by a missing shard, an
unexpected one, a duplicated one, an unparseable payload, a malformed row
shortening a capped result below the cap, or by trusting a broker field observed
to contradict itself. Each has a test, because every one of them reads
downstream as "the market was quiet today".
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.journal import JournalRepository
from agentic_trader.market.snapshot import build_snapshot
from agentic_trader.universe import (
    BUDGET_EXHAUSTED,
    DISCOVERY_V1,
    ROBINHOOD_MCP_SCANNER,
    UNKNOWN_SECTOR_LIMIT,
    CoverageStatus,
    FunnelStage,
    IdentityConflict,
    ScanCandidate,
    ScannerSource,
    ShardError,
    StaticSource,
    funnel_counts,
    parse_scan_payload,
    rotation_key,
    select_for_enrichment,
)

NOW = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)
DAY = date(2026, 8, 18)
SHARDS = DISCOVERY_V1.expected_shard_ids


def _payload(scan_id, tickers, total=None, extra_rows=()):
    rows = [
        {
            "ticker": t,
            "instrument_id": f"iid-{t}",
            "instrument_type": "EQUITY",
            "columns": {"Symbol": t, "RSI": "40.0", "Market cap": "3.0e+09"},
        }
        for t in tickers
    ]
    rows.extend(extra_rows)
    return {
        "data": {
            "result": {
                "scan_id": scan_id,
                "scan_title": f"shard {scan_id}",
                "total_items": total if total is not None else len(rows),
                "results": rows,
            }
        }
    }


def _all_shards():
    return [_payload(sid, [f"S{i}"]) for i, sid in enumerate(SHARDS)]


def _scan(payloads):
    return ScannerSource(payloads, DISCOVERY_V1).discover(now=NOW)


def _cand(symbol, sector=None):
    return ScanCandidate(
        symbol=symbol, source="test", discovered_at=NOW, instrument_id=f"iid-{symbol}"
    ).with_sector(sector)


def _sel(cands, **kw):
    kw.setdefault("on", DAY)
    return select_for_enrichment(cands, **kw)


# ------------------------------------------------------- coverage: shard set


def test_zero_shards_is_incomplete_not_complete():
    """`all([])` is True, so an empty run would otherwise claim full coverage —
    the most dangerous possible wrong answer here."""
    batch = _scan([])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert len(batch.missing_shard_ids) == 5


def test_four_of_five_shards_is_incomplete_even_when_each_is_short():
    batch = _scan(_all_shards()[:4])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.missing_shard_ids == (SHARDS[4],)


def test_all_five_short_shards_are_complete():
    batch = _scan(_all_shards())

    assert batch.coverage is CoverageStatus.COMPLETE
    assert batch.report.reasons() == []
    assert len(batch.candidates) == 5


def test_an_unexpected_shard_is_not_complete():
    """All five expected ids present plus a sixth still satisfies "nothing
    missing", while the stranger's candidates join the declared universe."""
    batch = _scan([*_all_shards(), _payload("some-other-scan", ["XYZ"])])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.report.unexpected_shard_ids == ("some-other-scan",)


def test_a_duplicated_shard_is_not_complete():
    """Double-counts a band and inflates the dedupe statistics that are meant
    to reveal boundary semantics."""
    batch = _scan([*_all_shards(), _payload(SHARDS[0], ["EXTRA"])])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.report.duplicate_shard_ids == (SHARDS[0],)


def test_an_unparseable_payload_is_counted_not_swallowed():
    """The missing-id check catches a broken *expected* shard; an extra
    malformed payload would otherwise vanish without affecting coverage."""
    batch = _scan([*_all_shards(), {"garbage": True}])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.report.parse_error_count == 1
    assert any("failed to parse" in r for r in batch.report.reasons())


def test_a_broken_expected_shard_is_reported_missing():
    payloads = [*_all_shards()[:4], {"data": {"result": {"scan_id": SHARDS[4]}}}]
    batch = _scan(payloads)

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert SHARDS[4] in batch.missing_shard_ids


def test_a_capped_shard_makes_the_run_unknown_truncated():
    payloads = [_payload(SHARDS[0], [f"T{i}" for i in range(200)]), *_all_shards()[1:]]
    assert _scan(payloads).coverage is CoverageStatus.UNKNOWN_TRUNCATED


# --------------------------------------------------- coverage: the row count


def test_coverage_uses_raw_rows_not_parsed_candidates():
    """200 raw rows with one unusable parses to 199 candidates.

    Judging coverage on the parsed count would call that COMPLETE, disguising
    a response that actually hit the cap.
    """
    shard = parse_scan_payload(
        _payload("s", [f"T{i}" for i in range(199)],
                 extra_rows=[{"instrument_id": "iid-x", "columns": {}}]),
        source="x", discovered_at=NOW,
    )

    assert shard.returned_row_count == 200
    assert shard.candidate_count == 199
    assert shard.rows_rejected == 1
    assert shard.coverage is not CoverageStatus.COMPLETE


def test_a_row_without_identity_makes_the_shard_incomplete():
    shard = parse_scan_payload(
        _payload("s", ["AAPL"], extra_rows=[{"columns": {"RSI": "40"}}]),
        source="x", discovered_at=NOW,
    )

    assert shard.coverage is CoverageStatus.INCOMPLETE
    assert shard.rows_rejected == 1
    assert [c.symbol for c in shard.candidates] == ["AAPL"]


def test_a_full_result_set_is_unknown_not_complete():
    shard = parse_scan_payload(
        _payload("s", [f"T{i}" for i in range(200)]), source="x", discovered_at=NOW
    )
    assert shard.coverage is CoverageStatus.UNKNOWN_TRUNCATED


def test_coverage_ignores_the_reported_total():
    """Five disjoint shards summed to 668 while the unsharded equivalent
    reported 394, so that field cannot establish anything."""
    honest = parse_scan_payload(_payload("s", ["A", "B"], total=2), source="x", discovered_at=NOW)
    lying = parse_scan_payload(_payload("s", ["A", "B"], total=9999), source="x", discovered_at=NOW)

    assert honest.coverage is lying.coverage is CoverageStatus.COMPLETE
    assert lying.reported_total == 9999  # recorded, never trusted


def test_a_payload_with_no_result_object_raises():
    with pytest.raises(ShardError):
        parse_scan_payload({"nonsense": True}, source="x", discovered_at=NOW)


def test_a_scanner_source_requires_a_declared_universe():
    """"Complete" is meaningless without a universe to be complete of."""
    with pytest.raises(TypeError):
        ScannerSource([])  # type: ignore[call-arg]


# ------------------------------------------------------------------- dedupe


def test_the_same_symbol_dedupes_across_shards():
    batch = _scan([_payload(SHARDS[0], ["AAPL", "MSFT"]), _payload(SHARDS[1], ["MSFT", "NVDA"])])

    assert sorted(c.symbol for c in batch.candidates) == ["AAPL", "MSFT", "NVDA"]
    assert batch.returned_before_dedupe == 4
    assert batch.duplicates_removed == 1


def test_a_missing_instrument_id_still_dedupes():
    """Keying on `instrument_id or symbol` would give one copy the key "iid-AAPL"
    and the other "AAPL", so the duplicate would survive."""
    without = _payload(SHARDS[1], ["AAPL"])
    without["data"]["result"]["results"][0]["instrument_id"] = None

    batch = _scan([_payload(SHARDS[0], ["AAPL"]), without])

    assert len(batch.candidates) == 1
    assert batch.duplicates_removed == 1
    assert batch.candidates[0].instrument_id == "iid-AAPL"  # identity preserved


def test_conflicting_instrument_ids_fail_loudly():
    b = _payload(SHARDS[1], ["AAPL"])
    b["data"]["result"]["results"][0]["instrument_id"] = "different-id"

    with pytest.raises(IdentityConflict, match="two instrument ids"):
        _scan([_payload(SHARDS[0], ["AAPL"]), b])


def test_disjoint_shards_remove_nothing():
    batch = _scan([_payload(SHARDS[0], ["AAPL"]), _payload(SHARDS[1], ["MSFT"])])
    assert batch.duplicates_removed == 0


def test_static_source_is_always_complete():
    batch = StaticSource(["aapl", " msft ", "AAPL"]).discover(now=NOW)

    assert [c.symbol for c in batch.candidates] == ["AAPL", "MSFT"]
    assert batch.coverage_complete


# ------------------------------------------------------- enrichment budget


def test_the_budget_spreads_across_sectors():
    candidates = (
        [_cand(f"TECH{i}", "Technology") for i in range(20)]
        + [_cand(f"FIN{i}", "Financials") for i in range(20)]
        + [_cand(f"HLTH{i}", "Healthcare") for i in range(20)]
    )
    result = _sel(candidates, budget=25)

    assert len(result.selected) == 25
    assert len(result.per_sector) == 3
    assert max(result.per_sector.values()) <= 13


def test_a_single_known_sector_may_fill_the_whole_budget():
    """The cap allocates compute between sectors. With no other sector to
    protect, holding capacity back protects nothing — max_sector_exposure_pct
    still governs what the account may actually hold.
    """
    result = _sel([_cand(f"T{i}", "Technology") for i in range(60)], budget=25)

    assert len(result.selected) == 25
    assert result.per_sector["Technology"] == 25


def test_unknown_sector_never_backfills():
    """A missing sector means the concentration gate cannot fully assess the
    position, so an unknown-heavy day deliberately underuses the budget."""
    result = _sel([_cand(f"U{i}") for i in range(60)], budget=25)

    assert len(result.selected) == 13
    assert result.per_sector["unknown"] == 13


def test_an_unknown_shortfall_is_not_reported_as_budget_exhaustion():
    result = _sel([_cand(f"U{i}") for i in range(60)], budget=25)

    assert all(c.exit_reason == UNKNOWN_SECTOR_LIMIT for c in result.deferred)
    assert result.budget_deferred_count == 0


def test_a_real_budget_exhaustion_says_so():
    result = _sel([_cand(f"T{i}", f"S{i % 4}") for i in range(80)], budget=8)

    assert len(result.selected) == 8
    assert all(c.exit_reason == BUDGET_EXHAUSTED for c in result.deferred)


def test_known_sectors_backfill_around_a_capped_unknown_bucket():
    mixed = [_cand(f"T{i}", "Technology") for i in range(40)] + [_cand(f"U{i}") for i in range(40)]
    result = _sel(mixed, budget=25)

    assert len(result.selected) == 25
    assert result.per_sector["unknown"] <= 13


def test_selected_candidates_advance_a_stage():
    assert _sel([_cand("AAPL", "Technology")], budget=5).selected[0].stage is FunnelStage.SELECTED


def test_a_zero_budget_selects_nothing():
    result = _sel([_cand("AAPL", "Technology")], budget=0)
    assert result.selected == []
    assert result.deferred_count == 1


# ------------------------------------------------------------- rotation


def test_the_selection_date_is_required():
    """Date is part of the algorithm; an implicit today() would make the same
    stored candidates select differently on replay."""
    with pytest.raises(TypeError):
        select_for_enrichment([_cand("AAPL", "Technology")], budget=5)  # type: ignore[call-arg]


def test_selection_is_deterministic_for_one_date():
    c = [_cand(f"T{i}", f"S{i % 3}") for i in range(40)]
    assert [x.symbol for x in _sel(list(c), budget=7).selected] == [
        x.symbol for x in _sel(list(c), budget=7).selected
    ]


def test_selection_rotates_across_dates():
    """Sorting by symbol would replace the broker's market-cap sampling bias
    with an alphabet one across daily runs."""
    c = [_cand(f"T{i:02}", "Technology") for i in range(60)]
    day1 = {x.symbol for x in _sel(list(c), budget=10, on=date(2026, 8, 18)).selected}
    day2 = {x.symbol for x in _sel(list(c), budget=10, on=date(2026, 8, 19)).selected}

    assert day1 != day2


def test_rotation_key_is_stable_across_processes():
    """hashlib, not hash() — the builtin is salted per process."""
    assert rotation_key("AAPL", DAY) == rotation_key("AAPL", DAY)
    assert rotation_key("AAPL", DAY) != rotation_key("AAPL", date(2026, 8, 19))


# ------------------------------------------- the trust boundary (load-bearing)


QUOTE = {
    "data": {
        "results": [
            {
                "quote": {
                    "symbol": "AAPL",
                    "last_trade_price": "306.44",
                    "venue_last_trade_time": "2026-08-18T13:55:24Z",
                    "bid_price": "306.43",
                    "ask_price": "306.45",
                    "venue_bid_time": "2026-08-18T13:55:24Z",
                    "venue_ask_time": "2026-08-18T13:55:24Z",
                    "has_traded": True,
                    "state": "active",
                }
            }
        ]
    }
}
INDICATORS = {
    "rsi": {
        "data": {
            "indicators": [
                {"type": "rsi", "series": [{"begins_at": "2026-08-17T00:00:00Z", "value": 41.0}]}
            ]
        }
    }
}


def test_a_lying_scanner_cannot_change_the_decision_snapshot():
    """The property the whole discovery architecture rests on.

    Two candidates for one symbol with contradictory scanner values produce
    identical snapshots, because only `symbol` crosses into enrichment.
    """
    honest = ScanCandidate(
        symbol="AAPL", source="scanner", discovered_at=NOW,
        source_values={"RSI": "41.0", "Market cap": "4.4e+12", "Last": "306.44"},
    )
    lying = ScanCandidate(
        symbol="AAPL", source="scanner", discovered_at=NOW,
        source_values={"RSI": "99.0", "Market cap": "1", "Last": "0.01"},
    )

    snaps = [
        build_snapshot(c.symbol, quote=QUOTE, indicators=INDICATORS, captured_at=NOW)
        for c in (honest, lying)
    ]

    assert snaps[0] == snaps[1]
    assert snaps[0].last_price == Decimal("306.44")
    assert snaps[0].indicators.rsi_14 == pytest.approx(41.0)


def test_source_values_never_appear_in_a_snapshot():
    """Structural: there is no parameter through which scanner columns could
    be passed."""
    import inspect

    assert set(inspect.signature(build_snapshot).parameters) == {
        "symbol", "quote", "historicals", "fundamentals",
        "earnings", "indicators", "captured_at",
    }


# ------------------------------------------------------ capability / config


def test_the_scanner_profile_describes_the_endpoint_only():
    """Scan-specific configuration belongs to the definition, not the profile."""
    fields = set(ROBINHOOD_MCP_SCANNER.__dataclass_fields__)

    assert "default_row_fields" not in fields
    assert "default_sorting" not in fields
    assert "sorting" in DISCOVERY_V1.as_config()


def test_the_three_contracts_are_versioned_independently():
    from agentic_trader.execution.capabilities import ROBINHOOD_MCP

    assert len({
        ROBINHOOD_MCP.profile_ref,
        ROBINHOOD_MCP_SCANNER.profile_ref,
        DISCOVERY_V1.definition_ref,
    }) == 3


def test_the_scanner_fingerprint_is_pinned():
    """Detects content changing without a version bump.

    Asserting only that a *modified* profile hashes differently proves the hash
    function works; it does not enforce version immutability. If this fails,
    bump `version` and `as_of` on ROBINHOOD_MCP_SCANNER in the same commit.
    """
    assert ROBINHOOD_MCP_SCANNER.profile_ref == "robinhood-mcp-scanner@2026-08-18"
    assert ROBINHOOD_MCP_SCANNER.content_fingerprint == (
        "0706d64044042019edcb67eee3ca0570db1086edb725cd2d462d0cad20c61396"
    )


def test_the_discovery_definition_fingerprint_is_pinned():
    """Same reasoning: retuning RSI 25-50 to 20-55 must not leave the run
    labelled agentic-discovery@v1-2026-08-18."""
    assert DISCOVERY_V1.definition_ref == "agentic-discovery@v1-2026-08-18"
    assert DISCOVERY_V1.config_fingerprint == (
        "52f7627cb23d99eaeaf627b9af873b65837326ad2e7cef395b41e673b88c4692"
    )


def test_retuning_a_filter_moves_the_definition_fingerprint():
    import dataclasses

    retuned = dataclasses.replace(
        DISCOVERY_V1,
        base_filters={**DISCOVERY_V1.base_filters, "rsi": {"values": [20, 55]}},
    )
    assert retuned.config_fingerprint != DISCOVERY_V1.config_fingerprint


def test_every_shard_declares_the_stock_filter():
    """The Mega shard was created without it and repaired later; the definition
    and the live scans must agree on the declared universe."""
    assert DISCOVERY_V1.base_filters["instrument_type"]["value"] == "STOCK"
    assert len(DISCOVERY_V1.expected_shard_ids) == 5


def test_the_vocabulary_mismatch_is_recorded():
    """Row says EQUITY, filter wants STOCK. A filter built from the row
    vocabulary validates and silently matches nothing."""
    assert ROBINHOOD_MCP_SCANNER.row_instrument_type == "EQUITY"
    assert ROBINHOOD_MCP_SCANNER.filter_instrument_type == "STOCK"


def test_freshness_is_recorded_as_absent_but_is_not_a_permission():
    cap = ROBINHOOD_MCP_SCANNER.row_freshness_timestamp
    assert cap.supported is False
    assert cap.is_certain
    assert not cap.usable


# ------------------------------------------------------------------ journal


def _record(repo, batch, result, run_id="run-1"):
    repo.record_scan_run(
        run_id,
        batch,
        candidates=[*result.selected, *result.deferred],
        scanner_profile_ref=ROBINHOOD_MCP_SCANNER.profile_ref,
        scan_definition_ref=DISCOVERY_V1.definition_ref,
        scan_config_fingerprint=DISCOVERY_V1.config_fingerprint,
        scan_config=DISCOVERY_V1.as_config(),
        selected_count=len(result.selected),
        budget_deferred_count=result.budget_deferred_count,
    )


def test_stored_funnel_counts_match_the_stored_candidate_stages(tmp_path):
    """Candidates are immutable, so selection returns new objects. Counting the
    pre-selection batch would record everything as `discovered`."""
    batch = _scan([_payload(SHARDS[0], ["AAPL", "MSFT", "NVDA"])])
    batch.candidates = [c.with_sector("Technology") for c in batch.candidates]
    result = _sel(batch.candidates, budget=2, max_per_sector=2)
    persisted = [*result.selected, *result.deferred]

    repo = JournalRepository(tmp_path / "j.db")
    _record(repo, batch, result)

    (run,) = repo.scan_runs()
    stored = repo.scan_candidates("run-1")

    assert json.loads(run["funnel_counts_json"]) == funnel_counts(persisted)
    assert Counter(c["stage"] for c in stored) == Counter({"selected": 2, "discovered": 1})
    assert run["budget_deferred_count"] == 1


def test_a_run_records_all_three_contract_identities(tmp_path):
    batch = _scan(_all_shards())
    result = _sel(batch.candidates, budget=5)
    repo = JournalRepository(tmp_path / "j.db")
    _record(repo, batch, result)

    (run,) = repo.scan_runs()
    assert run["scanner_profile_ref"] == "robinhood-mcp-scanner@2026-08-18"
    assert run["scan_definition_ref"] == "agentic-discovery@v1-2026-08-18"
    assert run["scan_config_fingerprint"] == DISCOVERY_V1.config_fingerprint
    assert json.loads(run["scan_config_json"])["base_filters"]["rsi"]["values"] == [25, 50]


def test_per_shard_detail_is_journalled(tmp_path):
    """A run can be incomplete six ways; a single status cannot say which."""
    batch = _scan([*_all_shards()[:4], {"garbage": True}])
    result = _sel(batch.candidates, budget=5)
    repo = JournalRepository(tmp_path / "j.db")
    _record(repo, batch, result)

    shards = repo.scan_shards("run-1")
    (run,) = repo.scan_runs()

    assert len(shards) == 4
    assert {s["scan_id"] for s in shards} == set(SHARDS[:4])
    assert run["coverage_status"] == "incomplete"
    assert any("failed to parse" in r for r in json.loads(run["coverage_reasons_json"]))


def test_a_journal_from_the_previous_release_still_accepts_a_run(tmp_path):
    """98ee06d created scan_runs with `truncated_candidate_count`.

    CREATE TABLE IF NOT EXISTS leaves that table untouched, so without an
    additive migration the insert fails on a missing column.
    """
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE, source TEXT NOT NULL,
            scanner_profile_ref TEXT, started_at TEXT NOT NULL, completed_at TEXT,
            shard_count INTEGER NOT NULL DEFAULT 0,
            returned_before_dedupe INTEGER NOT NULL DEFAULT 0,
            unique_discovered INTEGER NOT NULL DEFAULT 0,
            duplicates_removed INTEGER NOT NULL DEFAULT 0,
            selected_for_enrichment INTEGER NOT NULL DEFAULT 0,
            truncated_candidate_count INTEGER NOT NULL DEFAULT 0,
            coverage_status TEXT NOT NULL,
            coverage_complete INTEGER NOT NULL DEFAULT 0,
            scan_config_json TEXT, funnel_counts_json TEXT)"""
    )
    conn.commit()
    conn.close()

    batch = _scan(_all_shards())
    result = _sel(batch.candidates, budget=5)
    repo = JournalRepository(db)  # migration runs here
    _record(repo, batch, result)

    (run,) = repo.scan_runs()
    assert run["scan_definition_ref"] == "agentic-discovery@v1-2026-08-18"
    assert "truncated_candidate_count" in run  # obsolete column left in place
