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
from pathlib import Path

import pytest

from agentic_trader.journal import JournalRepository
from agentic_trader.market.snapshot import build_snapshot
from agentic_trader.universe import (
    BUDGET_EXHAUSTED,
    CURRENT_DISCOVERY,
    ROBINHOOD_MCP_SCANNER,
    UNKNOWN_SECTOR_LIMIT,
    CoverageStatus,
    DefinitionDriftStatus,
    FunnelStage,
    IdentityConflict,
    ScanCandidate,
    ScannerSource,
    ShardError,
    StaticSource,
    check_definition_drift,
    funnel_counts,
    parse_scan_payload,
    rotation_key,
    select_for_enrichment,
)

NOW = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)
DAY = date(2026, 8, 18)
SHARDS = CURRENT_DISCOVERY.expected_shard_ids


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
    return ScannerSource(payloads, CURRENT_DISCOVERY).discover(now=NOW)


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
    assert len(batch.missing_shard_ids) == len(SHARDS)


def test_one_shard_short_is_incomplete_even_when_each_returned_is_short():
    batch = _scan(_all_shards()[:-1])

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.missing_shard_ids == (SHARDS[-1],)


def test_all_short_shards_are_complete():
    batch = _scan(_all_shards())

    assert batch.coverage is CoverageStatus.COMPLETE
    assert batch.report.reasons() == []
    assert len(batch.candidates) == len(SHARDS)


def test_an_unexpected_shard_is_not_complete():
    """Every expected id present plus one more still satisfies "nothing
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
    assert "sorting" in CURRENT_DISCOVERY.as_config()


def test_the_three_contracts_are_versioned_independently():
    from agentic_trader.execution.capabilities import ROBINHOOD_MCP

    assert len({
        ROBINHOOD_MCP.profile_ref,
        ROBINHOOD_MCP_SCANNER.profile_ref,
        CURRENT_DISCOVERY.definition_ref,
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
    labelled with the version that denoted the old universe.

    v3 re-shards the universe from five bands to eight, because two v2 bands
    were within 20 rows of the scanner cap. Both the shard ids and the bands
    changed, so the version had to move with them.
    """
    assert CURRENT_DISCOVERY.definition_ref == "agentic-discovery@v3-2026-08-20"
    assert CURRENT_DISCOVERY.config_fingerprint == (
        "fca219ad30643b26953c051edd887dbce744cd91201beca0a79e480771314fed"
    )


def test_retuning_a_filter_moves_the_definition_fingerprint():
    import dataclasses

    retuned = dataclasses.replace(
        CURRENT_DISCOVERY,
        base_filters={**CURRENT_DISCOVERY.base_filters, "rsi": {"values": [20, 55]}},
    )
    assert retuned.config_fingerprint != CURRENT_DISCOVERY.config_fingerprint


def test_every_shard_declares_the_stock_filter():
    """The Mega shard was created without it and repaired later; the definition
    and the live scans must agree on the declared universe."""
    assert CURRENT_DISCOVERY.base_filters["instrument_type"]["value"] == "STOCK"
    assert len(CURRENT_DISCOVERY.expected_shard_ids) == 8


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
        scan_definition_ref=CURRENT_DISCOVERY.definition_ref,
        scan_config_fingerprint=CURRENT_DISCOVERY.config_fingerprint,
        scan_config=CURRENT_DISCOVERY.as_config(),
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
    assert run["scan_definition_ref"] == "agentic-discovery@v3-2026-08-20"
    assert run["scan_config_fingerprint"] == CURRENT_DISCOVERY.config_fingerprint
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
    assert run["scan_definition_ref"] == "agentic-discovery@v3-2026-08-20"
    assert "truncated_candidate_count" in run  # obsolete column left in place


# --------------------------------------------------------------------------
# v3 topology: the eight-band re-shard, verified against live configuration
# captured on 2026-08-20.
#
# The fixture is real broker output, not a fixture derived from the definition
# it is checked against. A definition-derived fixture can only prove the drift
# checker is self-consistent; this one can fail if Legend and the repo diverge.
# --------------------------------------------------------------------------

_LIVE_SCANS = Path(__file__).parent / "fixtures" / "live_scans_2026-08-20.json"

PROBE_SCAN_ID = "6bdcb42f-c651-41b9-98a3-d56ac256cda0"
"""The original unsharded $2B+ scan. Retained in the account as a capability
probe artifact; it also carries no instrument-type filter, so it never belonged
in the declared universe."""


def _live_payload():
    return json.loads(_LIVE_SCANS.read_text(encoding="utf-8"))


def test_v3_declares_exactly_the_eight_verified_shards():
    assert CURRENT_DISCOVERY.expected_shard_ids == (
        "7e65d9ff-a2b9-4a82-9c84-bee57475ef50",  # B1  $2B-$3B
        "3eeeb183-4a96-442b-8b01-404f33978c2b",  # B2  $3B-$4.5B
        "3bf67683-fc75-4e67-a9f1-2a7500bb0256",  # B3  $4.5B-$7B
        "813dd065-f51b-47de-9eff-ef112295a3de",  # B4  $7B-$10B
        "bd7d315f-db05-4fda-b937-b31b1989ce24",  # B5  $10B-$17.5B
        "795e9148-25fa-4678-a2b6-13ced4cbb025",  # B6  $17.5B-$35B
        "cccffcd4-8c3d-452c-ba23-23c71308030e",  # B7  $35B-$100B
        "cc72022a-5f93-4c66-a69e-369dc6c89d92",  # B8  >$100B
    )


def test_the_unsharded_probe_is_not_part_of_the_universe():
    assert PROBE_SCAN_ID not in CURRENT_DISCOVERY.expected_shard_ids


def test_the_bands_are_contiguous_and_cover_without_gaps():
    """Adjacent BETWEEN bands must meet exactly: a gap silently drops a slice of
    the market, and no coverage check would ever notice."""
    bounds = [s.filters["market_cap"] for s in CURRENT_DISCOVERY.shards]
    for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
        top = lower["values"][1]
        floor = upper["values"][0] if "values" in upper else upper["value"]
        assert top == floor, f"gap or overlap between {top} and {floor}"
    assert bounds[0]["values"][0] == 2_000_000_000
    assert bounds[-1] == {"predicate": ">", "value": 100_000_000_000}


def test_live_configuration_matches_the_v3_definition():
    report = check_definition_drift(_live_payload(), CURRENT_DISCOVERY)
    assert report.as_dicts() == []
    assert report.status is DefinitionDriftStatus.MATCHES
    assert not report.has_blocking
    assert not report.blocks(any_shard_capped=True)


def test_a_removed_shard_blocks_against_live_configuration():
    payload = _live_payload()
    payload["data"]["scans"] = payload["data"]["scans"][:-1]
    report = check_definition_drift(payload, CURRENT_DISCOVERY)
    assert report.status is DefinitionDriftStatus.DRIFTED
    assert report.has_blocking


def test_a_changed_band_blocks_against_live_configuration():
    payload = _live_payload()
    for f in payload["data"]["scans"][0]["filter_summary"]:
        if f["filter_type_enum"] == "FILTER_TYPE_MARKET_CAP":
            f["values"] = ["1000000000", "3000000000"]
    report = check_definition_drift(payload, CURRENT_DISCOVERY)
    assert report.status is DefinitionDriftStatus.DRIFTED
    assert report.has_blocking


def test_regular_session_semantics_block_against_live_configuration():
    """all-session and regular-session RSI are different numbers for the same
    symbol; the change is invisible except in the expression."""
    payload = _live_payload()
    for f in payload["data"]["scans"][0]["filter_summary"]:
        if "session=" in str(f.get("expression", "")):
            f["expression"] = f["expression"].replace('session="all"', 'session="regular"')
    report = check_definition_drift(payload, CURRENT_DISCOVERY)
    assert report.status is DefinitionDriftStatus.DRIFTED
    assert report.has_blocking


def test_the_probe_scan_appearing_live_does_not_join_the_universe():
    """Extra saved scans are irrelevant: the definition names its shards, so an
    unrelated scan in the account cannot widen what we discover."""
    payload = _live_payload()
    payload["data"]["scans"].append({
        "scan_id": PROBE_SCAN_ID,
        "title": "AgenticTrader Discovery v1",
        "sorting": "Market cap desc",
        "cortex_managed": False,
        "filter_summary": [],
    })
    report = check_definition_drift(payload, CURRENT_DISCOVERY)
    assert report.status is DefinitionDriftStatus.MATCHES
