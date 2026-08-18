"""Discovery, coverage, and the enrichment budget.

The load-bearing test here is the trust boundary: a candidate's `source_values`
can be arbitrarily wrong without changing any decision, because only the symbol
crosses into authoritative enrichment.

Most of the rest defend against a single failure mode — claiming to have seen
more of the universe than we did. Coverage can be overstated by a missing
shard, by an unparseable shard, by a malformed row shortening a capped result
below the cap, or by trusting a broker field observed to contradict itself.
Each has a test, because every one of them reads downstream as "the market was
quiet today".
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

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


def _payload(scan_id: str, tickers, total: int | None = None, extra_rows=()):
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


def _cand(symbol: str, sector: str | None = None) -> ScanCandidate:
    return ScanCandidate(
        symbol=symbol, source="test", discovered_at=NOW, instrument_id=f"iid-{symbol}"
    ).with_sector(sector)


# ------------------------------------------------------- coverage: shard set


def test_zero_shards_is_incomplete_not_complete():
    """`all([])` is True, so an empty run would otherwise claim full coverage —
    the most dangerous possible wrong answer here."""
    batch = ScannerSource([], definition=DISCOVERY_V1).discover(now=NOW)

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert not batch.coverage_complete
    assert len(batch.missing_shard_ids) == 5


def test_four_of_five_shards_is_incomplete_even_when_each_is_short():
    batch = ScannerSource(_all_shards()[:4], definition=DISCOVERY_V1).discover(now=NOW)

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert batch.missing_shard_ids == (SHARDS[4],)


def test_all_five_short_shards_are_complete():
    batch = ScannerSource(_all_shards(), definition=DISCOVERY_V1).discover(now=NOW)

    assert batch.coverage is CoverageStatus.COMPLETE
    assert batch.missing_shard_ids == ()
    assert len(batch.candidates) == 5


def test_an_unparseable_shard_does_not_silently_vanish():
    payloads = [*_all_shards()[:4], {"data": {"result": {"scan_id": SHARDS[4]}}}]
    batch = ScannerSource(payloads, definition=DISCOVERY_V1).discover(now=NOW)

    assert batch.coverage is CoverageStatus.INCOMPLETE
    assert SHARDS[4] in batch.missing_shard_ids


def test_a_capped_shard_makes_the_run_unknown_truncated():
    full = _payload(SHARDS[0], [f"T{i}" for i in range(200)])
    payloads = [full, *_all_shards()[1:]]
    batch = ScannerSource(payloads, definition=DISCOVERY_V1).discover(now=NOW)

    assert batch.coverage is CoverageStatus.UNKNOWN_TRUNCATED


# --------------------------------------------------- coverage: the row count


def test_coverage_uses_raw_rows_not_parsed_candidates():
    """200 raw rows with one unusable parses to 199 candidates.

    Judging coverage on the parsed count would call that COMPLETE, disguising
    a response that actually hit the cap.
    """
    rows = [f"T{i}" for i in range(199)]
    shard = parse_scan_payload(
        _payload("s", rows, extra_rows=[{"instrument_id": "iid-x", "columns": {}}]),
        source="x",
        discovered_at=NOW,
    )

    assert shard.returned_row_count == 200
    assert shard.candidate_count == 199
    assert shard.rows_rejected == 1
    assert shard.coverage is not CoverageStatus.COMPLETE


def test_a_row_without_identity_makes_the_shard_incomplete():
    shard = parse_scan_payload(
        _payload("s", ["AAPL"], extra_rows=[{"columns": {"RSI": "40"}}]),
        source="x",
        discovered_at=NOW,
    )

    assert shard.coverage is CoverageStatus.INCOMPLETE
    assert shard.rows_rejected == 1
    assert [c.symbol for c in shard.candidates] == ["AAPL"]


def test_a_full_result_set_is_unknown_not_complete():
    """Exactly-200 and truncated-from-more are indistinguishable without
    pagination, and there is none."""
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


# ------------------------------------------------------------------- dedupe


def test_the_same_symbol_dedupes_across_shards():
    batch = ScannerSource(
        [_payload("a", ["AAPL", "MSFT"]), _payload("b", ["MSFT", "NVDA"])]
    ).discover(now=NOW)

    assert sorted(c.symbol for c in batch.candidates) == ["AAPL", "MSFT", "NVDA"]
    assert batch.returned_before_dedupe == 4
    assert batch.duplicates_removed == 1


def test_a_missing_instrument_id_still_dedupes():
    """Keying on `instrument_id or symbol` would give one copy the key "abc"
    and the other "AAPL", so the duplicate would survive."""
    with_id = _payload("a", ["AAPL"])
    without = _payload("b", ["AAPL"])
    without["data"]["result"]["results"][0]["instrument_id"] = None

    batch = ScannerSource([with_id, without]).discover(now=NOW)

    assert len(batch.candidates) == 1
    assert batch.duplicates_removed == 1
    assert batch.candidates[0].instrument_id == "iid-AAPL"  # identity preserved


def test_conflicting_instrument_ids_fail_loudly():
    a = _payload("a", ["AAPL"])
    b = _payload("b", ["AAPL"])
    b["data"]["result"]["results"][0]["instrument_id"] = "different-id"

    with pytest.raises(IdentityConflict, match="two instrument ids"):
        ScannerSource([a, b]).discover(now=NOW)


def test_disjoint_shards_remove_nothing():
    batch = ScannerSource([_payload("a", ["AAPL"]), _payload("b", ["MSFT"])]).discover(now=NOW)
    assert batch.duplicates_removed == 0


def test_static_source_is_always_complete():
    batch = StaticSource(["aapl", " msft ", "AAPL"]).discover(now=NOW)

    assert [c.symbol for c in batch.candidates] == ["AAPL", "MSFT"]
    assert batch.coverage_complete


# ------------------------------------------------------- enrichment budget


def _sel(cands, **kw):
    kw.setdefault("on", DAY)
    return select_for_enrichment(cands, **kw)


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
    """The budget was not exhausted, so saying so would be false."""
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
    assert result.per_sector["Technology"] >= 12


def test_selected_candidates_advance_a_stage():
    result = _sel([_cand("AAPL", "Technology")], budget=5)
    assert result.selected[0].stage is FunnelStage.SELECTED


def test_a_zero_budget_selects_nothing():
    result = _sel([_cand("AAPL", "Technology")], budget=0)
    assert result.selected == []
    assert result.deferred_count == 1


# ------------------------------------------------------------- rotation


def test_selection_is_deterministic_for_one_date():
    c = [_cand(f"T{i}", f"S{i % 3}") for i in range(40)]
    a = _sel(list(c), budget=7)
    b = _sel(list(c), budget=7)

    assert [x.symbol for x in a.selected] == [x.symbol for x in b.selected]


def test_selection_rotates_across_dates():
    """Sorting by symbol is deterministic but biased: run daily, alphabetically
    early tickers would consume the budget every time, replacing the broker's
    market-cap sampling bias with an alphabet one."""
    c = [_cand(f"T{i:02}", "Technology") for i in range(60)]
    day1 = {x.symbol for x in _sel(list(c), budget=10, on=date(2026, 8, 18)).selected}
    day2 = {x.symbol for x in _sel(list(c), budget=10, on=date(2026, 8, 19)).selected}

    assert day1 != day2


def test_rotation_key_is_stable_across_processes():
    """hashlib, not hash() — the builtin is salted per process and would not
    replay."""
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

    Two candidates for one symbol carrying contradictory scanner values — RSI
    12 vs 99, market cap $9T vs $1 — produce identical snapshots, because only
    `symbol` crosses into enrichment and every value a decision uses is
    re-fetched.
    """
    honest = ScanCandidate(
        symbol="AAPL", source="scanner", discovered_at=NOW,
        source_values={"RSI": "41.0", "Market cap": "4.4e+12", "Last": "306.44"},
    )
    lying = ScanCandidate(
        symbol="AAPL", source="scanner", discovered_at=NOW,
        source_values={"RSI": "99.0", "Market cap": "1", "Last": "0.01"},
    )

    snapshots = [
        build_snapshot(c.symbol, quote=QUOTE, indicators=INDICATORS, captured_at=NOW)
        for c in (honest, lying)
    ]

    assert snapshots[0] == snapshots[1]
    assert snapshots[0].last_price == Decimal("306.44")
    assert snapshots[0].indicators.rsi_14 == pytest.approx(41.0)


def test_source_values_never_appear_in_a_snapshot():
    """Structural, not conventional: there is no parameter through which a
    candidate's scanner columns could be passed."""
    import inspect

    params = set(inspect.signature(build_snapshot).parameters)
    assert params == {
        "symbol", "quote", "historicals", "fundamentals",
        "earnings", "indicators", "captured_at",
    }


# ------------------------------------------------------ capability / config


def test_the_scanner_profile_describes_the_endpoint_only():
    """Scan-specific configuration belongs to the definition, not the profile.

    "Asset type" appearing in the returned columns is an artifact of how these
    scans were configured; it is not an invariant of run_scan.
    """
    fields = {f for f in ROBINHOOD_MCP_SCANNER.__dataclass_fields__}

    assert "default_row_fields" not in fields
    assert "default_sorting" not in fields
    assert "Asset type" in DISCOVERY_V1.columns


def test_profiles_and_definitions_are_versioned_independently():
    from agentic_trader.execution.capabilities import ROBINHOOD_MCP

    refs = {
        ROBINHOOD_MCP.profile_ref,
        ROBINHOOD_MCP_SCANNER.profile_ref,
        DISCOVERY_V1.definition_ref,
    }
    assert len(refs) == 3


def test_the_scanner_fingerprint_covers_semantics_not_just_booleans():
    import dataclasses

    base = ROBINHOOD_MCP_SCANNER.content_fingerprint
    for name, value in [("max_rows_per_run", 500), ("filter_instrument_type", "EQUITY")]:
        altered = dataclasses.replace(ROBINHOOD_MCP_SCANNER, **{name: value})
        assert altered.content_fingerprint != base, name


def test_retuning_a_filter_moves_the_definition_fingerprint():
    """A run recorded under RSI 25-50 must stay interpretable after a retune."""
    import dataclasses

    retuned = dataclasses.replace(
        DISCOVERY_V1,
        base_filters={**DISCOVERY_V1.base_filters, "rsi": {"values": [20, 55]}},
    )
    assert retuned.config_fingerprint != DISCOVERY_V1.config_fingerprint


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


def test_stored_funnel_counts_match_the_stored_candidate_stages(tmp_path):
    """Candidates are immutable, so selection returns new objects. Counting the
    pre-selection batch would record everything as `discovered` while the rows
    said otherwise.
    """
    from agentic_trader.journal import JournalRepository

    batch = ScannerSource([_payload("a", ["AAPL", "MSFT", "NVDA"])]).discover(now=NOW)
    batch.candidates = [c.with_sector("Technology") for c in batch.candidates]
    result = _sel(batch.candidates, budget=2, max_per_sector=2)
    persisted = [*result.selected, *result.deferred]

    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run(
        "run-1",
        batch,
        candidates=persisted,
        scanner_profile_ref=ROBINHOOD_MCP_SCANNER.profile_ref,
        scan_config=DISCOVERY_V1.as_config(),
        selected_count=len(result.selected),
        budget_deferred_count=result.budget_deferred_count,
    )

    import json

    (run,) = repo.scan_runs()
    stored = repo.scan_candidates("run-1")

    assert json.loads(run["funnel_counts_json"]) == funnel_counts(persisted)
    assert json.loads(run["funnel_counts_json"]) == {"selected": 2, "discovered": 1}
    from collections import Counter

    assert Counter(c["stage"] for c in stored) == Counter({"selected": 2, "discovered": 1})
    assert run["unique_discovered"] == 3
    assert run["budget_deferred_count"] == 1
