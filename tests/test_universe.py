"""Discovery, coverage, and the enrichment budget.

The load-bearing test in this file is the last one: a candidate's
`source_values` can be arbitrarily wrong without changing any decision, because
only the symbol crosses into authoritative enrichment. Everything else here
protects the two properties that make discovery honest — coverage is proven
from row counts rather than a field observed to be unreliable, and the budget
spreads across sectors instead of reproducing the concentration the sector cap
exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from agentic_trader.market.snapshot import build_snapshot
from agentic_trader.universe import (
    BUDGET_EXHAUSTED,
    ROBINHOOD_MCP_SCANNER,
    CoverageStatus,
    FunnelStage,
    ScanCandidate,
    ScannerSource,
    StaticSource,
    parse_scan_payload,
    select_for_enrichment,
)

NOW = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)


def _payload(scan_id: str, tickers: list[str], total: int | None = None):
    return {
        "data": {
            "result": {
                "scan_id": scan_id,
                "scan_title": f"shard {scan_id}",
                "total_items": total if total is not None else len(tickers),
                "results": [
                    {
                        "ticker": t,
                        "instrument_id": f"iid-{t}",
                        "instrument_type": "EQUITY",
                        "columns": {"Symbol": t, "RSI": "40.0", "Market cap": "3.0e+09"},
                    }
                    for t in tickers
                ],
            }
        }
    }


def _cand(symbol: str, sector: str | None = None) -> ScanCandidate:
    return ScanCandidate(
        symbol=symbol, source="test", discovered_at=NOW, instrument_id=f"iid-{symbol}"
    ).with_sector(sector)


# ------------------------------------------------------------------ coverage


def test_a_short_result_set_proves_complete_coverage():
    shard = parse_scan_payload(_payload("s1", ["AAPL", "MSFT"]), source="x", discovered_at=NOW)

    assert shard.returned_count == 2
    assert shard.coverage is CoverageStatus.COMPLETE
    assert shard.coverage_complete


def test_a_full_result_set_is_unknown_not_complete():
    """200 rows is ambiguous: exactly-200 and truncated-from-more look identical.

    There is no pagination, so nothing can distinguish them — and the broker's
    own total cannot be used to try.
    """
    rows = [f"T{i}" for i in range(ROBINHOOD_MCP_SCANNER.max_rows_per_run)]
    shard = parse_scan_payload(_payload("s1", rows), source="x", discovered_at=NOW)

    assert shard.returned_count == 200
    assert shard.coverage is CoverageStatus.UNKNOWN_TRUNCATED
    assert not shard.coverage_complete


def test_coverage_ignores_the_reported_total():
    """total_items is not consulted, because it has been observed unreliable.

    Five disjoint shards over one filter definition summed to 668 while the
    unsharded equivalent reported 394.
    """
    honest = parse_scan_payload(_payload("s", ["A", "B"], total=2), source="x", discovered_at=NOW)
    lying = parse_scan_payload(_payload("s", ["A", "B"], total=9999), source="x", discovered_at=NOW)

    assert honest.coverage is lying.coverage is CoverageStatus.COMPLETE
    assert lying.reported_total == 9999  # recorded, never trusted


def test_a_run_is_complete_only_if_every_shard_is():
    full = [f"T{i}" for i in range(200)]
    batch = ScannerSource([_payload("a", ["AAPL"]), _payload("b", full)]).discover(now=NOW)

    assert batch.coverage is CoverageStatus.UNKNOWN_TRUNCATED
    assert not batch.coverage_complete


# ------------------------------------------------------------------- shards


def test_shards_are_unioned_and_deduplicated_by_instrument_id():
    batch = ScannerSource(
        [_payload("a", ["AAPL", "MSFT"]), _payload("b", ["MSFT", "NVDA"])]
    ).discover(now=NOW)

    assert sorted(c.symbol for c in batch.candidates) == ["AAPL", "MSFT", "NVDA"]
    assert batch.returned_before_dedupe == 4
    assert batch.duplicates_removed == 1


def test_disjoint_shards_remove_nothing():
    """A non-zero duplicate count is evidence about band boundaries, so zero
    is worth asserting rather than assuming."""
    batch = ScannerSource([_payload("a", ["AAPL"]), _payload("b", ["MSFT"])]).discover(now=NOW)

    assert batch.duplicates_removed == 0


def test_rows_without_a_ticker_are_skipped_not_guessed():
    payload = _payload("a", ["AAPL"])
    payload["data"]["result"]["results"].append({"instrument_id": "iid-x", "columns": {}})
    shard = parse_scan_payload(payload, source="x", discovered_at=NOW)

    assert [c.symbol for c in shard.candidates] == ["AAPL"]


def test_static_source_is_always_complete():
    batch = StaticSource(["aapl", " msft ", "AAPL"]).discover(now=NOW)

    assert [c.symbol for c in batch.candidates] == ["AAPL", "MSFT"]
    assert batch.coverage_complete


# ------------------------------------------------------- enrichment budget


def test_the_budget_spreads_across_sectors_instead_of_taking_a_global_top_n():
    """The concentration problem this selector exists to avoid.

    Twenty technology candidates and a handful elsewhere must not yield a
    technology-only budget — the sector gate would then block everything after
    the first entry, and the run would spend its whole budget to find one trade.
    """
    candidates = (
        [_cand(f"TECH{i}", "Technology") for i in range(20)]
        + [_cand(f"FIN{i}", "Financials") for i in range(5)]
        + [_cand(f"HLTH{i}", "Healthcare") for i in range(5)]
    )
    result = select_for_enrichment(candidates, budget=9)

    assert len(result.selected) == 9
    assert len(result.per_sector) == 3
    assert max(result.per_sector.values()) <= 5


def test_one_sector_cannot_consume_the_whole_budget():
    result = select_for_enrichment([_cand(f"T{i}", "Technology") for i in range(50)], budget=10)

    assert result.per_sector["Technology"] <= 5
    assert len(result.selected) <= 5


def test_a_single_sector_still_fills_up_to_its_cap():
    result = select_for_enrichment([_cand(f"T{i}", "Technology") for i in range(3)], budget=10)
    assert len(result.selected) == 3


def test_unexamined_candidates_are_recorded_with_a_budget_reason():
    """A compute limit must never read as a strategy rejection."""
    result = select_for_enrichment([_cand(f"T{i}", f"S{i % 4}") for i in range(40)], budget=8)

    assert len(result.selected) == 8
    assert result.deferred_count == 32
    assert all(c.exit_reason == BUDGET_EXHAUSTED for c in result.deferred)


def test_selected_candidates_advance_a_stage():
    result = select_for_enrichment([_cand("AAPL", "Technology")], budget=5)
    assert result.selected[0].stage is FunnelStage.SELECTED


def test_unknown_sector_is_ranked_last_but_not_dropped():
    known = [_cand(f"K{i}", "Technology") for i in range(2)]
    unknown = [_cand("U1"), _cand("U2")]
    result = select_for_enrichment(known + unknown, budget=4)

    assert len(result.selected) == 4
    assert "unknown" in result.per_sector


def test_selection_is_deterministic():
    """A discovery run must replay identically from stored payloads."""
    c = [_cand(f"T{i}", f"S{i % 3}") for i in range(30)]
    a = select_for_enrichment(list(c), budget=7)
    b = select_for_enrichment(list(c), budget=7)

    assert [x.symbol for x in a.selected] == [x.symbol for x in b.selected]


def test_a_zero_budget_selects_nothing():
    result = select_for_enrichment([_cand("AAPL", "Technology")], budget=0)
    assert result.selected == []
    assert result.deferred_count == 1


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

    Two candidates for the same symbol carrying contradictory scanner values —
    RSI 12 vs 99, market cap $9T vs $1 — produce byte-identical snapshots,
    because only `symbol` crosses into enrichment and every value a decision
    uses is re-fetched. The scanner may be stale, wrong, or computed over a
    different session; none of it can reach a trade.
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
        build_snapshot(
            c.symbol, quote=QUOTE, indicators=INDICATORS, captured_at=NOW
        )
        for c in (honest, lying)
    ]

    assert snapshots[0] == snapshots[1]
    assert snapshots[0].last_price == Decimal("306.44")
    assert snapshots[0].indicators.rsi_14 == pytest.approx(41.0)


def test_source_values_never_appear_in_a_snapshot():
    """Structural, not conventional: build_snapshot takes a symbol and payloads.

    There is no parameter through which a candidate's scanner columns could be
    passed, which is what makes the guarantee hold without relying on care.
    """
    import inspect

    params = set(inspect.signature(build_snapshot).parameters)
    assert "source_values" not in params
    assert "candidate" not in params
    assert params == {
        "symbol", "quote", "historicals", "fundamentals",
        "earnings", "indicators", "captured_at",
    }


# ------------------------------------------------------ capability profile


def test_the_scanner_profile_is_separate_from_the_broker_profile():
    from agentic_trader.execution.capabilities import ROBINHOOD_MCP

    assert ROBINHOOD_MCP_SCANNER.profile_ref == "robinhood-mcp-scanner@2026-08-18"
    assert ROBINHOOD_MCP_SCANNER.profile_ref != ROBINHOOD_MCP.profile_ref
    assert ROBINHOOD_MCP_SCANNER.content_fingerprint != ROBINHOOD_MCP.content_fingerprint


def test_the_scanner_fingerprint_covers_semantic_content_not_just_booleans():
    """A vocabulary change must move the hash.

    Hashing only Capability fields would let the instrument-type vocabulary or
    the row cap change while profile_ref stayed put, so a stored reference
    would denote two different contracts.
    """
    import dataclasses

    base = ROBINHOOD_MCP_SCANNER.content_fingerprint
    for field_name, value in [
        ("max_rows_per_run", 500),
        ("filter_instrument_type", "EQUITY"),
        ("default_sorting", "Volume desc"),
        ("default_row_fields", ("Symbol",)),
    ]:
        altered = dataclasses.replace(ROBINHOOD_MCP_SCANNER, **{field_name: value})
        assert altered.content_fingerprint != base, field_name


def test_the_vocabulary_mismatch_is_recorded():
    """Row says EQUITY, filter wants STOCK. A filter built from the row
    vocabulary validates and silently matches nothing."""
    assert ROBINHOOD_MCP_SCANNER.row_instrument_type == "EQUITY"
    assert ROBINHOOD_MCP_SCANNER.filter_instrument_type == "STOCK"


def test_freshness_is_recorded_as_absent_but_is_not_a_permission():
    cap = ROBINHOOD_MCP_SCANNER.row_freshness_timestamp
    assert cap.supported is False
    assert cap.is_certain  # observed, not inferred
    assert not cap.usable


# ------------------------------------------------------------------ journal


def test_a_discovery_run_persists_every_candidate_including_the_unexamined(tmp_path):
    """Deferred candidates must survive to the journal.

    A funnel recording only survivors cannot answer why a day produced no
    trades: it conflates "nothing qualified" with "we ran out of budget".
    """
    from agentic_trader.journal import JournalRepository

    batch = ScannerSource([_payload("a", ["AAPL", "MSFT", "NVDA"])]).discover(now=NOW)
    batch.candidates = [c.with_sector("Technology") for c in batch.candidates]
    result = select_for_enrichment(batch.candidates, budget=2, max_per_sector=2)

    repo = JournalRepository(tmp_path / "j.db")
    repo.record_scan_run(
        "run-1",
        batch,
        candidates=[*result.selected, *result.deferred],
        scanner_profile_ref=ROBINHOOD_MCP_SCANNER.profile_ref,
        scan_config={"rsi": [25, 50]},
        selected_count=len(result.selected),
        truncated_count=result.deferred_count,
    )

    (run,) = repo.scan_runs()
    assert run["unique_discovered"] == 3
    assert run["selected_for_enrichment"] == 2
    assert run["truncated_candidate_count"] == 1
    assert run["coverage_complete"] == 1
    assert run["scanner_profile_ref"] == "robinhood-mcp-scanner@2026-08-18"

    stored = repo.scan_candidates("run-1")
    assert len(stored) == 3
    assert sum(1 for c in stored if c["exit_reason"] == BUDGET_EXHAUSTED) == 1
