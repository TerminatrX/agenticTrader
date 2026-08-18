"""Where candidates come from.

Two sources, one protocol. Neither performs broker I/O — `src/` never does. The
agent calls `run_scan` through MCP and hands the raw payloads here, exactly as
it already does for quotes and indicators via `market.snapshot`. That keeps the
guarantee that no test, import, or stray call can reach the broker, and it makes
a discovery run replayable from stored payloads.

`StaticSource` carries essentially all unit testing: it needs no payloads, no
network, and no scanner contract, so tests of the funnel exercise the funnel
rather than the integration. `ScannerSource` appears only at the seam.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from agentic_trader.universe.candidate import (
    CoverageStatus,
    DiscoveryBatch,
    ScanCandidate,
    ShardError,
    ShardResult,
    resolve_coverage,
)
from agentic_trader.universe.scan_definition import ScanDefinition
from agentic_trader.universe.scanner_capabilities import (
    ROBINHOOD_MCP_SCANNER,
    ScannerCapabilities,
)


class DiscoverySource(Protocol):
    """Anything that can propose symbols for evaluation."""

    name: str

    def discover(self, now: datetime | None = None) -> DiscoveryBatch: ...


class IdentityConflict(ValueError):
    """One symbol arrived under two different instrument ids.

    Raised rather than resolved. Silently keeping one would mean the union is
    quietly wrong about what it discovered, and the two plausible causes — a
    ticker reused across instruments, or shards drawn from inconsistent
    snapshots — both need a human to look.
    """


class StaticSource:
    """A fixed symbol list. Coverage is complete by definition.

    The default source for tests and for running without any saved scan. It
    cannot discover anything not written down, which is exactly why it is
    useless in production and ideal in a test.
    """

    name = "static"

    def __init__(self, symbols: Iterable[str]) -> None:
        self.symbols = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))

    def discover(self, now: datetime | None = None) -> DiscoveryBatch:
        at = now or datetime.now(UTC)
        candidates = [
            ScanCandidate(symbol=s, source=self.name, discovered_at=at) for s in self.symbols
        ]
        return DiscoveryBatch(
            source=self.name,
            started_at=at,
            coverage=CoverageStatus.COMPLETE,
            shards=[],
            candidates=candidates,
            returned_before_dedupe=len(candidates),
        )


def parse_scan_payload(
    payload: Any,
    *,
    source: str,
    discovered_at: datetime,
    capabilities: ScannerCapabilities = ROBINHOOD_MCP_SCANNER,
) -> ShardResult:
    """Turn one raw `run_scan` response into a shard result.

    Coverage is judged on the **raw** row count, never the parsed candidate
    count. Rejecting one unidentifiable row out of a capped 200 would otherwise
    yield 199 parsed and a false claim of complete coverage.

    A row with no usable identity makes the shard `INCOMPLETE` rather than
    disappearing. This module is strict about malformed data by design, and a
    silently shortened shard is exactly the kind of degradation that reads as a
    quiet market rather than a parsing failure.
    """
    data = payload.get("data", payload) if isinstance(payload, dict) else None
    result = data.get("result", data) if isinstance(data, dict) else None
    if not isinstance(result, dict):
        raise ShardError("scan payload has no result object")

    rows = result.get("results")
    if not isinstance(rows, list):
        raise ShardError("scan payload has no results list")

    candidates: list[ScanCandidate] = []
    rejected = 0
    for row in rows:
        ticker = None
        if isinstance(row, dict):
            ticker = row.get("ticker") or (row.get("columns") or {}).get("Symbol")
        if not ticker:
            rejected += 1
            continue
        columns = row.get("columns") or {}
        candidates.append(
            ScanCandidate(
                symbol=str(ticker).strip().upper(),
                source=source,
                discovered_at=discovered_at,
                instrument_id=row.get("instrument_id"),
                # Diagnostic only. Never reaches build_snapshot.
                source_values={str(k): str(v) for k, v in columns.items() if v is not None},
            )
        )

    coverage = capabilities.coverage_for(len(rows))
    if rejected:
        coverage = CoverageStatus.INCOMPLETE

    reported = result.get("total_items")
    return ShardResult(
        scan_id=str(result.get("scan_id", "")),
        scan_title=str(result.get("scan_title", "")),
        returned_row_count=len(rows),
        candidate_count=len(candidates),
        rows_rejected=rejected,
        reported_total=int(reported) if isinstance(reported, int) else None,
        coverage=coverage,
        candidates=tuple(candidates),
    )


class ScannerSource:
    """Union of the saved scans a `ScanDefinition` declares.

    Takes payloads the agent already fetched rather than fetching them, so this
    class is pure and testable. Multiple shards are the design from day one:
    the scanner caps every run at 200 rows with no pagination, so a single scan
    can only ever describe a universe it happens to fit inside.

    The definition's shard set is *expected*. Receiving four of five shards is
    not complete coverage, and without a stated expectation four short shards
    are indistinguishable from a full sweep.
    """

    name = "robinhood_scanner"

    def __init__(
        self,
        payloads: Sequence[Any],
        *,
        definition: ScanDefinition | None = None,
        capabilities: ScannerCapabilities = ROBINHOOD_MCP_SCANNER,
    ) -> None:
        self.payloads = list(payloads)
        self.definition = definition
        self.capabilities = capabilities

    def discover(self, now: datetime | None = None) -> DiscoveryBatch:
        at = now or datetime.now(UTC)
        expected = self.definition.expected_shard_ids if self.definition else ()

        shards: list[ShardResult] = []
        for p in self.payloads:
            try:
                shards.append(
                    parse_scan_payload(
                        p, source=self.name, discovered_at=at, capabilities=self.capabilities
                    )
                )
            except ShardError:
                # A shard that could not be parsed is a shard we did not query.
                # Skipping it silently would let the run claim coverage it does
                # not have; the missing-id check below turns it into INCOMPLETE.
                continue

        candidates, returned = _dedupe([c for s in shards for c in s.candidates])
        coverage, missing = resolve_coverage(shards, expected_shard_ids=expected)

        return DiscoveryBatch(
            source=self.name,
            started_at=at,
            coverage=coverage,
            shards=shards,
            candidates=candidates,
            returned_before_dedupe=returned,
            missing_shard_ids=missing,
        )


def _dedupe(rows: list[ScanCandidate]) -> tuple[list[ScanCandidate], int]:
    """Resolve identity across shards on symbol *and* instrument id.

    Keying on `instrument_id or symbol` is not enough: the same stock arriving
    from one shard with an id and another without produces two different keys
    and survives as a duplicate. Symbol is the primary identity — it is always
    present — and the instrument id is checked for agreement rather than used
    as the key.
    """
    by_symbol: dict[str, ScanCandidate] = {}
    for c in rows:
        existing = by_symbol.get(c.symbol)
        if existing is None:
            by_symbol[c.symbol] = c
            continue
        if (
            existing.instrument_id
            and c.instrument_id
            and existing.instrument_id != c.instrument_id
        ):
            raise IdentityConflict(
                f"{c.symbol} appeared with two instrument ids "
                f"({existing.instrument_id} and {c.instrument_id}) — shards may be "
                "drawn from inconsistent snapshots, or the ticker is reused"
            )
        # Prefer the copy that carries an id, so identity is not lost to
        # whichever shard happened to be parsed first.
        if existing.instrument_id is None and c.instrument_id is not None:
            by_symbol[c.symbol] = c
    return list(by_symbol.values()), len(rows)
