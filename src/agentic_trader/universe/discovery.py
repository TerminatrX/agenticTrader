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
    DiscoveryBatch,
    ScanCandidate,
    ShardResult,
)
from agentic_trader.universe.scanner_capabilities import (
    ROBINHOOD_MCP_SCANNER,
    ScannerCapabilities,
)


class DiscoverySource(Protocol):
    """Anything that can propose symbols for evaluation."""

    name: str

    def discover(self, now: datetime | None = None) -> DiscoveryBatch: ...


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

    Strict about malformed data, forgiving about missing data — the same
    contract `market.snapshot` follows. A row without a ticker is skipped
    rather than guessed at.
    """
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    result = data.get("result", data) if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        raise ValueError("scan payload has no result object")

    rows = result.get("results") or []
    candidates: list[ScanCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = row.get("ticker") or (row.get("columns") or {}).get("Symbol")
        if not ticker:
            continue
        columns = row.get("columns")
        candidates.append(
            ScanCandidate(
                symbol=str(ticker).strip().upper(),
                source=source,
                discovered_at=discovered_at,
                instrument_id=row.get("instrument_id"),
                # Diagnostic only. Never reaches build_snapshot.
                source_values={
                    str(k): str(v) for k, v in (columns or {}).items() if v is not None
                },
            )
        )

    reported = result.get("total_items")
    return ShardResult(
        scan_id=str(result.get("scan_id", "")),
        scan_title=str(result.get("scan_title", "")),
        returned_count=len(candidates),
        reported_total=int(reported) if isinstance(reported, int) else None,
        # Row count against the hard cap, never the reported total.
        coverage=capabilities.coverage_for(len(candidates)),
        candidates=tuple(candidates),
    )


class ScannerSource:
    """Union of several saved scans, deduplicated.

    Takes payloads the agent already fetched rather than fetching them, so this
    class is pure and testable. Multiple shards are the design from day one:
    the scanner caps every run at 200 rows with no pagination, so a single scan
    can only ever describe a universe it happens to fit inside.
    """

    name = "robinhood_scanner"

    def __init__(
        self,
        payloads: Sequence[Any],
        *,
        capabilities: ScannerCapabilities = ROBINHOOD_MCP_SCANNER,
    ) -> None:
        self.payloads = list(payloads)
        self.capabilities = capabilities

    def discover(self, now: datetime | None = None) -> DiscoveryBatch:
        at = now or datetime.now(UTC)
        shards = [
            parse_scan_payload(
                p, source=self.name, discovered_at=at, capabilities=self.capabilities
            )
            for p in self.payloads
        ]

        # Dedupe on instrument_id where present, symbol otherwise. Shard bands
        # are meant to be disjoint, but BETWEEN endpoint semantics are not
        # documented, so overlap is handled rather than assumed away.
        seen: dict[str, ScanCandidate] = {}
        returned = 0
        for shard in shards:
            for c in shard.candidates:
                returned += 1
                seen.setdefault(c.instrument_id or c.symbol, c)

        return DiscoveryBatch(
            source=self.name,
            started_at=at,
            shards=shards,
            candidates=list(seen.values()),
            returned_before_dedupe=returned,
        )
