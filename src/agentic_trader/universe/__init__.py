"""Candidate discovery: which symbols are worth evaluating today.

Discovery proposes; it never decides. Only a candidate's symbol crosses into
authoritative enrichment — every value a trade depends on is re-fetched through
`market.snapshot`, so a discovery source may be stale or wrong without any of it
reaching a decision.
"""

from agentic_trader.universe.candidate import (
    CoverageStatus,
    DiscoveryBatch,
    FunnelStage,
    ScanCandidate,
    ShardResult,
)
from agentic_trader.universe.discovery import (
    DiscoverySource,
    ScannerSource,
    StaticSource,
    parse_scan_payload,
)
from agentic_trader.universe.scanner_capabilities import (
    DISCOVERY_SHARD_IDS,
    ROBINHOOD_MCP_SCANNER,
    ScannerCapabilities,
)
from agentic_trader.universe.selection import (
    BUDGET_EXHAUSTED,
    SelectionResult,
    select_for_enrichment,
)

__all__ = [
    "BUDGET_EXHAUSTED",
    "DISCOVERY_SHARD_IDS",
    "ROBINHOOD_MCP_SCANNER",
    "CoverageStatus",
    "DiscoveryBatch",
    "DiscoverySource",
    "FunnelStage",
    "ScanCandidate",
    "ScannerCapabilities",
    "ScannerSource",
    "SelectionResult",
    "ShardResult",
    "StaticSource",
    "parse_scan_payload",
    "select_for_enrichment",
]
