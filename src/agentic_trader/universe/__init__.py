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
    ShardError,
    ShardResult,
    funnel_counts,
    resolve_coverage,
)
from agentic_trader.universe.discovery import (
    DiscoverySource,
    IdentityConflict,
    ScannerSource,
    StaticSource,
    parse_scan_payload,
)
from agentic_trader.universe.scan_definition import (
    DISCOVERY_V1,
    ScanDefinition,
    ShardSpec,
)
from agentic_trader.universe.scanner_capabilities import (
    ROBINHOOD_MCP_SCANNER,
    ScannerCapabilities,
)
from agentic_trader.universe.selection import (
    BUDGET_EXHAUSTED,
    UNKNOWN_SECTOR_LIMIT,
    SelectionResult,
    rotation_key,
    select_for_enrichment,
)

__all__ = [
    "BUDGET_EXHAUSTED",
    "DISCOVERY_V1",
    "ROBINHOOD_MCP_SCANNER",
    "UNKNOWN_SECTOR_LIMIT",
    "CoverageStatus",
    "DiscoveryBatch",
    "DiscoverySource",
    "FunnelStage",
    "IdentityConflict",
    "ScanCandidate",
    "ScanDefinition",
    "ScannerCapabilities",
    "ScannerSource",
    "SelectionResult",
    "ShardError",
    "ShardResult",
    "ShardSpec",
    "StaticSource",
    "funnel_counts",
    "parse_scan_payload",
    "resolve_coverage",
    "rotation_key",
    "select_for_enrichment",
]
