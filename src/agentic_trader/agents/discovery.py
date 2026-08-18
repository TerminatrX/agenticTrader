"""One discovery run, as a pure function.

Discovery is deliberately a *separate phase* from evaluation, because the two
have different data costs. Everything here works on batched payloads the agent
already fetched — `get_scans`, five `run_scan` responses, and batched
fundamentals at ten symbols per call. Evaluation is what costs seven
single-symbol calls per candidate, so this phase exists to decide which
candidates are worth paying that for.

The output is a symbol list. The agent then fetches enrichment for exactly
those symbols and runs the existing `evaluate` cycle per symbol — unchanged.
No broker I/O happens here, as everywhere in `src/`.

Order matters and is enforced:

    drift check  ->  blocking drift stops the run before any scan is read
    coverage     ->  sort drift escalates here, once caps are known
    sector       ->  authoritative fundamentals, never scanner columns
    selection    ->  diversity-aware, bounded by the enrichment budget

A run that stops early still journals its full funnel. "We refused to look" and
"we looked and found nothing" must never be confusable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from agentic_trader.market.snapshot import parse_sectors
from agentic_trader.universe.candidate import (
    CoverageStatus,
    DiscoveryBatch,
    FunnelStage,
    ScanCandidate,
    funnel_counts,
)
from agentic_trader.universe.discovery import ScannerSource
from agentic_trader.universe.drift import (
    DefinitionDriftStatus,
    DriftReport,
    check_definition_drift,
)
from agentic_trader.universe.scan_definition import ScanDefinition
from agentic_trader.universe.scanner_capabilities import (
    ROBINHOOD_MCP_SCANNER,
    ScannerCapabilities,
)
from agentic_trader.universe.selection import SelectionResult, select_for_enrichment

DRIFT_ABORT = "scan_definition_drift"
"""Exit reason for candidates on a run stopped by membership-affecting drift.

Distinct from every rejection reason: nothing about these candidates was
evaluated. The universe itself was untrustworthy.
"""


@dataclass
class DiscoveryResult:
    """Everything one discovery run produced, including why it stopped."""

    run_id: str
    trading_date: date
    started_at: datetime
    definition_ref: str
    config_fingerprint: str
    scanner_profile_ref: str

    drift: DriftReport = field(default_factory=DriftReport)
    batch: DiscoveryBatch | None = None
    selection: SelectionResult | None = None
    aborted_reason: str | None = None

    @property
    def coverage(self) -> CoverageStatus:
        return self.batch.coverage if self.batch else CoverageStatus.INCOMPLETE

    @property
    def drift_status(self) -> DefinitionDriftStatus:
        return self.drift.status

    @property
    def selected_symbols(self) -> list[str]:
        """The only output the agent acts on.

        Empty whenever the run aborted, which is what makes the drift boundary
        structural: there is nothing to enrich, so no enrichment call can be
        made regardless of what a caller intends.
        """
        return [c.symbol for c in self.selection.selected] if self.selection else []

    @property
    def all_candidates(self) -> list[ScanCandidate]:
        if self.selection:
            return [*self.selection.selected, *self.selection.deferred]
        return list(self.batch.candidates) if self.batch else []

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trading_date": self.trading_date.isoformat(),
            "definition_ref": self.definition_ref,
            "coverage": self.coverage.value,
            "coverage_reasons": self.batch.report.reasons() if self.batch else [],
            "drift_status": self.drift_status.value,
            "drift_findings": self.drift.as_dicts(),
            "aborted_reason": self.aborted_reason,
            "funnel": funnel_counts(self.all_candidates),
            "discovered": len(self.batch.candidates) if self.batch else 0,
            "duplicates_removed": self.batch.duplicates_removed if self.batch else 0,
            "selected": self.selected_symbols,
        }


def run_discovery(
    *,
    scans_payload: Any,
    run_payloads: list[Any],
    fundamentals_payloads: list[Any] | None = None,
    definition: ScanDefinition,
    trading_date: date,
    budget: int = 25,
    max_per_sector: int | None = None,
    capabilities: ScannerCapabilities = ROBINHOOD_MCP_SCANNER,
    now: datetime | None = None,
    run_id: str | None = None,
) -> DiscoveryResult:
    """Discover, verify, and select — without evaluating anything."""
    at = now or datetime.now(UTC)
    result = DiscoveryResult(
        run_id=run_id or uuid.uuid4().hex[:12],
        trading_date=trading_date,
        started_at=at,
        definition_ref=definition.definition_ref,
        config_fingerprint=definition.config_fingerprint,
        scanner_profile_ref=capabilities.profile_ref,
    )

    # 1. Drift. Checked first and against the live configuration, so a widened
    # filter stops the run before a single scan result is even interpreted.
    result.drift = check_definition_drift(scans_payload, definition)
    if result.drift.has_blocking:
        result.aborted_reason = DRIFT_ABORT
        return result

    # 2. Coverage.
    result.batch = ScannerSource(
        run_payloads, definition, capabilities=capabilities
    ).discover(now=at)

    # Sort drift becomes membership-affecting exactly when a shard caps, which
    # is only knowable now — still before anything expensive.
    any_capped = any(
        s.returned_row_count >= capabilities.max_rows_per_run for s in result.batch.shards
    )
    if result.drift.blocks(any_shard_capped=any_capped):
        result.aborted_reason = DRIFT_ABORT
        return result

    # 3. Sector, from authoritative fundamentals. Scanner columns are never
    # consulted — they are diagnostic, and computed over a different session.
    sectors = parse_sectors(fundamentals_payloads or [])
    enriched = [
        c.with_sector(sectors.get(c.symbol)).advanced_to(FunnelStage.ELIGIBLE)
        for c in result.batch.candidates
    ]

    # 4. Selection, bounded by the enrichment budget.
    result.selection = select_for_enrichment(
        enriched, on=trading_date, budget=budget, max_per_sector=max_per_sector
    )
    return result


def abort_candidates(result: DiscoveryResult) -> list[ScanCandidate]:
    """Candidates of an aborted run, tagged so the funnel says what happened."""
    if not result.aborted_reason or not result.batch:
        return result.all_candidates
    return [c.dropped(result.aborted_reason) for c in result.batch.candidates]
