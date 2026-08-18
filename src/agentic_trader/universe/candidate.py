"""Discovery types: what a candidate is, and where it left the funnel.

A candidate is a *symbol under consideration*, not a decision. The single most
important property in this package is that only `symbol` crosses into
authoritative enrichment — `source_values` is diagnostic and never an input to
any trading decision. That is what allows the scanner to be wrong, stale, or
computed over a different session than our own indicators without any of it
reaching a trade.

The funnel is recorded in full, including candidates that never got examined.
A count of what was dropped for budget reasons is not the same as a count of
what the strategy rejected, and a performance review that cannot tell those
apart will mistake a compute limit for a lack of opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class CoverageStatus(StrEnum):
    """Whether a discovery run saw its whole declared universe."""

    COMPLETE = "complete"
    """Fewer rows came back than the hard cap, so nothing was withheld."""

    UNKNOWN_TRUNCATED = "unknown_truncated"
    """The row count reached the cap. Whether the universe was exactly that
    size or larger is unknowable: there is no pagination, and the broker's own
    `total_items` has been observed to disagree with itself across equivalent
    filter sets. Deliberately not `INCOMPLETE` — the honest claim is that
    coverage is unknown, not that it is known to be partial."""


class FunnelStage(StrEnum):
    """Where a candidate got to. Ordered from first to last."""

    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    SELECTED = "selected"
    ENRICHED = "enriched"
    EVALUATED = "evaluated"
    TRADED = "traded"


@dataclass(frozen=True)
class ScanCandidate:
    """One symbol a discovery source proposed for evaluation.

    `source_values` holds whatever the source reported — scanner column cells,
    say. It exists for funnel analysis and debugging and must never reach
    `build_snapshot`. Recall the scanner computes RSI over `session="all"`
    while our own indicators use regular hours, so its numbers are not merely
    less fresh than ours, they answer a different question.
    """

    symbol: str
    source: str
    discovered_at: datetime
    instrument_id: str | None = None
    source_values: dict[str, str] = field(default_factory=dict)

    # Filled in as the candidate moves through, or stops.
    stage: FunnelStage = FunnelStage.DISCOVERED
    exit_reason: str | None = None

    # Authoritative, fetched during this run — not from the source.
    sector: str | None = None

    def advanced_to(self, stage: FunnelStage) -> ScanCandidate:
        from dataclasses import replace

        return replace(self, stage=stage)

    def dropped(self, reason: str) -> ScanCandidate:
        from dataclasses import replace

        return replace(self, exit_reason=reason)

    def with_sector(self, sector: str | None) -> ScanCandidate:
        from dataclasses import replace

        return replace(self, sector=sector)


@dataclass(frozen=True)
class ShardResult:
    """One saved scan's contribution to a discovery run."""

    scan_id: str
    scan_title: str
    returned_count: int
    reported_total: int | None
    coverage: CoverageStatus
    candidates: tuple[ScanCandidate, ...] = ()

    @property
    def coverage_complete(self) -> bool:
        return self.coverage is CoverageStatus.COMPLETE


@dataclass
class DiscoveryBatch:
    """Every candidate a discovery run produced, plus how it went.

    Deduplication is by `instrument_id` where available, falling back to
    `symbol`. Shard boundaries are defined by market-cap bands whose endpoint
    semantics are not documented, so overlap is assumed rather than trusted
    away — and `duplicates_removed` is recorded, because a non-zero count is
    evidence about those semantics rather than something to hide.
    """

    source: str
    started_at: datetime
    shards: list[ShardResult] = field(default_factory=list)
    candidates: list[ScanCandidate] = field(default_factory=list)
    returned_before_dedupe: int = 0

    @property
    def duplicates_removed(self) -> int:
        return self.returned_before_dedupe - len(self.candidates)

    @property
    def coverage(self) -> CoverageStatus:
        """A run is complete only if every shard was complete."""
        if all(s.coverage_complete for s in self.shards):
            return CoverageStatus.COMPLETE
        return CoverageStatus.UNKNOWN_TRUNCATED

    @property
    def coverage_complete(self) -> bool:
        return self.coverage is CoverageStatus.COMPLETE

    def counts(self) -> dict[str, int]:
        by_stage: dict[str, int] = {}
        for c in self.candidates:
            by_stage[c.stage.value] = by_stage.get(c.stage.value, 0) + 1
        return by_stage
