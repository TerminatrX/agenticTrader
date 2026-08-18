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

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum


class CoverageStatus(StrEnum):
    """Whether a discovery run saw its whole declared universe.

    Three states rather than two, because "we could not tell" and "we know we
    did not look" are different failures and only one of them is ambiguous.
    """

    COMPLETE = "complete"
    """Every expected shard reported, and each returned fewer raw rows than the
    hard cap, so nothing was withheld."""

    UNKNOWN_TRUNCATED = "unknown_truncated"
    """A shard's raw row count reached the cap. Whether the universe was
    exactly that size or larger is unknowable: there is no pagination, and the
    broker's own `total_items` has been observed to disagree with itself across
    equivalent filter sets."""

    INCOMPLETE = "incomplete"
    """An expected shard was absent, unparseable, or carried an unidentifiable
    row. Distinct from UNKNOWN_TRUNCATED on purpose — here we *know* the
    declared universe was not fully queried, which is a stronger and more
    actionable claim than not knowing."""


class FunnelStage(StrEnum):
    """Where a candidate got to. Ordered from first to last."""

    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    SELECTED = "selected"
    ENRICHED = "enriched"
    EVALUATED = "evaluated"
    TRADED = "traded"


class ShardError(ValueError):
    """A scan payload was present but could not be interpreted."""


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

    stage: FunnelStage = FunnelStage.DISCOVERED
    exit_reason: str | None = None

    # Authoritative, fetched during this run — not from the source.
    sector: str | None = None

    def advanced_to(self, stage: FunnelStage) -> ScanCandidate:
        return replace(self, stage=stage)

    def dropped(self, reason: str) -> ScanCandidate:
        return replace(self, exit_reason=reason)

    def with_sector(self, sector: str | None) -> ScanCandidate:
        return replace(self, sector=sector)


@dataclass(frozen=True)
class ShardResult:
    """One saved scan's contribution to a discovery run.

    `returned_row_count` is the raw row count from the payload and is what
    coverage is judged on. `candidate_count` is how many of those rows carried
    a usable identity. They differ only when rows were rejected, and using the
    parsed count for coverage would let a malformed row disguise a capped
    response as a complete one: 200 raw rows minus one unusable is 199 parsed,
    which would otherwise read as "below the cap, therefore complete".
    """

    scan_id: str
    scan_title: str
    returned_row_count: int
    candidate_count: int
    rows_rejected: int
    reported_total: int | None
    coverage: CoverageStatus
    candidates: tuple[ScanCandidate, ...] = ()

    @property
    def coverage_complete(self) -> bool:
        return self.coverage is CoverageStatus.COMPLETE


@dataclass(frozen=True)
class CoverageReport:
    """Run-level coverage plus every reason it might not be COMPLETE.

    A single status is not enough to answer "why was August 18 incomplete?"
    six months later, so each failure mode is carried separately.
    """

    status: CoverageStatus
    missing_shard_ids: tuple[str, ...] = ()
    unexpected_shard_ids: tuple[str, ...] = ()
    duplicate_shard_ids: tuple[str, ...] = ()
    parse_error_count: int = 0

    @property
    def complete(self) -> bool:
        return self.status is CoverageStatus.COMPLETE

    def reasons(self) -> list[str]:
        out = []
        if self.missing_shard_ids:
            out.append(f"missing shards: {', '.join(self.missing_shard_ids)}")
        if self.unexpected_shard_ids:
            out.append(f"unexpected shards: {', '.join(self.unexpected_shard_ids)}")
        if self.duplicate_shard_ids:
            out.append(f"duplicate shards: {', '.join(self.duplicate_shard_ids)}")
        if self.parse_error_count:
            out.append(f"{self.parse_error_count} payload(s) failed to parse")
        return out


def resolve_coverage(
    shards: list[ShardResult],
    *,
    expected_shard_ids: tuple[str, ...],
    parse_error_count: int = 0,
) -> CoverageReport:
    """Require the observed shard set to *equal* the expected one.

    Checking only for missing ids is not enough. All five expected shards plus
    an unrelated sixth still satisfies "nothing missing", while that sixth
    scan's candidates quietly join the declared universe. A duplicated shard is
    equally an integration defect — it double-counts a band and inflates the
    dedupe statistics that are supposed to reveal boundary semantics.

    Missing shards are also checked independently of `all(...)`, because an
    empty shard list makes `all([])` true: a run that queried nothing would
    otherwise report complete coverage.
    """
    observed = [s.scan_id for s in shards]
    expected = set(expected_shard_ids)

    missing = tuple(sorted(expected - set(observed)))
    unexpected = tuple(sorted(set(observed) - expected))
    duplicates = tuple(sorted({sid for sid in observed if observed.count(sid) > 1}))

    # An empty shard list is checked explicitly. `all([])` is True, so a run
    # that queried nothing would otherwise fall through to COMPLETE.
    incomplete = (
        bool(missing or unexpected or duplicates or parse_error_count)
        or not shards
        or any(s.coverage is CoverageStatus.INCOMPLETE for s in shards)
    )
    if incomplete:
        status = CoverageStatus.INCOMPLETE
    elif any(s.coverage is CoverageStatus.UNKNOWN_TRUNCATED for s in shards):
        status = CoverageStatus.UNKNOWN_TRUNCATED
    else:
        status = CoverageStatus.COMPLETE

    return CoverageReport(
        status=status,
        missing_shard_ids=missing,
        unexpected_shard_ids=unexpected,
        duplicate_shard_ids=duplicates,
        parse_error_count=parse_error_count,
    )


@dataclass
class DiscoveryBatch:
    """Every candidate a discovery run produced, plus how it went.

    Deduplication resolves identity across shards. Shard boundaries are defined
    by market-cap bands whose endpoint semantics are not documented, so overlap
    is handled rather than trusted away — and `duplicates_removed` is recorded,
    because a non-zero count is evidence about those semantics rather than
    something to hide.
    """

    source: str
    started_at: datetime
    report: CoverageReport
    shards: list[ShardResult] = field(default_factory=list)
    candidates: list[ScanCandidate] = field(default_factory=list)
    returned_before_dedupe: int = 0

    @property
    def coverage(self) -> CoverageStatus:
        return self.report.status

    @property
    def missing_shard_ids(self) -> tuple[str, ...]:
        return self.report.missing_shard_ids

    @property
    def duplicates_removed(self) -> int:
        return self.returned_before_dedupe - len(self.candidates)

    @property
    def coverage_complete(self) -> bool:
        return self.report.complete


def funnel_counts(candidates: list[ScanCandidate]) -> dict[str, int]:
    """Stage histogram over exactly the candidates given.

    Must be computed from the collection actually persisted. Candidates are
    immutable, so selection returns new objects; counting the pre-selection
    batch would record every candidate as `discovered` while the stored rows
    said otherwise.
    """
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.stage.value] = counts.get(c.stage.value, 0) + 1
    return counts
