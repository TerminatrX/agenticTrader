"""Choosing which candidates are worth expensive enrichment.

Enrichment costs about seven single-symbol MCP calls per candidate — six
indicators plus earnings, none of which batch. A few hundred candidates is
therefore not a slow run, it is an impossible one, and the budget has to bind
somewhere. This module decides where.

**This is a compute-budget policy, not a risk control.** It decides which
symbols get looked at, and nothing else. `max_sector_exposure_pct` in
`risk.limits` remains the only authority on how much of one sector the account
may actually hold; the two must not be confused, and a change here can never
loosen a risk limit. The naming reflects that: nothing in this module says
"exposure".

Selection is diversity-aware for a concrete reason. Ranking every candidate
globally and taking the top N reproduces the concentration problem the sector
cap exists to solve — a technology-heavy list yields a technology-heavy budget,
the sector gate then blocks all but the first entry, and the run does a lot of
work to find one trade.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from agentic_trader.universe.candidate import FunnelStage, ScanCandidate

UNKNOWN_SECTOR = "unknown"

BUDGET_EXHAUSTED = "enrichment_budget"
"""Exit reason for a candidate that was never examined.

Distinct from any strategy or risk rejection on purpose. A performance review
that cannot separate "the strategy declined this" from "we ran out of budget
before looking" will read a compute limit as an absence of opportunity.
"""


@dataclass
class SelectionResult:
    selected: list[ScanCandidate] = field(default_factory=list)
    deferred: list[ScanCandidate] = field(default_factory=list)
    per_sector: dict[str, int] = field(default_factory=dict)

    @property
    def deferred_count(self) -> int:
        return len(self.deferred)


def select_for_enrichment(
    candidates: list[ScanCandidate],
    *,
    budget: int = 25,
    max_per_sector: int | None = None,
) -> SelectionResult:
    """Pick up to `budget` candidates, spread across sectors.

    Round-robin across sectors rather than a global sort: take one from each
    sector in turn until the budget runs out. A sector with many candidates
    contributes more only once every other sector has had its turn, so a
    dominant sector cannot consume the budget while others go unexamined.

    `max_per_sector` caps any one sector's share outright. It defaults to
    roughly half the budget, which allows a concentrated day to still use the
    capacity while preventing a single sector from taking all of it.

    Sector comes from authoritatively fetched fundamentals, never from scanner
    columns — the same field `risk.limits` gates on, so one taxonomy governs
    both. Candidates whose sector is unknown are ranked last within the
    round-robin rather than dropped: unknown is a data gap, not a disqualifier,
    and the sector gate downstream will warn about it on its own.
    """
    if budget <= 0:
        return SelectionResult(deferred=list(candidates))

    cap = max_per_sector if max_per_sector is not None else max(1, (budget + 1) // 2)

    buckets: dict[str, list[ScanCandidate]] = defaultdict(list)
    for c in candidates:
        buckets[c.sector or UNKNOWN_SECTOR].append(c)

    # Deterministic ordering: known sectors first (alphabetically), unknown
    # last. Determinism matters because a discovery run must replay identically.
    ordered_sectors = sorted(k for k in buckets if k != UNKNOWN_SECTOR)
    if UNKNOWN_SECTOR in buckets:
        ordered_sectors.append(UNKNOWN_SECTOR)
    for k in buckets:
        buckets[k].sort(key=lambda c: c.symbol)

    selected: list[ScanCandidate] = []
    taken: dict[str, int] = defaultdict(int)
    cursor = {k: 0 for k in ordered_sectors}

    while len(selected) < budget:
        progressed = False
        for sector in ordered_sectors:
            if len(selected) >= budget:
                break
            if taken[sector] >= cap:
                continue
            i = cursor[sector]
            if i >= len(buckets[sector]):
                continue
            selected.append(buckets[sector][i])
            cursor[sector] = i + 1
            taken[sector] += 1
            progressed = True
        if not progressed:
            break

    chosen = {id(c) for c in selected}
    deferred = [
        c.dropped(BUDGET_EXHAUSTED) for c in candidates if id(c) not in chosen
    ]
    return SelectionResult(
        selected=[c.advanced_to(FunnelStage.SELECTED) for c in selected],
        deferred=deferred,
        per_sector=dict(taken),
    )
