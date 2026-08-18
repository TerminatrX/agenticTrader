"""Choosing which candidates are worth expensive enrichment.

Enrichment costs about seven single-symbol MCP calls per candidate — six
indicators plus earnings, none of which batch. A few hundred candidates is
therefore not a slow run, it is an impossible one, and the budget has to bind
somewhere. This module decides where.

**This is a compute-budget policy, not a risk control.** It decides which
symbols get looked at, and nothing else. `max_sector_exposure_pct` in
`risk.limits` remains the only authority on how much of one sector the account
may actually hold; the two must not be confused, and a change here can never
loosen a risk limit. The naming reflects that: nothing here says "exposure".

Selection is diversity-aware because ranking every candidate globally and
taking the top N reproduces the concentration the sector cap exists to solve —
a technology-heavy list yields a technology-heavy budget, the sector gate then
blocks all but the first entry, and the run does a lot of work to find one
trade.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from agentic_trader.universe.candidate import FunnelStage, ScanCandidate

UNKNOWN_SECTOR = "unknown"

BUDGET_EXHAUSTED = "enrichment_budget"
"""The budget ran out before this candidate's turn.

Distinct from any strategy or risk rejection on purpose. A review that cannot
separate "the strategy declined this" from "we ran out of budget before
looking" will read a compute limit as an absence of opportunity.
"""

UNKNOWN_SECTOR_LIMIT = "unknown_sector_limit"
"""Budget remained, but this candidate's sector is unknown and the unknown
bucket was already at its cap.

Deliberately not `enrichment_budget`, which would be a false statement: the
budget was not exhausted. Unknown-sector candidates are capped because a
missing sector also means the downstream concentration gate cannot fully
evaluate them — this is a data-quality limit wearing its own name.
"""


@dataclass
class SelectionResult:
    selected: list[ScanCandidate] = field(default_factory=list)
    deferred: list[ScanCandidate] = field(default_factory=list)
    per_sector: dict[str, int] = field(default_factory=dict)

    @property
    def deferred_count(self) -> int:
        return len(self.deferred)

    @property
    def budget_deferred_count(self) -> int:
        return sum(1 for c in self.deferred if c.exit_reason == BUDGET_EXHAUSTED)


def rotation_key(symbol: str, on: date) -> str:
    """Deterministic per-date ordering key.

    Sorting each bucket by symbol is deterministic but systematically biased:
    run daily, alphabetically early tickers consume the budget every time, and
    the eliminated market-cap sampling bias is simply replaced by a smaller
    alphabet one. Hashing symbol with the trading date rotates which candidates
    get examined while keeping any single date perfectly replayable.

    `hashlib` rather than `hash()` — the builtin is salted per process, so it
    would not replay across runs.
    """
    return hashlib.sha256(f"{on.isoformat()}|{symbol}".encode()).hexdigest()


def select_for_enrichment(
    candidates: list[ScanCandidate],
    *,
    budget: int = 25,
    max_per_sector: int | None = None,
    on: date | None = None,
) -> SelectionResult:
    """Pick up to `budget` candidates, spread across sectors.

    Two passes:

    1. Round-robin across every sector, each capped at `max_per_sector`, so a
       dominant sector cannot crowd others out while they go unexamined.
    2. If budget remains, backfill from **known** sectors only, ignoring the
       cap. The cap allocates compute between sectors; when no other sector has
       candidates left, holding capacity back protects nothing — the real
       concentration limit is enforced later by `max_sector_exposure_pct`.

    Unknown-sector candidates never backfill. A missing sector means the
    downstream concentration gate cannot fully assess the position, so an
    unknown-heavy day deliberately uses less than the full budget rather than
    filling it with candidates whose risk profile is partly unreadable.
    """
    if budget <= 0:
        return SelectionResult(deferred=[c.dropped(BUDGET_EXHAUSTED) for c in candidates])

    day = on or date.today()
    cap = max_per_sector if max_per_sector is not None else max(1, (budget + 1) // 2)

    buckets: dict[str, list[ScanCandidate]] = defaultdict(list)
    for c in candidates:
        buckets[c.sector or UNKNOWN_SECTOR].append(c)
    for bucket in buckets.values():
        bucket.sort(key=lambda c: rotation_key(c.symbol, day))

    known = sorted(
        (k for k in buckets if k != UNKNOWN_SECTOR),
        key=lambda s: rotation_key(s, day),
    )
    order = [*known, UNKNOWN_SECTOR] if UNKNOWN_SECTOR in buckets else list(known)

    selected: list[ScanCandidate] = []
    taken: dict[str, int] = defaultdict(int)
    cursor: dict[str, int] = dict.fromkeys(order, 0)

    def take(sector: str) -> bool:
        i = cursor[sector]
        if i >= len(buckets[sector]):
            return False
        selected.append(buckets[sector][i])
        cursor[sector] = i + 1
        taken[sector] += 1
        return True

    # Pass 1 — fair share.
    while len(selected) < budget:
        progressed = False
        for sector in order:
            if len(selected) >= budget:
                break
            if taken[sector] >= cap:
                continue
            progressed |= take(sector)
        if not progressed:
            break

    # Pass 2 — known sectors may use what nobody else claimed.
    while len(selected) < budget:
        progressed = False
        for sector in known:
            if len(selected) >= budget:
                break
            progressed |= take(sector)
        if not progressed:
            break

    chosen = {id(c) for c in selected}
    exhausted = len(selected) >= budget
    deferred = [
        c.dropped(BUDGET_EXHAUSTED if exhausted else UNKNOWN_SECTOR_LIMIT)
        for c in candidates
        if id(c) not in chosen
    ]
    return SelectionResult(
        selected=[c.advanced_to(FunnelStage.SELECTED) for c in selected],
        deferred=deferred,
        per_sector=dict(taken),
    )
