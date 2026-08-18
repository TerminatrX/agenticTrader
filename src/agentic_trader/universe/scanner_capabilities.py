"""What the Robinhood scanner can do, as observed on 2026-08-18.

Separate from the order-execution profile in `execution.capabilities` because
the two evolve independently: a change to the filter vocabulary should not
force a version bump on claims about `place_equity_order`, and vice versa.

Most claims here are `EMPIRICALLY_VERIFIED` — this profile was written from
observed responses rather than tool descriptions, and in two cases the
observation contradicted what the documentation implied. Where a claim was not
exercised its evidence says so; read the evidence rather than assuming.

Deliberately excluded: anything describing a *particular saved scan*. Its
filters, its sort, its column list, its shard ids all live in
`scan_definition`, because retuning a filter must not bump a profile about the
endpoint, and a change to the endpoint must not invalidate a record of the
filters we were running.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from agentic_trader.models.capabilities import (
    Capability,
    CapabilityProfile,
    Evidence,
    capability_items,
)
from agentic_trader.universe.candidate import CoverageStatus


@dataclass(frozen=True)
class ScannerCapabilities(CapabilityProfile):
    """Scanner contract: what it can do, and what its results mean.

    Unlike the execution profile, this one carries semantic content as well as
    booleans — vocabularies, row fields, the row cap. Those are fingerprinted
    too. A profile that hashed only its booleans would let the sector or
    instrument-type vocabulary change while `profile_ref` stayed put, which
    would make a stored reference denote two different contracts.
    """

    result_pagination: Capability
    row_freshness_timestamp: Capability
    total_items_reliable: Capability
    atomic_filter_update: Capability
    configurable_sorting: Capability

    max_rows_per_run: int
    row_instrument_type: str
    filter_instrument_type: str

    def fingerprint_items(self) -> tuple[str, ...]:
        return (
            *capability_items(self),
            f"max_rows_per_run={self.max_rows_per_run}",
            f"row_instrument_type={self.row_instrument_type}",
            f"filter_instrument_type={self.filter_instrument_type}",
        )

    def coverage_for(self, returned_count: int) -> CoverageStatus:
        """Classify a shard's completeness from its row count alone.

        Deliberately does not consult the broker's `total_items`. Summing five
        disjoint shards over the same filter definition produced 668 rows while
        the unsharded equivalent reported 394, so that field cannot establish
        completeness. What can: a row count strictly below the hard cap means
        nothing was withheld. A count *at* the cap is ambiguous — exactly-N and
        truncated-from-more are indistinguishable without pagination, and
        there is none.
        """
        if returned_count < self.max_rows_per_run:
            return CoverageStatus.COMPLETE
        return CoverageStatus.UNKNOWN_TRUNCATED


# Observed against the live scanner on 2026-08-18 while building the discovery
# shards. Bump `version` and `as_of` whenever any claim below changes.
ROBINHOOD_MCP_SCANNER = ScannerCapabilities(
    profile_id="robinhood-mcp-scanner",
    version="2026-08-18",
    as_of=date(2026, 8, 18),
    result_pagination=Capability(
        False,
        Evidence.EMPIRICALLY_VERIFIED,
        "run_scan takes no cursor and returns at most 200 rows; a scan "
        "reporting 394 matches returned 200 with no way to reach the rest",
    ),
    row_freshness_timestamp=Capability(
        False,
        Evidence.EMPIRICALLY_VERIFIED,
        "no timestamp field appears anywhere in a run_scan response. Records a "
        "fact, not a permission: freshness is established downstream from "
        "quote_as_of and indicators.as_of, so this never gates a candidate",
    ),
    total_items_reliable=Capability(
        False,
        Evidence.EMPIRICALLY_VERIFIED,
        "five disjoint market-cap shards over one filter definition summed to "
        "668 while the unsharded equivalent reported 394; use row counts "
        "against max_rows_per_run instead",
    ),
    atomic_filter_update=Capability(
        True,
        Evidence.EMPIRICALLY_VERIFIED,
        "update_scan_filters applies all filters or none, which makes it the "
        "safe way to repair a scan created with a bad filter",
    ),
    configurable_sorting=Capability(
        True,
        Evidence.SCHEMA_DOCUMENTED,
        "update_scan_config accepts sorting_column; not exercised, and with "
        "complete shards the sort affects ordering rather than coverage",
    ),
    max_rows_per_run=200,
    # The same instrument is EQUITY in a result row and STOCK to a filter. A
    # filter built from the row vocabulary validates and silently matches
    # nothing, which is how this was found.
    row_instrument_type="EQUITY",
    filter_instrument_type="STOCK",
)

