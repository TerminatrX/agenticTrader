"""What the Robinhood scanner can do, as observed on 2026-08-18.

Separate from the order-execution profile in `execution.capabilities` because
the two evolve independently: a change to the filter vocabulary should not
force a version bump on claims about `place_equity_order`, and vice versa.

Every claim here is `EMPIRICALLY_VERIFIED` — this profile was written from
observed responses rather than from tool descriptions, and in two cases the
observation contradicted what the documentation implied.
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
    default_row_fields: tuple[str, ...]
    default_sorting: str

    def fingerprint_items(self) -> tuple[str, ...]:
        return (
            *capability_items(self),
            f"max_rows_per_run={self.max_rows_per_run}",
            f"row_instrument_type={self.row_instrument_type}",
            f"filter_instrument_type={self.filter_instrument_type}",
            "default_row_fields=" + ",".join(sorted(self.default_row_fields)),
            f"default_sorting={self.default_sorting}",
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
    default_row_fields=(
        "% Change",
        "Asset type",
        "Average volume",
        "Last",
        "Market cap",
        "Name",
        "Net change",
        "RSI",
        "Relative volume",
        "Symbol",
        "Volume",
    ),
    default_sorting="Market cap desc",
)


# The saved scans backing production discovery. Market-cap bands chosen so each
# returns strictly under `max_rows_per_run`; if one ever reaches the cap, split
# it again rather than accepting a truncated universe.
DISCOVERY_SHARD_IDS: tuple[str, ...] = (
    "cc72022a-5f93-4c66-a69e-369dc6c89d92",  # Mega   >$100B
    "cccffcd4-8c3d-452c-ba23-23c71308030e",  # Large  $20B-$100B
    "795e9148-25fa-4678-a2b6-13ced4cbb025",  # Mid    $8B-$20B
    "bd7d315f-db05-4fda-b937-b31b1989ce24",  # Small  $4B-$8B
    "813dd065-f51b-47de-9eff-ef112295a3de",  # Micro  $2B-$4B
)
