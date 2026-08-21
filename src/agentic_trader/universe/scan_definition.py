"""The saved scans that define our discovery universe.

Kept separate from `ScannerCapabilities` because the two answer different
questions and change for different reasons:

    ScannerCapabilities   what run_scan is capable of        -> the endpoint
    ScanDefinition        what these particular scans ask    -> our config

Retuning an RSI band or adding a shard must not bump a profile describing
Robinhood's row cap, and Robinhood changing its row cap must not invalidate a
record of which filters we were running. Conflating them would force spurious
version churn in both directions.

`"Asset type"` appearing among the returned columns is the clearest example:
that column exists because of how *these scans* were configured, not because
`run_scan` always returns it.

Columns are deliberately **not** declared here either. They were observed to
vary per shard — one carries ten, the others eleven, in differing order — and
they are display-only, affecting `source_values` richness rather than which
candidates are returned. Drift checking reads the live column set per shard
instead of asserting a uniform one that was never true.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ShardSpec:
    """One saved scan and the band it is responsible for."""

    scan_id: str
    label: str
    filters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"scan_id": self.scan_id, "label": self.label, "filters": self.filters}


@dataclass(frozen=True)
class ScanDefinition:
    """A named, versioned set of saved scans covering one declared universe.

    The shard set is *expected*, not merely observed. A run that receives seven
    of eight shards has not covered the universe this definition declares, and
    only a stated expectation makes that detectable — without it, four short
    shards look exactly like complete coverage.
    """

    name: str
    version: str
    as_of: date
    shards: tuple[ShardSpec, ...]
    base_filters: dict[str, Any] = field(default_factory=dict)
    sorting: str | None = None

    # Whether a run may proceed when coverage is anything but COMPLETE.
    #
    # The entire point of sharding is that the declared universe is fully
    # reachable. If a shard grows to the cap, the correct response is to split
    # that band again — not to keep collecting shadow observations from a
    # biased first 200 while the journal records them as ordinary results.
    # A definition that genuinely wants best-effort discovery can turn this off.
    require_complete_coverage: bool = True

    @property
    def definition_ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def expected_shard_ids(self) -> tuple[str, ...]:
        return tuple(s.scan_id for s in self.shards)

    def as_config(self) -> dict[str, Any]:
        """The full configuration, for `scan_config_json` on a run."""
        return {
            "name": self.name,
            "version": self.version,
            "as_of": self.as_of.isoformat(),
            "base_filters": self.base_filters,
            "sorting": self.sorting,
            "require_complete_coverage": self.require_complete_coverage,
            "shards": [s.as_dict() for s in self.shards],
        }

    @property
    def config_fingerprint(self) -> str:
        """SHA-256 over the configuration, so a retune is detectable.

        A stored `definition_ref` is only meaningful if it denotes one filter
        set. Changing RSI 25-50 to 20-55 without bumping the version would
        otherwise make historical runs silently unreadable.
        """
        payload = json.dumps(self.as_config(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# `session` is declared explicitly. It is not a filter parameter we send — the
# broker bakes it into the underlying expression, e.g.
# `rsi(candlePeriod="1d", candleCount=14, session="all")` — but it changes what
# the number means, and "all" includes extended hours while our own indicators
# use regular hours only. Recording it makes a silent change to that semantics
# detectable as drift instead of invisible.
_BASE_FILTERS = {
    "average_volume": {
        "predicate": ">", "value": 500_000, "length": 30, "interval": "1d", "session": "all",
    },
    "rsi": {
        "predicate": "BETWEEN", "values": [25, 50], "length": 14, "interval": "1d",
        "session": "all",
    },
    "instrument_type": {"predicate": "=", "value": "STOCK"},
}


def _shard(scan_id: str, label: str, low: int | None, high: int | None) -> ShardSpec:
    band: dict[str, Any] = {}
    if low is not None and high is not None:
        band = {"predicate": "BETWEEN", "values": [low, high]}
    elif low is not None:
        band = {"predicate": ">", "value": low}
    return ShardSpec(scan_id=scan_id, label=label, filters={"market_cap": band})


# Market-cap bands chosen so every shard returns strictly under the scanner's
# 200-row cap. If one ever reaches it, split that band again rather than
# accepting a truncated universe — the whole point of sharding is that the
# declared universe is fully reachable.
#
# No sector filter: selecting every sector is a no-op, and sector is fetched
# authoritatively from fundamentals during enrichment anyway. No price filter,
# deliberately — a ceiling chosen to suit the current account size would
# distort which setups the strategy ever sees.
# The definition in force. Named for its role rather than its version --
# `ScanDefinition.version` is the authoritative historical identity, and a
# constant called DISCOVERY_V1 holding a v2 definition was already one
# mismatch too many.
CURRENT_DISCOVERY = ScanDefinition(
    name="agentic-discovery",
    version="v3-2026-08-20",
    as_of=date(2026, 8, 20),
    base_filters=_BASE_FILTERS,
    sorting="Market cap desc",
    shards=(
        _shard("7e65d9ff-a2b9-4a82-9c84-bee57475ef50", "B1 $2B-$3B",
               2_000_000_000, 3_000_000_000),
        _shard("3eeeb183-4a96-442b-8b01-404f33978c2b", "B2 $3B-$4.5B",
               3_000_000_000, 4_500_000_000),
        _shard("3bf67683-fc75-4e67-a9f1-2a7500bb0256", "B3 $4.5B-$7B",
               4_500_000_000, 7_000_000_000),
        _shard("813dd065-f51b-47de-9eff-ef112295a3de", "B4 $7B-$10B",
               7_000_000_000, 10_000_000_000),
        _shard("bd7d315f-db05-4fda-b937-b31b1989ce24", "B5 $10B-$17.5B",
               10_000_000_000, 17_500_000_000),
        _shard("795e9148-25fa-4678-a2b6-13ced4cbb025", "B6 $17.5B-$35B",
               17_500_000_000, 35_000_000_000),
        _shard("cccffcd4-8c3d-452c-ba23-23c71308030e", "B7 $35B-$100B",
               35_000_000_000, 100_000_000_000),
        _shard("cc72022a-5f93-4c66-a69e-369dc6c89d92", "B8 >$100B",
               100_000_000_000, None),
    ),
)
