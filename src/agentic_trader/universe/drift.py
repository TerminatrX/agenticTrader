"""Does the live Legend configuration still mean what our code says it means?

Saved scans are editable. `ScanDefinition` records what we believe five scan
ids contain; nothing stops someone widening a shard's RSI band in the UI. Runs
would keep being labelled `agentic-discovery@v1-2026-08-18` while describing a
different universe, which makes the journal actively misleading rather than
merely incomplete.

This is kept deliberately separate from `CoverageStatus`, because they answer
different questions:

    CoverageStatus          did we see the declared universe?
    DefinitionDriftStatus   does the declared universe still mean what we think?

Folding drift into INCOMPLETE would make coverage stop meaning coverage.

Severity is graded by whether a difference can change *membership*:

    FILTER_DRIFT            BLOCKING       changes which symbols exist
    SHARD_DEFINITION_DRIFT  BLOCKING       changes which scans define the set
    SORT_DRIFT              CONDITIONAL    matters only when a shard is capped
    DISPLAY_DRIFT           INFORMATIONAL  columns and titles cannot reach a trade

Drift is never repaired automatically. A mismatch means either Legend or the
definition should change, and which one is a human decision — silently
rewriting either would defeat the point of versioning them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentic_trader.universe.scan_definition import ScanDefinition

_FILTER_ENUMS = {
    "average_volume": "FILTER_TYPE_AVERAGE_VOLUME",
    "rsi": "FILTER_TYPE_RSI",
    "instrument_type": "FILTER_TYPE_INSTRUMENT_TYPE",
    "market_cap": "FILTER_TYPE_MARKET_CAP",
}


class DriftKind(StrEnum):
    FILTER_DRIFT = "filter_drift"
    SHARD_DEFINITION_DRIFT = "shard_definition_drift"
    SORT_DRIFT = "sort_drift"
    DISPLAY_DRIFT = "display_drift"


class DriftSeverity(StrEnum):
    BLOCKING = "blocking"
    CONDITIONAL = "conditional"
    INFORMATIONAL = "informational"


class DefinitionDriftStatus(StrEnum):
    MATCHES = "matches"
    DISPLAY_ONLY = "display_only"
    SORT_CHANGED = "sort_changed"
    DRIFTED = "drifted"


@dataclass(frozen=True)
class DriftFinding:
    kind: DriftKind
    severity: DriftSeverity
    detail: str
    scan_id: str | None = None
    expected: Any = None
    observed: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "scan_id": self.scan_id,
            "detail": self.detail,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class DriftReport:
    findings: tuple[DriftFinding, ...] = ()

    @property
    def status(self) -> DefinitionDriftStatus:
        kinds = {f.kind for f in self.findings}
        if kinds & {DriftKind.FILTER_DRIFT, DriftKind.SHARD_DEFINITION_DRIFT}:
            return DefinitionDriftStatus.DRIFTED
        if DriftKind.SORT_DRIFT in kinds:
            return DefinitionDriftStatus.SORT_CHANGED
        if DriftKind.DISPLAY_DRIFT in kinds:
            return DefinitionDriftStatus.DISPLAY_ONLY
        return DefinitionDriftStatus.MATCHES

    @property
    def has_blocking(self) -> bool:
        """True when discovery must not proceed to enrichment at all."""
        return any(f.severity is DriftSeverity.BLOCKING for f in self.findings)

    @property
    def has_sort_drift(self) -> bool:
        return any(f.kind is DriftKind.SORT_DRIFT for f in self.findings)

    def blocks(self, *, any_shard_capped: bool) -> bool:
        """Final verdict, once coverage is known.

        Sort order decides *which* rows survive only when a shard is truncated.
        Below the cap every match is returned, ordering is irrelevant — and this
        system's selector ignores scanner order anyway, ranking by an explicit
        trading date and a deterministic hash. At the cap, sorting becomes
        membership-affecting and the run must stop.
        """
        return self.has_blocking or (self.has_sort_drift and any_shard_capped)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [f.as_dict() for f in self.findings]


_SESSION_RE = re.compile(r'session\s*=\s*"([^"]+)"')


def _session_of(expression: str | None) -> str:
    """Pull `session="all"` out of a broker expression.

    `get_scans` does not surface session as a filter field, but it does return
    the underlying expression, which carries it. Extracting it is what makes a
    change from all-session to regular-session detectable — the two produce
    materially different RSI and volume values for the same symbol.
    """
    if not expression:
        return ""
    m = _SESSION_RE.search(str(expression))
    return m.group(1) if m else ""


def _canonical(enum_name: str, spec: dict[str, Any]) -> tuple:
    """Normalize a filter to a comparable shape.

    The broker returns everything as strings; the definition holds native
    numbers. Comparing raw would report drift on every run.
    """
    if "values" in spec:
        values = tuple(str(v) for v in spec["values"])
    elif "value" in spec:
        values = (str(spec["value"]),)
    else:
        values = ()
    return (
        enum_name,
        str(spec.get("predicate", "")),
        values,
        str(spec.get("interval") or ""),
        str(spec.get("length") or ""),
        str(spec.get("session") or ""),
    )


def _expected_filters(definition: ScanDefinition, shard) -> set[tuple]:
    merged = {**definition.base_filters, **shard.filters}
    return {
        _canonical(_FILTER_ENUMS[name], spec)
        for name, spec in merged.items()
        if name in _FILTER_ENUMS and spec
    }


def _observed_filters(scan: dict[str, Any]) -> set[tuple]:
    out = set()
    for f in scan.get("filter_summary") or []:
        out.add(
            _canonical(
                str(f.get("filter_type_enum", "")),
                {
                    "predicate": f.get("predicate", ""),
                    "values": f.get("values") or [],
                    "interval": f.get("interval"),
                    "length": f.get("length"),
                    "session": _session_of(f.get("expression")),
                },
            )
        )
    return out


def check_definition_drift(
    scans_payload: Any,
    definition: ScanDefinition,
) -> DriftReport:
    """Compare live saved-scan configuration against the declared definition.

    Reads a `get_scans` response. Performs no I/O — the agent fetches, this
    compares, exactly like every other parser in the codebase.
    """
    data = scans_payload.get("data", scans_payload) if isinstance(scans_payload, dict) else {}
    scans = data.get("scans") if isinstance(data, dict) else None
    live = {s.get("scan_id"): s for s in (scans or []) if isinstance(s, dict)}

    findings: list[DriftFinding] = []

    for shard in definition.shards:
        scan = live.get(shard.scan_id)
        if scan is None:
            findings.append(
                DriftFinding(
                    kind=DriftKind.SHARD_DEFINITION_DRIFT,
                    severity=DriftSeverity.BLOCKING,
                    scan_id=shard.scan_id,
                    detail=f"{shard.label}: saved scan no longer exists",
                )
            )
            continue

        expected = _expected_filters(definition, shard)
        observed = _observed_filters(scan)
        if expected != observed:
            findings.append(
                DriftFinding(
                    kind=DriftKind.FILTER_DRIFT,
                    severity=DriftSeverity.BLOCKING,
                    scan_id=shard.scan_id,
                    detail=(
                        f"{shard.label}: live filters differ from the definition — "
                        "this can change which symbols are discovered"
                    ),
                    expected=sorted(str(e) for e in expected - observed),
                    observed=sorted(str(o) for o in observed - expected),
                )
            )

        # A vanished sort is drift, not a match. Treating absent as equal is
        # most dangerous exactly at the cap, where sorting decides which rows
        # survive.
        live_sort = scan.get("sorting")
        if definition.sorting and live_sort != definition.sorting:
            findings.append(
                DriftFinding(
                    kind=DriftKind.SORT_DRIFT,
                    severity=DriftSeverity.CONDITIONAL,
                    scan_id=shard.scan_id,
                    detail=f"{shard.label}: sort changed; matters only if this shard caps",
                    expected=definition.sorting,
                    observed=live_sort,
                )
            )

        if scan.get("cortex_managed"):
            findings.append(
                DriftFinding(
                    kind=DriftKind.SHARD_DEFINITION_DRIFT,
                    severity=DriftSeverity.BLOCKING,
                    scan_id=shard.scan_id,
                    detail=f"{shard.label}: now Cortex-managed and no longer ours to rely on",
                )
            )

    # Display-only differences are recorded and never block. Columns cannot
    # reach a decision — only `symbol` crosses into enrichment.
    for shard in definition.shards:
        scan = live.get(shard.scan_id)
        if scan and shard.label and scan.get("title") and shard.label.split()[0] not in str(
            scan.get("title")
        ):
            findings.append(
                DriftFinding(
                    kind=DriftKind.DISPLAY_DRIFT,
                    severity=DriftSeverity.INFORMATIONAL,
                    scan_id=shard.scan_id,
                    detail="scan title changed",
                    expected=shard.label,
                    observed=scan.get("title"),
                )
            )

    return DriftReport(findings=tuple(findings))
