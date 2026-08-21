"""What an external system can do, and how confidently we know it.

Two facts are tracked separately for every capability, because conflating them
is how a system ends up trusting a guess:

- **supported** — can it do this? `None` means unknown.
- **evidence** — how that was established, from a verified round trip down to
  an inference nobody has checked.

These primitives live in `models/` rather than beside any one integration
because more than one subsystem needs them, and the alternative — importing
capability types from `execution/` into `universe/` — would invert the
dependency direction. Concrete profiles live with the integration they
describe; only the vocabulary is shared.

A profile is **versioned and fingerprinted** so a stored decision stays
readable. `profile_ref` identifies the claim set a past decision was made
against; `content_fingerprint` is what makes that reference honest, by failing
loudly when the claims move without the version moving with them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from datetime import date
from enum import StrEnum


class Evidence(StrEnum):
    """How a capability claim was established, strongest first."""

    EMPIRICALLY_VERIFIED = "empirically_verified"
    """Observed directly against the system. The only category that proves
    behaviour rather than describing it."""

    SCHEMA_DOCUMENTED = "schema_documented"
    """Stated in the MCP tool schema. Authoritative about intent, but the
    schema and the running system can disagree — and have."""

    PUBLIC_DOCUMENTED = "public_documented"
    """Stated in public documentation, which may lag the API."""

    INFERRED = "inferred"
    """Deduced from adjacent facts. Reasoning, not a source."""

    UNKNOWN = "unknown"
    """No basis. Must be treated as unsupported wherever failing closed matters."""


@dataclass(frozen=True)
class Capability:
    """One thing a system can or cannot do, plus the basis for saying so."""

    supported: bool | None
    evidence: Evidence
    note: str = ""

    @property
    def is_certain(self) -> bool:
        """True only for a claim actually observed against the system."""
        return self.supported is not None and self.evidence is Evidence.EMPIRICALLY_VERIFIED

    @property
    def usable(self) -> bool:
        """Fail closed: unknown support is not permission.

        Deliberately not `supported is not False` — `None` must never read as a
        yes, and a capability nobody has established is exactly the case where
        an optimistic default does the most damage.

        Note this asks "may we rely on this?". Some capabilities record a fact
        rather than a permission — "the scanner cannot date its rows" is not a
        reason to refuse the scanner — and those must simply not be consulted
        by a gate.
        """
        return self.supported is True

    def describe(self) -> str:
        state = {True: "supported", False: "not supported", None: "unknown"}[self.supported]
        return f"{state} ({self.evidence.value})" + (f": {self.note}" if self.note else "")


@dataclass(frozen=True)
class CapabilityProfile:
    """Versioned metadata and the fingerprint mechanism, shared by all profiles.

    The base deliberately does **not** decide what gets hashed. An earlier
    design had it hash every `Capability` field automatically, which works for a
    profile whose claims are all booleans and fails silently for one carrying
    semantic content: adding a value to an observed vocabulary would leave the
    fingerprint unchanged, and a stored `profile_ref` would then denote two
    different contracts. Each profile declares its own `fingerprint_items`, so
    "what is semantically load-bearing here" is answered where it is known.
    """

    profile_id: str
    version: str
    as_of: date

    @property
    def profile_ref(self) -> str:
        """Stable identifier to store alongside a decision."""
        return f"{self.profile_id}@{self.version}"

    def fingerprint_items(self) -> tuple[str, ...]:
        """Every piece of content a stored `profile_ref` must pin down."""
        raise NotImplementedError

    @property
    def content_fingerprint(self) -> str:
        """SHA-256 over this profile's declared semantic content.

        The same trick `config/risk.lock` plays on risk values: hash the
        meaning, pin it in a test, and a change that skips the version bump
        fails loudly instead of quietly rewriting history.
        """
        payload = "|".join(self.fingerprint_items())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capability_items(profile: CapabilityProfile) -> tuple[str, ...]:
    """Fingerprint items for every `Capability` field, in declaration order.

    `note` is excluded on purpose — prose may be improved without invalidating
    a claim. Only support and evidence are the claim.
    """
    return tuple(
        f"{f.name}={cap.supported}:{cap.evidence.value}"
        for f in fields(profile)
        if isinstance(cap := getattr(profile, f.name), Capability)
    )
