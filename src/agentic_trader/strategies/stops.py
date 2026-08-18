"""Stop construction, as a pure function strategies compose.

A stop is the single most consequential number a strategy produces. It decides
where the thesis is declared wrong, and — because sizing is
`notional = risk_budget / stop_distance` — it decides the position size too. A
stop chosen to justify a position size has the causality backwards and will
eventually produce a position nobody sized deliberately.

Three inputs, in priority order:

1. **Volatility (ATR).** How far this stock moves in a normal day. A stop
   inside that range is not a stop, it is a delayed market order: ordinary
   noise removes the position while the thesis is still intact.
2. **Structure.** A level where the premise actually fails — for a trend
   strategy, the 50-day. Only ever *widens* the stop, never tightens it.
3. **Hard bounds.** A floor, because sizing divides by this number and a tiny
   stop implies an enormous position; and a ceiling, because a setup needing a
   very wide stop is one this account should not be taking.

The flat-percentage fallback exists because ATR can be missing, and a strategy
that refuses to act on partial data would never trade. It is recorded as a
distinct basis rather than silently substituted — a stop derived from a
constant is a different risk claim than one derived from measured volatility,
and the journal has to be able to tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

CENTS = Decimal("0.01")


class StopBasis(StrEnum):
    """What actually determined the stop distance."""

    ATR = "atr"
    """Scaled from measured volatility. The intended path."""

    STRUCTURE = "structure"
    """A structural level sat below the volatility stop and widened it."""

    FLAT_PCT = "flat_pct"
    """ATR was unavailable; a fixed percentage was used instead. Honest
    degradation, but a weaker claim — flag it wherever it matters."""


@dataclass(frozen=True)
class StopPlan:
    """A stop level and the full account of how it was derived."""

    stop_price: Decimal
    distance: Decimal
    distance_pct: Decimal
    basis: StopBasis
    clamped_at_min: bool = False
    clamped_at_max: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_volatility_derived(self) -> bool:
        return self.basis in (StopBasis.ATR, StopBasis.STRUCTURE)

    @property
    def too_volatile_to_trade(self) -> bool:
        """True when volatility alone demanded a wider stop than allowed.

        Deliberately not treated as "clamp it and carry on". If ATR says the
        stock routinely travels further than the ceiling permits, a stop at the
        ceiling gets hit by ordinary movement — the position would be sized as
        though the risk were bounded when it is not. The honest response is to
        decline the setup, which is what the strategy does with this flag.
        """
        return self.clamped_at_max


def build_stop(
    price: Decimal,
    *,
    atr: Decimal | None,
    atr_multiple: Decimal,
    min_stop_pct: Decimal,
    max_stop_pct: Decimal,
    flat_stop_pct: Decimal,
    structural_level: Decimal | None = None,
) -> StopPlan:
    """Derive a stop for a long entry at `price`.

    `structural_level` is a price (not a distance) below which the premise is
    broken — the 50-day, a prior swing low. It may widen the stop past
    `max_stop_pct`: a structural stop that far away is a real signal that the
    setup is too loose, and the risk engine's own `max_stop_pct` gate is what
    refuses it. Silently pulling the stop up to the ceiling would place it
    somewhere the thesis does not support.
    """
    if price <= 0:
        raise ValueError(f"cannot build a stop at price {price}")

    notes: list[str] = []

    if atr is not None and atr > 0:
        raw_distance = atr * atr_multiple
        basis = StopBasis.ATR
        notes.append(f"ATR {atr:.2f} x {atr_multiple} = {raw_distance:.2f}")
    else:
        raw_distance = price * flat_stop_pct
        basis = StopBasis.FLAT_PCT
        notes.append(
            f"no ATR available; flat {flat_stop_pct:.1%} of {price:.2f} "
            f"= {raw_distance:.2f}"
        )

    raw_pct = raw_distance / price

    # The floor matters more than it looks: sizing divides by this number, so a
    # 0.1% stop implies a thousand times the risk budget in notional.
    clamped_at_min = raw_pct < min_stop_pct
    clamped_at_max = raw_pct > max_stop_pct
    distance_pct = min(max(raw_pct, min_stop_pct), max_stop_pct)

    if clamped_at_min:
        notes.append(f"widened to the {min_stop_pct:.1%} floor (was {raw_pct:.2%})")
    if clamped_at_max:
        notes.append(
            f"volatility implies a {raw_pct:.2%} stop, beyond the "
            f"{max_stop_pct:.1%} ceiling"
        )

    distance = price * distance_pct
    stop_price = (price - distance).quantize(CENTS)

    # Structure only ever widens. An average sitting below the volatility stop
    # is where the trend premise actually breaks, and a stop above it gets taken
    # out by movement the thesis explicitly allows for.
    if structural_level is not None and 0 < structural_level < stop_price:
        notes.append(
            f"widened to structure {structural_level:.2f} "
            f"(volatility stop was {stop_price:.2f})"
        )
        stop_price = structural_level.quantize(CENTS)
        basis = StopBasis.STRUCTURE
        # A structural stop may exceed the ceiling; risk.limits refuses it there
        # rather than this function quietly tightening it back.
        clamped_at_max = False

    distance = price - stop_price
    return StopPlan(
        stop_price=stop_price,
        distance=distance,
        distance_pct=distance / price,
        basis=basis,
        clamped_at_min=clamped_at_min,
        clamped_at_max=clamped_at_max,
        notes=notes,
    )
