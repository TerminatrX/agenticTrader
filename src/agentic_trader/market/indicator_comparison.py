"""Measuring local indicators against the broker's, without trusting either.

**Diagnostics only.** Nothing here is read by a strategy, a risk gate, or the
critic, and nothing here writes to `MarketSnapshot.indicators`. Broker values
remain the authoritative decision inputs; this module exists to find out whether
that could safely change, which is a different question from changing it.

Two kinds of comparison, and the second matters more
----------------------------------------------------

Numeric error is easy to compute and easy to over-trust. A 0.0001 difference in
a MACD histogram sounds negligible until both the current and previous values
carry it and the strategy's actual question is::

    macd_hist > macd_hist_prev

which is the *sign of a difference between two noisy numbers*. That comparison
can flip while the absolute error stays far below any tolerance one would
naively write down. So `DecisionFlags` re-derives the booleans the strategy
really evaluates, from each side independently, and disagreement between them is
reported separately from — and weighted above — numeric distance.

ATR gets a third treatment because its error is economic rather than logical:
it sets stop distance, and `notional = risk_budget / stop_distance`, so a small
relative error becomes a proportional position-size error, and a larger one can
cross `max_stop_pct` and decline the setup outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from agentic_trader.strategies.stops import StopPlan, build_stop


def _dec(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ValueDelta:
    """One indicator field at one bar, from both sides."""

    field: str
    begins_at: datetime | None
    local: Decimal | None
    broker: Decimal | None

    @property
    def comparable(self) -> bool:
        """Both sides present. A missing value is not a zero difference, and
        counting it as agreement would let an unavailable local indicator
        inflate the pass rate."""
        return self.local is not None and self.broker is not None

    @property
    def abs_error(self) -> Decimal | None:
        return abs(self.local - self.broker) if self.comparable else None

    @property
    def rel_error(self) -> Decimal | None:
        """Relative to the broker value, which is the reference side.

        `None` when the broker value is zero — a relative error against zero is
        undefined, and returning something large or something small would both
        be inventions. MACD histograms legitimately sit at zero.
        """
        if not self.comparable or self.broker == 0:
            return None
        return abs(self.local - self.broker) / abs(self.broker)


@dataclass(frozen=True)
class DecisionFlags:
    """Every boolean `trend_pullback` derives from indicators.

    Recomputed from one side's values so the two sides can be compared on what
    the strategy would actually conclude, not merely on how close the numbers
    look. `None` means the inputs were absent, and is never treated as False.
    """

    above_sma200: bool | None = None
    stack_bullish: bool | None = None
    in_pullback: bool | None = None
    pullback_within_depth: bool | None = None
    rsi_in_band: bool | None = None
    rsi_improving: bool | None = None
    macd_hist_rising: bool | None = None
    volatility_within_ceiling: bool | None = None
    structure_widens_stop: bool | None = None
    exit_rsi_reached: bool | None = None

    def disagreements(self, other: DecisionFlags) -> tuple[str, ...]:
        """Fields where the two sides would reach different conclusions.

        A flag that is `None` on one side and a boolean on the other counts as a
        disagreement: "we could not tell" and "yes" are different answers, and
        the strategy treats them differently — indeterminate fails the
        condition, so it is not merely a missing datum.
        """
        return tuple(
            name
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(other, name)
        )


@dataclass(frozen=True)
class StopImpact:
    """What an ATR difference does to the money."""

    local: StopPlan | None
    broker: StopPlan | None

    @property
    def comparable(self) -> bool:
        return self.local is not None and self.broker is not None

    @property
    def stop_price_diff(self) -> Decimal | None:
        if not self.comparable:
            return None
        return abs(self.local.stop_price - self.broker.stop_price)

    @property
    def distance_pct_diff(self) -> Decimal | None:
        if not self.comparable:
            return None
        return abs(self.local.distance_pct - self.broker.distance_pct)

    @property
    def notional_rel_diff(self) -> Decimal | None:
        """Fractional position-size difference the stop difference implies.

        Sizing is `notional = risk_budget / stop_distance_pct`, so notional is
        inversely proportional to stop distance and the risk budget cancels.
        This is the number that says whether an ATR discrepancy is economically
        real or merely arithmetic.
        """
        if not self.comparable:
            return None
        local_pct, broker_pct = self.local.distance_pct, self.broker.distance_pct
        if local_pct <= 0 or broker_pct <= 0:
            return None
        local_notional = Decimal(1) / local_pct
        broker_notional = Decimal(1) / broker_pct
        return abs(local_notional - broker_notional) / broker_notional

    @property
    def basis_disagrees(self) -> bool:
        """Different stop *bases* is a bigger deal than a different level: it
        means one side thinks volatility drove the stop and the other thinks
        structure or a flat percentage did."""
        if not self.comparable:
            return False
        return self.local.basis is not self.broker.basis

    @property
    def tradability_disagrees(self) -> bool:
        """One side declines the setup as too volatile and the other does not.
        This is a trade/no-trade disagreement, not a sizing one."""
        if not self.comparable:
            return False
        return self.local.too_volatile_to_trade != self.broker.too_volatile_to_trade


# ------------------------------------------------------------------ deriving


def decision_flags(
    *,
    price: Decimal,
    sma_20: Decimal | None,
    sma_50: Decimal | None,
    sma_200: Decimal | None,
    rsi_14: Decimal | None,
    rsi_prev: Decimal | None,
    macd_hist: Decimal | None,
    macd_hist_prev: Decimal | None,
    atr_14: Decimal | None,
    rsi_floor: Decimal = Decimal("30"),
    rsi_ceiling: Decimal = Decimal("45"),
    exit_rsi: Decimal = Decimal("72"),
    max_pullback_pct: Decimal = Decimal("0.12"),
    atr_multiple: Decimal = Decimal("2.0"),
    min_stop_pct: Decimal = Decimal("0.02"),
    max_stop_pct: Decimal = Decimal("0.12"),
    flat_stop_pct: Decimal = Decimal("0.05"),
) -> DecisionFlags:
    """Re-derive the strategy's conditions from one side's indicator values.

    Deliberately a *reimplementation* of the comparisons in `signals.py` and
    `trend_pullback.py` rather than a call into them. Calling the real strategy
    would need a full `MarketSnapshot` per side, which would mean building
    snapshots out of comparison data — and a snapshot is the thing this branch
    must not let local values into. Keeping the comparison logic separate is
    what stops a diagnostic from becoming a decision path.

    The defaults mirror `TrendPullbackStrategy.DEFAULTS`. They are duplicated
    knowingly and a test asserts they still match; if the strategy retunes, the
    comparison must retune with it or it stops measuring the real decision.
    """
    stop = _stop_for(
        price, atr_14, sma_50,
        atr_multiple=atr_multiple, min_stop_pct=min_stop_pct,
        max_stop_pct=max_stop_pct, flat_stop_pct=flat_stop_pct,
    )

    depth = None if sma_20 is None or sma_20 == 0 else (price - sma_20) / sma_20

    return DecisionFlags(
        above_sma200=None if sma_200 is None else price > sma_200,
        stack_bullish=(
            None if sma_50 is None or sma_200 is None else sma_50 > sma_200
        ),
        in_pullback=None if sma_20 is None else price < sma_20,
        pullback_within_depth=None if depth is None else abs(depth) <= max_pullback_pct,
        rsi_in_band=None if rsi_14 is None else rsi_floor <= rsi_14 <= rsi_ceiling,
        rsi_improving=(
            None if rsi_14 is None or rsi_prev is None else rsi_14 > rsi_prev
        ),
        macd_hist_rising=(
            None
            if macd_hist is None or macd_hist_prev is None
            else macd_hist > macd_hist_prev
        ),
        volatility_within_ceiling=(
            None if stop is None else not stop.too_volatile_to_trade
        ),
        structure_widens_stop=(
            None if sma_50 is None else sma_50 < price
        ),
        exit_rsi_reached=None if rsi_14 is None else rsi_14 >= exit_rsi,
    )


def _stop_for(
    price: Decimal,
    atr: Decimal | None,
    sma_50: Decimal | None,
    *,
    atr_multiple: Decimal,
    min_stop_pct: Decimal,
    max_stop_pct: Decimal,
    flat_stop_pct: Decimal,
) -> StopPlan | None:
    if price <= 0:
        return None
    structural = (
        (sma_50 * Decimal("0.99")).quantize(Decimal("0.01"))
        if sma_50 is not None and sma_50 < price
        else None
    )
    return build_stop(
        price,
        atr=atr,
        atr_multiple=atr_multiple,
        min_stop_pct=min_stop_pct,
        max_stop_pct=max_stop_pct,
        flat_stop_pct=flat_stop_pct,
        structural_level=structural,
    )


def stop_impact(
    *,
    price: Decimal,
    local_atr: Decimal | None,
    broker_atr: Decimal | None,
    sma_50: Decimal | None,
    atr_multiple: Decimal = Decimal("2.0"),
    min_stop_pct: Decimal = Decimal("0.02"),
    max_stop_pct: Decimal = Decimal("0.12"),
    flat_stop_pct: Decimal = Decimal("0.05"),
) -> StopImpact:
    """Both sides' stops from the same price and structure, differing only in ATR."""
    kwargs = dict(
        atr_multiple=atr_multiple, min_stop_pct=min_stop_pct,
        max_stop_pct=max_stop_pct, flat_stop_pct=flat_stop_pct,
    )
    return StopImpact(
        local=_stop_for(price, local_atr, sma_50, **kwargs),
        broker=_stop_for(price, broker_atr, sma_50, **kwargs),
    )


# ------------------------------------------------------------------ summary


@dataclass(frozen=True)
class FieldSummary:
    """Aggregate error for one indicator field across the validation sample."""

    field: str
    compared: int
    missing: int
    max_abs_error: Decimal | None
    median_abs_error: Decimal | None
    max_rel_error: Decimal | None

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "compared": self.compared,
            "missing": self.missing,
            "max_abs_error": _s(self.max_abs_error),
            "median_abs_error": _s(self.median_abs_error),
            "max_rel_error": _s(self.max_rel_error),
        }


def _s(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def summarize(field: str, deltas: list[ValueDelta]) -> FieldSummary:
    """Aggregate one field. Median as well as max, because a single outlier and
    a systematically shifted series need to be distinguishable."""
    comparable = [d for d in deltas if d.comparable]
    errors = sorted(d.abs_error for d in comparable)
    rel = [d.rel_error for d in comparable if d.rel_error is not None]

    median: Decimal | None = None
    if errors:
        mid = len(errors) // 2
        median = (
            errors[mid]
            if len(errors) % 2
            else (errors[mid - 1] + errors[mid]) / Decimal(2)
        )

    return FieldSummary(
        field=field,
        compared=len(comparable),
        missing=len(deltas) - len(comparable),
        max_abs_error=errors[-1] if errors else None,
        median_abs_error=median,
        max_rel_error=max(rel) if rel else None,
    )


# --------------------------------------------------------------- tolerances

PROPOSED_TOLERANCES: dict[str, Decimal] = {
    # Observed maxima over 12 symbols x 30 sessions on 2026-08-25, against
    # identical 289-bar regular-session inputs. Each tolerance sits orders of
    # magnitude above the observed error and orders of magnitude below anything
    # a decision could notice -- chosen after measuring, not before.
    #
    #   field         observed max abs      proposed
    #   sma_20/50/200        4.0e-13          1e-6
    #   rsi_14               1.2e-07          1e-4   (RSI points)
    #   atr_14               1.8e-08          1e-6
    #   macd / signal / hist 6.8e-08          1e-5
    "sma_20": Decimal("1e-6"),
    "sma_50": Decimal("1e-6"),
    "sma_200": Decimal("1e-6"),
    "rsi_14": Decimal("1e-4"),
    "atr_14": Decimal("1e-6"),
    "macd": Decimal("1e-5"),
    "macd_signal": Decimal("1e-5"),
    "macd_hist": Decimal("1e-5"),
}
"""Per-indicator numeric tolerances, scaled to each indicator's consequence.

A single tolerance across all eight fields would be wrong in both directions at
once: 1e-6 is loose for an SMA that agrees to 4e-13, and tight for a MACD
histogram that legitimately passes through zero.

**Necessary, not sufficient.** Numeric agreement is the weaker half of the test.
The binding criterion is `REQUIRED_DECISION_AGREEMENT` below, because the
comparisons the strategy actually makes are directional, and a difference far
inside any of these tolerances can still flip one.
"""

REQUIRED_DECISION_AGREEMENT: frozenset[str] = frozenset(
    DecisionFlags.__dataclass_fields__
)
"""Flags that must agree exactly. Every one of them, with no numeric allowance.

Written as "all of them" rather than a hand-picked subset so a flag added later
is required by default rather than silently exempt. `macd_hist_rising` is the
one that motivated it: two histogram readings agreeing to 1e-8 can still
disagree on `hist > hist_prev`, and that is entry condition 5 -- the check that
separates buying a pullback from catching a falling knife.
"""


def within_tolerance(delta: ValueDelta) -> bool | None:
    """Does one observation meet its field's proposed tolerance?

    `None` when the pair is not comparable, so a missing value can never be
    counted as a pass.
    """
    if not delta.comparable:
        return None
    limit = PROPOSED_TOLERANCES.get(delta.field)
    if limit is None:
        raise KeyError(f"no tolerance proposed for {delta.field!r}")
    return delta.abs_error <= limit


__all__ = [
    "PROPOSED_TOLERANCES",
    "REQUIRED_DECISION_AGREEMENT",
    "DecisionFlags",
    "FieldSummary",
    "StopImpact",
    "ValueDelta",
    "decision_flags",
    "stop_impact",
    "summarize",
    "within_tolerance",
]
