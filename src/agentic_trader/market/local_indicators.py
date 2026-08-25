"""Deterministic indicators computed from authoritative historical bars.

**Nothing here feeds a decision.** Broker indicator endpoints remain the
authoritative inputs to `MarketSnapshot.indicators`; these functions exist so
that local-vs-broker equivalence can be *measured* before any replacement is
considered. Wiring them into the decision path is a separate, separately
reviewed milestone.

Why write them out by hand
--------------------------

No pandas-ta, no TA-Lib. Not out of preference for reinvention, but because the
question this module exists to answer is "does our arithmetic agree with the
broker's, and where does it not" — and an opaque dependency answers that with
another opaque dependency. Every seed and recurrence below is stated in the
docstring and pinned by a hand-calculable test, so a disagreement can be
localized to a formula rather than to a library version.

Precision
---------

Everything is computed in `Decimal` and returned in `Decimal`, including RSI and
MACD, which the snapshot model carries as `float`. Two reasons: `Decimal` sums
are order-independent where binary floats are not, so a rolling mean has one
answer rather than one per summation order; and the conversion to `float` is
then a single, visible step at the comparison boundary rather than accumulated
noise throughout a 200-step recursion.

Intermediates are never rounded to resemble broker output. Rounding to match
would hide exactly the disagreement being measured.

Seeds are conventions, not laws
-------------------------------

Recursive indicators need a starting value, and there is more than one
defensible choice. `EmaSeed` makes that explicit and selectable rather than
buried, so the validation harness can ask which convention the broker uses
instead of assuming one. With enough warm-up the choice stops mattering — seed
influence decays geometrically, which is precisely the argument for the
convergence allowance in `acquisition.py`, and `required_bars` below quantifies
it rather than asserting it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from agentic_trader.models import Bar

# Seed influence is considered spent below this relative weight. At 1e-6 the
# starting value contributes under a millionth of the result -- for RSI, whose
# range is 0-100, under 1e-4 RSI points, which is orders of magnitude inside any
# tolerance a decision could care about.
SEED_DECAY_TARGET = Decimal("1e-6")


class EmaSeed(StrEnum):
    """How a recursive average starts.

    Both are in common use and both are defensible. Neither is "correct"; what
    matters is knowing which one produced a given number.
    """

    SMA = "sma"
    """Seed with the simple mean of the first `period` values, then recurse.
    The convention Wilder describes and the more common one in charting
    packages. First output appears at index `period - 1`."""

    FIRST_VALUE = "first_value"
    """Seed with the first value itself, then recurse from index 1. Produces
    output earlier and converges to the same series; used by several libraries
    and by some streaming implementations."""


@dataclass(frozen=True)
class LocalIndicatorSet:
    """Locally derived values for one symbol, mirroring `Indicators`.

    Every field is optional and `None` means *insufficient history*, never zero
    and never a shortened period. A missing SMA200 and an SMA200 that happens to
    equal 0 must remain distinguishable: one says "we could not compute this",
    the other is a value.
    """

    rsi_14: Decimal | None = None
    rsi_prev: Decimal | None = None

    macd: Decimal | None = None
    macd_signal: Decimal | None = None
    macd_hist: Decimal | None = None
    macd_hist_prev: Decimal | None = None

    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None

    atr_14: Decimal | None = None

    #: Bars the computation was given, for interpreting any `None` above.
    bar_count: int = 0

    #: Why a value is missing, keyed by field name. Populated only for absences.
    unavailable: tuple[tuple[str, str], ...] = ()


# --------------------------------------------------------------------- SMA


def sma(closes: Sequence[Decimal], period: int) -> Decimal | None:
    """Arithmetic mean of the last `period` closes.

    A finite window: it carries no seed and no memory beyond `period`, which is
    why it needs no convergence allowance and why it is the one indicator whose
    value cannot drift with how much history was requested.

    Returns `None` below `period` bars. A partial-window mean would be a
    different statistic wearing the same name.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(closes) < period:
        return None
    window = closes[-period:]
    return sum(window, Decimal(0)) / Decimal(period)


def sma_series(closes: Sequence[Decimal], period: int) -> list[Decimal]:
    """Every SMA value the input supports, oldest first. Empty if too short."""
    if len(closes) < period:
        return []
    return [
        sum(closes[i - period + 1 : i + 1], Decimal(0)) / Decimal(period)
        for i in range(period - 1, len(closes))
    ]


# --------------------------------------------------------------------- EMA


def ema_series(
    values: Sequence[Decimal], period: int, *, seed: EmaSeed = EmaSeed.SMA
) -> list[Decimal]:
    """Exponential moving average, oldest first.

    Smoothing factor is the standard `alpha = 2 / (period + 1)`, and the
    recurrence is::

        ema_i = value_i * alpha + ema_(i-1) * (1 - alpha)

    Under `EmaSeed.SMA` the first output corresponds to input index
    `period - 1` and equals the mean of the first `period` values. Under
    `EmaSeed.FIRST_VALUE` the first output corresponds to input index 0 and
    equals `values[0]`.

    The returned list is therefore shorter than the input under SMA seeding, and
    the same length under first-value seeding. Callers aligning series by
    position must account for that; `macd_series` does.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if not values:
        return []

    alpha = Decimal(2) / Decimal(period + 1)
    one_minus = Decimal(1) - alpha

    if seed is EmaSeed.SMA:
        if len(values) < period:
            return []
        current = sum(values[:period], Decimal(0)) / Decimal(period)
        out = [current]
        rest = values[period:]
    else:
        current = values[0]
        out = [current]
        rest = values[1:]

    for value in rest:
        current = value * alpha + current * one_minus
        out.append(current)
    return out


# --------------------------------------------------------------------- RSI


def rsi_series(
    closes: Sequence[Decimal], period: int = 14
) -> list[Decimal]:
    """Wilder's RSI, oldest first. Stated in full rather than called standard.

    1. **Delta.** ``delta_i = close_i - close_(i-1)`` for ``i >= 1``. There is
       no delta for the first bar, so `n` closes yield `n - 1` deltas.
    2. **Gain / loss.** ``gain_i = max(delta_i, 0)``,
       ``loss_i = max(-delta_i, 0)``. Both are non-negative; an unchanged close
       contributes zero to each.
    3. **Seed.** The first averages are the *simple* means of the first
       `period` gains and losses — Wilder's own seeding, not an EMA of them.
       This lands at close index `period`.
    4. **Recurrence.** ``avg_i = (avg_(i-1) * (period - 1) + x_i) / period``,
       i.e. Wilder smoothing with ``alpha = 1 / period``. Note this is *not*
       the ``2 / (period + 1)`` used by MACD's EMAs; conflating them is the
       single most common way to produce an RSI that is close but never right.
    5. **Output.** ``RS = avg_gain / avg_loss``,
       ``RSI = 100 - 100 / (1 + RS)``.

    Boundary conventions, all reachable and all tested:

    - ``avg_loss == 0`` and ``avg_gain > 0`` → **100**. RS is unbounded and the
      limit of the formula is 100.
    - ``avg_gain == 0`` and ``avg_loss > 0`` → **0**, by the same limit.
    - **both zero** → **50**. A perfectly flat stretch has no directional
      pressure either way. Taking the zero-loss branch first would return 100
      and claim maximum strength for a price that has not moved, which is
      actively wrong — the exit gate fires at RSI >= 72. This case is
      vanishingly rare in real data but it is a genuine convention choice, so
      it is stated rather than inherited from branch ordering.

    Returns `[]` when there are fewer than ``period + 1`` closes.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(closes) < period + 1:
        return []

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else Decimal(0) for d in deltas]
    losses = [-d if d < 0 else Decimal(0) for d in deltas]

    p = Decimal(period)
    avg_gain = sum(gains[:period], Decimal(0)) / p
    avg_loss = sum(losses[:period], Decimal(0)) / p

    out = [_rsi_from(avg_gain, avg_loss)]
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (p - 1) + gains[i]) / p
        avg_loss = (avg_loss * (p - 1) + losses[i]) / p
        out.append(_rsi_from(avg_gain, avg_loss))
    return out


def _rsi_from(avg_gain: Decimal, avg_loss: Decimal) -> Decimal:
    if avg_loss == 0 and avg_gain == 0:
        return Decimal(50)
    if avg_loss == 0:
        return Decimal(100)
    if avg_gain == 0:
        return Decimal(0)
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))


# --------------------------------------------------------------------- ATR


def true_range_series(bars: Sequence[Bar]) -> list[Decimal]:
    """True range, oldest first, starting at the *second* bar.

    ``TR_i = max(high_i - low_i, |high_i - close_(i-1)|, |low_i - close_(i-1)|)``

    The first bar has no previous close, so it has no true range. Substituting
    ``high - low`` there is a common shortcut and is not taken: it silently
    changes the seed average, which then propagates through every later value.
    `n` bars therefore yield `n - 1` true ranges.
    """
    return [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        for i in range(1, len(bars))
    ]


def atr_series(bars: Sequence[Bar], period: int = 14) -> list[Decimal]:
    """Wilder's ATR, oldest first.

    Seeded with the simple mean of the first `period` true ranges, then smoothed
    by ``atr_i = (atr_(i-1) * (period - 1) + tr_i) / period`` — the same
    ``alpha = 1 / period`` as RSI, and again not MACD's ``2 / (period + 1)``.

    Needs ``period + 1`` bars for a first value: one to supply the previous
    close, `period` to average. Returns `[]` below that.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    trs = true_range_series(bars)
    if len(trs) < period:
        return []

    p = Decimal(period)
    current = sum(trs[:period], Decimal(0)) / p
    out = [current]
    for i in range(period, len(trs)):
        current = (current * (p - 1) + trs[i]) / p
        out.append(current)
    return out


# -------------------------------------------------------------------- MACD


@dataclass(frozen=True)
class MacdPoint:
    """One MACD observation. `line - signal` is `histogram`, exactly."""

    line: Decimal
    signal: Decimal
    histogram: Decimal


def macd_series(
    closes: Sequence[Decimal],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    seed: EmaSeed = EmaSeed.SMA,
) -> list[MacdPoint]:
    """MACD line, signal, and histogram — oldest first.

    1. **Fast and slow EMAs** of the closes, both seeded per `seed`.
    2. **Alignment.** The two EMAs begin at different input indices under SMA
       seeding (`fast - 1` and `slow - 1`), so the fast series is trimmed from
       the front to line up with the slow one before subtracting. Subtracting
       them by position without this step silently offsets the line by
       ``slow - fast`` bars, which is the kind of error that produces a
       plausible-looking series that is wrong everywhere.
    3. **Line.** ``macd_i = ema_fast_i - ema_slow_i``.
    4. **Signal.** An EMA of the *MACD line series* — not of price — seeded the
       same way, so it begins `signal - 1` points into the line series under SMA
       seeding.
    5. **Histogram.** ``histogram = line - signal``, no scaling. Some platforms
       plot ``2 x (line - signal)``; that convention is not used here, and if
       the broker used it the comparison would show a clean factor of two rather
       than noise.

    Warm-up under SMA seeding: the line needs `slow` closes, and the signal
    needs `signal` line points, so the first histogram appears at close index
    ``slow + signal - 2`` — 33 for the (12, 26, 9) defaults, i.e. 34 closes. A
    second, for `macd_hist_prev`, needs 35.

    Returns `[]` when the input is too short.
    """
    for name, value in (("fast", fast), ("slow", slow), ("signal", signal)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")

    fast_ema = ema_series(closes, fast, seed=seed)
    slow_ema = ema_series(closes, slow, seed=seed)
    if not fast_ema or not slow_ema:
        return []

    # Trim the longer (earlier-starting) fast series so index 0 of both refers
    # to the same close.
    offset = len(fast_ema) - len(slow_ema)
    if offset > 0:
        fast_ema = fast_ema[offset:]
    elif offset < 0:  # pragma: no cover - slow always starts later
        slow_ema = slow_ema[-offset:]

    line = [f - s for f, s in zip(fast_ema, slow_ema, strict=True)]
    signal_line = ema_series(line, signal, seed=seed)
    if not signal_line:
        return []

    line_tail = line[len(line) - len(signal_line) :]
    return [
        MacdPoint(line=m, signal=s, histogram=m - s)
        for m, s in zip(line_tail, signal_line, strict=True)
    ]


# ------------------------------------------------------------ the whole set


def _last(series: Sequence[Decimal], back: int = 0) -> Decimal | None:
    index = len(series) - 1 - back
    return series[index] if index >= 0 else None


def compute_all(
    bars: Sequence[Bar],
    *,
    seed: EmaSeed = EmaSeed.SMA,
    rsi_period: int = 14,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> LocalIndicatorSet:
    """Every value the strategy reads, from one bar series.

    Anything the history cannot support comes back `None` with a recorded
    reason. Nothing is approximated, substituted, or computed over a shortened
    period — a caller must be able to tell "not enough history" from a value.
    """
    closes = [b.close for b in bars]
    missing: list[tuple[str, str]] = []

    def need(field: str, value: Decimal | None, required: int) -> Decimal | None:
        if value is None:
            missing.append(
                (field, f"needs {required} bars, given {len(bars)}")
            )
        return value

    rsi = rsi_series(closes, rsi_period)
    atr = atr_series(bars, atr_period)
    macd = macd_series(
        closes, fast=macd_fast, slow=macd_slow, signal=macd_signal, seed=seed
    )

    return LocalIndicatorSet(
        sma_20=need("sma_20", sma(closes, 20), 20),
        sma_50=need("sma_50", sma(closes, 50), 50),
        sma_200=need("sma_200", sma(closes, 200), 200),
        rsi_14=need("rsi_14", _last(rsi), rsi_period + 1),
        rsi_prev=need("rsi_prev", _last(rsi, 1), rsi_period + 2),
        atr_14=need("atr_14", _last(atr), atr_period + 1),
        macd=need("macd", macd[-1].line if macd else None, macd_slow + macd_signal - 1),
        macd_signal=need(
            "macd_signal", macd[-1].signal if macd else None, macd_slow + macd_signal - 1
        ),
        macd_hist=need(
            "macd_hist", macd[-1].histogram if macd else None, macd_slow + macd_signal - 1
        ),
        macd_hist_prev=need(
            "macd_hist_prev",
            macd[-2].histogram if len(macd) >= 2 else None,
            macd_slow + macd_signal,
        ),
        bar_count=len(bars),
        unavailable=tuple(missing),
    )


# ----------------------------------------------- how much history is needed


@dataclass(frozen=True)
class BarRequirement:
    """What one indicator needs, split into the two questions that differ."""

    label: str

    minimum_bars: int
    """Fewest bars that produce a value at all. Mathematically sufficient and
    numerically inadequate for anything recursive."""

    converged_bars: int
    """Bars for the result to be independent of the seed convention, to within
    `SEED_DECAY_TARGET`. This is the number that matters for provider
    equivalence: two implementations differing only in seed agree here, and
    disagree visibly at `minimum_bars`."""

    note: str = ""


def _decay_bars(alpha: Decimal) -> int:
    """Updates until seed weight `(1 - alpha)^k` falls below the target.

    Solved directly rather than iterated so the arithmetic is inspectable:
    ``k > ln(target) / ln(1 - alpha)``.
    """
    return math.ceil(
        math.log(float(SEED_DECAY_TARGET)) / math.log(float(Decimal(1) - alpha))
    )


def bar_requirements(
    *,
    rsi_period: int = 14,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> tuple[BarRequirement, ...]:
    """Derived, not chosen. Each number is arithmetic over the recurrences above.

    SMA carries no seed, so its two figures are equal — the honest statement
    that no amount of extra history changes a 200-bar mean.
    """
    wilder_rsi = _decay_bars(Decimal(1) / Decimal(rsi_period))
    wilder_atr = _decay_bars(Decimal(1) / Decimal(atr_period))
    ema_slow = _decay_bars(Decimal(2) / Decimal(macd_slow + 1))
    ema_signal = _decay_bars(Decimal(2) / Decimal(macd_signal + 1))

    return (
        BarRequirement("sma_20", 20, 20, "finite window; no seed to forget"),
        BarRequirement("sma_50", 50, 50, "finite window; no seed to forget"),
        BarRequirement("sma_200", 200, 200, "finite window; no seed to forget"),
        BarRequirement(
            "rsi_14",
            # +1 for the delta, +1 more because the strategy reads rsi_prev.
            rsi_period + 2,
            rsi_period + 2 + wilder_rsi,
            f"Wilder alpha=1/{rsi_period}; seed spent after {wilder_rsi} updates",
        ),
        BarRequirement(
            "atr_14",
            atr_period + 1,
            atr_period + 1 + wilder_atr,
            f"Wilder alpha=1/{atr_period}; seed spent after {wilder_atr} updates",
        ),
        BarRequirement(
            "macd",
            # Line needs `slow` closes; signal needs `signal` line points; the
            # strategy reads the previous histogram too.
            macd_slow + macd_signal,
            macd_slow + macd_signal + ema_slow + ema_signal,
            (
                f"EMA alpha=2/{macd_slow + 1} and 2/{macd_signal + 1}; seeds "
                f"spent after {ema_slow} and {ema_signal} updates, and they "
                "compose because the signal smooths an already-smoothed series"
            ),
        ),
    )


def required_bars(**kwargs: int) -> int:
    """The binding converged requirement across all six indicators."""
    return max(r.converged_bars for r in bar_requirements(**kwargs))


__all__ = [
    "SEED_DECAY_TARGET",
    "BarRequirement",
    "EmaSeed",
    "LocalIndicatorSet",
    "MacdPoint",
    "atr_series",
    "bar_requirements",
    "compute_all",
    "ema_series",
    "macd_series",
    "required_bars",
    "rsi_series",
    "sma",
    "sma_series",
    "true_range_series",
]
