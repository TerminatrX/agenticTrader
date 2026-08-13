"""Pure derivations over a snapshot, shared by strategies.

Nothing here decides anything. These are the small, individually testable
predicates that strategies compose into an opinion — kept separate so that a
condition can be unit-tested against a hand-built snapshot without constructing
a whole strategy.

Every function returns `None` when the inputs it needs are absent. Callers must
treat `None` as "unknown" and refuse to trade on it, never coerce it to False.
"""

from __future__ import annotations

from decimal import Decimal

from agentic_trader.models import MarketSnapshot


def pct_from(value: Decimal, reference: Decimal) -> Decimal | None:
    """Signed fractional distance of `value` from `reference`."""
    if not reference:
        return None
    return (value - reference) / reference


def above_long_term_trend(snapshot: MarketSnapshot) -> bool | None:
    """Price above the 200-day average — the coarsest "is this a bull" filter."""
    sma = snapshot.indicators.sma_200
    if sma is None:
        return None
    return snapshot.reference_price > sma


def trend_stack_bullish(snapshot: MarketSnapshot) -> bool | None:
    """50-day above 200-day: the medium-term trend agrees with the long-term one."""
    ind = snapshot.indicators
    if ind.sma_50 is None or ind.sma_200 is None:
        return None
    return ind.sma_50 > ind.sma_200


def in_pullback(snapshot: MarketSnapshot) -> bool | None:
    """Price has retraced below its short-term average while the trend holds."""
    sma = snapshot.indicators.sma_20
    if sma is None:
        return None
    return snapshot.reference_price < sma


def rsi_in_band(snapshot: MarketSnapshot, low: float, high: float) -> bool | None:
    rsi = snapshot.indicators.rsi_14
    if rsi is None:
        return None
    return low <= rsi <= high


def momentum_stabilizing(snapshot: MarketSnapshot) -> bool | None:
    """MACD histogram rising — selling pressure easing rather than accelerating.

    This is the condition that separates buying a pullback from catching a
    falling knife. A pullback that still has a deepening histogram has not
    finished falling.
    """
    return snapshot.indicators.macd_hist_improving


def distance_below_sma20(snapshot: MarketSnapshot) -> Decimal | None:
    sma = snapshot.indicators.sma_20
    if sma is None:
        return None
    return pct_from(snapshot.reference_price, sma)


def extension_from_52w_high(snapshot: MarketSnapshot) -> Decimal | None:
    if snapshot.high_52w is None:
        return None
    return pct_from(snapshot.reference_price, snapshot.high_52w)


def liquid_enough(snapshot: MarketSnapshot, min_avg_volume: Decimal) -> bool | None:
    if snapshot.average_volume_30d is None:
        return None
    return snapshot.average_volume_30d >= min_avg_volume


def realized_volatility_pct(snapshot: MarketSnapshot, lookback: int = 20) -> Decimal | None:
    """Mean absolute daily range over `lookback` bars, as a fraction of close.

    A cheap stand-in for ATR when the ATR indicator was not fetched. Used to
    widen or tighten a percentage stop to the symbol's own behaviour instead of
    applying one fixed number to every name.
    """
    bars = snapshot.bars[-lookback:]
    if len(bars) < 2:
        return None
    ranges = [(b.high - b.low) / b.close for b in bars if b.close]
    if not ranges:
        return None
    return sum(ranges) / Decimal(len(ranges))


def gap_pct(snapshot: MarketSnapshot) -> Decimal | None:
    """Overnight gap between the previous close and the latest price."""
    if snapshot.previous_close is None:
        return None
    return pct_from(snapshot.last_price, snapshot.previous_close)
