"""Trend classification for a *single symbol*.

Answers one question: does this stock currently satisfy the trend premise a
strategy needs? A pullback-buying strategy is reasonable in an uptrend and
actively harmful in a downtrend, and the cheapest way to avoid that mistake is
to refuse to run it there at all. This one **gates** — the critic blocks a long
whose symbol regime contradicts it.

Do not confuse this with `market.market_regime`, which asks what SPY and QQQ are
doing and deliberately gates nothing. The two answer different questions at
different scopes, and conflating them is easy enough to be worth naming apart:

    SymbolTrendRegime   does THIS stock satisfy the trend premise?  -> may gate
    MarketRegime        what is the broad market doing?             -> journal only

The classification is intentionally crude — a handful of moving-average
relationships. A more sophisticated model belongs here later; what matters now
is that the seam exists and strategies consult it.
"""

from __future__ import annotations

from enum import StrEnum

from agentic_trader.market import signals
from agentic_trader.models import MarketSnapshot


class SymbolTrendRegime(StrEnum):
    UPTREND = "uptrend"
    PULLBACK_IN_UPTREND = "pullback_in_uptrend"
    RANGE = "range"
    DOWNTREND = "downtrend"
    UNKNOWN = "unknown"

    @property
    def allows_long_entry(self) -> bool:
        return self in (SymbolTrendRegime.UPTREND, SymbolTrendRegime.PULLBACK_IN_UPTREND)


def classify_symbol_regime(snapshot: MarketSnapshot) -> SymbolTrendRegime:
    ind = snapshot.indicators
    if ind.sma_50 is None or ind.sma_200 is None or ind.sma_20 is None:
        return SymbolTrendRegime.UNKNOWN

    price = snapshot.reference_price
    long_ok = signals.above_long_term_trend(snapshot)
    stack_ok = signals.trend_stack_bullish(snapshot)

    if long_ok is None or stack_ok is None:
        return SymbolTrendRegime.UNKNOWN

    if long_ok and stack_ok:
        # Still structurally bullish; the short-term average tells us whether
        # we are extended or resting.
        return (
            SymbolTrendRegime.PULLBACK_IN_UPTREND
            if price < ind.sma_20
            else SymbolTrendRegime.UPTREND
        )

    if not long_ok and not stack_ok:
        return SymbolTrendRegime.DOWNTREND

    # Averages disagree — trend is in transition, which is neither a clean
    # uptrend to buy nor a clean downtrend to avoid outright.
    return SymbolTrendRegime.RANGE
