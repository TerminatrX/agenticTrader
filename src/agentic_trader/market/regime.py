"""Coarse market-state classification.

Regime gates which strategies are allowed to fire. A pullback-buying strategy
is reasonable in an uptrend and actively harmful in a downtrend, and the
cheapest way to avoid that mistake is to refuse to run it there at all.

The classification is intentionally crude — a handful of moving-average
relationships. A more sophisticated model belongs here later; what matters now
is that the seam exists and strategies consult it.
"""

from __future__ import annotations

from enum import StrEnum

from agentic_trader.market import signals
from agentic_trader.models import MarketSnapshot


class Regime(StrEnum):
    UPTREND = "uptrend"
    PULLBACK_IN_UPTREND = "pullback_in_uptrend"
    RANGE = "range"
    DOWNTREND = "downtrend"
    UNKNOWN = "unknown"

    @property
    def allows_long_entry(self) -> bool:
        return self in (Regime.UPTREND, Regime.PULLBACK_IN_UPTREND)


def classify_regime(snapshot: MarketSnapshot) -> Regime:
    ind = snapshot.indicators
    if ind.sma_50 is None or ind.sma_200 is None or ind.sma_20 is None:
        return Regime.UNKNOWN

    price = snapshot.reference_price
    long_ok = signals.above_long_term_trend(snapshot)
    stack_ok = signals.trend_stack_bullish(snapshot)

    if long_ok is None or stack_ok is None:
        return Regime.UNKNOWN

    if long_ok and stack_ok:
        # Still structurally bullish; the short-term average tells us whether
        # we are extended or resting.
        return Regime.PULLBACK_IN_UPTREND if price < ind.sma_20 else Regime.UPTREND

    if not long_ok and not stack_ok:
        return Regime.DOWNTREND

    # Averages disagree — trend is in transition, which is neither a clean
    # uptrend to buy nor a clean downtrend to avoid outright.
    return Regime.RANGE
