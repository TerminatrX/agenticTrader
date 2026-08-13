"""Breakout / continuation momentum. STUB — not yet implemented.

Intended thesis, the mirror image of `trend_pullback`: rather than buying
weakness inside strength, buy strength confirming itself — a push through a
recent range high on expanding volume, with the trend stack already aligned.

Sketch of the entry conditions, to be filled in:

    1. Price above the 20-, 50-, and 200-day averages
    2. Close above the highest high of the prior N bars (Donchian upper band)
    3. Volume on the breakout bar above its own 30-day average by some factor
    4. RSI strong but short of exhaustion, roughly [55, 75]
    5. Not extended more than `max_extension_pct` above the 20-day

Two design notes worth settling before writing it:

    - Momentum entries and pullback entries can fire on the same symbol days
      apart and produce a doubled position. The risk engine's per-symbol
      exposure cap catches that, but the orchestrator should also decide
      whether two strategies may hold the same name at all.
    - Breakouts need the *live* price, not the last completed bar, since the
      whole point is acting on today's move. That makes this the first
      strategy where `snapshot.reference_price` is the wrong anchor — it will
      need `last_price` plus a staleness guard.

`evaluate` deliberately returns NONE rather than raising, so an accidentally
enabled stub cannot halt a cycle.
"""

from __future__ import annotations

from agentic_trader.models import MarketSnapshot, Signal
from agentic_trader.strategies.base import Strategy, StrategyContext, register


@register
class MomentumStrategy(Strategy):
    name = "momentum"

    DEFAULTS = {
        "breakout_lookback": 20,
        "volume_surge_multiple": 1.5,
        "rsi_floor": 55.0,
        "rsi_ceiling": 75.0,
        "max_extension_pct": 0.08,
        "stop_pct": 0.06,
    }

    def evaluate(self, snapshot: MarketSnapshot, context: StrategyContext) -> Signal:
        return self._no_signal(snapshot, "momentum strategy not yet implemented")
