"""Market-wide regime, classified from index snapshots.

Distinct from `market.regime`, which classifies one *symbol's* trend and gates
whether a strategy may fire on it. This module asks a different question: what
is the whole market doing? A pullback in a name whose own chart looks perfect
is a different trade when the index is falling, and the symbol-local classifier
cannot see that at all.

**This is recorded, not enforced.** Nothing branches on the result and no gate
consults it. That is deliberate: the useful version of this rule is

    trend_pullback in BULL_TREND      expectancy +0.34R
    trend_pullback in RANGE           expectancy -0.18R
    trend_pullback in HIGH_VOLATILITY expectancy -0.52R

and none of those numbers exist yet. Wiring a guess into a gate would suppress
exactly the trades needed to find out whether the guess was right. Journal
first; let the rule follow the evidence.

The classification is deliberately crude and fully deterministic — the same
snapshots always produce the same regime, so a stored decision replays exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agentic_trader.models import MarketSnapshot

# ATR as a fraction of price, above which the market is treated as
# high-volatility regardless of direction. SPY's long-run daily range is well
# under 1%; sustained readings above this mark 2020- and 2022-style conditions,
# where trend signals degrade and gaps routinely jump stops.
HIGH_VOLATILITY_ATR_PCT = Decimal("0.02")


class MarketRegime(StrEnum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MarketContext:
    """The market backdrop a decision was made against.

    Carries the inputs alongside the verdict. A regime label with no record of
    what produced it cannot be re-examined later, and re-examining it later is
    the entire reason this is being collected.
    """

    regime: MarketRegime
    reasons: list[str] = field(default_factory=list)
    inputs: dict[str, float | None] = field(default_factory=dict)
    as_of: datetime | None = None

    @property
    def is_known(self) -> bool:
        return self.regime is not MarketRegime.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "reasons": list(self.reasons),
            "inputs": dict(self.inputs),
            "as_of": self.as_of.isoformat() if self.as_of else None,
        }


def _trend_of(snapshot: MarketSnapshot) -> tuple[str | None, list[str]]:
    """Directional read on one index: 'up', 'down', 'mixed', or None."""
    ind = snapshot.indicators
    if ind.sma_50 is None or ind.sma_200 is None:
        return None, [f"{snapshot.symbol}: missing SMA50/SMA200"]

    price = snapshot.reference_price
    above_long = price > ind.sma_200
    stack_up = ind.sma_50 > ind.sma_200
    detail = (
        f"{snapshot.symbol}: price {price:.2f} vs SMA200 {ind.sma_200:.2f}, "
        f"SMA50 {ind.sma_50:.2f}"
    )

    if above_long and stack_up:
        return "up", [detail + " -> up"]
    if not above_long and not stack_up:
        return "down", [detail + " -> down"]
    return "mixed", [detail + " -> mixed"]


def _volatility_of(snapshot: MarketSnapshot) -> Decimal | None:
    ind = snapshot.indicators
    price = snapshot.reference_price
    if ind.atr_14 is None or price <= 0:
        return None
    return ind.atr_14 / price


def classify_market_regime(
    spy: MarketSnapshot | None,
    qqq: MarketSnapshot | None = None,
    *,
    high_volatility_atr_pct: Decimal = HIGH_VOLATILITY_ATR_PCT,
) -> MarketContext:
    """Classify the market backdrop from SPY, optionally confirmed by QQQ.

    QQQ is a second opinion rather than an equal vote: SPY defines the regime,
    and QQQ disagreeing downgrades a clean trend to RANGE. Broad and tech-heavy
    measures parting company is itself the signal — it means the move is
    sectoral, not market-wide, and a single-name trend read borrows confidence
    it has not earned.
    """
    if spy is None:
        return MarketContext(
            regime=MarketRegime.UNKNOWN,
            reasons=["no index snapshot supplied"],
        )

    reasons: list[str] = []
    inputs: dict[str, float | None] = {}

    # Volatility outranks direction. A violently moving market is its own
    # regime: trend labels stop describing much, and the gap risk that breaks
    # managed stops is at its worst.
    for snap in (s for s in (spy, qqq) if s is not None):
        vol = _volatility_of(snap)
        inputs[f"{snap.symbol.lower()}_atr_pct"] = float(vol) if vol is not None else None
        if vol is not None and vol > high_volatility_atr_pct:
            reasons.append(
                f"{snap.symbol}: ATR {vol:.2%} of price exceeds "
                f"{high_volatility_atr_pct:.1%}"
            )
            return MarketContext(
                regime=MarketRegime.HIGH_VOLATILITY,
                reasons=reasons,
                inputs=inputs,
                as_of=spy.quote_as_of or spy.captured_at,
            )

    spy_trend, spy_notes = _trend_of(spy)
    reasons.extend(spy_notes)
    inputs["spy_price"] = float(spy.reference_price)

    if spy_trend is None:
        return MarketContext(
            regime=MarketRegime.UNKNOWN,
            reasons=reasons,
            inputs=inputs,
            as_of=spy.quote_as_of or spy.captured_at,
        )

    regime = {
        "up": MarketRegime.BULL_TREND,
        "down": MarketRegime.BEAR_TREND,
        "mixed": MarketRegime.RANGE,
    }[spy_trend]

    if qqq is not None:
        qqq_trend, qqq_notes = _trend_of(qqq)
        reasons.extend(qqq_notes)
        inputs["qqq_price"] = float(qqq.reference_price)
        if (
            qqq_trend is not None
            and qqq_trend != spy_trend
            and regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)
        ):
            reasons.append(
                f"SPY reads {spy_trend} and QQQ reads {qqq_trend} — a split "
                "market is not a trend"
            )
            regime = MarketRegime.RANGE

    return MarketContext(
        regime=regime,
        reasons=reasons,
        inputs=inputs,
        as_of=spy.quote_as_of or spy.captured_at,
    )
