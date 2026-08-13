"""Buy an orderly pullback inside an intact uptrend.

The thesis: in a name whose long-term trend is up, a retracement to the
short-term average is a discount rather than a warning — but only once the
selling has actually stopped. Every condition below exists to separate those
two cases.

Entry requires all of:

    1. Price above the 200-day average          (long-term trend intact)
    2. 50-day above the 200-day average         (medium term agrees)
    3. Price below the 20-day average           (an actual pullback)
    4. RSI(14) inside [rsi_floor, rsi_ceiling]  (soft, not capitulating)
    5. MACD histogram rising                    (selling pressure easing)
    6. Pullback depth within max_pullback_pct   (not a trend break)

Condition 5 is the one that earns its keep. Conditions 1-4 will happily fire in
the middle of a collapse; a rising histogram is what says the fall is decaying.
Dropping it turns this into a knife-catcher.

Exits are deliberately simpler than entries. A position closes when the trend
premise itself breaks — price losing the 50-day — or when momentum has run to
an extreme worth harvesting. Stops are enforced by the risk layer, not here.
"""

from __future__ import annotations

from decimal import Decimal

from agentic_trader.market import signals
from agentic_trader.market.regime import classify_regime
from agentic_trader.models import MarketSnapshot, Side, Signal, SignalStrength
from agentic_trader.strategies.base import Strategy, StrategyContext, register


@register
class TrendPullbackStrategy(Strategy):
    name = "trend_pullback"

    # Defaults are overridable per-symbol via config/strategies.yaml.
    DEFAULTS = {
        "rsi_floor": 30.0,
        "rsi_ceiling": 45.0,
        "max_pullback_pct": Decimal("0.12"),
        "stop_pct": Decimal("0.05"),
        "target_r_multiple": Decimal("2.0"),
        "exit_rsi": 72.0,
        "min_confidence": 0.5,
    }

    def evaluate(self, snapshot: MarketSnapshot, context: StrategyContext) -> Signal:
        if not snapshot.tradable:
            return self._no_signal(snapshot, f"symbol not tradable: {snapshot.staleness_note}")

        if context.holds_position:
            return self._evaluate_exit(snapshot, context)
        return self._evaluate_entry(snapshot, context)

    # ------------------------------------------------------------------ entry

    def _evaluate_entry(self, snapshot: MarketSnapshot, context: StrategyContext) -> Signal:
        p = lambda k: context.param(k, self.DEFAULTS[k])  # noqa: E731
        price = snapshot.reference_price
        ind = snapshot.indicators

        reasons: list[str] = []
        failed: list[str] = []
        metrics: dict[str, object] = {}

        regime = classify_regime(snapshot)
        metrics["regime"] = regime.value

        # Each check records why it passed or failed. `None` means the data
        # needed to judge it was absent, which counts as a failure.
        def check(label: str, result: bool | None, detail: str) -> bool:
            if result is None:
                failed.append(f"{label}: indeterminate ({detail})")
                return False
            (reasons if result else failed).append(f"{label}: {detail}")
            return result

        above_200 = check(
            "trend",
            signals.above_long_term_trend(snapshot),
            f"price {price:.2f} vs SMA200 {_fmt(ind.sma_200)}",
        )
        stack = check(
            "ma_stack",
            signals.trend_stack_bullish(snapshot),
            f"SMA50 {_fmt(ind.sma_50)} vs SMA200 {_fmt(ind.sma_200)}",
        )
        pullback = check(
            "pullback",
            signals.in_pullback(snapshot),
            f"price {price:.2f} vs SMA20 {_fmt(ind.sma_20)}",
        )

        rsi_floor, rsi_ceiling = float(p("rsi_floor")), float(p("rsi_ceiling"))
        rsi_ok = check(
            "rsi_band",
            signals.rsi_in_band(snapshot, rsi_floor, rsi_ceiling),
            f"RSI {_fmt(ind.rsi_14)} vs [{rsi_floor:g}, {rsi_ceiling:g}]",
        )
        metrics["rsi_14"] = ind.rsi_14

        stabilizing = check(
            "momentum_stabilizing",
            signals.momentum_stabilizing(snapshot),
            f"MACD hist {_fmt(ind.macd_hist)} vs prior {_fmt(ind.macd_hist_prev)}",
        )
        metrics["macd_hist"] = ind.macd_hist

        depth = signals.distance_below_sma20(snapshot)
        max_depth = p("max_pullback_pct")
        metrics["pullback_depth_pct"] = float(depth) if depth is not None else None
        depth_ok = check(
            "pullback_depth",
            None if depth is None else abs(depth) <= max_depth,
            f"{_pct(depth)} below SMA20 (max {_pct(max_depth)})",
        )

        all_passed = all([above_200, stack, pullback, rsi_ok, stabilizing, depth_ok])

        stop_price = (price * (Decimal("1") - p("stop_pct"))).quantize(Decimal("0.01"))
        # Anchor the stop below the 50-day when that sits lower than a flat
        # percentage would put it: the average is where the thesis actually
        # fails, and a stop above it gets taken out by noise the premise allows.
        if ind.sma_50 is not None and ind.sma_50 < price:
            structural = (ind.sma_50 * Decimal("0.99")).quantize(Decimal("0.01"))
            stop_price = min(stop_price, structural)

        risk_per_share = price - stop_price
        target_price = (price + risk_per_share * p("target_r_multiple")).quantize(Decimal("0.01"))

        if not all_passed:
            strength = (
                SignalStrength.WATCH
                if above_200 and stack and pullback
                else SignalStrength.NONE
            )
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                strength=strength,
                side=Side.BUY if strength is SignalStrength.WATCH else None,
                confidence=0.0,
                reference_price=price,
                reasons=reasons,
                failed_conditions=failed,
                metrics=metrics,
            )

        confidence = self._confidence(snapshot, rsi_floor, rsi_ceiling)
        metrics["confidence_inputs"] = {
            "rsi": ind.rsi_14,
            "hist_delta": (
                None
                if ind.macd_hist is None or ind.macd_hist_prev is None
                else ind.macd_hist - ind.macd_hist_prev
            ),
        }

        floor = float(p("min_confidence"))
        if confidence < floor:
            failed.append(f"confidence {confidence:.2f} below floor {floor:.2f}")
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                strength=SignalStrength.WATCH,
                side=Side.BUY,
                confidence=confidence,
                reference_price=price,
                reasons=reasons,
                failed_conditions=failed,
                metrics=metrics,
            )

        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            strength=SignalStrength.ENTER,
            side=Side.BUY,
            confidence=confidence,
            reference_price=price,
            stop_price=stop_price,
            target_price=target_price,
            reasons=reasons,
            failed_conditions=failed,
            metrics=metrics,
        )

    # ------------------------------------------------------------------- exit

    def _evaluate_exit(self, snapshot: MarketSnapshot, context: StrategyContext) -> Signal:
        p = lambda k: context.param(k, self.DEFAULTS[k])  # noqa: E731
        price = snapshot.reference_price
        ind = snapshot.indicators

        reasons: list[str] = []

        # The premise was "uptrend intact". Losing the 50-day says it is not.
        if ind.sma_50 is not None and price < ind.sma_50:
            reasons.append(f"price {price:.2f} closed below SMA50 {ind.sma_50:.2f} — thesis broken")

        if ind.rsi_14 is not None and ind.rsi_14 >= float(p("exit_rsi")):
            reasons.append(f"RSI {ind.rsi_14:.1f} at/above exit threshold {float(p('exit_rsi')):g}")

        if not reasons:
            return Signal(
                symbol=snapshot.symbol,
                strategy=self.name,
                strength=SignalStrength.NONE,
                reference_price=price,
                reasons=["holding: no exit condition met"],
                metrics={"rsi_14": ind.rsi_14},
            )

        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            strength=SignalStrength.EXIT,
            side=Side.SELL,
            confidence=1.0,
            reference_price=price,
            reasons=reasons,
            metrics={"rsi_14": ind.rsi_14},
        )

    # ------------------------------------------------------------- confidence

    def _confidence(self, snapshot: MarketSnapshot, rsi_floor: float, rsi_ceiling: float) -> float:
        """Blend how deep the pullback ran with how firmly it is turning.

        Kept simple and legible on purpose: a confidence score nobody can
        explain is worse than no score, because sizing depends on it.
        """
        ind = snapshot.indicators
        score = 0.5

        # Lower RSI within the accepted band means a better discount.
        if ind.rsi_14 is not None and rsi_ceiling > rsi_floor:
            position_in_band = (ind.rsi_14 - rsi_floor) / (rsi_ceiling - rsi_floor)
            score += 0.2 * (1.0 - min(max(position_in_band, 0.0), 1.0))

        # A sharply improving histogram is a firmer turn than a marginal one.
        if ind.macd_hist is not None and ind.macd_hist_prev is not None:
            delta = ind.macd_hist - ind.macd_hist_prev
            if delta > 0:
                magnitude = abs(ind.macd_hist_prev) or 1.0
                score += 0.2 * min(delta / magnitude, 1.0)

        # A rising RSI confirms the same turn from a second angle.
        if ind.rsi_improving:
            score += 0.1

        return round(min(max(score, 0.0), 1.0), 3)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{abs(value) * 100:.2f}%"
