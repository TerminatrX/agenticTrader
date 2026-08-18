"""Strategy tests.

The structure is one test per entry condition, each breaking exactly one thing
from a known-good snapshot. That proves every condition is load-bearing — a
condition no test can break is a condition that is not actually doing anything.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trader.models import Indicators, SignalStrength
from agentic_trader.strategies.base import StrategyContext
from agentic_trader.strategies.trend_pullback import TrendPullbackStrategy


def _evaluate(snapshot, position=None, **params):
    return TrendPullbackStrategy().evaluate(
        snapshot, StrategyContext(position=position, params=params)
    )


def _with_indicators(snapshot, **changes):
    return snapshot.model_copy(
        update={"indicators": snapshot.indicators.model_copy(update=changes)}
    )


def test_full_setup_produces_enter(bullish_pullback_snapshot):
    signal = _evaluate(bullish_pullback_snapshot)

    assert signal.strength is SignalStrength.ENTER
    assert signal.side is not None and signal.side.value == "buy"
    assert signal.stop_price is not None and signal.stop_price < signal.reference_price
    assert signal.target_price > signal.reference_price
    assert not signal.failed_conditions
    assert len(signal.reasons) == 7


def test_deepening_momentum_downgrades_to_watch(bullish_pullback_snapshot):
    """The knife-catch filter: same setup, histogram still falling.

    This is the real AAPL configuration as of 2026-08-12.
    """
    falling = _with_indicators(
        bullish_pullback_snapshot, macd_hist=-3.3428, macd_hist_prev=-3.2711
    )
    signal = _evaluate(falling)

    assert signal.strength is SignalStrength.WATCH
    assert any("momentum_stabilizing" in f for f in signal.failed_conditions)


def test_below_200day_is_not_an_entry(bullish_pullback_snapshot):
    broken = _with_indicators(bullish_pullback_snapshot, sma_200=Decimal("320.00"))
    signal = _evaluate(broken)

    assert signal.strength is SignalStrength.NONE
    assert any("trend" in f for f in signal.failed_conditions)


def test_inverted_ma_stack_is_not_an_entry(bullish_pullback_snapshot):
    inverted = _with_indicators(
        bullish_pullback_snapshot, sma_50=Decimal("270.00"), sma_200=Decimal("280.09")
    )
    signal = _evaluate(inverted)

    assert signal.strength is SignalStrength.NONE
    assert any("ma_stack" in f for f in signal.failed_conditions)


def test_no_pullback_means_no_entry(bullish_pullback_snapshot):
    """Price above the 20-day is an extension, not the setup."""
    extended = _with_indicators(bullish_pullback_snapshot, sma_20=Decimal("290.00"))
    signal = _evaluate(extended)

    assert signal.strength is SignalStrength.NONE
    assert any("pullback" in f for f in signal.failed_conditions)


def test_rsi_outside_band_downgrades(bullish_pullback_snapshot):
    for rsi in (25.0, 55.0):
        signal = _evaluate(_with_indicators(bullish_pullback_snapshot, rsi_14=rsi))
        assert signal.strength is not SignalStrength.ENTER
        assert any("rsi_band" in f for f in signal.failed_conditions)


def test_pullback_too_deep_is_rejected(bullish_pullback_snapshot):
    deep = _with_indicators(bullish_pullback_snapshot, sma_20=Decimal("400.00"))
    signal = _evaluate(deep)

    assert signal.strength is not SignalStrength.ENTER
    assert any("pullback_depth" in f for f in signal.failed_conditions)


def test_missing_indicators_never_produce_an_entry(bullish_pullback_snapshot):
    """Absent data is a failed condition, never an assumed pass."""
    blank = bullish_pullback_snapshot.model_copy(update={"indicators": Indicators()})
    signal = _evaluate(blank)

    assert signal.strength is SignalStrength.NONE
    assert any("indeterminate" in f for f in signal.failed_conditions)


def test_untradable_symbol_short_circuits(bullish_pullback_snapshot):
    halted = bullish_pullback_snapshot.model_copy(
        update={"tradable": False, "staleness_note": "halted"}
    )
    signal = _evaluate(halted)

    assert signal.strength is SignalStrength.NONE
    assert "not tradable" in signal.failed_conditions[0]


def test_params_override_defaults(bullish_pullback_snapshot):
    tightened = _evaluate(bullish_pullback_snapshot, rsi_ceiling=35.0)

    assert tightened.strength is not SignalStrength.ENTER
    assert any("rsi_band" in f for f in tightened.failed_conditions)


def test_low_confidence_downgrades_to_watch(bullish_pullback_snapshot):
    signal = _evaluate(bullish_pullback_snapshot, min_confidence=0.99)

    assert signal.strength is SignalStrength.WATCH
    assert any("confidence" in f for f in signal.failed_conditions)


def test_structure_widens_the_stop_below_the_50day(bullish_pullback_snapshot):
    """The stop belongs where the thesis fails, not at a fixed percentage.

    With the 50-day at 295.00, its 1%-under level (292.05) sits below where a
    tight percentage stop would land. The average is where this strategy's
    premise actually breaks, so the stop is widened to clear it — a stop above
    it gets taken out by movement the thesis explicitly allows for.
    """
    near = _with_indicators(bullish_pullback_snapshot, sma_50=Decimal("295.00"))
    signal = _evaluate(near, stop_pct=Decimal("0.01"))

    assert signal.strength is SignalStrength.ENTER
    assert signal.stop_price == Decimal("292.05")
    assert signal.metrics["stop_basis"] == "structure"


def test_the_minimum_stop_floor_binds_before_structure(bullish_pullback_snapshot):
    """A very tight stop is clamped up to the floor first.

    Sizing divides by the stop distance, so a 1% stop implies a hundred times
    the risk budget in notional. The floor bounds that at the source, before
    any structural anchor is considered.
    """
    near = _with_indicators(bullish_pullback_snapshot, sma_50=Decimal("300.00"))
    signal = _evaluate(near, stop_pct=Decimal("0.01"), min_stop_pct=Decimal("0.02"))

    # 2% under 302.25, not the 1% the parameter asked for.
    assert signal.stop_price == Decimal("296.20")
    assert signal.metrics["stop_distance_pct"] == pytest.approx(0.02, abs=1e-4)


def test_stop_already_below_sma50_is_left_alone(bullish_pullback_snapshot):
    """No widening when the flat stop is already clear of the average."""
    near = _with_indicators(bullish_pullback_snapshot, sma_50=Decimal("300.00"))
    signal = _evaluate(near, stop_pct=Decimal("0.05"))

    assert signal.strength is SignalStrength.ENTER
    assert signal.stop_price == Decimal("287.14")  # 5% under 302.25


# ------------------------------------------------------------------- exits


def test_holding_without_an_exit_condition_stays_put(
    bullish_pullback_snapshot, held_position
):
    calm = _with_indicators(bullish_pullback_snapshot, sma_50=Decimal("280.00"), rsi_14=55.0)
    signal = _evaluate(calm, position=held_position)

    assert signal.strength is SignalStrength.NONE
    assert "holding" in signal.reasons[0]


def test_losing_the_50day_triggers_an_exit(bullish_pullback_snapshot, held_position):
    signal = _evaluate(bullish_pullback_snapshot, position=held_position)

    assert signal.strength is SignalStrength.EXIT
    assert signal.side is not None and signal.side.value == "sell"
    assert any("thesis broken" in r for r in signal.reasons)


def test_overbought_rsi_triggers_an_exit(bullish_pullback_snapshot, held_position):
    hot = _with_indicators(
        bullish_pullback_snapshot, sma_50=Decimal("280.00"), rsi_14=75.0
    )
    signal = _evaluate(hot, position=held_position)

    assert signal.strength is SignalStrength.EXIT
    assert any("exit threshold" in r for r in signal.reasons)
