"""Stop construction from volatility, and market-wide regime classification.

Two properties matter most here and are tested from several directions:

- The stop is derived *before* the size, never fitted to one. Sizing divides by
  the stop distance, so anything that quietly changes the stop changes the
  position — the floor and ceiling exist to bound that, and the fallback is
  recorded rather than substituted silently.
- Market regime is recorded and never enforced. A test asserts the absence of
  gating, because the value of this data comes from having collected it across
  trades that were *allowed to happen*.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from agentic_trader.market.market_regime import (
    MarketContext,
    MarketRegime,
    classify_market_regime,
)
from agentic_trader.market.snapshot import parse_indicators
from agentic_trader.models import Indicators, MarketSnapshot
from agentic_trader.strategies.stops import StopBasis, build_stop

PRICE = Decimal("300.00")
NOW = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)


def _stop(**overrides):
    kwargs = {
        "atr": Decimal("6.00"),          # 2% of price
        "atr_multiple": Decimal("2.0"),  # -> 4% stop
        "min_stop_pct": Decimal("0.02"),
        "max_stop_pct": Decimal("0.12"),
        "flat_stop_pct": Decimal("0.05"),
        "structural_level": None,
    }
    kwargs.update(overrides)
    return build_stop(PRICE, **kwargs)


# ----------------------------------------------------------------- ATR basis


def test_the_stop_is_scaled_from_measured_volatility():
    plan = _stop()

    assert plan.basis is StopBasis.ATR
    assert plan.stop_price == Decimal("288.00")   # 300 - (6 x 2)
    assert plan.distance_pct == Decimal("0.04")
    assert plan.is_volatility_derived


def test_a_wider_multiple_gives_a_wider_stop_and_a_smaller_position():
    """The point of the change: volatility drives distance, distance drives size."""
    tight = _stop(atr_multiple=Decimal("1.0"))
    wide = _stop(atr_multiple=Decimal("3.0"))

    assert tight.distance_pct < wide.distance_pct
    # notional = risk_budget / distance, so a wider stop buys strictly less.
    budget = Decimal("1.00")
    assert budget / wide.distance_pct < budget / tight.distance_pct


@pytest.mark.parametrize("atr", [None, Decimal("0"), Decimal("-1")])
def test_unusable_atr_falls_back_to_a_flat_percentage(atr):
    """Degradation, not failure — but recorded as a weaker claim."""
    plan = _stop(atr=atr)

    assert plan.basis is StopBasis.FLAT_PCT
    assert not plan.is_volatility_derived
    assert plan.stop_price == Decimal("285.00")   # 5% of 300
    assert any("no ATR" in n for n in plan.notes)


# --------------------------------------------------------------------- bounds


def test_a_tiny_stop_is_widened_to_the_floor():
    """Sizing divides by this number; 0.1% would imply 1000x the risk budget."""
    plan = _stop(atr=Decimal("0.15"))   # 0.05% x 2 = 0.1%

    assert plan.clamped_at_min
    assert plan.distance_pct == Decimal("0.02")
    assert plan.stop_price == Decimal("294.00")


def test_volatility_beyond_the_ceiling_marks_the_setup_untradeable():
    """Not clamped-and-carry-on.

    A stop at the ceiling when ATR says the stock travels further is inside
    ordinary daily movement — the position would be sized as though risk were
    bounded when it is not. The flag is what makes the strategy decline.
    """
    plan = _stop(atr=Decimal("30.00"))   # 10% x 2 = 20%, ceiling is 12%

    assert plan.clamped_at_max
    assert plan.too_volatile_to_trade
    assert plan.distance_pct == Decimal("0.12")
    assert any("beyond the" in n for n in plan.notes)


def test_a_normal_stop_is_not_flagged():
    assert not _stop().too_volatile_to_trade
    assert not _stop().clamped_at_min


# ------------------------------------------------------------------ structure


def test_structure_below_the_volatility_stop_widens_it():
    plan = _stop(structural_level=Decimal("280.00"))

    assert plan.basis is StopBasis.STRUCTURE
    assert plan.stop_price == Decimal("280.00")
    assert plan.distance_pct > Decimal("0.04")


def test_structure_above_the_volatility_stop_is_ignored():
    """Structure may only widen.

    Tightening would place the stop where the thesis does not support it.
    """
    plan = _stop(structural_level=Decimal("295.00"))

    assert plan.basis is StopBasis.ATR
    assert plan.stop_price == Decimal("288.00")


def test_a_structural_stop_may_exceed_the_ceiling():
    """The max_stop_pct gate in risk.limits is what refuses it.

    Quietly pulling it back to the ceiling would claim a risk boundary the
    structure does not actually provide.
    """
    plan = _stop(structural_level=Decimal("250.00"))   # ~16.7%

    assert plan.stop_price == Decimal("250.00")
    assert plan.distance_pct > Decimal("0.12")
    assert not plan.too_volatile_to_trade   # risk.limits rejects, not this


def test_a_nonsense_price_raises():
    with pytest.raises(ValueError, match="cannot build a stop"):
        build_stop(
            Decimal("0"), atr=Decimal("1"), atr_multiple=Decimal("2"),
            min_stop_pct=Decimal("0.02"), max_stop_pct=Decimal("0.12"),
            flat_stop_pct=Decimal("0.05"),
        )


# -------------------------------------------------------------- market regime


def _index(symbol: str, price: str, sma_50: str, sma_200: str, atr: str | None = None):
    return MarketSnapshot(
        symbol=symbol,
        captured_at=NOW,
        last_price=Decimal(price),
        quote_as_of=NOW,
        indicators=Indicators(
            sma_50=Decimal(sma_50),
            sma_200=Decimal(sma_200),
            atr_14=Decimal(atr) if atr else None,
        ),
    )


BULL_SPY = _index("SPY", "600", "580", "540")
BEAR_SPY = _index("SPY", "500", "520", "560")


def test_a_rising_index_above_its_averages_is_a_bull_trend():
    ctx = classify_market_regime(BULL_SPY)

    assert ctx.regime is MarketRegime.BULL_TREND
    assert ctx.is_known
    assert ctx.inputs["spy_price"] == 600.0


def test_a_falling_index_below_its_averages_is_a_bear_trend():
    assert classify_market_regime(BEAR_SPY).regime is MarketRegime.BEAR_TREND


def test_disagreeing_averages_are_a_range():
    mixed = _index("SPY", "600", "520", "560")   # above 200, but 50 under 200
    assert classify_market_regime(mixed).regime is MarketRegime.RANGE


def test_spy_and_qqq_disagreeing_downgrades_a_trend_to_range():
    """A split market is not a trend — the move is sectoral, not market-wide."""
    ctx = classify_market_regime(BULL_SPY, _index("QQQ", "400", "420", "450"))

    assert ctx.regime is MarketRegime.RANGE
    assert any("split market" in r for r in ctx.reasons)


def test_agreeing_indices_keep_the_trend():
    ctx = classify_market_regime(BULL_SPY, _index("QQQ", "500", "480", "440"))
    assert ctx.regime is MarketRegime.BULL_TREND


def test_high_volatility_outranks_direction():
    """A violently moving market is its own regime, whichever way it points."""
    wild = _index("SPY", "600", "580", "540", atr="18")   # 3% of price

    ctx = classify_market_regime(wild)

    assert ctx.regime is MarketRegime.HIGH_VOLATILITY
    assert ctx.inputs["spy_atr_pct"] == pytest.approx(0.03)


def test_volatility_in_qqq_alone_still_flags_the_market():
    ctx = classify_market_regime(BULL_SPY, _index("QQQ", "400", "380", "350", atr="20"))
    assert ctx.regime is MarketRegime.HIGH_VOLATILITY


def test_missing_averages_are_unknown_not_assumed_calm():
    bare = MarketSnapshot(
        symbol="SPY", captured_at=NOW, last_price=Decimal("600"), quote_as_of=NOW
    )
    assert classify_market_regime(bare).regime is MarketRegime.UNKNOWN


def test_no_index_snapshot_is_unknown():
    ctx = classify_market_regime(None)
    assert ctx.regime is MarketRegime.UNKNOWN
    assert not ctx.is_known


def test_the_context_records_what_produced_it():
    """A label with no inputs cannot be re-examined, which is the whole point."""
    ctx = classify_market_regime(BULL_SPY)

    assert ctx.reasons
    assert ctx.as_of == NOW
    assert set(ctx.to_dict()) == {"regime", "reasons", "inputs", "as_of"}


def test_regime_gates_nothing():
    """Guards the deliberate absence of a rule.

    Expectancy per regime does not exist yet. Wiring a guess into a gate would
    suppress exactly the trades needed to find out whether the guess was right,
    so MarketContext must stay inert until the evidence is in.
    """
    assert not hasattr(MarketContext, "allows_entry")
    assert not hasattr(MarketRegime, "allows_long_entry")


# ------------------------------------------------- indicator parsing / staleness


def _series(kind: str, when: str, **values):
    return {
        "data": {
            "indicators": [
                {"type": kind, "series": [{"begins_at": when, **values}]}
            ]
        }
    }


def test_atr_is_parsed_in_dollars_per_share():
    ind = parse_indicators({"atr": _series("atr", "2026-08-17T00:00:00Z", value=7.63)})

    assert ind.atr_14 == Decimal("7.63")


def test_as_of_reports_the_oldest_indicator_not_the_newest():
    """Staleness is set by the stalest input being relied on.

    Last-writer-wins would let one fresh series mask a stale one — a same-day
    ATR hiding an RSI computed a week earlier — and the critic blocks on this
    value. The conservative reading is the only safe one.
    """
    ind = parse_indicators(
        {
            "rsi": _series("rsi", "2026-08-11T00:00:00Z", value=40.0),
            "atr": _series("atr", "2026-08-17T00:00:00Z", value=7.63),
        }
    )

    assert ind.as_of == datetime(2026, 8, 11, tzinfo=UTC)


def test_as_of_is_none_when_nothing_carries_a_timestamp():
    assert parse_indicators({}).as_of is None
