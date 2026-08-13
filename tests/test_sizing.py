"""Sizing tests.

The property that matters most: dollars actually at risk never exceed the
configured budget, whatever the stop width or account size. Everything else is
secondary to that invariant.
"""

from __future__ import annotations

from decimal import Decimal

from agentic_trader.models import Side, Signal, SignalStrength
from agentic_trader.risk.sizing import exit_notional, size_position


def _signal(entry: str, stop: str, confidence: float = 1.0) -> Signal:
    return Signal(
        symbol="AAPL",
        strategy="trend_pullback",
        strength=SignalStrength.ENTER,
        side=Side.BUY,
        confidence=confidence,
        reference_price=Decimal(entry),
        stop_price=Decimal(stop),
        reasons=["test"],
    )


def test_notional_derives_from_risk_budget_and_stop(account, risk_config):
    # $100 account, 1% risk = $1 budget. A 5% stop implies a $20 position.
    result = size_position(_signal("100.00", "95.00"), account, risk_config)

    assert result.approved
    assert result.risk_budget == Decimal("1.00")
    assert result.notional == Decimal("20.00")


def test_wider_stop_produces_smaller_position(account, risk_config):
    """The stop, not conviction, decides the size."""
    tight = size_position(_signal("100.00", "95.00"), account, risk_config)
    wide = size_position(_signal("100.00", "90.00"), account, risk_config)

    assert tight.notional == Decimal("20.00")
    assert wide.notional == Decimal("10.00")


def test_dollars_at_risk_never_exceed_budget(account, risk_config):
    """The core invariant, across a range of stop widths."""
    budget = account.total_value * risk_config.risk_per_trade_pct

    for stop_pct in ("0.02", "0.05", "0.08", "0.11"):
        entry = Decimal("300.00")
        stop = entry * (Decimal("1") - Decimal(stop_pct))
        result = size_position(_signal(str(entry), str(stop)), account, risk_config)
        if not result.approved:
            continue
        at_risk = result.notional * ((entry - stop) / entry)
        assert at_risk <= budget * Decimal("1.01"), f"stop {stop_pct} risked {at_risk}"


def test_position_ceiling_caps_a_tight_stop(account, risk_config):
    # A 1% stop implies a $100 position, above the 25% ceiling.
    result = size_position(_signal("100.00", "99.00"), account, risk_config)

    assert result.approved
    assert result.notional == Decimal("25.00")
    assert result.binding_constraint == "max_position_pct"
    assert any("max_position_pct" in c for c in result.caps_applied)


def test_confidence_scales_down_but_never_up(account, risk_config):
    full = size_position(_signal("100.00", "95.00", confidence=1.0), account, risk_config)
    half = size_position(_signal("100.00", "95.00", confidence=0.5), account, risk_config)

    assert full.notional == Decimal("20.00")
    assert half.notional == Decimal("10.00")
    assert half.notional < full.notional


def test_buying_power_is_a_hard_wall(account, risk_config):
    poor = account.model_copy(update={"buying_power": Decimal("5.00")})
    result = size_position(_signal("100.00", "95.00"), poor, risk_config)

    assert result.notional <= Decimal("5.00")
    assert result.binding_constraint == "buying_power"


def test_rejects_when_below_minimum_notional(account, risk_config):
    tiny = account.model_copy(
        update={"total_value": Decimal("2.00"), "buying_power": Decimal("2.00")}
    )
    result = size_position(_signal("100.00", "95.00"), tiny, risk_config)

    assert not result.approved
    assert "below minimum" in result.rejection_reason


def test_rejects_stop_at_or_above_entry(account, risk_config):
    result = size_position(_signal("100.00", "100.00"), account, risk_config)

    assert not result.approved
    assert "valid stop" in result.rejection_reason


def test_fractional_quantity_is_usable_on_a_small_account(account, risk_config):
    """A $100 account must still be able to take a position in a $300 stock."""
    result = size_position(_signal("302.25", "287.14"), account, risk_config)

    assert result.approved
    quantity = result.notional / Decimal("302.25")
    assert 0 < quantity < 1  # fractional, and non-zero


def test_exit_notional_uses_full_position(account, held_position):
    holding = account.model_copy(update={"positions": [held_position]})

    assert exit_notional(holding, "AAPL", Decimal("310.00")) == Decimal("14.20")
    assert exit_notional(holding, "MSFT", Decimal("310.00")) == Decimal("0")
