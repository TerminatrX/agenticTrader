"""Risk gate tests.

Each gate gets a test proving it blocks, because a limit that silently fails
open is worse than no limit — it produces false confidence.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from agentic_trader.models import EarningsEvent, Position, Side, Signal, SignalStrength
from agentic_trader.risk.limits import check_limits

TODAY = date(2026, 8, 13)


def _exit_signal() -> Signal:
    return Signal(
        symbol="AAPL",
        strategy="trend_pullback",
        strength=SignalStrength.EXIT,
        side=Side.SELL,
        reference_price=Decimal("302.25"),
        reasons=["thesis broken"],
    )


def test_clean_entry_passes(entry_signal, bullish_pullback_snapshot, account, risk_config):
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, account, risk_config, as_of=TODAY
    )
    assert result.passed, result.breaches


def test_halt_flag_blocks_everything_immediately(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, account, risk_config,
        is_halted=True, as_of=TODAY,
    )
    assert not result.passed
    assert "HALT" in result.breaches[0]
    # Short-circuits — nothing else is evaluated once trading is disabled.
    assert len(result.breaches) == 1


def test_earnings_blackout_blocks_entry(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    imminent = bullish_pullback_snapshot.model_copy(
        update={"earnings": EarningsEvent(report_date=date(2026, 8, 15), verified=True)}
    )
    result = check_limits(entry_signal, imminent, account, risk_config, as_of=TODAY)

    assert not result.passed
    assert any("earnings in 2d" in b for b in result.breaches)


def test_earnings_just_outside_blackout_warns_but_allows(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    result = check_limits(
        entry_signal,
        bullish_pullback_snapshot.model_copy(
            update={"earnings": EarningsEvent(report_date=date(2026, 8, 20), verified=True)}
        ),
        account, risk_config, as_of=TODAY,
    )
    assert result.passed
    assert any("earnings in 7d" in w for w in result.warnings)


def test_daily_loss_limit_blocks_new_entries(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    bleeding = account.model_copy(update={"realized_pnl_today": Decimal("-3.50")})
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, bleeding, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("daily loss" in b for b in result.breaches)


def test_max_open_positions_blocks(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    full = account.model_copy(
        update={
            "positions": [
                Position(symbol="MSFT", quantity=Decimal("1"), average_cost=Decimal("10")),
                Position(symbol="NVDA", quantity=Decimal("1"), average_cost=Decimal("10")),
            ]
        }
    )
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, full, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("already holding 2 positions" in b for b in result.breaches)


def test_existing_position_blocks_a_second_entry(
    entry_signal, bullish_pullback_snapshot, account, risk_config, held_position
):
    holding = account.model_copy(update={"positions": [held_position]})
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, holding, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("already holding AAPL" in b for b in result.breaches)


def test_open_order_blocks_duplicate_submission(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    pending = account.model_copy(update={"open_order_symbols": ["AAPL"]})
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, pending, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("already open" in b for b in result.breaches)


def test_cooldown_blocks_re_entry_after_a_loss(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, account, risk_config,
        as_of=TODAY, last_loss_exit=date(2026, 8, 12),
    )
    assert not result.passed
    assert any("cooldown" in b for b in result.breaches)


def test_cooldown_expires(entry_signal, bullish_pullback_snapshot, account, risk_config):
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, account, risk_config,
        as_of=TODAY, last_loss_exit=date(2026, 8, 1),
    )
    assert result.passed, result.breaches


def test_illiquid_symbol_blocked(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    thin = bullish_pullback_snapshot.model_copy(
        update={"average_volume_30d": Decimal("1000")}
    )
    result = check_limits(entry_signal, thin, account, risk_config, as_of=TODAY)

    assert not result.passed
    assert any("average volume" in b for b in result.breaches)


def test_stop_wider_than_max_is_blocked(bullish_pullback_snapshot, account, risk_config):
    loose = Signal(
        symbol="AAPL", strategy="trend_pullback", strength=SignalStrength.ENTER,
        side=Side.BUY, reference_price=Decimal("300.00"),
        stop_price=Decimal("250.00"),  # ~16.7%, above the 12% cap
        reasons=["test"],
    )
    result = check_limits(loose, bullish_pullback_snapshot, account, risk_config, as_of=TODAY)

    assert not result.passed
    assert any("exceeds maximum" in b for b in result.breaches)


def test_unsettled_funds_produce_an_explanatory_note(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """A cash account's shrunken buying power must have a stated cause."""
    settling = account.model_copy(
        update={"unsettled_funds": Decimal("40.00"), "buying_power": Decimal("60.00")}
    )
    result = check_limits(
        entry_signal, bullish_pullback_snapshot, settling, risk_config, as_of=TODAY
    )

    assert result.passed
    assert any("good-faith violation" in n for n in result.notes)


def test_exit_is_not_blocked_by_entry_gates(
    bullish_pullback_snapshot, risk_config, account, held_position
):
    """Blocking an exit increases risk; entry gates must not apply to sells."""
    holding = account.model_copy(
        update={
            "positions": [held_position],
            "realized_pnl_today": Decimal("-50.00"),  # would block any entry
        }
    )
    result = check_limits(
        _exit_signal(), bullish_pullback_snapshot, holding, risk_config, as_of=TODAY
    )
    assert result.passed, result.breaches


def test_exit_without_a_position_is_blocked(
    bullish_pullback_snapshot, account, risk_config
):
    result = check_limits(
        _exit_signal(), bullish_pullback_snapshot, account, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("no open position" in b for b in result.breaches)
