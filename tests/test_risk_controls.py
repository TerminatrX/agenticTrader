"""Tests for the controls added on top of the original gate battery.

Each control gets a test proving it *blocks*, and where a control has an
intentional exception (the kill switch permitting exits, sector caps degrading
to a warning on unknown data) that exception is tested too — an exception
nobody tests is an exception nobody knows is there.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from agentic_trader.agents.critic import CriticReport, CriticVerdict
from agentic_trader.config import (
    ConfigError,
    RiskConfig,
    risk_fingerprint,
    verify_risk_lock,
    write_risk_lock,
)
from agentic_trader.models import Position, Side, Signal, SignalStrength
from agentic_trader.risk.limits import check_limits
from agentic_trader.risk.sizing import size_position

TODAY = date(2026, 8, 13)
SECTOR = "Electronic Technology"


def _signal(entry="302.25", stop="287.14", target="332.47") -> Signal:
    return Signal(
        symbol="AAPL",
        strategy="trend_pullback",
        strength=SignalStrength.ENTER,
        side=Side.BUY,
        confidence=0.7,
        reference_price=Decimal(entry),
        stop_price=Decimal(stop),
        target_price=Decimal(target) if target else None,
        reasons=["test"],
    )


# ------------------------------------------------------------ risk / reward


def test_risk_reward_ratio_computed_from_levels():
    # (332.47 - 302.25) / (302.25 - 287.14) = 30.22 / 15.11 = 2.0
    assert _signal().risk_reward_ratio == Decimal("2")


def test_reward_below_minimum_is_blocked(bullish_pullback_snapshot, account, risk_config):
    poor = _signal(target="310.00")  # ~0.51R
    result = check_limits(poor, bullish_pullback_snapshot, account, risk_config, as_of=TODAY)

    assert not result.passed
    assert any("reward-to-risk" in b and "below minimum" in b for b in result.breaches)


def test_missing_target_is_blocked(bullish_pullback_snapshot, account, risk_config):
    """Unknown reward is not acceptable reward."""
    result = check_limits(
        _signal(target=None), bullish_pullback_snapshot, account, risk_config, as_of=TODAY
    )
    assert not result.passed
    assert any("no target" in b for b in result.breaches)


def test_exactly_at_minimum_passes(bullish_pullback_snapshot, account, risk_config):
    """trend_pullback builds targets at exactly 2R, so the boundary must pass.

    If this ever fails, the gate has silently disabled the only live strategy.
    """
    result = check_limits(_signal(), bullish_pullback_snapshot, account, risk_config, as_of=TODAY)
    assert result.passed, result.breaches


# ---------------------------------------------------------- sector exposure


def _snapshot_with_sector(snapshot, sector=SECTOR):
    return snapshot.model_copy(update={"sector": sector})


def test_sector_at_cap_blocks_entry(bullish_pullback_snapshot, account, risk_config):
    # 25% of a $100 account is $25; a $25 same-sector holding fills it.
    held = account.model_copy(
        update={
            "positions": [
                Position(
                    symbol="MSFT", quantity=Decimal("1"),
                    average_cost=Decimal("25"), market_value=Decimal("25"), sector=SECTOR,
                )
            ]
        }
    )
    result = check_limits(
        _signal(), _snapshot_with_sector(bullish_pullback_snapshot), held, risk_config, as_of=TODAY
    )

    assert not result.passed
    assert any("sector" in b and "cap" in b for b in result.breaches)


def test_different_sector_does_not_count(bullish_pullback_snapshot, account, risk_config):
    held = account.model_copy(
        update={
            "positions": [
                Position(
                    symbol="XOM", quantity=Decimal("1"),
                    average_cost=Decimal("25"), market_value=Decimal("25"), sector="Energy",
                )
            ]
        }
    )
    result = check_limits(
        _signal(), _snapshot_with_sector(bullish_pullback_snapshot), held, risk_config, as_of=TODAY
    )
    assert result.passed, result.breaches


def test_unknown_sector_warns_rather_than_blocks(
    bullish_pullback_snapshot, account, risk_config
):
    """Unknown data must not silently pass as 'diversified' without saying so."""
    result = check_limits(
        _signal(), bullish_pullback_snapshot, account, risk_config, as_of=TODAY
    )
    assert result.passed
    assert any("sector unknown for AAPL" in w for w in result.warnings)


def test_unknown_sector_on_a_holding_is_surfaced(
    bullish_pullback_snapshot, account, risk_config
):
    held = account.model_copy(
        update={
            "positions": [
                Position(symbol="MSFT", quantity=Decimal("1"), average_cost=Decimal("5"))
            ]
        }
    )
    result = check_limits(
        _signal(), _snapshot_with_sector(bullish_pullback_snapshot), held, risk_config, as_of=TODAY
    )
    assert any("under-counted" in w for w in result.warnings)


def test_sizing_caps_at_remaining_sector_headroom(account, risk_config):
    """The gate only warns; sizing is what actually enforces the ceiling."""
    held = account.model_copy(
        update={
            "positions": [
                Position(
                    symbol="MSFT", quantity=Decimal("1"),
                    average_cost=Decimal("20"), market_value=Decimal("20"), sector=SECTOR,
                )
            ]
        }
    )
    # $25 cap minus $20 held leaves $5 of headroom.
    result = size_position(
        _signal("100.00", "95.00"), held, risk_config, scale_by_confidence=False, sector=SECTOR
    )

    assert result.approved
    assert result.notional == Decimal("5.00")
    assert result.binding_constraint == f"sector_exposure[{SECTOR}]"


# ---------------------------------------------------------------- kill switch


def test_kill_switch_blocks_entry_and_trips(
    bullish_pullback_snapshot, account, risk_config
):
    # 6% of $100 is $6.
    wrecked = account.model_copy(update={"realized_pnl_today": Decimal("-6.50")})
    result = check_limits(
        _signal(), bullish_pullback_snapshot, wrecked, risk_config, as_of=TODAY
    )

    assert not result.passed
    assert result.trip_kill_switch
    assert any("KILL SWITCH" in b for b in result.breaches)


def test_kill_switch_still_allows_exits(
    bullish_pullback_snapshot, account, risk_config, held_position
):
    """An automatic control that strands open positions is worse than the loss."""
    wrecked = account.model_copy(
        update={"realized_pnl_today": Decimal("-6.50"), "positions": [held_position]}
    )
    exit_signal = Signal(
        symbol="AAPL", strategy="trend_pullback", strength=SignalStrength.EXIT,
        side=Side.SELL, reference_price=Decimal("302.25"), reasons=["thesis broken"],
    )
    result = check_limits(
        exit_signal, bullish_pullback_snapshot, wrecked, risk_config, as_of=TODAY
    )

    assert result.passed, result.breaches
    assert result.trip_kill_switch  # still flagged, so HALT is written
    assert any("allowing this exit" in w for w in result.warnings)


def test_kill_switch_must_exceed_daily_loss_limit():
    with pytest.raises(ValueError, match="must exceed max_daily_loss_pct"):
        RiskConfig(max_daily_loss_pct=Decimal("0.05"), kill_switch_daily_loss_pct=Decimal("0.03"))


def test_position_cap_above_sector_cap_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        RiskConfig(
            max_position_pct=Decimal("0.40"), max_sector_exposure_pct=Decimal("0.25")
        )


# ------------------------------------------------------- critic: veto only


def test_critic_adjustment_can_only_lower_confidence():
    report = CriticReport()
    report.penalize(0.2)

    assert report.confidence_adjustment == -0.2
    assert report.adjusted_confidence(0.7) == pytest.approx(0.5)


def test_penalize_ignores_sign_so_it_cannot_raise():
    """A positive magnitude must still reduce — the model may never amplify."""
    report = CriticReport()
    report.penalize(0.3)
    report.penalize(-0.1)  # sign is stripped

    assert report.confidence_adjustment == pytest.approx(-0.4)
    assert report.adjusted_confidence(0.5) <= 0.5


def test_adjusted_confidence_never_exceeds_original_and_floors_at_zero():
    generous = CriticReport(confidence_adjustment=0.5)  # hand-set, hostile input
    assert generous.adjusted_confidence(0.4) == 0.4  # clamped, not raised

    harsh = CriticReport()
    harsh.penalize(5.0)
    assert harsh.adjusted_confidence(0.4) == 0.0


def test_concern_can_carry_a_penalty():
    report = CriticReport()
    report.concern("earnings in 4d", 0.15)

    assert report.verdict is CriticVerdict.CONCERN
    assert report.confidence_adjustment == pytest.approx(-0.15)


def test_lower_confidence_produces_a_smaller_order(account, risk_config):
    full = size_position(_signal("100.00", "95.00"), account, risk_config)
    reduced = size_position(
        _signal("100.00", "95.00"), account, risk_config, confidence_override=0.4
    )
    assert reduced.notional < full.notional


# ------------------------------------------------------------ risk.yaml lock


def test_fingerprint_ignores_formatting_but_tracks_values():
    a = RiskConfig()
    b = RiskConfig()
    assert risk_fingerprint(a) == risk_fingerprint(b)

    changed = RiskConfig(max_position_pct=Decimal("0.20"))
    assert risk_fingerprint(changed) != risk_fingerprint(a)


def test_lock_accepts_matching_config(tmp_path):
    config = RiskConfig()
    lock = tmp_path / "risk.lock"
    write_risk_lock(config, lock)

    verify_risk_lock(config, lock)  # must not raise


def test_lock_rejects_a_loosened_limit_and_names_it(tmp_path):
    """Quadrupling per-trade risk is exactly what the lock exists to catch."""
    lock = tmp_path / "risk.lock"
    write_risk_lock(RiskConfig(risk_per_trade_pct=Decimal("0.01")), lock)

    loosened = RiskConfig(risk_per_trade_pct=Decimal("0.04"))
    with pytest.raises(ConfigError) as exc:
        verify_risk_lock(loosened, lock)

    message = str(exc.value)
    assert "risk_per_trade_pct" in message
    assert "0.01 -> 0.04" in message
    assert "lock-risk --confirm" in message


def test_missing_lock_is_not_an_error(tmp_path):
    """Absence means no baseline yet; only a mismatch is fatal."""
    verify_risk_lock(RiskConfig(), tmp_path / "absent.lock")


def test_corrupt_lock_is_an_error(tmp_path):
    lock = tmp_path / "risk.lock"
    lock.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="unreadable"):
        verify_risk_lock(RiskConfig(), lock)


def test_lock_file_records_values_for_diffing(tmp_path):
    lock = tmp_path / "risk.lock"
    write_risk_lock(RiskConfig(), lock)
    payload = json.loads(lock.read_text(encoding="utf-8"))

    assert payload["algorithm"] == "sha256"
    assert "max_daily_loss_pct" in payload["values"]
