"""Shared fixtures.

Values mirror real AAPL data from 2026-08-12 so the tests exercise the same
shapes the system sees in production rather than round numbers that hide
rounding and precision bugs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.config import RiskConfig
from agentic_trader.market.earnings_capabilities import ROBINHOOD_MCP_EARNINGS
from agentic_trader.models import (
    AccountState,
    Bar,
    EarningsAssessment,
    EarningsEvent,
    EarningsStatus,
    Indicators,
    MarketSnapshot,
    Position,
    Side,
    Signal,
    SignalStrength,
)


def clear_earnings(
    symbol: str,
    *,
    as_of: date,
    report_date: date | None = None,
    timing: str | None = None,
    eps_estimate: Decimal | None = None,
    verified: bool = True,
) -> EarningsAssessment:
    """An authoritative earnings assessment, for tests that are not about earnings.

    Exists because the blackout gate now fails closed: a snapshot with no
    assessment blocks every entry, so a fixture that omits one would make every
    unrelated test fail for the wrong reason. Passing `report_date=None` models
    a symbol the broker resolved with nothing scheduled.
    """
    if report_date is None:
        return EarningsAssessment(
            symbol=symbol, status=EarningsStatus.NONE_SCHEDULED, as_of=as_of,
            source="get_earnings_results", profile_ref=ROBINHOOD_MCP_EARNINGS.profile_ref,
        )
    return EarningsAssessment(
        symbol=symbol, status=EarningsStatus.UPCOMING, as_of=as_of,
        source="get_earnings_results", profile_ref=ROBINHOOD_MCP_EARNINGS.profile_ref,
        event=EarningsEvent(
            symbol=symbol, report_date=report_date, timing=timing,
            eps_estimate=eps_estimate, verified=verified,
        ),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=Decimal("0.01"),
        max_position_pct=Decimal("0.25"),
        min_order_notional=Decimal("1.00"),
        max_order_notional=Decimal("25.00"),
        max_open_positions=2,
        max_daily_loss_pct=Decimal("0.03"),
        default_stop_pct=Decimal("0.05"),
        max_stop_pct=Decimal("0.12"),
        earnings_blackout_days=3,
        symbol_cooldown_days=2,
        min_avg_volume_30d=Decimal("500000"),
    )


# Obviously fake, so a real account number can never be mistaken for a fixture
# value — and so nothing identifying enters git history.
TEST_ACCOUNT = "TEST0000"


@pytest.fixture
def account() -> AccountState:
    return AccountState(
        account_number=TEST_ACCOUNT,
        is_cash_account=True,
        total_value=Decimal("100"),
        cash=Decimal("100"),
        buying_power=Decimal("100"),
        unsettled_funds=Decimal("0"),
        positions=[],
        open_order_symbols=[],
        realized_pnl_today=Decimal("0"),
    )


def make_bars(closes: list[str], start: datetime | None = None) -> list[Bar]:
    base = start or datetime(2026, 8, 1, tzinfo=UTC)
    return [
        Bar(
            begins_at=base.replace(day=min(1 + i, 28)),
            open=Decimal(c),
            high=Decimal(c) * Decimal("1.01"),
            low=Decimal(c) * Decimal("0.99"),
            close=Decimal(c),
            volume=40_000_000,
        )
        for i, c in enumerate(closes)
    ]


@pytest.fixture
def bullish_pullback_snapshot() -> MarketSnapshot:
    """AAPL as of 2026-08-12, with momentum edited to be improving.

    Every condition of `trend_pullback` passes here. Individual tests break one
    condition at a time to confirm each is load-bearing.
    """
    return MarketSnapshot(
        symbol="AAPL",
        captured_at=datetime(2026, 8, 13, 17, 44, tzinfo=UTC),
        last_price=Decimal("303.54"),
        previous_close=Decimal("302.25"),
        # Quote printed a second before we captured it, with a ~0.02% book —
        # roughly what AAPL actually quotes. Preflight now requires both: a
        # venue timestamp to age the price, and a usable book to bound spread.
        quote_as_of=datetime(2026, 8, 13, 17, 43, 59, tzinfo=UTC),
        bid=Decimal("303.51"),
        ask=Decimal("303.57"),
        book_as_of=datetime(2026, 8, 13, 17, 43, 59, tzinfo=UTC),
        bars=make_bars(["308.26", "304.91", "302.25"]),
        indicators=Indicators(
            as_of=datetime(2026, 8, 12, tzinfo=UTC),
            rsi_14=40.21,
            rsi_prev=39.0,
            macd=-1.73,
            macd_signal=1.61,
            macd_hist=-3.34,
            macd_hist_prev=-3.90,  # deeper before => improving now
            sma_20=Decimal("321.22"),
            sma_50=Decimal("309.48"),
            sma_200=Decimal("280.09"),
        ),
        earnings=clear_earnings(
            "AAPL",
            as_of=date(2026, 8, 13),
            report_date=date(2026, 10, 29),
            timing="pm",
            eps_estimate=Decimal("1.98"),
            verified=False,
        ),
        average_volume_30d=Decimal("54378256"),
        market_cap=Decimal("4451817942107"),
        high_52w=Decimal("344.57"),
        low_52w=Decimal("223.78"),
    )


@pytest.fixture
def entry_signal() -> Signal:
    return Signal(
        symbol="AAPL",
        strategy="trend_pullback",
        strength=SignalStrength.ENTER,
        side=Side.BUY,
        confidence=0.7,
        reference_price=Decimal("302.25"),
        stop_price=Decimal("287.14"),
        target_price=Decimal("332.47"),
        reasons=["trend intact", "pullback to SMA20", "momentum stabilizing"],
    )


@pytest.fixture
def held_position() -> Position:
    return Position(
        symbol="AAPL",
        quantity=Decimal("0.045810"),
        average_cost=Decimal("302.55"),
        market_value=Decimal("13.86"),
    )
