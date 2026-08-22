"""End-to-end tests: snapshot parsing, idempotency, and the safety invariants.

The tests that matter most here are the negative ones. A trading system's worst
failure is not a missed opportunity — it is an order placed twice, or an order
placed at all when something upstream was broken.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from agentic_trader.agents.orchestrator import run_cycle
from agentic_trader.config import AppConfig, StrategyConfig
from agentic_trader.execution.executor import PreflightError, build_order_payload
from agentic_trader.execution.shadow_executor import ShadowExecutor
from agentic_trader.journal import JournalRepository
from agentic_trader.journal.models import CycleOutcome, TradeRecord
from agentic_trader.market.snapshot import SnapshotError, build_snapshot
from agentic_trader.models import EarningsStatus, Side
from agentic_trader.risk.engine import RiskEngine, build_client_key

# Obviously fake — see the note in conftest.py.
TEST_ACCOUNT = "TEST0000"

QUOTE = {
    "data": {
        "results": [
            {
                "quote": {
                    "symbol": "AAPL",
                    "last_trade_price": "303.540000",
                    "adjusted_previous_close": "302.250000",
                    "has_traded": True,
                    "state": "active",
                },
                "close": {"symbol": "AAPL", "price": "302.25"},
            }
        ]
    }
}

HISTORICALS = {
    "data": {
        "results": [
            {
                "symbol": "AAPL",
                "bars": [
                    {
                        "begins_at": "2026-08-11T00:00:00Z",
                        "open_price": "307.75", "close_price": "304.91",
                        "high_price": "309.97", "low_price": "302.79", "volume": 37476746,
                    },
                    {
                        "begins_at": "2026-08-12T00:00:00Z",
                        "open_price": "305.10", "close_price": "302.25",
                        "high_price": "305.66", "low_price": "300.57", "volume": 41657768,
                    },
                ],
            }
        ]
    }
}


def _indicator(kind: str, series: list[dict]) -> dict:
    return {"data": {"symbol": "AAPL", "indicators": [{"type": kind, "series": series}]}}


# ---------------------------------------------------------------- parsing


def test_snapshot_parses_broker_payloads():
    snapshot = build_snapshot(
        "aapl",  # lowercase input must normalize
        quote=QUOTE,
        historicals=HISTORICALS,
        indicators={
            "rsi": _indicator("rsi", [{"begins_at": "2026-08-11T00:00:00Z", "value": 41.9},
                                      {"begins_at": "2026-08-12T00:00:00Z", "value": 40.2}]),
            "sma_20": _indicator("sma", [{"begins_at": "2026-08-12T00:00:00Z", "value": 321.22}]),
        },
    )

    assert snapshot.symbol == "AAPL"
    assert snapshot.last_price == Decimal("303.540000")
    assert len(snapshot.bars) == 2
    assert snapshot.indicators.rsi_14 == 40.2
    assert snapshot.indicators.rsi_prev == 41.9
    assert snapshot.indicators.sma_20 == Decimal("321.22")
    # Indicators are computed through the last completed bar, so decisions
    # anchor on that close rather than the live tick.
    assert snapshot.reference_price == Decimal("302.25")


def test_interpolated_bars_are_dropped():
    payload = {
        "data": {"results": [{"symbol": "AAPL", "bars": [
            {"begins_at": "2026-08-11T00:00:00Z", "open_price": "1", "close_price": "1",
             "high_price": "1", "low_price": "1", "volume": 1, "interpolated": True},
            {"begins_at": "2026-08-12T00:00:00Z", "open_price": "2", "close_price": "2",
             "high_price": "2", "low_price": "2", "volume": 1},
        ]}]}
    }
    snapshot = build_snapshot("AAPL", quote=QUOTE, historicals=payload)

    assert len(snapshot.bars) == 1
    assert snapshot.bars[0].close == Decimal("2")


def test_malformed_price_raises_rather_than_defaulting():
    bad = {"data": {"results": [{"symbol": "AAPL", "bars": [
        {"begins_at": "2026-08-12T00:00:00Z", "open_price": "n/a", "close_price": "1",
         "high_price": "1", "low_price": "1", "volume": 1},
    ]}]}}
    with pytest.raises(SnapshotError):
        build_snapshot("AAPL", quote=QUOTE, historicals=bad)


def test_earnings_rows_without_symbol_identity_are_not_trusted():
    """The old parser read these rows happily. It never checked whose they were.

    A row with no `symbol` cannot be shown to be about the symbol under
    evaluation, so the assessment is UNKNOWN rather than a confident date.
    """
    payload = {"data": {"results": [
        {"eps": {"actual": "2.02"}, "report": {"date": "2026-07-30", "verified": True}},
        {"eps": {"actual": None}, "report": {"date": "2026-10-29", "verified": False}},
    ]}}
    snapshot = build_snapshot("AAPL", quote=QUOTE, earnings=payload)

    assert snapshot.earnings.status is EarningsStatus.UNKNOWN
    assert "absent from earnings response" in snapshot.earnings.reason


def test_next_earnings_is_the_nearest_future_dated_row_for_this_symbol():
    payload = {"data": {"results": [
        {"symbol": "AAPL", "eps": {"actual": "2.02"},
         "report": {"date": "2026-07-30", "timing": "pm", "verified": True}},
        {"symbol": "AAPL", "eps": {"actual": None},
         "report": {"date": "2026-10-29", "timing": "pm", "verified": False}},
    ]}}
    snapshot = build_snapshot("AAPL", quote=QUOTE, earnings=payload)

    assert snapshot.earnings.status is EarningsStatus.UPCOMING
    assert snapshot.earnings.event.symbol == "AAPL"
    assert snapshot.earnings.event.report_date == date(2026, 10, 29)
    assert snapshot.earnings.event.verified is False


def test_missing_quote_falls_back_to_last_bar():
    snapshot = build_snapshot("AAPL", historicals=HISTORICALS)

    assert snapshot.last_price == Decimal("302.25")
    assert "no live quote" in snapshot.staleness_note


# ------------------------------------------------------------ idempotency


def test_client_key_is_stable_and_a_valid_uuid():
    args = (TEST_ACCOUNT, "AAPL", Side.BUY, "trend_pullback", date(2026, 8, 13))
    first, second = build_client_key(*args), build_client_key(*args)

    assert first == second
    assert len(first) == 36 and first.count("-") == 4


def test_client_key_differs_across_side_symbol_and_day():
    day = date(2026, 8, 13)
    keys = {
        build_client_key(TEST_ACCOUNT, "AAPL", Side.BUY, "trend_pullback", day),
        build_client_key(TEST_ACCOUNT, "MSFT", Side.BUY, "trend_pullback", day),
        build_client_key(TEST_ACCOUNT, "AAPL", Side.SELL, "trend_pullback", day),
        build_client_key(TEST_ACCOUNT, "AAPL", Side.BUY, "trend_pullback", date(2026, 8, 14)),
    }
    assert len(keys) == 4


def test_duplicate_client_key_is_refused_by_the_journal(tmp_path):
    repo = JournalRepository(tmp_path / "j.db")
    trade = TradeRecord(
        client_key="dup", symbol="AAPL", strategy="trend_pullback", mode="shadow",
        opened_at=datetime.now(UTC), entry_price=Decimal("300"),
        quantity=Decimal("0.05"), notional=Decimal("15"),
    )
    repo.record_trade(trade)

    with pytest.raises(ValueError, match="already recorded"):
        repo.record_trade(trade)


def test_preflight_blocks_a_known_client_key(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account, as_of=date(2026, 8, 13)
    )
    assert decision.is_executable

    with pytest.raises(PreflightError, match="already submitted"):
        build_order_payload(
            decision, bullish_pullback_snapshot, account.account_number,
            known_client_keys={decision.intent.client_key},
            now=bullish_pullback_snapshot.captured_at,
        )


# ------------------------------------------------------------- safety gates


def test_stale_snapshot_blocks_order_construction(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account, as_of=date(2026, 8, 13)
    )
    with pytest.raises(PreflightError, match="old"):
        build_order_payload(
            decision, bullish_pullback_snapshot, account.account_number,
            now=bullish_pullback_snapshot.captured_at + timedelta(minutes=30),
        )


def test_price_drift_beyond_tolerance_blocks(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account, as_of=date(2026, 8, 13)
    )
    gapped = bullish_pullback_snapshot.model_copy(update={"last_price": Decimal("330.00")})

    with pytest.raises(PreflightError, match="price moved"):
        build_order_payload(
            decision, gapped, account.account_number,
            now=bullish_pullback_snapshot.captured_at,
        )


def test_order_payload_matches_broker_constraints(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Dollar-denominated orders must be market + regular hours, per the API."""
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account, as_of=date(2026, 8, 13)
    )
    plan = build_order_payload(
        decision, bullish_pullback_snapshot, account.account_number,
        now=bullish_pullback_snapshot.captured_at,
    )

    assert plan.payload["type"] == "market"
    assert plan.payload["market_hours"] == "regular_hours"
    assert "dollar_amount" in plan.payload
    assert "limit_price" not in plan.payload  # fractional limits are rejected
    assert plan.payload["ref_id"] == decision.intent.client_key
    assert any("MANAGED" in w for w in plan.warnings)


def test_shadow_slippage_always_works_against_the_trade(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account, as_of=date(2026, 8, 13)
    )
    plan = build_order_payload(
        decision, bullish_pullback_snapshot, account.account_number,
        now=bullish_pullback_snapshot.captured_at,
    )
    fill = ShadowExecutor(Decimal("0.001")).submit(plan)

    assert fill.fill_price > decision.intent.reference_price  # buys fill higher


# ------------------------------------------------------------------- cycle


def _app_config(tmp_path, risk_config) -> AppConfig:
    return AppConfig(
        risk=risk_config, strategies=StrategyConfig(), project_root=tmp_path
    )


def test_full_cycle_produces_a_shadow_fill_and_audit(
    bullish_pullback_snapshot, account, risk_config, tmp_path
):
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode="shadow", now=bullish_pullback_snapshot.captured_at,
    )

    assert result.outcome is CycleOutcome.SHADOW_FILLED
    assert result.trade is not None
    assert result.audit is not None
    assert result.audit.snapshot_json is not None  # replayable
    assert result.should_submit is False  # shadow never submits


def test_halt_file_stops_the_cycle(
    bullish_pullback_snapshot, account, risk_config, tmp_path
):
    (tmp_path / "HALT").write_text("stop")
    result = run_cycle(
        bullish_pullback_snapshot, account, _app_config(tmp_path, risk_config),
        mode="shadow", now=bullish_pullback_snapshot.captured_at,
    )

    assert result.outcome is CycleOutcome.REJECTED_BY_RISK
    assert result.trade is None
    assert any("HALT" in b for b in result.risk_decision.breached_limits)


def test_every_cycle_produces_an_audit_entry(
    bullish_pullback_snapshot, account, risk_config, tmp_path
):
    """Including the ones that do nothing — that record is the whole point."""
    flat = bullish_pullback_snapshot.model_copy(
        update={
            "indicators": bullish_pullback_snapshot.indicators.model_copy(
                update={"sma_200": Decimal("400.00")}
            )
        }
    )
    result = run_cycle(
        flat, account, _app_config(tmp_path, risk_config),
        mode="shadow", now=flat.captured_at,
    )

    assert result.outcome is CycleOutcome.NO_SIGNAL
    assert result.audit is not None
    assert result.audit.failed_conditions


def test_journal_round_trip_and_performance(tmp_path):
    repo = JournalRepository(tmp_path / "j.db")
    opened = datetime(2026, 8, 1, tzinfo=UTC)
    repo.record_trade(
        TradeRecord(
            client_key="k1", symbol="AAPL", strategy="trend_pullback", mode="shadow",
            opened_at=opened, entry_price=Decimal("300"), quantity=Decimal("1"),
            notional=Decimal("300"), stop_price=Decimal("285"),
        )
    )
    repo.close_trade("k1", Decimal("330"), opened + timedelta(days=10), "target reached")

    summary = repo.performance_summary()
    assert summary["trade_count"] == 1
    assert summary["wins"] == 1
    # +30 on 15 of risk = 2R
    assert summary["expectancy_r"] == "2.00"
    assert repo.has_client_key("k1")
    assert repo.last_losing_exit("AAPL") is None
