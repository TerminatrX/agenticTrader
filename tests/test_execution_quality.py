"""Tests for the execution-quality controls: quote age, spread, and the stop.

Each of these controls existed before and could not fire. The tests here are
written to fail against that earlier behaviour, so they are the thing that keeps
the controls honest rather than merely present:

- staleness measured `captured_at`, which is stamped `now()` at construction, so
  every snapshot looked fresh — including one replayed from a stored bundle
- the "spread" check compared the live price to the decision price, which is
  drift; bid and ask were never read at all
- the stop sized the position and was then never compared to price again

The recurring shape is: a value that is *unknown* must be refused, never treated
as benign. Absent data reading as a passing check is the failure mode all three
of these shared.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from agentic_trader.agents.critic import critique
from agentic_trader.execution.executor import PreflightError, build_order_payload, preflight
from agentic_trader.market.snapshot import build_snapshot
from agentic_trader.models import MarketSnapshot, Position, SignalStrength
from agentic_trader.risk.engine import RiskEngine
from agentic_trader.strategies.base import StrategyContext
from agentic_trader.strategies.trend_pullback import TrendPullbackStrategy

NOW = datetime(2026, 8, 13, 17, 44, tzinfo=UTC)


def _decision(entry_signal, snapshot, account, risk_config):
    return RiskEngine(risk_config).evaluate(
        entry_signal, snapshot, account, as_of=date(2026, 8, 13)
    )


# --------------------------------------------------------------- quote age


def test_a_freshly_captured_stale_quote_is_rejected(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The regression that motivated this control.

    `captured_at` is now, so the old age check saw a zero-second-old snapshot
    and passed. The quote itself is ten minutes stale, which is what actually
    matters when the order about to be sent is a market order.
    """
    stale = bullish_pullback_snapshot.model_copy(
        update={
            "captured_at": NOW,
            "quote_as_of": NOW - timedelta(minutes=10),
        }
    )
    decision = _decision(entry_signal, stale, account, risk_config)

    with pytest.raises(PreflightError, match="quote is 600s old"):
        build_order_payload(decision, stale, account.account_number, now=NOW)


def test_missing_quote_timestamp_is_refused_rather_than_assumed_fresh(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    unstamped = bullish_pullback_snapshot.model_copy(update={"quote_as_of": None})
    decision = _decision(entry_signal, unstamped, account, risk_config)

    with pytest.raises(PreflightError, match="no venue timestamp"):
        build_order_payload(decision, unstamped, account.account_number, now=NOW)


def test_quote_age_is_independent_of_capture_time(bullish_pullback_snapshot):
    """An old capture of a fresh quote is fine; the reverse is not."""
    snap = bullish_pullback_snapshot.model_copy(
        update={
            "captured_at": NOW - timedelta(hours=3),
            "quote_as_of": NOW - timedelta(seconds=5),
        }
    )
    assert snap.quote_age_seconds(NOW) == 5


def test_critic_blocks_a_quote_it_cannot_date(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    unstamped = bullish_pullback_snapshot.model_copy(update={"quote_as_of": None})
    decision = _decision(entry_signal, unstamped, account, risk_config)

    report = critique(decision, entry_signal, unstamped, account, risk_config, now=NOW)

    assert not report.approved
    assert any("no venue timestamp" in b for b in report.blocks)


# ------------------------------------------------------------------ spread


def test_wide_spread_blocks_the_order(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    wide = bullish_pullback_snapshot.model_copy(
        update={"bid": Decimal("300.00"), "ask": Decimal("303.00")}
    )
    decision = _decision(entry_signal, wide, account, risk_config)

    with pytest.raises(PreflightError, match="spread is 1.00%"):
        build_order_payload(decision, wide, account.account_number, now=NOW)


@pytest.mark.parametrize(
    ("bid", "ask", "why"),
    [
        (None, Decimal("303.57"), "missing bid"),
        (Decimal("303.51"), None, "missing ask"),
        (Decimal("0"), Decimal("0"), "broker's no-book sentinel"),
        (Decimal("303.60"), Decimal("303.50"), "crossed book"),
        (Decimal("303.50"), Decimal("303.50"), "locked book"),
    ],
)
def test_unusable_book_is_unknown_not_zero(bullish_pullback_snapshot, bid, ask, why):
    """`None`, never `0`. A zero spread would pass the tightest possible check."""
    snap = bullish_pullback_snapshot.model_copy(update={"bid": bid, "ask": ask})
    assert snap.spread_pct is None, why


def test_unknown_spread_blocks_the_order(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    no_book = bullish_pullback_snapshot.model_copy(
        update={"bid": Decimal("0"), "ask": Decimal("0")}
    )
    decision = _decision(entry_signal, no_book, account, risk_config)

    with pytest.raises(PreflightError, match="no usable bid/ask"):
        build_order_payload(decision, no_book, account.account_number, now=NOW)


def test_spread_and_drift_are_separately_tunable(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The two keys must gate different things, which is why they were split.

    A tight book with a price that has run away trips drift alone; a wide book
    at the decision price trips spread alone.
    """
    decision = _decision(entry_signal, bullish_pullback_snapshot, account, risk_config)

    drifted = bullish_pullback_snapshot.model_copy(update={"last_price": Decimal("330.00")})
    with pytest.raises(PreflightError, match="price moved"):
        preflight(
            decision, drifted,
            max_spread_pct=Decimal("0.10"),      # spread wide open
            max_price_drift_pct=Decimal("0.005"),
            now=NOW,
        )

    wide = bullish_pullback_snapshot.model_copy(
        update={"bid": Decimal("300.00"), "ask": Decimal("303.00")}
    )
    with pytest.raises(PreflightError, match="spread is"):
        preflight(
            decision, wide,
            max_spread_pct=Decimal("0.005"),
            max_price_drift_pct=Decimal("0.10"),  # drift wide open
            now=NOW,
        )


def test_preflight_reports_the_observed_spread(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Recorded so a threshold can later come from data rather than a guess."""
    decision = _decision(entry_signal, bullish_pullback_snapshot, account, risk_config)
    plan = build_order_payload(decision, bullish_pullback_snapshot, account.account_number, now=NOW)

    assert any("spread 0.02%" in n for n in plan.preflight_notes)
    assert any("quote age" in n for n in plan.preflight_notes)
    # Latency is reported but must never gate.
    assert any("pipeline latency" in n for n in plan.preflight_notes)


# --------------------------------------------------- price selection (parser)


def _quote_payload(**overrides):
    quote = {
        "symbol": "AAPL",
        "last_trade_price": "305.31",
        "venue_last_trade_time": "2026-08-13T19:59:59.998116249Z",
        "last_non_reg_trade_price": "304.85",
        "venue_last_non_reg_trade_time": "2026-08-13T21:42:14.458977236Z",
        "adjusted_previous_close": "302.25",
        "bid_price": "304.85",
        "ask_price": "304.92",
        "venue_bid_time": "2026-08-13T21:42:19.191656076Z",
        "venue_ask_time": "2026-08-13T21:42:19.191656076Z",
        "has_traded": True,
        "state": "active",
    }
    quote.update(overrides)
    return {"data": {"results": [{"quote": quote}]}}


def test_the_more_recent_print_wins_after_hours():
    """Real shape: the extended-hours print is newer than the closing print.

    Reading `last_trade_price` unconditionally quotes a price hours old while
    the live book (304.85/304.92) brackets the other one.
    """
    snap = build_snapshot("AAPL", quote=_quote_payload())

    assert snap.last_price == Decimal("304.85")
    assert snap.quote_as_of == datetime(2026, 8, 13, 21, 42, 14, 458977, tzinfo=UTC)


def test_the_regular_print_wins_when_it_is_newer():
    snap = build_snapshot(
        "AAPL",
        quote=_quote_payload(venue_last_non_reg_trade_time="2026-08-13T12:00:00Z"),
    )
    assert snap.last_price == Decimal("305.31")
    assert snap.quote_as_of == datetime(2026, 8, 13, 19, 59, 59, 998116, tzinfo=UTC)


def test_a_single_print_needs_no_comparison():
    snap = build_snapshot(
        "AAPL",
        quote=_quote_payload(last_non_reg_trade_price=None, venue_last_non_reg_trade_time=None),
    )
    assert snap.last_price == Decimal("305.31")


def test_zero_bid_ask_is_dropped_at_the_parser():
    snap = build_snapshot("AAPL", quote=_quote_payload(bid_price="0", ask_price="0"))

    assert snap.bid is None
    assert snap.ask is None
    assert snap.spread_pct is None


def test_book_time_is_the_older_of_the_two_sides():
    snap = build_snapshot(
        "AAPL", quote=_quote_payload(venue_bid_time="2026-08-13T21:00:00Z")
    )
    assert snap.book_as_of == datetime(2026, 8, 13, 21, 0, tzinfo=UTC)


def test_a_real_book_produces_a_real_spread():
    snap = build_snapshot("AAPL", quote=_quote_payload())
    # (304.92 - 304.85) / 304.885
    assert snap.spread_pct is not None
    assert Decimal("0.0002") < snap.spread_pct < Decimal("0.0003")


# ------------------------------------------------------------- stop breach


def _held(snapshot: MarketSnapshot, stop: Decimal | None) -> StrategyContext:
    return StrategyContext(
        position=Position(
            symbol=snapshot.symbol,
            quantity=Decimal("0.05"),
            average_cost=Decimal("302.25"),
        ),
        active_stop=stop,
    )


def test_live_price_at_or_below_the_stop_exits(bullish_pullback_snapshot):
    snap = bullish_pullback_snapshot.model_copy(update={"last_price": Decimal("287.00")})
    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, Decimal("287.14")))

    assert signal.strength is SignalStrength.EXIT
    assert "STOP BREACHED" in signal.reasons[0]


def test_the_stop_reason_leads_even_when_other_exits_also_fire(bullish_pullback_snapshot):
    """Price below the stop is also below the SMA50 — the stop must be told first."""
    snap = bullish_pullback_snapshot.model_copy(update={"last_price": Decimal("250.00")})
    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, Decimal("287.14")))

    assert signal.reasons[0].startswith("STOP BREACHED")
    assert len(signal.reasons) > 1  # the SMA50 break is still recorded


def test_an_intrabar_touch_exits_even_though_price_recovered(bullish_pullback_snapshot):
    """The stop is managed, not resting at the broker.

    A bar that pierced the level and closed back above it would still have taken
    a real stop order out. Holding on would be claiming protection the position
    never had.
    """
    bar = bullish_pullback_snapshot.bars[-1]
    pierced = bar.model_copy(update={"low": Decimal("286.00")})
    snap = bullish_pullback_snapshot.model_copy(
        update={"bars": [*bullish_pullback_snapshot.bars[:-1], pierced],
                "last_price": Decimal("303.54")}
    )

    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, Decimal("287.14")))

    assert signal.strength is SignalStrength.EXIT
    assert "intrabar" in signal.reasons[0]


def _holding_quietly(snapshot: MarketSnapshot) -> MarketSnapshot:
    """A held position with no exit condition met.

    The shared fixture is a pullback *below* the 50-day, which is itself an exit
    ("thesis broken") for anyone already holding. Drop the 50-day under price so
    these tests isolate the stop rather than re-testing that rule.
    """
    return snapshot.model_copy(
        update={
            "last_price": Decimal("303.54"),
            "indicators": snapshot.indicators.model_copy(update={"sma_50": Decimal("295.00")}),
        }
    )


def test_price_above_the_stop_does_not_exit(bullish_pullback_snapshot):
    snap = _holding_quietly(bullish_pullback_snapshot)
    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, Decimal("287.14")))

    assert signal.strength is SignalStrength.NONE


def test_no_stop_on_record_is_reported_not_silently_ignored(bullish_pullback_snapshot):
    """An unknown stop must not read as 'this position has no stop'."""
    snap = _holding_quietly(bullish_pullback_snapshot)
    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, None))

    assert signal.strength is SignalStrength.NONE
    assert "no stop on record" in signal.reasons[0]


def test_the_exit_signal_carries_the_stop_that_triggered_it(bullish_pullback_snapshot):
    snap = bullish_pullback_snapshot.model_copy(update={"last_price": Decimal("287.00")})
    signal = TrendPullbackStrategy().evaluate(snap, _held(snap, Decimal("287.14")))

    assert signal.stop_price == Decimal("287.14")
    assert signal.metrics["active_stop"] == 287.14
