"""The regular-session calendar, and the live admission it enforces.

The calendar is checked against dates that are externally verifiable rather
than against itself: real NYSE holidays and early closes for 2025 and 2026,
including the awkward ones — a Saturday Independence Day observed on the
Friday, Good Friday moving with Easter, Juneteenth only from 2022.

The admission tests exist because the completed-bar rule leans on "actionable
decisions happen during the regular session", and that was previously an
assumption rather than something the code enforced.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from agentic_trader.execution.executor import PreflightError, build_order_payload
from agentic_trader.execution.market_session import (
    SessionStatus,
    early_closes,
    is_trading_day,
    market_holidays,
    session_close,
    session_status,
    to_eastern,
)


def _et(y, m, d, hh, mm=0) -> datetime:
    """An Eastern wall-clock moment, expressed as the UTC instant."""
    naive = datetime(y, m, d, hh, mm, tzinfo=UTC)
    # Walk to the UTC instant whose Eastern rendering is the wanted clock time.
    for offset in (4, 5):
        candidate = naive + timedelta(hours=offset)
        local = to_eastern(candidate)
        if (local.hour, local.minute) == (hh, mm) and local.date() == date(y, m, d):
            return candidate
    raise AssertionError("could not construct that Eastern instant")


# ================================================================= calendar


@pytest.mark.parametrize(
    ("day", "name"),
    [
        (date(2025, 1, 1), "New Year's Day"),
        (date(2025, 1, 20), "MLK Day"),
        (date(2025, 2, 17), "Washington's Birthday"),
        (date(2025, 4, 18), "Good Friday"),
        (date(2025, 5, 26), "Memorial Day"),
        (date(2025, 6, 19), "Juneteenth"),
        (date(2025, 7, 4), "Independence Day"),
        (date(2025, 9, 1), "Labor Day"),
        (date(2025, 11, 27), "Thanksgiving"),
        (date(2025, 12, 25), "Christmas"),
        (date(2026, 4, 3), "Good Friday 2026"),
        (date(2026, 7, 3), "Independence Day observed (4th is a Saturday)"),
        (date(2026, 11, 26), "Thanksgiving 2026"),
    ],
)
def test_known_market_holidays_are_closed(day, name):
    assert day in market_holidays(day.year), name
    assert not is_trading_day(day), name


@pytest.mark.parametrize(
    "day",
    [date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
     date(2026, 11, 27), date(2026, 12, 24)],
)
def test_known_early_closes(day):
    assert day in early_closes(day.year)
    assert session_close(day).hour == 13


def test_july_third_is_not_an_early_close_when_it_is_itself_a_holiday():
    """2026: the 4th is a Saturday, so the 3rd is the observed holiday. It
    cannot also be a shortened session."""
    assert date(2026, 7, 3) in market_holidays(2026)
    assert date(2026, 7, 3) not in early_closes(2026)


def test_juneteenth_is_not_applied_before_2022():
    """Asserting it earlier would mark sessions closed that were open."""
    assert date(2021, 6, 18) not in market_holidays(2021)
    assert date(2022, 6, 20) in market_holidays(2022)


def test_a_normal_trading_day_closes_at_sixteen_hundred():
    assert session_close(date(2026, 8, 25)).hour == 16


def test_a_weekend_has_no_close():
    assert session_close(date(2026, 8, 29)) is None


# ================================================================== status


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        (_et(2026, 8, 25, 13, 0), SessionStatus.OPEN),
        (_et(2026, 8, 25, 9, 30), SessionStatus.OPEN),
        (_et(2026, 8, 25, 9, 29), SessionStatus.BEFORE_OPEN),
        (_et(2026, 8, 25, 16, 0), SessionStatus.AFTER_CLOSE),
        (_et(2026, 8, 25, 20, 0), SessionStatus.AFTER_CLOSE),
        (_et(2026, 8, 29, 12, 0), SessionStatus.WEEKEND),
        (_et(2026, 8, 30, 12, 0), SessionStatus.WEEKEND),
        (_et(2026, 11, 26, 12, 0), SessionStatus.HOLIDAY),
        (_et(2026, 11, 27, 12, 0), SessionStatus.OPEN),      # early close day
        (_et(2026, 11, 27, 13, 0), SessionStatus.AFTER_CLOSE),
        (_et(2026, 11, 27, 15, 0), SessionStatus.AFTER_CLOSE),
    ],
)
def test_session_status(instant, expected):
    assert session_status(instant) is expected


def test_the_early_close_is_the_case_a_naive_rule_gets_wrong():
    """14:00 on the day after Thanksgiving looks open to a 09:30-16:00 rule and
    is not. That is a thin, gappy session — the worst place to be wrong."""
    assert session_status(_et(2026, 11, 27, 14, 0)) is SessionStatus.AFTER_CLOSE
    assert session_status(_et(2026, 11, 30, 14, 0)) is SessionStatus.OPEN


def test_status_is_derived_from_the_given_instant_not_the_clock():
    """Replay must reach the same answer forever."""
    instant = _et(2026, 8, 25, 13, 0)
    assert session_status(instant) is session_status(instant)
    assert session_status(instant) is SessionStatus.OPEN


def test_dst_boundaries_resolve_correctly():
    """Eastern is UTC-4 in summer and UTC-5 in winter; getting this backwards
    would shift every session by an hour."""
    assert to_eastern(datetime(2026, 7, 1, 16, 0, tzinfo=UTC)).hour == 12
    assert to_eastern(datetime(2026, 1, 15, 16, 0, tzinfo=UTC)).hour == 11


# ======================================================== live admission


def _live_plan(decision, snapshot, risk_config, *, now, mode="live"):
    return build_order_payload(
        decision, snapshot, "TEST0000",
        max_spread_pct=risk_config.max_spread_pct,
        max_price_drift_pct=risk_config.max_price_drift_pct,
        mode=mode, now=now,
    )


@pytest.fixture
def approved(entry_signal, bullish_pullback_snapshot, account, risk_config):
    from agentic_trader.risk.engine import RiskEngine

    return RiskEngine(risk_config).evaluate(
        entry_signal, bullish_pullback_snapshot, account,
        as_of=bullish_pullback_snapshot.captured_at.date(),
    )


@pytest.mark.parametrize(
    ("label", "instant"),
    [
        ("before open", _et(2026, 8, 13, 9, 0)),
        ("after close", _et(2026, 8, 13, 17, 0)),
        ("weekend", _et(2026, 8, 15, 12, 0)),
        ("holiday", _et(2026, 11, 26, 12, 0)),
        ("after an early close", _et(2026, 11, 27, 14, 0)),
    ],
)
def test_a_live_payload_is_refused_outside_the_session(
    approved, bullish_pullback_snapshot, risk_config, label, instant
):
    """Every closed case, including the two a naive rule would admit."""
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    with pytest.raises(PreflightError, match="outside the regular session"):
        _live_plan(approved, stale, risk_config, now=instant)


def test_shadow_still_evaluates_outside_the_session(
    approved, bullish_pullback_snapshot, risk_config
):
    """Out-of-hours evaluation is how analysis and replay work, and a shadow
    fill reaches nothing."""
    instant = _et(2026, 8, 15, 12, 0)          # a Saturday
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    plan = _live_plan(approved, stale, risk_config, now=instant, mode="shadow")
    assert plan.mode == "shadow"


def test_a_sell_exit_is_covered_not_only_a_buy_entry(
    bullish_pullback_snapshot, account, risk_config, held_position
):
    """The gap this closes. The protection floor only guards entries, so an
    exit built after hours previously faced no session check at all."""
    from agentic_trader.models import Side, Signal, SignalStrength
    from agentic_trader.risk.engine import RiskEngine

    holding = account.model_copy(update={"positions": [held_position]})
    exit_signal = Signal(
        symbol="AAPL", strategy="trend_pullback", strength=SignalStrength.EXIT,
        side=Side.SELL, confidence=1.0,
        reference_price=bullish_pullback_snapshot.reference_price,
        reasons=["lost the 50-day"],
    )
    decision = RiskEngine(risk_config).evaluate(
        exit_signal, bullish_pullback_snapshot, holding,
        as_of=bullish_pullback_snapshot.captured_at.date(),
    )
    assert decision.is_executable
    assert decision.intent.side is Side.SELL

    instant = _et(2026, 8, 13, 20, 0)
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    with pytest.raises(PreflightError, match="outside the regular session"):
        _live_plan(decision, stale, risk_config, now=instant)


def test_a_live_payload_builds_inside_the_session(
    approved, bullish_pullback_snapshot, risk_config
):
    """The gate must not block the case it exists to permit. This still hits
    the protection floor afterwards, which is the pre-existing behaviour."""
    instant = _et(2026, 8, 13, 13, 44)
    fresh = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    with pytest.raises(PreflightError, match="not provably protected"):
        _live_plan(approved, fresh, risk_config, now=instant)


# ======================================================= calendar horizon


def test_the_pinned_horizon_is_explicit():
    from agentic_trader.execution.market_session import PINNED_SESSION_YEARS

    assert frozenset({2026, 2027, 2028}) == PINNED_SESSION_YEARS


@pytest.mark.parametrize(
    ("day", "name"),
    [
        (date(2027, 1, 1), "New Year's Day 2027"),
        (date(2027, 3, 26), "Good Friday 2027"),
        (date(2027, 6, 18), "Juneteenth observed (19th is a Saturday)"),
        (date(2027, 7, 5), "Independence observed (4th is a Sunday)"),
        (date(2027, 11, 25), "Thanksgiving 2027"),
        (date(2027, 12, 24), "Christmas observed (25th is a Saturday)"),
        (date(2028, 1, 17), "MLK Day 2028"),
        (date(2028, 4, 14), "Good Friday 2028"),
        (date(2028, 6, 19), "Juneteenth 2028"),
        (date(2028, 7, 4), "Independence Day 2028"),
        (date(2028, 11, 23), "Thanksgiving 2028"),
        (date(2028, 12, 25), "Christmas 2028"),
    ],
)
def test_published_2027_and_2028_holidays(day, name):
    assert day in market_holidays(day.year), name


@pytest.mark.parametrize(
    "day", [date(2027, 11, 26), date(2028, 7, 3), date(2028, 11, 24)]
)
def test_published_2027_and_2028_early_closes(day):
    assert day in early_closes(day.year)
    assert session_close(day).hour == 13


def test_a_saturday_new_year_is_not_observed_on_the_prior_year_end():
    """1 January 2028 is a Saturday. The exchange does not close 31 December
    2027 for it, and asserting otherwise would mark a full session closed."""
    assert date(2027, 12, 31) not in market_holidays(2027)
    assert date(2028, 1, 1) not in market_holidays(2028)


def test_christmas_eve_2027_is_a_holiday_not_an_early_close():
    assert date(2027, 12, 24) in market_holidays(2027)
    assert date(2027, 12, 24) not in early_closes(2027)


def test_an_unpinned_year_is_unsupported_even_mid_session():
    """A 2029 Wednesday at 13:00 ET looks like a textbook open session. The
    rules would happily produce an answer; nobody has checked it against a
    published schedule, so the gate refuses instead."""
    instant = _et(2029, 8, 15, 13, 0)
    assert instant.astimezone(UTC).weekday() < 5
    assert session_status(instant) is SessionStatus.UNSUPPORTED_CALENDAR
    assert not session_status(instant).is_open


@pytest.mark.parametrize("year", [2025, 2029, 2035])
def test_years_outside_the_horizon_are_all_unsupported(year):
    assert session_status(_et(year, 6, 17, 13, 0)) is (
        SessionStatus.UNSUPPORTED_CALENDAR
    )


def test_a_live_buy_is_refused_in_an_unpinned_year(
    approved, bullish_pullback_snapshot, risk_config
):
    instant = _et(2029, 8, 15, 13, 0)
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    with pytest.raises(PreflightError, match="outside the regular session"):
        _live_plan(approved, stale, risk_config, now=instant)


def test_a_live_sell_is_refused_in_an_unpinned_year(
    bullish_pullback_snapshot, account, risk_config, held_position
):
    from agentic_trader.models import Side, Signal, SignalStrength
    from agentic_trader.risk.engine import RiskEngine

    holding = account.model_copy(update={"positions": [held_position]})
    decision = RiskEngine(risk_config).evaluate(
        Signal(
            symbol="AAPL", strategy="trend_pullback", strength=SignalStrength.EXIT,
            side=Side.SELL, confidence=1.0,
            reference_price=bullish_pullback_snapshot.reference_price,
            reasons=["lost the 50-day"],
        ),
        bullish_pullback_snapshot, holding,
        as_of=bullish_pullback_snapshot.captured_at.date(),
    )
    instant = _et(2029, 8, 15, 13, 0)
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    with pytest.raises(PreflightError, match="outside the regular session"):
        _live_plan(decision, stale, risk_config, now=instant)


def test_shadow_still_works_in_an_unpinned_year(
    approved, bullish_pullback_snapshot, risk_config
):
    """Analysis and replay over any period must stay possible."""
    instant = _et(2029, 8, 15, 13, 0)
    stale = bullish_pullback_snapshot.model_copy(
        update={"captured_at": instant, "quote_as_of": instant}
    )
    plan = _live_plan(approved, stale, risk_config, now=instant, mode="shadow")
    assert plan.mode == "shadow"


def test_the_refusal_message_names_the_horizon():
    from agentic_trader.execution.market_session import describe

    assert "outside the verified session calendar" in describe(_et(2029, 8, 15, 13, 0))
