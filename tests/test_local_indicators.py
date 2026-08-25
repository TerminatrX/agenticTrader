"""Mathematical tests for the local indicators.

The oracle here is arithmetic, never the broker. Broker agreement is an
*equivalence* question answered separately, against live data; if these tests
used broker output as their expectation they would define the implementation as
"whatever Robinhood returns" and could never detect the two of them being wrong
together, nor localize a disagreement to a formula.

So: series whose answers can be worked out by hand or derived from a closed
form — constant, monotonic, all-gain, all-loss, gaps — plus the boundaries where
history runs out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from agentic_trader.market.local_indicators import (
    SEED_DECAY_TARGET,
    EmaSeed,
    atr_series,
    bar_requirements,
    compute_all,
    ema_series,
    macd_series,
    required_bars,
    rsi_series,
    sma,
    sma_series,
    true_range_series,
)
from agentic_trader.models import Bar

D = Decimal
START = datetime(2026, 1, 5, tzinfo=UTC)


def _bars(rows: list[tuple[str, str, str, str]]) -> list[Bar]:
    """(open, high, low, close) tuples -> Bar list on consecutive days."""
    return [
        Bar(
            begins_at=START + timedelta(days=i),
            open=D(o), high=D(h), low=D(low), close=D(c), volume=1_000_000,
        )
        for i, (o, h, low, c) in enumerate(rows)
    ]


def _flat_bars(closes: list[str], *, spread: str = "0") -> list[Bar]:
    """Bars whose high/low straddle the close by a fixed amount."""
    s = D(spread)
    return [
        Bar(
            begins_at=START + timedelta(days=i),
            open=D(c), high=D(c) + s, low=D(c) - s, close=D(c), volume=1_000_000,
        )
        for i, c in enumerate(closes)
    ]


# ============================================================== SMA


def test_sma_of_a_constant_series_is_that_constant():
    closes = [D("50.00")] * 30
    assert sma(closes, 20) == D("50.00")


def test_sma_of_a_monotonic_run_is_the_window_midpoint():
    """1..20 has mean 10.5 — a closed form, so no hand-arithmetic to get wrong."""
    closes = [D(i) for i in range(1, 21)]
    assert sma(closes, 20) == D("10.5")


def test_sma_reads_only_the_last_window():
    """Prepending ancient history must not move a finite-window mean. This is
    the property that makes SMA immune to how much history was requested."""
    window = [D(i) for i in range(1, 21)]
    assert sma([D("9999")] * 500 + window, 20) == sma(window, 20)


def test_sma_is_none_below_its_period():
    assert sma([D(1)] * 19, 20) is None
    assert sma([D(1)] * 20, 20) is not None


def test_sma_series_length_and_alignment():
    closes = [D(i) for i in range(1, 11)]
    series = sma_series(closes, 3)
    assert len(series) == 8            # 10 closes - 3 + 1
    assert series[0] == D(2)           # mean(1,2,3)
    assert series[-1] == D(9)          # mean(8,9,10)
    assert series[-1] == sma(closes, 3)


def test_sma_rejects_a_nonpositive_period():
    with pytest.raises(ValueError, match="period must be positive"):
        sma([D(1)], 0)


# ============================================================== EMA


def test_ema_of_a_constant_series_is_that_constant_under_both_seeds():
    """A constant input has no transient, so both conventions agree exactly and
    immediately. Any drift here would be an arithmetic bug, not a seed choice."""
    values = [D("7.00")] * 40
    for seed in EmaSeed:
        series = ema_series(values, 9, seed=seed)
        assert series[-1] == D("7.00")
        assert all(v == D("7.00") for v in series)


def test_ema_sma_seed_starts_at_the_simple_mean():
    values = [D(i) for i in range(1, 21)]
    series = ema_series(values, 5, seed=EmaSeed.SMA)
    assert series[0] == D(3)           # mean(1..5)
    assert len(series) == 16           # 20 - 5 + 1


def test_ema_first_value_seed_starts_at_the_first_value():
    values = [D(i) for i in range(1, 21)]
    series = ema_series(values, 5, seed=EmaSeed.FIRST_VALUE)
    assert series[0] == D(1)
    assert len(series) == 20


def test_ema_one_step_matches_the_stated_recurrence():
    """alpha = 2/(period+1) = 2/5 for period 4. From a seed of 10 with the next
    value 20: 20*0.4 + 10*0.6 = 14."""
    values = [D(10)] * 4 + [D(20)]
    series = ema_series(values, 4, seed=EmaSeed.SMA)
    assert series[0] == D(10)
    assert series[1] == D(14)


def test_the_two_ema_seeds_converge_with_enough_history():
    """The claim the convergence allowance rests on. Same data, different seed,
    difference decays below the target once the seed is spent."""
    values = [D(100) + D(i % 7) for i in range(400)]
    a = ema_series(values, 26, seed=EmaSeed.SMA)[-1]
    b = ema_series(values, 26, seed=EmaSeed.FIRST_VALUE)[-1]
    assert abs(a - b) < SEED_DECAY_TARGET * D(100)


def test_ema_of_too_short_a_series_is_empty_under_sma_seed():
    assert ema_series([D(1)] * 4, 5, seed=EmaSeed.SMA) == []
    assert ema_series([], 5, seed=EmaSeed.FIRST_VALUE) == []


# ============================================================== RSI


def test_rsi_of_an_unbroken_advance_is_100():
    """Every delta a gain means avg_loss is 0, and the formula's limit is 100."""
    closes = [D(i) for i in range(1, 40)]
    assert rsi_series(closes)[-1] == D(100)


def test_rsi_of_an_unbroken_decline_is_0():
    closes = [D(i) for i in range(40, 1, -1)]
    assert rsi_series(closes)[-1] == D(0)


def test_rsi_of_a_flat_series_is_50_not_100():
    """The convention that branch ordering would get wrong.

    With no gains and no losses, testing `avg_loss == 0` first returns 100 and
    declares a motionless price maximally overbought — which the exit gate
    reads at 72. Neutral is the only defensible answer.
    """
    assert rsi_series([D("25.00")] * 40)[-1] == D(50)


def test_rsi_is_hand_calculable_on_a_designed_sequence():
    """15 closes: 14 deltas, of which the first 13 are +1 and the last is -2.

    Seed (and only) value: avg_gain = 13/14, avg_loss = 2/14.
    RS = 6.5, RSI = 100 - 100/7.5 = 86.6666...
    """
    closes = [D(100 + i) for i in range(14)] + [D(111)]
    assert len(closes) == 15
    series = rsi_series(closes)
    assert len(series) == 1
    expected = D(100) - (D(100) / (D(1) + (D(13) / D(14)) / (D(2) / D(14))))
    assert series[0] == expected
    assert abs(series[0] - D("86.6667")) < D("0.001")


def test_rsi_uses_wilder_smoothing_not_an_ema():
    """alpha = 1/14, not 2/15. The two are close enough to look right and far
    enough apart to never match the broker, which is the worst combination."""
    closes = [D(100 + i) for i in range(14)] + [D(111), D(112)]
    series = rsi_series(closes)

    # Recompute the second point by hand under Wilder: avg_gain from 13/14
    # smooths toward the new gain of 1, avg_loss from 2/14 toward 0.
    avg_gain = (D(13) / D(14) * D(13) + D(1)) / D(14)
    avg_loss = (D(2) / D(14) * D(13) + D(0)) / D(14)
    expected = D(100) - (D(100) / (D(1) + avg_gain / avg_loss))
    assert series[1] == expected


def test_rsi_needs_period_plus_one_closes():
    assert rsi_series([D(i) for i in range(14)]) == []       # 14 closes, 13 deltas
    assert len(rsi_series([D(i) for i in range(15)])) == 1   # 15 closes, 14 deltas


def test_rsi_series_grows_one_point_per_extra_close():
    base = [D(100 + (i % 5)) for i in range(30)]
    assert len(rsi_series(base)) == len(base) - 14
    assert len(rsi_series(base + [D(103)])) == len(base) - 13


# ============================================================== ATR


def test_true_range_skips_the_first_bar():
    """No previous close exists for bar zero, and substituting high-low there
    would quietly change the seed average and every value after it."""
    bars = _bars([("10", "11", "9", "10"), ("10", "12", "10", "11")])
    trs = true_range_series(bars)
    assert len(trs) == 1
    assert trs[0] == D(2)   # max(12-10, |12-10|, |10-10|) = 2


def test_true_range_takes_the_gap_up_branch():
    """Gap up: high - previous_close exceeds the bar's own range."""
    bars = _bars([("10", "10", "10", "10"), ("20", "22", "20", "21")])
    assert true_range_series(bars)[0] == D(12)   # |22 - 10|


def test_true_range_takes_the_gap_down_branch():
    bars = _bars([("50", "50", "50", "50"), ("30", "32", "28", "30")])
    assert true_range_series(bars)[0] == D(22)   # |28 - 50|


def test_atr_of_a_constant_range_series_is_that_range():
    """Every TR equal means seed and recurrence both land on the same number,
    whatever the smoothing does."""
    bars = _bars([("10", "11", "9", "10")] * 40)
    assert atr_series(bars)[-1] == D(2)


def test_atr_seed_is_the_simple_mean_of_the_first_fourteen_ranges():
    bars = _bars([("10", "11", "9", "10")] * 8 + [("10", "14", "6", "10")] * 7)
    trs = true_range_series(bars)
    assert len(trs) == 14
    expected = sum(trs, D(0)) / D(14)
    series = atr_series(bars)
    assert len(series) == 1
    assert series[0] == expected


def test_atr_recurrence_is_wilder():
    bars = _bars([("10", "11", "9", "10")] * 16)
    trs = true_range_series(bars)
    seed = sum(trs[:14], D(0)) / D(14)
    expected = (seed * D(13) + trs[14]) / D(14)
    assert atr_series(bars)[1] == expected


def test_atr_needs_period_plus_one_bars():
    assert atr_series(_bars([("10", "11", "9", "10")] * 14)) == []
    assert len(atr_series(_bars([("10", "11", "9", "10")] * 15))) == 1


# ============================================================== MACD


def test_macd_of_a_constant_series_is_zero():
    """Both EMAs sit on the constant, so the line, the signal and the histogram
    are all exactly zero. A non-zero result here means a misalignment."""
    closes = [D("42.00")] * 120
    point = macd_series(closes)[-1]
    assert point.line == D(0)
    assert point.signal == D(0)
    assert point.histogram == D(0)


def test_macd_line_is_positive_on_a_rising_series():
    """Fast EMA tracks a rise more closely than slow, so the line is above zero
    and the histogram sign is well defined."""
    closes = [D(100 + i) for i in range(120)]
    point = macd_series(closes)[-1]
    assert point.line > 0
    assert point.histogram == point.line - point.signal


def test_macd_line_is_negative_on_a_falling_series():
    closes = [D(400 - i) for i in range(120)]
    assert macd_series(closes)[-1].line < 0


def test_macd_histogram_equals_line_minus_signal_everywhere():
    """No 2x scaling. Some platforms plot double; if the broker did, the
    comparison would show a clean factor of two rather than noise."""
    closes = [D(100) + D((i * 7) % 23) for i in range(200)]
    for point in macd_series(closes):
        assert point.histogram == point.line - point.signal


def test_macd_warm_up_boundary_is_slow_plus_signal_minus_one():
    """First histogram at close index 33 — so 34 closes for one point, 35 for
    the current-and-previous pair the strategy reads."""
    closes = [D(100 + (i % 11)) for i in range(40)]
    assert macd_series(closes[:33]) == []
    assert len(macd_series(closes[:34])) == 1
    assert len(macd_series(closes[:35])) == 2


def test_macd_fast_must_be_shorter_than_slow():
    with pytest.raises(ValueError, match="must be shorter"):
        macd_series([D(1)] * 100, fast=26, slow=12)


def test_macd_seeds_converge_with_enough_history():
    """Same series, different seed convention. At the pinned lookback the two
    are indistinguishable, which is what lets the seed choice be a documented
    convention rather than a correctness question."""
    closes = [D(100) + D((i * 13) % 29) for i in range(420)]
    a = macd_series(closes, seed=EmaSeed.SMA)[-1]
    b = macd_series(closes, seed=EmaSeed.FIRST_VALUE)[-1]
    assert abs(a.histogram - b.histogram) < D("1e-6")


def test_macd_seeds_disagree_at_the_bare_minimum():
    """The other half, and the reason the contract asks for 277 bars rather
    than 35: at minimum history the seed convention is plainly visible."""
    closes = [D(100) + D((i * 13) % 29) for i in range(35)]
    a = macd_series(closes, seed=EmaSeed.SMA)[-1]
    b = macd_series(closes, seed=EmaSeed.FIRST_VALUE)[-1]
    assert abs(a.histogram - b.histogram) > D("0.01")


# ================================================ insufficient history


def test_compute_all_reports_every_absence_with_a_reason():
    """A value missing for want of history must be distinguishable from a
    value that happens to be zero, and must say which it is."""
    result = compute_all(_flat_bars([str(100 + i) for i in range(30)], spread="1"))

    assert result.sma_20 is not None
    assert result.sma_50 is None
    assert result.sma_200 is None
    assert result.bar_count == 30

    reasons = dict(result.unavailable)
    assert "sma_50" in reasons and "needs 50 bars, given 30" in reasons["sma_50"]
    assert "sma_200" in reasons
    assert "macd_hist_prev" in reasons


def test_compute_all_never_substitutes_a_shorter_period():
    """With 199 bars there is no authoritative SMA200, and a 199-bar mean is a
    different statistic. It must be absent, not approximated."""
    bars = _flat_bars([str(100 + (i % 17)) for i in range(199)], spread="1")
    result = compute_all(bars)
    assert result.sma_200 is None
    assert result.sma_50 is not None

    with_one_more = compute_all(
        _flat_bars([str(100 + (i % 17)) for i in range(200)], spread="1")
    )
    assert with_one_more.sma_200 is not None


def test_a_zero_value_is_not_reported_as_unavailable():
    """The distinction stated as a test: a genuine zero must survive."""
    closes = [str(D(400) - D(i)) for i in range(300)]
    result = compute_all(_flat_bars(closes, spread="1"))
    assert result.rsi_14 == D(0)          # unbroken decline
    assert "rsi_14" not in dict(result.unavailable)


def test_compute_all_on_an_empty_series_reports_everything_missing():
    result = compute_all([])
    assert result.bar_count == 0
    assert dict(result.unavailable).keys() >= {
        "sma_20", "sma_50", "sma_200", "rsi_14", "rsi_prev",
        "atr_14", "macd", "macd_signal", "macd_hist", "macd_hist_prev",
    }


def test_a_full_history_produces_every_value():
    bars = _flat_bars([str(100 + (i * 7) % 31) for i in range(300)], spread="2")
    result = compute_all(bars)
    assert result.unavailable == ()
    for field in ("sma_20", "sma_50", "sma_200", "rsi_14", "rsi_prev",
                  "atr_14", "macd", "macd_signal", "macd_hist", "macd_hist_prev"):
        assert getattr(result, field) is not None, field


# ============================================ the derived bar requirement


def test_macd_binds_the_history_requirement_not_sma200():
    """The finding that set the lookback. The obvious guess is SMA200 at 200
    bars; MACD needs more, because the signal EMA smooths an already-smoothed
    line and the two seeds compose."""
    reqs = {r.label: r for r in bar_requirements()}
    assert reqs["sma_200"].converged_bars == 200
    assert reqs["macd"].converged_bars > reqs["sma_200"].converged_bars
    assert required_bars() == reqs["macd"].converged_bars == 277


def test_a_finite_window_needs_no_convergence_allowance():
    """SMA's two figures are equal, and saying so is a real claim: no amount of
    extra history changes a 200-bar mean."""
    for label in ("sma_20", "sma_50", "sma_200"):
        req = next(r for r in bar_requirements() if r.label == label)
        assert req.minimum_bars == req.converged_bars


def test_recursive_indicators_need_far_more_than_their_minimum():
    """15 bars produce an ATR. They do not produce an ATR anyone should compare
    against a provider's."""
    for label in ("rsi_14", "atr_14", "macd"):
        req = next(r for r in bar_requirements() if r.label == label)
        assert req.converged_bars > req.minimum_bars * 5, label


def test_the_acquisition_contract_carries_the_derived_requirement():
    """The two must not drift. If the arithmetic changes, the pinned contract
    has to change with it — in the same commit, visibly."""
    from agentic_trader.market.acquisition import CURRENT_ACQUISITION

    historicals = CURRENT_ACQUISITION.historicals
    assert historicals.derivation_bars == required_bars()
    assert historicals.lookback_calendar_days >= historicals.minimum_calendar_days()


def test_the_pinned_lookback_actually_supplies_the_required_bars():
    """420 calendar days at ~252 trading days a year is ~290 bars against 277
    needed. Asserted rather than assumed, because 330 -- the figure an earlier
    review floated, and one I repeated -- yields only ~228 and is short."""
    from agentic_trader.market.acquisition import CURRENT_ACQUISITION

    lookback = CURRENT_ACQUISITION.historicals.lookback_calendar_days
    trading_bars = lookback * 252 / 365
    assert trading_bars >= required_bars()
    assert required_bars() > 330 * 252 / 365, "the earlier estimate was short"
