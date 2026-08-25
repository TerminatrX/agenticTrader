"""Tests for the comparison machinery, and for its isolation from decisions.

Two jobs. First, prove the comparison measures what it claims — especially that
a direction flip is caught even when the absolute error is tiny, which is the
failure mode a numeric tolerance alone would miss. Second, prove the whole
apparatus is inert: local values are diagnostics in this branch, and a test has
to hold that open rather than trusting that nobody wired them in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from agentic_trader.market.indicator_comparison import (
    DecisionFlags,
    ValueDelta,
    decision_flags,
    stop_impact,
    summarize,
)

D = Decimal
TS = datetime(2026, 8, 24, tzinfo=UTC)


def _flags(**overrides) -> DecisionFlags:
    base: dict = {
        "price": D("100"),
        "sma_20": D("102"), "sma_50": D("95"), "sma_200": D("90"),
        "rsi_14": D("38"), "rsi_prev": D("36"),
        "macd_hist": D("0.5"), "macd_hist_prev": D("0.4"),
        "atr_14": D("2.0"),
    }
    return decision_flags(**{**base, **overrides})


# ================================================== ValueDelta


def test_a_missing_side_is_not_counted_as_agreement():
    """An unavailable local value must not read as zero error — that would let
    a broken local implementation inflate the pass rate to 100%."""
    delta = ValueDelta("sma_200", TS, local=None, broker=D("50"))
    assert delta.comparable is False
    assert delta.abs_error is None
    assert delta.rel_error is None


def test_absolute_and_relative_error():
    delta = ValueDelta("sma_20", TS, local=D("101"), broker=D("100"))
    assert delta.abs_error == D("1")
    assert delta.rel_error == D("0.01")


def test_relative_error_against_zero_is_undefined_not_invented():
    """MACD histograms sit at zero legitimately. Returning something huge or
    something tiny would both be fabrications."""
    delta = ValueDelta("macd_hist", TS, local=D("0.001"), broker=D("0"))
    assert delta.abs_error == D("0.001")
    assert delta.rel_error is None


def test_summarize_reports_median_as_well_as_max():
    """One outlier and a uniformly shifted series need to be distinguishable;
    a max alone cannot tell them apart."""
    deltas = [
        ValueDelta("x", TS, D("100"), D("100")),
        ValueDelta("x", TS, D("100.01"), D("100")),
        ValueDelta("x", TS, D("105"), D("100")),
        ValueDelta("x", TS, None, D("100")),
    ]
    summary = summarize("x", deltas)
    assert summary.compared == 3
    assert summary.missing == 1
    assert summary.max_abs_error == D("5")
    assert summary.median_abs_error == D("0.01")


def test_summarize_of_nothing_comparable_is_empty_not_zero():
    summary = summarize("x", [ValueDelta("x", TS, None, None)])
    assert summary.compared == 0
    assert summary.max_abs_error is None
    assert summary.median_abs_error is None


# ================================================== DecisionFlags


def test_flags_reproduce_the_strategy_conditions():
    flags = _flags()
    assert flags.above_sma200 is True       # 100 > 90
    assert flags.stack_bullish is True      # 95 > 90
    assert flags.in_pullback is True        # 100 < 102
    assert flags.rsi_in_band is True        # 30 <= 38 <= 45
    assert flags.rsi_improving is True      # 38 > 36
    assert flags.macd_hist_rising is True   # 0.5 > 0.4
    assert flags.exit_rsi_reached is False


def test_a_missing_input_yields_none_never_false():
    """`None` is "we could not tell". The strategy fails the condition on it,
    which is different from evaluating it as False and must stay visible."""
    flags = _flags(sma_200=None, rsi_14=None)
    assert flags.above_sma200 is None
    assert flags.rsi_in_band is None
    assert flags.rsi_improving is None


def test_none_versus_a_boolean_counts_as_a_disagreement():
    """Two sides where one knows and the other does not have reached different
    conclusions, and the funnel treats them differently."""
    known = _flags()
    unknown = _flags(sma_200=None)
    assert "above_sma200" in known.disagreements(unknown)


def test_identical_inputs_disagree_on_nothing():
    assert _flags().disagreements(_flags()) == ()


def test_a_tiny_error_that_flips_direction_is_caught():
    """The central case for this whole module.

    Two histogram readings 0.0002 apart. An absolute-error tolerance of 0.001
    calls that agreement; the strategy asks `hist > hist_prev`, and the two
    sides answer differently. Numeric closeness is not decision equivalence.
    """
    local = _flags(macd_hist=D("0.1001"), macd_hist_prev=D("0.1000"))
    broker = _flags(macd_hist=D("0.0999"), macd_hist_prev=D("0.1000"))

    delta = ValueDelta("macd_hist", TS, D("0.1001"), D("0.0999"))
    assert delta.abs_error == D("0.0002")
    assert delta.abs_error < D("0.001")          # "within tolerance"

    assert local.macd_hist_rising is True
    assert broker.macd_hist_rising is False
    assert "macd_hist_rising" in local.disagreements(broker)


def test_a_large_error_that_preserves_direction_is_not_a_disagreement():
    """The converse, and the reason numeric error alone also over-reports: a
    much bigger absolute difference that leaves every conclusion intact."""
    local = _flags(sma_200=D("50"))
    broker = _flags(sma_200=D("70"))
    assert ValueDelta("sma_200", TS, D("50"), D("70")).abs_error == D("20")
    assert local.disagreements(broker) == ()


def test_rsi_band_membership_flips_at_the_boundary():
    inside = _flags(rsi_14=D("45.00"))
    outside = _flags(rsi_14=D("45.01"))
    assert inside.rsi_in_band is True
    assert outside.rsi_in_band is False
    assert "rsi_in_band" in inside.disagreements(outside)


def test_the_comparison_defaults_match_the_strategy():
    """These thresholds are duplicated from `TrendPullbackStrategy.DEFAULTS` so
    the comparison does not need a full snapshot per side. Duplication is only
    safe while something checks it."""
    from agentic_trader.strategies.trend_pullback import TrendPullbackStrategy

    defaults = TrendPullbackStrategy.DEFAULTS
    assert D(str(defaults["rsi_floor"])) == D("30")
    assert D(str(defaults["rsi_ceiling"])) == D("45")
    assert D(str(defaults["exit_rsi"])) == D("72")
    assert defaults["max_pullback_pct"] == D("0.12")
    assert defaults["atr_stop_multiple"] == D("2.0")
    assert defaults["min_stop_pct"] == D("0.02")
    assert defaults["max_stop_pct"] == D("0.12")
    assert defaults["stop_pct"] == D("0.05")


# ================================================== ATR economics


def test_an_atr_difference_moves_position_size_proportionally():
    """Sizing is `risk_budget / stop_distance`, so a 5% ATR difference is a ~5%
    position-size difference. The risk budget cancels, which is why this can be
    measured without an account."""
    impact = stop_impact(
        price=D("100"), local_atr=D("2.00"), broker_atr=D("2.10"), sma_50=None,
    )
    assert impact.comparable
    assert impact.stop_price_diff == D("0.20")
    rel = impact.notional_rel_diff
    assert D("0.04") < rel < D("0.05")


def test_an_atr_difference_across_the_ceiling_changes_trade_or_no_trade():
    """Above `max_stop_pct` the strategy declines the setup rather than
    clamping, so this is not a sizing disagreement — it is one side trading and
    the other not."""
    impact = stop_impact(
        price=D("100"), local_atr=D("5.99"), broker_atr=D("6.01"), sma_50=None,
    )
    assert impact.local.too_volatile_to_trade is False
    assert impact.broker.too_volatile_to_trade is True
    assert impact.tradability_disagrees is True


def test_identical_atr_produces_no_impact():
    impact = stop_impact(
        price=D("100"), local_atr=D("2.00"), broker_atr=D("2.00"), sma_50=D("95"),
    )
    assert impact.stop_price_diff == 0
    assert impact.notional_rel_diff == 0
    assert impact.basis_disagrees is False
    assert impact.tradability_disagrees is False


def test_a_basis_disagreement_is_reported_separately():
    """One side volatility-derived, the other flat-percentage: same shape of
    number, different claim about where it came from."""
    impact = stop_impact(
        price=D("100"), local_atr=None, broker_atr=D("2.00"), sma_50=None,
    )
    assert impact.basis_disagrees is True


def test_stop_impact_is_incomparable_without_both_sides():
    impact = stop_impact(price=D("0"), local_atr=D("1"), broker_atr=D("1"), sma_50=None)
    assert impact.comparable is False
    assert impact.notional_rel_diff is None


# ============================================ isolation from the decision path


def test_local_indicators_never_reach_a_snapshot(
    bullish_pullback_snapshot, account, risk_config, tmp_path
):
    """Phase 13's requirement, as an executable check.

    Compute local indicators from the snapshot's own bars, confirm they differ
    from the broker values the snapshot carries, then run the real cycle twice —
    once ignoring the local set entirely. The decision must be identical,
    because nothing consults it.
    """
    from agentic_trader.agents.orchestrator import run_cycle
    from agentic_trader.config import AppConfig, StrategyConfig
    from agentic_trader.market.local_indicators import compute_all
    from agentic_trader.models import ExecutionMode

    config = AppConfig(
        risk=risk_config, strategies=StrategyConfig(), project_root=tmp_path
    )
    trading_date = bullish_pullback_snapshot.captured_at.date()

    local = compute_all(bullish_pullback_snapshot.bars)
    # The fixture carries two bars, so almost everything is unavailable — which
    # is itself the point: a decision path reading this would collapse.
    assert local.sma_200 is None
    assert bullish_pullback_snapshot.indicators.sma_200 is not None

    before = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=trading_date,
        now=bullish_pullback_snapshot.captured_at,
    )
    after = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=trading_date,
        now=bullish_pullback_snapshot.captured_at,
    )

    assert before.outcome is after.outcome
    assert before.signal.confidence == after.signal.confidence
    assert before.signal.stop_price == after.signal.stop_price
    assert before.risk_decision.approved_notional == after.risk_decision.approved_notional


def test_changing_a_local_diagnostic_cannot_change_a_decision(
    bullish_pullback_snapshot, account, risk_config, tmp_path
):
    """Deliberately absurd local values, deliberately no effect.

    `LocalIndicatorSet` is a plain frozen dataclass with no route into
    `MarketSnapshot`; this asserts that structurally rather than by inspection.
    """
    import dataclasses

    from agentic_trader.agents.orchestrator import run_cycle
    from agentic_trader.config import AppConfig, StrategyConfig
    from agentic_trader.market.local_indicators import compute_all
    from agentic_trader.models import ExecutionMode

    config = AppConfig(
        risk=risk_config, strategies=StrategyConfig(), project_root=tmp_path
    )
    trading_date = bullish_pullback_snapshot.captured_at.date()

    baseline = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=trading_date,
        now=bullish_pullback_snapshot.captured_at,
    )

    absurd = dataclasses.replace(
        compute_all(bullish_pullback_snapshot.bars),
        sma_200=D("999999"), rsi_14=D("99"), atr_14=D("500"),
        macd_hist=D("-42"), macd_hist_prev=D("42"),
    )
    assert absurd.sma_200 == D("999999")

    after = run_cycle(
        bullish_pullback_snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=trading_date,
        now=bullish_pullback_snapshot.captured_at,
    )

    assert after.outcome is baseline.outcome
    assert after.signal.stop_price == baseline.signal.stop_price
    assert after.risk_decision.approved_notional == (
        baseline.risk_decision.approved_notional
    )
    # And the snapshot still reports the broker's numbers, unchanged.
    assert bullish_pullback_snapshot.indicators.sma_200 != D("999999")


def test_no_decision_module_imports_the_local_indicators():
    """The structural half. If a strategy, risk gate, or the critic ever imports
    this module, that is the cutover — and it belongs in its own reviewed
    change, not as a quiet edit here."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentic_trader"
    decision_paths = [
        root / "strategies",
        root / "risk",
        root / "agents" / "critic.py",
        root / "market" / "snapshot.py",
        root / "market" / "signals.py",
        root / "market" / "symbol_regime.py",
    ]

    offenders = []
    for path in decision_paths:
        files = path.rglob("*.py") if path.is_dir() else [path]
        for file in files:
            if "local_indicators" in file.read_text(encoding="utf-8"):
                offenders.append(str(file.relative_to(root)))

    assert offenders == [], f"local indicators reached the decision path: {offenders}"


@pytest.mark.parametrize(
    "module",
    ["agentic_trader.market.local_indicators", "agentic_trader.market.indicator_comparison"],
)
def test_the_validation_modules_perform_no_broker_io(module):
    """`src/` never touches the network, validation code included."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    for banned in ("import requests", "import httpx", "urllib.request", "socket"):
        assert banned not in source


# ============================================ measured tolerances (Phase 11)


def test_every_compared_field_has_a_tolerance():
    """A field with no tolerance must raise rather than silently pass. An
    unmeasured indicator quietly counting as equivalent is the failure mode
    this whole branch exists to avoid."""
    from agentic_trader.market.indicator_comparison import (
        PROPOSED_TOLERANCES,
        within_tolerance,
    )

    assert set(PROPOSED_TOLERANCES) == {
        "sma_20", "sma_50", "sma_200", "rsi_14", "atr_14",
        "macd", "macd_signal", "macd_hist",
    }
    with pytest.raises(KeyError, match="no tolerance proposed"):
        within_tolerance(ValueDelta("unmeasured", TS, D("1"), D("1")))


def test_tolerances_are_indicator_specific_not_uniform():
    """One number across all eight would be simultaneously loose for SMA, which
    agrees to 4e-13, and tight for a MACD histogram that crosses zero."""
    from agentic_trader.market.indicator_comparison import PROPOSED_TOLERANCES

    assert len(set(PROPOSED_TOLERANCES.values())) > 1
    assert PROPOSED_TOLERANCES["sma_200"] < PROPOSED_TOLERANCES["rsi_14"]


def test_each_tolerance_clears_the_observed_maximum_with_headroom():
    """Proposed after measuring, not before. Each sits at least an order of
    magnitude above what was actually observed on 2026-08-25."""
    from agentic_trader.market.indicator_comparison import PROPOSED_TOLERANCES

    observed_max = {
        "sma_20": D("4.0e-13"), "sma_50": D("4.0e-13"), "sma_200": D("4.0e-13"),
        "rsi_14": D("1.204e-7"), "atr_14": D("1.766e-8"),
        "macd": D("6.157e-8"), "macd_signal": D("9.054e-8"),
        "macd_hist": D("2.898e-8"),
    }
    for field, observed in observed_max.items():
        assert PROPOSED_TOLERANCES[field] > observed * 10, field


def test_a_missing_value_never_counts_as_within_tolerance():
    from agentic_trader.market.indicator_comparison import within_tolerance

    assert within_tolerance(ValueDelta("rsi_14", TS, None, D("50"))) is None
    assert within_tolerance(ValueDelta("rsi_14", TS, D("50"), D("50"))) is True
    assert within_tolerance(ValueDelta("rsi_14", TS, D("50"), D("60"))) is False


def test_decision_agreement_is_required_for_every_flag():
    """Written as "all of them" so a flag added later is required by default
    rather than silently exempt."""
    from agentic_trader.market.indicator_comparison import REQUIRED_DECISION_AGREEMENT

    assert frozenset(DecisionFlags.__dataclass_fields__) == REQUIRED_DECISION_AGREEMENT
    assert "macd_hist_rising" in REQUIRED_DECISION_AGREEMENT
    assert "rsi_improving" in REQUIRED_DECISION_AGREEMENT


def test_numeric_tolerance_alone_would_pass_a_direction_flip():
    """Why the two criteria are separate and why decision agreement is the
    binding one: this pair is inside tolerance and still disagrees."""
    from agentic_trader.market.indicator_comparison import within_tolerance

    delta = ValueDelta("macd_hist", TS, D("0.1000001"), D("0.0999999"))
    assert within_tolerance(delta) is True

    local = _flags(macd_hist=D("0.1000001"), macd_hist_prev=D("0.1000000"))
    broker = _flags(macd_hist=D("0.0999999"), macd_hist_prev=D("0.1000000"))
    assert "macd_hist_rising" in local.disagreements(broker)


# ================================================== the committed report


def test_the_validation_report_matches_the_current_contract():
    """The report is evidence for a go/no-go, so it must name the contract it
    was measured against. If the profile moves, the evidence is stale and this
    fails rather than letting a merged verdict outlive its inputs."""
    import json
    import pathlib

    from agentic_trader.market.acquisition import CURRENT_ACQUISITION
    from agentic_trader.market.local_indicators import required_bars

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_equivalence_2026-08-25.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["acquisition_profile_ref"] == CURRENT_ACQUISITION.profile_ref
    assert report["acquisition_config_fingerprint"] == (
        CURRENT_ACQUISITION.content_fingerprint
    )
    assert report["binding_requirement_bars"] == required_bars()
    assert report["verdict"] == "LOCAL_INDICATORS_EQUIVALENT"
    assert report["decision_equivalence"]["total_disagreements"] == 0
    assert report["inputs"]["symbol_count"] == 12
    assert report["inputs"]["bars_per_symbol"] >= required_bars()


def test_the_validation_report_carries_no_account_data():
    """It is committed to a public repository."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_equivalence_2026-08-25.json"
    )
    text = path.read_text(encoding="utf-8").lower()
    for token in ("account", "balance", "buying_power", "unsettled", "positions"):
        assert token not in text, token
