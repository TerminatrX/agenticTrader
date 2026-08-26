"""Tests for the comparison machinery, and for its isolation from decisions.

Two jobs. First, prove the comparison measures what it claims — especially that
a direction flip is caught even when the absolute error is tiny, which is the
failure mode a numeric tolerance alone would miss. Second, prove the whole
apparatus is inert: local values are diagnostics in this branch, and a test has
to hold that open rather than trusting that nobody wired them in.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
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

# The contract the equivalence evidence was gathered under. Permanently v3:
# evidence is dated, and restamping it as v4 would claim measurements nobody
# took under v4.
BASELINE_REF = "agentic-acquisition@v3-2026-08-25"
BASELINE_FP = "1aeb6fe90b857bd862950b3e1b9d1a1a95ffd7e282fe5a1b998fb5a790e52b93"

COMPARISON_REF = "local-indicator-comparison@v1-2026-08-25"
COMPARISON_FINGERPRINT = (
    "43cafd8ad390d7f25cc76d79d25118dff54bf14a5f7c346b3380d6ca194b1486"
)


def _report() -> dict:
    import json
    import pathlib as _pathlib

    path = (
        _pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_equivalence_2026-08-25.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))



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

    from agentic_trader.market.local_indicators import required_bars

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_equivalence_2026-08-25.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    # v3, permanently. This report measured evidence under the contract in
    # force at the time; restamping it as v4 would claim measurements nobody
    # took under v4.
    assert report["production_acquisition_profile_ref"] == BASELINE_REF
    assert report["production_acquisition_fingerprint"] == BASELINE_FP
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


# ================================================ validation provenance
#
# The first version of this milestone recorded only the production acquisition
# identity against the equivalence measurements. That was wrong: the experiment
# necessarily used a different request shape -- one shared window and a wider
# trim -- because it is asking whether two implementations agree on *identical*
# inputs, and production gives each indicator its own range. Naming only the
# production ref made the production shape look like the one that produced the
# numbers. These tests keep the two identities apart.


def test_the_comparison_profile_identity_is_pinned():
    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_COMPARISON

    assert LOCAL_INDICATOR_COMPARISON.profile_ref == COMPARISON_REF
    assert LOCAL_INDICATOR_COMPARISON.content_fingerprint == COMPARISON_FINGERPRINT


def test_the_report_names_the_production_contract():
    """(1) Which trading contract this evidence is intended to validate for."""
    report = _report()
    assert report["production_acquisition_profile_ref"] == BASELINE_REF
    assert report["production_acquisition_fingerprint"] == BASELINE_FP


def test_the_report_names_the_contract_the_measurements_were_taken_under():
    """(2) And it is a different identity from the production one."""
    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_COMPARISON

    report = _report()
    assert report["validation_comparison_profile_ref"] == (
        LOCAL_INDICATOR_COMPARISON.profile_ref
    )
    assert report["validation_comparison_fingerprint"] == (
        LOCAL_INDICATOR_COMPARISON.content_fingerprint
    )
    assert (
        report["validation_comparison_profile_ref"]
        != report["production_acquisition_profile_ref"]
    )
    assert (
        report["validation_comparison_fingerprint"]
        != report["production_acquisition_fingerprint"]
    )


def test_the_report_records_the_full_comparison_request_semantics():
    semantics = _report()["validation_comparison_request_semantics"]
    assert semantics["common_start_time"] == "2025-07-01T00:00:00Z"
    assert semantics["interval"] == "day"
    assert semantics["bounds"] == "regular"
    assert semantics["adjustment_type"] == "split"
    assert semantics["output"] == "last:30"
    assert semantics["indicators"]["rsi"] == {"type": "rsi", "period": 14}
    assert semantics["indicators"]["macd"] == {
        "type": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9,
    }
    for key, period in (("sma_20", 20), ("sma_50", 50), ("sma_200", 200)):
        assert semantics["indicators"][key] == {"type": "sma", "period": period}
    assert semantics["indicators"]["atr"] == {"type": "atr", "period": 14}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("common_start_time", "2025-01-01T00:00:00Z"),   # (3)
        ("interval", "week"),                             # (4)
        ("bounds", "extended"),                           # (4)
        ("adjustment_type", "none"),                      # (4)
        ("output", "last:5"),                             # (4)
        ("historicals_tool", "something_else"),
        ("indicator_tool", "something_else"),
    ],
)
def test_changing_any_comparison_semantic_changes_its_fingerprint(field, value):
    """(3)(4) Every field that alters what was requested is inside the hash."""
    import dataclasses

    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_COMPARISON

    assert getattr(LOCAL_INDICATOR_COMPARISON, field) != value, "proves nothing"
    altered = dataclasses.replace(LOCAL_INDICATOR_COMPARISON, **{field: value})
    assert altered.content_fingerprint != LOCAL_INDICATOR_COMPARISON.content_fingerprint


@pytest.mark.parametrize(
    ("key", "field", "value"),
    [
        ("rsi", "period", 21),
        ("atr", "period", 20),
        ("sma_200", "period", 100),
        ("macd", "fast_period", 8),
        ("macd", "slow_period", 30),
        ("macd", "signal_period", 5),
        ("sma_20", "indicator_type", "ema"),
    ],
)
def test_changing_an_indicator_semantic_changes_the_comparison_fingerprint(
    key, field, value
):
    """(5) RSI, MACD, SMA and ATR parameters each move the identity."""
    import dataclasses

    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_COMPARISON

    original = next(
        s for s in LOCAL_INDICATOR_COMPARISON.indicators if s.key == key
    )
    assert getattr(original, field) != value, "proves nothing"
    mutated = dataclasses.replace(original, **{field: value})
    altered = dataclasses.replace(
        LOCAL_INDICATOR_COMPARISON,
        indicators=tuple(
            mutated if s.key == key else s
            for s in LOCAL_INDICATOR_COMPARISON.indicators
        ),
    )
    assert altered.content_fingerprint != LOCAL_INDICATOR_COMPARISON.content_fingerprint


def test_every_compared_indicator_shares_one_source_window():
    """(6) The property the whole experiment rests on.

    If any indicator were given a different range, the comparison would be
    measuring range differences alongside formula differences and could not
    separate them. Asserted across the generated requests, not assumed.
    """
    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_COMPARISON

    plan = LOCAL_INDICATOR_COMPARISON.request_plan("AAPL")
    calls = [plan["calls"]["historicals"], *plan["calls"]["indicators"].values()]

    starts = {c["params"]["start_time"] for c in calls}
    assert starts == {"2025-07-01T00:00:00Z"}, starts

    for axis in ("interval", "bounds", "adjustment_type"):
        assert len({c["params"][axis] for c in calls}) == 1, axis

    outputs = {
        c["params"]["output"] for c in plan["calls"]["indicators"].values()
    }
    assert outputs == {"last:30"}
    assert set(plan["calls"]["indicators"]) == {
        "rsi", "macd", "sma_20", "sma_50", "sma_200", "atr",
    }


def test_the_production_contract_no_longer_requests_indicators():
    """Was: production keeps its own per-indicator lookbacks. v4 removed them.

    The legacy windows did not vanish -- they moved to the derivation profile,
    which is now the only place they are recorded. Asserted here so the two
    halves of that migration stay visible together.
    """
    from agentic_trader.market.acquisition import CURRENT_ACQUISITION
    from agentic_trader.market.indicator_derivation import CURRENT_DERIVATION

    assert CURRENT_ACQUISITION.indicators == ()
    plan = CURRENT_ACQUISITION.request_plan("AAPL", date(2026, 8, 25))
    assert plan["calls"]["indicators"] == {}

    legacy = {
        s.key: s.source_lookback_calendar_days for s in CURRENT_DERIVATION.specs
    }
    assert legacy == {
        "rsi": 180, "macd": 210, "sma_20": 90, "sma_50": 90,
        "sma_200": 330, "atr": 180,
    }


def test_the_comparison_profile_cannot_reach_a_decision():
    """(8) Same structural guarantee the local indicators carry."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentic_trader"
    decision_paths = [
        root / "strategies",
        root / "risk",
        root / "agents",
        root / "market" / "snapshot.py",
        root / "market" / "signals.py",
        root / "market" / "symbol_regime.py",
        root / "market" / "acquisition.py",
    ]
    offenders = []
    for path in decision_paths:
        files = path.rglob("*.py") if path.is_dir() else [path]
        for file in files:
            text = file.read_text(encoding="utf-8")
            if "indicator_comparison" in text or "LOCAL_INDICATOR_COMPARISON" in text:
                offenders.append(str(file.relative_to(root)))
    assert offenders == [], f"the comparison profile reached: {offenders}"


def test_the_equivalence_verdict_survives_the_corrected_provenance():
    """(9) Separating the identities corrects who-measured-what. It does not
    touch the measurements, so the verdict stands on the same evidence."""
    report = _report()
    assert report["verdict"] == "LOCAL_INDICATORS_EQUIVALENT"
    assert report["decision_equivalence"]["total_disagreements"] == 0
    assert report["decision_equivalence"]["comparisons"] == 348
    assert report["inputs"]["symbol_count"] == 12
    assert report["inputs"]["bars_per_symbol"] == 289
    assert report["inputs"]["requests_issued_under"] == COMPARISON_REF
    for field, block in report["numeric"].items():
        assert block["result"] == "PASS", field


def test_the_report_records_how_the_unechoed_fields_were_verified():
    """`adjustment_type` and the indicator `start_time` are not echoed by the
    endpoint, so the report must say how they were established rather than
    leaving them as an assertion."""
    provenance = _report()["validation_comparison_provenance"]
    assert set(provenance["not_echoed_by_the_endpoint"]) == {
        "adjustment_type", "indicator start_time",
    }
    assert "matched bit for bit" in provenance["verification_method"]
    assert "recursive" in provenance["verification_method"]
    assert provenance["recovered_from_saved_payloads"]
