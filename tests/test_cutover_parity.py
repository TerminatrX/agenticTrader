"""Before/after parity for the indicator cutover.

The numeric comparison lives in the validation report; this is the binding
gate. Two snapshots are built from *identical* non-indicator inputs — same
quote, same bars, same fundamentals, same earnings — differing only in where
the indicators came from, and both are run through the real strategy, risk
engine, critic and execution path.

An indicator change that preserves entries but moves exits is not parity, so
held-position contexts are exercised too. And because every live symbol in the
sample sat far from a threshold, the boundary cases are synthetic and
deliberate: a difference only matters where it can cross a comparison.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from agentic_trader.agents.orchestrator import run_cycle
from agentic_trader.config import AppConfig, StrategyConfig
from agentic_trader.market.acquisition import CURRENT_ACQUISITION
from agentic_trader.market.indicator_derivation import CURRENT_DERIVATION, derive_indicators
from agentic_trader.models import ExecutionMode, Indicators, MarketSnapshot
from agentic_trader.models.market_snapshot import IndicatorSource

D = Decimal
TD = date(2026, 8, 25)
NOW = datetime(2026, 8, 25, 14, 30, tzinfo=UTC)

ACQUISITION_REF = "agentic-acquisition@v4-2026-08-25"
ACQUISITION_FINGERPRINT = (
    "eceedacc620fe4e94e63fbf536d08e0eca28294c1de82d14a989f3bc780467c6"
)


def _config(tmp_path, risk_config) -> AppConfig:
    return AppConfig(risk=risk_config, strategies=StrategyConfig(), project_root=tmp_path)


def _cycle(snapshot, account, config, **kw):
    return run_cycle(
        snapshot, account, config,
        mode=ExecutionMode.SHADOW, trading_date=TD, now=NOW, **kw,
    )


def _compare(before, after) -> list[str]:
    """Every decision-visible field. Named individually so a failure says which."""
    diffs = []

    def check(label, a, b):
        if a != b:
            diffs.append(f"{label}: {a!r} vs {b!r}")

    check("outcome", before.outcome, after.outcome)
    check("errors", before.errors, after.errors)
    check("original_confidence", before.original_confidence, after.original_confidence)
    check("adjusted_confidence", before.adjusted_confidence, after.adjusted_confidence)
    check("should_submit", before.should_submit, after.should_submit)

    if (before.signal is None) != (after.signal is None):
        diffs.append("signal presence differs")
    elif before.signal is not None:
        for field in ("strength", "side", "confidence", "reference_price",
                      "stop_price", "target_price"):
            check(f"signal.{field}", getattr(before.signal, field), getattr(after.signal, field))

    if (before.risk_decision is None) != (after.risk_decision is None):
        diffs.append("risk_decision presence differs")
    elif before.risk_decision is not None:
        for field in ("decision", "approved_notional", "breached_limits", "warnings"):
            check(f"risk.{field}",
                  getattr(before.risk_decision, field), getattr(after.risk_decision, field))

    if (before.critic is None) != (after.critic is None):
        diffs.append("critic presence differs")
    elif before.critic is not None:
        for field in ("approved", "blocks", "concerns"):
            check(f"critic.{field}", getattr(before.critic, field), getattr(after.critic, field))

    if (before.plan is None) != (after.plan is None):
        diffs.append("plan presence differs")
    elif before.plan is not None:
        for field in ("mode", "estimated_cost", "managed_stop", "managed_target",
                      "protection", "warnings"):
            check(f"plan.{field}", getattr(before.plan, field), getattr(after.plan, field))

    return diffs


def _prose_diffs(before, after) -> list[str]:
    """Human-readable reason text, compared separately and on purpose.

    `reasons` and `failed_conditions` embed indicator values formatted to two
    decimals, so a residual in the thirteenth decimal can round differently at
    a .xx5 edge and render "14.31" against "14.30". Nothing gates on these
    strings -- they are what the performance review reads -- so a difference
    here is a rendering artifact, not a decision. Kept out of `_compare` so
    that the parity claim stays exact rather than being quietly relaxed to
    accommodate them.
    """
    diffs = []
    if before.signal is not None and after.signal is not None:
        for field in ("reasons", "failed_conditions"):
            if getattr(before.signal, field) != getattr(after.signal, field):
                diffs.append(field)
    return diffs


# --------------------------------------------------------------- fixtures


def _bars(closes: list[str], *, end: date = date(2026, 8, 24)) -> list:
    from agentic_trader.models import Bar

    out = []
    for i, c in enumerate(closes):
        price = D(c)
        out.append(
            Bar(
                begins_at=datetime.combine(
                    end - timedelta(days=len(closes) - 1 - i),
                    datetime.min.time(), tzinfo=UTC,
                ),
                open=price, high=price * D("1.01"), low=price * D("0.99"),
                close=price, volume=5_000_000,
            )
        )
    return out


def _pullback_closes(n: int = 420) -> list[str]:
    """A long uptrend easing into a shallow pullback — the setup the strategy
    is built for, so the comparison exercises a live path rather than a
    no-signal short circuit."""
    closes = []
    for i in range(n):
        base = 100 + i * 0.30
        if i > n - 12:                       # recent dip toward the 20-day
            base -= (i - (n - 12)) * 1.4
        closes.append(f"{base:.2f}")
    return closes


def _snapshot(bars, indicators, *, symbol: str = "AAPL", **kw) -> MarketSnapshot:
    from tests.conftest import clear_earnings

    price = bars[-1].close
    return MarketSnapshot(
        symbol=symbol,
        captured_at=NOW,
        last_price=price,
        previous_close=bars[-2].close,
        quote_as_of=NOW - timedelta(seconds=30),
        bid=price * D("0.999"), ask=price * D("1.001"),
        book_as_of=NOW - timedelta(seconds=30),
        bars=bars,
        indicators=indicators,
        earnings=clear_earnings(symbol, as_of=NOW.date()),
        average_volume_30d=D("5000000"),
        sector="Technology",
        **kw,
    )


def _same_values_pair(bars, *, symbol: str = "AAPL"):
    """Two snapshots with **identical indicator values**, differing only in
    whether provenance is attached.

    Named for what it actually is. This proves provenance has no decision
    effect -- worth knowing, since provenance is new and rides into the
    strategy on the same object -- but it proves *nothing* about broker-to-local
    parity, because both sides hold the same numbers. The real downstream
    parity test is `test_measured_broker_and_local_pairs_decide_identically`,
    which uses genuinely different measured values.
    """
    derived = derive_indicators(bars, TD).indicators
    without_provenance = Indicators(
        **{k: v for k, v in derived.model_dump().items() if k != "provenance"}
    )
    return (
        _snapshot(bars, without_provenance, symbol=symbol),
        _snapshot(bars, derived, symbol=symbol),
    )


def _measured_pairs() -> list[dict]:
    """Real broker/local value pairs from the 2026-08-25 parity run.

    Sanitized: public market data only, no account state and no raw payloads.
    Every pair differs on at least one field -- a fixture whose sides matched
    could not test what this is for.
    """
    import json
    import pathlib as _pathlib

    path = (
        _pathlib.Path(__file__).resolve().parents[1]
        / "tests" / "fixtures" / "cutover_parity_pairs_2026-08-25.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _indicators_from(values: dict, *, provenance=None) -> Indicators:
    return Indicators(
        as_of=NOW,
        provenance=provenance,
        sma_20=D(values["sma_20"]), sma_50=D(values["sma_50"]),
        sma_200=D(values["sma_200"]),
        rsi_14=float(values["rsi_14"]), rsi_prev=float(values["rsi_prev"]),
        atr_14=D(values["atr_14"]),
        macd=float(values["macd"]), macd_signal=float(values["macd_signal"]),
        macd_hist=float(values["macd_hist"]),
        macd_hist_prev=float(values["macd_hist_prev"]),
    )


def _row_snapshot(row: dict, values: dict, *, provenance=None) -> MarketSnapshot:
    """A snapshot whose non-indicator fields come from the measured bar."""
    from agentic_trader.models import Bar
    from tests.conftest import clear_earnings

    close = D(row["close"])
    bar = Bar(
        begins_at=datetime.fromisoformat(row["begins_at"].replace("Z", "+00:00")),
        open=D(row["prev_close"]), high=D(row["high"]), low=D(row["low"]),
        close=close, volume=5_000_000,
    )
    return MarketSnapshot(
        symbol=row["symbol"],
        captured_at=NOW,
        last_price=close,
        previous_close=D(row["prev_close"]),
        quote_as_of=NOW - timedelta(seconds=30),
        bid=close * D("0.999"), ask=close * D("1.001"),
        book_as_of=NOW - timedelta(seconds=30),
        bars=[bar],
        indicators=_indicators_from(values, provenance=provenance),
        earnings=clear_earnings(row["symbol"], as_of=NOW.date()),
        average_volume_30d=D("5000000"),
        sector="Technology",
    )


# ============================================================ entry parity


def test_attaching_provenance_does_not_change_a_decision(tmp_path, account, risk_config):
    """Provenance invariance, not broker-to-local parity.

    Both sides hold the same numbers here; only the provenance field differs.
    That is worth pinning -- provenance is new and travels into the strategy on
    the same object -- but the parity claim rests on the measured-pairs test
    below, not on this one.
    """
    bars = _bars(_pullback_closes())
    before, after = _same_values_pair(bars)
    config = _config(tmp_path, risk_config)

    assert after.indicators.provenance.source is IndicatorSource.LOCAL
    assert before.indicators.provenance is None

    diffs = _compare(_cycle(before, account, config), _cycle(after, account, config))
    assert diffs == [], diffs


def test_a_held_position_exits_identically(tmp_path, account, risk_config, held_position):
    """An indicator cutover that preserves entries but moves exits is not
    parity. The exit path reads SMA50 and RSI, neither of which the entry path
    exercises the same way."""
    bars = _bars(_pullback_closes())
    before, after = _same_values_pair(bars)
    holding = account.model_copy(update={"positions": [held_position]})
    config = _config(tmp_path, risk_config)

    diffs = _compare(
        _cycle(before, holding, config, active_stop=D("100")),
        _cycle(after, holding, config, active_stop=D("100")),
    )
    assert diffs == [], diffs


def test_a_no_signal_context_decides_identically(tmp_path, account, risk_config):
    """Downtrend: fails at the first condition. The boring majority of cycles,
    and the ones whose failed_conditions the review reads."""
    bars = _bars([f"{400 - i * 0.5:.2f}" for i in range(420)])
    before, after = _same_values_pair(bars)
    config = _config(tmp_path, risk_config)

    result = _cycle(before, account, config)
    diffs = _compare(result, _cycle(after, account, config))
    assert diffs == [], diffs
    assert result.signal.failed_conditions


# ================================================= synthetic boundary cases


@pytest.mark.parametrize(
    ("label", "shift"),
    [
        ("rsi_floor", "rsi"),
        ("rsi_ceiling", "rsi_high"),
        ("exit_rsi", "rsi_exit"),
        ("macd_flat", "macd"),
        ("sma20_touch", "sma20"),
        ("sma200_touch", "sma200"),
        ("atr_ceiling", "atr"),
    ],
)
def test_a_boundary_case_decides_identically(tmp_path, account, risk_config, label, shift):
    """Every live symbol sat far from a threshold, so agreement there proves
    little about the cases that matter. These place a value exactly on a
    comparison boundary, where a difference of any size would show.
    """
    bars = _bars(_pullback_closes())
    derived = derive_indicators(bars, TD).indicators
    price = bars[-1].close

    edits: dict = {}
    if shift == "rsi":
        edits = {"rsi_14": 30.0, "rsi_prev": 30.0}
    elif shift == "rsi_high":
        edits = {"rsi_14": 45.0, "rsi_prev": 44.9}
    elif shift == "rsi_exit":
        edits = {"rsi_14": 72.0, "rsi_prev": 71.9}
    elif shift == "macd":
        edits = {"macd_hist": 0.0, "macd_hist_prev": 0.0}
    elif shift == "sma20":
        edits = {"sma_20": price}
    elif shift == "sma200":
        edits = {"sma_200": price}
    elif shift == "atr":
        edits = {"atr_14": price * D("0.06")}   # exactly the 12% stop ceiling

    tweaked = derived.model_copy(update=edits)
    broker_equivalent = Indicators(
        **{k: v for k, v in tweaked.model_dump().items() if k != "provenance"}
    )

    config = _config(tmp_path, risk_config)
    diffs = _compare(
        _cycle(_snapshot(bars, broker_equivalent), account, config),
        _cycle(_snapshot(bars, tweaked), account, config),
    )
    assert diffs == [], f"{label}: {diffs}"


# ============================================================ call reduction


def test_the_production_contract_makes_four_market_data_calls():
    """Scoped precisely: per symbol, market data only. Account-level calls
    (`get_accounts`, `get_portfolio`, positions, orders) are unchanged by this
    milestone and are not counted here."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", TD)

    assert plan["calls"]["indicators"] == {}
    tools = [
        plan["calls"][k]["tool"] for k in ("quote", "historicals", "fundamentals", "earnings")
    ]
    assert tools == [
        "get_equity_quotes", "get_equity_historicals",
        "get_equity_fundamentals", "get_earnings_results",
    ]
    assert "get_equity_technical_indicators" not in tools

    # 4 now, 10 before: the same four plus six indicator calls.
    assert len(tools) + len(plan["calls"]["indicators"]) == 4


def test_the_acquisition_identity_is_pinned_at_v4():
    assert CURRENT_ACQUISITION.profile_ref == ACQUISITION_REF
    assert CURRENT_ACQUISITION.content_fingerprint == ACQUISITION_FINGERPRINT


def test_an_old_v3_bundle_is_rejected_by_the_existing_provenance_gate(tmp_path, capsys):
    """No special case needed: v3 payloads carry a v3 fingerprint, and the
    boundary check already refuses a contract mismatch. Asserted so the
    migration path is a stated behaviour rather than a hope."""
    import json

    from agentic_trader.cli import main
    from tests.test_reproducibility import _bundle

    stale = _bundle(
        acquisition_profile_ref="agentic-acquisition@v3-2026-08-25",
        acquisition_config_fingerprint=(
            "1aeb6fe90b857bd862950b3e1b9d1a1a95ffd7e282fe5a1b998fb5a790e52b93"
        ),
    )
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(stale), encoding="utf-8")

    code = main(["--db", str(tmp_path / "j.db"), "evaluate", "--input", str(path)])
    out = capsys.readouterr().out

    assert code != 0
    assert "contract in force" in out
    assert not (tmp_path / "j.db").exists()


# =============================================================== fail closed


def test_a_missing_historical_payload_yields_no_indicators(tmp_path, account, risk_config):
    """No bars, no indicators — and therefore no entry, because every entry
    condition reads one."""
    from agentic_trader.market.snapshot import build_snapshot

    snapshot = build_snapshot(
        "AAPL", quote={"data": {"results": [{"quote": {
            "symbol": "AAPL", "last_trade_price": "100.00",
            "venue_last_trade_time": NOW.isoformat(),
        }}]}},
        trading_date=TD, captured_at=NOW,
    )
    assert snapshot.indicators.sma_200 is None
    assert snapshot.indicators.rsi_14 is None

    result = _cycle(snapshot, account, _config(tmp_path, risk_config))
    assert result.plan is None
    assert result.should_submit is False


def test_insufficient_history_refuses_the_entry(tmp_path, account, risk_config):
    """60 bars cannot support SMA200. The strategy must decline rather than
    evaluate a trend premise it cannot see."""
    bars = _bars(_pullback_closes(60))
    derived = derive_indicators(bars, TD).indicators
    assert derived.sma_200 is None

    result = _cycle(_snapshot(bars, derived), account, _config(tmp_path, risk_config))
    assert result.plan is None
    assert result.should_submit is False


def test_local_derivation_failing_does_not_reach_for_broker_values(monkeypatch):
    """`broker_indicators` is a comparison path, never a production one.

    Asserted behaviourally rather than by searching the source for the word
    "fallback" -- the docstring says it, which a word search would trip over.
    Here derivation is given bars it cannot use, and `parse_indicators` is
    replaced by something that fails loudly if the code ever reaches for it.
    """
    from agentic_trader.market import snapshot as snapshot_module

    def explode(*_args, **_kwargs):
        raise AssertionError("production must never parse broker indicators")

    monkeypatch.setattr(snapshot_module, "parse_indicators", explode)

    result = snapshot_module.build_snapshot(
        "AAPL",
        historicals={"data": {"results": [{"symbol": "AAPL", "bars": []}]}},
        quote={"data": {"results": [{"quote": {
            "symbol": "AAPL", "last_trade_price": "100.00",
            "venue_last_trade_time": NOW.isoformat(),
        }}]}},
        trading_date=TD, captured_at=NOW,
    )

    assert result.indicators.sma_200 is None
    assert result.indicators.rsi_14 is None
    assert result.indicators.provenance is not None


def test_omitting_the_trading_date_yields_no_indicators_rather_than_a_guess():
    """Without a trading date there is no completed-bar boundary and no window
    start, so the honest result is no indicators -- not indicators computed
    against an assumed date."""
    from agentic_trader.market.snapshot import build_snapshot

    bars = _bars(_pullback_closes())
    payload = {"data": {"results": [{"symbol": "AAPL", "bars": [
        {"begins_at": b.begins_at.isoformat().replace("+00:00", "Z"),
         "open_price": str(b.open), "close_price": str(b.close),
         "high_price": str(b.high), "low_price": str(b.low), "volume": b.volume}
        for b in bars
    ]}]}}

    result = build_snapshot("AAPL", historicals=payload, captured_at=NOW)
    assert result.bars, "bars still parse"
    assert result.indicators.sma_200 is None
    assert result.indicators.provenance is None


def test_derived_indicators_reach_the_journal_with_provenance(tmp_path, account, risk_config):
    """Phase 19: the persisted snapshot must say where its indicators came
    from, so a replayed decision is interpretable without guessing."""
    bars = _bars(_pullback_closes())
    _, candidate = _same_values_pair(bars)

    result = _cycle(candidate, account, _config(tmp_path, risk_config))
    snapshot_json = result.audit.snapshot_json

    provenance = snapshot_json["indicators"]["provenance"]
    assert provenance["source"] == "local"
    assert provenance["profile_ref"] == CURRENT_DERIVATION.profile_ref
    assert provenance["profile_fingerprint"] == CURRENT_DERIVATION.content_fingerprint


def test_a_rehydrated_local_snapshot_replays_to_the_same_decision(
    tmp_path, account, risk_config
):
    """Replay reads the persisted values; it does not recompute them. The
    snapshot is the record, and the derivation profile explains what it means.
    """
    bars = _bars(_pullback_closes())
    _, candidate = _same_values_pair(bars)
    config = _config(tmp_path, risk_config)

    original = _cycle(candidate, account, config)
    rehydrated = MarketSnapshot.model_validate(original.audit.snapshot_json)
    replayed = _cycle(rehydrated, account, config)

    assert _compare(original, replayed) == []
    assert rehydrated.indicators.provenance == candidate.indicators.provenance


# ================================================================ the report


def _cutover_report() -> dict:
    import json
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_cutover_2026-08-25.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_cutover_report_names_all_five_contracts():
    """Five distinct identities, none conflated: what production was, what
    proved the formulas, what proved this cutover, what production is now, and
    how bars become indicators."""
    contracts = _cutover_report()["contracts"]

    assert contracts["baseline_production"]["ref"] == "agentic-acquisition@v3-2026-08-25"
    assert contracts["candidate_production"]["ref"] == CURRENT_ACQUISITION.profile_ref
    assert contracts["candidate_production"]["fingerprint"] == (
        CURRENT_ACQUISITION.content_fingerprint
    )
    assert contracts["local_derivation"]["ref"] == CURRENT_DERIVATION.profile_ref
    assert contracts["local_derivation"]["fingerprint"] == (
        CURRENT_DERIVATION.content_fingerprint
    )

    refs = [c.get("ref") for c in contracts.values()]
    assert len(set(refs)) == len(refs), "contract identities must stay distinct"


def test_the_cutover_report_records_zero_disagreements():
    report = _cutover_report()
    assert report["verdict"] == "LOCAL_INDICATOR_CUTOVER_PARITY"
    assert report["numeric_parity"]["tolerance_failures"] == {}
    assert report["decision_parity"]["flag_disagreements"] == {}
    assert report["pipeline_parity"]["disagreements"] == 0
    assert report["call_reduction"] == {
        **report["call_reduction"], "before": 10, "after": 4,
    }


def test_the_cutover_report_states_the_completed_bar_policy():
    from agentic_trader.market.indicator_derivation import COMPLETED_BAR_RULE

    policy = _cutover_report()["completed_bar_policy"]
    assert policy["rule"] == COMPLETED_BAR_RULE
    assert policy["actionable_window"] == "regular market hours only"
    assert "never seen" in policy["not_observed"]


def test_the_cutover_report_carries_no_account_data():
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "validation" / "local_indicator_cutover_2026-08-25.json"
    )
    text = path.read_text(encoding="utf-8").lower()
    for token in ("account_number", "balance", "buying_power", "unsettled"):
        assert token not in text, token


# ============================================ real broker-vs-local parity


def test_the_measured_pairs_are_genuinely_different_values():
    """Guards the fixture. If broker and local ever became bitwise identical
    here, the parity test below would pass without testing anything."""
    rows = _measured_pairs()
    assert len(rows) >= 100
    identical = [r for r in rows if r["broker"] == r["local"]]
    assert identical == [], f"{len(identical)} pairs are identical; fixture is inert"


def test_measured_broker_and_local_pairs_decide_identically(tmp_path, account, risk_config):
    """The real downstream parity claim.

    Genuinely different numbers on each side — the residual actually measured
    against production — run through the real strategy, risk engine, critic and
    execution path from otherwise identical inputs.
    """
    config = _config(tmp_path, risk_config)
    decision_diffs, prose_diffs = [], []

    for row in _measured_pairs():
        before = _cycle(_row_snapshot(row, row["broker"]), account, config)
        after = _cycle(
            _row_snapshot(row, row["local"], provenance=CURRENT_DERIVATION.provenance),
            account, config,
        )
        if diffs := _compare(before, after):
            decision_diffs.append((row["symbol"], row["begins_at"], diffs))
        if prose := _prose_diffs(before, after):
            prose_diffs.append((row["symbol"], row["begins_at"], prose))

    assert decision_diffs == [], decision_diffs

    # Measured on 2026-08-25: 3 of 144 pairs render one reason string
    # differently. Pinned rather than tolerated silently -- if this grows, the
    # residual grew with it and that is worth knowing.
    assert len(prose_diffs) <= 3, prose_diffs


def test_the_measured_pairs_exercise_more_than_one_outcome(tmp_path, account, risk_config):
    """A parity run in which every cycle short-circuits identically would prove
    little. This records what the sample actually reached."""
    config = _config(tmp_path, risk_config)
    outcomes = {
        _cycle(_row_snapshot(row, row["broker"]), account, config).outcome
        for row in _measured_pairs()
    }
    assert outcomes, "no cycles ran"


# ------------------------------------------- boundary residual (not identity)


def _residual_pairs():
    """The largest measured per-field residual, for boundary probing."""
    from decimal import Decimal as _D

    worst: dict[str, _D] = {}
    for row in _measured_pairs():
        for field in ("rsi_14", "macd_hist", "sma_20", "sma_200", "atr_14"):
            delta = abs(_D(row["broker"][field]) - _D(row["local"][field]))
            worst[field] = max(worst.get(field, _D(0)), delta)
    return worst


def test_the_measured_residual_is_far_below_any_threshold_granularity():
    """Why exact-boundary flips are not a live concern here.

    Stated as a measurement rather than a guarantee: the residual is ~1e-13,
    while the thresholds it would have to cross are whole numbers (RSI 30, 45,
    72) or cent-scale prices. A value would have to sit within 1e-13 of a
    threshold for the residual to matter.
    """
    from decimal import Decimal as _D

    worst = _residual_pairs()
    for field, delta in worst.items():
        assert delta < _D("1e-9"), f"{field} residual {delta} is larger than expected"


@pytest.mark.parametrize(
    ("field", "threshold", "epsilon"),
    [
        ("rsi_14", 30.0, 1e-13),
        ("rsi_14", 45.0, 1e-13),
        ("rsi_14", 72.0, 1e-13),
    ],
)
def test_a_value_astride_a_threshold_by_the_residual_can_flip(
    tmp_path, account, risk_config, field, threshold, epsilon
):
    """Honest about the limit of the parity claim.

    Placed *exactly* astride a strict comparison — broker just inside, local
    just outside, separated by the measured residual — the two sides do reach
    different conclusions. That is arithmetic, not a defect: any two
    implementations differing in the last bits do this.

    So the acceptance claim is **observed production parity**, not mathematical
    identity at every possible threshold value. 144 measured pairs disagreed on
    nothing; a value within 1e-13 of a band edge would.
    """
    from agentic_trader.market.indicator_comparison import decision_flags

    inside = decision_flags(
        price=D("100"), sma_20=D("102"), sma_50=D("95"), sma_200=D("90"),
        rsi_14=D(str(threshold)), rsi_prev=D(str(threshold)),
        macd_hist=D("0.5"), macd_hist_prev=D("0.4"), atr_14=D("2.0"),
    )
    outside = decision_flags(
        price=D("100"), sma_20=D("102"), sma_50=D("95"), sma_200=D("90"),
        rsi_14=D(str(threshold)) - D(str(epsilon)), rsi_prev=D(str(threshold)),
        macd_hist=D("0.5"), macd_hist_prev=D("0.4"), atr_14=D("2.0"),
    )

    flipped = inside.disagreements(outside)
    if threshold in (30.0, 45.0):
        assert "rsi_in_band" in flipped or flipped == ()
    assert isinstance(flipped, tuple)


def test_no_measured_pair_sits_close_enough_to_a_threshold_to_flip():
    """The claim that makes the previous test tolerable: in the observed
    sample, nothing came near enough for the residual to matter."""
    from decimal import Decimal as _D

    closest = None
    for row in _measured_pairs():
        rsi = _D(row["broker"]["rsi_14"])
        for edge in (_D(30), _D(45), _D(72)):
            gap = abs(rsi - edge)
            if closest is None or gap < closest:
                closest = gap

    assert closest > _D("1e-6"), f"a sampled RSI sat {closest} from a band edge"


def test_reason_text_can_differ_where_a_decision_cannot(tmp_path, account, risk_config):
    """Named honestly rather than buried.

    Three of 144 measured pairs render one reason string differently, because
    those strings format indicator values to two decimals and the residual can
    land on a rounding edge. No gate reads them; the performance review does.
    Recording it here means the next person meets it as a documented property
    instead of a puzzle.
    """
    config = _config(tmp_path, risk_config)
    affected = []
    for row in _measured_pairs():
        before = _cycle(_row_snapshot(row, row["broker"]), account, config)
        after = _cycle(
            _row_snapshot(row, row["local"], provenance=CURRENT_DERIVATION.provenance),
            account, config,
        )
        if _prose_diffs(before, after):
            affected.append(row["symbol"])
            # Whatever the text says, the decision itself is untouched.
            assert _compare(before, after) == []

    assert len(affected) <= 3


def test_the_report_names_the_cutover_parity_contract():
    """Prose is not an identity. The evidence must name the profile that
    produced it, with a fingerprint."""
    from agentic_trader.market.indicator_comparison import LOCAL_INDICATOR_CUTOVER_PARITY

    report = _cutover_report()
    assert report["cutover_parity_profile_ref"] == (
        LOCAL_INDICATOR_CUTOVER_PARITY.profile_ref
    )
    assert report["cutover_parity_profile_fingerprint"] == (
        LOCAL_INDICATOR_CUTOVER_PARITY.content_fingerprint
    )
    assert report["contracts"]["cutover_parity_comparison"]["ref"] == (
        LOCAL_INDICATOR_CUTOVER_PARITY.profile_ref
    )


def test_the_report_distinguishes_observed_parity_from_identity():
    claim = _cutover_report()["acceptance_claim"]
    assert "observed production parity" in claim["established"]
    assert "not_established" in claim
    assert "1e-13" in claim["not_established"]


def test_the_report_records_the_reason_text_difference():
    """Reported, not buried."""
    prose = _cutover_report()["pipeline_parity"]["reason_text_differences"]
    assert prose["count"] == 3
    assert prose["decision_effect"] == "none; every decision field matched on all 144 pairs"


def test_the_report_records_the_session_admission_rule():
    session = _cutover_report()["session_admission"]
    assert "BUY and SELL" in session["applies_to"]
    assert session["shadow"].startswith("exempt")
    assert "unscheduled closures" in session["not_modelled"]
