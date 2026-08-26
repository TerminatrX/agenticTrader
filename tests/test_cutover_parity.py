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
                      "stop_price", "target_price", "failed_conditions", "reasons"):
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


def _pair(bars, *, symbol: str = "AAPL"):
    """Baseline and candidate snapshots differing only in indicator source.

    The baseline stands in for the retired broker payloads: the same values,
    carried without local provenance, exactly as `parse_indicators` would have
    produced them.
    """
    derived = derive_indicators(bars, TD).indicators
    broker_equivalent = Indicators(
        **{
            k: v for k, v in derived.model_dump().items()
            if k != "provenance"
        }
    )
    return (
        _snapshot(bars, broker_equivalent, symbol=symbol),
        _snapshot(bars, derived, symbol=symbol),
    )


# ============================================================ entry parity


def test_an_entry_context_decides_identically(tmp_path, account, risk_config):
    bars = _bars(_pullback_closes())
    before, after = _pair(bars)
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
    before, after = _pair(bars)
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
    before, after = _pair(bars)
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
    _, candidate = _pair(bars)

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
    _, candidate = _pair(bars)
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
