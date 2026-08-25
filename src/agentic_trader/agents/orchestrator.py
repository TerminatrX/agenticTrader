"""One evaluation cycle, start to finish, as a pure function.

Despite the name, this is not a loop and it owns no schedule. The agent drives
the loop: it fetches data through MCP tools, calls `run_cycle`, and — only if
the result says so, and only after the critic and any human gate agree —
submits the order. `run_cycle` itself is deterministic and side-effect-free, so
the same inputs always produce the same decision.

"The same inputs" includes the clock. `now` reaches the risk gate's `as_of`,
the critic, and preflight quote-age and drift, so a past cycle replays only
when it is given the instant it originally ran -- persisted as
`AuditEntry.occurred_at`, alongside the snapshot and the trading date. Passing
the snapshot alone, or substituting `captured_at`, re-decides rather than
replays.

The pipeline:

    snapshot + account
        -> strategy.evaluate      (an opinion)
        -> RiskEngine.evaluate    (a sized, permitted order)
        -> critique               (an independent re-derivation)
        -> build_order_payload    (an exact broker payload)
        -> CycleResult            (decision + audit trail)

Any stage may end the cycle. Whatever happens, an `AuditEntry` is produced —
including for the boring majority of cycles that decide to do nothing, since
those are what tell you later whether the filters were working.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

from agentic_trader.agents.critic import CriticReport, critique
from agentic_trader.config import AppConfig
from agentic_trader.execution.executor import ExecutionPlan, PreflightError, build_order_payload
from agentic_trader.execution.shadow_executor import ShadowExecutor
from agentic_trader.journal.models import AuditEntry, CycleOutcome, TradeRecord
from agentic_trader.market.acquisition import (
    CURRENT_ACQUISITION,
    MarketDataAcquisitionProfile,
)
from agentic_trader.market.market_regime import MarketContext
from agentic_trader.models import (
    SELECTABLE_EXECUTION_MODES,
    AccountState,
    ExecutionMode,
    MarketSnapshot,
    RiskDecision,
    Signal,
    SignalStrength,
)
from agentic_trader.risk.engine import RiskEngine
from agentic_trader.strategies.base import StrategyContext, get_strategy

# How an ExecutionMode is expressed in the executor's own vocabulary.
#
# Exhaustive and total: a mode absent from this table has no plan mode, and
# `run_cycle` refuses rather than choosing one. The previous form --
# `"shadow" if mode is SHADOW else "live"` -- was fail-closed against the
# protection floor but fail-*open* against enablement: adding APPROVAL to
# SELECTABLE_EXECUTION_MODES would, with no other edit, have handed it a
# `mode="live"` plan. That is a one-line accidental go-live, which is exactly
# the shape of change this project must not leave lying around.
_PLAN_MODE: dict[ExecutionMode, Literal["live", "shadow"]] = {
    ExecutionMode.SHADOW: "shadow",
    ExecutionMode.LIVE: "live",
}


@dataclass
class CycleResult:
    """Everything one symbol's evaluation produced."""

    cycle_id: str
    symbol: str
    strategy: str
    outcome: CycleOutcome
    # Neither has a default. A CycleResult that has not been told its mode or
    # its acquisition date must not be constructible, because `_build_audit`
    # copies both straight onto the persisted record -- a default would put an
    # unexamined value into the journal for a cycle nobody confirmed.
    mode: ExecutionMode
    trading_date: date
    acquisition: MarketDataAcquisitionProfile = CURRENT_ACQUISITION

    signal: Signal | None = None
    risk_decision: RiskDecision | None = None
    critic: CriticReport | None = None
    plan: ExecutionPlan | None = None
    trade: TradeRecord | None = None
    audit: AuditEntry | None = None
    errors: list[str] = field(default_factory=list)

    # Both recorded so the critic's effect on position size is measurable
    # rather than invisible — the performance review reads the gap between them.
    original_confidence: float | None = None
    adjusted_confidence: float | None = None

    # Recorded for later analysis; nothing in this cycle branches on it.
    market_context: MarketContext | None = None

    @property
    def should_submit(self) -> bool:
        """True only when every gate agreed and the mode is live.

        Both mode readings must agree. `self.mode` is the context the caller
        asked for; `plan.mode` is what the payload was actually built under.
        Requiring both means a disagreement between them -- a mapping bug, a
        hand-built CycleResult, a future mode slipping through -- resolves to
        "do not submit" rather than to whichever field a reader happened to
        check. Shadow is excluded outright: a shadow cycle has already recorded
        its simulated fill and must never reach the broker.
        """
        return (
            self.mode is ExecutionMode.LIVE
            and self.plan is not None
            and self.plan.mode == "live"
            and self.critic is not None
            and self.critic.approved
        )

    def headline(self) -> str:
        if self.plan is not None:
            return f"{self.symbol}: {self.plan.summary()}"
        if self.signal is not None and self.signal.strength is not SignalStrength.NONE:
            return f"{self.symbol}: {self.signal.strength.value} ({self.outcome.value})"
        return f"{self.symbol}: no action ({self.outcome.value})"


def run_cycle(
    snapshot: MarketSnapshot,
    account: AccountState,
    config: AppConfig,
    *,
    strategy_name: str = "trend_pullback",
    mode: ExecutionMode | str = ExecutionMode.SHADOW,
    trading_date: date,
    cycle_id: str | None = None,
    acquisition: MarketDataAcquisitionProfile = CURRENT_ACQUISITION,
    known_client_keys: set[str] | None = None,
    last_loss_exit: date | None = None,
    recent_symbol_trades: int = 0,
    active_stop: Decimal | None = None,
    market_context: MarketContext | None = None,
    now: datetime | None = None,
) -> CycleResult:
    """Evaluate one symbol under one strategy.

    `trading_date` is required and has no default. It is the date the
    acquisition profile built every `start_time` from, so it decided which
    bars the broker computed the inputs over. Defaulting it to today would
    make a replayed cycle claim inputs it never had.
    """
    current = now or datetime.now(UTC)
    cid = cycle_id or uuid.uuid4().hex[:12]

    # Normalized once, at the entry point, so every downstream record and gate
    # reads one value of one type. A string that is not a mode is rejected here
    # rather than defaulting to something safe-sounding: silently treating an
    # unrecognized mode as shadow would make a typo look like a deliberate
    # choice in the journal.
    execution_mode = ExecutionMode(mode)
    if execution_mode not in SELECTABLE_EXECUTION_MODES:
        raise ValueError(
            f"execution mode {execution_mode.value!r} is declared but not "
            f"implemented in this build; selectable modes are "
            f"{sorted(m.value for m in SELECTABLE_EXECUTION_MODES)}"
        )

    # Resolved here, before anything else happens, so a mode with no executor
    # vocabulary stops the cycle instead of reaching payload construction and
    # picking a branch by elimination. Deliberately independent of the check
    # above: enabling a mode and teaching the executor what it means are two
    # separate decisions, and neither should imply the other.
    try:
        plan_mode = _PLAN_MODE[execution_mode]
    except KeyError:
        raise NotImplementedError(
            f"execution mode {execution_mode.value!r} has no executor "
            "vocabulary; add it to _PLAN_MODE deliberately, having decided "
            "what it means for payload construction and the protection floor"
        ) from None

    result = CycleResult(
        cycle_id=cid,
        symbol=snapshot.symbol,
        strategy=strategy_name,
        outcome=CycleOutcome.NO_SIGNAL,
        mode=execution_mode,
        trading_date=trading_date,
        acquisition=acquisition,
        market_context=market_context,
    )

    def finish(outcome: CycleOutcome) -> CycleResult:
        result.outcome = outcome
        result.audit = _build_audit(result, snapshot, current)
        return result

    # --- 1. Strategy ------------------------------------------------------

    try:
        strategy = get_strategy(strategy_name)
        entry = config.strategies.strategies.get(strategy_name)
        context = StrategyContext(
            position=account.position_in(snapshot.symbol),
            params=dict(entry.params) if entry else {},
            active_stop=active_stop,
        )
        signal = strategy.evaluate(snapshot, context)
    except Exception as exc:  # A broken strategy must not halt the whole run.
        result.errors.append(f"strategy {strategy_name} raised: {exc!r}")
        return finish(CycleOutcome.ERROR)

    result.signal = signal

    if not signal.is_actionable:
        return finish(
            CycleOutcome.WATCH
            if signal.strength is SignalStrength.WATCH
            else CycleOutcome.NO_SIGNAL
        )

    # --- 2. Risk ----------------------------------------------------------

    engine = RiskEngine(config.risk, is_halted=config.is_halted())
    decision = engine.evaluate(
        signal, snapshot, account, as_of=current.date(), last_loss_exit=last_loss_exit
    )
    result.risk_decision = decision

    if not decision.is_executable:
        return finish(CycleOutcome.REJECTED_BY_RISK)

    # --- 3. Critic --------------------------------------------------------

    report = critique(
        decision,
        signal,
        snapshot,
        account,
        config.risk,
        recent_symbol_trades=recent_symbol_trades,
        now=current,
    )
    result.critic = report

    if not report.approved:
        return finish(CycleOutcome.REJECTED_BY_CRITIC)

    # --- 3b. Bounded re-size -----------------------------------------------
    # The critic reviews a concrete sized order, so its confidence adjustment
    # arrives after sizing. Re-size exactly once so the adjustment reaches the
    # position, then verify it only ever shrank. Gates are not re-run: a
    # smaller order cannot breach a limit the larger one already satisfied.

    result.original_confidence = signal.confidence
    adjusted = report.adjusted_confidence(signal.confidence)
    result.adjusted_confidence = adjusted

    if adjusted < signal.confidence:
        before = decision.approved_notional
        resized = engine.resize(
            decision, signal, snapshot, account, adjusted, as_of=current.date()
        )
        if resized.is_executable and resized.approved_notional is not None:
            # An LLM must never be able to enlarge a position. If this ever
            # fires it is a logic error, not a market condition.
            assert resized.approved_notional <= before, (
                f"critic re-size grew the order: {before} -> {resized.approved_notional}"
            )
            decision = resized
            result.risk_decision = decision
        else:
            # Shrinking pushed the order under the minimum notional. That is a
            # rejection, not a silent fall back to the original size.
            result.risk_decision = resized
            return finish(CycleOutcome.REJECTED_BY_RISK)

    # --- 4. Payload -------------------------------------------------------

    try:
        plan = build_order_payload(
            decision,
            snapshot,
            account.account_number,
            max_spread_pct=config.risk.max_spread_pct,
            max_price_drift_pct=config.risk.max_price_drift_pct,
            allow_unprotected_shadow_entries=config.risk.allow_unprotected_shadow_entries,
            mode=plan_mode,
            known_client_keys=known_client_keys,
            now=current,
        )
    except PreflightError as exc:
        result.errors.append(str(exc))
        if result.risk_decision is not None:
            result.risk_decision = result.risk_decision.model_copy(
                update={"breached_limits": [*decision.breached_limits, f"preflight: {exc}"]}
            )
        return finish(CycleOutcome.REJECTED_BY_RISK)

    result.plan = plan

    # --- 5. Shadow fill ---------------------------------------------------
    # Live submission happens in the agent, not here. This function's most
    # important property is that it cannot place an order.

    if execution_mode is ExecutionMode.SHADOW:
        fill = ShadowExecutor().submit(plan, now=current)
        result.trade = TradeRecord(
            client_key=fill.client_key,
            symbol=fill.symbol,
            strategy=fill.strategy,
            mode=execution_mode,
            opened_at=fill.submitted_at,
            entry_price=fill.fill_price,
            quantity=fill.quantity,
            notional=fill.notional,
            stop_price=fill.managed_stop,
            target_price=fill.managed_target,
            entry_rationale=signal.reasons,
            thesis=signal.thesis,
            invalidation_reason=signal.invalidation_reason,
            sector=snapshot.sector,
            protection_state=fill.protection,
            capability_profile=plan.capability_profile,
            market_regime=(
                market_context.regime.value if market_context is not None else None
            ),
            stop_basis=signal.metrics.get("stop_basis"),
        )
        return finish(CycleOutcome.SHADOW_FILLED)

    return finish(CycleOutcome.LIVE_FILLED)


def _build_audit(
    result: CycleResult, snapshot: MarketSnapshot, occurred_at: datetime
) -> AuditEntry:
    signal = result.signal
    decision = result.risk_decision
    plan = result.plan
    market = result.market_context
    return AuditEntry(
        mode=result.mode,
        trading_date=result.trading_date,
        acquisition_profile_ref=result.acquisition.profile_ref,
        acquisition_config_fingerprint=result.acquisition.content_fingerprint,
        protection_state=plan.protection if plan is not None else None,
        capability_profile=plan.capability_profile if plan is not None else None,
        market_regime=market.regime.value if market is not None else None,
        market_context=market.to_dict() if market is not None else None,
        cycle_id=result.cycle_id,
        occurred_at=occurred_at,
        symbol=result.symbol,
        strategy=result.strategy,
        outcome=result.outcome,
        reference_price=snapshot.reference_price,
        confidence=signal.confidence if signal else 0.0,
        signal_strength=signal.strength.value if signal else None,
        original_confidence=result.original_confidence,
        adjusted_confidence=result.adjusted_confidence,
        thesis=signal.thesis if signal else None,
        invalidation_reason=signal.invalidation_reason if signal else None,
        reasons=list(signal.reasons) if signal else [],
        failed_conditions=([*signal.failed_conditions] if signal else []) + result.errors,
        risk_breaches=list(decision.breached_limits) if decision else [],
        critic_notes=(
            [*result.critic.blocks, *result.critic.concerns] if result.critic else []
        ),
        snapshot_json=snapshot.model_dump(mode="json"),
    )


def summarize_cycles(results: list[CycleResult]) -> dict[str, object]:
    """Roll several symbols' results into one report for the agent to relay."""
    by_outcome: dict[str, int] = {}
    for r in results:
        by_outcome[r.outcome.value] = by_outcome.get(r.outcome.value, 0) + 1

    actionable = [r for r in results if r.plan is not None]
    return {
        "evaluated": len(results),
        "by_outcome": by_outcome,
        "actionable": [r.headline() for r in actionable],
        "total_notional": str(
            sum((r.plan.estimated_cost for r in actionable), Decimal("0"))
        ),
        "errors": [e for r in results for e in r.errors],
    }
