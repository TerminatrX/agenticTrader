"""One evaluation cycle, start to finish, as a pure function.

Despite the name, this is not a loop and it owns no schedule. The agent drives
the loop: it fetches data through MCP tools, calls `run_cycle`, and — only if
the result says so, and only after the critic and any human gate agree —
submits the order. `run_cycle` itself is deterministic and side-effect-free, so
the same inputs always produce the same decision and any past cycle can be
replayed from its stored snapshot.

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

from agentic_trader.agents.critic import CriticReport, critique
from agentic_trader.config import AppConfig
from agentic_trader.execution.executor import ExecutionPlan, PreflightError, build_order_payload
from agentic_trader.execution.shadow_executor import ShadowExecutor
from agentic_trader.journal.models import AuditEntry, CycleOutcome, TradeRecord
from agentic_trader.models import (
    AccountState,
    MarketSnapshot,
    RiskDecision,
    Signal,
    SignalStrength,
)
from agentic_trader.risk.engine import RiskEngine
from agentic_trader.strategies.base import StrategyContext, get_strategy


@dataclass
class CycleResult:
    """Everything one symbol's evaluation produced."""

    cycle_id: str
    symbol: str
    strategy: str
    outcome: CycleOutcome

    signal: Signal | None = None
    risk_decision: RiskDecision | None = None
    critic: CriticReport | None = None
    plan: ExecutionPlan | None = None
    trade: TradeRecord | None = None
    audit: AuditEntry | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def should_submit(self) -> bool:
        """True only when every gate agreed and the mode is live.

        Shadow plans are deliberately excluded: a shadow cycle has already
        recorded its simulated fill and must never reach the broker.
        """
        return (
            self.plan is not None
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
    mode: str = "shadow",
    cycle_id: str | None = None,
    known_client_keys: set[str] | None = None,
    last_loss_exit: date | None = None,
    recent_symbol_trades: int = 0,
    now: datetime | None = None,
) -> CycleResult:
    """Evaluate one symbol under one strategy."""
    current = now or datetime.now(UTC)
    cid = cycle_id or uuid.uuid4().hex[:12]

    result = CycleResult(
        cycle_id=cid,
        symbol=snapshot.symbol,
        strategy=strategy_name,
        outcome=CycleOutcome.NO_SIGNAL,
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

    # --- 4. Payload -------------------------------------------------------

    try:
        plan = build_order_payload(
            decision,
            snapshot,
            account.account_number,
            max_spread_pct=config.risk.max_spread_pct,
            mode="live" if mode == "live" else "shadow",
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

    if mode == "shadow":
        fill = ShadowExecutor().submit(plan, now=current)
        result.trade = TradeRecord(
            client_key=fill.client_key,
            symbol=fill.symbol,
            strategy=fill.strategy,
            mode="shadow",
            opened_at=fill.submitted_at,
            entry_price=fill.fill_price,
            quantity=fill.quantity,
            notional=fill.notional,
            stop_price=fill.managed_stop,
            target_price=fill.managed_target,
            entry_rationale=signal.reasons,
        )
        return finish(CycleOutcome.SHADOW_FILLED)

    return finish(CycleOutcome.LIVE_FILLED)


def _build_audit(
    result: CycleResult, snapshot: MarketSnapshot, occurred_at: datetime
) -> AuditEntry:
    signal = result.signal
    decision = result.risk_decision
    return AuditEntry(
        cycle_id=result.cycle_id,
        occurred_at=occurred_at,
        symbol=result.symbol,
        strategy=result.strategy,
        outcome=result.outcome,
        reference_price=snapshot.reference_price,
        confidence=signal.confidence if signal else 0.0,
        signal_strength=signal.strength.value if signal else None,
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
