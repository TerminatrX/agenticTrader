"""Adversarial review of a proposed trade — the deterministic half.

The critique is split across two layers on purpose. This module holds the
checks that are mechanical and therefore worth making unskippable: arithmetic
that must reconcile, data that must be fresh, patterns that are objectively
present in the journal. The `critique-trade` skill holds the judgement calls
that genuinely need a model — is this thesis actually supported, does the
reasoning contain a rationalization, is there context the data does not show.

Putting the mechanical checks here means the agent cannot talk itself past
them, which is the entire point of having a critic. An LLM asked to review its
own trade will agree with itself far more often than it should; a function that
recomputes the risk and finds a mismatch will not.

Findings are severity-tagged. BLOCK is fatal. CONCERN is surfaced for a human
and, in a fully autonomous run, should also block — nothing here fires on
weak evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from agentic_trader.config import RiskConfig
from agentic_trader.market.regime import Regime, classify_regime
from agentic_trader.models import (
    AccountState,
    MarketSnapshot,
    RiskDecision,
    Side,
    Signal,
)


class CriticVerdict(StrEnum):
    PASS = "pass"
    CONCERN = "concern"
    BLOCK = "block"


@dataclass
class CriticReport:
    verdict: CriticVerdict = CriticVerdict.PASS
    blocks: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    # Cumulative downward adjustment to confidence, always <= 0.
    #
    # Confidence multiplies notional in `risk.sizing`, so this value moves real
    # position size. It is clamped to be non-positive precisely because an
    # adjustment that could *raise* confidence would let a language model
    # enlarge a position — the one coupling this architecture exists to
    # prevent. The model may veto or shrink. It may never amplify.
    confidence_adjustment: float = 0.0

    def block(self, reason: str) -> None:
        self.verdict = CriticVerdict.BLOCK
        self.blocks.append(reason)

    def concern(self, reason: str, confidence_penalty: float = 0.0) -> None:
        if self.verdict is not CriticVerdict.BLOCK:
            self.verdict = CriticVerdict.CONCERN
        self.concerns.append(reason)
        if confidence_penalty:
            self.penalize(confidence_penalty)

    def penalize(self, amount: float) -> None:
        """Reduce confidence by `amount` (given as a positive magnitude)."""
        self.confidence_adjustment -= abs(amount)

    def observe(self, note: str) -> None:
        self.observations.append(note)

    @property
    def approved(self) -> bool:
        return self.verdict is not CriticVerdict.BLOCK

    def adjusted_confidence(self, original: float) -> float:
        """Apply the adjustment, clamped so it can only ever lower confidence."""
        return max(0.0, min(original, original + self.confidence_adjustment))

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "blocks": self.blocks,
            "concerns": self.concerns,
            "observations": self.observations,
            "confidence_adjustment": round(self.confidence_adjustment, 4),
        }


def critique(
    decision: RiskDecision,
    signal: Signal,
    snapshot: MarketSnapshot,
    account: AccountState,
    config: RiskConfig,
    *,
    recent_symbol_trades: int = 0,
    now: datetime | None = None,
) -> CriticReport:
    """Re-derive the trade independently and report anything that fails to reconcile."""
    report = CriticReport()

    if not decision.is_executable or decision.intent is None:
        report.observe(f"nothing to critique: decision is {decision.decision.value}")
        return report

    intent = decision.intent
    current = now or datetime.now(UTC)

    # --- Internal consistency --------------------------------------------
    # A mismatch here means two layers disagree about what is being ordered,
    # which is a bug, not a judgement call.

    if intent.symbol != signal.symbol:
        report.block(f"intent symbol {intent.symbol} does not match signal {signal.symbol}")

    if signal.side is not None and intent.side is not signal.side:
        report.block(
            f"intent side {intent.side.value} contradicts signal side {signal.side.value}"
        )

    if (
        intent.side is Side.BUY
        and intent.stop_price is not None
        and intent.stop_price >= intent.reference_price
    ):
        report.block(
            f"stop {intent.stop_price} is at or above entry {intent.reference_price} "
            "— this position cannot lose money in the intended direction"
        )

    # --- Risk arithmetic, recomputed from scratch -------------------------

    if intent.side is Side.BUY and intent.stop_price is not None:
        implied_risk = intent.notional * (
            (intent.reference_price - intent.stop_price) / intent.reference_price
        )
        budget = account.total_value * config.risk_per_trade_pct
        # A small tolerance absorbs rounding to cents; anything beyond it means
        # the sizing math and the config disagree.
        if implied_risk > budget * Decimal("1.05"):
            report.block(
                f"position risks {implied_risk:.2f} against a budget of {budget:.2f} "
                f"({config.risk_per_trade_pct:.1%} of {account.total_value:.2f})"
            )
        else:
            report.observe(f"risk check: {implied_risk:.2f} against {budget:.2f} budget")

    if intent.notional > account.buying_power:
        report.block(
            f"notional {intent.notional:.2f} exceeds buying power {account.buying_power:.2f}"
        )

    concentration = intent.notional / account.total_value if account.total_value else Decimal("0")
    if concentration > config.max_position_pct:
        report.block(
            f"position is {concentration:.1%} of the account, above the "
            f"{config.max_position_pct:.1%} ceiling"
        )
    elif concentration > config.max_position_pct * Decimal("0.9"):
        report.concern(
            f"position is {concentration:.1%} of the account — near the ceiling", 0.05
        )

    # --- Data quality -----------------------------------------------------

    # Age of the price itself, from the venue's print time. `captured_at` cannot
    # answer this: it is stamped when the snapshot object is built, so it
    # reports "fresh" even for a quote served from a stale cache.
    age_seconds = snapshot.quote_age_seconds(current)
    if age_seconds is None:
        report.block(
            "quote carries no venue timestamp — its age is unknown, and unknown "
            "is not the same as fresh"
        )
    elif age_seconds > 900:
        report.block(f"quote is {age_seconds / 60:.0f} minutes old — refuse to trade on it")
    elif age_seconds > 300:
        report.concern(f"quote is {age_seconds / 60:.0f} minutes old", 0.10)

    if snapshot.indicators.as_of is not None:
        indicator_age = (current.date() - snapshot.indicators.as_of.date()).days
        if indicator_age > 4:
            report.concern(
                f"indicators computed through {snapshot.indicators.as_of.date()} "
                f"({indicator_age}d ago) — stale relative to the live price",
                0.10,
            )

    missing = [
        name
        for name, value in (
            ("RSI", snapshot.indicators.rsi_14),
            ("MACD", snapshot.indicators.macd_hist),
            ("SMA20", snapshot.indicators.sma_20),
            ("SMA50", snapshot.indicators.sma_50),
            ("SMA200", snapshot.indicators.sma_200),
        )
        if value is None
    ]
    if missing:
        report.block(f"missing indicators the strategy claims to use: {', '.join(missing)}")

    # --- Thesis coherence -------------------------------------------------

    # UNKNOWN is checked first: it does not permit long entry either, so leaving
    # it to the general branch below would report "contradicts the strategy
    # premise" for what is really absent data. Both block; only one is true.
    regime = classify_regime(snapshot)
    if regime is Regime.UNKNOWN:
        if intent.side is Side.BUY:
            report.block(
                "regime could not be classified — entering long without trend "
                "context is trading on absent data"
            )
        else:
            report.concern(
                "regime could not be classified — trading without trend context", 0.15
            )
    elif intent.side is Side.BUY and not regime.allows_long_entry:
        report.block(f"long entry in a {regime.value} regime contradicts the strategy premise")

    # A flat-percentage stop is a weaker claim than a measured one: it asserts
    # a risk boundary without having looked at how far this stock actually
    # moves. It still trades — refusing would mean no trades whenever an
    # indicator call fails — but it is sized slightly smaller for it.
    if intent.side is Side.BUY and signal.metrics.get("stop_basis") == "flat_pct":
        report.concern(
            "stop is a flat percentage, not derived from measured volatility "
            "(ATR missing) — the risk boundary is assumed rather than observed",
            0.10,
        )

    if intent.side is Side.BUY and signal.confidence < 0.5:
        report.concern(f"confidence {signal.confidence:.2f} is weak for a new position")

    if not signal.reasons:
        report.block("signal carries no stated reasons — an unexplainable trade is not reviewable")

    # --- Behavioural patterns ---------------------------------------------

    if recent_symbol_trades >= 3:
        report.concern(
            f"{recent_symbol_trades} recent trades in {intent.symbol} — "
            "check for revenge trading or an over-fitted setup",
            0.10,
        )

    if snapshot.earnings is not None:
        days_out = snapshot.earnings.days_until(current.date())
        if 0 <= days_out <= 10:
            note = (
                f"earnings in {days_out}d ({snapshot.earnings.report_date}, "
                f"{'confirmed' if snapshot.earnings.verified else 'tentative'})"
            )
            report.concern(f"{note} — plan the exit before the report", 0.15)

    gap = snapshot.previous_close
    if gap is not None and gap > 0:
        move = abs(snapshot.last_price - gap) / gap
        if move > Decimal("0.05"):
            report.concern(
                f"price moved {move:.1%} from the prior close — "
                "indicators computed before this move may no longer describe the setup",
                0.20,
            )

    return report
