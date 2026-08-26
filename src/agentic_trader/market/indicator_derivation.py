"""Turning one authoritative bar payload into the strategy's indicators.

The acquisition profile answers *what we asked Robinhood for*. Once indicators
stop being fetched, that no longer explains *how bars became RSI* — the seed
conventions, the per-indicator source windows, and the rule for which bars count
as complete are all decision-affecting and none of them appear in an acquisition
version number. This module owns that second contract, versioned and
fingerprinted separately for the same reason `scan_definition` is separate from
`scanner_capabilities`: the two move for different reasons.

Preserving semantics, not improving them
----------------------------------------

This is deliberately a *semantic-preserving* change. The previous milestone
established two things, and only one of them licenses a cutover:

- the local formulas agree with the provider's on identical inputs, to float64
  round-trip; and
- **range length itself moves the values** — up to 5.8e-3 RSI points and 4.1e-4
  of MACD histogram between a production-length window and a fully converged
  one.

Replacing the formula source and lengthening the history are therefore two
changes, and combining them would leave any disagreement unattributable. So
each indicator here is computed over *the same calendar window the production
broker call requested*, cut from the 420-day superset. Adopting the converged
420-day values is a later, separately reviewed decision.

Legacy semantics, then ours
---------------------------

The per-indicator windows and warm-up counts below were measured against the
broker endpoints this module replaces, so that swapping the source of a value
does not also change the value. After cutover nothing calls those endpoints and
these stop being compatibility shims: they are this system's pinned indicator
semantics, and a later provider change cannot move a production number. Moving
them becomes a deliberate version bump here.

Slicing by timestamp, never by count
------------------------------------

Production windows were defined by `start_time`, not by a number of points. A
90-day span holds a different number of sessions depending on where holidays
fall, so `bars[-62:]` is an approximation of the request rather than a
reproduction of it. Every slice below is `begins_at >= trading_date - lookback`,
which is the boundary the acquisition contract actually sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from agentic_trader.market.local_indicators import (
    EmaSeed,
    atr_series,
    macd_series,
    rsi_series,
    sma,
)
from agentic_trader.models import Bar, Indicators
from agentic_trader.models.capabilities import CapabilityProfile
from agentic_trader.models.market_snapshot import IndicatorProvenance, IndicatorSource

COMPLETED_BAR_RULE = "exclude_bars_dated_on_or_after_trading_date"
"""Which bars are eligible to feed an indicator.

`Indicators.as_of` means *the last completed daily bar*, and local computation
must not quietly start consuming a partial session. Two facts forced an explicit
rule rather than trust:

1. The historicals guide states "close_price on the most recent bar is NOT the
   official settled close". That sentence only makes sense if the newest bar can
   be provisional.
2. Observations on 2026-08-25 at 08:45 and 16:28 ET — pre-market and post-close —
   both showed the newest bar as the *previous* session, and both matched the
   indicator endpoint. But neither observation was taken during regular hours,
   so the in-progress case was never seen. Absence of evidence is not evidence
   of absence, and this feeds a hard entry path.

The rule keys on `trading_date` rather than on a clock: a bar dated on or after
the evaluation date is excluded. Deterministic without a market calendar,
identical for normal and early-close sessions (a short session is still dated
that day), and free of wall-clock sensitivity, which matters because every
other date input in this system is explicit for exactly that reason.

**Why no wall-clock rule is needed.** The application's actionable window is
regular market hours and cannot be anything else: `build_order_payload` sends
`market_hours="regular_hours"` unconditionally, and the broker restricts
fractional and dollar-denominated orders -- every order this account size can
place -- to `type=market` in regular hours. An indicator advancing to today's
bar after the close could therefore never inform an order placed that day.
Clock sensitivity would buy nothing and would make the derivation depend on
when a cycle happened to run.

Its boundary, stated plainly: a cycle run in the evening of a trading day
excludes that day's now-final bar. That is the same thing the broker indicator
endpoint was still doing 28 minutes after the close, so it preserves behaviour
rather than changing it, and erring one bar short is conservative in a way that
consuming a partial bar could never be.

If the application ever supports acting after hours, this rule needs a bounded
post-close observation of when the endpoint advances -- not a guess.
"""


WARMUP_EVIDENCE = (
    "Measured 2026-08-25 against production-range broker calls. Slicing to the "
    "literal requested window reproduces the request boundary but not the "
    "values, because the endpoint prepends warm-up before start_time. "
    "Effective bar counts, against a 124/145/62-bar literal window: RSI(14) "
    "139 (+15 = period+1), ATR(14) 138 (+14 = period), MACD(12,26,9) 179 "
    "(+34 = slow+signal-1). With the prepend, local matches broker to ~1e-14 "
    "on AAPL and KO; without it RSI differs by 2.3e-03. SMA is a finite window "
    "and is unaffected either way (5e-14). Consistent with the tool schema's "
    "note that a request costs 'the requested range plus the indicator's "
    "warm-up'."
)
"""How the warm-up figures were established, kept next to the numbers.

Provenance for a migration, not a live dependency. The figures were
reverse-engineered from the endpoint being migrated *off*, so that changing
where a value comes from would not also change the value. Once production stops
calling that endpoint -- the point of the cutover -- these become this system's
own pinned semantics, and a later change to Robinhood's warm-up policy cannot
reach a production number.

They are empirical legacy-compatibility constants, not universal indicator
mathematics: another provider, or a design starting from scratch, would have no
reason to choose them. The fingerprint is what makes them a stated contract
rather than an accident of how they were first obtained.
"""


def completed_bars(bars: Sequence[Bar], trading_date: date) -> list[Bar]:
    """Bars eligible to feed an indicator, oldest first.

    Drops interpolated gap-fill (which carries no information by the provider's
    own description) and anything dated on or after `trading_date`. Sorted
    defensively: a duplicate or out-of-order timestamp would silently corrupt a
    recursive indicator, and `deduplicate` below refuses rather than guessing.
    """
    eligible = [
        b for b in bars if not b.interpolated and b.begins_at.date() < trading_date
    ]
    return sorted(eligible, key=lambda b: b.begins_at)


def duplicate_timestamps(bars: Sequence[Bar]) -> list[datetime]:
    """Repeated `begins_at` values, if any. Empty is the healthy case."""
    seen: set[datetime] = set()
    dupes: list[datetime] = []
    for bar in bars:
        if bar.begins_at in seen:
            dupes.append(bar.begins_at)
        seen.add(bar.begins_at)
    return dupes


def window_start(trading_date: date, lookback_calendar_days: int) -> datetime:
    """The RFC3339 boundary the production acquisition contract sent.

    Deliberately identical arithmetic to `acquisition._start_time`, so the slice
    reproduces the request rather than approximating it. A test pins the two
    together; if acquisition ever changes how it builds a start, this must move
    with it or the cutover stops being semantic-preserving.
    """
    start = trading_date - timedelta(days=lookback_calendar_days)
    return datetime.combine(start, time.min, tzinfo=UTC)


def slice_from(
    bars: Sequence[Bar], trading_date: date, lookback_calendar_days: int
) -> list[Bar]:
    """Completed bars at or after the production window boundary."""
    boundary = window_start(trading_date, lookback_calendar_days)
    return [b for b in completed_bars(bars, trading_date) if b.begins_at >= boundary]


@dataclass(frozen=True)
class DerivedIndicatorSpec:
    """One indicator, its parameters, and the window it is computed over."""

    key: str
    kind: str  # "sma" | "rsi" | "atr" | "macd"

    source_lookback_calendar_days: int
    """The production broker call's lookback, reproduced exactly.

    Not a convergence requirement and not a minimum — a *legacy compatibility*
    number. Whether it is long enough for the recurrence to converge is a
    separate question this branch intentionally does not answer.
    """

    warmup_bars: int = 0
    """Completed bars included *before* the window start.

    Originally measured from the provider, now **our own pinned semantics**.
    The distinction matters: it was derived empirically so the cutover would
    preserve values rather than silently retune them, but after cutover nothing
    calls the indicator endpoint, so a later change on Robinhood's side cannot
    move a production number. What began as compatibility is now simply the
    definition of this system's indicator windows, fixed by this fingerprint.

    The value came from the migration target rather than from indicator
    mathematics: the endpoint prepended each indicator's own minimum-history
    requirement so its first returned point landed *at* the requested
    `start_time`. Slicing to the literal window without it reproduced the
    request boundary but not the values -- RSI(14) on AAPL moved 2.3e-03
    points, 23x the tolerance the formula-equivalence milestone justified.

    See `WARMUP_EVIDENCE`.
    """

    period: int | None = None
    fast_period: int | None = None
    slow_period: int | None = None
    signal_period: int | None = None

    def fingerprint_items(self) -> tuple[str, ...]:
        items = [
            f"derive.{self.key}.kind={self.kind}",
            f"derive.{self.key}.source_lookback_calendar_days="
            f"{self.source_lookback_calendar_days}",
            f"derive.{self.key}.warmup_bars={self.warmup_bars}",
        ]
        for name in ("period", "fast_period", "slow_period", "signal_period"):
            value = getattr(self, name)
            if value is not None:
                items.append(f"derive.{self.key}.{name}={value}")
        return tuple(items)


@dataclass(frozen=True)
class DerivationResult:
    """Indicators plus an account of anything that could not be produced."""

    indicators: Indicators
    unavailable: tuple[tuple[str, str], ...] = ()
    bars_considered: int = 0

    @property
    def complete(self) -> bool:
        return not self.unavailable


@dataclass(frozen=True)
class IndicatorDerivationProfile(CapabilityProfile):
    """How bars become indicators. Versioned because every field here can move
    a decision."""

    source_description: str
    completed_bar_rule: str

    ema_seed: EmaSeed
    """The MACD seeding convention, stated rather than defaulted.

    `EmaSeed.SMA` — Wilder's own convention and the more common one in charting
    packages. Chosen because it is the convention the validation milestone ran
    its primary comparison under, not because it scored marginally better on a
    sample: picking a convention to improve one measurement is how an
    implementation gets fitted to its test set. At production window lengths the
    two conventions differ well below any decision boundary anyway.
    """

    specs: tuple[DerivedIndicatorSpec, ...]

    def spec_for(self, key: str) -> DerivedIndicatorSpec:
        for spec in self.specs:
            if spec.key == key:
                return spec
        raise KeyError(f"no derivation spec named {key!r} in {self.profile_ref}")

    @property
    def max_lookback_calendar_days(self) -> int:
        return max(s.source_lookback_calendar_days for s in self.specs)

    @property
    def provenance(self) -> IndicatorProvenance:
        return IndicatorProvenance(
            source=IndicatorSource.LOCAL,
            profile_ref=self.profile_ref,
            profile_fingerprint=self.content_fingerprint,
        )

    def fingerprint_items(self) -> tuple[str, ...]:
        items = [
            f"profile_id={self.profile_id}",
            f"version={self.version}",
            f"source_description={self.source_description}",
            f"completed_bar_rule={self.completed_bar_rule}",
            f"ema_seed={self.ema_seed.value}",
        ]
        for spec in sorted(self.specs, key=lambda s: s.key):
            items.extend(spec.fingerprint_items())
        return tuple(items)


CURRENT_DERIVATION = IndicatorDerivationProfile(
    profile_id="local-indicator-derivation",
    version="v1-2026-08-25",
    as_of=date(2026, 8, 25),
    source_description="authoritative regular-session split-adjusted daily bars",
    completed_bar_rule=COMPLETED_BAR_RULE,
    ema_seed=EmaSeed.SMA,
    specs=(
        # Each lookback mirrors the corresponding broker call in
        # agentic-acquisition@v3, so this cutover reproduces today's semantics
        # rather than adopting the longer converged window. A test pins them to
        # v3's own values so the two cannot drift apart silently.
        DerivedIndicatorSpec("rsi", "rsi", 180, warmup_bars=15, period=14),
        DerivedIndicatorSpec(
            "macd", "macd", 210, warmup_bars=34,
            fast_period=12, slow_period=26, signal_period=9,
        ),
        # SMA takes no warm-up, and this is a real distinction rather than an
        # omission. A finite window's last value depends only on the trailing
        # `period` closes, so bars before the window start cannot move it --
        # confirmed empirically: local SMA20 with no prepend matched the
        # provider to 5e-14. Requiring a prepend here would also make SMA200
        # unsatisfiable inside the 420-day envelope for no gain, which is how
        # the distinction was found.
        DerivedIndicatorSpec("sma_20", "sma", 90, warmup_bars=0, period=20),
        DerivedIndicatorSpec("sma_50", "sma", 90, warmup_bars=0, period=50),
        DerivedIndicatorSpec("sma_200", "sma", 330, warmup_bars=0, period=200),
        DerivedIndicatorSpec("atr", "atr", 180, warmup_bars=14, period=14),
    ),
)
"""The derivation contract in force.

Its lookbacks are legacy-compatibility values copied from the broker calls they
replace, **not** convergence requirements. `local_indicators.bar_requirements()`
says RSI needs ~203 bars to shed its seed and MACD ~277; these windows supply
fewer. That is intentional and is the whole point of the sequencing: today's
production values carry that same seed dependence, and reproducing them is what
makes this cutover attributable. Lengthening them is a deliberate later change
with its own decision analysis.
"""


def _envelope_note(available: int, needed: int, label: str) -> str:
    return (
        f"{label}: window needs bars from {needed} calendar days back, but the "
        f"payload only reaches {available}"
    )


def derive_indicators(
    bars: Sequence[Bar],
    trading_date: date,
    *,
    profile: IndicatorDerivationProfile = CURRENT_DERIVATION,
) -> DerivationResult:
    """Compute every indicator the strategy reads, from one bar payload.

    Anything the history cannot support comes back `None` with a reason. Nothing
    is approximated: no shortened period, no partial window, no substitution,
    and never a silent fall back to a broker value. A strategy treats a missing
    indicator as a failed condition, so failing closed here refuses the entry
    rather than inventing one.
    """
    eligible = completed_bars(bars, trading_date)
    missing: list[tuple[str, str]] = []

    dupes = duplicate_timestamps(eligible)
    if dupes:
        # A repeated session would double-count a delta and quietly corrupt
        # every recursive value after it. Refuse the whole set rather than
        # de-duplicating on a guess about which row is authoritative.
        reason = f"duplicate bar timestamps: {sorted({d.date().isoformat() for d in dupes})}"
        return DerivationResult(
            indicators=Indicators(provenance=profile.provenance),
            unavailable=tuple((s.key, reason) for s in profile.specs),
            bars_considered=len(eligible),
        )

    # How far back the payload actually reaches, so an under-supplied envelope
    # is reported as such rather than silently producing a shorter window.
    oldest = eligible[0].begins_at if eligible else None

    fields: dict[str, Any] = {}

    def window(spec: DerivedIndicatorSpec) -> list[Bar] | None:
        """The bars the provider would have computed this indicator over.

        The literal requested window, plus the measured warm-up prepended from
        immediately before it. Both halves matter: the boundary reproduces the
        request, the prepend reproduces the values.
        """
        boundary = window_start(trading_date, spec.source_lookback_calendar_days)
        if oldest is None or oldest > boundary:
            reach = (trading_date - oldest.date()).days if oldest is not None else 0
            missing.append(
                (
                    spec.key,
                    _envelope_note(
                        reach, spec.source_lookback_calendar_days, "insufficient envelope"
                    ),
                )
            )
            return None

        at_or_after = [i for i, b in enumerate(eligible) if b.begins_at >= boundary]
        first = at_or_after[0] if at_or_after else len(eligible)
        start = first - spec.warmup_bars
        if start < 0:
            missing.append(
                (
                    spec.key,
                    f"needs {spec.warmup_bars} warm-up bars before its "
                    f"{spec.source_lookback_calendar_days}d window; only {first} "
                    "available -- the payload envelope is too short to reproduce "
                    "the provider's window",
                )
            )
            return None
        return eligible[start:]

    def record_missing(spec: DerivedIndicatorSpec, needed: int, got: int) -> None:
        missing.append(
            (
                spec.key,
                f"needs {needed} bars in its "
                f"{spec.source_lookback_calendar_days}d window, got {got}",
            )
        )

    for spec in profile.specs:
        rows = window(spec)
        if rows is None:
            continue
        closes = [b.close for b in rows]

        if spec.kind == "sma":
            value = sma(closes, spec.period)
            if value is None:
                record_missing(spec, spec.period, len(rows))
            else:
                fields[spec.key] = value

        elif spec.kind == "rsi":
            series = rsi_series(closes, spec.period)
            if len(series) < 2:
                record_missing(spec, spec.period + 2, len(rows))
            else:
                fields["rsi_14"] = float(series[-1])
                fields["rsi_prev"] = float(series[-2])

        elif spec.kind == "atr":
            series = atr_series(rows, spec.period)
            if not series:
                record_missing(spec, spec.period + 1, len(rows))
            else:
                fields["atr_14"] = series[-1]

        elif spec.kind == "macd":
            points = macd_series(
                closes,
                fast=spec.fast_period,
                slow=spec.slow_period,
                signal=spec.signal_period,
                seed=profile.ema_seed,
            )
            if len(points) < 2:
                record_missing(
                    spec, spec.slow_period + spec.signal_period, len(rows)
                )
            else:
                fields["macd"] = float(points[-1].line)
                fields["macd_signal"] = float(points[-1].signal)
                fields["macd_hist"] = float(points[-1].histogram)
                fields["macd_hist_prev"] = float(points[-2].histogram)
        else:  # pragma: no cover - guarded by the profile's own construction
            raise ValueError(f"unknown derivation kind {spec.kind!r}")

    return DerivationResult(
        indicators=Indicators(
            # The last completed bar every value was computed through. Not the
            # wall clock, and not the newest bar in the payload if that bar is
            # today's.
            as_of=eligible[-1].begins_at if eligible else None,
            provenance=profile.provenance,
            **fields,
        ),
        unavailable=tuple(missing),
        bars_considered=len(eligible),
    )


def envelope_covers(
    bars: Sequence[Bar],
    trading_date: date,
    *,
    profile: IndicatorDerivationProfile = CURRENT_DERIVATION,
) -> dict[str, bool]:
    """Whether the payload reaches back far enough for each indicator's window.

    The 420-day acquisition request is meant to be a superset of all six. This
    reports it per indicator rather than asserting it, so an under-supplied
    payload is visible instead of producing quietly shortened windows.
    """
    eligible = completed_bars(bars, trading_date)
    oldest = eligible[0].begins_at if eligible else None
    return {
        spec.key: oldest is not None
        and oldest <= window_start(trading_date, spec.source_lookback_calendar_days)
        for spec in profile.specs
    }


def decimal_or_none(value: float | Decimal | None) -> Decimal | None:
    """Small helper for callers comparing float indicator fields as Decimal."""
    return None if value is None else Decimal(str(value))


__all__ = [
    "COMPLETED_BAR_RULE",
    "WARMUP_EVIDENCE",
    "CURRENT_DERIVATION",
    "DerivationResult",
    "DerivedIndicatorSpec",
    "IndicatorDerivationProfile",
    "completed_bars",
    "decimal_or_none",
    "derive_indicators",
    "duplicate_timestamps",
    "envelope_covers",
    "slice_from",
    "window_start",
]
