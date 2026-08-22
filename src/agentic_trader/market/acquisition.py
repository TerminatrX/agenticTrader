"""What to request from the broker for one symbol's evaluation.

This is a *definition*, not a capability profile. The distinction is the same
one `scan_definition` draws against `scanner_capabilities`, and it matters for
the same reason:

    BrokerCapabilities / EarningsCapabilities   what the endpoint can do
    MarketDataAcquisitionProfile                what we choose to ask it

Retuning a lookback must not bump a profile describing Robinhood's contract,
and Robinhood changing its contract must not invalidate the record of what we
asked for. It reuses `CapabilityProfile` only for the versioning and
fingerprint machinery -- `profile_id`, `version`, `as_of`, `content_fingerprint`
-- which is generic identity plumbing rather than anything capability-specific.
**Do not add `Capability` fields here.** A claim about what the broker supports
belongs in the profile that owns that claim.

Why this exists
---------------

During the eight-shard scanner smoke, different acquisition workers fetched
different amounts of indicator history for the same logical input: roughly 30
points in one case, 57 in another, 265 in a third. The decisions happened to
come out the same, because every worker had *some* value at the last two
positions. That is luck, not reproducibility.

It matters beyond tidiness. RSI, ATR and MACD are recursive smoothers -- each
value depends on the one before it, back to a seed at the start of the
requested range. Ask for 30 bars and the returned RSI still carries visible
weight from its seed; ask for 300 and it does not. The two are *different
numbers for the same indicator on the same day*, and which one the strategy saw
was decided by whichever worker happened to run. SMA is a finite window and is
immune, which is precisely why the problem is invisible until it isn't.

The request shape, not the response
-----------------------------------

Everything here describes what goes *out*. No returned value, timestamp,
symbol, count, or market state appears in the fingerprint -- those belong to
the snapshot, which is persisted separately and already replayable.

Two schema facts shape the design, read from the MCP tool definitions:

1. **There is no point-count request parameter.** `get_equity_technical_-
   indicators` takes a *range* (`start_time`, optional `end_time`) and trims
   the response with `output` (`series` | `latest` | `last:N`). The indicator is
   computed over the full range first. So "how many points do we need" is a
   *sufficiency requirement we must satisfy by choosing a range*, never
   something we can ask for -- and the range is what has to be pinned.

2. **Omitting `interval` is not neutral.** For `get_equity_historicals`, "when
   omitted, the server picks an interval that targets ~2,500 bars across the
   requested range". An unpinned interval means the *range silently determines
   the granularity*: two workers with different lookbacks get different bar
   sizes. `interval` is therefore always sent explicitly.

Note the parameter is `bounds`, not `session`. The scanner's filters carry a
baked-in `session="all"` (see `scan_definition`), and that is a different
vocabulary on a different endpoint -- one more reason both are written down
rather than remembered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from agentic_trader.models.capabilities import CapabilityProfile

# Calendar days per trading day. US equities trade ~252 days a year against a
# 365-day calendar. Used only to convert a bar requirement into a range, and
# deliberately generous in the safe direction: over-asking costs a little
# server-side computation, under-asking silently truncates an indicator's
# warm-up and changes its value.
_CALENDAR_PER_TRADING_DAY = 365 / 252

# Added on top of the converted requirement. Covers a holiday cluster, a long
# weekend, or an unscheduled closure landing inside the window.
_HOLIDAY_BUFFER_DAYS = 10

# Pinned lookbacks are rounded up to this grid. Pinning the exact computed
# minimum would tie the fingerprint to the arithmetic above, so adjusting the
# buffer by a day would bump the contract without any real change in what is
# requested. A coarse grid decouples them.
_LOOKBACK_GRID_DAYS = 30

# No indicator is requested over a shorter window than this regardless of its
# warm-up. A 20-period average technically resolves in six weeks, but a range
# that short leaves no room for the bar-count conversion to be wrong.
_MIN_LOOKBACK_DAYS = 90


def _required_calendar_days(min_bars: int) -> int:
    """Smallest calendar range that reliably contains `min_bars` daily bars."""
    return math.ceil(min_bars * _CALENDAR_PER_TRADING_DAY) + _HOLIDAY_BUFFER_DAYS


def _pinned_lookback(min_bars: int) -> int:
    """The published lookback for a requirement: rounded up to the grid."""
    required = max(_required_calendar_days(min_bars), _MIN_LOOKBACK_DAYS)
    return math.ceil(required / _LOOKBACK_GRID_DAYS) * _LOOKBACK_GRID_DAYS


def _start_time(trading_date: date, lookback_calendar_days: int) -> str:
    """RFC3339 UTC start, derived from the trading date and nothing else.

    Not from the wall clock. The trading date is already the seed discovery
    selection rotates on, and deriving the range from it is what lets a request
    spec be regenerated identically months later.
    """
    start = trading_date - timedelta(days=lookback_calendar_days)
    return datetime.combine(start, time.min).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class HistoricalsSpec:
    """The pinned `get_equity_historicals` request.

    Kept on a shorter leash than the indicators because this endpoint has **no
    `output` parameter** -- the full bar series comes back, so range length is
    paid for in payload volume on every call. The indicator endpoints trim
    server-side and can afford a long warm-up.
    """

    tool: str
    interval: str
    bounds: str
    adjustment_type: str
    lookback_calendar_days: int

    #: Bars the deterministic core actually reads. `realized_volatility_pct`
    #: takes `bars[-20:]`; the intrabar stop check takes `bars[-1]`.
    required_bars: int

    def minimum_calendar_days(self) -> int:
        return _required_calendar_days(self.required_bars)

    def request(self, symbol: str, trading_date: date) -> dict[str, Any]:
        return {
            "symbols": [symbol],
            "interval": self.interval,
            "bounds": self.bounds,
            "adjustment_type": self.adjustment_type,
            "start_time": _start_time(trading_date, self.lookback_calendar_days),
        }

    def fingerprint_items(self) -> tuple[str, ...]:
        return (
            f"historicals.tool={self.tool}",
            f"historicals.interval={self.interval}",
            f"historicals.bounds={self.bounds}",
            f"historicals.adjustment_type={self.adjustment_type}",
            f"historicals.lookback_calendar_days={self.lookback_calendar_days}",
            f"historicals.required_bars={self.required_bars}",
        )


@dataclass(frozen=True)
class IndicatorSpec:
    """One pinned `get_equity_technical_indicators` request.

    The three bar counts are a *derivation*, not request parameters -- the
    endpoint accepts no point count. They exist so the pinned lookback can be
    proved sufficient rather than asserted, and so a future change to the
    strategy's needs fails a test instead of silently under-fetching.
    """

    #: Key this response is filed under in the snapshot payload dict, matching
    #: `parse_indicators`. Fingerprinted: routing an `sma` response to the
    #: `sma_200` slot would be a silent misattribution.
    key: str

    indicator_type: str
    interval: str
    bounds: str
    adjustment_type: str
    output: str
    lookback_calendar_days: int

    warmup_bars: int
    """Bars consumed before the first value exists. `period` for a simple
    lookback; `slow_period + signal_period` for MACD."""

    required_output_bars: int
    """Trailing values the core reads. Two wherever the strategy compares the
    current bar to the prior one -- that comparison is what separates buying a
    pullback from catching a falling knife, and it disappears silently if only
    one point comes back."""

    convergence_bars: int
    """Extra bars for a recursive smoother to shed its seed. Zero for SMA,
    which is a finite window and genuinely does not care.

    Sized so the seed's residual weight falls below 0.1%. For a smoother with
    coefficient a, that weight after k bars is (1-a)^k: Wilder RSI/ATR at
    a=1/14 needs k>=94, MACD's slow EMA at a=2/27 needs k>=90. Hence 100."""

    period: int | None = None
    fast_period: int | None = None
    slow_period: int | None = None
    signal_period: int | None = None

    @property
    def minimum_bars(self) -> int:
        return self.warmup_bars + self.convergence_bars + self.required_output_bars

    def minimum_calendar_days(self) -> int:
        return _required_calendar_days(self.minimum_bars)

    def request(self, symbol: str, trading_date: date) -> dict[str, Any]:
        """Exact MCP kwargs. Only parameters this indicator type accepts.

        The schema rejects a parameter the chosen type does not take, so the
        `None` fields are omitted rather than sent as null.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "type": self.indicator_type,
            "interval": self.interval,
            "bounds": self.bounds,
            "adjustment_type": self.adjustment_type,
            "output": self.output,
            "start_time": _start_time(trading_date, self.lookback_calendar_days),
        }
        for name in ("period", "fast_period", "slow_period", "signal_period"):
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        return params

    def fingerprint_items(self) -> tuple[str, ...]:
        items = [
            f"indicator.{self.key}.type={self.indicator_type}",
            f"indicator.{self.key}.interval={self.interval}",
            f"indicator.{self.key}.bounds={self.bounds}",
            f"indicator.{self.key}.adjustment_type={self.adjustment_type}",
            f"indicator.{self.key}.output={self.output}",
            f"indicator.{self.key}.lookback_calendar_days={self.lookback_calendar_days}",
            f"indicator.{self.key}.warmup_bars={self.warmup_bars}",
            f"indicator.{self.key}.required_output_bars={self.required_output_bars}",
            f"indicator.{self.key}.convergence_bars={self.convergence_bars}",
        ]
        for name in ("period", "fast_period", "slow_period", "signal_period"):
            value = getattr(self, name)
            if value is not None:
                items.append(f"indicator.{self.key}.{name}={value}")
        return tuple(items)


@dataclass(frozen=True)
class SingleCallSpec:
    """An endpoint taking no range or window -- one call, one symbol.

    Quote and earnings both land here. Neither accepts a lookback, and the
    honest record of that is a spec with no lookback field rather than an
    invented one that reads as though a window were chosen.
    """

    label: str
    tool: str
    note: str = ""

    def request(self, symbol: str) -> dict[str, Any]:
        return {"symbols": [symbol]} if self.label == "quote" else {"symbol": symbol}

    def fingerprint_items(self) -> tuple[str, ...]:
        return (f"{self.label}.tool={self.tool}",)


@dataclass(frozen=True)
class MarketDataAcquisitionProfile(CapabilityProfile):
    """The pinned request contract for one symbol's evaluation inputs."""

    historicals: HistoricalsSpec
    indicators: tuple[IndicatorSpec, ...]
    quote: SingleCallSpec
    earnings: SingleCallSpec

    end_time_policy: str = "omitted_defaults_to_request_time"
    """How `end_time` is handled, fingerprinted because it is a real choice.

    It is omitted. The evaluation needs the current bar, so pinning an end
    would exclude it and change the decision. Omitting is also the *more*
    deterministic option for the spec itself: an absent key is identical on
    every regeneration, whereas an explicit "now" would make the same
    (symbol, date, profile) produce a different request on every call.
    """

    #: Fields carried for readability that must never reach the fingerprint.
    _NON_SEMANTIC: tuple[str, ...] = field(default=("note",), repr=False)

    def spec_for(self, key: str) -> IndicatorSpec:
        for spec in self.indicators:
            if spec.key == key:
                return spec
        raise KeyError(f"no indicator spec named {key!r} in {self.profile_ref}")

    @property
    def indicator_keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self.indicators)

    def request_plan(self, symbol: str, trading_date: date) -> dict[str, Any]:
        """Every read-only call needed for one symbol, fully specified.

        A worker receives `(symbol, profile)` and translates this into calls.
        It chooses nothing: no lookback, no interval, no output width. That is
        the whole point -- prose describing "roughly 120 days" is what produced
        three different histories for one indicator.
        """
        ticker = symbol.upper()
        return {
            "symbol": ticker,
            "trading_date": trading_date.isoformat(),
            "acquisition_profile_ref": self.profile_ref,
            "acquisition_config_fingerprint": self.content_fingerprint,
            "calls": {
                "quote": {
                    "tool": self.quote.tool,
                    "params": self.quote.request(ticker),
                },
                "historicals": {
                    "tool": self.historicals.tool,
                    "params": self.historicals.request(ticker, trading_date),
                },
                "earnings": {
                    "tool": self.earnings.tool,
                    "params": self.earnings.request(ticker),
                },
                "indicators": {
                    spec.key: {
                        "tool": "get_equity_technical_indicators",
                        "params": spec.request(ticker, trading_date),
                    }
                    for spec in self.indicators
                },
            },
        }

    def fingerprint_items(self) -> tuple[str, ...]:
        items: list[str] = [
            f"profile_id={self.profile_id}",
            f"version={self.version}",
            f"end_time_policy={self.end_time_policy}",
            *self.historicals.fingerprint_items(),
            *self.quote.fingerprint_items(),
            *self.earnings.fingerprint_items(),
        ]
        # Sorted by key so reordering the declaration is not a contract change,
        # while adding, removing, or retuning one is.
        for spec in sorted(self.indicators, key=lambda s: s.key):
            items.extend(spec.fingerprint_items())
        return tuple(items)


def _wilder(
    key: str, indicator_type: str, period: int, output: str, required: int
) -> IndicatorSpec:
    """RSI and ATR: same Wilder smoothing, same convergence requirement."""
    warmup = period
    return IndicatorSpec(
        key=key,
        indicator_type=indicator_type,
        interval="day",
        bounds="regular",
        adjustment_type="split",
        output=output,
        lookback_calendar_days=_pinned_lookback(warmup + 100 + required),
        warmup_bars=warmup,
        required_output_bars=required,
        convergence_bars=100,
        period=period,
    )


def _sma(key: str, period: int) -> IndicatorSpec:
    """A finite window: no seed to shed, so no convergence allowance."""
    return IndicatorSpec(
        key=key,
        indicator_type="sma",
        interval="day",
        bounds="regular",
        adjustment_type="split",
        output="latest",
        lookback_calendar_days=_pinned_lookback(period + 1),
        warmup_bars=period,
        required_output_bars=1,
        convergence_bars=0,
        period=period,
    )


_MACD = IndicatorSpec(
    key="macd",
    indicator_type="macd",
    interval="day",
    bounds="regular",
    adjustment_type="split",
    # Two points, not one. `momentum_stabilizing` compares this bar's histogram
    # to the previous bar's, and `latest` would disable that check in silence.
    output="last:2",
    lookback_calendar_days=_pinned_lookback(26 + 9 + 100 + 2),
    warmup_bars=35,
    required_output_bars=2,
    convergence_bars=100,
    fast_period=12,
    slow_period=26,
    signal_period=9,
)


CURRENT_ACQUISITION = MarketDataAcquisitionProfile(
    profile_id="agentic-acquisition",
    version="v1-2026-08-22",
    as_of=date(2026, 8, 22),
    historicals=HistoricalsSpec(
        tool="get_equity_historicals",
        # Explicit, never omitted -- an omitted interval lets the server pick
        # one from the range length.
        interval="day",
        # Regular hours only, matching the indicators. A bar series including
        # extended-hours prints would put `last_bar.low` on a different session
        # from the SMA it is compared against, and the intrabar stop check
        # reads exactly that low.
        bounds="regular",
        # The documented default, and the one the indicator endpoint documents
        # too. Dividend adjustment would shift these closes relative to values
        # computed server-side under a different convention, producing a
        # systematic mismatch nobody would see.
        adjustment_type="split",
        required_bars=20,
        lookback_calendar_days=_pinned_lookback(20),
    ),
    indicators=(
        _wilder("rsi", "rsi", 14, "last:2", 2),
        _MACD,
        _sma("sma_20", 20),
        _sma("sma_50", 50),
        _sma("sma_200", 200),
        _wilder("atr", "atr", 14, "latest", 1),
    ),
    quote=SingleCallSpec(
        label="quote",
        tool="get_equity_quotes",
        note=(
            "No lookback exists and none is invented. Freshness is enforced "
            "downstream by preflight, which bounds quote age from the venue "
            "timestamps rather than by anything requested here."
        ),
    ),
    earnings=SingleCallSpec(
        label="earnings",
        tool="get_earnings_results",
        note=(
            "One symbol per call. get_earnings_calendar takes no symbol "
            "argument and must never be substituted -- doing so attributed one "
            "company's report date to another. What the response *means* is "
            "EarningsCapabilities' business, versioned separately: this pins "
            "only which tool is called and how."
        ),
    ),
)
"""The acquisition contract in force.

Deliberately does **not** embed the earnings capability version. Which tool to
call is an acquisition decision and belongs here; what its answer means is a
claim about the broker, owned by `EarningsCapabilities` and already recorded on
every snapshot through `EarningsAssessment.profile_ref`. Folding that ref in
would couple two contracts that move for different reasons and force an
acquisition bump every time an earnings claim was refined.

The historicals lookback is sized for what the deterministic core reads today
(20 bars). The local-indicator milestone will compute SMA200 from these bars
and will need roughly 200 -- that is a real contract change and should arrive
as a version bump, not as a quiet widening.
"""

__all__ = [
    "CURRENT_ACQUISITION",
    "HistoricalsSpec",
    "IndicatorSpec",
    "MarketDataAcquisitionProfile",
    "SingleCallSpec",
]
