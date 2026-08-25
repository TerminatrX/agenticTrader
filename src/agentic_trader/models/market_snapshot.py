"""Normalized view of everything the system knows about one symbol at one instant.

A snapshot is the sole input to strategy evaluation. If a fact is not on the
snapshot, no strategy may consider it. That rule is what makes a decision
replayable: persist the snapshot, and you can reproduce the decision exactly.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class Bar(BaseModel):
    """One OHLCV bar. Mirrors get_equity_historicals output."""

    model_config = ConfigDict(frozen=True)

    begins_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    interpolated: bool = False


class IndicatorSource(StrEnum):
    """Where a snapshot's indicator values actually came from."""

    BROKER = "broker"
    """Returned by `get_equity_technical_indicators`, one call per indicator."""

    LOCAL = "local"
    """Computed here from the authoritative historical bars in this snapshot."""


class IndicatorProvenance(BaseModel):
    """Which contract produced these indicator values.

    Once indicators are derived rather than fetched, "which acquisition version
    was in force" no longer explains how bars became RSI. The arithmetic, the
    seed conventions, the per-indicator source windows and the completed-bar
    rule are all decision-affecting, and they live in a profile of their own --
    so the snapshot records that profile rather than leaving a reader to infer
    it from an acquisition version number.

    Carried on `Indicators`, so it lands in `snapshot_json` automatically and
    needs no journal column. Nothing queries provenance across rows today; when
    something does, that is the moment to justify an audit column, not before.
    """

    model_config = ConfigDict(frozen=True)

    source: IndicatorSource
    profile_ref: str
    profile_fingerprint: str


class Indicators(BaseModel):
    """Technical state as of the last completed bar.

    Every field is optional: the orchestrator may build a partial snapshot when
    an indicator call fails, and strategies are required to degrade gracefully
    rather than assume presence. `as_of` is the bar the values were computed
    through, which is normally the prior session's close, not the live tick.
    """

    model_config = ConfigDict(frozen=True)

    as_of: datetime | None = None

    # `None` on rows written before provenance existed. Deliberately not
    # defaulted to BROKER: those snapshots predate the distinction, and
    # stamping them with a source nobody recorded would invent evidence.
    provenance: IndicatorProvenance | None = None

    rsi_14: float | None = None
    rsi_prev: float | None = None

    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    macd_hist_prev: float | None = None

    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None

    atr_14: Decimal | None = None

    @property
    def macd_hist_improving(self) -> bool | None:
        """True when the histogram is rising — a pullback losing downside momentum.

        Returns None when either reading is missing, which callers must treat as
        "unknown", never as False.
        """
        if self.macd_hist is None or self.macd_hist_prev is None:
            return None
        return self.macd_hist > self.macd_hist_prev

    @property
    def rsi_improving(self) -> bool | None:
        if self.rsi_14 is None or self.rsi_prev is None:
            return None
        return self.rsi_14 > self.rsi_prev


def _norm_symbol(value: str) -> str:
    """One spelling of a ticker, so identity comparisons cannot miss on case.

    Both models normalise, so `event.symbol == assessment.symbol` compares like
    with like no matter which construction path produced them.
    """
    return value.strip().upper()


class EarningsStatus(StrEnum):
    """What we were able to establish about a symbol's next earnings report.

    Three states, not two. The whole class of bug this replaces came from
    collapsing "we know there is nothing scheduled" together with "we could not
    find out" — the first is a fact about the company, the second is a fact
    about our data, and only the first may permit an entry.
    """

    UPCOMING = "upcoming"
    """An authoritative future-dated report exists for this symbol."""

    NONE_SCHEDULED = "none_scheduled"
    """The source resolved the symbol and shows no report on or after the
    evaluation date. Authoritative absence."""

    UNKNOWN = "unknown"
    """No usable answer: payload missing, malformed, symbol unresolved, or the
    source is not capable of a per-symbol answer. **Blocks new entries.**"""


class EarningsEvent(BaseModel):
    """One scheduled earnings report, tied to the symbol it belongs to.

    `symbol` is mandatory and is not decoration. Without it an event parsed from
    a market-wide payload is indistinguishable from the right one, which is
    exactly how a snapshot for NVO came to carry NVZMY's report date.

    Deliberately a `date` and not a `datetime`. The broker publishes a bare
    calendar date with no time and no timezone; inventing midnight to satisfy a
    type would manufacture a precision the source does not have, and any
    downstream comparison would silently pick up the local zone.
    """

    model_config = ConfigDict(frozen=True)

    symbol: Annotated[str, AfterValidator(_norm_symbol)]
    report_date: date
    timing: str | None = None  # "am" | "pm" | None when the broker omits it
    eps_estimate: Decimal | None = None
    verified: bool = False

    def days_until(self, as_of: date) -> int:
        return (self.report_date - as_of).days


class EarningsAssessment(BaseModel):
    """The normalized answer the blackout gate reads, and its provenance.

    Carries enough to reconstruct the decision months later without calling the
    broker again: which symbol, as of which date, from which source and
    capability profile, and — when the answer was UNKNOWN — why.
    """

    model_config = ConfigDict(frozen=True)

    symbol: Annotated[str, AfterValidator(_norm_symbol)]
    status: EarningsStatus
    as_of: date
    source: str
    profile_ref: str
    event: EarningsEvent | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _enforce_status_event_invariants(self) -> EarningsAssessment:
        """Make the impossible combinations unconstructable.

        The gate reads `status` and `event` as a pair. A model that permits
        UPCOMING-with-no-event, or NONE_SCHEDULED-carrying-an-event, hands the
        gate a contradiction to interpret — and every interpretation of a
        contradiction is a guess. Rejecting them here means the gate's
        defence-in-depth checks are a second line rather than the only one.
        """
        if self.status is EarningsStatus.UPCOMING:
            if self.event is None:
                raise ValueError("UPCOMING assessment must carry an event")
            if self.event.symbol != self.symbol:
                raise ValueError(
                    f"event is for {self.event.symbol}, assessment is for {self.symbol}"
                )
            if self.event.report_date < self.as_of:
                # "Upcoming" and "already happened" are not compatible claims.
                # Left unchecked this yields a negative `days_until`, which
                # misses the 0..N blackout test while still satisfying the
                # `<= N + 5` warn test -- so a past report would clear the gate
                # with an "earnings in -1d" note. Same-day stays valid: it is
                # upcoming until it is reported, and the blackout catches it.
                raise ValueError(
                    f"UPCOMING event {self.event.report_date} is before the "
                    f"assessment date {self.as_of}"
                )
        elif self.event is not None:
            raise ValueError(f"{self.status.value} assessment must not carry an event")

        if self.status is EarningsStatus.UNKNOWN and not (self.reason or "").strip():
            raise ValueError("UNKNOWN assessment must record a reason")
        return self

    @property
    def is_authoritative(self) -> bool:
        return self.status is not EarningsStatus.UNKNOWN

    def days_until(self) -> int | None:
        return self.event.days_until(self.as_of) if self.event else None


class MarketSnapshot(BaseModel):
    """Everything known about one symbol at `captured_at`."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    captured_at: datetime

    last_price: Decimal
    previous_close: Decimal | None = None

    # When the broker's venue printed the trade `last_price` came from — not
    # when we built this object. `captured_at` cannot answer "is this quote
    # stale?", because it is set to `now()` at construction and so is always
    # fresh by definition. `None` means the payload carried no timestamp, which
    # callers must treat as unknown rather than as recent.
    quote_as_of: datetime | None = None

    # Top of book, and when it was printed. Tracked separately from the trade
    # time: outside regular hours a quote can carry a recent book and a stale
    # last print, or the reverse.
    bid: Decimal | None = None
    ask: Decimal | None = None
    book_as_of: datetime | None = None

    bars: list[Bar] = Field(default_factory=list)
    indicators: Indicators = Field(default_factory=Indicators)
    # The normalized earnings answer, not a raw event. `None` means no
    # assessment was attached at all, and the risk gate treats that exactly
    # like UNKNOWN — a snapshot that was never asked the question cannot be
    # evidence that the answer was reassuring.
    earnings: EarningsAssessment | None = None

    # Liquidity and context, sourced from fundamentals.
    average_volume_30d: Decimal | None = None
    market_cap: Decimal | None = None
    high_52w: Decimal | None = None
    low_52w: Decimal | None = None

    # Drives the sector-exposure cap. `None` means fundamentals were absent or
    # did not carry it — the gate reports that rather than assuming diversity.
    sector: str | None = None
    industry: str | None = None

    # Set when the quote is stale or the symbol is not actively trading.
    tradable: bool = True
    staleness_note: str | None = None

    @property
    def last_bar(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    @property
    def reference_price(self) -> Decimal:
        """Price to base decisions on.

        Indicators are computed through the last completed bar, so pairing them
        with a live intraday tick compares values from two different instants.
        Strategies use the last bar's close when bars are present, keeping the
        whole decision internally consistent; `last_price` is reserved for
        sizing and slippage checks at execution time.
        """
        bar = self.last_bar
        return bar.close if bar is not None else self.last_price

    @property
    def spread_pct(self) -> Decimal | None:
        """Bid/ask spread as a fraction of the mid, or None when unknowable.

        Returns `None` — never `Decimal("0")` — for an unusable book. The broker
        documents zero bid/ask as its "no book" sentinel, and a crossed or
        locked book (ask <= bid) is data we cannot interpret. Reporting any of
        those as a zero spread would pass the tightest possible check on the
        worst possible information, which is the failure mode the spread gate
        exists to prevent. Callers must treat None as "refuse", not "fine".
        """
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0 or self.ask <= self.bid:
            return None
        mid = (self.bid + self.ask) / Decimal("2")
        return (self.ask - self.bid) / mid

    def quote_age_seconds(self, now: datetime) -> float | None:
        """Seconds since the venue printed this price. None when unknown.

        This is the real freshness question for a market order. Contrast
        `captured_at`, which only measures how long ago this process assembled
        the snapshot.
        """
        if self.quote_as_of is None:
            return None
        return (now - self.quote_as_of).total_seconds()
