"""Normalized view of everything the system knows about one symbol at one instant.

A snapshot is the sole input to strategy evaluation. If a fact is not on the
snapshot, no strategy may consider it. That rule is what makes a decision
replayable: persist the snapshot, and you can reproduce the decision exactly.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


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


class Indicators(BaseModel):
    """Technical state as of the last completed bar.

    Every field is optional: the orchestrator may build a partial snapshot when
    an indicator call fails, and strategies are required to degrade gracefully
    rather than assume presence. `as_of` is the bar the values were computed
    through, which is normally the prior session's close, not the live tick.
    """

    model_config = ConfigDict(frozen=True)

    as_of: datetime | None = None

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


class EarningsEvent(BaseModel):
    """Next scheduled earnings report. Drives the entry blackout window."""

    model_config = ConfigDict(frozen=True)

    report_date: date
    timing: str | None = None  # "am" | "pm"
    eps_estimate: Decimal | None = None
    verified: bool = False

    def days_until(self, as_of: date) -> int:
        return (self.report_date - as_of).days


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
    earnings: EarningsEvent | None = None

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
