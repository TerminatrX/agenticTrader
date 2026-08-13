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

    bars: list[Bar] = Field(default_factory=list)
    indicators: Indicators = Field(default_factory=Indicators)
    earnings: EarningsEvent | None = None

    # Liquidity and context, sourced from fundamentals.
    average_volume_30d: Decimal | None = None
    market_cap: Decimal | None = None
    high_52w: Decimal | None = None
    low_52w: Decimal | None = None

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
