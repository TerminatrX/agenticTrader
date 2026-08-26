"""Build a `MarketSnapshot` from raw Robinhood MCP tool payloads.

This module is the boundary between the agent and the deterministic core. The
orchestrating agent calls MCP tools, hands the raw JSON here unmodified, and
receives a normalized snapshot. Keeping the parsing on this side of the line —
rather than asking the agent to reshape JSON in prose — is what makes a cycle
reproducible: the same payloads and the same `now` always yield the same
snapshot, and the snapshot is what gets persisted for replay.

`now` is load-bearing rather than incidental. It stamps `captured_at` and dates
the earnings assessment, so re-parsing an old payload today produces a snapshot
that is *not* the one the decision was made from. Replay reads the persisted
snapshot instead of rebuilding it.

Parsers here are deliberately forgiving about *missing* data and strict about
*malformed* data. A missing RSI yields a snapshot without RSI, and strategies
handle that. A price that will not parse as a number is an error, because
silently coercing it to zero would produce a confident, wrong decision.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from agentic_trader.market.earnings_capabilities import (
    ROBINHOOD_MCP_EARNINGS,
    EarningsCapabilities,
)
from agentic_trader.market.indicator_derivation import derive_indicators
from agentic_trader.models import (
    Bar,
    EarningsAssessment,
    EarningsEvent,
    EarningsStatus,
    Indicators,
    MarketSnapshot,
)


class SnapshotError(ValueError):
    """Raised when a payload is present but cannot be interpreted."""


def _unwrap(payload: Any) -> Any:
    """MCP responses nest the useful part under `data`; tolerate either form."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _dec(value: Any, field: str) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise SnapshotError(f"Could not parse {field!r} as a number: {value!r}") from exc


def _req_dec(value: Any, field: str) -> Decimal:
    result = _dec(value, field)
    if result is None:
        raise SnapshotError(f"Required field {field!r} is missing")
    return result


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive(value: Decimal | None) -> Decimal | None:
    """Treat non-positive prices as absent.

    The broker's quote guide says to drop bid/ask when zero: zero is its "no
    book right now" sentinel, not a real price. Carrying it through would make
    an absent book look like an infinitely tight one.
    """
    return value if value is not None and value > 0 else None


def _select_last_price(q: dict[str, Any]) -> tuple[Decimal | None, datetime | None]:
    """Pick the current price the way the broker documents it.

    The quote carries two prints — the regular-session `last_trade_price` and
    the extended-hours `last_non_reg_trade_price` — each with its own venue
    timestamp, and the more recent one is the live price. Reading only the
    regular print means that after the close we quote a price hours old while
    the book has moved on; the returned timestamp is what makes that detectable
    downstream, so price and clock always come from the same print.
    """
    candidates = [
        (_dec(q.get("last_trade_price"), "last_trade_price"), _dt(q.get("venue_last_trade_time"))),
        (
            _dec(q.get("last_non_reg_trade_price"), "last_non_reg_trade_price"),
            _dt(q.get("venue_last_non_reg_trade_time")),
        ),
    ]
    priced = [(p, t) for p, t in candidates if p is not None]
    if not priced:
        return None, None

    timed = [(p, t) for p, t in priced if t is not None]
    if timed:
        return max(timed, key=lambda pair: pair[1])
    # Prices but no timestamps: fall back to the regular print and report an
    # unknown age rather than inventing one.
    return priced[0][0], None


def _first_result(payload: Any, symbol: str) -> dict[str, Any] | None:
    """Pull this symbol's entry out of a multi-symbol MCP response."""
    data = _unwrap(payload)
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("symbol") == symbol:
            return entry
        quote = entry.get("quote")
        if isinstance(quote, dict) and quote.get("symbol") == symbol:
            return entry
    # Single-symbol calls need not echo the symbol back.
    return results[0] if isinstance(results[0], dict) else None


def parse_bars(historicals: Any, symbol: str) -> list[Bar]:
    entry = _first_result(historicals, symbol)
    if not entry:
        return []
    bars: list[Bar] = []
    for raw in entry.get("bars", []):
        begins_at = _dt(raw.get("begins_at"))
        if begins_at is None:
            continue
        bars.append(
            Bar(
                begins_at=begins_at,
                open=_req_dec(raw.get("open_price"), "open_price"),
                high=_req_dec(raw.get("high_price"), "high_price"),
                low=_req_dec(raw.get("low_price"), "low_price"),
                close=_req_dec(raw.get("close_price"), "close_price"),
                volume=int(raw.get("volume") or 0),
                interpolated=bool(raw.get("interpolated", False)),
            )
        )
    bars.sort(key=lambda b: b.begins_at)
    # Interpolated bars are gap-fill and carry no new information; including
    # them would let a synthetic price trigger a real order.
    return [b for b in bars if not b.interpolated]


def _indicator_series(payload: Any, kind: str) -> list[dict[str, Any]]:
    data = _unwrap(payload)
    if not isinstance(data, dict):
        return []
    for indicator in data.get("indicators", []):
        if isinstance(indicator, dict) and indicator.get("type") == kind:
            series = indicator.get("series")
            return series if isinstance(series, list) else []
    return []


def _stamp(stamps: list[datetime], entry: dict[str, Any]) -> None:
    when = _dt(entry.get("begins_at"))
    if when is not None:
        stamps.append(when)


def parse_indicators(payloads: dict[str, Any]) -> Indicators:
    """Assemble indicators from separate per-indicator MCP responses.

    `payloads` maps a label to the raw response, e.g.::

        {"rsi": <rsi response>, "macd": <macd response>,
         "sma_20": ..., "sma_50": ..., "sma_200": ..., "atr": ...}
    """
    fields: dict[str, Any] = {}
    # Every indicator's own bar timestamp. The snapshot reports the *oldest* of
    # them, because the honest answer to "how current is this view?" is set by
    # the stalest input being relied on, not the freshest. Last-writer-wins
    # would let one fresh series mask a stale one — a fresh ATR hiding an RSI
    # computed a week ago, say — and the critic's staleness check reads this.
    stamps: list[datetime] = []

    rsi = _indicator_series(payloads.get("rsi"), "rsi")
    if rsi:
        fields["rsi_14"] = _float(rsi[-1].get("value"))
        if len(rsi) >= 2:
            fields["rsi_prev"] = _float(rsi[-2].get("value"))
        _stamp(stamps, rsi[-1])

    macd = _indicator_series(payloads.get("macd"), "macd")
    if macd:
        latest = macd[-1]
        fields["macd"] = _float(latest.get("macd"))
        fields["macd_signal"] = _float(latest.get("signal"))
        fields["macd_hist"] = _float(latest.get("histogram"))
        if len(macd) >= 2:
            fields["macd_hist_prev"] = _float(macd[-2].get("histogram"))
        _stamp(stamps, latest)

    for label, field in (("sma_20", "sma_20"), ("sma_50", "sma_50"), ("sma_200", "sma_200")):
        series = _indicator_series(payloads.get(label), "sma")
        if series:
            fields[field] = _dec(series[-1].get("value"), label)
            _stamp(stamps, series[-1])

    atr = _indicator_series(payloads.get("atr"), "atr")
    if atr:
        # ATR is quoted in dollars per share, not as a percentage — stop
        # construction converts it against the price.
        fields["atr_14"] = _dec(atr[-1].get("value"), "atr")
        _stamp(stamps, atr[-1])

    return Indicators(as_of=min(stamps) if stamps else None, **fields)


def parse_sectors(payloads: Any) -> dict[str, str | None]:
    """Map symbol -> sector from one or more batched `get_equity_fundamentals`.

    Authoritative, unlike anything a discovery source reports. This is the same
    field `build_snapshot` puts on `MarketSnapshot.sector` and the same one
    `risk.limits` gates concentration on, so the enrichment-budget selector and
    the risk cap share one taxonomy rather than two that drift apart.

    `get_equity_fundamentals` batches ten symbols per call, which is why sector
    can be fetched for every discovered candidate while the per-symbol
    indicators cannot.
    """
    out: dict[str, str | None] = {}
    for payload in payloads if isinstance(payloads, list) else [payloads]:
        data = _unwrap(payload)
        if not isinstance(data, dict):
            continue
        for entry in data.get("results") or []:
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            sector = entry.get("sector")
            out[str(symbol).strip().upper()] = str(sector).strip() if sector else None
    return out


def assess_earnings(
    payload: Any,
    symbol: str,
    as_of: date,
    *,
    capabilities: EarningsCapabilities = ROBINHOOD_MCP_EARNINGS,
) -> EarningsAssessment:
    """Normalize a `get_earnings_results` response into a blackout answer.

    Always returns an assessment — never `None`, and never raises. Every path
    that cannot establish an answer returns `UNKNOWN` with a reason, so the gate
    has something explicit to refuse on and the journal has something to show.

    Three rules earned by observation rather than assumed:

    1. **Rows for other symbols are discarded.** Every entry must carry a
       `symbol` matching the one asked about. A payload with no matching rows is
       `UNKNOWN`, never "nothing scheduled" — that distinction is the entire
       reason this function exists.
    2. **Pendingness comes from the date, not from `eps.actual`.** That field is
       unreliable in both directions (see `EARNINGS_ACTUAL_NOTE`). Using the
       date can only over-block.
    3. **A malformed date invalidates the whole assessment.** Skipping an
       unparseable row could drop the very event that mattered and leave the
       remaining rows looking like a clean bill of health.
    """
    if not capabilities.usable_for_blackout:
        return _unknown(
            symbol, as_of, capabilities,
            "earnings source not capable of a per-symbol answer",
        )

    data = _unwrap(payload)
    if not isinstance(data, dict):
        return _unknown(symbol, as_of, capabilities, "earnings payload missing or not an object")

    wanted = symbol.strip().upper()
    if wanted in {str(x).strip().upper() for x in (data.get("not_found") or [])}:
        return _unknown(symbol, as_of, capabilities, f"broker could not resolve symbol {wanted}")

    results = data.get("results")
    if not isinstance(results, list):
        return _unknown(symbol, as_of, capabilities, "earnings payload has no results list")

    mine: list[EarningsEvent] = []
    for entry in results:
        if not isinstance(entry, dict):
            return _unknown(
                symbol, as_of, capabilities, "earnings payload contains a malformed entry"
            )
        if str(entry.get("symbol", "")).strip().upper() != wanted:
            continue  # another company's row; never ours to interpret
        report = entry.get("report")
        if report is None:
            report = {}
        if not isinstance(report, dict):
            return _unknown(symbol, as_of, capabilities, "malformed report object")
        raw_date = report.get("date")
        try:
            report_date = date.fromisoformat(str(raw_date))
        except (TypeError, ValueError):
            return _unknown(
                symbol, as_of, capabilities, f"unparseable report date {raw_date!r}"
            )
        if report_date < as_of:
            continue  # already happened; the gate looks forward only
        eps = entry.get("eps")
        if eps is None:
            eps = {}
        if not isinstance(eps, dict):
            return _unknown(symbol, as_of, capabilities, "malformed eps object")
        timing = report.get("timing")
        if timing is not None and not isinstance(timing, str):
            return _unknown(symbol, as_of, capabilities, f"malformed timing {timing!r}")
        try:
            event = EarningsEvent(
                symbol=wanted,
                report_date=report_date,
                timing=timing,
                eps_estimate=_dec(eps.get("estimate"), "eps.estimate"),
                # Only a real boolean counts as confirmation. `bool("no")` is
                # True, so coercing a string here would upgrade a tentative
                # date to confirmed -- wrong in the unsafe direction.
                verified=report.get("verified") is True,
            )
        except Exception as exc:  # noqa: BLE001 - any construction failure is ignorance
            # This function's contract is that it always returns an assessment.
            # A validation error escaping here would surface as a crash in the
            # middle of snapshot building, which is a far worse failure mode
            # than a blocked entry.
            return _unknown(
                symbol, as_of, capabilities, f"could not normalize earnings entry: {exc}"
            )
        mine.append(event)

    if not any(str(e.get("symbol", "")).strip().upper() == wanted
               for e in results if isinstance(e, dict)):
        # The symbol is absent from a response that resolved fine. That is a
        # market-wide payload, a stale ticker, or the wrong tool -- all of which
        # are ignorance, not an all-clear.
        return _unknown(symbol, as_of, capabilities, f"{wanted} absent from earnings response")

    if not mine:
        # Rows for this symbol exist and every one is behind us. Whether that
        # *means* nothing is coming is a separate claim about the source, and
        # one nobody has established -- the endpoint can return future events,
        # but "can" is not "always does when one exists". Until that is
        # evidenced, this is ignorance rather than an all-clear.
        if not capabilities.future_event_absence_authoritative.usable:
            return _unknown(
                symbol, as_of, capabilities,
                f"{wanted} has no future-dated report and the source is not "
                "established as authoritative about absence",
            )
        return EarningsAssessment(
            symbol=wanted, status=EarningsStatus.NONE_SCHEDULED, as_of=as_of,
            source=capabilities.source_tool, profile_ref=capabilities.profile_ref,
        )

    # Nearest first. Ties broken on the full tuple so a symbol carrying two rows
    # for one date resolves identically on every replay.
    nearest = min(mine, key=lambda e: (e.report_date, e.timing or "", not e.verified))
    return EarningsAssessment(
        symbol=wanted, status=EarningsStatus.UPCOMING, as_of=as_of,
        source=capabilities.source_tool, profile_ref=capabilities.profile_ref,
        event=nearest,
    )


def _unknown(
    symbol: str, as_of: date, capabilities: EarningsCapabilities, reason: str
) -> EarningsAssessment:
    return EarningsAssessment(
        symbol=symbol.strip().upper(),
        status=EarningsStatus.UNKNOWN,
        as_of=as_of,
        source=capabilities.source_tool,
        profile_ref=capabilities.profile_ref,
        reason=reason,
    )


def build_snapshot(
    symbol: str,
    *,
    quote: Any = None,
    historicals: Any = None,
    fundamentals: Any = None,
    earnings: Any = None,
    trading_date: date | None = None,
    captured_at: datetime | None = None,
    broker_indicators: dict[str, Any] | None = None,
) -> MarketSnapshot:
    """Assemble a snapshot from whichever MCP payloads are available.

    Only a price is strictly required, and it may come from either the quote or
    the most recent bar.

    **Indicators are derived here, not fetched.** They come from the same
    historical bars this snapshot already carries, through
    `indicator_derivation`, which reproduces the windows the six broker
    indicator calls used to request. `trading_date` decides which bars count as
    complete and where each window starts, so it is required for indicators to
    exist at all -- omitting it yields a snapshot with none rather than one
    computed against an assumed date.

    `broker_indicators` is a **testing and comparison** path only. It exists so
    a baseline snapshot can be built from the old payloads for parity work, and
    it is deliberately *not* a fallback: production passes bars and gets local
    values, or gets nothing. A silent fallback would mean two production
    semantics and would quietly reintroduce six calls per symbol.
    """
    symbol = symbol.strip().upper()
    now = captured_at or datetime.now(UTC)

    bars = parse_bars(historicals, symbol) if historicals else []

    if broker_indicators is not None:
        indicator_state = parse_indicators(broker_indicators)
    elif trading_date is not None:
        indicator_state = derive_indicators(bars, trading_date).indicators
    else:
        indicator_state = Indicators()

    last_price: Decimal | None = None
    previous_close: Decimal | None = None
    quote_as_of: datetime | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    book_as_of: datetime | None = None
    tradable = True
    staleness_note: str | None = None

    quote_entry = _first_result(quote, symbol) if quote else None
    if quote_entry:
        q = quote_entry.get("quote", quote_entry)
        if isinstance(q, dict):
            last_price, quote_as_of = _select_last_price(q)
            previous_close = _dec(
                q.get("adjusted_previous_close") or q.get("previous_close"),
                "previous_close",
            )
            bid = _positive(_dec(q.get("bid_price"), "bid_price"))
            ask = _positive(_dec(q.get("ask_price"), "ask_price"))
            bid_time = _dt(q.get("venue_bid_time"))
            ask_time = _dt(q.get("venue_ask_time"))
            # The older of the two sides is how stale the book is as a whole.
            times = [t for t in (bid_time, ask_time) if t is not None]
            book_as_of = min(times) if times else None
            state = q.get("state")
            if q.get("has_traded") is False or (state and state != "active"):
                tradable = False
                staleness_note = f"quote state={state!r}, has_traded={q.get('has_traded')!r}"
        close = quote_entry.get("close")
        if isinstance(close, dict) and previous_close is None:
            previous_close = _dec(close.get("price"), "close.price")

    if last_price is None and bars:
        last_price = bars[-1].close
        staleness_note = staleness_note or "no live quote; using last completed bar close"
    if last_price is None:
        raise SnapshotError(f"No price available for {symbol}: pass a quote or historicals")

    extras: dict[str, Any] = {}
    fundamental_entry = _first_result(fundamentals, symbol) if fundamentals else None
    if fundamental_entry:
        extras["average_volume_30d"] = _dec(
            fundamental_entry.get("average_volume_30_days"), "average_volume_30_days"
        )
        extras["market_cap"] = _dec(fundamental_entry.get("market_cap"), "market_cap")
        extras["high_52w"] = _dec(fundamental_entry.get("high_52_weeks"), "high_52_weeks")
        extras["low_52w"] = _dec(fundamental_entry.get("low_52_weeks"), "low_52_weeks")
        # Free-text from the broker (e.g. "Electronic Technology"). Used only
        # for grouping, so its exact taxonomy does not matter — only that the
        # same name maps to the same bucket consistently.
        sector = fundamental_entry.get("sector")
        extras["sector"] = str(sector).strip() if sector else None
        industry = fundamental_entry.get("industry")
        extras["industry"] = str(industry).strip() if industry else None

    return MarketSnapshot(
        symbol=symbol,
        captured_at=now,
        last_price=last_price,
        previous_close=previous_close,
        quote_as_of=quote_as_of,
        bid=bid,
        ask=ask,
        book_as_of=book_as_of,
        bars=bars,
        indicators=indicator_state,
        earnings=assess_earnings(earnings, symbol, now.date()),
        tradable=tradable,
        staleness_note=staleness_note,
        **{k: v for k, v in extras.items() if v is not None},
    )
