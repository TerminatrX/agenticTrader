"""Build a `MarketSnapshot` from raw Robinhood MCP tool payloads.

This module is the boundary between the agent and the deterministic core. The
orchestrating agent calls MCP tools, hands the raw JSON here unmodified, and
receives a normalized snapshot. Keeping the parsing on this side of the line —
rather than asking the agent to reshape JSON in prose — is what makes a cycle
reproducible: the same payloads always yield the same snapshot, and the
snapshot is what gets persisted for replay.

Parsers here are deliberately forgiving about *missing* data and strict about
*malformed* data. A missing RSI yields a snapshot without RSI, and strategies
handle that. A price that will not parse as a number is an error, because
silently coercing it to zero would produce a confident, wrong decision.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from agentic_trader.models import Bar, EarningsEvent, Indicators, MarketSnapshot


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


def parse_indicators(payloads: dict[str, Any]) -> Indicators:
    """Assemble indicators from separate per-indicator MCP responses.

    `payloads` maps a label to the raw response, e.g.::

        {"rsi": <rsi response>, "macd": <macd response>,
         "sma_20": ..., "sma_50": ..., "sma_200": ..., "atr": ...}
    """
    fields: dict[str, Any] = {}
    as_of: datetime | None = None

    rsi = _indicator_series(payloads.get("rsi"), "rsi")
    if rsi:
        fields["rsi_14"] = _float(rsi[-1].get("value"))
        if len(rsi) >= 2:
            fields["rsi_prev"] = _float(rsi[-2].get("value"))
        as_of = _dt(rsi[-1].get("begins_at")) or as_of

    macd = _indicator_series(payloads.get("macd"), "macd")
    if macd:
        latest = macd[-1]
        fields["macd"] = _float(latest.get("macd"))
        fields["macd_signal"] = _float(latest.get("signal"))
        fields["macd_hist"] = _float(latest.get("histogram"))
        if len(macd) >= 2:
            fields["macd_hist_prev"] = _float(macd[-2].get("histogram"))
        as_of = _dt(latest.get("begins_at")) or as_of

    for label, field in (("sma_20", "sma_20"), ("sma_50", "sma_50"), ("sma_200", "sma_200")):
        series = _indicator_series(payloads.get(label), "sma")
        if series:
            fields[field] = _dec(series[-1].get("value"), label)
            as_of = _dt(series[-1].get("begins_at")) or as_of

    atr = _indicator_series(payloads.get("atr"), "atr")
    if atr:
        fields["atr_14"] = _dec(atr[-1].get("value"), "atr")

    return Indicators(as_of=as_of, **fields)


def parse_next_earnings(payload: Any, as_of: date) -> EarningsEvent | None:
    """Find the next unreported earnings event.

    `eps.actual is None` is the reliable "has not reported yet" marker; the date
    alone is not, since a report filed this morning still carries today's date.
    """
    data = _unwrap(payload)
    if not isinstance(data, dict):
        return None
    upcoming: list[EarningsEvent] = []
    for entry in data.get("results", []):
        if not isinstance(entry, dict):
            continue
        eps = entry.get("eps") or {}
        report = entry.get("report") or {}
        if eps.get("actual") is not None:
            continue
        raw_date = report.get("date")
        if not raw_date:
            continue
        try:
            report_date = date.fromisoformat(str(raw_date))
        except ValueError:
            continue
        if report_date < as_of:
            continue
        upcoming.append(
            EarningsEvent(
                report_date=report_date,
                timing=report.get("timing"),
                eps_estimate=_dec(eps.get("estimate"), "eps.estimate"),
                verified=bool(report.get("verified", False)),
            )
        )
    return min(upcoming, key=lambda e: e.report_date) if upcoming else None


def build_snapshot(
    symbol: str,
    *,
    quote: Any = None,
    historicals: Any = None,
    fundamentals: Any = None,
    earnings: Any = None,
    indicators: dict[str, Any] | None = None,
    captured_at: datetime | None = None,
) -> MarketSnapshot:
    """Assemble a snapshot from whichever MCP payloads are available.

    Only a price is strictly required, and it may come from either the quote or
    the most recent bar.
    """
    symbol = symbol.strip().upper()
    now = captured_at or datetime.now(UTC)

    bars = parse_bars(historicals, symbol) if historicals else []
    indicator_state = parse_indicators(indicators or {})

    last_price: Decimal | None = None
    previous_close: Decimal | None = None
    tradable = True
    staleness_note: str | None = None

    quote_entry = _first_result(quote, symbol) if quote else None
    if quote_entry:
        q = quote_entry.get("quote", quote_entry)
        if isinstance(q, dict):
            last_price = _dec(q.get("last_trade_price"), "last_trade_price")
            previous_close = _dec(
                q.get("adjusted_previous_close") or q.get("previous_close"),
                "previous_close",
            )
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
        bars=bars,
        indicators=indicator_state,
        earnings=parse_next_earnings(earnings, now.date()) if earnings else None,
        tradable=tradable,
        staleness_note=staleness_note,
        **{k: v for k, v in extras.items() if v is not None},
    )
