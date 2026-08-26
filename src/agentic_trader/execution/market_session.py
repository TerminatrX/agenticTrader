"""When the US equity regular session is actually open.

Exists because a claim was being leaned on without being enforced. The
completed-bar rule for local indicators is only sound if actionable decisions
happen *during* the regular session, and the justification offered for that was
`market_hours="regular_hours"` on the order payload. That field is an
instruction to the broker, not a guarantee about when this process runs: a
payload built at 20:00 can still be submitted, and the broker may queue it for
the next open. For a SELL exit there is not even a protection floor to catch it,
because that check only guards entries.

So the session is checked here, structurally, and live payload construction
refuses outside it.

Why a real calendar
-------------------

A weekday plus 09:30-16:00 rule is wrong on roughly a dozen days a year, and
wrong in the direction that matters: it would admit trading on Thanksgiving and
on Good Friday, and would treat the 13:00 early closes as three more hours of
open session. Those are exactly the thin, gappy sessions where a market order
sized off a stale indicator behaves worst.

No dependency is added for this. `pandas_market_calendars` and friends bring a
large transitive surface into a process whose defining property is that it
performs no I/O and stays auditable; the rules below are a page of arithmetic
and are tested against known dates.

Scope, stated honestly: NYSE/Nasdaq regular sessions, holidays and early
closes as scheduled. It does not model unscheduled closures (a hurricane, a
national day of mourning, a circuit-breaker halt). Those make it *more*
permissive than reality, never less — it would admit a session the exchange
cancelled. That residual is the reason live execution still requires human
confirmation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone
from enum import StrEnum

EST = timezone(timedelta(hours=-5), "EST")
EDT = timezone(timedelta(hours=-4), "EDT")


def _dst_bounds(year: int) -> tuple[datetime, datetime]:
    """US daylight saving window, as fixed by law since 2007.

    Second Sunday in March at 02:00 local standard time through the first
    Sunday in November at 02:00 local daylight time.
    """
    start = _nth_weekday(year, 3, 6, 2)
    end = _nth_weekday(year, 11, 6, 1)
    return (
        datetime.combine(start, time(2, 0), tzinfo=EST),
        datetime.combine(end, time(2, 0), tzinfo=EDT),
    )


def to_eastern(instant: datetime) -> datetime:
    """Convert to US Eastern without an IANA database.

    `zoneinfo` would be the obvious choice, but it needs the platform tz
    database or the `tzdata` package, and neither is guaranteed here -- this
    process runs on Windows, where the lookup fails outright. The rule is a
    dozen lines and has been stable since the Energy Policy Act took effect in
    2007, so it is written out rather than depended upon, which also keeps it
    inspectable alongside every other externally-imposed rule in this codebase.

    Pre-2007 dates are outside the rule's validity and outside anything this
    system trades.
    """
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    utc = instant.astimezone(UTC)
    start, end = _dst_bounds(utc.year)
    in_dst = start.astimezone(UTC) <= utc < end.astimezone(UTC)
    return utc.astimezone(EDT if in_dst else EST)

PINNED_SESSION_YEARS = frozenset({2026, 2027, 2028})
"""Years whose schedule has been checked against a published NYSE calendar.

The rules below *extrapolate* -- give them 2035 and they will confidently
produce a holiday set nobody has verified, and the exchange does move dates
(Juneteenth was added in 2022; special closures are announced ad hoc). For a
gate that admits live orders, confident extrapolation is the wrong failure
mode.

So live admission trusts only these years. Outside them the answer is
`UNSUPPORTED_CALENDAR`, which is not open, so payload construction refuses.
Extending the horizon is a deliberate edit: check the published schedule, add
the year, add its cases to the tests.

Shadow is unaffected -- analysis and replay over any period stay possible.
"""

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


class SessionStatus(StrEnum):
    """Why a moment is or is not inside the regular session."""

    OPEN = "open"
    BEFORE_OPEN = "before_open"
    AFTER_CLOSE = "after_close"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

    UNSUPPORTED_CALENDAR = "unsupported_calendar"
    """Outside the verified horizon. The recurrence rules would still produce
    an answer; it just would not be one anybody checked."""

    @property
    def is_open(self) -> bool:
        return self is SessionStatus.OPEN


def _easter(year: int) -> date:
    """Gregorian Easter Sunday. Needed only to locate Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lu = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lu) // 451
    month, day = divmod(h + lu - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """`n`th `weekday` (Mon=0) of a month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """Weekend holidays shift, the way the exchange observes them.

    Saturday moves to the preceding Friday, Sunday to the following Monday.
    New Year's Day is the one exception the exchange makes in the other
    direction: a Saturday 1 January is simply not observed rather than closing
    the previous 31 December, which is a different trading year.
    """
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def market_holidays(year: int) -> frozenset[date]:
    """Scheduled full closures for a year."""
    days = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),          # MLK Day
        _nth_weekday(year, 2, 0, 3),          # Washington's Birthday
        _easter(year) - timedelta(days=2),    # Good Friday
        _last_weekday(year, 5, 0),            # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),          # Labor Day
        _nth_weekday(year, 11, 3, 4),         # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    # Juneteenth became a market holiday in 2022; asserting it earlier would
    # mark historical sessions closed that were open.
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))

    # A New Year's Day falling on Saturday is not observed at all -- the
    # preceding Friday belongs to the previous year and stays a full session.
    if date(year, 1, 1).weekday() == 5:
        days.discard(date(year - 1, 12, 31))
        days.discard(date(year, 1, 1) - timedelta(days=1))
    return frozenset(days)


def early_closes(year: int) -> frozenset[date]:
    """Scheduled 13:00 ET closes.

    The day after Thanksgiving always. Christmas Eve and 3 July only when they
    are themselves trading days -- if 24 December is a Saturday there is no
    session to shorten, and if 4 July falls on a Monday then 3 July is a normal
    Sunday.
    """
    holidays = market_holidays(year)
    days = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}

    for candidate in (date(year, 12, 24), date(year, 7, 3)):
        if candidate.weekday() < 5 and candidate not in holidays:
            days.add(candidate)
    return frozenset(d for d in days if d.weekday() < 5 and d not in holidays)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def session_close(day: date) -> time | None:
    """Closing time for a trading day, or `None` if it is not one."""
    if not is_trading_day(day):
        return None
    return EARLY_CLOSE if day in early_closes(day.year) else REGULAR_CLOSE


def session_status(instant: datetime) -> SessionStatus:
    """Where `instant` falls relative to the regular session.

    Takes an explicit instant and never reads the clock, so a replay given the
    recorded `occurred_at` reaches the same answer forever. A naive datetime is
    read as UTC, matching how every other timestamp in this system is handled.
    """
    local = to_eastern(instant)
    day = local.date()

    if day.year not in PINNED_SESSION_YEARS:
        return SessionStatus.UNSUPPORTED_CALENDAR
    if day.weekday() >= 5:
        return SessionStatus.WEEKEND
    if day in market_holidays(day.year):
        return SessionStatus.HOLIDAY

    close = session_close(day)
    if local.time() < REGULAR_OPEN:
        return SessionStatus.BEFORE_OPEN
    if local.time() >= close:
        return SessionStatus.AFTER_CLOSE
    return SessionStatus.OPEN


def describe(instant: datetime) -> str:
    """One line for a refusal message or a journal note."""
    status = session_status(instant)
    local = to_eastern(instant)
    if status is SessionStatus.OPEN:
        return f"regular session open ({local:%Y-%m-%d %H:%M %Z})"
    if status is SessionStatus.UNSUPPORTED_CALENDAR:
        return (
            f"{local:%Y} is outside the verified session calendar "
            f"({min(PINNED_SESSION_YEARS)}-{max(PINNED_SESSION_YEARS)})"
        )
    close = session_close(local.date())
    window = (
        f"; session runs {REGULAR_OPEN:%H:%M}-{close:%H:%M} ET" if close else ""
    )
    return f"{status.value} at {local:%Y-%m-%d %H:%M %Z}{window}"


__all__ = [
    "EARLY_CLOSE",
    "PINNED_SESSION_YEARS",
    "EDT",
    "EST",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "SessionStatus",
    "describe",
    "early_closes",
    "is_trading_day",
    "market_holidays",
    "session_close",
    "session_status",
    "to_eastern",
]
