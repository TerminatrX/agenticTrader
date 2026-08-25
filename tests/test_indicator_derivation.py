"""Deriving the strategy's indicators from one bar payload.

Two things are being held open here. First, that the derivation reproduces the
*production* window rather than an approximation of it — which turned out to
need more than slicing by timestamp. Second, that nothing it produces can reach
a decision on this branch: the cutover itself is not done, and local values
remain diagnostics until it is.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from agentic_trader.market.acquisition import CURRENT_ACQUISITION
from agentic_trader.market.indicator_derivation import (
    COMPLETED_BAR_RULE,
    CURRENT_DERIVATION,
    completed_bars,
    derive_indicators,
    duplicate_timestamps,
    envelope_covers,
    slice_from,
    window_start,
)
from agentic_trader.models import Bar
from agentic_trader.models.market_snapshot import IndicatorSource

D = Decimal
TD = date(2026, 8, 25)

DERIVATION_REF = "local-indicator-derivation@v1-2026-08-25"
DERIVATION_FINGERPRINT = (
    "45f2d4fd3b7ca091bacf8d7cc61d2ef46758f6b401639381c115f5e81ec3764b"
)


def _bar(day: date, close: str, *, interpolated: bool = False) -> Bar:
    c = D(close)
    return Bar(
        begins_at=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        open=c, high=c + 1, low=c - 1, close=c,
        volume=1_000_000, interpolated=interpolated,
    )


def _run(days: int, *, end: date = date(2026, 8, 24)) -> list[Bar]:
    """`days` consecutive calendar bars ending at `end`, gently trending."""
    return [
        _bar(end - timedelta(days=days - 1 - i), str(100 + (i % 23) * 0.5))
        for i in range(days)
    ]


# ============================================ identity


def test_the_derivation_profile_is_pinned():
    assert CURRENT_DERIVATION.profile_ref == DERIVATION_REF
    assert CURRENT_DERIVATION.content_fingerprint == DERIVATION_FINGERPRINT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_bar_rule", "something_else"),
        ("source_description", "something_else"),
    ],
)
def test_changing_a_derivation_semantic_changes_the_fingerprint(field, value):
    import dataclasses

    altered = dataclasses.replace(CURRENT_DERIVATION, **{field: value})
    assert altered.content_fingerprint != CURRENT_DERIVATION.content_fingerprint


@pytest.mark.parametrize(
    ("key", "field", "value"),
    [
        ("rsi", "source_lookback_calendar_days", 200),
        ("rsi", "warmup_bars", 0),
        ("macd", "warmup_bars", 10),
        ("sma_200", "period", 100),
        ("atr", "warmup_bars", 20),
    ],
)
def test_changing_a_spec_changes_the_fingerprint(key, field, value):
    """Warm-up is inside the hash because it moves values. It is
    reverse-engineered provider behaviour, so a change to it must be a visible
    contract change rather than a silent edit."""
    import dataclasses

    original = CURRENT_DERIVATION.spec_for(key)
    assert getattr(original, field) != value, "proves nothing"
    altered = dataclasses.replace(
        CURRENT_DERIVATION,
        specs=tuple(
            dataclasses.replace(s, **{field: value}) if s.key == key else s
            for s in CURRENT_DERIVATION.specs
        ),
    )
    assert altered.content_fingerprint != CURRENT_DERIVATION.content_fingerprint


# ============================================ legacy window fidelity


def test_derivation_lookbacks_mirror_the_production_broker_calls():
    """The cutover reproduces today's semantics, so every derivation window
    must equal the broker call it replaces. Pinned against v3 itself rather
    than restated, so the two cannot drift apart."""
    production = {s.key: s.lookback_calendar_days for s in CURRENT_ACQUISITION.indicators}
    derivation = {s.key: s.source_lookback_calendar_days for s in CURRENT_DERIVATION.specs}
    assert derivation == production == {
        "rsi": 180, "macd": 210, "sma_20": 90, "sma_50": 90,
        "sma_200": 330, "atr": 180,
    }


def test_the_window_boundary_matches_the_acquisition_start_time():
    """Slicing must reproduce the request, so the two arithmetics must agree
    exactly — not merely land on the same day."""
    plan = CURRENT_ACQUISITION.request_plan("AAPL", TD)
    for spec in CURRENT_DERIVATION.specs:
        sent = plan["calls"]["indicators"][spec.key]["params"]["start_time"]
        derived = window_start(TD, spec.source_lookback_calendar_days)
        assert derived.strftime("%Y-%m-%dT%H:%M:%SZ") == sent, spec.key


def test_slicing_is_by_timestamp_not_by_bar_count():
    """A calendar span holds a different number of sessions depending on where
    holidays fall, so a fixed count approximates the request instead of
    reproducing it. Here a gap removes bars without moving the boundary."""
    bars = [_bar(date(2026, 8, 24) - timedelta(days=i), "100") for i in range(60)][::-1]
    sparse = [b for b in bars if b.begins_at.day % 3 != 0]

    dense_slice = slice_from(bars, TD, 30)
    sparse_slice = slice_from(sparse, TD, 30)
    boundary = window_start(TD, 30)

    assert len(dense_slice) != len(sparse_slice)
    assert all(b.begins_at >= boundary for b in dense_slice + sparse_slice)


def test_warm_up_is_recorded_for_recursive_indicators_only():
    """SMA takes none: a finite window's last value cannot be moved by bars
    before its own span, and demanding a prepend would make SMA200
    unsatisfiable inside the 420-day envelope for no gain."""
    warmup = {s.key: s.warmup_bars for s in CURRENT_DERIVATION.specs}
    assert warmup == {
        "rsi": 15, "atr": 14, "macd": 34,
        "sma_20": 0, "sma_50": 0, "sma_200": 0,
    }


# ============================================ completed-bar rule


def test_todays_bar_is_excluded():
    """`Indicators.as_of` means the last *completed* bar. The provider warns its
    newest close is not the settled one, and the in-progress case was never
    observed — so the rule refuses rather than trusts."""
    bars = _run(40) + [_bar(TD, "999")]
    eligible = completed_bars(bars, TD)
    assert all(b.begins_at.date() < TD for b in eligible)
    assert D("999") not in [b.close for b in eligible]


def test_interpolated_bars_are_dropped():
    bars = _run(40)
    bars.insert(10, _bar(date(2026, 7, 20), "500", interpolated=True))
    assert all(not b.interpolated for b in completed_bars(bars, TD))


def test_the_completed_bar_rule_is_named_in_the_fingerprint():
    assert COMPLETED_BAR_RULE in "|".join(CURRENT_DERIVATION.fingerprint_items())


def test_bars_are_returned_oldest_first_regardless_of_input_order():
    bars = _run(30)
    shuffled = list(reversed(bars))
    assert completed_bars(shuffled, TD) == completed_bars(bars, TD)


# ============================================ fail-closed


def test_duplicate_timestamps_refuse_every_indicator():
    """A repeated session double-counts a delta and corrupts every recursive
    value after it. Refuse the set rather than guess which row is real."""
    bars = _run(300)
    bars.append(bars[-1])
    result = derive_indicators(bars, TD)

    assert duplicate_timestamps(bars)
    assert len(result.unavailable) == len(CURRENT_DERIVATION.specs)
    assert all("duplicate" in reason for _, reason in result.unavailable)
    assert result.indicators.rsi_14 is None


def test_an_empty_payload_produces_no_indicators_and_says_why():
    result = derive_indicators([], TD)
    assert result.indicators.as_of is None
    assert result.unavailable
    assert result.indicators.rsi_14 is None


def test_a_short_envelope_reports_which_windows_it_cannot_cover():
    """Never a shortened window, never a substituted period."""
    result = derive_indicators(_run(120), TD)
    missing = dict(result.unavailable)
    assert "sma_200" in missing
    assert "envelope" in missing["sma_200"] or "warm-up" in missing["sma_200"]
    assert result.indicators.sma_200 is None


def test_insufficient_bars_inside_a_covered_window_still_fails_closed():
    """The envelope reaches back far enough but holds too few sessions — a
    holiday-riddled span, or a thinly traded name."""
    sparse = [_bar(date(2026, 8, 24) - timedelta(days=i * 9), "100") for i in range(40)]
    result = derive_indicators(sparse[::-1], TD)
    assert result.indicators.sma_200 is None
    assert result.unavailable


def test_an_absent_indicator_is_none_and_never_zero():
    """The distinction that matters downstream: a strategy treats `None` as a
    failed condition, but would read 0.0 as a real reading and trade on it."""
    result = derive_indicators(_run(60), TD)
    missing = dict(result.unavailable)

    assert "sma_200" in missing
    assert result.indicators.sma_200 is None
    assert result.indicators.sma_200 != 0
    assert all(reason.strip() for reason in missing.values())


def test_a_genuine_zero_is_preserved_rather_than_reported_missing():
    """A flat series produces a MACD histogram of exactly zero. That is a
    value, and it must not be confused with an absent one."""
    flat = [
        _bar(date(2026, 8, 24) - timedelta(days=400 - i), "100.00")
        for i in range(400)
    ]
    result = derive_indicators(flat, TD)

    assert result.indicators.macd_hist == 0.0
    assert "macd" not in dict(result.unavailable)


# ============================================ provenance


def test_derived_indicators_carry_local_provenance():
    result = derive_indicators(_run(400), TD)
    provenance = result.indicators.provenance
    assert provenance is not None
    assert provenance.source is IndicatorSource.LOCAL
    assert provenance.profile_ref == DERIVATION_REF
    assert provenance.profile_fingerprint == DERIVATION_FINGERPRINT


def test_provenance_survives_a_snapshot_round_trip():
    """It rides in `snapshot_json`, which is why no journal column is needed."""
    from agentic_trader.models import Indicators

    result = derive_indicators(_run(400), TD)
    dumped = result.indicators.model_dump(mode="json")
    assert dumped["provenance"]["source"] == "local"

    restored = Indicators.model_validate(dumped)
    assert restored.provenance == result.indicators.provenance


def test_a_snapshot_without_provenance_stays_readable():
    """Rows written before provenance existed must not be back-filled with a
    source nobody recorded."""
    from agentic_trader.models import Indicators

    legacy = Indicators.model_validate({"rsi_14": 42.0})
    assert legacy.provenance is None
    assert legacy.rsi_14 == 42.0


def test_as_of_is_the_last_completed_bar():
    bars = _run(400) + [_bar(TD, "500")]
    result = derive_indicators(bars, TD)
    assert result.indicators.as_of.date() == date(2026, 8, 24)


# ============================================ envelope


def test_the_420_day_envelope_covers_every_derivation_window():
    """Phase 5's superset claim, asserted rather than assumed."""
    lookback = CURRENT_ACQUISITION.historicals.lookback_calendar_days
    assert lookback == 420
    bars = _run(lookback)
    assert all(envelope_covers(bars, TD).values())


@pytest.mark.parametrize("trading_day", [date(2026, 8, 25), date(2026, 3, 2), date(2026, 1, 5)])
def test_the_envelope_covers_the_windows_on_other_trading_dates(trading_day):
    bars = [
        _bar(trading_day - timedelta(days=420 - i), str(100 + (i % 17)))
        for i in range(420)
    ]
    assert all(envelope_covers(bars, trading_day).values())


# ============================================ still not wired in


def test_derivation_has_not_been_wired_into_the_decision_path():
    """The cutover is not done on this commit. Until it is, `build_snapshot`
    must still take broker indicators and no decision module may import the
    derivation."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentic_trader"
    offenders = []
    for path in (root / "strategies", root / "risk", root / "agents"):
        for file in path.rglob("*.py"):
            if "indicator_derivation" in file.read_text(encoding="utf-8"):
                offenders.append(str(file.relative_to(root)))
    assert offenders == [], offenders
