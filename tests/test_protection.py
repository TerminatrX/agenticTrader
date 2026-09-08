"""Protective-stop feasibility, capability provenance, and the entry gate.

The finding these tests encode: a fractional position cannot carry a resting
protective stop, so at this account size every position is unprotectable. That
is honoured, but it could not be confirmed empirically — `review_equity_order`
accepted a fractional `stop_market` sell without complaint, while also accepting
a short sale in an account holding none of the symbol, which suggests it does
not validate order parameters at all. The uncertainty lives in
`Capability.evidence`, never in the protection state.

The gate these tests defend is asymmetric on purpose: shadow may carry an
unprotected position because it costs nothing and the record is worth having;
live may not, and no configuration value can change that.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agentic_trader.execution.capabilities import (
    ROBINHOOD_MCP,
    BrokerCapabilities,
    Capability,
    Evidence,
)
from agentic_trader.execution.executor import (
    PreflightError,
    ProtectionNotPlaceable,
    assess_protection,
    build_order_payload,
    build_protective_stop_payload,
)
from agentic_trader.execution.shadow_executor import SHADOW_PRODUCIBLE_STATES, ShadowExecutor
from agentic_trader.journal import JournalRepository
from agentic_trader.journal.models import TradeRecord
from agentic_trader.models import LIVE_PERMITTED_PROTECTION, ProtectionState, Side
from agentic_trader.risk.engine import RiskEngine

NOW = datetime(2026, 8, 13, 17, 44, tzinfo=UTC)


def _profile(**overrides) -> BrokerCapabilities:
    """ROBINHOOD_MCP with individual capabilities swapped out."""
    base = {
        "profile_id": "test",
        "version": "0",
        "as_of": date(2026, 8, 13),
        "market_orders": ROBINHOOD_MCP.market_orders,
        "limit_orders": ROBINHOOD_MCP.limit_orders,
        "stop_market_orders": ROBINHOOD_MCP.stop_market_orders,
        "stop_limit_orders": ROBINHOOD_MCP.stop_limit_orders,
        "fractional_market_orders": ROBINHOOD_MCP.fractional_market_orders,
        "fractional_stop_orders": ROBINHOOD_MCP.fractional_stop_orders,
        "gtc_orders": ROBINHOOD_MCP.gtc_orders,
        "cancel_orders": ROBINHOOD_MCP.cancel_orders,
        "replace_orders": ROBINHOOD_MCP.replace_orders,
        "bracket_orders": ROBINHOOD_MCP.bracket_orders,
        "stops_regular_hours_only": ROBINHOOD_MCP.stops_regular_hours_only,
    }
    base.update(overrides)
    return BrokerCapabilities(**base)


# ------------------------------------------------------- integrality, not size


@pytest.mark.parametrize(
    ("quantity", "protectable"),
    [
        ("0.5", False),
        ("0.999999", False),   # a `>= 1` check gets this one right by luck
        ("1", True),
        ("1.000000", True),    # trailing zeros are still a whole share
        ("1.5", False),        # a `>= 1` check gets this one WRONG
        ("2", True),
    ],
)
def test_protection_requires_a_whole_share_count(quantity, protectable):
    """A stop order carries a quantity like any other order.

    1.5 shares is a fractional order exactly as 0.5 is, so the test is
    integrality rather than size. `quantity >= 1` would wave 1.5 through.
    """
    result = ROBINHOOD_MCP.protection_feasibility(Decimal(quantity))
    assert result.protectable is protectable, result.reason


def test_zero_and_negative_quantities_are_not_protectable():
    assert ROBINHOOD_MCP.protection_feasibility(Decimal("0")).protectable is False
    assert ROBINHOOD_MCP.protection_feasibility(Decimal("-1")).protectable is False


# ----------------------------------------------------------------- provenance


def test_the_fractional_stop_restriction_is_documented_not_verified():
    """Guards the honesty of the claim, not the claim itself.

    If someone later verifies this against the broker they should update the
    evidence deliberately, and this test is what makes that a conscious act.
    """
    cap = ROBINHOOD_MCP.fractional_stop_orders
    assert cap.supported is False
    assert cap.evidence is Evidence.SCHEMA_DOCUMENTED
    assert not cap.is_certain


def test_unknown_support_is_not_permission():
    """`None` must never read as a yes."""
    unknown = Capability(supported=None, evidence=Evidence.UNKNOWN)
    assert unknown.usable is False
    assert unknown.is_certain is False


def test_unknown_fractional_stop_support_fails_closed():
    """An unestablished capability gates exactly like a known-absent one."""
    profile = _profile(
        fractional_stop_orders=Capability(None, Evidence.UNKNOWN, "never established")
    )
    result = profile.protection_feasibility(Decimal("0.039"))

    assert result.protectable is None
    assert result.certainly_impossible is False  # unknown is not proven-impossible
    assert "unknown" in result.reason


def test_capability_profile_is_versioned():
    """Historical decisions must stay attributable to what was known then."""
    assert ROBINHOOD_MCP.profile_ref == "robinhood-mcp@2026-08-14"


def test_the_profile_ref_identifies_exactly_one_set_of_claims():
    """Pins content to version, so a claim cannot change without a bump.

    A journal row storing `robinhood-mcp@2026-08-14` is only reconstructible if
    that string means one thing forever. Editing a capability in place without
    bumping `version` would silently repoint every past decision at claims that
    were never used to make it.

    If this fails: you changed a capability. Bump `version` and `as_of` on
    ROBINHOOD_MCP, then update the values here — deliberately, in the same
    commit, so the change is reviewable.
    """
    assert ROBINHOOD_MCP.content_fingerprint == (
        "0263726da167b08776283676e453ce5ea4146bbc03ab72760a810c954624a9d7"
    )


def test_changing_a_capability_changes_the_fingerprint():
    """The guard above is only meaningful if the hash actually moves."""
    flipped = _profile(
        fractional_stop_orders=Capability(True, Evidence.EMPIRICALLY_VERIFIED, "hypothetical")
    )
    assert flipped.content_fingerprint != ROBINHOOD_MCP.content_fingerprint

    # Prose is not a claim: improving a note must not invalidate history.
    reworded = _profile(
        fractional_stop_orders=Capability(
            ROBINHOOD_MCP.fractional_stop_orders.supported,
            ROBINHOOD_MCP.fractional_stop_orders.evidence,
            "same claim, different wording",
        )
    )
    assert reworded.content_fingerprint == ROBINHOOD_MCP.content_fingerprint


# ----------------------------------------------------- state derivation


def _entry_intent(entry_signal, snapshot, account, risk_config):
    decision = RiskEngine(risk_config).evaluate(
        entry_signal, snapshot, account, as_of=date(2026, 8, 13)
    )
    assert decision.intent is not None
    return decision


def test_a_fractional_entry_is_unavailable_with_a_reason(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    state, note = assess_protection(decision.intent)

    assert state is ProtectionState.UNAVAILABLE
    assert "fractional" in note
    assert "type=market" in note


def test_a_whole_share_entry_is_pending_not_protected(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Placeable is not placed.

    Nothing submits the stop yet, so a position that *could* be protected still
    is not. Calling it PROTECTED would be the exact false assurance this
    tracking exists to prevent.
    """
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    whole = decision.intent.model_copy(
        update={"notional": Decimal("302.25"), "reference_price": Decimal("302.25")}
    )
    state, note = assess_protection(whole)

    assert state is ProtectionState.PENDING
    assert "not implemented" in note


def test_exits_and_stopless_intents_need_no_protection(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)

    sell = decision.intent.model_copy(update={"side": Side.SELL})
    assert assess_protection(sell)[0] is ProtectionState.NOT_REQUIRED

    stopless = decision.intent.model_copy(update={"stop_price": None})
    assert assess_protection(stopless)[0] is ProtectionState.NOT_REQUIRED


# ------------------------------------------------------------------- the gate


def test_live_refuses_an_unprotectable_entry_regardless_of_config(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """The structural floor, tested against the most permissive config there is.

    `allow_unprotected_shadow_entries=True` is the loosest setting reachable,
    and it must still not admit an unprotected live entry.
    """
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)

    with pytest.raises(PreflightError, match="not provably protected"):
        build_order_payload(
            decision, bullish_pullback_snapshot, account.account_number,
            mode="live",
            allow_unprotected_shadow_entries=True,
            now=NOW,
        )


def test_shadow_refuses_when_the_flag_is_off(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)

    with pytest.raises(PreflightError, match="allow_unprotected_shadow_entries"):
        build_order_payload(
            decision, bullish_pullback_snapshot, account.account_number,
            mode="shadow",
            allow_unprotected_shadow_entries=False,
            now=NOW,
        )


def test_shadow_allows_it_and_records_why(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    plan = build_order_payload(
        decision, bullish_pullback_snapshot, account.account_number,
        mode="shadow",
        allow_unprotected_shadow_entries=True,
        now=NOW,
    )

    assert plan.protection is ProtectionState.UNAVAILABLE
    assert plan.capability_profile == "robinhood-mcp@2026-08-14"
    assert any("CANNOT BE PROTECTED" in w for w in plan.warnings)
    assert any("protection unavailable" in n for n in plan.preflight_notes)


def test_live_still_permits_an_exit(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    """Blocking a close increases risk. NOT_REQUIRED must pass the live gate."""
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    exit_decision = decision.model_copy(
        update={"intent": decision.intent.model_copy(update={"side": Side.SELL})}
    )

    plan = build_order_payload(
        exit_decision, bullish_pullback_snapshot, account.account_number,
        mode="live", now=NOW,
    )
    assert plan.protection is ProtectionState.NOT_REQUIRED


# -------------------------------------------------------- shadow state limits


def test_shadow_does_not_claim_positions_were_protected():
    assert ProtectionState.PROTECTED not in SHADOW_PRODUCIBLE_STATES


def test_shadow_executor_rejects_a_state_it_cannot_earn(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    plan = build_order_payload(
        decision, bullish_pullback_snapshot, account.account_number, mode="shadow", now=NOW
    )
    plan.protection = ProtectionState.PROTECTED  # as if the live path leaked in

    with pytest.raises(ValueError, match="cannot produce protection state"):
        ShadowExecutor().submit(plan, now=NOW)


def test_the_shadow_fill_carries_the_protection_state(
    entry_signal, bullish_pullback_snapshot, account, risk_config
):
    decision = _entry_intent(entry_signal, bullish_pullback_snapshot, account, risk_config)
    plan = build_order_payload(
        decision, bullish_pullback_snapshot, account.account_number, mode="shadow", now=NOW
    )
    fill = ShadowExecutor().submit(plan, now=NOW)

    assert fill.protection is ProtectionState.UNAVAILABLE
    assert fill.to_row()["protection"] == "unavailable"


# --------------------------------------------------------------------- journal


def test_protection_survives_a_journal_round_trip(tmp_path):
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_trade(
        TradeRecord(
            client_key="k1",
            symbol="AAPL",
            strategy="trend_pullback",
            mode="shadow",
            opened_at=NOW,
            entry_price=Decimal("302.25"),
            quantity=Decimal("0.045856"),
            notional=Decimal("13.86"),
            stop_price=Decimal("287.14"),
            protection_state=ProtectionState.UNAVAILABLE,
            capability_profile="robinhood-mcp@2026-08-14",
        )
    )
    (stored,) = repo.open_trades()

    assert stored.protection_state is ProtectionState.UNAVAILABLE
    assert stored.capability_profile == "robinhood-mcp@2026-08-14"


def test_rows_predating_protection_tracking_do_not_read_as_protected(tmp_path):
    """A NULL column means "we did not track this", never "it was covered"."""
    repo = JournalRepository(tmp_path / "j.db")
    repo.record_trade(
        TradeRecord(
            client_key="k2", symbol="AAPL", strategy="s", mode="shadow", opened_at=NOW,
            entry_price=Decimal("100"), quantity=Decimal("1"), notional=Decimal("100"),
        )
    )
    with sqlite3.connect(tmp_path / "j.db") as conn:
        conn.execute("UPDATE trades SET protection_state = NULL WHERE client_key = 'k2'")

    (stored,) = repo.open_trades()
    assert stored.protection_state is ProtectionState.NOT_REQUIRED
    assert stored.protection_state is not ProtectionState.PROTECTED


def test_protective_orders_table_carries_the_lifecycle_columns(tmp_path):
    """The shape the lifecycle writes into.

    Kept as its own assertion after the lifecycle landed: the columns below are
    the ones recovery reads, and a rename that only broke reconciliation would
    otherwise show up as a passing suite and an unprotected position.
    """
    JournalRepository(tmp_path / "j.db")
    with sqlite3.connect(tmp_path / "j.db") as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(protective_orders)")}

    assert {
        "trade_client_key", "broker_order_id", "state", "stop_price",
        "requested_quantity", "accepted_quantity", "filled_quantity",
        "submitted_at", "accepted_at", "triggered_at", "cancelled_at",
        "updated_at", "last_reconciled_at", "supersedes_protective_order_id",
    } <= columns


# --------------------------------------------------------- the stop lifecycle
#
# The states below are the ones a live position's safety actually turns on.
# Each test breaks exactly one condition, because a refusal no test can
# provoke is a refusal nobody has evidence works.


def test_submitted_is_not_permission_to_trade_live():
    """"We asked and do not know" must gate exactly as "we never asked" does.

    The allowlist is what makes this true by construction — a state added later
    is excluded until somebody deliberately admits it — so this asserts the
    property rather than the membership test that currently provides it.
    """
    assert ProtectionState.SUBMITTED not in LIVE_PERMITTED_PROTECTION
    assert ProtectionState.PENDING not in LIVE_PERMITTED_PROTECTION
    assert ProtectionState.FAILED not in LIVE_PERMITTED_PROTECTION


def _stop_payload(**overrides):
    kwargs = {
        "account_number": "123456789",
        "symbol": "F",
        "quantity": Decimal("3"),
        "stop_price": Decimal("11.40"),
        "fill_price": Decimal("12.00"),
        "entry_client_key": "2f1d7a6e-0000-5000-8000-000000000001",
    }
    kwargs.update(overrides)
    return build_protective_stop_payload(**kwargs)


def test_the_resting_stop_is_gtc_and_regular_hours():
    """A day order would expire at the close and uncover the position overnight."""
    payload = _stop_payload()

    assert payload["type"] == "stop_market"
    assert payload["side"] == "sell"
    assert payload["time_in_force"] == "gtc"
    assert payload["market_hours"] == "regular_hours"
    assert payload["quantity"] == "3"
    assert payload["stop_price"] == "11.40"


@pytest.mark.parametrize("quantity", [Decimal("0.5"), Decimal("1.5"), Decimal("0.000001")])
def test_a_fractional_quantity_gets_no_stop_payload(quantity):
    """The restriction that makes every position at this account size unprotectable."""
    with pytest.raises(ProtectionNotPlaceable, match="fractional"):
        _stop_payload(quantity=quantity)


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-1")])
def test_a_non_positive_quantity_gets_no_stop_payload(quantity):
    with pytest.raises(ProtectionNotPlaceable, match="not positive"):
        _stop_payload(quantity=quantity)


@pytest.mark.parametrize("stop", [Decimal("12.00"), Decimal("12.01")])
def test_a_stop_at_or_above_the_fill_is_refused(stop):
    """Such a stop triggers on placement and exits the position it protects.

    The broker would accept it — the schema has no opinion about where a stop
    sits relative to the market — so refusing it has to happen here.
    """
    with pytest.raises(ProtectionNotPlaceable, match="at or above"):
        _stop_payload(stop_price=stop, fill_price=Decimal("12.00"))


def test_rounding_a_stop_can_only_tighten_it():
    """A stop rounded away from the fill would raise realized risk by accident."""
    payload = _stop_payload(stop_price=Decimal("11.4501"), fill_price=Decimal("12.00"))

    # 11.4501 -> 11.46, never 11.45: the rounded stop sits closer to the fill,
    # so the loss it caps is no larger than the one the position was sized to.
    assert payload["stop_price"] == "11.46"
    assert Decimal(payload["stop_price"]) >= Decimal("11.4501")


def test_an_identical_resubmission_reuses_the_broker_idempotency_key():
    """A retry must collide at the broker rather than rest a second stop."""
    assert _stop_payload()["ref_id"] == _stop_payload()["ref_id"]


def test_a_moved_stop_is_a_different_order():
    """This broker cannot modify a resting order, so a move is a new placement."""
    original = _stop_payload(stop_price=Decimal("11.40"))
    moved = _stop_payload(stop_price=Decimal("11.60"))
    resized = _stop_payload(quantity=Decimal("2"))

    assert original["ref_id"] != moved["ref_id"]
    assert original["ref_id"] != resized["ref_id"]


def _submit(repo, client_key="stop-key-1", trade_key="entry-key-1"):
    repo.record_protection_submitted(
        trade_client_key=trade_key,
        client_key=client_key,
        stop_price=Decimal("11.40"),
        requested_quantity=Decimal("3"),
        capability_profile=ROBINHOOD_MCP.profile_ref,
        submitted_at=datetime(2026, 9, 4, 14, 30, tzinfo=UTC),
    )


def test_a_submitted_stop_is_recorded_before_it_is_protection(tmp_path):
    """The row exists so a crashed submission is not invisible."""
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo)

    (row,) = repo.protective_orders_for("entry-key-1")
    assert row["state"] == ProtectionState.SUBMITTED.value
    assert row["broker_order_id"] is None


def test_promoting_to_protected_requires_a_broker_order_id(tmp_path):
    """PROTECTED is the one claim that may not originate inside this system."""
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo)

    with pytest.raises(ValueError, match="broker_order_id is required"):
        repo.record_protection_accepted(
            client_key="stop-key-1",
            broker_order_id="   ",
            accepted_quantity=Decimal("3"),
            accepted_at=datetime(2026, 9, 4, 14, 31, tzinfo=UTC),
        )

    (row,) = repo.protective_orders_for("entry-key-1")
    assert row["state"] == ProtectionState.SUBMITTED.value


def test_a_broker_confirmation_is_what_makes_a_position_protected(tmp_path):
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo)

    repo.record_protection_accepted(
        client_key="stop-key-1",
        broker_order_id="rh-order-abc",
        accepted_quantity=Decimal("3"),
        accepted_at=datetime(2026, 9, 4, 14, 31, tzinfo=UTC),
    )

    (row,) = repo.protective_orders_for("entry-key-1")
    assert row["state"] == ProtectionState.PROTECTED.value
    assert row["broker_order_id"] == "rh-order-abc"
    assert row["last_reconciled_at"] is not None


def test_a_rejected_stop_cannot_later_be_called_protected(tmp_path):
    """A stale confirmation must not manufacture protection for a dead order."""
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo)
    repo.record_protection_failed(
        client_key="stop-key-1",
        reason="insufficient shares",
        failed_at=datetime(2026, 9, 4, 14, 31, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="no submitted protective order"):
        repo.record_protection_accepted(
            client_key="stop-key-1",
            broker_order_id="rh-order-abc",
            accepted_quantity=Decimal("3"),
            accepted_at=datetime(2026, 9, 4, 14, 32, tzinfo=UTC),
        )

    (row,) = repo.protective_orders_for("entry-key-1")
    assert row["state"] == ProtectionState.FAILED.value


def test_the_same_stop_cannot_be_submitted_twice(tmp_path):
    """Two rows for one key would mean two resting stops on one position."""
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo)

    with pytest.raises(ValueError, match="already recorded"):
        _submit(repo)


def test_an_unresolved_submission_is_what_recovery_looks_for(tmp_path):
    """The restart question: which positions might have a stop we never confirmed?"""
    repo = JournalRepository(tmp_path / "j.db")
    _submit(repo, client_key="stop-key-1", trade_key="entry-key-1")
    _submit(repo, client_key="stop-key-2", trade_key="entry-key-2")
    repo.record_protection_accepted(
        client_key="stop-key-2",
        broker_order_id="rh-order-xyz",
        accepted_quantity=Decimal("3"),
        accepted_at=datetime(2026, 9, 4, 14, 31, tzinfo=UTC),
    )

    unresolved = repo.unconfirmed_protection()

    assert [r["client_key"] for r in unresolved] == ["stop-key-1"]
