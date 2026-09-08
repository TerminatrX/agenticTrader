"""JSON-in / JSON-out command line — the seam between the agent and the core.

The agent fetches data with MCP tools, pipes the raw payloads here, and gets
back a decision. Everything crosses the boundary as JSON on stdin and stdout so
that the interface is inspectable and diffable.

Replay is a property of the **journal**, not of the bundle. Re-running a saved
bundle later does not reproduce the original decision: `build_snapshot` stamps
`captured_at` from the current clock, the earnings assessment is normalized
against that date, and `run_cycle` measures quote age, price drift and the risk
gate's `as_of` against the instant it is given. A bundle replayed tomorrow is
evaluated as tomorrow.

What does support replay is the audit row, which records the three things
separately because they answer different questions:

    acquisition contract + trading_date   what was requested
    snapshot_json                         what came back
    occurred_at                           when the decision was evaluated

Replaying with all three -- rehydrate the snapshot, pass the stored mode and
trading date, and set `now` to `occurred_at` -- reproduces the decision. The
freshness controls deliberately have no historical-timestamp override on the
normal path: they must keep measuring against the real decision time, or they
stop being freshness controls.

    agentic-trader evaluate  < bundle.json
    agentic-trader report    --json
    agentic-trader config-check
    agentic-trader strategies

The `evaluate` bundle:

    {
      "symbol": "AAPL",
      "mode": "shadow",                  // "shadow" | "live"
      "strategy": "trend_pullback",

      // Required. Copy all three verbatim from `acquisition-spec`, which is
      // what produced the payloads below. `trading_date` is the date every
      // start_time was derived from; the other two say which contract was in
      // force. A bundle whose contract does not match the one in force is
      // refused rather than re-stamped -- the lookbacks differ, so the
      // payloads are genuinely not what the current profile would have asked
      // for, and recording them as such would be a false provenance claim.
      "trading_date": "2026-08-22",
      "acquisition_profile_ref": "agentic-acquisition@v4-2026-08-25",
      "acquisition_config_fingerprint": "eceedacc...",

      "account": { ...AccountState... },
      "payloads": {
        "quote":        <get_equity_quotes response>,
        "historicals":  <get_equity_historicals response>,
        "fundamentals": <get_equity_fundamentals response>,
        "earnings":     <get_earnings_results response>,
        // No "indicators" key since v4. RSI, MACD, SMA20/50/200 and ATR are
        // derived from "historicals" above; sending them changes nothing.
        "market": {                        // optional; recorded, never gated
          "SPY": {"quote": <...>, "indicators": {...}},
          "QQQ": {"quote": <...>, "indicators": {...}}
        }
      }
    }

Exit codes are meaningful, since the agent branches on them:
0 evaluated (whatever the outcome), 1 bad input or config, 2 internal error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentic_trader.agents.discovery import abort_candidates, run_discovery
from agentic_trader.agents.orchestrator import CycleResult, run_cycle
from agentic_trader.config import (
    ConfigError,
    find_project_root,
    load_config,
    load_risk_config,
    risk_fingerprint,
    write_risk_lock,
)
from agentic_trader.execution.capabilities import ROBINHOOD_MCP
from agentic_trader.execution.executor import (
    ProtectionNotPlaceable,
    build_flatten_payload,
    build_protective_stop_payload,
)
from agentic_trader.journal import JournalRepository
from agentic_trader.journal.models import TradeRecord
from agentic_trader.market.acquisition import CURRENT_ACQUISITION
from agentic_trader.market.market_regime import MarketContext, classify_market_regime
from agentic_trader.market.snapshot import SnapshotError, build_snapshot
from agentic_trader.market.symbol_regime import classify_symbol_regime
from agentic_trader.models import AccountState, ExecutionMode, ProtectionState
from agentic_trader.risk.sizing import max_whole_share_price
from agentic_trader.strategies.base import available_strategies
from agentic_trader.universe import CURRENT_DISCOVERY

EXIT_OK, EXIT_BAD_INPUT, EXIT_ERROR = 0, 1, 2


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, default=_json_default)
    sys.stdout.write("\n")


def _fail(message: str, code: int = EXIT_BAD_INPUT) -> int:
    _emit({"ok": False, "error": message})
    return code


def _read_bundle(path: str | None) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    if not raw.strip():
        raise ValueError("no input received on stdin (pass a bundle or use --input)")
    bundle = json.loads(raw)
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be a JSON object")
    return bundle


def _default_db(config_root: Path) -> Path:
    return config_root / "data" / "journal.db"


# ------------------------------------------------------------------- evaluate


def _check_acquisition_provenance(bundle: dict[str, Any]) -> str | None:
    """Return an error message if the bundle's contract claim is unusable.

    Both halves are checked. The ref alone would accept a profile edited
    without a version bump; the fingerprint alone would accept a value copied
    from an unrelated contract. Together they say "these payloads were fetched
    under exactly this contract", which is the claim the audit row will make.
    """
    ref = bundle.get("acquisition_profile_ref")
    fingerprint = bundle.get("acquisition_config_fingerprint")

    if not ref or not fingerprint:
        missing = [
            name
            for name, value in (
                ("acquisition_profile_ref", ref),
                ("acquisition_config_fingerprint", fingerprint),
            )
            if not value
        ]
        return (
            f"bundle is missing {' and '.join(missing)} — copy the values "
            "verbatim from `acquisition-spec`, which is what fetched these "
            "payloads. Evaluating without them would record a decision whose "
            "inputs cannot be traced to a contract."
        )

    if ref != CURRENT_ACQUISITION.profile_ref:
        return (
            f"bundle was fetched under acquisition contract {ref!r} but the "
            f"contract in force is {CURRENT_ACQUISITION.profile_ref!r}. "
            "Re-fetch with the current profile; the payloads are not "
            "re-stamped, because the lookbacks that produced them differ."
        )

    if fingerprint != CURRENT_ACQUISITION.content_fingerprint:
        return (
            f"bundle claims contract {ref!r} but carries fingerprint "
            f"{fingerprint!r}, and the contract in force hashes to "
            f"{CURRENT_ACQUISITION.content_fingerprint!r}. The profile was "
            "edited without a version bump, or the value was copied from "
            "elsewhere. Re-run `acquisition-spec` and re-fetch."
        )

    return None


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        bundle = _read_bundle(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(f"could not read bundle: {exc}")

    try:
        config = load_config(Path(args.project_root) if args.project_root else None)
    except ConfigError as exc:
        return _fail(str(exc))

    symbol = bundle.get("symbol")
    if not symbol:
        return _fail("bundle is missing 'symbol'")

    try:
        account = AccountState(**bundle["account"])
    except KeyError:
        return _fail("bundle is missing 'account'")
    except Exception as exc:
        return _fail(f"invalid account state: {exc}")

    # --- acquisition provenance -------------------------------------------
    #
    # The bundle must state which contract fetched its payloads, and that claim
    # must match the contract in force. Without this the whole milestone is
    # decorative: `run_cycle` would fall back to CURRENT_ACQUISITION and a
    # payload gathered weeks ago under different lookbacks would be journalled
    # as though it came from today's profile -- a false provenance record,
    # which is worse than none at all.
    #
    # There is deliberately no override flag. Replaying a bundle built under an
    # older contract means checking out the commit that defined it; a bypass
    # here would be reached for on exactly the day it should not be.
    provenance_error = _check_acquisition_provenance(bundle)
    if provenance_error:
        return _fail(provenance_error)

    try:
        trading_date = date.fromisoformat(bundle["trading_date"])
    except (KeyError, TypeError):
        return _fail(
            "bundle is missing 'trading_date' — copy it from the "
            "acquisition-spec output that produced these payloads. It is the "
            "date every start_time was derived from, and it is not "
            "recoverable from the payloads themselves."
        )
    except ValueError:
        return _fail(
            f"bundle 'trading_date' must be YYYY-MM-DD, got "
            f"{bundle['trading_date']!r}"
        )


    payloads = bundle.get("payloads", {})
    try:
        snapshot = build_snapshot(
            symbol,
            quote=payloads.get("quote"),
            historicals=payloads.get("historicals"),
            fundamentals=payloads.get("fundamentals"),
            earnings=payloads.get("earnings"),
            # Indicators are derived from these bars, not fetched. The trading
            # date decides which bars are complete and where each window
            # starts, which is why it is parsed before the snapshot is built.
            trading_date=trading_date,
        )
    except SnapshotError as exc:
        return _fail(f"could not build snapshot: {exc}")

    # Index snapshots for market-regime context. Entirely optional: the bundle
    # may omit them, and a malformed one must not stop a decision — the regime
    # is recorded for later analysis and gates nothing.
    market_context: MarketContext | None = None
    market_payloads = payloads.get("market") or {}
    if market_payloads:
        try:
            index_snaps = {
                sym.upper(): build_snapshot(
                    sym,
                    quote=data.get("quote"),
                    historicals=data.get("historicals"),
                    trading_date=trading_date,
                )
                for sym, data in market_payloads.items()
            }
            market_context = classify_market_regime(
                index_snaps.get("SPY"), index_snaps.get("QQQ")
            )
        except (SnapshotError, AttributeError, TypeError) as exc:
            print(f"warning: market context unavailable ({exc})", file=sys.stderr)

    mode = bundle.get("mode", args.mode)
    if mode not in ("shadow", "live"):
        return _fail(f"mode must be 'shadow' or 'live', got {mode!r}")

    strategy_name = bundle.get("strategy", args.strategy)

    # Journal lookups feed the cooldown gate and duplicate-order guard. A
    # missing journal is not fatal — it just means no history to consult.
    repo: JournalRepository | None = None
    known_keys: set[str] = set()
    last_loss: date | None = None
    recent_trades = 0
    active_stop: Decimal | None = None
    if not args.no_journal:
        try:
            repo = JournalRepository(args.db or _default_db(config.project_root))
            known_keys = repo.known_client_keys()
            loss_dt = repo.last_losing_exit(symbol.upper())
            last_loss = loss_dt.date() if loss_dt else None
            recent_trades = len(
                [t for t in repo.closed_trades() if t.symbol == symbol.upper()]
            )
            # The stop that was set when this position was opened. Nothing at
            # the broker enforces it — stops here are managed — so the journal
            # is the only record of it, and the strategy cannot check a level it
            # is never told. Where several lots are open, the highest stop wins:
            # it is the one that would trigger first.
            stops = [
                t.stop_price
                for t in repo.open_trades()
                if t.symbol == symbol.upper() and t.stop_price is not None
            ]
            active_stop = max(stops) if stops else None
        except Exception as exc:  # noqa: BLE001 - journal must never block a decision
            repo = None
            known_keys = set()
            print(f"warning: journal unavailable ({exc})", file=sys.stderr)

    try:
        result = run_cycle(
            snapshot,
            account,
            config,
            strategy_name=strategy_name,
            mode=mode,
            trading_date=trading_date,
            known_client_keys=known_keys,
            last_loss_exit=last_loss,
            recent_symbol_trades=recent_trades,
            active_stop=active_stop,
            market_context=market_context,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cycle failed: {exc!r}", EXIT_ERROR)

    # The kill switch is the one place this CLI acts rather than reports. The
    # risk engine detects the breach and stays pure; writing HALT here means
    # the next cycle stops before evaluating anything, and only a human
    # deleting the file resumes trading.
    if result.risk_decision is not None and result.risk_decision.trip_kill_switch:
        _write_halt(config, result, dry_run=args.dry_run)

    if repo is not None and not args.dry_run:
        try:
            if result.audit is not None:
                repo.record_audit(result.audit)
            if result.trade is not None:
                repo.record_trade(result.trade)
        except ValueError as exc:
            # A duplicate client_key means this cycle already ran. Surface it
            # rather than swallowing it — it is the idempotency guard working.
            result.errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: journal write failed ({exc})", file=sys.stderr)

    _emit(_render_result(result, symbol_regime=classify_symbol_regime(snapshot).value))
    return EXIT_OK


def cmd_discover(args: argparse.Namespace) -> int:
    """Phase one of a multi-symbol cycle: decide what is worth evaluating.

    Consumes batched payloads the agent already fetched and emits a symbol
    list. Evaluation stays a separate command because it costs roughly seven
    single-symbol calls per candidate — this phase exists to bound that.
    """
    try:
        bundle = _read_bundle(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(f"could not read bundle: {exc}")

    try:
        config = load_config(Path(args.project_root) if args.project_root else None)
    except ConfigError as exc:
        return _fail(str(exc))

    payloads = bundle.get("payloads", {})
    try:
        trading_date = date.fromisoformat(args.date)
    except ValueError:
        return _fail(f"--date must be YYYY-MM-DD, got {args.date!r}")

    try:
        result = run_discovery(
            scans_payload=payloads.get("scans") or {},
            run_payloads=payloads.get("runs") or [],
            # Omitted key means "not fetched yet"; an empty list means the
            # fetch happened and returned nothing. Do not collapse them.
            fundamentals_payloads=payloads.get("fundamentals"),
            definition=CURRENT_DISCOVERY,
            trading_date=trading_date,
            budget=args.budget,
            fundamentals_budget=args.fundamentals_budget,
            max_per_sector=args.max_per_sector,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"discovery failed: {exc!r}", EXIT_ERROR)

    # Journalled unconditionally. A run stopped by membership drift produces no
    # DiscoveryBatch at all, and that is precisely the run worth keeping — "Legend
    # changed, so we refused to discover" must survive in the record rather than
    # only in stdout.
    if not args.no_journal and not args.dry_run:
        try:
            repo = JournalRepository(args.db or _default_db(config.project_root))
            repo.record_scan_run(
                result.run_id,
                result.batch,
                # The date selection actually rotated on, carried from the
                # result rather than re-derived here -- two sources for one
                # input is how they drift apart.
                trading_date=result.trading_date,
                candidates=abort_candidates(result),
                scanner_profile_ref=result.scanner_profile_ref,
                scan_definition_ref=result.definition_ref,
                scan_config_fingerprint=result.config_fingerprint,
                scan_config=CURRENT_DISCOVERY.as_config(),
                selected_count=len(result.selected_symbols),
                budget_deferred_count=(
                    result.selection.budget_deferred_count if result.selection else 0
                ),
                drift_status=result.drift_status.value,
                drift_findings=result.drift.as_dicts(),
                aborted_reason=result.aborted_reason,
                coverage_status=result.coverage.value,
                source="robinhood_scanner",
                started_at=result.started_at,
            )
        except Exception as exc:  # noqa: BLE001 - journal must never block a decision
            print(f"warning: journal unavailable ({exc})", file=sys.stderr)

    payload = {"ok": True, **result.summary()}
    # A blocking drift is not an error the agent should retry around; it is a
    # deliberate refusal, so it exits 0 with an empty selection.
    _emit(payload)
    return EXIT_OK


def _write_halt(config: Any, result: CycleResult, *, dry_run: bool) -> None:
    """Write the HALT file so no further cycle can construct an order."""
    if dry_run:
        result.errors.append("KILL SWITCH tripped (HALT not written: --dry-run)")
        return
    try:
        config.halt_path.write_text(
            f"Kill switch tripped by cycle {result.cycle_id} on {result.symbol} "
            f"at {datetime.now(UTC).isoformat()}.\n\n"
            "Trading is disabled until this file is deleted. Review the day's "
            "realized losses and the journal before removing it.\n",
            encoding="utf-8",
        )
        result.errors.append(f"KILL SWITCH tripped — HALT written to {config.halt_path}")
    except OSError as exc:
        # Failing to write HALT is severe: the guard silently would not exist.
        result.errors.append(
            f"KILL SWITCH tripped but HALT could not be written ({exc}) — "
            "stop the system manually"
        )


def _render_result(result: CycleResult, *, symbol_regime: str) -> dict[str, Any]:
    """Shape the result for the agent, foregrounding what it must act on."""
    payload: dict[str, Any] = {
        "ok": True,
        "cycle_id": result.cycle_id,
        "symbol": result.symbol,
        "strategy": result.strategy,
        "outcome": result.outcome.value,
        # Reported at cycle level rather than only under `plan`, because the
        # outcomes that produce no plan are exactly the ones whose execution
        # context could not otherwise be established.
        "mode": result.mode.value,
        "acquisition_profile_ref": result.acquisition.profile_ref,
        "acquisition_config_fingerprint": result.acquisition.content_fingerprint,
        "symbol_regime": symbol_regime,
        "market_regime": (
            result.market_context.to_dict() if result.market_context else None
        ),
        "headline": result.headline(),
        "should_submit": result.should_submit,
        "errors": result.errors,
    }

    if result.signal is not None:
        payload["signal"] = {
            "strength": result.signal.strength.value,
            "side": result.signal.side.value if result.signal.side else None,
            "confidence": result.signal.confidence,
            "reference_price": result.signal.reference_price,
            "stop_price": result.signal.stop_price,
            "target_price": result.signal.target_price,
            "reasons": result.signal.reasons,
            "failed_conditions": result.signal.failed_conditions,
            "metrics": result.signal.metrics,
        }

    if result.risk_decision is not None:
        payload["risk"] = {
            "decision": result.risk_decision.decision.value,
            "approved_notional": result.risk_decision.approved_notional,
            "breached_limits": result.risk_decision.breached_limits,
            "warnings": result.risk_decision.warnings,
            "notes": result.risk_decision.notes,
        }

    if result.critic is not None:
        payload["critic"] = result.critic.to_dict()

    if result.plan is not None:
        # `order_payload` is verbatim arguments for place_equity_order. The
        # agent should pass it through unmodified — editing it here would mean
        # the executed order differs from the one risk approved.
        payload["plan"] = {
            "mode": result.plan.mode,
            "summary": result.plan.summary(),
            "estimated_cost": result.plan.estimated_cost,
            "managed_stop": result.plan.managed_stop,
            "managed_target": result.plan.managed_target,
            "warnings": result.plan.warnings,
            "preflight_notes": result.plan.preflight_notes,
            "order_payload": result.plan.payload,
        }

    if result.trade is not None:
        payload["shadow_trade"] = result.trade.model_dump(mode="json")

    return payload


# --------------------------------------------------------------------- report


# --------------------------------------------------------- the stop lifecycle
#
# Four commands covering the window between a live entry filling and its stop
# resting at the broker. That window cannot be removed — a stop needs a position
# to attach to — so these exist to make it short, observable, and terminating.
#
# The order is not advisory. `protect` writes SUBMITTED *before* emitting the
# payload, so a crash between emitting and placing leaves a row saying "we may
# have placed a stop" rather than nothing at all. Reversing it would make a
# lost reply indistinguishable from an order never sent, and the recovery
# question — is there already a stop resting? — unanswerable without guessing.


def _repo_for(args: argparse.Namespace) -> tuple[JournalRepository, Path] | int:
    try:
        config = load_config(Path(args.project_root) if args.project_root else None)
    except ConfigError as exc:
        return _fail(str(exc))
    return JournalRepository(args.db or _default_db(config.project_root)), config.project_root


def cmd_record_fill(args: argparse.Namespace) -> int:
    """Record a live entry that actually filled.

    Nothing else writes a live position to the journal — `run_cycle` builds the
    payload and stops. Until this runs, the position exists at the broker and
    not in the record, which means `unprotected_live_positions` cannot see it
    and the entry guard cannot fire. Skipping this step does not just lose an
    audit row; it disarms the check that would otherwise notice the next
    unprotected entry.
    """
    resolved = _repo_for(args)
    if isinstance(resolved, int):
        return resolved
    repo, _ = resolved

    try:
        fill = _read_bundle(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(str(exc))

    required = ("client_key", "symbol", "strategy", "fill_price", "quantity")
    missing = [k for k in required if not fill.get(k)]
    if missing:
        return _fail(f"fill is missing required field(s): {', '.join(missing)}")

    try:
        quantity = Decimal(str(fill["quantity"]))
        fill_price = Decimal(str(fill["fill_price"]))
        stop_price = Decimal(str(fill["stop_price"])) if fill.get("stop_price") else None
        target_price = Decimal(str(fill["target_price"])) if fill.get("target_price") else None
        notional = Decimal(str(fill["notional"])) if fill.get("notional") else quantity * fill_price
        opened_at = (
            datetime.fromisoformat(fill["opened_at"])
            if fill.get("opened_at")
            else datetime.now(UTC)
        )
    except (ArithmeticError, ValueError) as exc:
        return _fail(f"unparseable fill value: {exc}")

    # COMMITTED, not PROTECTED. The position is live and uncovered at this
    # instant; the state records what the caller is bound to do next, and the
    # entry guard treats it as unprotected until a broker id says otherwise.
    record = TradeRecord(
        client_key=str(fill["client_key"]),
        symbol=str(fill["symbol"]).upper(),
        strategy=str(fill["strategy"]),
        mode=ExecutionMode.LIVE,
        opened_at=opened_at,
        entry_price=fill_price,
        quantity=quantity,
        notional=notional,
        stop_price=stop_price,
        target_price=target_price,
        entry_rationale=list(fill.get("entry_rationale") or []),
        thesis=fill.get("thesis"),
        invalidation_reason=fill.get("invalidation_reason"),
        sector=fill.get("sector"),
        protection_state=(
            ProtectionState.COMMITTED if stop_price is not None else ProtectionState.NOT_REQUIRED
        ),
        capability_profile=fill.get("capability_profile") or ROBINHOOD_MCP.profile_ref,
        market_regime=fill.get("market_regime"),
        stop_basis=fill.get("stop_basis"),
    )

    try:
        repo.record_trade(record)
    except ValueError as exc:
        return _fail(str(exc))

    _emit(
        {
            "ok": True,
            "recorded": record.client_key,
            "symbol": record.symbol,
            "quantity": record.quantity,
            "protection_state": record.protection_state,
            "next": (
                "protect" if stop_price is not None else "none: this position carries no stop"
            ),
        }
    )
    return EXIT_OK


def cmd_protect(args: argparse.Namespace) -> int:
    """Build the resting stop for a filled position, and record it as submitted.

    Emits the payload to place. The SUBMITTED row is written first, on purpose
    — see the note above this section.
    """
    resolved = _repo_for(args)
    if isinstance(resolved, int):
        return resolved
    repo, _ = resolved

    open_trades = {t.client_key: t for t in repo.open_trades()}
    trade = open_trades.get(args.client_key)
    if trade is None:
        return _fail(f"no open trade with client_key {args.client_key}")
    if trade.stop_price is None:
        return _fail(f"trade {args.client_key} carries no stop level; nothing to place")

    # The filled quantity, not the intended one. A partial fill protected at
    # the requested size rests a stop for shares the account does not hold.
    quantity = Decimal(args.filled_quantity) if args.filled_quantity else trade.quantity
    fill_price = Decimal(args.fill_price) if args.fill_price else trade.entry_price

    try:
        payload = build_protective_stop_payload(
            account_number=args.account_number,
            symbol=trade.symbol,
            quantity=quantity,
            stop_price=trade.stop_price,
            fill_price=fill_price,
            entry_client_key=trade.client_key,
        )
    except ProtectionNotPlaceable as exc:
        # Not an error to swallow: the position is open and cannot be covered.
        # The flatten payload is emitted with the refusal so the caller is never
        # left holding an uncoverable position with no next move.
        _emit(
            {
                "ok": False,
                "error": str(exc),
                "position_is_unprotected": True,
                "flatten_payload": build_flatten_payload(
                    account_number=args.account_number,
                    symbol=trade.symbol,
                    quantity=quantity,
                    entry_client_key=trade.client_key,
                ),
                "next": "place flatten_payload to close the position, then record-exit",
            }
        )
        return EXIT_BAD_INPUT

    try:
        repo.record_protection_submitted(
            trade_client_key=trade.client_key,
            client_key=payload["ref_id"],
            stop_price=Decimal(payload["stop_price"]),
            requested_quantity=quantity,
            capability_profile=ROBINHOOD_MCP.profile_ref,
            submitted_at=datetime.now(UTC),
        )
    except ValueError as exc:
        return _fail(str(exc))

    _emit(
        {
            "ok": True,
            "protective_client_key": payload["ref_id"],
            "state": ProtectionState.SUBMITTED,
            "order_payload": payload,
            "next": (
                "place order_payload, then `protect-resolve` with the broker order id "
                "— or with --failed if it was rejected"
            ),
        }
    )
    return EXIT_OK


def cmd_protect_resolve(args: argparse.Namespace) -> int:
    """Close the uncertainty a submitted stop opened, in one direction or the other.

    Accepting requires the broker's own order id, because PROTECTED is the only
    state asserting that a resting order exists and the id is the only evidence
    of it that does not originate here.
    """
    resolved = _repo_for(args)
    if isinstance(resolved, int):
        return resolved
    repo, _ = resolved

    if args.failed:
        try:
            repo.record_protection_failed(
                client_key=args.client_key,
                reason=args.failed,
                failed_at=datetime.now(UTC),
            )
        except ValueError as exc:
            return _fail(str(exc))

        rows = [
            r
            for r in repo.protective_orders_for(args.trade_client_key)
            if r["client_key"] == args.client_key
        ]
        quantity = Decimal(rows[0]["requested_quantity"]) if rows else Decimal(args.quantity or "0")
        _emit(
            {
                "ok": True,
                "state": ProtectionState.FAILED,
                "position_is_unprotected": True,
                "flatten_payload": build_flatten_payload(
                    account_number=args.account_number,
                    symbol=args.symbol,
                    quantity=quantity,
                    entry_client_key=args.trade_client_key,
                ),
                "next": "place flatten_payload — the position is open and uncovered",
            }
        )
        return EXIT_OK

    if not args.accepted_order_id:
        return _fail("pass --accepted-order-id or --failed")

    try:
        repo.record_protection_accepted(
            client_key=args.client_key,
            broker_order_id=args.accepted_order_id,
            accepted_quantity=Decimal(args.quantity) if args.quantity else Decimal("0"),
            accepted_at=datetime.now(UTC),
        )
    except ValueError as exc:
        return _fail(str(exc))

    # The trade row too, so a later review does not read COMMITTED for a
    # position whose stop was confirmed. The guard does not depend on this.
    try:
        repo.update_trade_protection(args.trade_client_key, ProtectionState.PROTECTED)
    except ValueError as exc:
        return _fail(str(exc))

    _emit(
        {
            "ok": True,
            "state": ProtectionState.PROTECTED,
            "broker_order_id": args.accepted_order_id,
            "next": "none: the position is covered",
        }
    )
    return EXIT_OK


def cmd_record_exit(args: argparse.Namespace) -> int:
    """Close a position in the journal after its exit filled.

    Required to finish a flatten, and not merely for tidiness: an uncovered
    live position blocks every subsequent live entry, and the block is keyed on
    the trade being open. A position sold at the broker but left open here
    stops the system permanently, for a position that no longer exists.
    """
    resolved = _repo_for(args)
    if isinstance(resolved, int):
        return resolved
    repo, _ = resolved

    try:
        repo.close_trade(
            client_key=args.client_key,
            exit_price=Decimal(args.exit_price),
            closed_at=datetime.now(UTC),
            exit_reason=args.reason,
        )
    except (ArithmeticError, ValueError) as exc:
        return _fail(str(exc))

    _emit(
        {
            "ok": True,
            "closed": args.client_key,
            "exit_price": args.exit_price,
            "reason": args.reason,
            "live_entries_blocked": bool(repo.unprotected_live_positions()),
        }
    )
    return EXIT_OK


def cmd_protection_status(args: argparse.Namespace) -> int:
    """What is open, what is uncovered, and what was never resolved.

    The first command a session should run. `unconfirmed` is the dangerous
    list: each row is a stop that may or may not be resting at the broker, and
    the answer is only obtainable by reading the broker's open orders. Placing
    a replacement without checking risks two resting stops on one position,
    where the second becomes a short the moment the first fills.
    """
    resolved = _repo_for(args)
    if isinstance(resolved, int):
        return resolved
    repo, _ = resolved

    uncovered = repo.unprotected_live_positions()
    _emit(
        {
            "ok": True,
            "unprotected_live_positions": uncovered,
            "unconfirmed": [
                {
                    "client_key": r["client_key"],
                    "trade_client_key": r["trade_client_key"],
                    "stop_price": r["stop_price"],
                    "submitted_at": r["submitted_at"],
                }
                for r in repo.unconfirmed_protection()
            ],
            "live_entries_blocked": bool(uncovered),
        }
    )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.project_root) if args.project_root else None)
    except ConfigError as exc:
        return _fail(str(exc))

    repo = JournalRepository(args.db or _default_db(config.project_root))
    summary = repo.performance_summary(args.strategy)
    open_trades = repo.open_trades()

    _emit(
        {
            "ok": True,
            "performance": summary,
            "open_positions": [
                {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "opened_at": t.opened_at,
                    "entry_price": t.entry_price,
                    "quantity": t.quantity,
                    "stop_price": t.stop_price,
                    "target_price": t.target_price,
                }
                for t in open_trades
            ],
            "rejection_reasons": repo.rejection_reasons(),
            "recent_audit": repo.recent_audit(limit=args.limit),
        }
    )
    return EXIT_OK


# ---------------------------------------------------------------- acquisition


def cmd_acquisition_spec(args: argparse.Namespace) -> int:
    """Emit the exact read-only calls to make for each symbol.

    This command is the reason the acquisition profile exists. An agent or a
    payload worker asks for a symbol and a date and receives fully-specified
    MCP parameters -- interval, bounds, adjustment, output width, start time.
    It chooses none of them. Prose saying "roughly 120 days back" is what
    produced 30, 57, and 265-point histories for the same logical indicator.

    Read-only by construction: every tool named here is a market-data lookup.
    Nothing in this output can place, review, cancel, or modify an order.
    """
    try:
        trading_date = date.fromisoformat(args.date)
    except ValueError:
        return _fail(f"--date must be YYYY-MM-DD, got {args.date!r}")

    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    if not symbols:
        return _fail("at least one symbol is required")

    profile = CURRENT_ACQUISITION
    _emit(
        {
            "ok": True,
            "acquisition_profile_ref": profile.profile_ref,
            "acquisition_config_fingerprint": profile.content_fingerprint,
            "trading_date": trading_date.isoformat(),
            "end_time_policy": profile.end_time_policy,
            "symbols": {
                symbol: profile.request_plan(symbol, trading_date)["calls"]
                for symbol in symbols
            },
        }
    )
    return EXIT_OK


# --------------------------------------------------------------------- config


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.project_root) if args.project_root else None)
    except ConfigError as exc:
        return _fail(str(exc))

    _emit(
        {
            "ok": True,
            "project_root": str(config.project_root),
            "halted": config.is_halted(),
            "halt_file": str(config.halt_path),
            # Masked deliberately: this output gets pasted into issues and
            # chat logs, and an account number does not belong in either.
            "account": (
                {"configured": True, "account_number": config.account.masked,
                 "nickname": config.account.nickname,
                 "is_cash_account": config.account.is_cash_account}
                if config.account
                else {"configured": False,
                      "note": "no config/account.local.yaml; "
                              "the agent resolves the account via get_accounts"}
            ),
            "risk": config.risk.model_dump(mode="json"),
            "enabled_strategies": config.strategies.enabled_names(),
            "registered_strategies": available_strategies(),
            **_whole_share_reach(config.risk, args.account_value),
        }
    )
    return EXIT_OK


def _whole_share_reach(risk: Any, account_value: str | None) -> dict[str, Any]:
    """What price range this account can hold as whole shares, by stop width.

    Surfaced here because it is the constraint that decides whether live
    trading is possible at all, and it is otherwise only discoverable one
    rejected symbol at a time. Above these prices a position is fractional,
    fractional positions cannot carry a resting stop, and unprotected positions
    cannot be held live.
    """
    if not account_value:
        return {}

    try:
        total = Decimal(account_value)
    except ArithmeticError:
        return {"whole_share_reach": {"error": f"unparseable account value {account_value!r}"}}

    account = AccountState(
        account_number="unused",
        is_cash_account=True,
        total_value=total,
        cash=total,
        # Buying power set to the full value on purpose: this is an upper
        # bound, and a lower figure would understate reach for an account whose
        # cash is merely unsettled today.
        buying_power=total,
        unsettled_funds=Decimal("0"),
    )
    return {
        "whole_share_reach": {
            "account_value": total,
            "note": (
                "Highest share price holdable as a whole share, so the highest "
                "price a live position can exist at. Upper bound: sector "
                "headroom and confidence below 1.0 lower it further."
            ),
            "by_stop_distance": {
                f"{pct}": {
                    "confidence_1.0": max_whole_share_price(account, risk, Decimal(pct)),
                    "confidence_0.6": max_whole_share_price(
                        account, risk, Decimal(pct), confidence=Decimal("0.6")
                    ),
                }
                for pct in ("0.02", "0.05", "0.08", "0.12")
            },
        }
    }


def cmd_strategies(args: argparse.Namespace) -> int:
    _emit({"ok": True, "registered": available_strategies()})
    return EXIT_OK


def cmd_lock_risk(args: argparse.Namespace) -> int:
    """Record the current risk values as the approved baseline.

    THIS IS A HUMAN COMMAND. The trading agent must never run it — doing so
    would let it edit a limit and immediately bless the edit, which defeats the
    entire purpose of the lock. `--confirm` is required so it cannot happen by
    reflex, and `.claude/settings.json` denies it at the harness level.
    """
    root = Path(args.project_root) if args.project_root else find_project_root()
    config_dir = root / "config"
    lock_path = config_dir / "risk.lock"

    # Load without verifying, since the whole point is to replace the baseline.
    try:
        risk = load_risk_config(config_dir / "risk.yaml")
    except ConfigError as exc:
        return _fail(str(exc))

    current = risk_fingerprint(risk)
    previous = None
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8")).get("fingerprint")
        except (OSError, json.JSONDecodeError):
            previous = None

    if previous == current:
        _emit({"ok": True, "changed": False, "fingerprint": current,
               "note": "risk config already matches its lock"})
        return EXIT_OK

    if not args.confirm:
        _emit({
            "ok": False,
            "changed": False,
            "error": "risk config differs from its lock; re-run with --confirm to approve",
            "previous_fingerprint": previous,
            "current_fingerprint": current,
            "values": risk.model_dump(mode="json"),
        })
        return EXIT_BAD_INPUT

    fingerprint = write_risk_lock(risk, lock_path)
    _emit({"ok": True, "changed": True, "fingerprint": fingerprint,
           "lock_file": str(lock_path),
           "note": "commit this lock file alongside the risk.yaml change"})
    return EXIT_OK


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-trader",
        description="Deterministic trading core. Reads JSON, writes JSON, places nothing.",
    )
    parser.add_argument("--project-root", help="Override project root discovery.")
    parser.add_argument("--db", help="Path to the journal database.")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="Run one cycle for one symbol.")
    ev.add_argument("--input", help="Read the bundle from a file instead of stdin.")
    ev.add_argument("--mode", default="shadow", choices=["shadow", "live"])
    ev.add_argument("--strategy", default="trend_pullback")
    ev.add_argument("--no-journal", action="store_true", help="Skip all journal access.")
    ev.add_argument("--dry-run", action="store_true", help="Evaluate but write nothing.")
    ev.set_defaults(func=cmd_evaluate)

    dc = sub.add_parser(
        "discover",
        help="Verify saved scans, union the shards, and select candidates to enrich.",
    )
    dc.add_argument("--input", help="Discovery bundle (default: stdin).")
    dc.add_argument("--budget", type=int, default=25, help="Max candidates to enrich.")
    dc.add_argument(
        "--fundamentals-budget",
        type=int,
        default=50,
        help="Max candidates to request authoritative sector data for.",
    )
    dc.add_argument("--max-per-sector", type=int, default=None)
    dc.add_argument(
        "--date",
        required=True,
        help=(
            "Trading date (YYYY-MM-DD). Required: candidate rotation keys on it, "
            "and UTC crosses midnight while the US session is still open, so an "
            "ambient clock would rotate a day early and mislabel the journal."
        ),
    )
    dc.add_argument("--no-journal", action="store_true")
    dc.add_argument("--dry-run", action="store_true")
    dc.set_defaults(func=cmd_discover)

    aq = sub.add_parser(
        "acquisition-spec",
        help="Emit the pinned read-only MCP calls for one or more symbols.",
    )
    aq.add_argument("symbols", nargs="+", help="Symbols to generate requests for.")
    aq.add_argument(
        "--date",
        required=True,
        help=(
            "Trading date (YYYY-MM-DD). Required for the same reason discover "
            "requires it: every start_time derives from this date, never from "
            "the wall clock, so the same date regenerates the same requests."
        ),
    )
    aq.set_defaults(func=cmd_acquisition_spec)

    rf = sub.add_parser(
        "record-fill",
        help="Record a live entry that filled. Nothing else writes a live position.",
    )
    rf.add_argument("--input", help="Fill JSON (default: stdin).")
    rf.set_defaults(func=cmd_record_fill)

    pr = sub.add_parser(
        "protect",
        help="Build and record the resting stop for a filled position.",
    )
    pr.add_argument("--client-key", required=True, help="The entry's client_key.")
    pr.add_argument("--account-number", required=True)
    pr.add_argument(
        "--filled-quantity",
        help="Quantity actually filled. Defaults to the recorded quantity; pass it "
        "explicitly after a partial fill.",
    )
    pr.add_argument("--fill-price", help="Actual fill price. Defaults to the recorded entry.")
    pr.set_defaults(func=cmd_protect)

    pv = sub.add_parser(
        "protect-resolve",
        help="Record whether the broker accepted the stop. Emits a flatten payload if not.",
    )
    pv.add_argument("--client-key", required=True, help="The protective order's client_key.")
    pv.add_argument("--trade-client-key", required=True, help="The entry's client_key.")
    pv.add_argument("--account-number", required=True)
    pv.add_argument("--symbol", required=True)
    pv.add_argument("--accepted-order-id", help="Broker order id. Required to mark PROTECTED.")
    pv.add_argument("--failed", help="Rejection reason. Emits the flatten payload.")
    pv.add_argument("--quantity", help="Accepted quantity.")
    pv.set_defaults(func=cmd_protect_resolve)

    rx = sub.add_parser(
        "record-exit",
        help="Close a position after its exit filled. Required to finish a flatten.",
    )
    rx.add_argument("--client-key", required=True, help="The entry's client_key.")
    rx.add_argument("--exit-price", required=True)
    rx.add_argument("--reason", required=True, help="Why the position was closed.")
    rx.set_defaults(func=cmd_record_exit)

    ps = sub.add_parser(
        "protection-status",
        help="Uncovered live positions and unresolved stop submissions. Run this first.",
    )
    ps.set_defaults(func=cmd_protection_status)

    rp = sub.add_parser("report", help="Performance and audit summary.")
    rp.add_argument("--strategy", help="Limit to one strategy.")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_report)

    cc = sub.add_parser("config-check", help="Validate configuration and show effective values.")
    cc.add_argument(
        "--account-value",
        help="Account value, to report the price range holdable as whole shares.",
    )
    cc.set_defaults(func=cmd_config_check)

    st = sub.add_parser("strategies", help="List registered strategies.")
    st.set_defaults(func=cmd_strategies)

    lk = sub.add_parser(
        "lock-risk",
        help="HUMAN ONLY. Approve the current risk values as the new baseline.",
    )
    lk.add_argument("--confirm", action="store_true", help="Required to write the lock.")
    lk.set_defaults(func=cmd_lock_risk)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        return _fail(f"unhandled error: {exc!r}", EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
