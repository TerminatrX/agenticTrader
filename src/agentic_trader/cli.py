"""JSON-in / JSON-out command line — the seam between the agent and the core.

The agent fetches data with MCP tools, pipes the raw payloads here, and gets
back a decision. Everything crosses the boundary as JSON on stdin and stdout so
that the interface is inspectable, diffable, and replayable: capture a request
bundle, and you can reproduce the exact decision offline forever.

    agentic-trader evaluate  < bundle.json
    agentic-trader report    --json
    agentic-trader config-check
    agentic-trader strategies

The `evaluate` bundle:

    {
      "symbol": "AAPL",
      "mode": "shadow",                  // "shadow" | "live"
      "strategy": "trend_pullback",
      "account": { ...AccountState... },
      "payloads": {
        "quote":        <get_equity_quotes response>,
        "historicals":  <get_equity_historicals response>,
        "fundamentals": <get_equity_fundamentals response>,
        "earnings":     <get_earnings_results response>,
        "indicators": {
          "rsi": <...>, "macd": <...>,
          "sma_20": <...>, "sma_50": <...>, "sma_200": <...>,
          "atr": <...>                     // sets the stop, and so the size
        },
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
from agentic_trader.journal import JournalRepository
from agentic_trader.market.market_regime import MarketContext, classify_market_regime
from agentic_trader.market.snapshot import SnapshotError, build_snapshot
from agentic_trader.market.symbol_regime import classify_symbol_regime
from agentic_trader.models import AccountState
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

    payloads = bundle.get("payloads", {})
    try:
        snapshot = build_snapshot(
            symbol,
            quote=payloads.get("quote"),
            historicals=payloads.get("historicals"),
            fundamentals=payloads.get("fundamentals"),
            earnings=payloads.get("earnings"),
            indicators=payloads.get("indicators", {}),
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
                    indicators=data.get("indicators", {}),
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
        }
    )
    return EXIT_OK


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

    rp = sub.add_parser("report", help="Performance and audit summary.")
    rp.add_argument("--strategy", help="Limit to one strategy.")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_report)

    cc = sub.add_parser("config-check", help="Validate configuration and show effective values.")
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
