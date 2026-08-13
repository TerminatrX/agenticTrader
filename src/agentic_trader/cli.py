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
          "sma_20": <...>, "sma_50": <...>, "sma_200": <...>
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
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentic_trader.agents.orchestrator import CycleResult, run_cycle
from agentic_trader.config import ConfigError, load_config
from agentic_trader.journal import JournalRepository
from agentic_trader.market.regime import classify_regime
from agentic_trader.market.snapshot import SnapshotError, build_snapshot
from agentic_trader.models import AccountState
from agentic_trader.strategies.base import available_strategies

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
    if not args.no_journal:
        try:
            repo = JournalRepository(args.db or _default_db(config.project_root))
            known_keys = repo.known_client_keys()
            loss_dt = repo.last_losing_exit(symbol.upper())
            last_loss = loss_dt.date() if loss_dt else None
            recent_trades = len(
                [t for t in repo.closed_trades() if t.symbol == symbol.upper()]
            )
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
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cycle failed: {exc!r}", EXIT_ERROR)

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

    _emit(_render_result(result, snapshot_regime=classify_regime(snapshot).value))
    return EXIT_OK


def _render_result(result: CycleResult, *, snapshot_regime: str) -> dict[str, Any]:
    """Shape the result for the agent, foregrounding what it must act on."""
    payload: dict[str, Any] = {
        "ok": True,
        "cycle_id": result.cycle_id,
        "symbol": result.symbol,
        "strategy": result.strategy,
        "outcome": result.outcome.value,
        "regime": snapshot_regime,
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

    rp = sub.add_parser("report", help="Performance and audit summary.")
    rp.add_argument("--strategy", help="Limit to one strategy.")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_report)

    cc = sub.add_parser("config-check", help="Validate configuration and show effective values.")
    cc.set_defaults(func=cmd_config_check)

    st = sub.add_parser("strategies", help="List registered strategies.")
    st.set_defaults(func=cmd_strategies)

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
