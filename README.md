# agentic-trader

An agent-driven equity trading system for Robinhood, built on the Robinhood MCP
server. Claude orchestrates; a deterministic Python core makes every decision.

> **This places real orders with real money.** It ships in shadow mode and
> should stay there until it has a track record. See [Safety](#safety).

## The idea

Most "AI trading bot" designs put the model in the decision path, where its
output is unreproducible and its risk math is unverifiable. This one inverts
that:

- **Claude does what it is good at** — fetching data through MCP tools,
  reading context an indicator cannot see, arguing adversarially against a
  proposed trade, and deciding when to ask a human.
- **Python does what it is good at** — the same arithmetic every time, gates
  that cannot be talked around, and a decision that can be replayed months
  later from its stored snapshot.

The boundary is JSON. Nothing in `src/` can reach the broker, so no test,
import, or stray call can place an order.

## How a cycle works

```
  Claude                                    Python core
  ──────                                    ───────────
  MCP: quotes, bars, indicators,
       fundamentals, earnings    ──────►    build_snapshot()
                                                 │
                                            strategy.evaluate()   → Signal
                                                 │
                                            RiskEngine.evaluate() → RiskDecision
                                                 │
                                            critique()            → CriticReport
                                                 │
                                 ◄──────    build_order_payload() → ExecutionPlan
  review_equity_order
  human confirmation
  place_equity_order
```

Every cycle writes an audit entry — including the overwhelming majority that
decide to do nothing. Those are the more valuable half of the record: a system
that logs only its trades cannot tell you whether its filters work or whether
it simply never sees a setup.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

.venv/Scripts/python.exe -m agentic_trader.cli config-check
.venv/Scripts/python.exe -m pytest -q
```

Then, in Claude Code:

```
/analyze-trade AAPL
```

Or drive the core directly with a bundle of raw MCP responses:

```bash
.venv/Scripts/python.exe -m agentic_trader.cli evaluate --input bundle.json
```

## What a real result looks like

AAPL on 2026-08-12, actual market data:

```
outcome: watch
reasons:
  trend: price 302.25 vs SMA200 280.09          ✓
  ma_stack: SMA50 309.48 vs SMA200 280.09       ✓
  pullback: price 302.25 vs SMA20 321.22        ✓
  rsi_band: RSI 40.21 vs [30, 45]               ✓
  pullback_depth: 5.91% below SMA20 (max 12%)   ✓
failed_conditions:
  momentum_stabilizing: MACD hist -3.34 vs prior -3.27   ✗
```

Five of six conditions pass. The sixth — a MACD histogram still deepening —
is the one that separates buying a pullback from catching a falling knife, so
the setup is `watch`, not `enter`. Remove that condition and the strategy will
happily buy the middle of a collapse.

## The strategy

`trend_pullback` buys an orderly retracement inside an intact uptrend, and only
once the selling has measurably stopped. Exits when the premise breaks (price
loses the 50-day) or momentum runs to an extreme.

`momentum` is a stub with a documented design sketch; it is disabled.

## Risk model

Sizing works backward from the loss you accept, not forward from the cash you
hold:

```
risk budget = account value × risk_per_trade_pct
notional    = risk budget ÷ stop distance
```

A 1% budget with a 5% stop is a 20% position; the same budget with a 10% stop
is 10%. The stop decides the size, not conviction.

Sizes are in **dollars**, not shares — a $100 account cannot buy one share of a
$300 stock, so a share-based sizer would simply never trade.

Gates, all configurable in `config/risk.yaml` and all schema-validated at
startup: daily loss limit, max open positions, portfolio exposure, per-symbol
duplicate and open-order checks, post-loss cooldown, earnings blackout,
liquidity floor, maximum stop width, and a `HALT` kill switch.

## Safety

| Control | Mechanism |
|---|---|
| No accidental orders | Nothing in `src/` can reach the broker |
| Kill switch | `touch HALT` at the project root |
| Idempotency | Deterministic UUIDv5 `ref_id` + a unique DB constraint |
| Stale data | Preflight rejects old snapshots and excess price drift |
| Independent review | `agents/critic.py` re-derives the trade mechanically |
| Wrong account | Only `agentic_allowed: true` accounts are permitted |
| Audit trail | Every cycle stored with its full snapshot, replayable |

**Known gap:** the entry order cannot carry a broker-native stop. Stops are
*managed* — a separate `stop_market` order must follow the fill, or the
position is unprotected between cycles. This is documented everywhere it
matters and is the first thing to close before live trading.

## Layout

```
src/agentic_trader/
  models/        domain types crossing every layer
  market/        MCP payloads → snapshot; signals; regime
  strategies/    opinions only — no account access, no sizing
  risk/          limits, sizing, engine
  agents/        orchestrator (one pure cycle), critic
  execution/     payload construction, shadow fills
  journal/       SQLite audit stream + trade records
  cli.py         the JSON seam
config/          risk.yaml, strategies.yaml (schema-validated)
.claude/skills/  analyze-trade, critique-trade, review-performance
```

## Status

Shadow mode. `trend_pullback` implemented and tested (58 tests); `momentum`
stubbed. No live trades placed.
