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
                                            re-size (once, may
                                            only shrink)
                                                 │
                                 ◄──────    build_order_payload() → ExecutionPlan
  review_equity_order
  human confirmation
  place_equity_order
```

Risk sizes *before* the critic runs, so the critic attacks a concrete order
rather than an abstract signal. Its confidence adjustment then feeds back
through a single bounded re-size pass.

Every cycle writes an audit entry — including the overwhelming majority that
decide to do nothing. Those are the more valuable half of the record: a system
that logs only its trades cannot tell you whether its filters work or whether
it simply never sees a setup.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Optional: record which account to trade. Gitignored — account numbers are
# identifying information and permanent once committed. The agent resolves the
# account at runtime via get_accounts regardless, so this only saves a lookup.
cp config/account.example.yaml config/account.local.yaml

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

When a setup does qualify, the intent carries its own reasoning:

```
outcome: shadow_filled
BUY AAPL $13.86 (~0.045856 sh @ ~302.25)

thesis:
  AAPL remains in an uptrend — price 302.25 holds above its 200-day
  (280.09) with the 50-day (309.48) above it — and has pulled back 5.91%
  to the 20-day (321.22) while momentum stops deteriorating. Buying the
  discount, not the breakdown.

invalidation_reason:
  A close below the 50-day (309.48) breaks the uptrend premise this trade
  rests on; the stop at 287.14 enforces that. RSI back below 30, or the
  MACD histogram resuming its decline, means the pullback became a
  breakdown and the setup was misread.

reward-to-risk    2.00 (minimum 2.00)
modeled loss      0.69  (not a floor — managed stop, market entry, gap risk)
sector            Electronic Technology — 0.00 of 25.00 cap
```

The thesis and invalidation condition are written at entry, before the outcome
is known. That is the difference between a post-mortem that reads what you
believed and one that reconstructs what you wish you had believed.

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

### Gates

All configured in `config/risk.yaml`, all schema-validated at startup, each
with a test proving it blocks.

| Gate | Behaviour |
|---|---|
| `max_daily_loss_pct` | Blocks new entries. Resets tomorrow. |
| `kill_switch_daily_loss_pct` | **Sticky.** Writes `HALT`; a human must clear it. |
| `max_sector_exposure_pct` | Caps combined exposure to one sector, and caps sizing. |
| `min_risk_reward` | Rejects setups whose target does not justify the stop. |
| `max_position_pct`, `max_open_positions`, `max_portfolio_exposure_pct` | Concentration ceilings. |
| `max_stop_pct` | A stop this wide means the setup is too loose to size. |
| `earnings_blackout_days`, `symbol_cooldown_days` | Event and behavioural gates. |
| `min_avg_volume_30d`, `max_spread_pct` | Liquidity and execution quality. |

Three behaviours that are deliberate and will otherwise look like bugs:

- **The kill switch does not block exits.** It fires automatically, and an
  automatic control that strands you in losing positions until you notice does
  more damage than the loss that triggered it. It writes `HALT`, so the *next*
  cycle stops entirely — by then a human is involved.
- **The sector cap binds immediately.** The default universe is all one sector,
  so one position at the ceiling blocks the next. The fix is a more diversified
  universe, never a looser cap.
- **`min_risk_reward` never fires for `trend_pullback`**, which builds its
  target at exactly 2R. It guards future strategies whose targets come from
  structure. Raising it above 2.0 blocks every entry instead of improving
  selectivity.

Config is cross-validated, so contradictory setups fail at startup rather than
behaving strangely later — a kill switch at or below the daily limit, or a
position ceiling above the sector ceiling, are both rejected outright.

### The critic may shrink a trade, never grow it

`agents/critic.py` re-derives the trade mechanically and can return a
`confidence_adjustment`. It is clamped non-positive, and the orchestrator
re-sizes exactly once and asserts the notional did not increase.

This is not stylistic. Confidence multiplies notional in the sizer, so an
adjustment that could raise it would let a language model enlarge a position —
the one coupling this architecture exists to prevent. The model may veto or
shrink. It may never amplify.

## Safety

| Control | Mechanism |
|---|---|
| No accidental orders | Nothing in `src/` can reach the broker |
| Manual kill switch | `touch HALT` at the project root |
| Automatic kill switch | Daily-loss breach writes `HALT`; sticky until cleared |
| Risk limits are immutable to the agent | Deny hook + integrity lock, below |
| Idempotency | Deterministic UUIDv5 `ref_id` + a unique DB constraint |
| Stale data | Preflight rejects old snapshots and excess price drift |
| Independent review | `agents/critic.py` re-derives the trade mechanically |
| LLM cannot enlarge a position | Critic confidence adjustment clamped non-positive |
| Wrong account | Only `agentic_allowed: true` accounts are permitted |
| Secrets | Account numbers live in gitignored local config, masked in output |
| Audit trail | Every cycle stored with its full snapshot, replayable |

### Why risk limits are enforced twice

The agent must not be able to widen a limit to fit a trade that was correctly
blocked. Two independent layers, because they fail differently:

1. **A `PreToolUse` deny hook** (`.claude/hooks/guard_risk_config.py`) refuses
   edits to `config/risk.yaml` and `config/risk.lock`, refuses the `lock-risk`
   command, and refuses shell commands that would write to those files while
   still allowing reads. It always exits 0 and returns its decision as JSON —
   a hook that could exit non-zero would block every tool call in the session.
2. **An integrity lock** (`config/risk.lock`) stores a SHA-256 of the
   *validated* risk values. It is verified on every config load, so a change
   made by any route — including one the hook never sees — stops the system and
   names the key that moved.

Changing a limit is therefore a deliberate two-step human act, both steps
visible in git history:

```bash
# 1. edit config/risk.yaml by hand, then:
python -m agentic_trader.cli lock-risk --confirm
```

Layer 1 constrains this agent; layer 2 catches everything. Neither is meant to
stop a determined human, and layer 1 is a speed bump rather than a boundary —
the lock is the boundary.

**Known gap:** the entry order cannot carry a broker-native stop. Stops are
*managed* — a separate `stop_market` order must follow the fill, or the
position is unprotected between cycles. This is documented everywhere it
matters and is the first thing to close before live trading. It is also why
`estimated_max_loss` is named a *modeled* loss and not a floor.

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
  config/        schema-validated loading + integrity lock
  cli.py         the JSON seam
config/
  risk.yaml            limits (agent cannot edit)
  risk.lock            SHA-256 baseline of the above
  strategies.yaml      per-strategy switches and params
  account.example.yaml template; copy to account.local.yaml (gitignored)
.claude/
  skills/        analyze-trade, critique-trade, review-performance
  hooks/         guard_risk_config.py — the deny hook
  settings.json  wires the hook (tracked, so protection travels with the repo)
```

## Next

In rough priority order:

1. **Close the managed-stop gap** — place the protective `stop_market` order
   after a fill. Until this lands, the configured stop is advisory between
   cycles, which makes it the most important open item by some distance.
2. **Bid/ask capture** — the quote is parsed but the spread is dropped, and
   spread is the real execution risk on a market order.
3. **ATR-scaled stops**, replacing the flat percentage.
4. **Market-level regime** from SPY, so the strategy stops buying pullbacks
   into a falling market.
5. **A more diversified universe**, which the sector cap now demands.
6. **Universe scanner**, `ApprovalExecutor` recording who approved what, and
   normalized `agent_decisions` / `daily_performance` journal tables.

## Status

Shadow mode. `trend_pullback` implemented and tested; `momentum` stubbed.
**No live trades placed.**

82 tests, ruff clean. Test coverage is weighted toward the negative cases —
every risk gate has a test proving it *blocks*, because a limit that silently
fails open is worse than no limit at all.
