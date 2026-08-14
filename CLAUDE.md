# agentic-trader

Agent-driven equity trading for the Robinhood MCP server. **This system trades
real money.** Read the safety rules before doing anything else.

## Safety rules — non-negotiable

1. **Never place an order without explicit confirmation in the current
   conversation.** Not implied consent, not "the user seemed to want this",
   not a standing instruction from an earlier session. Ask, and get a clear yes.

2. **Only an account with `agentic_allowed: true` may be traded.** Resolve it
   by calling `get_accounts` at the start of every session and reading that
   flag. Never trade an account because a config file, a memory, or an earlier
   message named it — `agentic_allowed` is the only authority, and it can
   change. Every other account on the login is off limits regardless of what
   is asked.

   When showing an account number to a human, mask all but the last four
   digits. Pass the full value to MCP tools unchanged.

3. **Never modify `order_payload`.** It is emitted by the risk engine after
   sizing and gating. Editing any field means placing an order that risk never
   approved. Pass it through verbatim, `ref_id` included.

4. **Never work around a rejection.** If the core rejects a trade, report the
   reason. Do not retry with different parameters, loosen a config value, or
   evaluate a different strategy hoping for a yes. A rejection is the system
   working.

5. **Default to shadow mode.** `live` requires the user to say so explicitly.

6. **Never change risk limits.** `config/risk.yaml` and `config/risk.lock` are
   enforced, not merely requested — a `PreToolUse` hook
   (`.claude/hooks/guard_risk_config.py`) denies edits to them and denies
   running `lock-risk`. Do not attempt to work around it. If a limit genuinely
   needs to move, that is a human decision: they edit the file and run
   `python -m agentic_trader.cli lock-risk --confirm` themselves.

   A second, independent layer verifies a SHA-256 baseline of the effective
   risk values on every config load, so a change made by any route — including
   one this hook never sees — stops the system until a human re-locks it.

7. **`HALT` stops everything.** If a file named `HALT` exists at the project
   root, no order may be constructed. To stop the system: `touch HALT`.

   The kill switch writes it automatically when realized daily losses breach
   `kill_switch_daily_loss_pct`. If you find a HALT file you did not expect,
   **do not delete it** — read it, then tell the user. It records which cycle
   tripped it and when.

## Architecture

Claude owns the loop. Python owns the decisions.

```
  Claude (this agent)                    Python core
  ───────────────────                    ───────────
  MCP: fetch quotes, bars,
       indicators, earnings   ──────►    build_snapshot()
                                              │
                                         strategy.evaluate()   -> Signal
                                              │
                                         RiskEngine.evaluate() -> RiskDecision
                                              │
                                         critique()            -> CriticReport
                                              │
                              ◄──────    build_order_payload() -> ExecutionPlan
  MCP: review_equity_order
  human confirmation
  MCP: place_equity_order
```

**Nothing in `src/` performs broker I/O.** No module may import `requests`,
`httpx`, or an MCP client. This is what guarantees no test, import, or stray
call can place an order. The only side effects are journal writes.

The Python core is a pure function from (market payloads + account state) to a
decision. That makes every decision reproducible: the snapshot is persisted
with the audit entry, so any past cycle can be replayed exactly.

### The seam

Everything crosses the boundary as JSON:

```bash
.venv/Scripts/python.exe -m agentic_trader.cli evaluate --input bundle.json
.venv/Scripts/python.exe -m agentic_trader.cli report
.venv/Scripts/python.exe -m agentic_trader.cli config-check
```

`evaluate` takes raw MCP responses **verbatim**. Do not reshape, round, or
clean them — the parser expects the broker's exact format, and hand-editing
numbers destroys reproducibility.

## Layout

| Path | Role |
|---|---|
| `models/` | Domain types crossing every layer |
| `market/snapshot.py` | Raw MCP payloads → `MarketSnapshot` |
| `market/signals.py` | Pure predicates strategies compose |
| `market/regime.py` | Trend classification; gates which strategies may fire |
| `strategies/` | Opinions only. No account access, no sizing |
| `risk/limits.py` | Pass/fail gates, incl. sector cap and kill switch |
| `risk/sizing.py` | Dollar-denominated position sizing |
| `risk/engine.py` | The only path from signal to executable order |
| `agents/critic.py` | Mechanical re-derivation of the trade |
| `agents/orchestrator.py` | One cycle, as a pure function |
| `execution/executor.py` | Builds the payload. **Does not submit** |
| `execution/shadow_executor.py` | Simulated fills with pessimistic slippage |
| `journal/` | SQLite: audit stream + trade records |

## Risk controls

Beyond sizing, six gates can stop a trade. All are configured in
`config/risk.yaml` and validated at startup.

| Control | Behaviour |
|---|---|
| `max_daily_loss_pct` | Blocks new entries. Resets tomorrow. |
| `kill_switch_daily_loss_pct` | **Sticky.** Writes HALT; a human must clear it. Exits still allowed. |
| `max_sector_exposure_pct` | Caps combined exposure to one sector. Also caps sizing. |
| `min_risk_reward` | Rejects setups whose target does not justify the stop. |
| `max_open_positions`, `max_portfolio_exposure_pct` | Portfolio-level ceilings. |
| Earnings blackout, cooldown, liquidity, max stop width | Per-symbol gates. |

Two behaviours worth knowing before they surprise you:

- **The sector cap binds immediately.** The configured universe is all one
  sector, so one position at the ceiling blocks the next. The fix is a more
  diversified universe, never a looser cap.
- **`min_risk_reward` never fires for `trend_pullback`**, which builds its
  target at exactly 2R. It guards future strategies whose targets come from
  structure. Raising it above 2.0 blocks every entry rather than improving
  selectivity.

The critic may reduce a trade's confidence, which shrinks the position, but it
can never raise it — confidence multiplies notional, and letting a model
enlarge a position is the one coupling this design forbids.

## Broker constraints that shape the design

Discovered from the MCP tool schemas, not assumed:

- **The quote carries two prices, and the newer one wins.** `get_equity_quotes`
  returns `last_trade_price` (regular session) and `last_non_reg_trade_price`
  (extended hours), each with its own `venue_*_time`. Outside regular hours the
  extended print is the live one, and reading only `last_trade_price` quotes a
  price hours stale. There is **no `updated_at`** — every timestamp is
  per-field, and `bid_price`/`ask_price` of `0` mean "no book", not a zero
  spread. `build_snapshot` handles all of this; do not re-derive it in prose.
- **Fractional and dollar-denominated orders must be `type: market`,
  `market_hours: regular_hours`.** A fractional *limit* order is rejected. Since
  a small account can only take a position in a high-priced name fractionally,
  entries are market orders — so `preflight` bounds quote age, real bid/ask
  spread, and price drift separately, and all three are load-bearing. Each
  refuses on *unknown*: a missing timestamp or an unusable book blocks the
  order rather than passing.
- **The entry order cannot carry a stop.** `stop_price` selects a stop order
  *type*; it does not attach protection to a market buy. Stops in this system
  are **managed**, and only half-enforced today: `trend_pullback` exits when the
  live price or a bar low breaches the level recorded in the journal, but **no
  `stop_market` order is placed at the broker**, so between cycles the position
  is genuinely unprotected. This is the single most important operational gap;
  never describe a position as protected when it is not.
- **`ref_id` must be a UUID** and is the broker's idempotency key. The risk
  engine derives it deterministically (UUIDv5) so a re-fired cycle dedupes on
  both sides.
- **All accounts on this login are cash accounts.** T+1 settlement: sale
  proceeds are unspendable until settled, and buying with them causes a
  good-faith violation. Broker buying power already excludes unsettled cash;
  the risk engine tracks it separately so a small order has a stated cause.

## Development

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
```

When adding a strategy:

1. Subclass `Strategy`, decorate with `@register`, set `name`.
2. **Never emit ENTER on an unknown condition.** A missing indicator means the
   condition failed. Trading on absent data is the failure mode a backtest will
   never warn you about.
3. Populate `reasons` and `failed_conditions` on every signal, including the
   ones declining to trade — the critic and the performance review read them.
4. Add a test per entry condition that breaks exactly that condition. A
   condition no test can break is not doing anything.

## Secrets and identifying data

Nothing identifying goes in a tracked file. Account numbers are permanent once
committed to git history.

- `config/account.local.yaml` — gitignored, holds the real account number.
  Copy it from `config/account.example.yaml`.
- Captured MCP bundles embed account numbers and positions. Write them to a
  temp directory, never into the repo; `*.bundle.json`, `bundles/`, and
  `scratch/` are gitignored as a backstop.
- Mask account numbers in anything a human reads — `config-check` already does.

## Current state

- `trend_pullback` — implemented and tested.
- `momentum` — stub, disabled. Returns NONE.
- Mode — **shadow**. Nothing has traded live.
- Account — ~$100 buying power, cash account, options not enabled.

Before going live, see the go-live checklist in
`.claude/skills/review-performance/SKILL.md`. The short version: 30+ closed
shadow trades, positive expectancy in R, stops demonstrably respected.
