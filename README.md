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

Discovery and evaluation are separate commands because they cost different
amounts. Discovery works on batched payloads — one `get_scans`, eight
`run_scan`, and fundamentals at ten symbols per call — and its job is to decide
which candidates justify the ~9 single-symbol calls that evaluation costs.

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

## Discovery

The sector cap made a single-sector universe untenable, so candidate selection
runs as its own phase against Robinhood's server-side scanner.

`run_scan` returns **at most 200 rows**, offers no pagination, and reports a
`total_items` that does not survive scrutiny. One scan of the tradable universe
hit that cap — meaning the result was the first 200 by market cap, not
everything matching. A biased universe that looks complete.

The fix is structural rather than a re-sort. The universe is partitioned into
eight market-cap bands, each a saved scan, sized so every one returns well
clear of the cap:

| | Band | | Band |
|---|---|---|---|
| B1 | $2B–$3B | B5 | $10B–$17.5B |
| B2 | $3B–$4.5B | B6 | $17.5B–$35B |
| B3 | $4.5B–$7B | B7 | $35B–$100B |
| B4 | $7B–$10B | B8 | >$100B |

The definition in force is `agentic-discovery@v3-2026-08-20`, fingerprinted
`fca219ad…`. Verified live on 2026-08-20: **754 symbols, zero duplicates across
bands**, every shard complete with at least 87 rows of headroom. If a band ever reaches
200 the run **aborts**. The answer is to split that band again — never to accept
a truncated universe, and never to relax the coverage requirement.

That union check proves the eight bands did not overlap for the observed
universe. It does not prove two inclusive `BETWEEN` predicates can never collide
on a shared boundary; the runtime duplicate detector remains the guard for that.

### Coverage and drift answer different questions

```
CoverageStatus          did we see the declared universe?
DefinitionDriftStatus   does the declared universe still mean what we think?
```

Saved scans are editable in Legend. `ScanDefinition` pins what eight scan ids
are believed to contain, fingerprinted with SHA-256, so a widened RSI band
cannot leave runs labelled with a version that denoted a different universe.
Severity is graded by whether a difference can change *membership*:

| Drift | Severity | |
|---|---|---|
| Filter | **Blocking** | changes which symbols exist |
| Shard definition | **Blocking** | changes which scans define the set |
| Sort | Conditional | matters only when a shard is capped |
| Display / title | Informational | columns cannot reach a trade |

Session semantics are part of this. The broker bakes `session="all"` into the
filter expression rather than exposing it as a field, and all-session and
regular-hours RSI are different numbers for the same symbol — so it is extracted
from the expression and compared. Drift is **never repaired automatically**:
whether Legend or the definition should change is a human decision.

### Two budgets, and three different reasons to stop

Sector comes from authoritative fundamentals, never from a scanner column. That
costs a call per ten symbols, so it gets a budget of its own, separate from the
much more expensive per-symbol enrichment:

```
754 discovered
  ├── 714  fundamentals_budget    deliberately not requested
  └──  40  fundamentals_selected
        ├──  0  fundamentals_missing   requested, broker returned nothing
        └── 35  enrichment_budget      eligible, but not selected
              └── 5 selected for full enrichment
```

Those three exits are kept distinct on purpose. "We chose not to ask", "we asked
and got nothing back", and "we asked, got an answer, and the answer was that the
sector is unknown" are different facts, and collapsing them would let a failed
batch look like ordinary budget rationing.

Selection is deterministic — seeded by `sha256(date|symbol)`, not builtin
`hash()`, so a re-run on the same trading date plans exactly the same symbols
and the agent can fetch them in a second pass without the set shifting.

**Scanner values never reach a decision.** The RSI, market cap and volume in a
scan row are discovery diagnostics only; the evaluator re-fetches everything
authoritatively. A test feeds contradictory `source_values` for one symbol and
asserts the resulting snapshots are identical.

## The strategy

`trend_pullback` buys an orderly retracement inside an intact uptrend, and only
once the selling has measurably stopped. Exits when the premise breaks (price
loses the 50-day) or momentum runs to an extreme.

`momentum` is a stub with a documented design sketch; it is disabled.

## Risk model

Sizing works backward from the loss you accept, not forward from the cash you
hold:

```
stop distance = clamp(ATR(14) × multiple, floor, ceiling)
risk budget   = account value × risk_per_trade_pct
notional      = risk budget ÷ stop distance
```

A 1% budget with a 5% stop is a 20% position; the same budget with a 10% stop
is 10%. The stop decides the size, not conviction — and the *stock* decides the
stop. Reversing that order, picking a size and then finding a stop to fit it, is
how a position ends up sized by preference rather than by risk.

**The stop comes from measured volatility.** ATR(14) scaled by a multiple, so a
name that routinely moves 3% a day gets a wider stop and a proportionally
smaller position than one that moves 1%. Three things bound it:

| Bound | Why |
|---|---|
| Floor (`min_stop_pct`) | Sizing divides by this. A 0.5% stop implies 200× the risk budget in notional. |
| Ceiling (`max_stop_pct`) | Volatility beyond it **declines the setup** rather than clamping — a stop at the ceiling would sit inside the stock's ordinary daily range, so the risk would only look bounded. |
| Structure | The 50-day may *widen* the stop, never tighten it. It may exceed the ceiling, and the risk engine refuses it there. |

When ATR is unavailable the stop falls back to a flat percentage and records
`stop_basis: flat_pct`. That still trades — refusing would mean no trades
whenever an indicator call fails — but the critic shrinks the position for it,
because a risk boundary asserted from a constant is a weaker claim than one
measured from the stock.

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
| `earnings_blackout_days`, `symbol_cooldown_days` | Event and behavioural gates. Earnings fails closed on unknown — see below. |
| `min_avg_volume_30d` | Liquidity floor. |
| `max_spread_pct` | Real bid/ask spread at submission — paid in full on a market order. |
| `max_price_drift_pct` | How far price may move from the decision price before the setup is re-evaluated rather than chased. |

Three behaviours that are deliberate and will otherwise look like bugs:

- **The kill switch does not block exits.** It fires automatically, and an
  automatic control that strands you in losing positions until you notice does
  more damage than the loss that triggered it. It writes `HALT`, so the *next*
  cycle stops entirely — by then a human is involved.
- **The sector cap is now load-bearing rather than instantly binding.** It was
  written when the universe was a handful of same-sector names, where one
  position at the ceiling blocked the next. Discovery now spans 754 symbols
  across every sector and selection spreads enrichment over them, so the cap
  does the job it was designed for instead of acting as a de-facto position
  limit.
- **`min_risk_reward` never fires for `trend_pullback`**, which builds its
  target at exactly 2R. It guards future strategies whose targets come from
  structure. Raising it above 2.0 blocks every entry instead of improving
  selectivity.

**The earnings blackout fails closed, and knows whose earnings it is looking
at.** This gate was rebuilt after a live run showed it doing something worse
than nothing. `get_earnings_calendar` takes no symbol argument — it is a
market-wide window scan — and the parser reading it never checked the `symbol`
field on a row. So a snapshot for NVO was assigned NVZMY's report date, from a
payload NVO did not appear in, and all five candidates in that run were
journalled with the same fabricated date.

Two rules now hold. Evidence must be **provably about the symbol**: rows are
matched on `symbol`, the source is the per-symbol `get_earnings_results`, and
the gate re-checks identity before reading a date. And **not knowing blocks**:
a missing payload, a malformed one, an unresolved ticker, evidence from another
trading date, or a symbol absent from the response all resolve to
`earnings_status_unknown` and refuse the entry. Only one silence is
authoritative — the source resolved the symbol and every report it holds is in
the past.

Pendingness is decided by `report.date`, never by `eps.actual`. That field was
observed unreliable in both directions: one symbol carried three past-dated
reports whose `actual` was never filled in, and the calendar returned a
future-dated row with `actual` already populated. Using the date can only
over-block.

Config is cross-validated, so contradictory setups fail at startup rather than
behaving strangely later — a kill switch at or below the daily limit, or a
position ceiling above the sector ceiling, are both rejected outright.

### Unknown is refused, never assumed benign

The last gate before an order exists is `preflight`, and all three of its checks
treat missing data as a failure rather than a pass:

| Check | Refuses when |
|---|---|
| Quote age | The quote's **venue timestamp** is older than 120s — **or absent**. |
| Spread | Bid/ask is missing, zero (the broker's no-book sentinel), or crossed. |
| Drift | Live price has run away from the price the decision was made at. |

This is worth stating because the earlier version of all three was unfalsifiable.
Staleness was measured against `captured_at`, which is stamped `now()` when the
snapshot is built — so every snapshot looked fresh, including one replayed from
a stored bundle months later. The spread check compared the live price to the
decision price, which is drift, not spread; bid and ask were never read at all.
Both passed every test they had.

A control that cannot fail is worse than a missing one, because it earns trust
it has not done anything to deserve. `spread_pct` returns `None` rather than
`Decimal("0")` for an unusable book for exactly this reason: a zero spread would
sail through the tightest possible threshold on the worst possible information.

### Market regime is recorded, not enforced

`market/market_regime.py` classifies the backdrop from SPY, confirmed by QQQ:
`BULL_TREND`, `BEAR_TREND`, `RANGE`, `HIGH_VOLATILITY`, `UNKNOWN`. Volatility
outranks direction, and SPY/QQQ disagreeing downgrades a trend to `RANGE` — a
split market means the move is sectoral, not market-wide.

**Nothing gates on it.** No strategy consults it and no limit reads it. The
useful version of this rule looks like

```
trend_pullback   BULL_TREND        expectancy +0.34R
                 RANGE             expectancy -0.18R
                 HIGH_VOLATILITY   expectancy -0.52R
```

and none of those numbers exist yet. Wiring a guess into a gate now would
suppress exactly the trades needed to find out whether the guess was right. The
regime and the inputs that produced it are journalled on every cycle; the rule
follows the evidence, not the other way round.

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

**Known gap: positions cannot be protected at this account size.**

The entry order cannot carry a broker-native stop, so stops here are *managed* —
a separate `stop_market` order has to follow the fill. Investigating whether the
MCP could place one turned up a harder constraint than "not implemented yet":

> Fractional quantities are accepted **only on `type=market`**, and a
> `stop_market` order is not `type=market`. So a fractional position cannot
> carry a resting stop at all.

That binds because of how sizing works. `notional = risk_budget / stop_distance`
is about $20 today, so a position is whole shares only for a stock under roughly
$20 — and the universe trades at $250+. **Every position this account can take
is unprotectable.**

Note what that threshold depends on, though: `stop_distance`, which is now
derived from ATR and so varies per symbol rather than sitting at a fixed
percentage. A $0.50 stop on a $1 risk budget affords a $2 notional; a $2.00 stop
affords $0.50. There is no single price ceiling that separates protectable from
unprotectable — it moves with each stock's volatility. So "trade cheaper stocks"
is *not* established as the answer, and forcing a price ceiling into the scanner
to accommodate a $100 test account would distort which setups the strategy
sees. The honest sequencing is: find what the strategy
actually wants, then ask what capital that requires under whole-share
protection. A $100 account may simply be adequate for shadow validation and
inadequate for protected execution, which is a fine answer.

What exists instead:

- The strategy enforces the stop each cycle — exiting when the live price is at
  or below the level, and when a bar's *low* touched it even if price recovered,
  since a resting order would have filled there.
- `ProtectionState` records the truth per position, and `UNAVAILABLE` is
  distinct from `FAILED`: one is a standing property of the account, the other
  an incident.
- Shadow mode may carry an unprotectable position and journals why. **Live and
  approval execution refuse it structurally** — no configuration reaches that
  check.

This is why `estimated_max_loss` is a *modeled* loss and not a floor.

**On confidence:** the restriction is `SCHEMA_DOCUMENTED`, not
`EMPIRICALLY_VERIFIED`. `review_equity_order` previewed a fractional
`stop_market` sell without complaint — but it also accepted a short sale in an
account holding none of the symbol, so it appears not to validate order
parameters at all. Confirming the rule would mean placing a real order, which
this project will not do to settle a question. `execution/capabilities.py` tracks
that distinction per capability rather than burying it in a comment.

## Layout

```
src/agentic_trader/
  models/        domain types crossing every layer
  market/        MCP payloads → snapshot; signals; regime
  universe/      scan definition, coverage, drift, candidate selection
  strategies/    opinions only — no account access, no sizing
  risk/          limits, sizing, engine
  agents/        orchestrator (one pure cycle), discovery, critic
  execution/     payload construction, shadow fills
  journal/       SQLite audit stream + trade records
  config/        schema-validated loading + integrity lock
  cli.py         the JSON seam
config/
  risk.yaml            limits (agent cannot edit)
  risk.lock            SHA-256 baseline of the above
  strategies.yaml      per-strategy switches and params
  account.example.yaml template; copy to account.local.yaml (gitignored)
tests/
  fixtures/      live saved-scan configuration, for drift tests
.claude/
  skills/        analyze-trade, critique-trade, review-performance
  hooks/         guard_risk_config.py — the deny hook
  settings.json  wires the hook (tracked, so protection travels with the repo)
```

## Next

In rough priority order:

1. **Close three journal and reproducibility gaps** the first live discovery run
   exposed: `scan_runs` does not persist `trading_date` even though selection is
   date-seeded; audit records do not persist execution mode; and indicator
   lookback parameters are not pinned centrally, so two fetches of the same
   indicator contract can return different series lengths.
2. **How much capital this strategy needs for whole-share broker protection.**
   Answering it earlier would be guessing — ATR makes stop distance vary per
   symbol, so the price ceiling implied by `risk_budget / stop_distance` is not
   one number. Now that a real universe exists, the question is answerable.
3. **The protective-stop lifecycle** — submit, confirm acceptance, record the
   broker order id, monitor, reconcile on restart. Gated on (2): until positions
   can be whole shares it could never leave its first state. There is no
   replace/modify tool, so moving a stop means cancel-then-place with an
   unprotected window in between.
4. **Normalized journal**, a session-aware `ShadowExecutor` with realistic
   spread, slippage and stop-gap modelling, and a baseline-vs-critic A/B to
   establish whether the critic actually improves expectancy.
5. **Compute RSI/MACD/SMA/ATR locally from authoritative bars.** Six of the ~9
   calls per symbol are indicator endpoints deriving from bars already fetched.
6. **`ApprovalExecutor`** — last, and gated on evidence rather than on a green
   test suite (see safety rule 8).

## Status

Shadow mode. `trend_pullback` implemented and tested; `momentum` stubbed.
Discovery runs live against an eight-shard universe. **No live trades placed.**

Last end-to-end shadow run, 2026-08-20: 754 symbols discovered, 40 fundamentals
requested, 5 selected and fully enriched, 5 evaluated. Four `watch`, one
`no_signal`, zero entries — so the sizing, critic and risk path were exercised
by unit tests but not by that run. Zero order, cancel or replace calls have ever
been made.

283 tests, ruff clean. Test coverage is weighted toward the negative cases —
every risk gate has a test proving it *blocks*, because a limit that silently
fails open is worse than no limit at all.

**That number is not evidence of live readiness**, and is deliberately not
offered as any. A green suite shows the code does what it was written to do. It
says nothing about whether the strategy has an edge, whether shadow fills
resemble real ones, whether the system survives a restart mid-position, or
whether its view of the account matches the broker's. None of those are
established yet.
