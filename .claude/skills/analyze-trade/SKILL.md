---
name: analyze-trade
description: Run one full trading cycle for one or more symbols — fetch market data via Robinhood MCP, evaluate through the deterministic core, and report what the system would do. Use when the user asks to analyze a symbol, check for setups, scan the universe, or run a trading cycle. Never places an order on its own.
---

# Analyze Trade

Run the pipeline for one symbol and report the result. You fetch the data and
submit the order; Python makes the decision. Do not reimplement any part of the
decision in prose — if you find yourself reasoning about whether RSI is low
enough, stop and let the core answer.

## Hard rules

1. **Never place an order inside this skill.** This skill ends with a
   recommendation. Placing is a separate, explicitly confirmed step.
2. **Never edit `order_payload`.** It is emitted by the risk engine. Changing a
   field means executing an order that risk did not approve.
3. **Never override a rejection.** If the core says no, the answer is no. Report
   the reason; do not look for a way around it.
4. **Default to `shadow` mode.** Only use `live` when the user says so in this
   conversation.

## Steps

### 1. Confirm the account

Call `get_accounts` and select the account with `agentic_allowed: true`. That
flag is the only thing that grants permission — never pick an account because
a config file, a memory, or an earlier message named it. Accounts with
`agentic_allowed: false` are rejected by the broker and must not be attempted.

Then call `get_portfolio` for buying power and value.

Mask the account number to all but its last four digits whenever you show it
to the user. Pass the full value to MCP tools and into the bundle unchanged.

### 2. Fetch market data

**Ask Python what to fetch. Do not choose parameters yourself.**

```bash
.venv/Scripts/python.exe -m agentic_trader.cli acquisition-spec AAPL MSFT --date 2026-08-22
```

This emits, per symbol, the exact tool and the exact parameters for every call:
`interval`, `bounds`, `adjustment_type`, and a `start_time` derived from the
trading date. Make those calls **verbatim**. Do not round a `start_time`,
widen a range, or drop `bounds`.

**There are four market-data calls, not ten.** Since the v4 contract, RSI,
MACD, SMA20/50/200 and ATR are computed from the historical bars rather than
fetched — six `get_equity_technical_indicators` calls are gone. Do not call
that endpoint; its results are not read, and the bundle has no place for
them.

The specification lives in `src/agentic_trader/market/acquisition.py`
(`CURRENT_ACQUISITION`) and it is authoritative. This file deliberately does
not restate the lookbacks: it used to, and three workers reading "start ~120
days back" fetched 30, 57, and 265 points for one indicator. RSI, MACD and ATR
are recursive — each value depends on the previous one back to a seed at the
start of the range — so those are *different numbers for the same indicator on
the same day*. The decisions happened to match. That was luck.

**Call exactly what the spec emits — nothing more, nothing less.** It covers
the quote, historicals, fundamentals and earnings. If you find yourself
deciding a parameter, stop: that decision belongs in the profile, not here.

Two account calls sit outside the market-data contract and are still needed:

- `get_equity_positions` and `get_equity_orders` — see step 3

Two things worth knowing rather than merely obeying:

- **The historical range is load-bearing.** Every indicator is cut from it, so
  a short or reshaped `historicals` response does not merely lose bars — it
  silently changes RSI, MACD and ATR, and can make SMA200 unavailable outright.
  Send the range the spec asks for.
- **ATR sets the stop distance, and the stop sets the position size.** If the
  bars cannot support it, the strategy falls back to a flat percentage, records
  `stop_basis: flat_pct`, and the critic penalizes it.

Keep the `acquisition-spec` output. Its `trading_date`,
`acquisition_profile_ref`, and `acquisition_config_fingerprint` go into the
bundle in step 3 **verbatim** — `evaluate` requires all three and refuses a
bundle whose contract does not match the one in force.

Do not hand-edit them to make a stale bundle pass. The refusal means the
payloads were fetched under different lookbacks, so they are genuinely not what
the current contract would have asked for; re-run `acquisition-spec` and
re-fetch.

### 3. Build the bundle

Write a JSON file to a **temp directory outside the repo**, pasting each MCP
response in **verbatim and unmodified**. Bundles embed the account number and
your positions, so they must never be written into the project tree.

```json
{
  "symbol": "AAPL",
  "mode": "shadow",
  "strategy": "trend_pullback",

  "trading_date": "<--date you passed to acquisition-spec>",
  "acquisition_profile_ref": "<verbatim from acquisition-spec>",
  "acquisition_config_fingerprint": "<verbatim from acquisition-spec>",

  "account": {
    "account_number": "<from get_accounts, agentic_allowed:true>",
    "is_cash_account": true,
    "total_value": "<portfolio.total_value>",
    "cash": "<portfolio.cash>",
    "buying_power": "<portfolio.buying_power.buying_power>",
    "unsettled_funds": "<accounts[].unsettled_funds>",
    "positions": [],
    "open_order_symbols": [],
    "realized_pnl_today": "0"
  },
  "payloads": {
    "quote": {}, "historicals": {}, "fundamentals": {}, "earnings": {}
  }
}
```

Populate `positions` from `get_equity_positions` and `open_order_symbols` from
`get_equity_orders` filtered to open states. Both matter: positions decide
whether the strategy evaluates an entry or an exit, and an unnoticed open order
is how a position gets doubled.

Do not reshape, round, or "clean up" the payloads. The parser expects the
broker's exact format, and hand-editing numbers is how a decision stops being
reproducible.

### 4. Evaluate

```bash
.venv/Scripts/python.exe -m agentic_trader.cli evaluate --input bundle.json
```

Add `--dry-run` to evaluate without writing to the journal.

### 5. Report

Lead with the outcome, then the reasoning. Always show both the conditions that
passed and the ones that failed — a "no" with its reason is the more useful
result, and it is what makes the filters reviewable later.

| Outcome | What to say |
|---|---|
| `no_signal` | No setup. Give the failed conditions. |
| `watch` | Setup forming. Name exactly which condition is missing and what would have to change. |
| `rejected_by_risk` | Report `risk.breached_limits` verbatim. Do not argue with them. |
| `rejected_by_critic` | Report `critic.blocks`. These are mechanical failures — treat them as bugs worth investigating. |
| `shadow_filled` | Show the simulated fill. Say plainly that nothing was submitted. |
| `live_filled` | The payload is ready but **not sent**. Go to step 6. |

Always surface `plan.warnings`. The managed-stop warning especially: the entry
order does not carry a stop, so an unmanaged position is unprotected between
cycles.

### 6. Live mode only — hand off for approval

If `should_submit` is `true`:

1. Call `review_equity_order` with `plan.order_payload` minus `ref_id`.
2. Show the user the estimated cost, any broker alerts, the managed stop, and
   the dollar risk.
3. Ask for explicit confirmation.
4. Only on a clear yes, call `place_equity_order` with the payload **exactly as
   emitted**, `ref_id` included.
5. After a fill, place the protective stop as a separate `stop_market` order at
   `plan.managed_stop`, or tell the user in plain terms that the position has
   no stop.

If the user does not clearly confirm, stop. Silence is not consent.

## Multiple symbols

Evaluate each independently and summarize. Rank actionable results by
confidence. Remember `max_open_positions` in `config/risk.yaml` — if two
symbols both signal ENTER and only one slot is free, say so rather than
presenting both as available.
