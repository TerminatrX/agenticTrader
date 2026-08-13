---
name: review-performance
description: Analyze the trading journal — realized performance, which risk gates are firing, whether the strategy edge is real or noise, and what to change. Use when the user asks how the system is doing, wants a post-mortem, or asks whether to go live.
---

# Review Performance

Assess whether the system is working. The bar is honesty, not encouragement: a
review that concludes "this is not working yet" is a successful review.

## Pull the data

```bash
.venv/Scripts/python.exe -m agentic_trader.cli report
.venv/Scripts/python.exe -m agentic_trader.cli report --strategy trend_pullback
```

Returns `performance` (counts, win rate, expectancy in R), `open_positions`,
`rejection_reasons` (tallied gate breaches), and `recent_audit`.

## Read it in this order

### 1. Sample size, before anything else

Under ~30 closed trades, **say so first and loudly**. Win rate on 8 trades is
noise, and every conclusion below is provisional until the sample supports it.
Resist the pull to find a pattern in a handful of results — that pull is the
main way a trading system gets destroyed by its own review process.

### 2. Expectancy in R, not dollars

`expectancy_r` is the number that matters: average result in units of initial
risk. Positive expectancy means the edge exists; win rate alone means nothing,
since a 35% win rate at 3R is excellent and an 80% win rate at 0.2R with
occasional full losses is a slow bleed.

Compare `avg_win` to `avg_loss`. If average loss exceeds average risk budget,
stops are not being respected — that is an execution defect, not a strategy
one, and it is more urgent than any edge question.

### 3. Which gates are actually firing

`rejection_reasons` shows what blocks trades. Read it as diagnosis:

- One gate dominating means either the filter is miscalibrated or the universe
  is wrong for the strategy. A `momentum_stabilizing` that blocks nearly
  everything may be correct in a falling market and much too strict in a rising
  one.
- `earnings` blocking often means the universe needs staggering across
  reporting calendars.
- `sized notional below minimum` means the account is too small for the current
  `risk_per_trade_pct`. Say so plainly rather than suggesting the minimum be
  lowered.
- Very few rejections is a warning, not a success. Filters that never fire are
  not filtering.

### 4. Are the losses the expected kind?

For each losing trade, check whether the loss came from the thesis breaking or
from something the strategy never modelled — an earnings gap, a market-wide
drop, a stop hit by noise before the move worked. Those need different fixes,
and averaging them together hides both.

### 5. Shadow versus live

Shadow results assume a fixed slippage (10bp by default). If live fills are
consistently worse, the shadow record is systematically optimistic and every
historical conclusion drawn from it needs discounting. Compare shadow entry
prices against real fills from `get_pnl_trade_history`.

## Recommendations

Give at most three, each with a specific change and its justification from the
data. Tie every recommendation to a number in the report.

Good: "Widen `rsi_ceiling` from 45 to 50 — 6 of 9 `watch` outcomes failed only
the RSI band, with RSI between 45 and 49, and 4 of those 6 rose more than 3%
over the following week."

Bad: "Consider tuning the RSI parameters."

Be direct about the possibility that the strategy has no edge. Parameter
tuning on a small losing sample is how overfitting starts, and the honest
recommendation is often "keep it in shadow and gather more data".

## The go-live question

If asked whether to trade live, require all of:

- 30+ closed shadow trades
- Positive expectancy in R
- Average loss within the configured risk budget (stops respected)
- No gate producing surprises on review
- Losses that are explainable and of the expected kind

If any fail, say which, and say what would have to change. Do not soften it —
this is the one recommendation where being agreeable costs real money.

Also confirm the operational preconditions, which are separate from
performance: the `HALT` file works, `max_order_notional` is set to something
survivable, and the managed-stop gap is understood — the entry order does not
carry a broker-native stop, so an unmonitored live position is unprotected
between cycles.
