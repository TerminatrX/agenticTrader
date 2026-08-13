---
name: critique-trade
description: Adversarially review a proposed trade before it is placed — challenge the thesis, hunt for the failure mode, and try to talk the system out of it. Use after analyze-trade produces an actionable signal and before any live order. Returns a verdict that can veto the trade.
---

# Critique Trade

Your job is to argue against the trade. Not to balance it, not to note some
pros and cons — to actively look for the reason it is a mistake. If you cannot
find one after genuinely trying, that is a meaningful result. If you find
yourself agreeing quickly, you are not doing the job.

This exists because the system that produced the signal cannot review itself.
The same reasoning that generated a thesis will generate a defense of it. A
separate pass with an adversarial mandate is the only structural fix.

## What is already checked — do not redo it

`agentic_trader.agents.critic` mechanically verifies:

- Intent matches the signal (symbol, side, stop below entry)
- Risk arithmetic reconciles against `risk_per_trade_pct`
- Notional is within buying power and the concentration ceiling
- Snapshot and indicator freshness
- No required indicator is missing
- Regime permits the direction
- Earnings proximity, large gaps, repeat trading in the symbol

Those run automatically and appear as `critic` in the `evaluate` output. If the
verdict is `block`, the trade is already dead — report why and stop.

Your job is what a function cannot check.

## What to actually examine

### 1. Is the thesis real, or a story fitted to the indicators?

The strategy says "pullback in an uptrend". Look at the actual bars. Is this an
orderly retracement on declining volume, or a sharp break on heavy volume that
happens to sit in the right RSI band? Those look identical to the indicators
and are opposite trades.

Check the volume on the down days specifically. Pullbacks on light volume are
the thesis. Pullbacks on volume heavier than the advance are distribution.

### 2. What has changed that the indicators cannot see?

Indicators are computed through the last completed bar and know nothing about
why price moved. Ask directly:

- Was there a recent earnings report? What did the stock do with it? A 7%
  drop on a beat means the market disagreed with something the numbers did not
  capture, and a technical pullback signal will not know that.
- Guidance changes, analyst actions, sector-wide moves, index events?
- Is the whole sector down, making this a market move rather than a stock one?

Use `get_earnings_results` and `get_equity_fundamentals`. Compare the symbol's
move against a sector or index peer via `get_index_quotes` or a comparable
ticker.

### 3. Where is the stop, and is it in the obvious place?

If the stop sits just below a level everyone can see — a round number, the
50-day, an obvious swing low — expect it to get taken out before the move
works. Ask whether the stop is placed where the thesis fails, or merely where a
percentage calculation landed.

Then check the honest question: if this stop is hit, will it be because the
thesis broke, or because of noise the thesis explicitly allows for? A stop that
noise can hit is not risk management, it is a donation.

### 4. What is the actual downside?

Not the stop distance — the gap risk. This is a cash account holding overnight.
Check the distance to earnings, and recall what the last report did. If the
position must be held through an event that can gap 7%, the stop is decorative.

### 5. Does the portfolio already own this risk?

If existing positions are in the same sector, or correlate with this one, the
real exposure is the sum, not the individual sizes. Two mega-cap tech names are
close to one position of double the size.

### 6. Is this the same trade that lost last time?

Query the journal:

```bash
.venv/Scripts/python.exe -m agentic_trader.cli report --strategy trend_pullback
```

Look at `rejection_reasons` and the closed trades. If this setup has failed
repeatedly in the same way, the strategy has found a pattern that does not
work, and the correct action is to fix the strategy rather than take the trade.

## Verdict

End with one of:

- **BLOCK** — a specific, concrete reason this trade should not happen. State
  the failure mode, not a vague concern.
- **PROCEED WITH CONCERN** — the trade is defensible but something specific
  should be watched. Name the trigger that would change the answer.
- **PROCEED** — you tried to break it and could not. Say what you checked, so
  the record shows the review was real.

Never hedge into meaninglessness. "It could go either way" is not a verdict; it
is a refusal to do the job. Commit to an answer.

## Recording

The verdict belongs in the journal alongside the decision. When the trade is
taken, the critique becomes the record of what was known and considered at the
time — which is the only thing that makes a later post-mortem honest rather
than reconstructed.
