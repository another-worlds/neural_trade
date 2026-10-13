# T08: trade only on high confidence; a conservative ensemble that trades only on consensus

Lead, 2026-10-13. Committed BEFORE any result.

**Owner:** "Подними метрику уверенности и исходя из нее делай решение о сделке. Также сделай ансамбль, который будет
консервативно торговать только на общем согласии".

**Why it might help.** T07 lost at 2 bps because it traded about 4,000 times a year, on an edge of about 0.4 bps per
trade. Trading rarely and only when confident cuts turnover. It works only if confidence really sorts the good trades
from the bad.

## Inherited from T07 (code `runs/tactical/tasks/T07_ensemble_strategy/`)
- Data, periods, fills (next-open), costs (0/1/2/5 bps, verdict at 2), and frozen forward models.
- The never-read span.
- The six members M1-M6 and their monthly refits.

## Confidence per member
Each member gives a signed score s with a direction sign(s) and a confidence |s|.

| Member | Score s |
|---|---|
| M1 minute reversal | P(up) - 0.5 of E1 (15-minute head) |
| M2 H62 recipe | P(up) - 0.5 |
| M3 1 h rich L1 | P(up) - 0.5 |
| M4 trend | (EMA20 - EMA50) / ATR14 |
| M5 RSI | (50 - RSI14) / 50, sign = reversion |
| M6 Bollinger | (close - mid) / (2 x sd20), sign = breakout direction |

- **Percentile confidence c in [0, 1]:** the rank of |s| within the PREVIOUS calendar month's |s| values of the same
  member (V_prev, H79). Nothing from the current month enters.
- **Decision grid:** hourly, at the first minute of each hour. The minute member M1 uses its score at that minute.

## Trades
- On a signal at hour t, enter at the next open, hold **4 hours**, non-overlapping (no new entry while in a trade).
- Fixed 1x notional.

## Family A: trade on high confidence (one member at a time)
- A member trades when c >= q, in the direction of its sign.
- q is in {0.90, 0.95, 0.98}.
- 6 members x 3 levels = **18 configurations.**

## Family B: conservative consensus ensemble
A member "votes" when c >= q. Trade only when:
- **K1 (models agree):** M1, M2 and M3 all vote, with the same sign.
- **K2 (no dissent):** at least 3 members vote, all with the same sign, and no member's sign opposes them at c >= 0.5.
- **K3 (unanimity):** all 6 vote with the same sign.

q is in {0.80, 0.90, 0.95}. That gives 3 x 3 = **9 configurations.**

## Selection: 2019-01..2024-06 on the cut file
- **One finalist per family:** the highest Calmar at 2 bps among configurations with at least 30 trades per year in
  selection.
- Both finalists go into `selection.json`, committed before any look.
- **Report also** each configuration's hit rate by confidence decile. This is the check that confidence sorts trades:
  the hit rate should rise with c.

## Look 1 (descriptive, partly seen): 2024-07..2025-07-20

## Verdict: the Binance forward file, 2025-11..2026-09
- Members are frozen at 2025-07-20. 2025-10 is calibration only.
- **Success for a finalist at 2 bps** iff:
  1. there are at least 30 forward trades. With fewer, the verdict is "inconclusive: too few trades";
  2. mean net bps per trade > 0, with the 95% lower bound > 0 (a block bootstrap by week);
  3. cumulative return is above the random-side null's 95th percentile (same trades, random signs, 2,000 draws);
  4. max drawdown < 15%.
- **Two finalists, Holm over 2** on criterion 2.
- **Reported:** the owner's goal (hit > 60%, DD < 5%), the hit rate by confidence decile on the forward data, the
  trades per year, and buy-and-hold.

## Tests (seconds)
- the percentile confidence uses only the previous month;
- the consensus rules on synthetic votes;
- non-overlapping 4-hour holds;
- next-open;
- selection.json committed before the looks;
- the never-read span.

## Outputs
`runs/tactical/tasks/T08_confidence_consensus/`: code, `selection.json`, `results.json`, `tables.txt`, and figures in
the tactical_report style. The figures:
- the hit rate against the confidence decile per member;
- the finalists' equity against buy-and-hold and the null band;
- their trades marked on the BTC price for a forward month.

The report text goes to the lead.
