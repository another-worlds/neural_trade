# O8 report: a vol-band strategy on the geometry-enhanced size forecast, backtested on independent data

SPEC: `SPEC.md` (e8a2c05), with an addendum committed before any test number (509d962, PESS fills).
- **Selection:** 2446ca2. The choice was committed alone in 3253588.
- **Test code:** PESS was committed in 6dc5ce6 before any test number.
- **Results:** 94933c7.

CPU only: selection 161 s, tests 157 s. Tests: `tests.log` (25 passed).

This report was written by the outlook lead from the implementer's text, because the implementer's Write tool refused the file. The verdict
lines were checked against `tables.txt`.

## Verdicts (pre-registered, pooled B + C, 26 months, 95% t-interval)
- **strategy works on independent data: no**
- **geometry adds to the strategy: no**

| criterion | value | result |
|---|---|---|
| 1. L1 monthly net, 2 bps, primary fills | +1.971 % [+0.050, +3.892], 17/26 months positive | pass (barely) |
| 1. same, PESS fills (stop checked in the entry bar) | -1.057 % [-3.374, +1.260] | **fail** |
| 2. L1 - SHUF (shuffled sigma, the null) | -1.242 % [-3.401, +0.916] | **fail** |
| 3. L1 - R0 (same strategy, sigma without geometry) | -0.375 % [-1.407, +0.657] | **fail** |

## What was built
- **Choice:** the selection months 2021-01..2024-06 chose BRK_k1.5_H60_s1.0 out of 24 configurations. The rule:
  - every 15 minutes, place stop orders at +-1.5 sigma_hat, where sigma_hat is L1's 60-minute size forecast;
  - ride the move up to 60 minutes;
  - stop out at 1 sigma.
- **Selection result:** +4.0 % per month at 2 bps, t = +3.49. All 12 band-reversion (REV) configurations lost.

## Numbers by span (L1, monthly net %, 2 bps; primary / PESS)
| span | L1 primary | L1 PESS | L1 - SHUF | L1 - R0 | buy and hold (log, max DD) |
|---|---|---|---|---|---|
| B 2024-07..2025-07-20 | +3.68 [+1.40, +5.97] | +1.18 [-1.82, +4.18] | -3.67 [-6.63, -0.71] | -0.66 [-2.69, +1.36] | +63.9 % (33.1 %) |
| C Binance 2025-10..2026-10-12 (frozen) | +0.26 [-2.86, +3.37] | -3.30 [-6.77, +0.18] | +1.19 [-1.70, +4.07] | -0.09 [-0.99, +0.81] | -33.2 % (75.4 %) |
| A 2018-07..2020-12 (not the verdict) | +4.65 [+0.53, +8.77] | -4.29 [-8.17, -0.41] | +1.80 [-2.09, +5.70] | -0.74 [-2.00, +0.52] | +152.0 % (102.7 %) |

Trade statistics:
- **Trades:** B 955 (hit 43.5 %, max DD 11.9 %); C 1,154 (hit 40.1 %, max DD 17.9 %, Calmar 0.18); A 2,015 (hit 37.2 %, max DD 56.6 %).
- **Mean trade:** about +5 bps in B and +0.3 bps in C. The median trade is about -11 bps, a typical breakout profile.
- **Exits:** about 40 % stop out and the rest hit the time stop.
- **5 bps:** B +1.48, C -2.41, A +2.64.

**The owner's target (hit > 60 %, drawdown < 5 %) is not met by any arm in any span.**

## Reading (lead)
1. **The edge does not survive honest fills.**
   - With the primary fills the strategy is positive on B, flat on C (the most recent, fully frozen period) and positive on A.
   - Once the stop is also checked inside the entry minute (PESS), it loses on every span.
   - The selection profit came largely from ignoring reversals inside the breakout minute: a fill-model artefact, not an edge.
2. **The size forecast does not drive it.** R0 (no geometry) matches L1, and shuffled sigma (SHUF) is as good or better (B: +7.4 % with 2.6x the trades).
   - Whatever the strategy earns comes from momentum after a band break in trending periods (B, A).
   - It does not come from WHEN the forecast says moves are large.
3. **Consistent with the whole project.** The size of moves is forecastable (O1, O5, H53, H83), but on spot a size forecast alone does not make money: direction is needed, and minute-scale direction has decayed (H51, H52). T02 (vol-timed exposure) and H78/H79 (vol-sized direction) failed the same way.

## Caveats
- **PESS** is a bound (the worst ordering inside the entry minute), not a fill model. The truth lies between primary and PESS.
- **Months:** the months are not independent. These months are partial: 2025-07 (to the 20th), 2025-10 (from the 5th) and 2026-10 (12 days).
- **C:** earlier tactical studies read these months; the 12 days of 2026-10 were unseen.
- **Fills and costs:**
  - stop-market fills carry no slippage beyond 2 bps, which is optimistic for BRK;
  - when both levels trade in one bar, the level nearer the open is taken (3 of 3,645 selection trades).
- **TRAIL** is not scale-matched (descriptive only).
- **Gap guard:** decisions whose next 75 bars cross the cut file's 19-hour gap (before 2025-03-15 19:21) place no orders.
- **SHUF** is not a clean null: shuffled sigma changes the number of trades.

## Deviations
1. **PESS** follows the addendum.
2. **Gap guard:** a decision needs consecutive minutes.
3. **Accounting:** a position's month is its entry month, and drawdown is taken in trade order.
4. **Skipped decisions:** a decision with an invalid feature mask (a zero-RV hour) is skipped (820 of 277k).
5. **Same-bar timing:** after a time stop, a decision on the same bar is allowed.
6. **Metrics:** daily Sharpe uses exit-day sums over all calendar days; Calmar is the annualised net return over max DD.
7. **A's early windows** are shorter than 365 days (as in O5).
8. **`run_o8.py test`** refuses to run twice: it exits if `tests.jsonl` exists.

## Files
`selection.jsonl`, `tables_selection.txt`, `choice.json`, `tests.jsonl`, `tables.txt`, `trades_{A,B,C}.csv.gz`, `equity_dd.png`
and `.plotly.json`, `windows.json` (6 real cut-file example trades, plus 2 forward trades as numbers), `run.log`, `tests.log`,
`runtime_*.json`.
