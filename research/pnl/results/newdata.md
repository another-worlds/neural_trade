# New-data test: T2-P on markets and periods the study never scored

Date 2026-11-03. Pre-registration: `prereg_newdata.md`, final at commit edcb157, before any scoring. Scored once at
commit 3c84f4a (`results/newdata.json`, `newdata_daily.npz`, `newdata_lock.json`). Code: `nd_fetch.py`, `nd_run.py`.
T2-P is unchanged: daily trend, lookbacks 7/14/28/56 days, 50% volatility target, band 0.2, long or short, at most 1x.

## 1. Answer

**On 79 other crypto perpetuals, T2-P passed the pre-registered test: REPLICATED. The pass is at the 5% line, not
well inside it.**

| Equal capital per market, 2021-01..2026-10, net of fees and funding | T2-P | buy-and-hold (same markets) |
|---|---:|---:|
| Mean net return per day | +4.49 bps | +0.60 bps |
| One-sided 95% lower bound (stationary bootstrap, block 20) | **+0.12 bps** | - |
| Bootstrap p | 0.047 | - |
| Per year (additive) | +16.4% | +2.2% |
| Sharpe | 0.67 | 0.03 |
| Maximum drawdown | 30% | 88% |

- **What it shows.** The rule's edge carries over to markets it was not chosen on. It earned in 5 of 6 years
  (log: 2021 +39%, 2022 +12%, 2023 +22%, 2024 +15%, 2025 -10%, 2026 to October +17%). 58 of 79 markets made money.
- **Why it is weak.** The independent part is small. The panel's returns correlate 0.58 with BTC's own T2-P. After
  removing that part, about 2.7 bps/day remain (HAC t 1.15, not significant). 83% of the days overlap the window on
  which T2-P was chosen for BTC. Small changes flip the verdict to CONSISTENT (section 5).
- **For the outlook.** The edge is more likely real than before, but it is still not proven. A spread of many markets
  had a smaller drawdown than BTC alone (30% against 46.5%).

## 2. Panel results (primary costs: 5 bps per unit of |x| increased, 10 bps reduced)

| Rule | %/yr | Sharpe | Max DD | Note |
|---|---:|---:|---:|---|
| **T2-P** | **+16.4** | **0.67** | **30%** | verdict REPLICATED |
| tsmom_unscaled (same signal, no volatility target) | +24.3 | 0.53 | 49% | descriptive |
| voltarget_long | +5.7 | 0.14 | 57% | descriptive |
| buy-and-hold | +2.2 | 0.03 | 88% | descriptive |
| T2-P at BTC costs (2 / 5 bps) | +18.0 | 0.74 | 30% | descriptive |
| T2-P at stress costs (25 / 25 bps) | +9.1 | 0.37 | 33% | descriptive |

- **Where the profit came from** (sum of daily returns, % of equity): long price moves +122, short price moves +24,
  funding -15, fees -18. Most profit came from catching rallies (2021, 2023, 2024). Shorts paid in 2022 and 2026.
  So the result is not mainly from shorting the known alt-coin falls, the risk named before the test.
- **Activity.** Mean |exposure| 0.36; turnover about 40 units per market per year (about 3% per year in fees).
- **Data.** 79 markets active (LEND stopped in 2020); 57-79 active per day, median 73; no day lost all its equity;
  no funding month missing inside a traded span. Delisted markets are included (22 of them); they did no worse
  (+18.5 %/yr) than the survivors (+15.8 %/yr).

## 3. Secondary verdicts (same rule; reported, not gates)

| Part | %/yr | Sharpe | Lower bound (bps/day) | p | Verdict |
|---|---:|---:|---:|---:|---|
| Dev window 2021-01..2025-10 | +15.3 | 0.62 | -0.75 | 0.084 | CONSISTENT |
| Sealed window 2025-11..2026-10 | +21.7 | 0.91 | -2.42 | 0.12 | CONSISTENT |
| 20 most liquid markets (chosen on 2020 volume) | +11.2 | 0.44 | -1.41 | 0.13 | CONSISTENT |
| Without the index contract DEFI | +16.4 | 0.67 | +0.15 | 0.045 | REPLICATED |
| BTC 2017-10..2019-12 (spot, no funding) | +39.6 | 0.97 | -3.51 | 0.12 | CONSISTENT |
| BTC 2020-02..2020-12 (perp) | +81 (log, 10 months) | 2.14 | +5.55 | 0.016 | REPLICATED |

- **BTC 2017-2019.** T2-P lost 13% in 2018, when holding BTC lost 73%. Over the whole period buy-and-hold made
  +18.0 %/yr with an 83% drawdown; T2-P +39.6 %/yr with a 39% drawdown.
- **BTC 2020.** T2-P made money, but buy-and-hold made more (+107% log).
- **BTC after the seal (2026-10-31..11-02, descriptive only).** T2-P was fully short: -0.24%, -1.01%, +1.91% (the
  last day includes the test's exit fee), about +0.7% in total. After the 11-02 close the rule is still fully short.
  Three days test nothing.

## 4. Checks

- **Engine.** Engine H (hourly) reproduces the study's engine E on BTC: dev 2021-01..2025-10 within 0.06 %/yr for
  every rule, sealed year within 0.005 log (`newdata_validation.json`).
- **Before scoring.** Two independent review rounds (causality, engine accounting against an independent reference
  engine, scoring and pre-registration). All findings were fixed before the pre-registration was final; the BTC
  validation did not change. The run-once guard checks the committed pre-registration, a clean code tree and the
  SHA-256 of every data file.
- **After scoring.** An agent that did not see `nd_run.py` re-implemented the test from the pre-registration. Its first
  reading gave a lower bound of +0.106 bps/day (REPLICATED). Three conventions that the pre-registration leaves open
  explain the gap. With them, it reproduces every scored number to 5e-15: the trade sets the exposure exactly to its
  target after fees (engine E sizes before fees, a 1e-3 difference); the fee of a trade at the daily close is booked
  on the decision day; a trim and a decision at the same close are one fill. One wording point: the first decision
  day carries the entry fee, so it is not exactly "cash".

## 5. How fragile the result is (post hoc, descriptive: the verdict stays as scored)

- **What holds.**
  - **Bootstrap noise:** all 100 bootstrap seeds give a lower bound above 0, and so do block lengths 5 to 100 days.
  - **No single market carries it:** the best market gives 3.4% of the profit, and 71 of 79 markets add to it.
  - **No hidden long bet:** the beta on buy-and-hold is about -0.03.
  - **Timing:** the rule's timing beats a placebo that shifts the market returns in time (p 0.015).
- **What flips it to CONSISTENT** (the lower bound goes below 0):
  - removing the 2 best markets;
  - costs 15% higher (5.75 / 11.5 bps);
  - leaving out 2021, 2022, 2023 or 2026;
  - keeping only the 49 markets with full history;
  - a plain t-test, or Newey-West at lag 10 or 20 (p 0.051-0.053).
- **What it leans on.**
  - **The first half of 2021:** 36% of the log profit in 8.5% of the days.
  - **The best 5 days:** without them the mean falls to 2.9 bps/day.
  - **The thinner markets:** the most liquid third is CONSISTENT (p 0.11) and the least liquid third is REPLICATED
    (p 0.035). Costs, fills and the no-impact assumption are least reliable in the thinnest markets.
- **Breadth.** The markets move together: the mean pairwise correlation of their T2-P returns is 0.35, which equals
  about 2.7 independent bets, not 79.
- **Recent state.** The portfolio peaked on 2024-12-08 and was still 3% below that peak at the end of 2026-10. It is
  about flat since 2024-08.

## 6. Limits

- **Not blind.** We knew the broad 2021-2026 path of these markets. T2-P was fixed before (SPEC, 2026-11-01), but the
  designers expected long falls.
- **Execution.** The engine checks trims hourly, so it misses wicks inside the hour. Costs are assumed: no market
  impact, and equal weights ignore how little the thin markets can take. Delistings settle at the last traded close;
  this changes almost nothing (the delisted markets' final days sum to -0.04%).
- **Archive.** 2026-10-28 is missing for every market, so no decision is made that day. Days missing from the monthly
  archives were filled from the daily archives.

## 7. What this changes

- **Before:** BTC alone showed a positive but unproven edge (sealed verdict CONSISTENT).
- **Now:** the same rule, unchanged, made money on a broad set of other crypto markets, at the edge of significance.
  That supports the trend edge, but only weakly, because the markets move with BTC and with each other.
- **Practical reading.** A spread of markets cut the drawdown, but the most liquid markets (the ones a real account
  can trade) showed the weaker result. The study's rule still applies: 12 months of paper trading before any money.
  A paper test should run the same rule on BTC and on a liquid basket at the same time.
