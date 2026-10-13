# T07: an ensemble TRADING strategy with a backtest

Lead, 2026-10-13. Committed BEFORE any result.

**Owner:** "дай ансамблевый не предсказатель, а ансамблевую трейдинг стратегию с бэктестом".

**The question.** Can several different strategies, combined into one portfolio, reach a positive, stable result after
costs where each alone is at about zero today? Diversity here comes from different horizons and different logic: a
minute reversal model, hourly models, and trend and mean-reversion TA rules. T06 averaged look-alike predictors; this
combines different strategies.

## Data, fills, costs
- **Selection and look 1:** the cut main file (last bar 2025-07-20 23:59).
- **Verdict:** the Binance forward file, 2025-10..2026-09. The protected span 2025-07-21..2025-09-30 is never read.
- **Fills:** next-open entry; stops are not used.
- **Costs:** 0 / 1 / 2 / 5 bps per round trip. **The verdict uses 2 bps.**
- **Thresholds:** from the previous month only (V_prev, H79).

## Members (fixed recipes; no tuning in this task)

| Member | Recipe | Rebalance | Model refit |
|---|---|---|---|
| **M1, minute reversal ensemble** | E1 of T06's walk-forward members L0, L5, L6 (mean P(up), 15-minute horizon). Trades when mean-of-3 \|P-0.5\| exceeds the previous month's 90th percentile. Non-overlapping 15-minute holds. | 15 min | monthly, 60-day training |
| **M2, hourly H62 recipe** | rich, logistic C 0.1, 730 days, 4-hour horizon, top 10% with V_prev; non-overlapping. | 4 h | monthly |
| **M3, hourly rich L1** | H56's 1-hour rich L1, 365 days, top 10% with V_prev. | 1 h | monthly |
| **M4, trend** | hourly EMA 20/50 cross, long or short, always in the market. | 1 h | none |
| **M5, mean reversion** | hourly RSI(14): long below 30, short above 70, exit at 50. | 1 h | none |
| **M6, breakout** | hourly Bollinger(20, 2) close outside the band: enter in the direction of the break, exit at the middle band. | 1 h | none |

- Each member is an hourly position series in [-1, 1].
- Its 1-minute or 15-minute trades are aggregated to the hourly net return.

## Combinations (6, fixed here)

| ID | Name | Rule |
|---|---|---|
| C1 | **Equal risk** | each member scaled to the same trailing 30-day volatility of its own returns, then averaged. |
| C2 | **Equal weight** | the plain mean of member positions. |
| C3 | **Vote** | sign of the sum of member positions, active only when at least 3 members are non-zero and agree. |
| C4 | **Inverse-loss** | C1, with members weighted by their trailing 90-day Sharpe clipped to [0, ∞). A member with a negative trailing Sharpe gets 0. |
| C5 | **C1 + HAR vol target** | total exposure scaled by sigma* / sigma_HAR(next 24 h), capped at 1. T03's HAR family, fitted on the 365 days before each month. |
| C6 | **C3 + HAR vol target** | as C5, applied to C3. |

- Gross exposure is capped at 1x notional. Costs are charged on turnover.

## Selection: 2019-01..2024-06
- The ONE combination with the highest Calmar at 2 bps is chosen. It must have at least 100 trades per year and must beat
  its own random-side null's 95th percentile on cumulative return.
- The choice is written to `selection.json` and committed before any look.
- All members and combinations are reported for transparency.

## Look 1 (descriptive, partly seen): 2024-07..2025-07-20

## Verdict: the Binance forward file, 2025-10..2026-09
- Member models are frozen as fitted on data to 2025-07-20.
- 2025-10 calibrates the V_prev thresholds and the HAR scale constant only. The scored months are 2025-11..2026-09
  (11 months).
- **Success at 2 bps** iff all of these hold:
  1. mean monthly net return > 0, with the 95% t-interval lower bound > 0;
  2. cumulative return above the random-side null's 95th percentile (the same exposure path with random signs, 1,000 draws);
  3. Calmar above buy-and-hold's Calmar over the same months;
  4. max drawdown < 15%.
- **Reported against the owner's goal:** hit rate > 60%, drawdown < 5%, and the leverage needed for DD < 5%.

## Also reported
- the members' correlation matrix of hourly returns;
- each member's stand-alone result on every period;
- equity, drawdown and exposure figures, and monthly returns.

## Tests (seconds)
- next-open fills;
- thresholds from the previous month only;
- models fitted only on data before the month;
- the never-read span;
- selection.json committed before the looks;
- costs charged on turnover;
- the null keeps the exposure path.

## Outputs
`runs/tactical/tasks/T07_ensemble_strategy/`: code, `selection.json`, `results.json`, `tables.txt`, figures in the
tactical_report.html style (theme.py), `tests.log`, `run.log`. The report text goes to the lead.
