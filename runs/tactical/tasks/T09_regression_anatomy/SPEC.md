# T09: what the regression finds (an anatomy of the direction signal)

Lead, 2026-10-13. Owner: "сделай глубокое исследование того, что регрессия находит".

**Kind.** Descriptive: no trading verdict, no choice of a model. The questions and methods are listed here before any
number, so the answers are not shaped after the fact.

## Overlap with the outlook session
Regression is the outlook session's topic (TACTICAL.md "Division of work"). The owner asked this session directly.
- This study writes only under `runs/tactical/tasks/T09_regression_anatomy/`.
- It is reported to the outlook session through LOG.md.

## The model
- **What:** the 13-feature logistic regression `tb7` (7 textbook indicators at the window's last bar, plus 6 trailing returns), C = 0.1.
- **Training:** monthly walk-forward, 60-day training, horizon 15 minutes, 5-bps deadband, as in H51 / H59.
- **Data:** the cut file 2017-08..2025-07-20, plus the Binance forward file 2025-10..2026-09 (descriptive only).
- **Never read:** 2025-07-21..2025-09-30.
- **Labels:** both close-entry and next-open-entry. Every number states which.

## Questions

**Q1. Which features carry it?**
- For each month, record the standardised coefficients.
- For each feature and year, report:
  - the univariate AUC;
  - the drop-one AUC loss;
  - the grouped permutation importance on the month's own data.
- Which 1-3 features explain most of the AUC, and is it the same features every year?

**Q2. What shape is it?**
- Partial dependence: binned P(up) against each top feature, in 20 quantile bins. Is it linear?
- Is it symmetric for up and down moves?
- Is it stronger for large moves (tails) or spread evenly?

**Q3. Is it reversal, and over which lag and horizon?**
- An impulse-response table: the AUC and mean signed return of fading the last k-minute return (k = 1, 2, 3, 5, 10, 15, 30, 60) for target horizons of 1, 5, 10, 15, 30 and 60 minutes, by year.
- **Equivalent simple rule:** "fade the last k minutes". What share of the regression's AUC does the best simple rule reproduce, per year?

**Q4. How much is microstructure (bid-ask bounce)?**
- Close-entry against next-open-entry, and against a mid proxy: entry at (high + low) / 2 of the next bar.
- The AUC and bps by a spread proxy: the 1-minute high-low range in bps, and the Corwin-Schultz estimator from two bars.
- Does the signal concentrate where the spread proxy is large?

**Q5. When does it work?**
- AUC and bps by:
  - volatility tercile;
  - volume tercile;
  - hour of day (UTC);
  - weekday versus weekend;
  - the size of the last move.

  Each from training-span terciles.
- Which conditions held up in 2024-25, and which faded?

**Q6. Why did it decay?**
- Per year, decompose the AUC drop into:
  - the feature strength (univariate AUC);
  - coefficient instability (stable 2019 coefficients against monthly refits);
  - conditions (Q5).
- Relate the yearly AUC to market-quality proxies: the median spread proxy, volume, the 1-minute return autocorrelation at lag 1, and the variance ratio VR(1, 15).

**Q7. Economic size.**
- Gross bps per trade of the top 10% by year, at close and at next-open entry, set against the spread proxy (which is the cost of trading it).
- At what cost, in bps per round trip, does it break even each year?

**Q8. The hourly cousin.**
- The same anatomy, Q1 and Q3 only, for the hourly `rich` logistic of H62 (4-hour horizon).
- Is it also reversal, or momentum?

**Q9. The present.**
- Q1, Q3 and Q4 on the Binance forward year, 2025-10..2026-09.
- Is anything left, and what?

## Outputs
`runs/tactical/tasks/T09_regression_anatomy/`: code, `results.json`, `tables.txt`, and figures in the tactical_report style (theme.py):
- a coefficient heatmap (feature × month);
- importance by year;
- partial dependence curves;
- the impulse-response heatmap (lag × horizon) by year;
- AUC by condition;
- the decay decomposition;
- break-even cost by year;
- one real BTC window with the signal and the "fade" trades drawn on price.

A plain-language summary goes to the lead. Tests: the cutoff, the never-read span, walk-forward causality, the
next-open labels.
