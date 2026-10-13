# T09 report: what the regression finds

The lead wrote this from the implementer's hand-back (2026-10-14). Owner: "сделай глубокое исследование того, что
регрессия находит".
- SPEC: [SPEC.md](SPEC.md) (52eb1f8, questions fixed before numbers).
- Code, tables and figures: 47751ad; tests 17 passed.
- Sanity: H51 and H59 reproduced to 6e-7 AUC.
- Literature side: [research/2026-10-13-regression-signal](../../research/2026-10-13-regression-signal/README.md) (H91).

**Data caveat.** The cut file is Bitstamp BTC/USD, with timestamps at UTC+4/+5 until the 2024-09-13 splice. The
2022-08-10 tick change ($0.01 → $1) and the splice confound 2022 and 2024. Q9 runs on Binance, a different venue. Hours
are converted to UTC. Seam minutes (0.57% of anchors) are excluded from Q2, Q3 and the hour split.

## In ten lines

1. **What it is:** the regression fades price stretched over the last 20-60 minutes. It sells after a run above the 20-minute band and buys after a drop below it.
2. **One factor:** all price-position features, trailing returns and RSI say the same thing. Any one group alone gives nearly the full AUC, and volatility features add nothing.
3. **Unstable weights, stable score:** the weights jump between collinear features from month to month (cosine to the 2019 weights falls to 0.01 by 2025). A frozen 2019 model scores the same; refitting adds at most 0.008.
4. **Two parts:**
   - bid-ask bounce or a stale close on a thin venue: 0.02-0.04 AUC of the 2017-20 close-entry edge, gone from 2021;
   - a real 15-60 minute reversal, which is what survives next-open entry.
5. **It lives in the tails:** the most stretched decile wins 56-61% (2017-19) and 53-55% (2024-25). The middle deciles are coin flips.
6. **A simple rule does the same job:** "fade the last 30-60 minutes" reproduces 90-100% of the regression's next-open edge every year. Since 2022 it is as good as the regression.
7. **Why it decayed:** the reversal itself weakened. Same weights and same conditions now predict less (-0.044 to -0.075 of the -0.045 to -0.078 drop). It tracks the market:
   - lag-1 autocorrelation went from -0.09/-0.11 (2017-20) to about 0 (2023+);
   - VR(1,15) went from 0.70-0.81 to 0.95-0.99;
   - the Roll spread went from 5.6-10 to 0.1-1.6 bps.
8. **Economics:** the top-10% gross edge at next open was 1.7-3.6 bps in 2018-21. It has been 0-1 bps since 2022, inside its noise and below any real cost.
9. **Today, on Binance 2025-11..2026-09:** AUC 0.535-0.538 with no bounce. The gross edge is 0.2-0.7 bps per top-10% trade, not separable from zero.
10. **The hourly model (H62) is also reversal** (RSI, VWAP and EMA distance over 1-24 h). It decayed less: 0.58-0.59 in 2019-20, 0.54 in 2024-25.

## Key tables

Yearly AUC (mean over months, ±~0.01):

| Entry | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| close | .568 | .590 | .599 | .590 | .549 | .544 | .554 | .531 | .521 |
| next open | .527 | .562 | .561 | .567 | .553 | .540 | .551 | .533 | .518 |
| mid of next bar | .524 | .563 | .566 | .571 | .554 | .544 | .554 | .533 | .516 |

Top-10% gross bps per signal, next open: 6.0, 2.5, 1.7, 2.8, 3.6, 0.0, -0.1, 1.0, 0.8 (2017-2025).

Decay decomposition against 2019:

| Year | Total change | From signal strength | From condition mix |
|---|---|---|---|
| 2021 | -0.049 | -0.069 | +0.012 |
| 2024 | -0.068 | -0.075 | +0.006 |

Detrended correlation of monthly AUC with lag-1 autocorrelation: -0.51 (p < 1e-6).

Conditions in 2024-25: weekends 0.547 against weekdays 0.521. The largest last moves are 0.55-0.57, with wide noise. No
hour-of-day structure.

## Figures

All in `figs/`, as PNG and plotly JSON:
- coefficient heatmap;
- importance by year;
- partial dependence;
- tail edge;
- impulse heatmaps (close, next open, bps);
- conditions;
- decay decomposition;
- break-even cost;
- real windows (2019, 2025);
- microstructure;
- hourly;
- forward year.

## Deviations

- The purge is stricter than in H51/H59, with an effect of 6e-7.
- The forward fits train only on Binance from 2025-10, so the first month has about 30 days of training.
- The subset and partial-dependence features were chosen after the fact (descriptive only).
- The Q6 condition mix uses fixed 2019 cells; a range-based variant is in tables.txt.
- The heavy pickles are out of git, in D:/nt/nt_t09_cache.
