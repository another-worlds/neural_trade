# T13 sprint: synthetic geometric indicators (support and resistance lines) for the predictable targets

Lead, 2026-10-15. Committed BEFORE any result.

**Owner:** "дай еще геометрию синтетических геометрических индикаторов - то есть линии поддержки. Сделай один спринт по
синтезу перспективных и реально помогающих направлений".

**Idea.** Support and resistance lines are geometry that technical analysts draw by hand. Here they are built
algorithmically and pointed at the targets that work:
- the hit probability (T10/T12 H3);
- the size of the next move (O5 L1, the next-hour volatility change, T01/T11).

H49 N4, convex-hull lines, was tested only for DIRECTION, where it failed. These targets have never been tried with
lines.

## The lines (causal; bars < anchor only), on 1-minute and 15-minute bars, lookback L ∈ {60, 240} bars

| Line | Construction |
|---|---|
| S1 hull lines | the upper and lower convex hull of highs and lows over the lookback. Resistance and support are the hull edges active at the last bar, extended to now. |
| S2 pivot levels | swing highs and lows (fractal, k = 3 bars each side, confirmed only k bars later). The nearest unbroken pivot above and below, with the touch count within 0.1 sigma. |
| S3 regression channel | an OLS line of close over the lookback, ± 2 residual sd. |

**Features per line, in HAR-sigma units** (sigma_HAR of the target horizon, so they are scale-free):
- the distance from close to resistance and to support;
- the channel width;
- the line slopes;
- the touch counts;
- the bars since the last touch;
- whether price is inside or outside the channel.

That gives about 30 features: the set **SR**.

## Targets and bases (fixed)

| Part | Base | Target | Variants |
|---|---|---|---|
| **A (primary)** | T10's H3 (unchanged code) | barrier hit, X = 50 bps, T = 60 min | A1 = H3 + SR (logistic, same fitting as H3). A0 = H3. Shuffled control AS = H3 + SR with features permuted within month. |
| **B** | O5's L1 (BANK + G1 + G15 ridge; code on origin/nt-tactical-outlook `runs/tactical/outlook/O5_geom_bestbase/`) | Spearman of the next-60-min size | B1 = L1 + SR. |
| **C** | T11's V3 | target A (next-hour change in log RV) | C1 = V3 + SR. |

## Periods (all in UTC via `runs/tactical/common/timefix.py`; seams masked)

**Selection, 2019-01..2024-06, cut file.**
- Choose the lookback L and the bar size (1 or 15 minutes): ONE pair per part, by the mean monthly gain.
- Written to `selection.json` and committed before any judgement.

**Judgement.** These are second looks at periods H3 and the bases have already seen. That is stated; the new element is the SR block.
- **J1:** Bitstamp 2017-10..2018-12, walk-forward. For part B, 2018-07..2018-12 only, as O5 needs history.
- **J2:** Binance 2025-11..2026-09, models frozen at 2025-07-20 (part A) or as in each base's own frozen recipe.

**Prospective.** A1, B1 and C1 frozen on Binance to 2026-09-30 with a sha256; `score_B.py` scores 3 complete months from 2026-10-02.

## Pre-registered criteria, Holm over A, B, C

| Part | Pass iff, on BOTH J1 and J2 |
|---|---|
| A | A1 vs A0: Brier improvement ≥ 1% relative, with a day-block bootstrap 95% CI above 0; AND A1 beats AS |
| B | B1 − L1 Spearman ≥ +0.003 with the monthly CI above 0 |
| C | C1 − V3 skill ≥ +0.005 with the monthly CI above 0 |

## Never
- No bar 2025-07-21..2025-09-30.
- No October-2026 target.
- Nothing chosen on J1 or J2.
- The main file is not modified.

## Visual deliverable
A section for `runs/tactical/report/positive_outlook.html` (build_positive.py), "Синтетическая геометрия: линии
поддержки и сопротивления". It shows the S1, S2 and S3 lines drawing themselves on real BTC windows (animated, in the
page's style), with the predicted P(hit) and sigma next to them. It reports the measured T13 numbers.

## Outputs
`runs/tactical/tasks/T13_support_lines/`: code, `selection.json`, `results.json`, `tables.txt`, `frozen_B.json`,
`score_B.py`, `tests.log`, figures. The report text goes to the lead.
