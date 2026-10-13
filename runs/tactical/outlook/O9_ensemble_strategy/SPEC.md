# O9: an ensemble trading strategy built on the strengths found (O1-O8, H72/T01)

Outlook lead, 2026-10-13. Committed BEFORE any number of this study is computed. Owner (2026-10-13): "сформируй ансамблевую
трейдинг стратегию основанную на опирании на сильные стороны найденного". Tactical, exploratory; CPU only.

## Honest framing (read first)
- **No span is clean for a verdict.** O8 ran a strategy of the same family (band breakout) on A (2018-07..2020-12), B
  (2024-07..2025-07-20) and C (Binance 2025-10-05..2026-10-12). The design below is informed by those results, so A, B and C are
  **descriptive only** here.
- **The verdict is prospective.** The models are frozen on data up to 2026-09-30, and the strategy is scored ONCE on the first 3
  complete months after 2026-10-12 that are downloaded later (expected 2026-11..2027-01; the owner's forward-data permission,
  TACTICAL.md).
- **Prior expectation: low.** Minute-scale direction has decayed (H51, H52). O8's breakout edge vanished under pessimistic fills.

## The strengths and the role each plays
| strength | evidence | role in the ensemble |
|---|---|---|
| size of the next 15-60 min (L1: S3 + bank + G1 + G15) | O1, O5 (QA PASS) | band distance, stop and target distance, position size (1 / sigma) |
| next-hour volatility CHANGE (ridge S3 of H72) | H72, T01 forward PASS (skill +0.142) | regime gate: trade breakouts only when volatility is forecast to EXPAND |
| momentum after a band break in trending periods | O8 primary fills, spans A and B | the entry trigger (a break of m +- k sigma) |
| residual direction of a regression (13 scale-free features, logistic) | H27, H51 (AUC ~0.52 recently) | agreement filter: the break must agree with P(up) |
| risk measurement (intervals) | O7, T03 | stops at quantiles of r / sigma |

## Signals at each decision (every 15 minutes on the clock, the open of the decision bar; everything from bars < t)
- **sigma_60:** L1's 60-minute size forecast. Monthly walk-forward on the 365 days before each month, as O8.
- **dV:** the H72 / T01 S3-ridge forecast of ln RV(next hour) - ln RV(last hour), with the same monthly walk-forward.
  T01's code is imported read-only.
- **p_up:** a logistic regression with L2 (C = 0.1) on the 13 tb7-type features (`lab.fs_tb7` at the decision bar).
  - Label: the sign of the next-open 60-minute return (`open[t+60] / open[t] - 1`), with a 5 bps dead band in training.
  - Fit: the same monthly walk-forward.
- **Break trigger:** O8's BRK mechanics, with stop orders at m(1 +- k * sigma_60) valid for 15 bars, one position at a time.

## Ensemble rule (one family, a small grid; the order fills only if all its gates are open at the decision)
- **Long stop order placed** only if p_up >= 0.5 + delta and dV >= g. The short side is the mirror: p_up <= 0.5 - delta and dV >= g.
- **Exits:**
  - a stop at s * sigma_60 beyond the entry;
  - a target at tau * sigma_60 beyond the entry (tau = none or 2);
  - a time stop at 60 bars.
- **Size:** w = min(2, sigma_ref / sigma_60), where sigma_ref is the median sigma_60 of the previous month (a vol-targeted size,
  capped). The return of a trade is w x (log return in its direction - cost).
- **Grid (24):** k in {1.0, 1.5}, delta in {0, 0.02}, g in {-inf (no gate), 0}, s = 1, tau in {none, 2}, sizing in {1x, vol-targeted}
  minus duplicates.
- **Meta-labelling variant (one more candidate, "META"):**
  - candidate trades: every BRK k = 1.5 trigger;
  - a logistic meta-model on [p_up aligned to the side, dV, ln sigma_60, the geometry top-15 features of O1, hour sin/cos] predicts
    P(net trade > 0);
  - the trade is taken if P > the 60th percentile of the previous month's meta-scores;
  - fit: monthly walk-forward on past trades only (labels end before the month).

## Fills and costs (lessons of O8: pessimistic is primary)
- **Primary:** O8's PESS fills (the stop is checked in the entry bar too; target and stop in one bar resolve to the stop).
- **Also reported:** O8's original fills.
- **Costs:** 2 bps round trip is primary; 0 and 5 bps are reported.

## Selection
- **Data:** 2021-01 .. 2024-06 (42 months), PESS, 2 bps.
- **Rule:** the configuration (of the 24 + META) with the highest t-statistic of the monthly net return, among those with at least
  20 trades per month on average.
- `choice.json` is committed alone before anything else is computed. The selection table shows each component's marginal value
  (gates on/off) descriptively.

## Descriptive runs (already-seen spans; no verdict)
- the choice on A, B and C (C with the frozen fit to 2025-07-20, as O8);
- arms for the choice:
  - FULL (the choice);
  - NO_REG (delta = 0, the regression gate off);
  - NO_DV (g = -inf);
  - NO_GEOM (sigma from R0, S3 only);
  - SHUF (sigma, dV and p_up each permuted within the month);
  - O8's BRK_k1.5_H60_s1.0 as the plain baseline.

## Prospective verdict (frozen now; scored once later)
- **Freeze:** every model (L1, R0, the dV ridge, the logistic regression, META if chosen) is fitted once on the 365 days ending
  2026-09-30 23:59 of the Binance file, with features from that file only. The artefact goes to `frozen_P.json` with its sha256.
  The month-to-month quantities (sigma_ref, META's threshold) roll forward from the previous scored month, starting with 2026-10.
- **Scoring:** `score_P.py` scores the first 3 complete calendar months after 2026-10 that the forward file contains, once. It
  refuses fewer months, and refuses a second scoring once its results file exists.
- **Criterion** (daily net returns, PESS, 2 bps; the 95% interval from a 5-day block bootstrap, 10,000 draws, seed 0):
  1. the mean daily net return of FULL is > 0 with the lower bound > 0;
  2. FULL - SHUF, daily and paired, has its lower bound > 0;
  3. FULL - O8 baseline, daily and paired, has its lower bound > 0 (the ensemble adds over the plain breakout).
- **Verdicts:** "ensemble works prospectively" iff 1 and 2 hold; "ensemble adds over the breakout" iff 3 holds.
- **Also reported:** the owner's target (hit > 60 %, drawdown < 5 %).

## Tests (seconds)
- every signal at t uses bars < t; p_up's training labels end before the month; the freeze reads no bar after 2026-09-30 23:59
  and no bar of 2025-07-21..2025-09-30 from the cut file;
- the gates work on hand-made inputs, and the vol-targeted size is capped at 2;
- META is trained on trades whose exits precede its month;
- `choice.json` is committed before `describe` runs (the gate as in O8); `score_P.py` refuses short data and a second run.

## Outputs
`runs/tactical/outlook/O9_ensemble_strategy/`: code, `selection.jsonl`, `choice.json`, `describe.jsonl`, `trades_*.csv.gz`, `tables.txt`,
`frozen_P.json` (+ sha256), `score_P.py`, figures, `run.log`, `tests.log`, `REPORT.md`, and an O9 section of the outlook report.
