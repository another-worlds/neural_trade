# T14 M8: distil M7 into a compact module and freeze it (SPEC, committed BEFORE anything runs)

Tactical sprint T14, candidate M8, 2026-10-17, CPU only. Parent spec: `../SPEC.md` (unchanged). Builds on M7_gbm (passed both targets on both
looks) and its post-hoc M7c_gbm_resid. Harness: `../core` (unchanged). Branch `nt-tactical-t14-m8`.

## Setup that every part shares
- Rows: the M7 row set `ok` (harness ok anchors with ALL 12 M6 keys finite and finite labels); the same rows in every view below, so increments
  are comparable (11,872 rows on Look 1, 8,346 on Look 2 as in M7_gbm). Pool, GBM parameters and seed: M7's `m7_lib.monthly_view` /
  `gbm_params` untouched (num_leaves 15, 200 rounds, lr 0.05, min_data_in_leaf 200, bagging 0.8, 365-day pool, labels ended before the month).
- Evaluation: `evaluate.py --view <name>` per view (Look 1 2024-07..2025-07, Look 2 2025-11..2026-09), n_perm 5, n_boot 10000, no C2b.
- Never read: 2025-07-21..2025-09-30, anything after 2026-09-30 23:59, the original Bitcoin_BTCUSDT.csv, any October-2026 target. Data only
  through the harness caches. Nothing is chosen on a look; the ablation and the linear test are descriptive (no Holm, no new pass/fail).

## (a) Ablation, view names `M8_abl_<group>`
M7_gbm refitted with the listed key columns removed (same four output columns `gbm_dA_hat, gbm_hit_hat, gbm_dA_rank, gbm_hit_rank`):

| group | removed keys |
|---|---|
| stretch | str20, str60 |
| volpct | vp_h, vp_d |
| hourwk | hsin, hcos, wkend |
| sr | sr_near, sr_asym |
| lrv | lrv0, lrv1, lrv2 |

A sixth refit with nothing removed must reproduce the stored M7_gbm view (sanity check, tolerance 1e-9; recorded in `checks_m8.json`).
Readout per group and target (A: monthly skill increment over V3; B: monthly relative Brier improvement over H3):
drop = full model minus ablated model, as the monthly series paired by month; mean drop with a 95% t-interval over the 24 pooled months
(and per look). **A group "carries the gain" on a target if its pooled drop is >= 25% of the full model's pooled mean increment AND the
pooled t-interval of the drop is above 0.** The main carrier is the group with the largest pooled drop. Each view's own harness numbers
(pooled increment with CI, B relative improvement with day-block CI) are tabulated beside it.

## (b) Cheap linear recovery, view `M8_lin`
Eight columns computed row-wise from the keys (so causal by construction), appended to V3 (A) and to H3 (B) by the harness, on the M7 row set:
`|str20|, |str60|, str20^2, str60^2` and the interactions `(vp_h-0.5)*hsin, (vp_h-0.5)*hcos, (vp_d-0.5)*hsin, (vp_d-0.5)*hcos`.
No other term is tried. Recovery ratio R_T = pooled mean monthly gain of M8_lin / pooled mean monthly gain of M7_gbm, for T in {A, B} (the 24
months). **Verdict: "recovers" if min(R_A, R_B) >= 0.75; "partly" if 0.40 <= min < 0.75; "no" if < 0.40.** Also reported: M8_lin's own harness
flags and the paired difference M7_gbm minus M8_lin (monthly t-interval). If it recovers, the module is an interpretable V4.

## (c) Freeze for the prospective period
- Module choice rule, fixed now: of M7_gbm and M7c_gbm_resid, the one with the larger mean over the two looks of the stored target-A
  increment (`results.json`, `looks.*.A.increment.view.mean`); ties go to M7_gbm. (Computed by `freeze_m8.py` from the stored files; the
  numbers on 2026-10-17 are 0.0273/0.0124 for M7_gbm and 0.0298/0.0217 for M7c.)
- Final models: the same GBM parameters, fitted once on the pool of label-known anchors to 2026-09-30 by M7's pool rule at m0 = 2026-10-01
  (ta >= m0 - 365 days, ta + 60 min <= m0, ok, finite keys and finite labels; cut and Binance anchors both enter if they satisfy it). For M7c the
  labels are the walk-forward residuals of V3 / H3 (`build_views.residuals`). Models are saved as LightGBM text files with sha256 in
  `frozen_P.json` (with the pool's row count and hash, parameters, library versions). A frozen kernel control (Nadaraya-Watson on the same keys
  and labels, bandwidth by the leave-one-month-out CV of M7) is frozen the same way, as the prospective C2b.
- `score_P.py` scores the months from 2026-10 once >= 3 complete months exist: a month is complete when the new anchor table reaches
  6 hours before the month's end. The history for the monthly refit of V3 / H3 and of the ridge / logistic is the forward base and the
  stored walk-forward view to 2026-09; new months use the FROZEN models (no GBM refit). V3 / H3 and the ridge / logistic are refitted monthly
  as in the harness (`evaluate_look`, unchanged, called from score_P). Looks: the first 3 complete months, then the first 6; at both the
  Pocock level alpha = 0.0294 (one-sided). **Criterion: A increment >= +0.005 with the one-sided monthly t p-value < 0.0294; B relative Brier
  improvement over H3 >= 1% with the one-sided day-block bootstrap p-value < 0.0294; and both above C1 (permuted view) and C2 (the harness
  C2a, and the frozen kernel C2b).** Pass = either target. With < 3 complete months the script writes a "withheld" result and computes nothing.
- score_P is tested on synthetic data and on a replay of already-used 2026 months (the cut-off date moved earlier); it is never run on
  real post-2026-09 data, and no October-2026 target is read.

## (d) Figure `M8_module/fig_m8.png` + `fig_m8.json` (plotly figure JSON written by hand, theme.py colours)
- Bars: the A increment (per look) and the B relative Brier improvement (per look) of the full model and of every ablated model, with the
  harness CIs, M8_lin beside them.
- Real-day strip: Binance 2026-03 (Look 2 month). Among the days of that month with >= 20 valid module rows, rank by the day's mean
  `A_LV0` (previous-hour log RV, i.e. the day's activity level) ascending and take the element at index (n-1)//2 (the lower median day). The
  strip shows, hour by hour, V3's forecast and V3 + module's forecast of the next-hour change in log RV (March refit, the frozen module's walk-forward
  view) against the realised change. The rule uses nothing about forecast quality.

## Tests (seconds)
ablation excludes exactly the listed columns; linear terms are row-wise (causal: changing other rows or later rows leaves a row's terms
unchanged); the frozen artefact reloads to identical predictions and sha256; score_P withholds with 2 complete months, passes the right
looks to the criterion, and the Pocock level is 0.0294.

## Budget
CPU only, below-normal priority, 4 threads. Estimate: 6 ablation refits ~10 min, evaluations ~1 min each, freeze ~1 min.
