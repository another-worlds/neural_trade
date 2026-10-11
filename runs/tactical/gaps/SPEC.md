# gaps SPEC (pre-registered 2026-10-11, committed before any result)

Three CPU studies from the "not covered" list of runs/tactical/research/2026-10-11-unknown/README.md (owner approved any target,
2026-10-11). CPU only, <= 4 processes, below-normal priority. Descriptive/exploratory rigour (TACTICAL.md); nothing here changes a default.

## Data and the hard cutoff (all three studies)
- File D:/nt/neural_trade/Bitcoin_BTCUSDT.csv, 1-minute bars, read ONLY through `shape/shapelib.load_bars` (stops reading at the first chunk
  crossing 2025-07-20 23:59 and drops later rows). **No bar after 2025-07-20 23:59 is read.** Every run script asserts
  `index.max() <= CUTOFF`; `test_gaps.py` pins it. The last month (2025-07) is therefore PARTIAL (07-01..07-20); every table is
  reported with and without it ("ex-July": months <= 2025-06).
- Month M is scored by training on data before M only (details per study). Anchor `a` sees bars `< a`.

## Study 1: structural breaks in predictability
Series (monthly, from the same protocols as before but recomputed on the cutoff data; the stored decay_*.json are NOT used because they
were computed on bars to 2025-09):
- S_auc: tb7 direction AUC, next-open labels, exactly `lab/decay_nextopen.py` (60 d logistic C 0.1, anchors every 10 bars, 15-bar
  horizon, 5-bps deadband), months 2017-04..2025-07.
- S_har: HAR Spearman, exactly `lab/decay_vol.py` column `har` (ridge on RV15/60/240/1440 + hour sin/cos, 60 d training, horizon 15),
  months 2017-04..2025-07. (`rv60` and `har_tb7` columns also stored, not tested.)
Reproduction check: months <= 2025-06 must match the stored json to 1e-6 on the AUC/Spearman (the months do not depend on later bars);
differences are reported.
Models for a series y_1..y_T (T about 100), in this order:
1. M_trend: y = a + b t (OLS). 2. M_k: piecewise-constant mean with k breaks, k = 0..5, minimum segment 6 months, globally optimal
by dynamic programming (Bai-Perron style). Parameters 2k+1 (k+1 means, k dates). BIC = T ln(RSS/T) + (#par) ln T. k* = argmin BIC over
M_0..M_5. 3. sup-Chow: break in BOTH intercept and slope at a date tau in the central 70% of the sample (15% trimming); statistic = sup of
the F over tau; p-value by moving-block bootstrap (block 6, 2000 replicates) of the residuals of the null M_trend.
Decision per series: **breaks supported** iff BIC(M_k*) + 2 < BIC(M_trend) AND sup-Chow p < 0.05; else "not supported" (trend
adequate or indistinguishable). Break-date intervals: moving-block bootstrap (block 6, 1000 replicates) of the residuals around the fitted
M_k*, k* fixed, re-estimating the dates; 2.5/97.5 percentiles of each break (matched by order). Reported even when the decision is
"not supported" (labelled).
Events. Candidate events are checked against the data, not memory. For a candidate date d: ratio R(d) = median daily notional volume
(volume x close) of the 14 days after d over the 14 days before, and the same for daily realised variance; rank of R(d) among R over all
dates in the sample (percentile). "Data-verified" = percentile >= 95 (up or down). A date taken from memory and not verified this way is
labelled "unverified (memory)". Candidates: Binance BTC zero-fee start 2022-07-21 and end 2023-03-31 (memory; additionally the
data's own largest volume jump in +-60 days is reported as the data date), FTX collapse 2022-11-08, COVID crash 2020-03-12, China mining
ban 2021-05-19, halvings 2020-05-11 and 2024-04-20, US spot-ETF launch 2024-01-11, Terra/Luna 2022-05-09. The same break search
(M_k, BIC) is also run on the monthly log notional volume and log realised variance series, giving data-dated structural changes in the
market. Coincidence = an event (or data date) within a break's 95% interval widened by 1 month. Descriptive; no multiplicity correction
(stated). Coincidence with an event is NOT causal evidence.

## Study 2: HAR with an intra-hour periodic multiplier (B8)
Protocol of volnet/SPEC.md (target y_H = log(|close[a+H-1]/close[a-1]-1| + 1e-5), H in {15, 60, 240}; monthly walk-forward; 365-day
training; rows whose label ends by M0; last 15% of the training rows (gap of 240 bars) is HOLD for the QLIKE scale; Spearman and QLIKE
exactly as volnet_lib.metrics; paired monthly diffs, 95% t-interval over months), with one change: **anchor stride 7** (primary), because
stride 10 fixes the anchor's minute-of-hour modulo 10 (H57 anchors sat at only 6 minute phases, 1/11/.../51, never at the :00/:15/:30/:45
bursts). The same run is repeated at stride 10 (the H57 stride) for comparability, descriptive.
Variants (ridge alpha 1 on standardised inputs, all horizons jointly as in volnet):
- `har` = HAR+tod (RV15/60/240/1440, hour sin/cos), the baseline.
- `m15`: har + 14 dummies for the anchor open time's minute-of-hour mod 15 (covers the 5-minute boundaries and the 0/15/30/45 bursts).
- `moh`: har + 59 dummies for the minute of the hour (0..59).
- `cal`: moh + 23 hour-of-day dummies (full calendar control: separates intra-day from intra-hour).
Windows: selection months 2021-01..2024-06 (42); ONE look 2024-07..2025-07 (partial July). Selection: for m15, moh, cal the t-statistic of the
paired monthly diff vs har for Spearman and QLIKE at each horizon; score = mean of the six; the highest is written to selection.json
and committed before the look. Look success per horizon: the selected variant's 95% t-interval lower bound > 0 on BOTH Spearman and
QLIKE (as H57). The look also reports the other variants, labelled descriptive.
"Explains H57's +0.006": H57's selected ensemble gained +0.006 Spearman at 15 and 60 min over har (look 2024-07..2025-09, stride 10).
Using H57's own stored monthly results (volnet/results_look_*.jsonl) restricted to the same months (2024-07..2025-06, those months
do not depend on later bars), the selected variant's mean Spearman diff at stride 10 is compared with the ensemble's diff on those
months. Pre-registered reading: "explains most" iff the variant's diff is >= 2/3 of the ensemble's diff at BOTH 15 and 60 minutes
(and its own CI > 0 there); "explains part" if >= 1/3 at both; else "does not explain".

## Study 3: vol-of-vol
Stride 7. Notation: RV(a,b) = sum of squared 1-minute log returns over bars a..b-1; hourly log RV L_k(a) = ln(RV(a-60(k+1), a-60k) + 1e-12).
- Target A (next-hour change in log realised volatility): dA = ln RV(a, a+60) - ln RV(a-60, a) (volatility = sqrt RV, so the change in
  log vol is dA/2; same ranking and R2).
- Target B (realised vol-of-vol, next 4 hours): vB = ln(std_ddof1(ln RV over the four hourly windows [a+60j, a+60(j+1)), j=0..3) + 0.01).
Rows where any of those windows has RV = 0 (flat bars, early file) are dropped from training and scoring (count reported).
Features (all from bars < a): S1 = RV15/60/240/1440 (log), hour sin/cos; S2 = S1 + volume/range extras of volnet (`Data.extra`);
S3 = S2 + hourly lags L_1, L_2, L_3 and the previous 4 hours' vol-of-vol vp = ln(std(L_0..L_3) + 0.01).
Baselines. A: `rw` predicts dA = 0; `ar` OLS on the single regressor L_0 - L_1 (the previous hour's change). B: `rw` predicts vp;
`ar` OLS of vB on vp. Models: ridge (alpha 1, standardised) on S1/S2/S3. Training: 365 days before month M with the label end <= M0
(label length 60 min for A, 240 for B).
Metrics per month: MSE, R2 against rw = 1 - MSE/MSE_rw, Spearman(pred, target); skill vs ar = 1 - MSE_model/MSE_ar.
Selection months 2019-01..2024-06 (66): per target, the feature set with the highest t-statistic of the paired monthly skill vs ar
is selected and written to selection.json before the look. ONE look 2024-07-01..2025-07-20 (14 months, July partial).
**Pre-registered threshold (per target):** success iff the look's mean monthly skill vs `ar` is >= 0.02 AND its 95% t-interval lower bound
is > 0 AND the mean R2 vs `rw` is > 0. Otherwise "no". Both targets are stated; no multiplicity correction.

## Outputs
runs/tactical/gaps/: SPEC.md, code (`gaps_lib.py`, `study1_breaks.py`, `study2_har_moh.py`, `study3_vov.py`, `test_gaps.py`), per-study results
(*.json/jsonl), tables (*.txt), logs, REPORT.md with the verdicts against this SPEC and run times.
