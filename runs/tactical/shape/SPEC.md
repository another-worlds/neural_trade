# SPEC: path shape in its own clock (hypothesis #1) and rotation vs shrinkage of the edge (hypothesis #4)

Pre-registered before any result (committed alone, first). Source: `runs/tactical/research/2026-10-11-unknown/README.md`
sections 1, #1, #4 and `verification.md` (every correction there is applied and marked "corr."). Tactical, exploratory
(D-062/063); CPU only. Branch `nt-tactical-shape`. Findings are not defaults.

## 0. Data and the hard cutoff

- Only `Bitcoin_BTCUSDT.csv` (1m OHLCV, D-050). **Last bar read: 2025-07-20 23:59 (inclusive).** The loader reads the file in
  chunks and stops at the first chunk that crosses the cutoff, dropping the rows past it; no bar after 2025-07-20 is held in
  memory or reaches any computation. Test: `test_shape.py` (loader on a CSV with later rows; every target window and every
  anchor inside the cutoff; features and targets unchanged when a copy of the data is edited after the cutoff).
- Anchor: stride 10 bars (the project's 1m convention), anchor i = decision at the close of bar i-1, window A[i-60:i] for
  window features, per-bar trailing statistics read at bar i-1. Anchors from 2017-01-02 (warm-up 1,440 bars) to the last
  anchor whose target window closes by the cutoff. Entry for direction labels is the NEXT OPEN (bar i), label
  r = C[i+14] / O[i] - 1, dead-band 5e-4 (as the H61/H56 next-open variants).
- Walk-forward: monthly, 365-day training window, purge: a training row's target window must CLOSE before the month start
  (`recent/wf.py split_masks` semantics, label end = anchor + T bars). Nothing is fitted on the scored month or later.

## 1. Hypothesis #1: targets (on the future window of T bars i..i+T-1, 1m log returns r_j = ln C_j - ln C_{j-1})

- rho_T = |ln C_{i+T-1} - ln C_{i-1}| / (ln max(C_{i-1}, max H) - ln min(C_{i-1}, min L)) over the window (corr.: C_t = C_{i-1}
  is inside the range, so rho is in [0,1]; logs, so exactly invariant under price reflection). Rows with a zero range are dropped.
- VR_T = (sum_j r_j)^2 / sum_j r_j^2.
- Z2_T = sum over pairs (a<b, b-a >= 2) of r_a r_b / sum r_j^2 (no lag 1: bid-ask bounce). Z3_T = the same with b-a >= 3. (corr.)
- T in {30, 60, 120} minutes. All 12 targets are unchanged when the path is reflected (sign-invariant): tested.
- Rows with sum r^2 = 0 are dropped for VR, Z2, Z3. Each (target, T) is trained and scored on its own valid rows.

## 2. Features

Baseline B (strengthened, corr.): log RV of 1m returns over the last 15/60/240/1440 bars (HAR; log sigma_HAR); Fourier clock of the
anchor's open time (time-of-day harmonics 1-3, minute-in-hour harmonics for periods 60/30/20/15 min, day-of-week harmonics 1-2,
sin and cos each); bounce proxy (mean |O_j - C_{j-1}| over the last 60 bars / sigma_60); jump state (log1p of bars since the last
bar with |r| >= 4 sigma, sigma = sqrt(RV_1440/1440) of the previous bar, capped at 1,440; bipower/RV over 60 bars); LeBaron and
Campbell-Grossman-Wang terms (sigma_hat, volatility surprise ln RV15 - ln RV240 and ln RV60 - ln RV1440, volume surprise
ln mean V60 - ln mean V1440, and its product with |past 60-bar return|/sigma_60). No tick term (constant at USD 0.01). (corr.)

Challenger C = B plus: past rho over the previous 30/60/240 bars (same definition, backward); VR_q (q = 5/15/60) over the previous
240 bars (overlapping q-sums); autocorrelations at lags 1-5 over the previous 240 bars; |logit| of the tb7 direction model
(`lab.fs_tb7`, logistic C=0.1, next-open labels, fitted on the month's training rows only; training rows get 2-fold contiguous
cross-fit values with a 40-anchor gap, test rows the full fit); RS+/RS- (ln of the positive and negative realised semivariance
over 60 bars minus ln RV60). The volume surprise is already in B.

## 3. Models and metrics

- Models: ridge (alpha 100, standardised features clipped at +-8) and HistGradientBoostingRegressor (100 iterations, lr 0.08, depth 4,
  min_samples_leaf 200, l2 1.0, seed 0). Targets VR, Z2, Z3 are rank-Gaussianised on the training rows; rho is fitted raw.
- Metric: Spearman correlation between prediction and the raw target over the scored month's rows, per month and (target, T, model,
  feature set). Delta = Spearman(C) - Spearman(B), same model class, same rows, same months.
- For rho also a calibrated score: MSE skill = 1 - MSE(C)/MSE(B) of the raw prediction (reported, not gating).
- Noise (D-012): n_eff = N / T per month; inference is over months: paired monthly differences, t interval (df = n_months - 1).
  Expected SE of the mean Delta is about 0.0046 over 66 months at T = 60 (0.037 per month).

## 4. Pre-registered verdicts for #1

- Selection months 2019-01 .. 2024-06 (66). These months were used for direction by H56, H63, H64; for the new shape targets they
  are new, but the result is still labelled selection, not confirmation.
- Family: 4 targets x 3 horizons x 2 models = 24 one-sided tests (Delta > 0), Holm at 0.05.
- A (target, model) cell is **selected** when at T = 60: mean Delta >= +0.02, the two-sided 95% lower bound > 0 and its Holm-adjusted
  p < 0.05. Cells at T = 30 and 120 are reported and informative only.
- **One confirmation look** 2024-07-01 .. 2025-07-20, run once, only for the selected cells (and their T = 30/120 siblings,
  descriptive), after the selection table is written. Confirmed when mean Delta >= +0.01 and the one-sided Bonferroni p over the
  selected cells < 0.05. Nothing selected = no look is run; the verdict is the selection result.
- The look window was already read for direction (H56, H57, H61, H63, H64 read it to 2025-09, before the cutoff rule) and for path
  functionals: excursion asymmetry (H54), barrier outcomes (H48, H60, H61). No rho, VR or Z target was ever scored there.
- Secondary prediction: decay ratio = mean Delta(rho, T=60, chosen model) in 2023-01..2024-06 / the same in 2019-01..2020-12, against
  the same ratio for the tb7 direction AUC excess over 0.5 (monthly next-open 365-day logistic). Predicted: rho ratio > 0.32
  (decays more slowly than the direction edge). Bootstrap CI over months. If rho is not selected, the ratio is reported but not interpreted.
- If nothing is selected: the verdict is "in its own clock an hour of BTC is Brownian to the strengthened baseline" (a clean negative).

## 5. Sanity check by simulation (before the real data)

- Simulator: latent log price with 1m volatility sigma_t = base x S(minute of day) x garch-like clock (GARCH(1,1) alpha 0.05,
  beta 0.94 on the deseasonalised return, normal innovations), S the empirical minute-of-day profile of mean |r| from the real
  2019-01 .. 2024-06 bars (it includes the minute-0 and 15/30/45 bursts); symmetric jumps (prob 1/2000, size N(0,(6 sigma)^2)); OHLC
  from 4 sub-steps per minute; bid-ask bounce: the observed O and C get an independent +-0.3 sigma_ref noise; volume lognormal tied to
  the clock. 3.5 simulated years from a fixed seed, stride 10; 30 scored months after 365 training days.
- **Null (A)**: pure clock, no path dependence on the sign. Expected: no predictability beyond the clock. Pass: all 24 cells have
  |mean Delta| < 0.01 and no Holm-significant positive Delta.
- **Positive control (B)**: the same plus a lag-2 autoregression phi in {0, +0.12, -0.12} switched by a Markov regime (mean duration
  one day). Pass: at T = 60 at least one of rho, VR, Z2, Z3 has mean Delta >= +0.02 with Holm significance. If B fails, the
  challenger lacks power and a real-data null is reported as uninformative at that effect size.
- Real-data results are interpreted only after A passes (if A fails, the failing cell is reported and the baseline is repaired by
  a documented amendment BEFORE the real selection).

## 6. Hypothesis #4: rotation vs shrinkage (descriptive/exploratory; next-open labels)

- Data: the same anchors; tb7 (13 features), standardised with one fixed scaler fitted on 2017-2018 (descriptive). Label y = next-open
  15-minute return > 0, dead-band 5e-4.
- Half-year fits 2017H1 .. 2024H1 (15 points), logistic regression C = 1. Reported: ||beta||, cosine/angle between consecutive
  halves and between each half and the reference beta_2019 (fit on calendar 2019), the full 15x15 cosine matrix.
- Shrinkage prediction: ||beta||(2024H1) / ||beta||(2019) <= 0.5 and cos(beta_2019, beta_2024H1) >= 0.8. (corr.: when the norm
  falls to its standard error the estimated angle grows even under pure shrinkage.)
- Scaling-model likelihood-ratio test (corr.): for each half h, reference beta_ref(h) = fit on all halves except h and its two neighbours.
  On the first 2/3 of the half (chronological, 40-anchor gap) fit the scaled model beta = c beta_ref (c and an intercept) and a free
  beta (+ intercept); score both on the last 1/3. Lambda = sum over halves of held-out log-likelihood(free) - (scaled). Null of
  pure shrinkage: parametric bootstrap, y ~ Bernoulli(sigmoid(a + c_h beta_ref . x)) on the real X, same procedure, B = 200;
  p = share of Lambda* >= Lambda. The same bootstrap gives the null distribution of the consecutive-halves angle. Rotation beyond
  shrinkage is supported when p < 0.05.
- Adaptation: a Kalman/RLS logistic (random-walk coefficients, daily Laplace update with labels closed before the day; process variance
  q per day chosen from {1e-8,1e-7,1e-6,1e-5} on AUC in 2019-01..2020-12; P0 = 0.01 I; initial beta from 2017-2018) and the
  Nagel recency-weighted momentum baseline (signed exponentially weighted past 1m returns, half-life in {5,15,30,60} min and the sign
  chosen on 2019-2020, no fit afterwards), scored by monthly AUC (dead-band mask) in 2021-01 .. 2024-06 (42 months, exploratory:
  those months were used by H56/H63/H64) against the static baseline (tb7 logistic refit monthly, 365 days).
- Verdict: **shrinkage** supported when the norm/cosine prediction holds, the Kalman gain over static is <= +0.002 AUC (paired monthly
  mean, t interval) and the LR test p >= 0.05. **Rotation** supported when the adaptation gain is >= +0.005 AUC (lower 95% bound > 0) or the
  LR test p < 0.05 with a falling cosine. Anything else = inconclusive. The Nagel baseline is a guard: an adaptive gain counts only if
  it also beats Nagel's AUC.

## 7. Budget, processes

CPU only, up to 6 processes, below-normal priority (psutil), resumable per month. Expected: cache build about 10 min; MC about 15 min;
selection about 30 min; look about 5 min; #4 about 10 min. Times are reported.
