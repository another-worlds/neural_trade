# SPEC: Omori aftershocks of excursions beyond HAR (hypothesis #2)

Pre-registered before any result (committed alone with the code and tests, first). Source:
`runs/tactical/research/2026-10-11-unknown/README.md` section 1 #2 and `verification.md` (A5 + H-2; corrections applied, marked "corr.").
Tactical, exploratory (D-062/063; the owner approved any target on 2026-10-11); CPU only, up to 4 processes, below-normal priority.
Branch `nt-tactical-omori`. Findings are not defaults.

## 0. Data and the hard cutoff

- Only `Bitcoin_BTCUSDT.csv` (1m OHLCV, D-050) through `shape/shapelib.load_bars`: **last bar read 2025-07-20 23:59**, rows after it are
  dropped chunk-wise and never reach an array (tests: `test_omori.py::test_cutoff_constant_and_loader_drops_later_rows`,
  `::test_no_scored_month_or_window_can_pass_the_cutoff`).
- Anchor i = decision at the close of bar i-1; stride 5; features read bars <= i-1; target window = bars i..i+T-1.
- Walk-forward: monthly, 365-day training window; a training row's window must close by the month start (purge, tested). The HAR
  coefficients, the shock stream, the kernel choice and all coefficients are fitted on the month's training rows only (test:
  features and fits are unchanged when bars inside the scored month are edited).

## 1. Event, horizons, k

- T in {15, 30, 60} minutes (primary T = 30; README names T = 30).
- sigma_hat_T(i) = HAR + time of day: OLS of log mean squared 1-bar return over bars i..i+T-1 on log mean squared return over the
  last 15/60/240/1440 bars and the first two daily harmonics at the window middle (volbench's HAR+tod; refitted per month).
- **Event** E_i = 1 if within bars i..i+T-1 the high or the low reaches C_{i-1} (1 +- k sigma_hat_T(i) sqrt(T)) in log terms, i.e.
  max(max ln H - ln C_{i-1}, ln C_{i-1} - min ln L) > k sigma_hat sqrt(T).
- **k fixed for a 2.0% event frequency** on the training year of the first scored month (2018-01..2018-12; no test, selection or
  look month, no skill number; `calibrate_k.py`, `results/k.json`): k = 3.882 (T=15), 3.845 (T=30), 3.867 (T=60), achieved rates
  0.0200 / 0.0200 / 0.0200. Fixed for every month. Expected events per month at stride 5 and T=30: about 170 anchor rows, roughly 29
  independent windows (the README's number).

## 2. Shock stream (past events)

Bar j is a shock bar when the window that ended at j (anchor a = j-T+1, C_{j-T} to C_j) left the HAR band: z_j = |ln C_j - ln C_{j-T}| /
(sigma_hat_T(a) sqrt(T)) > k (sigma_hat read bars <= j-T). Magnitude m_j = min(z_j/k, 3) (0 otherwise). Foreshock indicator: z_j > k/2.
Both are known at the close of bar j. (corr.: events are exceedances beyond the HAR forecast, not raw returns.)

## 3. Models (logistic regression, L2 C = 1, standardised features clipped at +-8)

- **B, the strengthened baseline (corr.):** log mean squared return at 1/5/15/60/240/1440 bars; Fourier clock of the anchor (time-of-day
  harmonics 1-3, minute-in-hour harmonics 60/30/20/15, day-of-week 1-2); log sigma_hat_T and log sigma_hat x the time-of-day harmonics
  1-3 (sin, cos) = "log sigma x clock"; the plain "time since last excursion" feature (log1p(bars since the last shock bar, cap 1440)
  and its square). Without that feature the kernel beats a strawman (verification A5).
- **C_pow (the challenger):** B + K_p,c(i) + foreshock count, K_p,c(i) = sum over shock bars j < i of m_j (i - j + c)^-p, lag at most 4,320
  bars. (p, c) is chosen per month and horizon on the training rows only, by the profile log-likelihood of one added standardised
  feature on top of B's logit (grid p in 0.4..1.6 step 0.1, c in {1, 10}; positive coefficient required), then the full model is refit.
- **C_exp (sum of exponentials, classic Hawkes):** B + the four sums of m_j exp(-(i-j)/tau), tau = 5/30/180/1440 bars + foreshock count.
- Ablations: C_pow without the foreshock; B + foreshock only; constant (training rate). Reported, not gating.
- Foreshock count = number of k/2 exceedance bars among the last 30 bars.

## 4. Metrics, thresholds, verdict (fixed here; "A beats B" is the challenger C_pow against B)

- Per month and horizon: test-row log-loss and Brier score of every arm. Skill_m = 1 - LL(C_pow)/LL(B) (Brier likewise). The estimate is
  the mean over months; the interval is the t interval over months (df = months - 1), D-012 (consecutive anchors share windows;
  n_eff = rows x 5 / T is reported per month).
- **Selection months 2019-01..2024-06 (66).** Family: 3 horizons, one-sided test of mean skill > 0 on the monthly log-loss skill, **Holm at
  0.05**. The Brier skill and C_exp against C_pow are reported, descriptive.
- **Pre-registered prediction (falsifiable):** at T = 30, mean log-loss skill >= +1.0% with the two-sided 95% lower bound > 0 and the Holm
  p < 0.05 (Brier skill >= +1.5% reported alongside), AND the Omori exponent p (per-month choices) has a mean in [0.6, 1.3] and its yearly
  means (2019, ..., 2023, 2024H1) differ by less than 0.2 between any two years. Threshold is below the designers' 2% because the
  baseline is stronger. Anything else = not supported.
- **Confirmation:** exactly one look, 2024-07-01..2025-07-20 (13 months, the last to 07-20), only if the selection prediction holds
  at T = 30; confirmed when mean skill >= +0.5% with the lower 95% bound > 0 (one-sided p < 0.05). This window was already read for
  RV and direction (H56, H57, H61, H63, H64 to 2025-09 before the cutoff rule) but no HAR-normalised exceedance was scored there
  (verification A5: partly contaminated). Nothing selected = no look; the verdict is the selection result.
- Secondary (descriptive): C_exp vs C_pow in log-loss; the mean coefficient of K and of the foreshock.

## 5. Control by simulation (before the real data)

Simulated 1m paths (36 months from a fixed seed, real minute-of-day volatility profile from the 2019-01..2024-06 bars is NOT used here: a
flat profile; sigma 1e-3 per bar): (A) null, a persistent GARCH-like clock (alpha 0.04, beta 0.95) and nothing else; (B) the same plus
Omori feedback: sigma_t = sigma_clock (1 + 6 * sum_j 1{|eps_j| > 3.5} (t-j+1)^-0.9) (truncated at 4,320 bars). Scored months
13..36 (train 365 days = months 1..12 are simulated for that). Pass: (A) mean skill at T = 30 below +0.3%; (B) mean skill >= +1.0% and
the mean chosen p within 0.9 +- 0.2. If B fails, the challenger lacks power and a real null is "uninformative", reported as such.

## 6. Budget

CPU only, 4 processes, below normal priority, resumable per month (`results/*.jsonl`). Times are reported.

## 7. Amendment 1 (before any real-data result; the control only)

The first draft of the section 5 simulator triggered feedback on single bars with |eps| > 3.5, which the T-window shock stream of section 2 cannot
see, and produced no measurable effect (trial runs on 2 simulated months, 14-month paths, nothing from the real file). The control is
repaired so that it matches the stream: feedback sigma_t = sigma_clock (1 + 4 sum over trigger bars j (t-j+1)^-0.9), a trigger being the
onset (60-bar refractory) of a T = 30 endpoint excursion |ln C_t - ln C_{t-30}| > 2.0 sigma_clock(t-30) sqrt(30). The Omori exponent is
still 0.9. k is recalibrated to a 2% rate on each simulated path's first year (`omorilib.calibrate_k`). Pass criteria as in section 5.
Settings (4, 2.0) were chosen among (1.5/2.5), (4/2.5), (8/2.5), (4/2.0) on two simulated months; the real data, the selection and the look
were not touched.
