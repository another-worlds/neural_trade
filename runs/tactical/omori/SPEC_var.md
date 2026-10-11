# SPEC_var: intraday VaR/ES with the FZ0 loss and the shock state of #2 (hypothesis #6)

Pre-registered before any #6 result (committed first, after the #2 selection result that fixes the state kernel, before any #6 run). Source:
`runs/tactical/research/2026-10-11-unknown/README.md` section 1 #6 and `verification.md` A7 (corrections applied, marked "corr."). Tactical,
exploratory; risk-branch work (D-063), taken because the owner approved any target on 2026-10-11. CPU only, up to 4 processes. Branch
`nt-tactical-omori`. Findings are not defaults. Shares the data, the cutoff (2025-07-20 23:59, tested), the walk-forward, the HAR sigma_hat and
the shock stream with `SPEC.md`.

## 1. Targets and tails

- Horizons T in {15, 60} minutes. Return R_i = ln C_{i+T-1} - ln C_{i-1} for the window of bars i..i+T-1 (anchor i, stride 5, features read
  bars <= i-1). z_i = R_i / (sigma_hat_T(i) sqrt(T)), sigma_hat the HAR + time-of-day forecast of `SPEC.md` section 1 (refitted per month on
  the training rows).
- Left tail: y = z; right tail: y = -z. VaR_a and ES_a are the lower-tail quantile and tail mean of y at level a, both negative, v = VaR,
  e = ES, e < v < 0. Levels a = 2.5% and 5% (**primary**), 1% (secondary: about 570 exceedances, probably underpowered; verification A7).

## 2. Forecasts

- **FHS (baseline, "HAR-FHS"):** v = empirical a-quantile and e = mean below it of y over the month's training rows (standardised by the same
  sigma_hat), i.e. VaR = sigma_hat sqrt(T) q, ES = sigma_hat sqrt(T) es.
- **FZ0-const:** the FHS parametrisation with a constant (v, e) fitted by minimising the training-row FZ0 (isolates "FZ0-fit constants" from the features).
- **Challenger C (the tail-state multiplier):** v = -exp(a0 + a.x), e = v - exp(b0 + b.x), (a0, a, b0, b) fitted by minimising the training-row
  mean FZ0 (L-BFGS, analytic gradient, ridge 1e-3 on a and b, started at FHS), x standardised on the training rows. Features x (8):
  the log ratio of the 60-bar realised semivariances ln RS-/RS+ for the left tail (the other way round for the right tail, so it is always
  "adverse over favourable"); the jump age log1p(bars since the last shock bar of `SPEC.md` section 2); the **Omori intensity of #2**
  log1p-free: K_{p*,c*}(i) (`results/state.json`: p* = the mean per-month exponent of the #2 selection at T = 30, rounded to 0.1, and the modal c);
  the foreshock count; the first two daily harmonics (sin, cos). FZ0 (Patton-Ziegel-Chen 2019): L(y, v, e) = -1{y<=v}(v-y)/(a e) + v/e + log(-e) - 1.
  (corr.: the multiplier is the only difference between C and FHS apart from fitted constants; the FZ0-const arm shows which part is which.)

## 3. Metrics and tests

- Per month: mean FZ0 in standardised units (L(z/c,...) = L(z,...) + log c is the same additive constant for every arm, so differences are
  unit-free), the hit rate, the Acerbi-Szekely Z2 term mean(y 1{y<=v}/(a e)) - 1 for every arm, and the 2x2 transition counts of the hit
  sequence on the non-overlapping windows (every T/5-th anchor), summed over months for the Christoffersen tests.
- **Gain** = 1 - L(C)/L(FHS), averaged over months; **Diebold-Mariano-type test:** one-sided t test of the monthly mean loss difference
  (df = months - 1; consecutive anchors share windows, so the month is the independent unit, D-012). **Bonferroni over the 8 primary
  cells** (a in {2.5%, 5%} x T in {15, 60} x tail in {left, right}); secondary cells (a = 1%) are reported without a verdict.
- Coverage: Kupiec unconditional coverage, Christoffersen independence and conditional coverage (cc) on the non-overlapping subsample;
  Z2 tested by a t test of the monthly means against 0 (two-sided). Reported for FHS, FZ0-const and C.

## 4. Power simulation (before the real run; `power_var.py`, `results/power_var.json`)

World: 1-minute returns r_t = s_t u_t with u_t iid Student-t(5) of unit variance, s_t = exp(gamma x_t), x_t a unit-variance AR(1) with a
240-bar time constant (a latent tail state); sigma_hat is perfect, so FHS is the unconditional quantile. The challenger sees x through a
proxy x + N(0, 0.7^2) plus 7 pure-noise features (its estimation cost is included). 12 training months, 66 scored months (the selection
length), stride 5, gamma in {0, 0.10, 0.15, 0.20, 0.30}, 20 replications each, lower tail, a in {2.5%, 5%, 1%}, T in {15, 60}. Reported: the
FZ0 gain, the rejection rate at the Bonferroni-8 level (size at gamma = 0), and the rejection rate requiring also gain >= 1%.
Reading rule fixed here: a primary cell is **adequately powered for the 1% claim** when the simulated rejection rate at the gamma whose mean
gain is nearest to 1% is at least 0.5; a real-data null in a cell that is not adequately powered is reported as "uninformative at 1%".
Size must be at most 0.01 at gamma = 0 (Bonferroni-8 level 0.00625); otherwise the test is reported as liberal and the claim threshold is
raised to the simulated 95th percentile of the gamma = 0 gain (stated before the real run).

## 5. Verdict (selection 2019-01..2024-06, 66 months; one look)

- **Supported on selection:** for a primary cell, gain >= +1.0% with Bonferroni-8 p < 0.05 for C against FHS, and C's conditional coverage
  (cc, p > 0.05) and Z2 (p > 0.05) not rejected. The cell list is written to `results/var_selected.json`.
- **Confirmation:** exactly one look 2024-07-01..2025-07-20, only if at least one primary cell was selected, run for the selected cells only;
  confirmed when gain >= +0.5% and the one-sided Bonferroni p over the selected cells < 0.05. The window was read for RV and direction
  (H56, H57, H61, H63, H64 to 2025-09, before the cutoff rule); VaR/ES was never scored there (verification A7: contaminated).
- Nothing selected = no look; the verdict is the selection result. The answer to "does the shock state add anything" is the C vs FZ0-const
  contrast (descriptive) and the share of the gain that FZ0-const already explains.

## 6. Budget

CPU only, 4 processes, below normal priority; the power simulation about 10 min, selection about 15-25 min, the look about 5 min. Times are reported.

## 7. Power simulation result (recorded before the real run; `results/power_var_table.txt`, 247 s)

Size at gamma = 0: rejection 0.00 in every cell (Bonferroni-8 level), mean gain -0.03% to -0.09% (the estimation cost of the 8-feature fit),
so no liberal-test adjustment. gamma = 0.10 gives a mean FZ0 gain of 1.2-1.6% in every cell (sd across replications 0.07-0.19%) and a
rejection rate of 1.00 (0.90 with the gain >= 1% requirement in the cell T = 60, a = 5%): every primary cell is adequately powered for the
1% claim under the section 4 reading rule. Caveat, stated now: this world is favourable (a persistent latent state seen through a proxy, a
perfect sigma_hat, iid t5 innovations, no regime change); in the real data a 1% gain is a stronger statement than it is here, and the simulation
says only that the estimator and the test do not lose a 1% effect of that kind to noise or to the 8-feature fit cost.
