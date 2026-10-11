# tdvar SPEC (pre-registered 2026-10-11, committed before any result)

Hypothesis #5 of `runs/tactical/research/2026-10-11-unknown/README.md` (TD estimation of the variance term structure; the
expected-variance tower property as a loss), with every correction of `verification.md` (A3 + H-5). Owner 2026-10-11 approved
any target. Risk-branch work (variance, not direction); tactical rigour (D-063): one seed, no default change.

## Question
Does a model of the H-minute realised variance that is trained by bootstrapping through the tower
`E_t[RV_(t,t+H)] = E_t[RV_(t,t+s)] + E_t[E_(t+s)[RV_(t+s,t+H)]]`, `s = H/2`, beat a direct regression on the noisy
overlapping H-window, and does that gain shrink as the training window grows (a bias-for-variance mechanism)?

## Data and the hard cutoff
- `Bitcoin_BTCUSDT.csv`, 1-minute close (D-050). **No bar at or after 2025-07-21 00:00 is ever loaded** (`load_bars` drops them on
  read; tests `test_cutoff_drops_every_later_bar`, `test_real_file_respects_cutoff_if_present`). The last bar is 2025-07-20 23:59.
- The file has one gap (19 h 21 min); an anchor whose bars `a-10080 .. a+H-1` are not consecutive minutes is dropped.
- Anchor `a` (a bar position) every 60 bars. Features read bars `< a`; the target is the interval `a .. a+H-1`.

## Variables (all in variance LEVELS, never log; verification fix 1)
- `rate(a,h) = mean over bars a..a+h-1 of (1-minute log return)^2`, times 1e6 (additive: `H rate_H = s rate_s + (H-s) rate_(H-s)`;
  test `test_tower_additivity_in_levels_not_in_logs`).
- Features `x_a` (HAR, levels): `rate` over the last 60, 240, 1440, 10080 bars (1 h, 4 h, 1 d, 7 d). No time-of-day term.
- Horizons H = 240 and 1440 minutes (main), 60 minutes (secondary, no criterion; the 60-minute horizon at a 30-day window is
  expected to give nothing). `s = H/2`.

## Models (all fit by OLS in levels on the same training anchors, intercept included; forecasts floored at 2% of the median
training target, the same floor for all)
1. `direct` - direct HAR: `rate_H(a)` on `x_a`.
2. `iterated` - iterated HAR: `g_s` direct on `rate_s(a)`, a level VAR(1) `x_(a+s) | x_a`, forecast
   `(s g_s(x_a) + s g_s(E[x_(a+s)|x_a])) / H`.
3. `td` - **the headline TD (LSTD-HAR)**: bootstrap target `Z(a) = (s rate_s(a) + s g_s(x_(a+s))) / H` with the realised
   `rate_s` and the realised later state, regressed on `x_a`. **Known identity, stated before the run:** because OLS is linear in
   the target, on identical anchors the linear `td` equals the `iterated` forecast exactly (test
   `test_linear_td_equals_iterated_har`). The tested claim for the linear model is therefore "bootstrapped / iterated beats
   direct"; the two rows are reported, and any difference between them is numerical.
4. `td_recent` - `td` fitted on more recent anchors: those whose H-label is incomplete at the month start but whose s-part and
   the later state `x_(a+s)` read only bars before the month start (leak-free; test `test_td_recent_uses_only_bars_before_t0`).
   This is TD's structural edge (a shorter wait for labels). Secondary, not the headline.
5. `midas` - direct MIDAS: `rate_H(a)` on one regressor `sum_k w_k(theta) b_k`, `b_k` the hourly rate of hour k before `a`
   (k = 1..168), exponential-Almon weights, `theta` by a 20-point grid on the SSE, intercept and slope by OLS.
6. `naive` - the trailing rate over the lag nearest to H (240 or 1440 for the main horizons), no fitting.
7. Neural tower (secondary, written below, run after the linear study).

## Walk-forward and purge (verification fix 2)
- Monthly. For month M starting at `t0`: training anchors have `a >= t0 - W days` and **`a + H <= t0`** (the label of every
  training anchor ends before the month starts; the bootstrap reads `x_(a+s)` with `a+s < a+H <= t0`, so the 2T reading is inside
  the gap). `td_recent` is the only model that uses anchors with `a + H > t0`, and only through bars `< t0`.
- Test anchors: `t0 <= a < next month start`, whose label also ends by the stage end (selection: 2024-07-01; look: the cutoff).
  The test month's own labels are never seen in training. Test `test_no_training_label_bar_at_or_after_the_month_start` perturbs
  every bar from `t0` on and checks that the fit is unchanged.
- Training windows W in {30, 60, 180, 365, 1095} days. A window that reaches before the start of the file is truncated and the
  row says so (`win_days_avail`). **The gain-vs-window curve and the criteria use only the common months where every window is
  complete: 2020-02 .. 2024-06 (53 months).** Months 2019-01 .. 2020-01 are run for every window and shown separately.

## Metrics
- QLIKE in levels on the realised H-rate: mean over a month's test anchors of `x - log x - 1`, `x = rate_H / forecast`
  (`rate_H` floored at 1e-4 of the median training target). Lower is better.
- Spearman between the forecast and the realised H-rate per month (secondary).
- Per month and (W, H): `gain = (QLIKE_direct - QLIKE_model) / QLIKE_direct` (relative, in %). A cell's gain is the mean of the
  monthly gains with a 95% t-interval over months (months are mildly autocorrelated: stated, not corrected) and a 95% moving-block
  bootstrap interval (blocks of 3 months, 2000 draws). n_eff per month is `n_test * 60 / H` (about 30 at H = 1440 and 180 at H = 240
  with a 30-day month at 60-minute anchors; it is printed).

## Prediction (research README, fixed before the run; judged on the headline `td` against `direct`)
- **C1** at W = 60 days: mean gain >= 1.0% at BOTH H = 240 and H = 1440, and the one-sided p of "gain > 0" below 0.05 after
  Holm over the two horizons (selection only; the look has 13 months and is judged on the point value).
- **C2** at W = 1095 days: mean gain <= 0.3% at both horizons.
- **C3** Spearman correlation between the gain and log W over the five windows is < 0 at both horizons (a monotone fall gives -1).
- **Verdict.** Mechanism SUPPORTED iff C1, C2 and C3 all hold on the selection common months, and CONFIRMED iff the look agrees on
  C1 (point), C2 and C3. REFUTED if C3 fails at either horizon (the research README: "if the gain does not fall, the hypothesis
  is rejected"). Anything else is PARTIAL; the exact numbers are reported either way. The key evidence is the curve of gain against
  training window with its intervals, not any one cell. A negative result is a valid outcome and is recorded as such.
- The comparison of `td` with `midas` and with `iterated` is reported on the same footing (the strongest baseline per cell).

## Months, the one look
- Selection months 2019-01 .. 2024-06 (66; the criteria use the 53 common ones). These months were used by earlier tactical
  hypotheses (H56, H63, H64), so the selection is exploratory.
- **ONE look: 2024-07-01 .. 2025-07-20** (13 months, the last partial), run once after `selection.json` and its tables are
  committed. The look period was already inspected by earlier tactical studies on other targets (H53, H57; volnet), so it is not
  clean; the gain-vs-window curve on the selection months is the main evidence. The protected tail (after 2025-07-20) is never
  read. No test fold.

## Neural tower (secondary, optional)
- Inputs: the four log-rates, standardised on the training window. `log f = b + w.z + MLP(z)`, MLP 16 tanh units, last layer
  zero-initialised, `b` initialised at the log of the training mean target, L2 1e-3 on the MLP. Loss: QLIKE in levels. Full-batch
  Adam, learning rate 0.01, 200 steps, no early stopping, seed 0. CPU only (one process, below-normal priority): small network.
- `nn_direct`: trained on `rate_H` (H-window). `nn_td`: stage 1 trains `g_s` on `rate_s(a)` from `x_a`; stage 2 trains the H head
  on `Z(a) = (s rate_s(a) + s g_s(x_(a+s))) / H` with `g_s` frozen (the target network). Same anchors and purge as above.
- Months: every second common month of the selection (2020-02, 2020-04, ...), all look months; all five windows; H = 240, 1440.
  Reported as `gain_nn = (QLIKE_nn_direct - QLIKE_nn_td) / QLIKE_nn_direct` and against linear `direct`. It does not enter C1-C3.

## Outputs
`runs/tactical/tdvar/`: this SPEC, `tdvar_lib.py`, `run.py`, `analyse.py`, `test_tdvar.py` (+ `tests.log`), `results_*.jsonl`,
`selection_tables.txt`, `look_tables.txt`, `curve_sel.png`, `curve_look.png`, `REPORT.md` with run times. Resources: linear study
on CPU, up to 3 processes at below-normal priority; no GPU used by the linear study.
