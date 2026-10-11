# tdconfirm SPEC (pre-registered 2026-10-11, committed before any result)

Confirmation of the one positive lead of `runs/tactical/tdvar/REPORT.md` (LOG H71): a neural QLIKE tower trained on the
frozen-target TD (tower-property) loss, `nn_td`, beat the same net trained directly on the H-window, `nn_direct`, on short
training windows (+5.6% / +12.0% QLIKE at 30 days, +3.6% / +8.1% at 60 days, about 0 at 1-3 years; H = 240 / 1440 min). That
was a SECONDARY result of tdvar (27 selection months, one seed, caveats: unstable direct net at short windows, MIDAS and the
naive rate better at 30 days, the look period not clean). This study tests it on months the TD study never touched, against
fairer baselines. Risk-branch work (variance, not direction); tactical rigour (D-063); no default change.

## Data, cutoff, anchors (identical to tdvar; code reused, not copied: `import tdvar_lib`, `import tdvar_nn`)
- `Bitcoin_BTCUSDT.csv`, 1-minute close. **No bar at or after 2025-07-21 00:00 is ever loaded** (the last bar is 2025-07-20
  23:59; `tdvar_lib.load_bars` drops later rows on read; test `test_cutoff_no_bar_after_2025_07_20`).
- Anchor `a` every 60 bars; features read bars `< a`; target = realised variance rate over bars `a .. a+H-1`; variance LEVELS
  (rate = mean squared 1-minute log return x 1e6); features = rates over the last 60, 240, 1440, 10080 bars. Anchors start
  at bar 10081 (2017-01-08). An (a, H) with a gap in the file is dropped (the file has one gap, in 2025).
- Horizons H = 240 and 1440 minutes; `s = H/2`. Training windows W = 30, 60, 90 days, and 365 days as the "no gain" control.

## Models (all per (month, W, H), fitted on training anchors only)
Neural nets are exactly `tdvar_nn.Net` (16 tanh units, log-linear skip, last layer zero-initialised, QLIKE in levels, full-batch
Adam, lr 0.01, features log-standardised on the training window); **no retuning of `nn_td` or `nn_direct`**: L2 1e-3, 200 steps.
Seeds 0..4 (the seed sets the first-layer initialisation; the nets of one `nn_td` share it); a month's QLIKE is the mean
over the 5 seeds (per-seed values are stored too).
1. `nn_td`: stage 1 `g_s` on `rate_s(a)`; stage 2 the H head on `Z(a) = (s rate_s(a) + s g_s(x_(a+s))) / H` with `g_s` frozen
   (as `tdvar_nn.cell_nn`).
2. `nn_direct`: the same net trained on `rate_H(a)` (L2 1e-3, 200 steps).
3. `nn_direct_reg` (baseline): the same architecture and loss trained on `rate_H(a)` with the L2 weight and the number of steps
   (early stopping by a fixed step budget) chosen ONCE on 2020-02..2021-12 and frozen, as below.
4. `midas_q` (baseline): direct MIDAS (168 hourly blocks, exponential-Almon weights, the same 20-point theta grid as tdvar) with
   intercept `c0 >= 0` and slope `c1 >= 0` AND theta all fitted by minimising the QLIKE (L-BFGS-B from the OLS start clipped to
   the bounds; theta = the grid point with the lowest training QLIKE). Forecast floored as for all models.
5. `naive` (baseline): the trailing rate over the lag nearest to H (240 or 1440 bars), no fitting.
6. `har_direct`, `har_iter` (baselines): the OLS-in-levels direct HAR and iterated HAR of tdvar (`tdvar_lib.cell`).
   (`midas_sse`, the tdvar SSE MIDAS, is stored for reference and is not a baseline of the criterion.)
All forecasts are floored at 2% of the median training target (same floor for all models); the target is floored at 1e-4 of that
median in the QLIKE. A cell whose median training target is 0 is skipped (and counted).

## Regularised direct net: the one tuning, before the confirmation (script `tune_reg.py`, output `reg_config.json`, committed
before the confirmation run)
- Months 2020-02..2021-12 (23, all of them), W in {30, 60, 90}, H in {240, 1440}, seed 0.
- Grid: L2 in {1e-3, 1e-2, 1e-1, 1.0} x steps in {25, 50, 100, 200} (16 configs; (1e-3, 200) is the unregularised `nn_direct`).
- Score of a config: mean over all (month, W, H) cells of `QLIKE(config) / QLIKE(1e-3, 200)`. Pick the lowest; a tie within
  1e-4 goes to the larger L2, then to fewer steps. Frozen in `reg_config.json` (with the full grid of scores) and used unchanged
  everywhere. The tuning months lie in the replication period only, never in the confirmation period.

## Walk-forward and purge (as tdvar)
Monthly. For the month starting at `t0`: training anchors have `a >= t0 - W days` and **`a + H <= t0`** (every training label,
and the bootstrap read `x_(a+s)`, ends before the month starts); test anchors `t0 <= a < next month start` with their labels ending
by the period end; a test month's labels are never in training. Tests below.

## Periods (exact months)
- **CONFIRMATION: 2017-07 .. 2018-12 (18 months).** Never used by the TD study (which ran 2019-01 onward). Bars before 2017-07
  appear only as training windows and features. Training windows 30, 60, 90 days are complete in every month (data start
  2017-01-01, anchors from 2017-01-08, so a 90-day window is complete from 2017-04-08). The 365-day window is complete only
  from 2018-02 (11 of the 18 months, 2018-02..2018-12); in earlier months it is truncated and the row says so (`trunc`).
  Early-2017 minute data has many zero-return minutes (34% in 2017-01, 13% in 2017-07, 3% in 2018-01); the QLIKE floors handle
  zero-rate targets; this regime difference is a stated limitation.
- **REPLICATION: 2020-02 .. 2024-06 (53 months, all months, not every second).** The months of tdvar's selection; not clean,
  reported as a replication of the secondary result with the baselines and seeds added. Test labels end by 2024-07-01.
  The look period (2024-07..2025-07) is NOT used. No test fold.

## Metrics, intervals
- QLIKE in levels (lower is better), per month, seed mean. Spearman of forecast against realised (secondary).
- `gain_best(W,H,month) = 100 (Q_best - Q_nn_td) / Q_best`, where **best = the baseline with the lowest mean monthly QLIKE over
  the period's months, in that (W, H) cell, among {nn_direct_reg, midas_q, naive, har_direct, har_iter}** (chosen ex post on
  the same months: conservative, it can only make the gain smaller). Sensitivity, reported: the per-month minimum of the five
  baselines (oracle), and each baseline separately.
- `gain_direct(W,H,month) = 100 (Q_nn_direct - Q_nn_td) / Q_nn_direct`.
- A cell's gain is the mean of the monthly gains with a 95% t-interval and a 95% moving-block bootstrap interval (blocks of 3
  months, 2000 draws, seed 0) over months; months are mildly autocorrelated (stated). A paired sign count (months with gain > 0)
  is shown. n_eff per month = `n_test * 60 / H` (printed).

## Prediction (fixed now) and verdict
- **C1** (W = 60, at BOTH H = 240 and 1440): mean `gain_best >= 2.0%`, and the lower end of BOTH the t-interval and the block
  bootstrap 95% interval is above 0.
- **C2** (at both H): Spearman correlation between the mean `gain_direct` and log W over the windows 30, 60, 90, 365 is < 0,
  using the 11 months where all four windows are complete (2018-02..2018-12). (Also shown: windows 30/60/90 on all 18 months.)
- Control (reported, no criterion): `gain_direct` at W = 365 on the 11 complete months should be about 0 (within +-2%).
- **Verdict (confirmation period only).** CONFIRMED iff C1 and C2 hold at both horizons. PARTIAL iff C2 holds at both horizons and
  the point `gain_best` at W = 60 is > 0 at both horizons but C1 fails. NOT CONFIRMED otherwise. The replication is judged by the
  same arithmetic and labelled replication; it cannot change the confirmation verdict. A negative or partial result is a valid
  outcome and is recorded as such. Nothing is retuned after the results.

## Tests (seconds, `test_tdconfirm.py`, `tests.log`)
Cutoff (no bar after 2025-07-20 in memory, synthetic crossing file and the real file); purge (every training anchor's label ends
before the scored month; perturbing every bar from `t0` on leaves every fitted parameter - nets, MIDAS, HAR - unchanged); the TD
target uses only the frozen stage-1 network (stage-2 target equals `(s rate_s + s g_s(x_(a+s)))/H` from the stage-1 net, the
stage-1 parameters are unchanged by stage 2, and the target does not read `rate_H`); the QLIKE-MIDAS fit does not read the SSE
and has a QLIKE no worse than the SSE fit on a heavy-tailed sample; the base `nn_td`/`nn_direct` constants are untouched by the
regularised fit.

## Resources
CPU only, up to 4 processes at below-normal priority, `OMP_NUM_THREADS=1`; no GPU. Outputs in `runs/tactical/tdconfirm/`: this
SPEC, `confirm_lib.py`, `tune_reg.py`, `run_confirm.py`, `analyse_confirm.py`, `test_tdconfirm.py`, `reg_config.json`,
`results_*.jsonl`, `tables_*.txt`, `confirm.json`, `curve_*.png`, `REPORT.md` with run times.
