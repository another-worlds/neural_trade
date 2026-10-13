# Hill-climb v2: 1-hour BTC/USDT trading, Calmar at 2 bps, confirmed on a lockbox of past months (SPEC, fixed 2026-10-13, before any score)

Owner, 2026-10-13 (docs/TACTICAL.md "Hill-climb v2"): the objective is trading **Calmar** (and net return) at **2 bps round trip**, scale **1 hour**; the
confirmation is a **lockbox of past months** set aside before the climb and used for no choice. Tactical, exploratory rigour (D-062/63): CPU only,
nothing here changes a default. Branch `nt-tactical-hc2`, worktree `D:/nt/nt_hc2`. Code: `hc2lib.py`, tests: `test_hc2.py`, months: `months.py`.

## Data, cutoff, execution (all fixed)
- All data is cut at `2024-07-01` (the spent period 2024-07 onward is neither in the climb nor in the lockbox, and no bar from it is read; the
  physical cut file of `common/cutoff.py` ends at 2025-07-20 and the loaders here truncate again at 2024-07-01, asserted). Hourly bars, features
  (`tb7` 13, `rich` 43 and `ctx` 17 per-bar context features, so `rich+ctx` has 60) and HAR volatility are the ones of the frozen H62 pipeline
  (`runs/tactical/fwd`, `sizing`, `stops`).
- **Entry:** at the open of the first 1-minute bar of the hour after the signal hour (next open). **Path:** read on 1-minute high/low/open/close:
  a stop fills at the worse of its level and the minute's open; a take-profit fills at its level; stop before take-profit in the same minute; the
  time exit is the close of the last minute of the holding horizon (`stoplib.sim_trade`, reused unchanged). A trade whose minute path is not fully
  inside the data is dropped before the non-overlap rule.
- **Non-overlap:** a signal is skipped while a position is open (`entry < previous actual exit`); entry == exit is taken. Cost: 2 bps round trip on
  every trade at its exit; fixed notional 1x (additive equity `1 + sum(w * pnl)`, no compounding); drawdown in percent of the running peak.
- **Information at entry only:** the selection threshold of a month is the (1 - q) quantile of |P - 0.5| over the **previous** month's predictions
  (V_prev of `stoplib.month_thresholds_prev`); the HAR sigma is the state at the signal bar; the drawdown brake uses closed trades only; models are
  refitted monthly on anchors whose label bar closed by the month start (`wf.split_masks`).

## The lockbox (drawn before any score, never scored during the climb)
- 12 months, stratified 2 per year over 2019-2024 (2024: from January-June only, because 2024-07 onward is spent), drawn with
  `numpy.random.default_rng(20261013).choice(months of the year, 2, replace=False)` year by year in order (`months.draw_lockbox`):
  **2019-05, 2019-12, 2020-03, 2020-07, 2021-02, 2021-10, 2022-09, 2022-10, 2023-01, 2023-04, 2024-01, 2024-06.**
- **DEV months:** 2019-01 .. 2024-06 minus the lockbox = 54 months. The month 2018-12 is a warm-up: predicted (its confidences give January 2019's
  threshold), never scored.
- A lockbox month is never scored during the climb: no signal, trade, path, P&L or metric of a lockbox month is computed, and the predictions of the
  climb are not even produced for lockbox months (their store holds NaN there). `test_hc2.py` enforces it (the evaluator refuses a lockbox month
  unless the look flag is set; every dev trade has its entry AND its exit inside dev months: a signal whose holding window would reach the next month
  when that month is in the lockbox is dropped; the cache files of the climb hold no lockbox prediction).
- **Lockbox months DO appear inside training windows (labels and features) of later months, and in the HAR training rows.** This is past information
  at the time of the later month's prediction and is how the live system would train; it is stated, not hidden. A dev month right after a lockbox
  month takes its threshold from the nearest earlier non-lockbox month (`months.ref_month`) during the climb, so no lockbox prediction is used by it.
  In the single look the plain previous calendar month is used.
- **Partial cleanliness (owner's caveat):** earlier tactical studies (H56/H61/H62/H73/H78) saw all these months (for direction, the frozen H62
  candidate was chosen on 2019-2024-06 months). The lockbox is clean of THIS climb's choices only.

## Baseline
The frozen H62 candidate: features `rich`, logistic (C = 0.1), training window 730 days, horizon 4 h, selection top 10% by |P - 0.5| with the threshold
from the previous month (V_prev), side = the model's (long if P > 0.5), time exit after 4 h, fixed notional 1x.

## Levers and values (1,944 configurations)
| lever | values |
|---|---|
| features | `tb7`, `rich`, `rich+ctx` |
| model | `logreg` (C = 0.1), `l1` (L1 logistic, C by the last-20% rule), `hgb` (HistGradientBoosting; the `wf` settings) |
| training window | 365, 730 days |
| horizon | 1, 2, 4 h (label = next-open return over the horizon; holding time = the horizon) |
| selection quantile | top 5%, 10%, 20% (previous month's quantile of |P - 0.5|) |
| side rule | `follow` the model; `agree`: trade only when the models of horizons 1, 2 and 4 (same features, model, window) all give the same side |
| exit | `time`; `tp1.5`: stop = take-profit = 1.5 sigma; `tp2`: stop = take-profit = 2 sigma, sigma = HAR forecast sigma x sqrt(horizon hours) at the signal bar |
| size | `fixed` 1x; `brake`: weight 0.5 while the running drawdown is >= 2% (until a new equity peak; `sizelib` rule R4) |

3 x 3 x 2 x 3 x 3 x 2 x 3 x 2 = 1,944. Prediction sets: 3 x 3 x 2 x 3 = 54 (each month refit independently; HGB and L1 are the slow ones).

## Objective and validity
- **Calmar** on the dev months = (cumulative return % / years) / max drawdown %, years = total days of the scored months / 365.25, the equity
  being the dev months' trades concatenated in time order. Net return (cumulative %, 2 bps) is reported beside it.
- A configuration with fewer than **500 dev trades** is invalid (Calmar = -inf): few trades make a tiny drawdown and an absurd Calmar.
- The **monthly net bps** of a configuration is the sum of its trades' net bps per entry month (a month without trades counts 0): the vector the paired
  condition uses.

## The climb procedure
- **Coordinate ascent:** from the incumbent, the levers are visited in a random order (seeded); for a lever, every other value is evaluated; the
  candidates that satisfy BOTH (i) dev Calmar strictly above the incumbent's AND (ii) the paired monthly net-bps difference (candidate - incumbent,
  over the 54 dev months) has a **positive mean** are collected, and the one with the best Calmar is accepted. One pass visits all 8 levers; the
  climb stops after a pass with no accepted step.
- **Starts:** the baseline plus **9 random starts** (configurations drawn uniformly, seed 20261013 + start index) = 10 climbs. The climb's result is
  the final incumbent with the best dev Calmar (ties: higher net return).
- **Budget (hard):** at most **6 CPU-hours** in total for everything in step 3 (prediction sets, paths, the real climb, the null climbs, the
  exhaustive table); the runners count process time and stop if exceeded. At most 6 worker processes at below-normal priority; no GPU.
- **Logged:** every evaluated configuration (key, dev trades, Calmar, return %, DD %, hit) to `trials_real.jsonl`; every accepted step to the
  trajectory. Evaluations are memoised, so the count of distinct configurations is the cost of the search.
- **Extra, descriptive:** the exhaustive dev table of all 1,944 configurations (where the winner ranks, how much of the climb's result is search),
  never used to choose.

## Controls
- **(a) Null climb:** the SAME climb procedure on a world where the direction skill is destroyed: for each draw, every dev month gets a random sign
  s_m = +/-1 (seeded) and every trade of that month takes the OPPOSITE side when s_m = -1 (its path is the other side's path of the same signal). This
  equals flipping the sign of the month's realised labels; the models are NOT retrained (training labels are the real ones; retraining 54 prediction
  sets per draw is the only thing this saves, and it keeps volatility, signal times and thresholds unchanged). **40 draws** (seeds 1..40), each with
  the same 10 starts (the "baseline" start is the baseline configuration). Reported: the distribution of the climb's final dev Calmar and net
  return (median, p90, p95, max) and the exhaustive best-of-1,944 on 40 draws: the Calmar a search finds from pure selection noise.
- **(b) The baseline** evaluated on the same dev months and, at the look, on the lockbox.

## Verdict on the lockbox (ONE look, for the final winner and the baseline, 2 bps, after `winner.json` is committed)
`run_lockbox.py` refuses unless `winner.json` exists; it computes predictions, paths and trades of the 12 lockbox months once; no re-selection.
The winner PASSES only if ALL hold:
1. **Mean monthly net bps** (sum of net bps per lockbox month, 12 months, empty months count 0) with the 95% t-interval (t_0.975,11 = 2.201) **above 0**;
2. the winner's lockbox **Calmar above the baseline's**;
3. the winner's cumulative return **above its own random-side null p95** (500 draws, a fair coin per signal for the side, the whole sequence incl.
   non-overlap, brake and exits re-simulated).
If the winner IS the baseline the comparison 2 cannot hold: the answer is "the climb found nothing better than the baseline".
Reported for the winner and the baseline, on dev and on the lockbox side by side: trades, hit (pooled), mean net bps/trade, cumulative return, max
DD, Calmar, months with hit > 60%, months with within-month DD < 5% (the owner's goal: hit > 60% and DD < 5%), the leverage frontier (fixed weights
0.1..1.0, and the largest weight with DD < 5% and the return there), exit-type shares.
Noise: 12 months is a small sample (the t-interval carries it); the lockbox drawdown is one path. A clear negative answer is a valid outcome and is recorded.

## Disclosures
- Exploratory, one seed per model, no multiple-testing correction beyond the null-climb comparison; the dev Calmar of the winner is optimistic by
  construction (the null climbs measure by how much).
- The prediction sets use `wf._fit` as frozen (HGB: 150 iterations, depth 4, seed 0).
