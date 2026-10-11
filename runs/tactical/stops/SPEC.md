# Stops and take-profits set by the HAR volatility forecast: SPEC (fixed 2026-10-11, before any result)

Owner, 2026-10-11 (tactical, any target allowed): test the combination "regression gives the SIDE; HAR and its improvements give position
SIZE, STOPS and ENTRY TIMING; the volatility-change forecast cuts risk before spikes"; goal: max drawdown < 5% (and, as before, hit > 60%).
Sizing, the vol-change filter and drawdown brakes are already tested (LOG H78, `runs/tactical/sizing/REPORT.md`: none reaches the goal; drawdown
falls only with size). THE UNTESTED PIECE HERE: stops and take-profits whose distance is set by the HAR forecast. CPU only, descriptive-
exploratory rigour (D-062/63); nothing here changes a default. Branch `nt-tactical-stops`, worktree `D:/nt/nt_tactical_stops`.

## Data, cutoff, execution (all fixed)
- `runs/tactical/common/cutoff.py`: the physically cut file (last bar 2025-07-20 23:59); every loader asserts the bound. The selection phase
  truncates ALL data (1-minute bars, side-signal features, HAR) before 2024-07-01; the look phase ends 2025-07-20.
- **Entry** at the open of the first 1-minute bar of the hour after the signal hour (next-open fill; the same entry as the sizing study).
  Signals are hour bars as in H78; the PATH of a trade is read on 1-MINUTE bars (high/low/open/close), which is the finest data.
- **Time exit** (the baseline and the fallback of every rule): the close of the last minute of the signal's holding horizon: h = 1 hour for
  side signal (a), h = 4 hours for (b). Equals the frozen `r_long` of the hourly code (a test checks it on every signal).
- **Stops on bar extremes, conservative:** on each minute after (and including) the entry minute, a long's stop is hit if the minute's LOW <=
  stop level, a short's if the HIGH >= stop level. Fill = the worse of the stop level and that minute's OPEN (a gap through the stop fills at
  the open). A take-profit is hit if HIGH >= level (long) / LOW <= level (short) and fills AT the level (no gap improvement). If a stop and a
  take-profit are both reachable in the same minute, the STOP is taken. A trailing stop on minute k uses the best extreme through minute
  k-1 only (the minute's own extreme never raises its own stop).
- The exit of an early exit counts as at the END of the exit minute (so the next position cannot open earlier than it could).
- **Non-overlap:** signals are scanned in entry order; a signal is skipped while a position is open (entry < the previous ACTUAL exit; with
  an early exit the position frees earlier, so a later signal can be taken that the time exit would have skipped); entry == exit is taken.
  A signal removed by a filter never blocks the next one. Round-trip cost c bps (0/1/2/5) is charged on every trade at its exit, whatever
  the exit type, on notional 1 (additive equity `1 + sum(w x pnl)`, no compounding), drawdown in percent of the running peak.

## Side signals (frozen; direction code unchanged), and the top-10% threshold in TWO versions
- **(a)** 1-hour `rich`, L1 logistic (C by the last-20% rule), 365-day monthly walk-forward, h = 1, top 10% by |P - 0.5|, long if P > 0.5.
- **(b)** the frozen H62 candidate: 1-hour `rich`, logistic C = 0.1, 730-day walk-forward, h = 4, same top-10% rule.
- **V_same (the frozen convention of H56/H61/H62/H78):** the 90th percentile of |P - 0.5| over THE SAME month's predictions. Not known at entry.
- **V_prev (known at entry):** the 90th percentile of |P - 0.5| over THE PREVIOUS calendar month's predictions (a signal in month m is taken when
  its |P - 0.5| >= that threshold; a month whose previous month has no predictions has no signals). V_prev is the PRIMARY version for the verdict
  (it is implementable); V_same is reported beside it and must reproduce the sizing study's R0 (a check).
Four (side signal, version) combinations in all; selection and the look treat each separately.

## Volatility inputs (`runs/tactical/sizing/sizelib.py`, unchanged; all from bars up to the signal bar j)
HAR forecast `pred_j` of log mean hourly RV over the next h hours (monthly OLS on the previous 365 days, rows whose targets closed by the month
start), `sigma_j = exp(pred_j / 2)` (per-hour volatility), `dlv_j = pred_j - log RV_j`, `thr_j` = trailing 180-day 80th percentile of dlv.
**Trade sigma** `s_j = sigma_j x sqrt(h)` (the HAR forecast of the std of the log return over the holding horizon), used as a fraction of the
entry price: a stop distance `m x s_j` below (long) / above (short) the entry price.

## The exit rules (six, fixed; no parameter is tuned; m multiples of s_j)
| id | rule |
|---|---|
| X0_time | time exit only (baseline) |
| X1_sym1 | stop-loss = take-profit = 1.0 s_j |
| X2_sym15 | stop-loss = take-profit = 1.5 s_j |
| X3_sym2 | stop-loss = take-profit = 2.0 s_j |
| X4_asym | take-profit 2.0 s_j, stop-loss 1.0 s_j |
| X5_trail1 | trailing stop 1.0 s_j from the best extreme since entry (no take-profit) |
Plus ONE derived rule, not a seventh parameter set: **X6_best_plus_filter** = the best of X1..X5 (fixed by selection, below) combined with
H78's vol-change filter (skip a signal when `dlv_j > thr_j`, the trailing top quintile of the predicted log-vol change; the filter acts on
signals first, then the non-overlap rule).

## Metrics (per combination, rule, cost, period)
n trades, hit (net P&L > 0), mean net bps per trade, cumulative return % (additive, 1x), max drawdown %, Calmar = (cum % / years) / max DD %,
months with hit > 60%, share of calendar months whose within-month max drawdown is < 5%, exit-type shares (stop / take-profit / time),
the average holding time. **Random-side null per rule:** 500 draws, a fair coin per signal for the side, the WHOLE sequence re-simulated
(the exits and the non-overlap depend on the side, so the trade set is re-derived per draw); reported: cumulative-return median/p95, max-
drawdown median/p5/p95, hit p95, P(null cum >= actual).

## Selection and the ONE look
- **Selection period 2021-01..2024-06.** Per combination: `best5` = the highest Calmar at 2 bps among X1..X5; X6 = best5 + filter. The
  **chosen** rule = the highest Calmar at 2 bps among X1..X6, provided it exceeds X0's Calmar on the selection period; otherwise none is chosen.
  Written to `selection.json` (committed) before the look; `run_stops.py look` refuses without it.
- **ONE look 2024-07-01..2025-07-20**, run once. **Disclosure:** the look period was already used for DIRECTION by these side signals (H56/H61/H73) and
  by the sizing look (H78); it is not fresh for them. The exit rules are new and their selection never saw it. All rules are shown on the look for
  description; only the chosen rule and X0 enter the verdict. No second look, no re-selection.

## Verdict rules (fixed now), at 2 bps, on the look, for the chosen rule of each combination (V_prev primary)
1. **Drawdown goal met:** max DD < 5% AND cumulative return > 0 AND share of months with within-month DD < 5% >= 0.9 AND >= 100 trades.
2. **Hit goal met:** hit > 0.60 pooled AND the 95% binomial interval excludes 0.50.
3. **Goal met** only if both. DD < 5% by trading too little, or with cumulative return <= 0, is not a success.
4. **Value of the exit rule:** the chosen rule beats X0 if its look Calmar exceeds X0's AND its cumulative return exceeds its own random-side null p95;
   otherwise "no evidence that the HAR-set exits help".
5. No rule chosen in selection: reported at X0 only; the answer is "no".
Noise: ~700-900 look trades give a hit standard error of ~1.7-1.9 points; the look drawdown is one path.

## The leverage frontier (price of DD < 5%)
For every rule, both periods and every cost: fixed notional weights w = 0.1, 0.2, ..., 1.0 (pnl x w, cost x w): cumulative return % and max DD %.
Also the largest w on a 0.01 grid (up to 1.0) with max DD < 5% and the return there. Reported for the chosen rule (or X0 if none) and X0.
The verdict question: how much return does DD < 5% cost with and without the HAR exits.

## Tests (seconds; `test_stops.py`)
1. A trade's exit and P&L use only the entry price, the entry-time stop distance and the bars from entry to the exit: corrupting minutes after the exit,
   or the HAR/volatility inputs after bar j, changes nothing; the trailing stop at minute k ignores minute k's own extreme.
2. Gap through the stop fills at the open; stop and take-profit in the same minute -> stop; take-profit fills at the level.
3. Non-overlap with early exits (stopped trade frees the position; entry == exit taken; filtered signal never blocks).
4. Cutoff: minute bars and signals end at or before 2025-07-20 23:59; selection data end before 2024-07-01.
5. V_prev threshold of month m depends only on month m-1's predictions.
6. X0 gross P&L equals the frozen hourly `r_long` on every real signal; hand-computed equity/drawdown path; null determinism.

## Outputs
`stoplib.py`, `run_stops.py`, `test_stops.py`, `make_report.py`, `results_sel.json`, `selection.json`, `results_look.json`, `REPORT.md`, logs, in
`runs/tactical/stops/`. Resource rule: <= 4 processes, below-normal priority. LOG.md, PLAN.md, the dashboard and the queues are not edited.
