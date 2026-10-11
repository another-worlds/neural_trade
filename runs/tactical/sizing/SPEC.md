# Can volatility forecasting turn a weak direction signal into the owner's drawdown and hit-rate goal? SPEC (fixed 2026-10-11, before any result)

Owner, 2026-10-11 (tactical, any target allowed): volatility is predictable (HAR Spearman ~0.40, LOG H53; next-hour log-vol change
R2 0.2, H72). Direction is not (hour AUC ~0.52-0.54). Can POSITION SIZING and TIMING, not better direction, bring the strategy to the
owner's goal: max drawdown < 5% and hit rate > 60%? CPU only, descriptive-exploratory rigour (D-062/63). Nothing here changes a default.

## What sizing can and cannot do (stated before the numbers)
- Sizing alone cannot change the hit rate (the sign of a trade's net P&L is the sign of its gross P&L minus a cost that scales with the
  notional). Only the trade filter (R2, R3, R5) changes which trades are taken, hence hit. A hit rate above 60% from a signal whose
  top-10% hit rate is ~0.53 is therefore not expected; the drawdown is what sizing can move.
- Rules that change the number of trades are compared on the same period, so a lower trade count is a cost the Calmar must absorb.

## Data and cutoff
`runs/tactical/common/cutoff.py`: the physically cut file (last bar 2025-07-20 23:59); every loader asserts the bound. The selection
phase additionally truncates all data (minute RV, side-signal features, labels) before 2024-07-01. Hourly bars: the 1-hour resample
used everywhere (`dropna` on the close). Costs 0/1/2/5 bps round trip per unit notional (a trade of weight w pays w x cost);
equity conventions of `runs/tactical/equity/eqlib.py`: next-open entry, additive equity `1 + sum(w x pnl)`, non-overlapping trades
(a signal is skipped while a position is open; entry == previous exit is taken), drawdown in percent of the running peak.

## Side signals (frozen, not re-selected; direction code unchanged)
- **(a)** 1-hour, `rich` features, L1 logistic (C by the last-20% rule), 365-day monthly walk-forward, H = 1 bar, top 10% of
  |P - 0.5| per month (the month's own quantile), long if P > 0.5 else short (`runs/tactical/recent/wf.py`, run_b convention; H56/H61).
- **(b)** the frozen H62 candidate: 1-hour, `rich`, logistic C = 0.1, 730-day walk-forward, H = 4 hours, top 10% per month
  (`runs/tactical/fwd/fwdlib.py`, FORWARD_SPEC.md). Entry at the open of the bar after the signal bar, exit at the close of the 4th bar.
Inherited caveat: the per-month top-10% threshold is the 90th percentile of that month's own confidences, which is not known at
entry; this is the frozen convention of both signals and is not changed here (it affects which signals exist, not any sizing input).

## Volatility inputs (all from bars up to and including the signal bar j; the trade enters at the open of bar j+1)
- Hourly realised variance RV_j = sum of squared 1-minute log returns of hour j.
- **HAR forecast** for a trade of horizon h bars (h = 1 for (a), 4 for (b)): OLS of y_j = log(mean RV over bars j+1..j+h) on
  [1, log RV_j, log mean RV(j-23..j), log mean RV(j-167..j)], refitted monthly on the 365 days before the month; a training row
  needs its target bars closed by the month start. `pred_j` is the fit, `sigma_j = exp(pred_j / 2)`.
- **Predicted log-vol change** `dlv_j = pred_j - log RV_j` (the forecast of the coming window against the hour just completed).
- Trailing statistics over the last 4,320 bars (180 days, minimum 720) including j: `target_j` = median of sigma, `thr_j` = 80th
  percentile of dlv.

## The six rules (fixed; no others, no parameter is tuned)
| id | rule |
|---|---|
| R0_fixed | weight 1 (the baseline) |
| R1_voltarget | w = clip(target_j / sigma_j, 0.25, 2.0) |
| R2_volchange_filter | skip a signal when dlv_j > thr_j (top quintile of the trailing predicted change); weight 1 |
| R3_vt_plus_filter | R1 and R2 |
| R4_brake | weight 1, halved while the rule's own realised equity is in a drawdown of at least 2% from its peak, restored at a new peak |
| R5_all | R1, R2 and R4 |
Order of operations: the filter acts on signals first, the skip-while-open rule on what is left; the brake state at an entry uses
trades closed before it (non-overlapping, so all earlier trades). The cost is charged on the weighted notional.

## Metrics (per rule, per side signal, per cost, per period)
n trades, hit (net P&L > 0), mean net bps per trade (weighted), cumulative return % (additive, 1x capital), max drawdown %, Calmar =
(cumulative return % / years of the period) / max drawdown %, months with hit > 60%, share of calendar months whose within-month max
drawdown is < 5%, average weight. **Random-side null per rule:** 500 draws, a fair coin per trade for the side on the SAME trades and
weights (the brake is re-simulated per draw); reported: cumulative-return median/p95, max-drawdown median/p5/p95, hit p95, and
the share of draws with cumulative return >= the actual.

## Selection and the one look
- **Selection period 2021-01..2024-06.** For each side signal choose at most one of R1..R5: the one with the highest Calmar at
  2 bps on the selection period, provided it exceeds R0's Calmar there; otherwise no rule is chosen. Nothing else is used.
- **ONE look 2024-07-01..2025-07-20**, computed once after `selection.json` and the selection table are committed (`run_sizing.py
  look` refuses without `selection.json`). **Disclosure:** the look period was already used for direction by these two side signals
  (H56/H61/H73), so it is not fresh for the side signals; the sizing rule is new, and its selection never saw the look period.
  All six rules are shown on the look for description; only the chosen rule and R0 enter the verdict. No second look, no re-selection.

## Verdict rules (fixed now), at 2 bps, on the look period, for the chosen rule of each side signal
1. **Drawdown goal met** if max drawdown < 5% AND cumulative return > 0 AND the share of months with within-month drawdown < 5% >= 0.9.
2. **Hit goal met** if hit > 0.60 (pooled over the look trades) AND the 95% binomial interval (n trades) excludes 0.50.
3. **Goal met** only if both. A rule that keeps drawdown < 5% by trading too little (< 100 trades) or with cumulative return <= 0 is not a success.
4. **Value of sizing**: the chosen rule beats R0 if its look Calmar exceeds R0's AND its cumulative return exceeds the random-side null p95
   of the same rule. Otherwise "no evidence that sizing helps".
5. If no rule is chosen for a side signal in selection, that side signal is reported at R0 only and the answer for it is "sizing does not help".
Noise: trades are non-overlapping and months are the unit for monthly shares; ~700 look trades give a hit standard error of ~1.9 points.
Look-period drawdown is a single path; no claim is made beyond it.

## Tests (seconds; `test_sizing.py`)
1. Sizing inputs at bar j (HAR forecast, dlv, target, thr, hence the weight and the skip flag) are unchanged when every bar after j
   is corrupted (a synthetic series; the test also checks the training rows end before the month start).
2. The brake's weight at trade i is unchanged when the P&L of trade i and later trades is corrupted.
3. Non-overlap: every rule's kept trades open at or after the previous close; a filtered signal never blocks the next one.
4. Cutoff: the hourly RV and the side-signal data end at or before 2025-07-20 23:59; the selection phase data end before 2024-07-01.
5. Hand-computed brake path and drawdown; the R0 sequence equals the frozen non-overlapping sequence of `fwdlib.trades_from` for (b).

## Outputs
`results_sel.json`, `selection.json`, `results_look.json`, `REPORT.md`, logs. Resource rule: <= 4 processes, below-normal priority.
