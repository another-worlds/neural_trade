# volnet SPEC (pre-registered 2026-10-11, committed before any result)

Question: on the recent market, can a neural volatility model beat a HAR-RV forecast of the next |move|?

## Protocol (as runs/tactical/lab/decay_vol.py, extended)
- Data D:/nt/neural_trade/Bitcoin_BTCUSDT.csv, 1-minute bars; anchors every 10 bars (first anchor 1441); anchor `a` sees bars `< a`
  (last bar a-1). Target `y_H = log(|close[a+H-1]/close[a-1] - 1| + 1e-5)`, H in {15, 60, 240} minutes. Scored against |r|.
- Monthly walk-forward. For month M: training window = anchors in [M0 - 365 d, M0) whose label ends by M0 (`t + H <= M0`); scored
  on anchors in M. The last 15% of the training window (chronological, with a gap of H bars after the fit part) is the HOLD part:
  early stopping (nets, HGB iteration count) and the QLIKE scale. All models fit their parameters on the first 85% only.
- Selection months 2021-01..2024-06 (42). THE ONE LOOK 2024-07..2025-09 (15): run only after `selection.json` is committed,
  for the selected model and the baseline only.
- Metrics per month and horizon: Spearman(prediction, |r|); QLIKE = mean(v/s - log(v/s) - 1), v = r^2 + 1e-10, s = c * exp(2 mu),
  mu = predicted log|r|, c = mean(v / exp(2 mu)) on the HOLD part (closed-form minimiser, scale fitted inside the training window).
  QLIKE lower is better.
- Paired monthly difference vs the baseline: Spearman diff = model - base; QLIKE diff = base - model (positive = model better).
  95% t-interval over months (monthly diffs are mildly autocorrelated: stated, not corrected).

## Models
1. `har` HAR+tod ridge (the baseline): log RV over 15/60/240/1440 bars + hour sin/cos, Ridge(alpha 1) on standardised inputs.
2. `hgb` HistGradientBoostingRegressor on HAR + tod + tb7 (13) + volume/range features, per horizon, depth 4, lr 0.05,
   min leaf 200, up to 300 iterations, count chosen on HOLD.
3. `mlp` MLP on the same vector: linear skip initialised at the ridge solution on that vector + zero-init residual MLP (96-96), ~12k params.
4. `seq` patch transformer (240 one-minute bars -> 24 patches of 10; d 32, 2 layers, 4 heads, FF 64) on scale-free channels
   (log return / trailing std of the window, log(high/low) / same std, log volume / window mean volume), pooled, MLP head with a
   zero-initialised last layer, PLUS a linear skip on the HAR vector initialised at the HAR ridge solution (equal to model 1 at init).
5. `ens` mean of mu of models 2, 3, 4.
Diagnostic only, never selected: `ridge_full` (ridge on the vector of model 2/3).
Loss for 3-4: Gaussian NLL on log|r| with one learned log-sigma per horizon (so its gradient in mu is MSE; chosen before running).
Adam 2e-3, batch 512, max 25 epochs, patience 4 on HOLD NLL, best epoch restored; tf.random.set_seed(month index).

## Selection (before the look)
For each of models 2-5: t-statistic of the paired monthly diff over the 42 selection months for Spearman and for QLIKE, at each
horizon; score = mean of those six t-values. The highest score is the selected model, written to `selection.json`.

## Success (the look)
Per horizon: the 95% t-interval of the selected model's paired difference vs `har` on the 15 look months is above 0 on BOTH
Spearman and QLIKE. Otherwise "no". Reported per horizon; no multiplicity correction (3 horizons stated).
