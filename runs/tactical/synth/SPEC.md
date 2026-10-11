# SPEC: neural feature engineering of SYNTHETIC indicators, then geometric technical analysis on them

Owner hypothesis (2026-10-11, verbatim): "протестируй гипотезу: фич-инжиниринг синтетических индикаторов признаков с помощью
нейросети, потом геометрический технический анализ поверх". Branch `nt-tactical-synth` (from `nt-tactical` e17b1fd), worktree
`D:/nt/nt_synth`. Written and committed BEFORE any result. Exploratory rigour (D-062/D-063); no default changes; hard cutoff
`runs/tactical/common/cutoff.py` (no bar after 2025-07-20 23:59 is read; test in `test_synth.py`).

Difference from H24, H43, H45, H47, H49: there the geometry (slopes, distance to channel, crossings, squeeze) was computed on
TEXTBOOK or period-learnable indicators. Here a network freely LEARNS K synthetic indicator series (no RSI/EMA form), and the
geometry runs on those only.

## Components (`synthlib.py`)
- **Input**: the 60-bar OHLCV window, turned into the 5 scale-free per-bar channels of `jepa/channels.py` (return, range, body,
  close position, log volume, all over the window's own scale), standardised with the TRAINING block's statistics, plus a 6th
  channel, the window's cumulative-return path p (the "price" the geometry compares with), scaled by 1/5.
- **Generator**: causal network -> K series of length 60. Causal TCN (Conv1D `padding=causal`, kernel 3, dilations 1,2,4,8,16,
  receptive field 63, 24 channels, tanh) plus a linear causal skip convolution (kernel 3, dilation 4) on the inputs, then a 1x1
  convolution to K channels. Primary K = 12. Output at time t depends on bars <= t only (tested).
- **Geometry layer**: `src/neural_trade/models/layers/indicator_geometry.py` on [series, p] (p is the last channel, so "distance of
  price to the channel" is p - s_k): per series slopes over 3 and 10 bars, asinh distance, smooth 5-bar crossing, squeeze (the
  features of p itself are dropped: 5K features), plus `PairGeometry`: for every pair (i<j) the asinh of the standardised
  difference s_i - s_j and its smooth 5-bar crossing (2 K(K-1)/2 features; K = 12: 132). 192 features, all differentiable.
- **Readout**: BatchNorm (affine) -> Dense, linear, on the geometry features only (a test pins that the readout's input is the
  geometry concatenation and nothing else). Direction: 3 logits (horizons 10/15/20), BCE on the bars beyond a 5 bps deadband (as
  the lab). Volatility: 1 output, MSE on the standardised target.
- **Targets**. Direction: AUC per horizon, the mean of 10/15/20 (the lab's metric). Volatility: log realised volatility of the next
  60 one-minute log returns (bars t+1..t+60 after the window's end); metric Spearman rank correlation per slice/month (and R2);
  baseline HAR+tod = ridge on log RV over the last 5/15/59 in-window returns plus time-of-day harmonics 1..3 (sin, cos).

## Arms
| arm | generator | geometry | readout |
|---|---|---|---|
| **A** end-to-end | trained through the geometry | yes | linear (trained jointly) |
| **B** pretrained | self-supervised, then frozen: from s_t a linear decoder predicts the 5 channels of bar t+1 (MSE, training block, inside windows only) | yes | linear, fitted on the frozen features (sklearn logistic C 0.1 / ridge) |
| C1 random | untrained, 3 random initialisations (mean over them) | yes | as B |
| C2a | trained end-to-end (like A) | NO: the K last values only | linear |
| C2b | the B generator | NO: the K last values only | as B |
| C3 textbook | none: `geom2/geomlib.py` `geom_rich` on the 22 textbook channels (164 features) | yes | as B |
| tb7 / har_tod | none | none | logistic (direction) / ridge (volatility) |

A is compared with C1, C2a, C3 and tb7 (har_tod); B with C1, C2b, C3 and tb7 (har_tod). "Geometry adds over learned series
alone" = A - C2a and B - C2b; "adds over random series" = A - C1, B - C1.
Training of A and C2a: Adam 2e-3, batch 256, at most 12 epochs, early stopping (patience 3) on the last 15% of the training
block (80-window gap), the best epoch kept; L2 1e-4 on the readout; one seed per slice/month. Pretraining of B: 6 epochs.
All models run on the CPU (the networks are tiny; the GPU cap of D-063 is not needed). `tf.random.set_seed` everywhere.

## Protocols
1. **Lab slices** (fast verdict): the 24 slices of `lab/cache` (train block -> fit, val block -> score; consecutive 1-minute
   windows; close-entry labels from the cache). Volatility labels come from the cut CSV after locating the slice's bars (the
   cut file stops at the cutoff). Unit of inference: the slice; paired differences with a 95% t-interval over 24 slices.
2. **Monthly walk-forward** on the selection months 2021-01 .. 2024-06 (42 months): for each month a model is trained on the
   365 days before it (labels must end before the month starts; anchors every 20 bars) and scored on the month (anchors every
   10 bars). Direction labels are NEXT-OPEN (`close[i+H] / open[i] - 1`, H = 10, 15, 20, deadband 5 bps), as in
   `lab/decay_nextopen.py`. Volatility as above. Unit of inference: the month (t-interval; months are weakly dependent, so
   intervals are a little optimistic: stated in the report).
3. **Sensitivity (not in the verdict)**: K = 8 and K = 16 for A and B on the lab slices, direction.

## Decision rule (fixed now)
For each primary arm X in {A, B}, target T in {direction, volatility} and protocol P in {lab, walk-forward}, four paired
differences are formed (X - baseline for tb7 or har_tod, C1, C2 of X, C3): 2 x 2 x 2 x 4 = 32 tests, Holm-adjusted together
(two-sided t p-values). A configuration (X, T) "wins" only if in BOTH protocols all four differences have a raw 95% interval
above 0 AND a Holm-adjusted p < 0.05. A win is the only thing that triggers the ONE look at 2024-07-01 .. 2025-07-20 (not run
otherwise). The direction look period was already used by H56/H61/H73, so even a pass there is weaker than a first look; the
volatility look period was used by H47-era checks only on the hourly/minute HAR comparisons. Anything short of a win is
reported as is ("no gain"). No parameter or arm is changed after seeing results; extra analyses are labelled exploratory.

## Outputs
`results_lab.jsonl`, `results_wf.jsonl` (one row per slice/month with every arm's metrics and the seconds), `report.txt`
(tables per arm and target, paired differences with intervals, Holm, the winner, run times), logs. Tests:
`test_synth.py` (generator causality, geometry shapes/finite/differentiable, readout sees only the geometry, cutoff).
