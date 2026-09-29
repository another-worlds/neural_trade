# Micro loop (D-041): hypothesis journal

Goal (owner, 2026-09-29): raise predictive power and strategy PnL, iterating on micro setups (minutes per
run) so hypotheses turn around in hours. Quick sweeps and descriptive analyses; anything "A beats B" still
goes through D-025. Reference block for CPU work: the 360-day run's dev block (2025-07-27 .. 2025-08-28).

| # | Hypothesis | Method | Cost | Result |
|---|---|---|---|---|
| H2 | Fewer, more selective trades of the existing cq signal survive the 26 bps cost | CPU rescore, entry_quantile 0.9-0.995 x max_hold 15/60/240 on the 360d cell (rescore/cq_selectivity_v1-20260929T123716Z) | 1 CPU-min | **Negative.** Gross edge stays ~1 bps per trade at every threshold (at 0.995 it is negative); tightening the quantile only cuts the trade count. max_hold is inert: the median-cross exit ends trades at ~8-10 bars whatever the cap. |
| H2b | The cq signal persists beyond the exit's ~10 bars, so longer holds could pay | Signal-decay curve on stored predictions: IC of sign(weighted_direction - cal median) vs k-bar forward returns, k = 5..480 | seconds | **Negative.** IC peaks at 15-20 bars (0.026-0.027, z ~1.3-1.5 on n_eff) and dies by 120; gross edge <=0.8 bps per trade at any k. The signal lives only at the trained horizons and is ~30x below cost. |
| H1 | Horizons of 1-4 h (move sd 2-4x the 26 bps cost) carry a tradable edge | Quick sweep configs/scenarios/micro_horizons.yaml: h_15m / h_1h / h_4h, micro layout (~10-day train, ~4 min per cell), one seed | ~15 GPU-min | running |

H2b table (360d cell, dev block, n = 46,544 bars):

| hold k (bars) | IC | gross bps per trade | z at n_eff |
|---|---|---|---|
| 5 | 0.018 | 0.19 | 1.8 |
| 10 | 0.022 | 0.31 | 1.5 |
| 15 | 0.026 | 0.44 | 1.4 |
| 20 | 0.027 | 0.54 | 1.3 |
| 30 | 0.016 | 0.40 | 0.6 |
| 60 | 0.023 | 0.79 | 0.6 |
| 120 | 0.007 | 0.36 | 0.1 |
| 240 | -0.007 | -0.40 | -0.1 |

Reading: to reach PnL, the model must be trained (and calibrated) on horizons where a move is several times
the cost, or on a P&L-aware target: reshaping the trading of the 10-20-bar signal cannot get there.
