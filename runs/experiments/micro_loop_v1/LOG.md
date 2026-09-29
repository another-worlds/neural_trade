# Micro loop (D-041): hypothesis journal

Goal (owner, 2026-09-29): raise predictive power and strategy PnL, iterating on micro setups (minutes per
run) so hypotheses turn around in hours. Quick sweeps and descriptive analyses; anything "A beats B" still
goes through D-025. Reference block for CPU work: the 360-day run's dev block (2025-07-27 .. 2025-08-28).

| # | Hypothesis | Method | Cost | Result |
|---|---|---|---|---|
| H2 | Fewer, more selective trades of the existing cq signal survive the 26 bps cost | CPU rescore, entry_quantile 0.9-0.995 x max_hold 15/60/240 on the 360d cell (rescore/cq_selectivity_v1-20260929T123716Z) | 1 CPU-min | **Negative.** Gross edge stays ~1 bps per trade at every threshold (at 0.995 it is negative); tightening the quantile only cuts the trade count. max_hold is inert: the median-cross exit ends trades at ~8-10 bars whatever the cap. |
| H2b | The cq signal persists beyond the exit's ~10 bars, so longer holds could pay | Signal-decay curve on stored predictions: IC of sign(weighted_direction - cal median) vs k-bar forward returns, k = 5..480 | seconds | **Negative.** IC peaks at 15-20 bars (0.026-0.027, z ~1.3-1.5 on n_eff) and dies by 120; gross edge <=0.8 bps per trade at any k. The signal lives only at the trained horizons and is ~30x below cost. |
| H3 | The owner's target (>60% stable hit rate, drawdown <5%) is reachable by trading only the current model's most confident bars | Conditional hit rate by calibrated-P(up) confidence bucket on the 360d cell's dev block | seconds | **Negative for the current model.** Top-10% bars: 51.0-52.9% hit (+-4.5-6.4 pp); the top-0.5% buckets are too small to score (n_eff 12-23, +-20-29 pp). Calibrated p stays inside [0.43, 0.58] on 98% of bars: the model is honest about knowing little. A stable >60% needs a stronger signal, not a threshold. |
| H1 | Horizons of 1-4 h (move sd 2-4x the 26 bps cost) carry a tradable edge | Quick sweep configs/scenarios/micro_horizons.yaml: h_15m / h_1h / h_4h, micro layout (~10-day train), one seed, 13.1 GPU-min; cells under runs/scenarios/micro_horizons/ | 13 GPU-min | **Direction: negative.** No horizon beats logreg_lags (h_1h h1 and h_4h h2 significantly worse); gross edge per trade negative in all three cells. **Variance: the edge grows with horizon** (CRPSS 0.010-0.017 at 15m, 0.015-0.024 at 1h, 0.040-0.067 at 4h), but coverage at 4h falls to 0.85-0.87 (the 10-day cal block has n_eff ~40 at 320 bars) and var/err^2 Spearman shrinks. Quick-sweep label: 1 seed, fold -2. |
| H1b | Confident tails at 1-4 h reach 60% hit | hitrate_buckets.py on the h_1h and h_4h cells | seconds | **Not measurable, and not promising.** Top-0.5% buckets show 36-69% hit at n_eff 1-5 (+-42-98 pp): noise. Top-10% buckets: 44-55%. The useful fact: at 4 h the median \|move\| is 26-39 bps (>= the 26 bps cost), and 35-51 bps on the top-10% bars, so a stable 55%+ direction call at 4 h would be tradable; direction, not the cost, is the bottleneck at every scale. |

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

## Iteration 2 (started 2026-09-29): richer inputs

H2, H2b, H3, H1 and H1b all point the same way: the close-only input carries no directional signal at any
horizon, and no strategy or threshold on top of it changes that. The next lever is the model's inputs:
OHLCV and the indicator catalogue (D-031, NT-046/NT-047), evaluated on the micro layout (~4 minutes per
cell) against the same dev block, with the owner's target (stable >60% hit, drawdown < 5%) as the yardstick
and logreg_lags as the bar to clear first.

| H4 | The 60-bar window was the mismatch: a 240-bar window unlocks the 1-4 h horizons | Quick sweep configs/scenarios/micro_lookback.yaml (LOOKBACK 240 x the three horizon sets), one seed; cells under runs/scenarios/micro_lookback/ | 32 GPU-min | **Direction: no change** (no horizon beats logreg_lags; deltas -0.034..+0.010, all within ~1 SE). Variance edge weaker than at L60 for 1 h and 4 h. **Confounded:** batch 2048 and 512 OOM at LOOKBACK 240 (the attention softmax is quadratic in the window), so these cells ran at batch 256 and early-stopped at epochs 7-14 serving epochs 1-8: the 4 h cell is barely trained. H4b re-runs it with more patience before any reading. |

Operational findings of H4: (1) the attention memory wall (OOM at batch >= 512 for LOOKBACK 240) is a
config-guard candidate (NT-038) and constrains the window-free plan's sizing; (2) five failed OOM run
directories sit under runs/scenarios/micro_lookback/ (kept, D-029); (3) the GPU-free check's 30% sm line
trips on the owner's active desktop with no compute process present - RUNBOOK clarification candidate.

| H4b | H4's 4 h cell was merely undertrained (it served epoch 1) | Re-run with EPOCHS 60, EARLY 12, PATIENCE 6 (configs/scenarios/micro_lookback_h4b.yaml); cell under runs/scenarios/micro_lookback_h4b/ | 8.5 GPU-min | **Negative: not undertraining.** Validation loss never improved after epoch 1 in 13 epochs; the served weights are epoch 1 again. Direction still within 1 SE of logreg_lags on every horizon; variance CRPSS no better than the 60-bar window's. On the close-only input, the (window, horizon) plane is exhausted. |
| H5 | The 4 h horizons were data-starved, not signal-free: 10-day training gives ~40 effective outcomes at 320 bars; 360 days gives ~1,600 | configs/scenarios/h4h_360d.yaml: 160/240/320-bar horizons on the 360-day block, window 60, batch 2048, one seed; cell under runs/scenarios/h4h_360d/ | 19.4 GPU-min | **Negative for direction, instructive for variance.** With n_eff 145-290 (vs ~40), direction stays within 1 SE of logreg_lags (deltas -0.011..-0.003). H1's large 4 h CRPSS (0.04-0.07) does not replicate (0.009-0.027 here): a short-block artifact. More data repairs calibration (coverage 0.906-0.921; var/err^2 Spearman 0.18-0.24, the loop's best), not direction. Betas 0.83-1.00 (the delta heads survive at these horizons), yet delta skill vs zero is ~0. |

## Iteration 1 closed (2026-09-29)

Eight hypotheses (H1-H5 with variants) exhaust the close-only input across horizon (10 min - 5.3 h) x window
(60 / 240 bars) x training size (10 / 360 days) x strategy shaping (selectivity, holds, confidence buckets):
**direction never beats a logistic regression on lagged returns anywhere**, and every gross edge is about
1 bps per trade against the 26 bps round trip. What data volume does buy is honest uncertainty: coverage at
target and the loop's best var/err^2 ranking. The owner's target (stable >60% hit, drawdown < 5%) is not
reachable from this input. Iteration 2 is running: OHLCV input and the ten new indicator families (NT-047),
then the same micro evaluation against logreg_lags on the same dev block.

| I2-duel | OHLCV input + 14 indicator families (NT-047) give direction the close-only input lacks | Quick sweep configs/scenarios/micro_ohlcv_duel.yaml on branch nt-047-duel (e1c8b93 + the batch edit): close4 vs ohlcv14, micro layout, 3 seeds each; cells under runs/scenarios/micro_ohlcv_duel/ | 27 GPU-min | **Negative (provisional).** All 18 cell-horizons within noise of logreg_lags (boot z -1.78..+0.37); at the 3-seed mean both variants sit BELOW logreg_lags on every horizon (AUC - logreg: close4 -0.004/-0.010/-0.007, ohlcv14 -0.008/-0.007/-0.004, inside the seed spread); ohlcv14's seed spread is 2-5x close4's; gross edge -0.24..+0.64 bps per trade. Cost: ohlcv14 ~2.2-2.5x GPU time per training sample and OOM at batch 2048 (attention over 128 tokens), so it ran at 1024: a confound. Provisional because 3 of the 14 families (stoch, willr, donchian) carried QA's soft-extremum scale bug; the 3 ohlcv14 cells re-run after NT-047's repair. **Final (re-run 2026-09-30 on 72d3838, soft extrema fixed; runs/scenarios/micro_ohlcv_duel_r1/, 18 GPU-min): still negative.** Seed-mean AUC - logreg_lags: ohlcv14 -0.006 / -0.005 / -0.016 vs close4 -0.004 / -0.010 / -0.007; every |boot z| < 1.5; gross edge -0.05..+0.58 bps per trade. |

## Reading after iteration 2 (2026-09-29)

Across 10 hypotheses, no model variant beats a logistic regression on 3 lagged returns, and that baseline
itself reaches AUC 0.51-0.53. On this data, price-and-volume history alone carries a directional signal
of that size at 10 min - 5 h. The owner's target (stable >60% hit) sits far above anything any variant or
the baseline shows: it would need a different information source or a different target, not a different
model. Next lever inside the owner's list: the P&L-aware target (the owner's point 3), research first.

| E1 | Direction exists on moves larger than the round trip, drowned by the small ones (P&L plan E1, pass lines pre-registered in a9f296f) | configs/scenarios/micro_pnl_e1.yaml: DIR_DEADBAND_BPS 5 vs 26 at horizons 60/120/240, micro layout, 3 seeds; cells under runs/scenarios/micro_pnl_e1/ | ~30 GPU-min | **Negative: signal gate failed.** Seed-mean AUC - logreg_lags: db26 -0.033 / -0.018 / -0.010, db5 -0.046 / -0.020 / -0.008 (gate: >= +0.02 on 2 of 3). The model sits below the linear baseline on the same (masked) labels at every horizon; boot z -0.1 .. -2.0. Trading: -95..-99% net on 1,196-1,568 trades, gross edge -1.2 .. +0.1 bps. Masking untradable moves does not reveal a direction the inputs do not carry. |
| E2 | Training the direction heads on net P&L after costs (owner's point 3; pnl_utility, NT-087) produces a gross edge above cost (pass lines pre-registered in a9f296f; lambdas fixed from init magnitudes before the run, 3ffb863) | configs/scenarios/micro_pnl_e2.yaml on branch nt-087-e2: LAMBDA_PNL 0 / 0.25 / 0.50, horizons 60/120/240, micro layout, SHUFFLE_BUFFER 0, 3 seeds; cells under runs/scenarios/micro_pnl_e2/ | 22.5 GPU-min | **Negative: both gates failed.** The term optimises (train pnl_val rises with lambda: 0 / ~0.10 / ~0.18), but seed-mean AUC - logreg_lags stays negative at every lambda (lam_a -0.039 / -0.020 / -0.011; lam_b -0.039 / -0.026 / -0.008; control -0.046 / -0.020 / -0.008); mean \|2p-1\| 0.08-0.12 with no trend; net -94..-99% on 1,129-1,570 trades, gross edge -0.84..+0.57 bps per trade with no lambda pattern. Putting the P&L into the target does not create an edge the inputs lack. |
