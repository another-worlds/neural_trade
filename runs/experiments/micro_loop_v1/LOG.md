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
| E3-bar | Triple-barrier labels ("which volatility-scaled barrier is hit first") are more predictable than the close-to-close sign (the P&L plan's model-free bar, run before building E3) | barrier_bar.py: logistic regression on 5 lagged returns + log sigma, fit 2025-01-01..04-30, scored 2025-05-01..07-20 (before the dev block), anchors every 5 bars, H 60/120/240, k 1 and 1.5 (barrier_bar.json) | 1 CPU-min | **Negative.** AUC sign 0.511 / 0.523 / 0.526, barrier k=1 0.513 / 0.507 / 0.507, k=1.5 0.542 / 0.526 / 0.540; every z <= 1.5 (n_eff 140-1,940). The k=1.5 point estimates sit above sign by 0.01-0.03, inside noise; top-10% hit rates (53-66%) rest on n_eff ~14-46 (+-15-26 pp). The E3 model would have to beat ~0.54 with no significant bar to aim at: not built (the lead's call; the plan's "what a negative means" applies). |

## Reading after the P&L plan (2026-09-30)

E1 (cost-sensitive labels), E2 (net-P&L objective) and the E3 model-free bar are negative, on top of the
eleven earlier hypotheses. Every lever inside the current data - horizon, window, training length, inputs
(close vs OHLCV + 14 families), labels, objective, strategy shaping - leaves direction at a linear
baseline's level (AUC 0.51-0.54, never significant). The P&L research note's verdict stands: the owner's
target needs a different information source (E5: taker-buy volume, basis, funding), which is the owner's
decision (VISION "Not in the MVP"; STATUS question 8).

## Screen mode, first GPU measurement (2026-09-30, NT-088 step 3)

configs/screens/example_6h.yaml (bundled CSV, 360-window = 6-hour blocks, 2 epochs, 16 trials: LR {1e-4, 1e-3} x
BATCH {64, 256} x 2 slices x 2 seeds); results runs/screens/example_6h/results.jsonl; 4.3 GPU-min.
- Timings (median of the 15 cache-hit trials): load 0.0001 s, prep 0.011 s, build 0.76 s, train 14.2 s, score
  1.04 s, wall 16.7 s. Trace (epoch_s[0] - median(epoch_s[1:])): median 12.2 s = **73% of a trial** (plan
  threshold for phase 2: 50%). Steady epoch 0.42-0.61 s at batch 256, 1.35-1.43 s at batch 64.
- Rules: 8/16 pass. Every LR 1e-4 trial fails max_clipped_share (0.92-1.0), every LR 1e-3 trial passes. The
  lead's reading: the pre-clip global norm at initialisation is above GRAD_CLIP_NORM 20 (the OHLCV dashboard's
  epoch-1 mean was ~26), and a 2-epoch run at a small LR never leaves that region, so the rule measures the
  start, not an instability. The campaign rule must skip the first epoch's steps (NT-092).

## Level-1 screen campaign (2026-09-30; configs/screens/campaign_l1/, results runs/screens/l1_*/)

960/960 trials (6-hour blocks, 8 epochs, 4 regime slices x 2 seeds, 3 shards), 2 h 45 min wall (cap 4 h),
537 pass / 423 fail. **No trial was non-finite and every train loss fell**: the maths is stable across the
screened space. Every failure is max_clipped_share (> 0.5 of steps clipped after the first epoch), plus 22
max_term_share failures in block B (nll_loss at 0.90-0.95 of the total with every step clipped).
- E maths: EWMA matrix vs scan agree within float drift (val loss 8.725 vs 8.706); scan is 3.4x slower.
- D loss choice: 50/64 pass; failures rise with LAMBDA_PNL (bce 1/2/2/4 of 8 at 0/0.25/0.5/1; focal_dice 0/0/1/2).
- A hyperparameters: 202/264; GRAD_CLIP_NORM 5 fails every trial, 20 fails 3/8, 100 none; LR >= 3e-3 never fails.
- C physics: 212/288; failures spread evenly over ablations, RHO_MAX and the physics weights: no term stands out.
- B loss weights: 61/328; failures track higher LAMBDA_SOFT_ECE (+0.12 vs -0.58 decades), DIR_OUTER, CRPS, VAR,
  VOL, NLL_OUTER; lower EXTENDED_TREND and COHERENCE.
- Seed 1 fails far more often than seed 0 in D, A and C (worst: the 2024-12-05 slice).
Reading: at the default clip norm 20 the pre-clip gradient of a 6-hour block often exceeds the clip; the rule
flags heavy clipping, not divergence. Level 1 cannot rank quality. Level 2 below tests the one lead with a
mechanism: the calibration terms (soft ECE, NLL, CRPS) that QA of NT-087 saw holding P(up) near 0.5.

| L2 | Level-1 survivors / the calibration-term lead give direction on the micro layout (configs/scenarios/micro_l2.yaml; pass line pre-registered in its header, ce1e2ed) | control, clip100, ece_low, calib_low, focal x 3 seeds, horizons 60/120/240; cells under runs/scenarios/micro_l2/ | ~36 GPU-min | **Negative, and sharper: the network is significantly WORSE than logreg_lags at 1 h in every variant** (seed-mean AUC - logreg h0 -0.043..-0.051, pooled Stouffer z -3.3..-3.5; h1 -0.020..-0.025, z -1.7..-2.4; h2 -0.005..-0.010, z -0.5..-0.8). Lowering soft ECE / CRPS / NLL does not free the direction head (ece_low -0.049 / -0.025 / -0.009). logreg_lags on this dev block: AUC 0.532 / 0.525 / 0.510. |

## The screen plan is closed (2026-09-30)

Level 1 (960 trials) found stable maths and one failure mode (heavy clipping); level 2 found no configuration
that recovers direction, and every one below a 3-lag logistic regression. With the P&L plan (E1, E2, E3-bar)
and the eleven earlier hypotheses, 17 lines of attack agree: on this data the network adds nothing over a
linear model of recent returns, whose own edge (AUC <= 0.53) is far below the owner's target. The remaining
lever is information the inputs do not contain (STATUS question 8).

| H6 | Predictability is concentrated in conditions (hour of day, volatility or volume regime, after a large move) rather than spread over all bars | conditional_scan.py: logistic regression (lags, log sigma, volume ratio, hour) fit 2023-2024, scored 2025-01..07-20 per bucket, H 15/60/240; bar fixed in the script: AUC >= 0.56 and z >= 3 at n_eff >= 200, same direction in FIT (conditional_scan.json) | 2 CPU-min | **Negative.** No bucket meets the bar. All bars: AUC 0.516 / 0.516 / 0.521. Hours and regimes 0.49-0.53. Strongest: after a 2-sigma 60-bar move at 15 min, AUC 0.554, hit 54.8% (z 2.2, n_eff 561). |
| H6b | After a large move the next move is predictable, out of sample over years | shock_event_study.py: events \|z60\| >= 2 or 3, rule fitted on 2017-06..2022 (continuation vs reversal), scored 2023-01..2025-07-20; plus a logistic model on the event (shock_event_study.json) | 3 CPU-min | **A real effect, economically empty.** k=2, H=15: reversal, hit **54.2%, z 4.1** out of sample (7,073 events, n_eff 2,358; FIT continuation share 45%), logistic AUC 0.541 (z 3.5). But the rule's gross edge per trade is **-0.39 bps** (wins smaller than losses) and the median 15-bar move after a shock is 18.6 bps < the 26 bps round trip. k=2 H=60: hit 55.2% (z 2.5), edge 0.01 bps. k=3: too few events. The first statistically solid out-of-sample directional effect of the loop, and it carries no money. |
| Q8-probe | Order flow (Binance taker-buy volume) adds directional signal over price (evidence for owner question 8; a measurement, no integration) | taker_flow_probe.py: public data.binance.vision spot 1m klines 2024-01..2025-07 in a scratch folder (D:/nt_data_probe, not in git); logistic price-only vs price + taker imbalance (1/5/15/60 bars) + volume and trade-count ratios; fit 2024, scored 2025-01..07-20; bar fixed in the script: lift >= +0.02 AUC, paired block-bootstrap z >= 3, AUC >= 0.55 (taker_flow_probe.json) | ~40 MB download, 3 CPU-min | **Small, real lift; bar not met.** Lift +0.0097 AUC at 15 min (z 3.2), +0.0124 at 60 min (z 2.3), +0.006 at 4 h (z 0.9); AUC with flow 0.524 / 0.526 / 0.532; hit 51.8-52.6%, top-10% 53.9-55.9%. The first input of the loop with a significant gain, and an order of magnitude short of the owner's target. |
| Q8-probe-2 | USD-M futures signals (basis = premium index level and changes, futures taker imbalance, futures-minus-spot recent return) add signal for spot over price + spot flow | futures_flow_probe.py: public futures/um 1m klines and premiumIndexKlines 2024-01..2025-07 (scratch D:/nt_data_probe, not in git), same fit/score split and bar as the first probe (futures_flow_probe.json) | ~50 MB, 5 CPU-min | **Negative.** Lift over the spot-flow model +0.001 (z 0.95) at 15 min, +0.0001 at 60 min, -0.004 (z -2.7) at 4 h; AUC 0.525 / 0.527 / 0.537. Futures carry nothing the spot price and spot order flow do not. |
| Q8-probe-3 | Order-book depth (Binance USD-M futures bookDepth: cumulative depth at +-1/2/5% of price, ~1-minute snapshots) adds signal over price + spot flow; and the owner's target read directly (hit >= 60% on all bars) | book_depth_probe.py: public futures/um/daily/bookDepth 2025-01-01..07-20 (78 MB, scratch, not in git); fit 2025-01..04, scored 2025-05..07-20; H 1/5/15/60 minutes; bars fixed in the script (book_depth_probe.json) | 78 MB, 5 CPU-min | **Negative.** Lift over price + spot flow +0.003 (z 1.1) at 1 min, 0.000 at 5, -0.006 at 15, -0.015 (z -2.2) at 60. Hit on all bars 51.1-53.0% at every horizon; the top-10% at 1 min reaches 59.5%, on a median move of 2.0 bps. No horizon meets the 60% target. |
