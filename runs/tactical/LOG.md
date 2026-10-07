# Tactical session journal (D-062)

Rules: [docs/TACTICAL.md](../../docs/TACTICAL.md). Every row is tactical, exploratory: one seed, dev folds
only, with its noise level. Earlier work this continues from: `runs/experiments/micro_loop_v1/LOG.md`.

## Handoff

_Rewritten at the end of every tactical session._ 2026-10-06: round 1 (3 direction-head switches) is closed with no effect. GPU hours used today: 0. Next: the owner names the goal, or the session starts from where the micro loop
stopped (H4b: re-run the LOOKBACK 240 cell with more patience; direction is the bottleneck at every horizon).

## Hypotheses

| # | Date | Hypothesis | Method | Cost | Result | Evidence |
|---|---|---|---|---|---|---|
| H5 | 2026-10-06 | R1: the direction head is spoiled by the high-capacity path; constrain it | hc_baseline (defaults, 100 trials) vs 3 switches (cc85a5c), same 5 slices x 20 seeds, screen layout, paired by (slice, seed), CI over slices | ~300 trials, ~27 s each | **No effect.** skip_only -0.012 [-0.101, +0.076]; shrink1 -0.010 [-0.037, +0.017] (83 of 100 trials: shard 2 segfaulted 3 times at 16/34; accepted as final by the owner); dropout 0.5 -0.015 [-0.029, -0.0001]. None beats the baseline (0.522); logreg_lags 0.554 is within noise of it. | runs/tactical/screens/hc_*, hc_logreg.json, configs/tactical/ |
| H6 | 2026-10-07 | R2/R3 (design v2: 40 climb slices x 3 seeds, paired, CI over slices; stage-1 gate on 8 slices) | base 0.5296; skip_only +0.0075 [-0.007,+0.022]; 3 epochs -0.0088 [-0.019,+0.001]; LR 3e-4 -0.0122 [-0.022,-0.002] worse; LAMBDA_DIR x5 -0.0031 [-0.014,+0.008]; LOOKBACK 20 -0.0253 [-0.042,-0.009] worse; no-physics dropped at stage 1 (24 trials, -0.017); calibration value -0.0013 [-0.008,+0.005]; calibration gradient -0.0054 [-0.012,+0.001] | ~1000 trials | **Nothing passes the rule** (CI above 0 and diff >= +0.01). Two variants are worse. No Config switch moves direction AUC. 459 of 975 trials took over 120 s (median 118 s, 3 shards in parallel): the 2-minute rule was not kept. | runs/tactical/screens/hc2_*, runs/tactical/hc2_compare.py |
| H7 | 2026-10-07 | Why every configuration scores high on the slice DATA_END 2022-04-19 (base mean 0.672; single heads up to 0.805) | slice_2022_04_19.py: block bounds, price path, model-free k-bar momentum/fade rules on the same val block and label mask; news search | CPU, minutes | **A regime, not skill.** The val block is Sat 2022-04-16 11:01-18:30 UTC (Easter weekend, US markets shut, BTC ~40,450 in a 0.52% range, thin liquidity; Cointelegraph, crypto.news, FS Insight). Fading the last 10 bars scores AUC 0.756 there (10-bar return autocorr -0.32), above the network. Contrast: 2024-07-09 (trend) momentum wins (0.605), network 0.446; 2022-10-23 fade wins (0.65) but the network is 0.44. Per-run AUC on ~24 independent points: pure noise reaches 0.8 in ~1%. | runs/tactical/slice_2022_04_19.json |
| H8 | 2026-10-07 | The 10 best-AUC configs of the old campaign hold up at the 7-day scale (owner: 'on the off chance') | make_hc3.py: 7-day blocks (10,080 train windows), 4 old slices x seed 0, vs default; owner exception to the 2-minute rule | ~8 min per run, 3 at a time | **Stopped by the owner after 3 configs** (one candidate instead). Default 0.537; cand01 0.537; cand02 0.532; cand03-05 one slice each (0.57-0.60, not comparable). At 7 days the val block has ~617 independent points per head (noise ~0.02 instead of ~0.10). | runs/tactical/screens/hc3_*, hc3_candidates.json |
| H9 | 2026-10-07 | The 0.805 candidate (skip_only, 2022-04-19, seed 2, h1) is real (SPEC runs/tactical/cand_0805/SPEC.md, fixed before the runs) | C1: same block, seeds 3-12, screen layout; C2: 7-day blocks, 6 slices x 2 seeds, vs default; all 9 heads measured | C1 20 runs ~1 min; C2 23 of 24 runs ~7 min (the last stopped by the owner) | **Closed: not real.** C1 fails all three parts: skip_only h1 mean 0.415 (rule >= 0.75), vs base -0.199 [-0.270, -0.129] (base 0.667 on that block). C2 fails (a): -0.008 [-0.032, +0.017] over 6 slices; guard-rails pass (CRPSS +0.010, coverage +0.006). **Found on the way:** at 7 days the variance head works (CRPSS +0.001..+0.024, 90% coverage 0.865-0.883, Spearman var~err^2 0.20-0.25), at 6 hours it does not (CRPSS -0.155, coverage 0.56); the delta head is below zero skill at both scales; direction AUC 0.52-0.55 at 7 days. | runs/tactical/screens/cand_c*, preds/*.npz |

## For the MVP lead

Findings worth adopting, with evidence; each becomes an MVP backlog item and a paired test (D-025) before
any default changes.

- Measurement, not a model finding: one slice's direction AUC has an error of about 0.06-0.12 (n_eff 20-40), the 5-slice mean about 0.03; seed sd 0.04-0.05 inside a slice. A 0.02-0.03 effect needs far more independent slices (30-50) or longer validation blocks. logreg_lags on the same blocks is 0.554 (0.42-0.67 per slice): not evidence of skill.
