# Is it the number of training steps, not the block length? (tactical, fixed before any run)

Owner, 2026-10-07: "check the hypothesis with the epochs".

**Observation.** The variance head of the default network learns at 7 days and not at 1 day (same 6 slices x 2 seeds):
90% coverage 0.87 vs 0.67, CRPSS +0.008 vs -0.216, Spearman variance~error^2 0.24 vs 0.13. The 7-day run makes
40 steps x 8 epochs = 320 updates; the 1-day run (bench A, batch 256) only 6 x 8 = 48.

**Hypothesis.** The gap is mostly the number of updates, not the amount of data: a 1-day block trained for about as
many updates as a 7-day one reaches 7-day quality on the variance head.

**Arms** (1-day block: MAX_SEQUENCE_COUNT 18,000, 1,440 train windows; the C2 slices x seeds 0-1; all 9 heads):
- `ep1d_bs256_e40`: batch 256, 40 epochs = 240 updates.
- `ep1d_bs64_e14`: batch 64, 14 epochs = 23 x 14 = 322 updates.
References, already run: bench A (1 day, batch 256, 8 epochs, 48 updates) and C2 default (7 days, 320 updates).

**Rules** (means over h0-h2 and the 12 runs):
- **Supported** if either arm reaches 90% coverage >= 0.80 AND CRPSS >= 0.
- **Refuted** if both arms stay at coverage < 0.75 or CRPSS < -0.10.
- Otherwise **partial**. Direction AUC and the delta head are reported, not judged.
- The screen path has no early stopping: 40 epochs on 1,440 windows may overfit; that is part of what is tested.

**Limits.** Runs start after the throughput benchmark ends (it measures speed and must not share the GPU with us).
Expected about 2-2.5 minutes per run (40 epochs x ~2.5 s), slightly over the 2-minute rule; the owner asked for this check.
