# O2 report: physics terms on the variance head at 8 epochs (tactical; verdict against SPEC.md)

Code pinned: worktree `D:/nt/nt_o2` at 287bbc6 (SPEC commit) + the two `configs/tactical/o2_none_*.yaml` copied in; `PYTHONPATH=D:/nt/nt_o2/src`.
Results were written with an absolute `--store D:/nt/nt_tactical_outlook/runs/tactical`, so they are in the outlook worktree's
`runs/tactical/screens/o2_none_{on,off}/`. Config diff against phys7b (line-ending-insensitive): `config_diff.txt` (name, seeds [0, 1], epochs 8; nothing else).
Setup: BTC/USDT 1-minute static excerpt, 6 climb slices (data ends 2017-05-02 .. 2024-11-11), fold -2, 126,000 windows, `PRICE_HEAD: none`,
physics on vs the six lambdas 0, 2 seeds, 8 epochs, calibration off. Scoring = `analyze_phys7.py` unchanged (OOS block of fold -2, cutoff assert kept; test fold never scored).

## Verdict (SPEC rule, 6 slices, 95% t, 2-seed means)
**NO DETECTABLE EFFECT** at 8 epochs: Spearman on minus off = **+0.0094 [-0.0068, +0.0257]**, sd 0.0155, 3 of 6 slices up. The interval contains 0 and
the mean is below the +0.010 bar. (QLIKE on-off -0.085 [-0.276, +0.106], negative = better, not significant; the guard does not matter without HELPS.)

**Epoch question: d8 - d4 = -0.0062 [-0.0238, +0.0114]** (paired over slices, same seeds 0, 1): "same within noise" (point estimate shrinks: 4-epoch effect
+0.0157 [+0.0031, +0.0283] on these seeds/slices, 5 of 6 slices up, falls to +0.0094 at 8 epochs). The H77 effect neither grows nor is distinguishably gone.
Reading: lean "shrinks slightly", not established. d8 - d4 per slice: +0.0018, -0.0305, -0.0050, -0.0008, +0.0169, -0.0199.

## On - off at 8 epochs (Spearman with |r|, mean over seeds 0, 1; 4-epoch values on the same seeds/slices in brackets)
| | mean | 95% interval | slices up |
|---|---|---|---|
| mean over horizons | +0.0094 [+0.0157] | [-0.0068, +0.0257] ([+0.0031, +0.0283]) | 3/6 (5/6) |
| h0 (10) | +0.0056 [+0.0140] | [-0.0122, +0.0234] | |
| h1 (15) | +0.0118 [+0.0117] | [-0.0068, +0.0304] | |
| h2 (20) | +0.0109 [+0.0213] | [-0.0072, +0.0290] | |
By seed (8 ep): seed 0 +0.0097 [-0.0112, +0.0306]; seed 1 +0.0092 [-0.0078, +0.0262]. Per slice (8 ep): +0.0276, -0.0002, -0.0078, +0.0164, +0.0250, -0.0043.

## Net minus HAR+tod (Spearman; mean over slices, interval over slices)
| cell | 8 epochs | 4 epochs | change 8 vs 4 |
|---|---|---|---|
| physics on | -0.0631 [-0.1084, -0.0178] | -0.0552 | -0.0078 [-0.0199, +0.0042] |
| physics off | -0.0725 [-0.1097, -0.0353] | -0.0709 | -0.0016 [-0.0125, +0.0093] |
Net means (Spearman) 8 ep: on 0.2209, off 0.2115 (4 ep: 0.2287, 0.2131). More training does NOT close the HAR gap (it widens slightly, within noise); the net stays about
0.06-0.07 below HAR+tod. QLIKE net minus HAR (positive = worse): on +0.153 (4 ep +0.164), off +0.237 (4 ep +0.193); changes not significant.
Full numbers: `report_tables.txt`, `report_tables.json`, `scores_o2_none_{on,off}.json`.

## Run facts and guard-rails
- 24 runs, all `passed`, finite, 0 non-finite grad steps, no constant-variance output; `epochs_run` = 8 in all 24 (no early stop). n = 6 slices, nothing dropped.
  The served (best-validation) epoch is not recorded in the screen results, so it is not reported.
- GPU wall: launch 19:56:43, cell on done 21:12:52 (76 min), cell off done 22:19:44 (67 min): **2 h 23 min wall** for 24 runs (budget 3.1 h, stop line 5 h);
  summed process time 4.71 h (on 2.50 h, off 2.21 h); ~11-15 min per run, 2 processes, BelowNormal, `NT_GPU_MEMORY_LIMIT_MB=4000`. The GPU was free at launch; no other heavy process seen.
- Deviations: none from the SPEC. (4-epoch comparison uses seed 0 from phys7 and seed 1 from phys7b, both with the same slices; the d8 - d4 pairs match by slice.)
  Caveat from SPEC: 2 seeds and 8 epochs, not 3 seeds / full length; the interval is wide (sd 0.0155 over 6 slices).

## What it means
H75/H77's +0.012 to +0.016 physics effect on the variance head's rank correlation does not grow with training; at 8 epochs it is smaller and not distinguishable from 0.
It is not a positive outlook for physics on the variance axis, and not evidence of harm either (QLIKE on-off leans better with physics, n.s.). A decisive answer would need more
slices or seeds; the HAR gap is the larger issue.
