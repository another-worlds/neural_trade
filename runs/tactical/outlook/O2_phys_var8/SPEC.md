# O2: physics terms on the variance head with longer training (outlook prospect 3; tactical, fixed before any run)

Outlook lead, 2026-10-13. OUTLOOK_positive.md prospect 3: "H77 ran 4 epochs. Try a full-length training ... with and without
physics, judged on variance Spearman and QLIKE against HAR". Owner rules: long GPU studies are shrunk at the expense of
precision (TACTICAL.md, 2026-10-11); the physics terms stay (D-003); no default changes.

## Question
H75/H77: with `PRICE_HEAD: none` (alive: t_perp, vacuum overflow, HD), the physics terms raised the variance head's Spearman
with |r| by +0.0116 [+0.0031, +0.0202] (6 slices x 3 seeds, **4 epochs**; the nets were under-trained, 115 s per epoch). Does
the effect grow, hold, or vanish when the same nets train **8 epochs**? A growing effect is the positive outlook for physics
on its natural axis; a vanishing one says H77 measured a faster start, not a better variance model.

## Design (shrunk; the precision loss stated)
- Cells: `configs/tactical/o2_none_on.yaml`, `o2_none_off.yaml` = the phys7b configs (`phys7b_none_{on,off}.yaml`) with
  `seeds: [0, 1]` and `run.epochs: 8`; everything else identical (hc4 long-block layout on the static excerpt, 126,000 windows,
  fold -2, batch 256, horizons 10/15/20, calibration off, `save_predictions`, `DATA_TAIL_TRIM`; physics off = the six lambdas 0).
- Slices: the 6 climb slices of hc4/phys7 (data ends 2017-05-02 .. 2024-11-11; all before the cutoff 2025-07-20).
- Runs: 2 cells x 6 slices x 2 seeds = 24. Code: `nt-tactical-outlook` at the SPEC commit, in a pinned worktree
  `D:/nt/nt_o2` with `PYTHONPATH=D:/nt/nt_o2/src`. `src/` is not edited.
- Precision: 2 seeds, not 3; 8 epochs, not "full length" (early stopping as the default config does; the served weights are the
  best-validation epoch, D-011).

## Scoring (unchanged code)
`runs/tactical/phys7/analyze_phys7.py` scoring per run (Spearman of the head's variance with |r_h| on the fold -2 OOS block;
QLIKE on its second half with the scale from the first half; HAR+tod on the same blocks), averaged over seeds per slice by an
`analyze_o2.py` that imports it. The cutoff assert stays. The test fold is never scored.

## Verdict rule (fixed now; the phys7 rule on the 2-seed means)
Per slice: mean over seeds 0, 1 of the mean-over-horizons Spearman, physics on minus off. Unit = slice (6), 95% t (2.571).
- HELPS: interval above 0 AND mean >= +0.010 (QLIKE difference not worse than +0.05).
- HURTS: interval below 0 AND mean <= -0.010. Otherwise NO DETECTABLE EFFECT.
- **Epoch question (the point of O2):** the 8-epoch effect is compared with the 4-epoch effect on the SAME seeds 0, 1 and slices
  (H75/H77 runs: `scores_phys7_none_*.json` seed 0, `scores_phys7b_none_*.json` seed 1). Reported as d8 - d4 with a paired
  t-interval over slices: "grows" if the interval is above 0, "shrinks" if below 0, else "same within noise".
- Descriptive: net minus HAR+tod (Spearman, QLIKE) per cell at 8 epochs against 4 epochs (does training close the HAR gap?);
  per-horizon intervals; epochs actually run and the served epoch per run.
- Guard-rails: a non-finite step or a constant variance output (sd < 1e-9) flags the run; its slice drops from the pair and n is
  stated.

## GPU budget (estimate) and rules
phys7b measured about 115 s per epoch with 2 processes. 24 runs x 8 epochs x ~115 s / 2 processes = about **3.1 h wall**. Stop
and report if the projection from the first measured runs exceeds 5 h wall. At most 2 processes (2 shards, one cell at a time),
below-normal priority, `NT_GPU_MEMORY_LIMIT_MB=4000`, `TF_FORCE_GPU_ALLOW_GROWTH`; together with the volatility session at most 2
heavy GPU processes; the owner's other project is never touched (RUNBOOK "GPU rules"). Each run exceeds the 2-minute rule: this is
the owner's "shrink, don't skip" exception (as H75/H77), stated here.

## Outputs
`runs/tactical/screens/o2_none_{on,off}/` (results committed; weights and `preds/` untracked), `runs/tactical/outlook/O2_phys_var8/`
(make_o2.py, run_o2.ps1, analyze_o2.py, scores_*.json, report_tables.txt, progress.txt, REPORT.md with the verdict against this SPEC).
