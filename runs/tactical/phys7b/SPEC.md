# phys7b: firming up H75 (physics terms on the variance head), primary pair, 3 seeds (tactical; fixed before any run)

Follow-up of `runs/tactical/phys7/` (SPEC, REPORT): with `PRICE_HEAD: none` the physics terms (t_perp, vacuum overflow alive)
raised the variance head's Spearman with |r| by +0.020 [+0.002, +0.037] (6 slices, 1 seed, 4 epochs), still about 0.05 below
HAR+tod. Branch `nt-tactical-phys7b` from `nt-tactical` (c35261b); pinned worktree `D:/nt/nt_phys7b`; `src/` and `tests/` are
untouched, so the code is that of c35261b; every run uses `PYTHONPATH=D:/nt/nt_phys7b/src`.

## Design (owner rule "shrink at the expense of precision"; the coordinator chose this option)

* Only the PRIMARY pair: `configs/tactical/phys7b_none_on.yaml` and `phys7b_none_off.yaml` = the phys7 configs with seeds [1, 2].
  Same 6 slices (data ends 2017-05-02, 2018-11-16, 2020-03-30, 2021-10-13, 2023-04-29, 2024-11-11), same layout, **4 epochs**.
* **Seed 0 is REUSED** from phys7 (`phys7_none_on`, `phys7_none_off`; same configs, same `src/`, scores in
  `runs/tactical/phys7/scores_phys7_none_{on,off}.json`). New runs: 2 cells x 6 slices x 2 seeds = 24. Total 3 seeds.
* The asked 8 epochs is NOT used: 36 runs at 8 epochs were projected at about 5.3 h wall. **4-epoch nets are under-trained**;
  a converged net could differ. This is a limit of the result, as in phys7.
* Scoring = `analyze_phys7.py` unchanged (Spearman and re-scaled QLIKE on the fold -2 OOS block, HAR+tod on the same blocks,
  scale fitted on the first half). `analyze_phys7b.py` only averages over seeds. Hard cutoff: no bar after 2025-07-20 (asserted
  in `score_slice`; the slices end by 2024-11-11). The test fold is never scored.

## Verdict rule (same as phys7, on the 3-seed means)

Per slice: mean over seeds 0, 1, 2 of the mean-over-horizons Spearman, on minus off (paired by slice and seed). Unit of inference =
slice (6), 95% t-interval (t 2.571).
* HELPS: interval above 0 AND mean difference >= +0.010. HURTS: interval below 0 AND mean <= -0.010. Otherwise NO DETECTABLE
  EFFECT (reported with the interval, not as "no effect").
* QLIKE guard (negative = better, mean over horizons, second half): a HELPS verdict needs the QLIKE difference not worse than +0.05.
* Guard-rails: a non-finite step or a constant variance output (sd < 1e-9) flags the run; its slice drops from the pair and n is stated.
* Reported also (descriptive): net minus HAR+tod (Spearman, QLIKE) for each cell; per-horizon intervals; the seeds-1-2-only and
  seed-0-only differences as a consistency check (not the verdict).

## GPU budget (estimate)

phys7 measured 530 to 710 s per 4-epoch run with 2 processes sharing a busy GPU. 24 runs x about 590 s / 2 processes = about
2.0 h wall (about 4 h summed process time). Stop and report if the projection from measured runs exceeds 4 GPU-hours.
GPU rules: at most 2 processes (2 shards per cell), below-normal priority, `NT_GPU_MEMORY_LIMIT_MB=4000`; the owner's other project
is never touched. Outputs: `runs/tactical/screens/phys7b_none_{on,off}/` (results committed, weights and `preds/` untracked).
