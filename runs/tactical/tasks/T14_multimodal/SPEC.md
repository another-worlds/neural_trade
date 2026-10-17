# T14: the multimodal sprint (six views, one stack)

Lead, 2026-10-17, while the owner sleeps. Committed BEFORE any model is run.
Catalogue and install recipes: [research/2026-10-17-multimodal](../../research/2026-10-17-multimodal/README.md).

**Stance (owner, 2026-10-17):** build on what works. Each candidate is a VIEW whose output is added to our working stack,
never a replacement. A miss triggers one documented improvement step, not a closure.

## Candidates (the catalogue's top 6)

| ID | View | Output added to the stack |
|---|---|---|
| M1 | Chronos-2 with covariates | its median and quantile spread for the next hour's log RV / range, with the HAR forecast as a covariate |
| M2 | TTM r2 | its forecast of log RV, then a 50/50 average with HAR |
| M3 | Kronos | range, volatility and P(max abs move ≥ 50 bps within 60 min) from sampled OHLC paths |
| M4 | Regime card | a CNN/CLIP embedding of a rendered chart image, with a templated text caption; the ladder numeric → +image → +caption → +both |
| M5 | VisionTS | a seasonal-profile forecast of range and volume (5-minute bars, period 288) |
| M6 | Regime retrieval | the mean HAR residual of the k = 50 nearest past windows (FAISS), keys = vol percentile, session, stretch, line distance |

## Stack and targets (unchanged code)
- **Target A:** the next-hour change in log RV (T01 definition). Base = T11's V3 (S3 + 96 UTC slots), ridge refitted monthly on the 365 days before each month. The candidate's view is added as extra columns.
- **Target B:** P(max abs move ≥ 50 bps within 60 min) (T10). Base = H3, fitted the same way. The candidate's view is added to H3's logistic features.
- Anchors are hourly, on the UTC grid. The hourly grid is a thinning of T01/T10's stride-7 anchors, so the bases are re-run on it.
- Time of day is in UTC via `runs/tactical/common/timefix.py`, with seam anchors masked.

## Periods

| Role | Period | Data | Note |
|---|---|---|---|
| Hyper-parameters of the fusion head only | 2019-01..2024-06 | cut file | The pretrained models may have seen these years. This is stated, and nothing is judged here. |
| Look 1 | 2024-07..2025-07-20 | cut file | Kronos' clean window (its pretraining ends 2024-06) |
| Look 2 | 2025-11..2026-09 | Binance forward file | Clean for every pretrained model; after all knowledge cut-offs |
| Prospective | months from 2026-10 | forward file | Each passing candidate is frozen with a sha256 |

## Controls (required for every candidate)
- **C1, random features:** same number of columns of permuted view values (within month).
- **C2, information-equivalent numeric control:**
  - for images: the S3 numeric features of the same window;
  - for retrieval: kNN on the same keys with shuffled residuals.
- **Honest baselines:** V3 (A) and H3 (B), plus a 50/50 average of V3 with the view's own point forecast.

## Pre-registered criterion, per candidate, on BOTH looks
- **A:** increment over V3 ≥ +0.005 in monthly skill vs AR, with the monthly t-interval above 0, and above C1 and C2.
- **B:** relative Brier improvement over H3 ≥ 1%, with a day-block bootstrap CI above 0, and above C1 and C2.
- **Holm over the 6 candidates × 2 targets.** A candidate "passes" if it passes either target on both looks.

## Improvement step (instead of closing)
A candidate that misses gets ONE documented variant before the report (for example, fine-tuning on the first 5% of the
months, a different context length, or the view entering the residual instead of the level). It is run on the same
looks, flagged as post-hoc, and reported next to the pre-registered result.

## Install rules
- Only in `D:/nt/nt_mm_env` (Python 3.11) and `D:/nt/nt_mm_legacy` (Python 3.10). Weights and caches go in `D:/nt/nt_mm_cache`.
- The `nt` conda env is untouched.
- The GPU is shared: at most 2 heavy processes at once, and `TF`/`torch` memory growth is on.
- No data is downloaded except model weights and the packages.

## Never
- No bar 2025-07-21..2025-09-30.
- No October-2026 target.
- No new data source.
- Nothing is chosen on a look.

## Outputs
`runs/tactical/tasks/T14_multimodal/`:
- an install log with versions and sha256 of weights;
- one folder per candidate (`M1_chronos2/` etc.) with code, `results.json`, `tables.txt`, figures and tests;
- `smoke.md` (what installed and ran);
- the report text goes to the lead.
