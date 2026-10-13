# T01: forward test of the next-hour volatility-change forecast (H72) on unseen data

Lead, 2026-10-13, committed BEFORE any forward number is computed. Owner: "В этой сессии занимайся тем, что предложил в
конце". REVIEW_2026-10-13.md U1.

## Question
H72 passed its one look (2024-07..2025-07-20): the next-hour change in log realised volatility is predictable. Skill vs
AR +0.163 [+0.147, +0.180], R2 vs random walk +0.205. Does it hold on 2025-10..2026-09, which no study has seen?

## Data
- **Training:** the cut main file `D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv`. The last bar is 2025-07-20 23:59 (`runs/tactical/common/cutoff.py`).
- **Forward:** `D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv` (Binance spot, sha256 b2b28099...a1bba; H82).
- **Never read:** no bar 2025-07-21..2025-09-30. The forward loader asserts it.
- **Different source:** the files come from different feeds (H82). The log-RV features (S1) are scale-free. The volume/range extras (S2, S3) may shift, so S1 is reported beside the primary.

## Method (exactly H72 study 3, `runs/tactical/gaps/SPEC.md`, code `gaps_lib.py` / `study3_vov.py`)
- **Target A:** dA = ln RV(a, a+60) - ln RV(a-60, a). Stride 7. Rows with a zero-RV window are dropped and counted.
- **Features:** S1, S2 and S3 as defined in the gaps SPEC, computed from bars < a of the SAME file. Forward anchors need their
  full feature history inside the forward file (warm-up: the first 1,440 + 180 bars of 2025-10 are skipped).
- **Models:**
  - ridge (alpha 1, standardised) on S3 is the primary, as selected in H72 (`selection3.json`);
  - ridge on S1 is secondary;
  - baselines: `ar` (OLS on L_0 - L_1) and `rw` (dA = 0).
- **Arm F (frozen, the verdict):** every model and baseline is fitted ONCE on the 365 days ending 2025-07-20 23:59 (label end at or before the cutoff), then frozen.
- **Arm R (refit, secondary, not a verdict):** each month M from 2025-11, all models are refitted on the forward file's bars before M (an expanding window from 2025-10-01).
- **Target B** (4-hour vol-of-vol) is reported descriptively only. It failed its threshold in H72.

## Scored months
2025-10 .. 2026-09 (12 complete months) for arm F; 2025-11 .. 2026-09 for arm R.

## Pre-registered criterion (H72's, unchanged)
**Success** iff, for arm F with S3:
1. the mean monthly skill vs `ar` is >= 0.02;
2. its 95% t-interval lower bound is > 0;
3. the mean monthly R2 vs `rw` is > 0.

Otherwise the verdict is "no". If S3 fails and S1 passes, report it as "S1 only (source-robust features)", not as a pass.

## Also reported
- monthly tables (MSE, R2 vs rw, Spearman, skill vs ar) for both arms and every model;
- the forward means next to H72's look values;
- per-month row counts and dropped rows.

## Tests (seconds)
- no bar 2025-07-21..2025-09-30 is read;
- training labels end at or before 2025-07-20 23:59;
- forward features use only bars < a;
- the frozen arm's coefficients are identical across months.

## Outputs
`runs/tactical/tasks/T01_vol_forward/`: code, `results.json`, `tables.txt`, `run.log`, `tests.log`, `REPORT.md` (the verdict
against this SPEC). CPU only.
