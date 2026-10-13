# T06 report: an ensemble of the best direction models

Lead, written from the implementer's hand-back (2026-10-13).
- Owner request: "Составь из них ансамбль и проверь его точность".
- SPEC: [SPEC.md](SPEC.md) (f3ff5bc, committed before results).
- Code and results: bed99f3; tests acca180, 10 passed.

## Verdict: criterion met by a hair on the history; no gain in today's market

| Where | E1 (mean of P) minus the best single model (regression L0) | Slices or months positive |
|---|---|---|
| 24 lab slices (2017-2025, the criterion) | **+0.0052 [+0.0011, +0.0092]** (threshold +0.005) | 18/24 |
| 6 FINAL slices (confirmation: same sign) | +0.0039 [-0.0028, +0.0106] | 5/6 |
| **Walk-forward 2024-07..2025-07 (today's market)** | **+0.0014 [-0.0022, +0.0050]**; absolute AUC 0.521 vs 0.520 | 9/13 |

**Reading.**
- The ensemble adds about +0.005 AUC (0.570 → 0.575) on the historical slices.
- The gain sits in the 2017-2021 slices. Three of the last four lab slices are negative.
- In 2024-25 every member and the ensemble are at about 0.52 AUC, with log loss at the constant and a top-10% hit of 54-55%.
- An ensemble cannot add information the members lack: 0.52 stays 0.52.
- Nothing here approaches the owner's goal: hit > 60% today.

## Members (6 used, all reproduced exactly; 2 not run)

| Member | AUC on the 24 slices |
|---|---|
| L0 regression tb7 | 0.5701 |
| L5 rich-43 linear | 0.5683 |
| L6 + MLP | 0.5674 |
| L7 + GRU | 0.5671 |
| R1y regression 1 y | 0.5700 |
| P1y PatchTST-lite 1 y | 0.5698 |

- The two 3-year PatchTST seed ensembles were not run. They saved no predictions, and re-running them would take about 5 GPU-hours per seed.
- Logit correlation:
  - within the lab group: 0.80-0.97;
  - within the 1-year group: 0.95;
  - across the two groups: 0.55-0.64. That cross-group gap is the only diversity the ensemble uses.

## The three ensembles on the 24 slices

| Ensemble | AUC | diff vs L0 | Log loss vs constant | Top-10% hit | Top-10% bps |
|---|---|---|---|---|---|
| L0 | 0.5701 | - | -0.0069 | 0.605 | +2.95 |
| **E1 mean P** | 0.5752 | +0.0052 | -0.0093 | 0.616 | +3.40 |
| E2 mean logit | 0.5751 | +0.0051 | -0.0090 | 0.616 | +3.42 |
| E3 stacker | 0.5646 | -0.0054 | +0.0219 (worse than constant) | 0.605 | +4.01 |

The stacker's tail predictions come from the members' inner models, a different model from the final one, so it is poorly calibrated.

## Deviations

1. The two 3-year PatchTST ensembles were dropped (budget).
2. The members have different training spans. The stacker and threshold tail is the last 15% of the lab block for every member; the 1-year networks early-stopped on a span containing it (mild overlap).
3. The series cache was rebuilt from the cut file and verified against the lab cache (0 mismatches).
4. The walk-forward uses a 20-minute label embargo; L0 matches H59 to 0.0003.
5. A tail-alignment bug on the first slice was found in the analysis, fixed, and that slice re-run (not a choice made on results).

Predictions are out of git, in D:/nt/nt_t06_preds.
