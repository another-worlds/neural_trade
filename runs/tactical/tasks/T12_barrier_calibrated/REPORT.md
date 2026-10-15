# T12 report: the barrier-hit model with monthly isotonic calibration

The lead wrote this from the implementer's hand-back (2026-10-15).
- **SPEC:** [SPEC.md](SPEC.md), committed at 26d9006 before any result.
- **Code and tests:** 5eda8f3 (15 tests passed).
- **Final commit:** 0848a73.

## Verdict on Look A (2017-10..2018-12, untouched for this target): FAIL

Primary cell: a move of 50 bps within 60 minutes.

| Criterion | H3c | Pass? |
|---|---|---|
| 1. Skill over B1 (Brier) | +21.4% [+19.0, +23.7] | PASS |
| 2. Calibration error (ECE) < 1 pp | 3.88 pp [3.17, 4.85] | FAIL |
| 3. Brier no more than 1% worse than H3 | +3.9% worse | FAIL |

- **The monthly calibration makes H3 worse.** H3c's Brier is higher than H3's in all 9 cells, with the CI excluding 0 every time.
- **Why:** an isotonic map fitted on one month of anchors (about 700 at T=60, about 180 at T=240) adds more noise than the bias it removes.

## H3 uncalibrated on the same months (a comparator, not the verdict model)

| Measure (primary cell) | H3 |
|---|---|
| Brier skill over B1 | **+24.4% [+22.2, +26.4]** |
| ECE | 1.74 pp [1.41, 2.66] |
| AUC | 0.874 |

- Skill is positive in all 9 cells, Holm-adjusted p = 0.0009.
- This is the third period where H3 holds, after Bitstamp 2024-25 and Binance 2025-26 (T10). It is also the earliest and least-trained one, with 15-365 days of training history.

## Look B (prospective)

- **What is frozen:** H3 plus the October 2026 isotonic map. Canonical sha256 382555ce...e5d4.
- **When it is scored:** `score_B.py` scores once at least 3 complete months from 2026-10-02 exist.
- **What to expect:** given Look A, the calibrated version is expected to fail.

## Deviations

- **Calibration rows:** "the previous month's H3 predictions" means the real walk-forward predictions of that month's model.
- **Holm:** applied per model, over the 9 cells.
- **Second figure:** it plots the Brier difference instead of the skill ratio, because the ratio explodes when B1's Brier is near 0 (T=240, X=25).
