# O7 report: geometry-enhanced volatility forecasts as 90% return intervals

SPEC: `SPEC.md` (0741db4). Code and results: 58c5b3e. CPU only, 238 s. Tests: `tests.log` (13 passed). Tables: `tables.txt`;
figures `cover_width_T{15,60,240}.png` (+ plotly `.json`). Written by the outlook lead from the implementer's text (its Write
tool refused the file); the criterion line checked against `tables.txt` lines 90-94.

## Verdict: **pass** (tiny, and not better than HAR)
The pre-registered criterion compares L1 with R0 at T = 60 over the judgement span 2019-02..2020-12 (23 months).
- **Interval score:** L1 - R0 is -1.75 bps on average, 95% t-interval [-2.64, -0.86], with L1 better in 18 of 23 months (15 needed). Met.
- **Coverage:** L1's pooled coverage is 0.8925, inside [0.88, 0.92]. Met.

**Lead's reading.** The gain is real by the rule but small:
- **Size:** -1.75 bps on an interval score of about 307 bps (0.6%).
- **Source of the gain:** the width does not change, so it comes from fewer and smaller misses at the same coverage.
- **Against HAR:** L1 is NOT better than T03's HAR form on these months. At T = 60 the difference is +0.65 bps [-3.97, +5.28], and 2020-03 alone favours HAR.
- **Conclusion:** the ranking gain of O1/O5 survives as an interval gain over S3, but it is too small to make L1 a better risk model than HAR on this span. On the later, spent span (2021-02..2024-06), L1 beats HAR in 35/41 months at T = 60 (-2.32 bps [-3.20, -1.44]), which is descriptive only.

| T = 60 (judgement) | pooled coverage | mean width bps | mean interval score bps |
|---|---|---|---|
| L1 | 0.8925 | 169.47 | 305.77 |
| R0 | 0.8918 | 169.46 | 307.52 |
| HAR | 0.8919 | 173.76 | 305.11 |

**L1 - R0 at the other horizons** (Holm over the two):
- T = 15: -1.84 bps [-2.22, -1.46], 23/23 months better.
- T = 240: +0.13 [-2.10, +2.36], 11/23 months better. No effect.

**L1 - HAR** (Holm over the three horizons):

| T | difference | 95% interval | months L1 better | Holm p |
|---|---|---|---|---|
| 15 | -1.49 | [-3.06, +0.09] | 22/23 | 0.19 |
| 60 | +0.65 | [-3.97, +5.28] | 16/23 | 1.0 |
| 240 | -1.27 | [-13.30, +10.75] | 15/23 | 1.0 |

**Descriptive spans, L1 - R0** (no verdict):

| span | T = 15 | T = 60 | T = 240 |
|---|---|---|---|
| 2021-02..2024-06 | -0.91 bps, 38/41 | -0.76 bps, 28/41 | -1.28 bps, 26/41 |
| 2024-08..2025-07 | -0.75 bps, 12/12 | -1.35 bps, 10/12 | -2.30 bps, 9/12 |

Every forecaster's pooled coverage on these spans is 0.896 to 0.900.

## Caveats
- **Months already used:** the judgement months are the ones O5 used for L1's ranking, so this is a further look at them.
- **Optimistic intervals:** the monthly differences are correlated (overlapping labels and volatility regimes), which makes the t-intervals optimistic.
- **Coverage in single months:** monthly coverage ranges from 0.80 to 0.95 for every forecaster. The worst months are 2020-03, 2020-11 and 2019-05.
- **Prospective test pending:** `score_B.py` will score O5's frozen L1 against L0 (not R0). Calibration uses 2026-10, and the scored months are 2026-11..2027-01, scored once. This is the only unspent check.
- **No QA agent:** the verdict is tiny and secondary to O5's QA'd result, and the lead checked the criterion line.

## Definitions and deviations
1. r_T = ln c[a+T-1] - ln c[a-1]. This is the same window as the RV target, and a test checks it.
2. Month M is scored with the fit made for month M-1, the same fit that gives sigma_hat on the calibration rows. So the forecaster is one month stale. The variant with the fit for month M was not run.
3. Calibration rows also need their labels to end by the start of M (T03's rule).
4. HAR is a ridge (alpha 1) on the same `ok` rows, with T03's features and target ln(|r| + 1e-5).
5. Prospective calibration is rolling by default; `--calib fixed` is also available.
6. The frame is O6's cache. The signed returns and the HAR block are in D:/nt/nt_scratch/o7/extra.npz, and their anchors are asserted to match the frame.
7. The PNGs are drawn with matplotlib in the theme colours, because kaleido is not installed. Plotly JSON files of the same panels sit next to them.
