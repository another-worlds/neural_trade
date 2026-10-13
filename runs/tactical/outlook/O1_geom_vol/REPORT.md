# O1 report: does indicator geometry add to a fair volatility forecast?

Study per SPEC.md (8d9d20c); code and results 6cb5617; CPU only (26 min). Numbers: `tables.txt`; rows: `results.jsonl` (2,278);
log: `run.log`; tests: `tests.log` (8 passed). Written by the outlook lead from the implementer's text (its Write tool refused the
file); the criterion lines were checked against `tables.txt` lines 253-256.

## Verdict: **holds** (on the SPEC's linear base, narrowly)
The SPEC criterion compares G* against R0 at H = 60, on the primary target ln RV. The unit is the month, with 95% t-intervals over months.

| span | months | Spearman gain (mean [95% CI], months > 0) | QLIKE diff (negative = better) | pass |
|---|---|---|---|---|
| selection 2021-01..2024-06 | 42 | +0.0073 [+0.0057, +0.0089], 40/42 | -0.0107 [-0.0159, -0.0055] | yes |
| look 2024-07..2025-07 | 13 | +0.0054 [+0.0032, +0.0076], 12/13 | -0.0083 [-0.0118, -0.0048] | yes |
| forward 2025-10..2026-09 | 12 | +0.0052 [+0.0029, +0.0075], 12/12 | -0.0077 [-0.0126, -0.0028] | yes |

- **Verdict rule:** "holds" means it passes on the look AND on the forward.
- **Sanity gate:** Rsh - R0 is -0.0013 on the selection months at H = 60, inside the +-0.003 limit. PASS.
- **Choice of G*:** on selection, R1 gains +0.0039 and R2 +0.0073. R2 - R1 = +0.0034 clears the +0.002 bar, so G* = R2 (S3 + G1 + G15).

**Lead's reading.** It passes, but the result is small and depends on the linear base:
- **Size:** about 0.5 Spearman points on a base of 0.73-0.80. Look and forward clear the +0.005 bar by only 0.0004 and 0.0002.
- **Consistency:** almost every month is positive.
- **Nonlinear base:** boosted S3 (H0) beats the linear R0 by +0.020 on the forward. Against H0, geometry's gain does not hold: H2 - H0 is +0.0045 on the look and -0.0030 [-0.0060, +0.0001] on the forward.

So the claim is "geometry adds to a linear HAR-type forecast". It is not "geometry adds to the best available forecast". The fair next test is O5.

## Key table, H = 60 (mean Spearman | mean QLIKE, primary target)
| span | R0 | R1 | R2 | Rsh | HGB S3 (H0) | HGB S3+G (H2) |
|---|---|---|---|---|---|---|
| selection | 0.7307 \| 0.3332 | 0.7347 \| 0.3290 | 0.7380 \| 0.3225 | 0.7294 \| 0.3340 | 0.7225 \| 0.3452 | 0.7273 \| 0.3388 |
| look | 0.7988 \| 0.2542 | 0.8016 \| 0.2504 | 0.8042 \| 0.2459 | 0.7979 \| 0.2550 | 0.7969 \| 0.2533 | 0.8014 \| 0.2502 |
| forward | 0.7761 \| 0.2884 | 0.7805 \| 0.2827 | 0.7813 \| 0.2807 | 0.7751 \| 0.2894 | 0.7964 \| 0.2710 | 0.7935 \| 0.2713 |

- **R1 alone (1-minute geometry):** +0.0039 / +0.0029 / +0.0044 (selection / look / forward).
- **15-minute geometry (new):** it lifts R2 above the bar.
- **Rsh:** it sits slightly below R0 on every span, so the gain is not an effect of adding more features.

## Secondary (not part of the verdict)
- **H = 15:** G* - R0 is +0.0143 / +0.0094 / +0.0081, every month positive, Holm p < 0.0001.
- **H = 240:** nothing (+0.0012 / -0.0015 / +0.0012; Holm p 0.40 / 0.34 / 0.28).
- **Target ln(|r_H| + 1e-5), H = 60:** no gain. Geometry helps predict realised volatility, not the size of the net move (consistent with H47 and H57).
- **HGB pair, H2 - H0:** +0.0048 on selection, +0.0045 [+0.0015, +0.0076] on the look, -0.0030 [-0.0060, +0.0001] on the forward.

## Interpretation
Top standardised coefficients of the frozen G* (fit on the 365 days ending 2025-07-20, H = 60):

| feature | coefficient |
|---|---|
| g15.ema20.slope3 | -0.544 |
| g15.ema20.pos | +0.368 |
| g15.ema40.slope3 | +0.331 |
| g1.ema10.slope10 | +0.315 |
| g1.ema20.slope3 | -0.246 |
| g15.sma20.pos | -0.228 |
| g15.ema10.slope10 | +0.215 |
| g15.atr14.slope3 | +0.177 |

- **Channel mix:** 13 of the top 15 are moving-average (price-band) channels. 9 of 15 are on the 15-minute scale. None are oscillators or volume.
- **How to read them:** the opposite-sign pairs are contrasts inside the MA family, roughly a curvature of the 15-minute trend, not single effects.
- **Each group added alone to S3** (G1 columns, selection, H = 60):

  | group | Spearman gain |
  |---|---|
  | price bands | +0.0036 [+0.0024, +0.0049] |
  | volatility channels | +0.0012 |
  | oscillators | +0.0009 |
  | volume | +0.0000 |

- **For the figures:** `interp.json` holds the top 15 and six real selection-span windows for each of the top 3 features.

## Caveats
- **Narrow pass:** see the verdict.
- **Linear base:** see the verdict. The SPEC tested against the linear S3.
- **Spent forward months:** T01 and T03 scored 2025-10..2026-09 for the HAR level. No geometry model was scored on them.
- **Different feed:** the forward bars are Binance spot, so volume levels may differ. Volume channels add about 0 on their own.
- **Partial first month:** October 2025 is partial (6,041 rows at H = 60).
- **Dropped rows:** rows with a zero-RV window are dropped.
- **Model ageing:** each forward month's gain also contains the ageing of the one frozen fit. It is still positive in 12/12.

## Deviations and notes
1. **QLIKE form for ln RV:** QLIKE uses the SPEC's v / (c e^mu), with c = mean(v / e^mu) on the hold rows. The first draft used exp(2 mu), was caught by the lead's note, and all results were recomputed. A test pins this.
2. **Ridge fits:** the scoring fit uses all 365 days. The scale c comes from a second fit on the first 85% of the rows, with a gap of ceil(H/7) rows. HGB is fit once, on the first 85% strided by 10.
3. **Ridge solver:** normal equations on standardised columns, tested equal to sklearn Ridge(alpha=1). Rsh permutes G1 and G15 together, independently in the train and test rows.
4. **Spearman and Holm:** Spearman is computed against realised RV. Holm uses two-sided t tests over months.
5. **Channel groups:** they follow the code kinds. The group runs use G1 columns only.
6. **Threads:** BLAS was limited to 4 threads. Without the limit, the run took 90-185 s per month at below-normal priority.

## Runtime
Features 110 s; walk-forward 22-95 s per month; total 1,579 s. The feature cache in D:/nt/nt_scratch/o1/ and `frozen_coefs_h60.npz` are local and not in git.

## QA (2026-10-13, qa agent, read-only at ac6a87e; scratch D:/nt/nt_qa_o1_scratch/)
**Verdict holds as stated: yes; all five checks PASS.**
1. **Causality and leakage.**
   - Every feature at anchor a uses bars < a, and the targets start at bar a.
   - The 15-minute groups end at a-1, which a pandas resample confirms.
   - A probe that overwrote every bar from a onward left all features unchanged.
   - Training labels end before each month (frozen fit: by 2025-07-20 20:58).
   - No forbidden bar exists in either file.
2. **Independent recompute.** The QA's own sklearn ridge reproduced every look and forward month at H = 60 exactly (|dSpearman| 0, |dQLIKE| 2e-14).
3. **Selection.** G* was chosen on the selection span only, and the SPEC predates the run.
4. **QLIKE.** QLIKE uses the SPEC form.
5. **Robustness** (information only):
   - **Forward gain decays over the year:** about +0.009 in 2025-10..2026-02, about +0.0025 in 2026-03..09. Per month: +0.0117, +0.0082, +0.0063, +0.0110, +0.0076, +0.0030, +0.0014, +0.0014, +0.0035, +0.0034, +0.0021, +0.0027.
   - **Without the partial 2025-10:** +0.0046 [+0.0025, +0.0067], 11/11 positive. The sign and the interval survive; the +0.005 size bar does not.
   - **Leave-one-month-out:** 4 of 12 forward cases and 1 of 13 look cases fall below +0.005.
   - **Reading:** a robust positive gain over linear S3 (the lower bound stays above 0 in every case). The pre-registered size bar is met on a knife edge, and the gain shrinks across the forward year, so the frozen fit ages.
6. **Minor.** `o1_lib.py` drops rows whose 5th previous hour has zero RV, although S3 uses only 4 previous hours (68 rows). This is causal and harmless.
