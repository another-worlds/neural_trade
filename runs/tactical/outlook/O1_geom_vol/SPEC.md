# O1: does indicator geometry add to a fair volatility forecast? (outlook prospect 2)

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. Owner: "позитивный аутлук насчет геометрии
..." (TACTICAL.md "Division of work"). OUTLOOK_positive.md prospect 2. Tactical, exploratory; CPU only.

## Question
H47 found geometry worth +0.017..+0.021 Spearman over a BARE HAR (log RV lags + time of day) on 24 lab slices, but nothing
once the network's 60-feature vector was present. The fairness gap: the bare HAR saw no volume, range or hour-by-hour RV
statistics. Against the strongest linear base we have (S3 of T01/H72: HAR + volume/range extras + the three previous hourly
log RVs + their dispersion), does geometry still add to the forecast of the size of the next move, on a monthly walk-forward,
on the one look, and on the Binance forward months?

New in this study (never tested): geometry at a second scale, on 60 bars of 15 minutes (the last 15 hours).

## Data
- **Train / selection / look:** `D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv` (last bar 2025-07-20 23:59,
  `runs/tactical/common/cutoff.py`). Loaded from 2019-12-01 (365 training days + feature history before 2021-01).
- **Forward:** `D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv` (Binance spot, H82). No bar 2025-07-21..2025-09-30 is read
  (the T01 loader `t01_lib.load_bars` asserts it). The first 1,440 + 900 bars of 2025-10 are warm-up only.
- **Caveat (spent months):** the forward months were scored by T01/T03 (HAR on the volatility level, H85). No geometry model
  was ever scored on them, and this hypothesis was not formed from them, but they are not virgin data.

## Rows and targets
- Anchors a with stride 7 (coprime with 60: no minute-of-hour phase, H72). Features use bars < a only.
- **Primary target:** y_H = ln RV(a, a+H) (sum of squared 1-minute log returns over the next H bars), H in {15, 60, 240}.
  Rows with a zero-RV feature or target window are dropped and counted.
- Secondary target (comparability with H47/H57): ln(|r_H| + 1e-5).

## Feature sets
- **S3** (the base; exactly `t01_lib.build`): log RV 15/60/240/1440, hour sin/cos, volume ratios 15/60/240 to 1440, mean range
  15/60, log RV 5, the three previous hourly log RVs, their dispersion.
- **G1:** `geom2/geomlib.geom_rich` on the window of the last 60 one-minute bars (22 channels of all 14 families, slopes 3/10,
  position, crossings 3/5/10/20, squeeze; 164 features).
- **G15:** the same function on the last 60 bars of 15 minutes (900 one-minute bars ending at a-1, grouped into 15-bar OHLCV
  bars: open of the first, max high, min low, close of the last, summed volume).

## Models (ridge alpha 1 on standardised features, one fit per horizon; monthly refit on the 365 days before the month,
labels ending before the month; scale c for QLIKE from the last 15% of the training year, a gap of H minutes before it)
- **R0:** S3. **R1:** S3 + G1. **R2:** S3 + G1 + G15.
- **Control Rsh:** S3 + G1 + G15 with the geometry rows permuted (independently in the training and test rows, seed 0).
- Secondary, not a verdict: HistGradientBoosting (max_iter 200, lr 0.05, 15 leaves, min leaf 300, training rows strided by
  10) on S3 and on S3 + G1 + G15.

## Metrics, units
Per month and horizon: Spearman(prediction, realised RV of the next H bars) and QLIKE (v / (c e^mu) - ln(...) - 1 with v the
realised RV). The unit of inference is the month (labels overlap only inside a month and at its edge); 95% t-intervals over
months of the paired differences against R0.

## Spans
- **Selection:** 2021-01 .. 2024-06 (42 months). Chooses the candidate G* in {R1, R2}: the one with the larger mean Spearman
  gain over R0 at H = 60; R2 only if it beats R1 by at least +0.002, otherwise R1.
- **Look (one):** 2024-07 .. 2025-07 (13 months; the last one to labels ending by 2025-07-20 23:59).
- **Forward:** 2025-10 .. 2026-09 (12 months), G* and R0 frozen: fitted once on the 365 days ending 2025-07-20 23:59.

## Pre-registered criterion
**Pass on a span** iff, for G* against R0 at H = 60:
1. the mean monthly Spearman gain is >= +0.005 and its 95% t-interval lower bound is > 0;
2. the mean monthly QLIKE difference is <= 0 (not worse).

Verdict: "holds" if it passes on the look AND on the forward; "look only" if it passes on the look and fails forward; "no" if
it fails on the look. H = 15 and 240 are secondary, Holm over the two, reported but not part of the verdict. The forward is
computed whatever the look gives, and the report says which applies.

**Sanity gate:** the control Rsh must stay within +-0.003 of R0 on the selection months (mean Spearman difference at H = 60);
otherwise the analysis is reported as broken, with no verdict.

## Interpretation outputs (the positive-outlook part)
- the 15 largest standardised geometry coefficients of G* at H = 60 (frozen fit), with sign and channel;
- the incremental Spearman of each channel group (price bands, oscillators, volatility channels, volume) added alone to S3 on
  the selection months (descriptive, no verdict);
- one figure per top feature drawn on real BTC windows in the tactical report style (TACTICAL.md "Report style").

## Tests (seconds)
- no bar 2025-07-21..2025-09-30 is read; the training labels end before each month (before 2025-07-20 23:59 for the frozen fit);
- G1 and G15 at anchor a are unchanged when every bar >= a is overwritten (causality);
- the 15-minute aggregation equals a pandas resample on an aligned example;
- R0 on H = 60 reproduces the S3 features of `t01_lib.build` row for row.

## Outputs
`runs/tactical/outlook/O1_geom_vol/`: code, `results.jsonl` (one row per month x horizon x model), `tables.txt`, `run.log`,
`tests.log`, `REPORT.md` (the verdict against this SPEC).
