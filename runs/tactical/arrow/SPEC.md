# Arrow of time: when is the market predictable? SPEC (fixed before any result is computed)

Tactical, exploratory (D-062/063; owner 2026-10-11: any target allowed). Hypothesis 3 of
`runs/tactical/research/2026-10-11-unknown/README.md` (H-4 + A1), with its five mandatory corrections and
`verification.md` section A1 + H-4. Branch `nt-tactical-arrow`, off `nt-tactical` 3702d1b. Nothing here changes a default.

## Data and the hard cutoff

- `Bitcoin_BTCUSDT.csv`, 1-minute OHLCV (D-050). **Last bar used: 2025-07-20 23:59 (hard cutoff, no bar after 2025-07-20).**
  `arrowlib.load_bars` drops every later row at load; `test_hard_cutoff_bars_and_anchors` asserts it for bars, anchors, label bars and
  the last evaluation month (2025-06).
- Anchors: the 306k anchors of the `recent` cache (`D:/nt/nt_recent_cache/1m_t.npy`, every 10th bar from 2019-12-01; window = the 60 bars
  ending at the anchor bar), kept only if the label bar (anchor + 15 bars) is inside the cutoff. Windows with zero realised
  volatility are dropped. The tb7 features are that cache's `1m_tb7.npy` rows (13 textbook features, causal, window-local).
- Label of tb7 (next-open, as H61): `r = close[i+15] / open[i+1] - 1` for anchor bar `i`; up = r > 0; rows with |r| <= 5 bps are masked
  (as `recent/wf.py`). tb7 model: logistic regression C = 0.1 on standardised, clipped (+-8) features, trained on the 365 days before each
  month whose labels closed by the month start (`t + 16 min <= month start`), tested on the month's anchors.

## Models (GPU, small 1D-CNN, NEEP-style, Kim et al. PRL 2020)

Window channels, all divided by the window's OWN realised volatility `sigma_w = sqrt(mean(body^2 + gap^2))` (a time-symmetric scalar; no
trailing EWMA, correction 2): `body = ln(c/o)`, `up = ln(h / max(o,c))`, `dn = ln(min(o,c) / l)` (scaled, clipped at +-20) and
`lv = ln(1+volume) - window mean`, `gap_k = ln(o_k / c_(k-1))` (0 for the first bar; scaled). The gaps are not negligible on this file (55% of
bars have |gap| > 1e-4, median 1.3e-4; `prep_info.json`), so they are a channel; time reversal negates them and re-attaches them to the new
neighbours (tested against reversing the raw bars).

The classifier sees a window `x` or its transform `T(x)` and says which (label 1 = original). Conv(32,k5) - Conv(32,k5,s2) -
Conv(32,k3,s2) - [global mean, global max] - Dense(16) - Dense(1); Adam 2e-3, batch 256 windows (512 samples), at most 12 epochs, early
stop with patience 2 on a validation block (the last 10% of the training rows, separated by 60 minutes), best weights restored.
`tf.random.set_seed(seed)`, seeds 0 and 1. Four models, differing by `T` and input:

| model | input | T | role |
|---|---|---|---|
| `full` | body, up, dn, lv, gap | time reversal: bar order reversed, open and close swapped (body negated, gaps negated and re-attached, wicks and volume unchanged) | main: EP_total |
| `abs` | |body|, up+dn, lv, |gap| (sign-free) | order reversal | EP_vol: asymmetry of magnitudes in time (Zumbach-type) |
| `order` | body, up, dn, lv, gap | order reversal only | control: time order without the sign flip |
| `sign` | body, up, dn, lv, gap | mirror image of the price path (body and gap negated, up and dn swapped), order kept | control: sign (skew) only |

(The README's "three controls" are `sign`, `order` and, for correction 3, `abs`; `full` is the estimator.)

Walk-forward by month, retrained every month: evaluation months **2020-07 .. 2025-06** (60); training = the 365 days of anchors ending
60 minutes before the month starts (no bar shared with a test window); test = the month's anchors. **EP estimate of a month = ln 2 - mean
held-out loss**, the loss of a window `x` being `0.5 [BCE(f(x), 1) + BCE(f(T(x)), 0)]` (mean over the month's windows; seeds averaged
per window). The per-window time-symmetric irreversibility score is `g(x) = |f(x) - f(T(x))| / 2`.

## Questions, with the criteria fixed now

**(i) Does detrended monthly EP track detrended monthly tb7 AUC? (months 2021-01 .. 2024-06, 42)** Series: monthly `EP_full` and
monthly tb7 AUC (masked rows). Detrending (correction 1), both reported: first differences; deviation from the calendar-year mean.
Spearman rho with a moving-block bootstrap 95% CI (block 3 months, 2000 reps). Also, as references: `EP_vol`, `EP_dir = EP_full - EP_abs`,
the monthly mean ln sigma_w, and the partial rho of EP and AUC given ln sigma_w. **"Tracks" = rho >= 0.4 in BOTH detrendings**; the raw
Spearman is printed only to show the trend effect.

**(ii) Gate (exploratory, 2021-01 .. 2024-06; months already used by H56/H63/H64, no clean confirmation exists).** For each month the
quintile edges of a gate are the quantiles of that gate over the **6 preceding months** (their own out-of-sample scores: causal).
Per month and quintile, the AUC of the tb7 prediction on the masked test rows (bins with < 150 rows or one class: NaN). Spread =
AUC(top quintile) - AUC(bottom quintile), mean over months with a t-interval. Gates: `EP_full` (= g of `full`; primary), the same for
`abs`, `sign`, `order`, `sigma` (window realised volatility), and `EP_full|sigma` (EP quintile within a sigma quintile).
**Criterion (point estimates, both):** mean spread(`EP_full`) >= +0.015 **and** mean spread(`EP_full`) - sgn * mean spread(`sigma`) >=
+0.01, where `sgn` is the sign of sigma's mean spread (sigma in its better orientation: conservative for EP). The prior bet recorded in
the README: EP will be a volatility proxy and will not beat the sigma gate. Also reported: Spearman of g with sigma over windows.

**(iii) Volatility vs return asymmetry (corrections 3, 5).** `EP_total` = EP of `full`; `EP_vol` = EP of `abs` (reversal of
sign-free magnitudes); `EP_dir` = the **paired per-window** gain `loss(abs) - loss(full)` (never a difference of two separately
biased scalars), pooled and per year, with a day-block bootstrap 95% CI (2000 reps). Controls: EP of `sign` and `order`. These are
checks of known stylised facts (leverage, Zumbach), not discoveries.

**Control "sign -> future sign" (correction 4).** Monthly walk-forward logistic models on 12 sign features (signs of the last 10 bar
bodies, of the 60-bar net body and of the last 5): (a) AUC for the future sign (the tb7 label), (b) AUC for "future |r| / (sigma_w sqrt 15)
above its training median" (leverage-type sign -> |r|). Reported: both mean AUCs and the detrended Spearman of monthly `EP_dir` with each.
If `EP_dir` follows (b) and not (a), the directional part of EP is the leverage effect, not a directional edge.

## Statistics and noise

Windows of one stride-10 grid overlap (6 per 60 bars): intervals over months use the month as the unit (t-interval, n = 42), window-level
intervals use day blocks. Seeds are averaged per window; the seed-to-seed difference of the monthly EP is reported. No multiple-comparison
correction (exploratory; one primary gate, one primary tracking test). A verdict on (ii) is exploratory only (README: direction questions
for 2021-2024 have no clean confirmation). No test-fold data, no choice on 2024-07 or later, no default changed.

## Resources and process

GPU cap 3000 MB per process, at most 2 processes, below-normal priority, `tf.random.set_seed`; the trainer works in chunks of at most
~100 s wall-clock (`run.py --budget 100`, resumable by result file), driven by `drive.sh`; every chunk is a separate process (D-063).
480 tasks (4 models x 2 seeds x 60 months). Order: `full`, `abs`, `order`, `sign`. Tests (seconds): `test_arrow.py` (reversal transforms
against the raw-bar construction, the mirror transform, scaling, cutoff, causality of the window features, gate edges from earlier months
only, month split, detrending). Run times are reported with the results.
