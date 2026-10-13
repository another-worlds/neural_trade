# O3: synthetic, interpretable volatility indicators (learned memory kernels) over a fair base (outlook prospect 4)

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. OUTLOOK_positive.md prospect 4: "Repeat H80
with RV statistics among the generator's inputs, and volatility as the target. Read the learned series as new, interpretable
volatility indicators". Tactical, exploratory; CPU only.

## Question
H80's free-form TCN series captured some volatility (+0.07..+0.12 Spearman over random series) but saw no realised-volatility
statistics and stayed 0.23-0.26 below HAR + time of day. Here the synthetic indicators are built FROM the volatility channels,
with a form that is readable by construction: each indicator is a learned memory kernel,

    I_k(t) = ln( sum_j softmax(a_k)_j * EWMA_tau_j(x_k)(t) + eps ),

an arithmetic mixture over a fixed bank of exponential moving averages (tau_j = 1, 2, 4, ..., 2048 minutes, 12 taus) of one
scale-free channel x_k (squared 1-minute log return, high-low range over close, log volume minus its 1,440-bar mean). The
learned weights a_k ARE the indicator: a kernel over the past that can be drawn and read ("this indicator remembers 30 minutes
and 1 day, nothing between"). HAR is the special case of a few fixed boxcar windows mixed geometrically (a ridge on logs). Does a
learned arithmetic kernel find volatility memory that the S3 base and a linear readout of the whole bank do not?

## Data, rows, targets, spans
As O1 (`runs/tactical/outlook/O1_geom_vol/SPEC.md`): the cut file to 2025-07-20 23:59, the Binance forward file 2025-10..2026-09,
no bar 2025-07-21..2025-09-30 read, anchors with stride 7, features from bars < a. Primary target y_H = ln RV(a, a+H), H in
{15, 60, 240}; rows with a zero-RV window dropped and counted. Selection 2021-01..2024-06 (42 months), one look 2024-07..2025-07,
forward 2025-10..2026-09 with frozen fits (365 days to the cutoff). The EWMA bank is computed causally on the whole series
(state carried from the file start; the first 3 x 2048 bars of each file are warm-up and never anchors).

## Models (one fit per horizon; monthly refit on the 365 days before the month, labels ending before it)
- **R0:** ridge on S3 (`t01_lib.build`), the base. Identical to O1's R0.
- **BANK:** ridge (alpha 1, standardised) on S3 + ln of every EWMA in the bank for the 3 channels (36 extra features): the fair
  linear control. If BANK equals LK, the "synthetic" form adds nothing over a richer HAR.
- **LK (the candidate):** S3 (linear) + K learned indicators, K = 2 per channel (6), with a linear readout; trained jointly by
  Adam (lr 1e-2, batch 4096, at most 30 epochs, early stopping patience 3 on the last 15% of the training year after an H-minute gap,
  best epoch kept), MSE on y_H, a_k initialised at 0 (uniform kernel), 2 inits (seed 0, 1), the one with the lower hold loss kept.
  TensorFlow on the CPU (CUDA_VISIBLE_DEVICES=-1) or numpy/scipy; the readout of S3 starts at the R0 ridge solution.
- **LKsh (control):** LK with the indicator rows permuted (train and test independently), to check the pipeline gives ~0.

## Metrics and units
Per month and horizon: Spearman(prediction, realised RV) and QLIKE (scale from the last 15% of the training year). Unit: the
month; 95% t-intervals over months of paired differences.

## Pre-registered criterion (H = 60)
**Pass on a span** iff LK against R0: mean monthly Spearman gain >= +0.005 with the 95% lower bound > 0, AND the mean QLIKE
difference <= 0. **Synthetic adds** iff, in addition, LK against BANK has a mean Spearman gain >= 0 with the mean QLIKE
difference <= 0 (LK is not worse than the linear bank: the learned form is at least as good with 6 instead of 36 features).
Verdict: "holds" (passes the look and the forward), "look only", or "no"; and separately "synthetic adds: yes/no" on the look.
H 15 and 240 are secondary (Holm over the two). The forward is computed whatever the look gives.
**Sanity gate:** LKsh within +-0.003 of R0 on the selection months (H = 60), else "broken".

## Interpretation outputs (the point of the positive outlook)
- the 6 learned kernels (weights over tau) of the frozen fit at H = 60, drawn as kernels over the past (log time axis);
- the kernels of monthly fits over 2021-2025 (do they drift? one figure per channel);
- each learned indicator drawn on real BTC windows (a quiet day and a shock day before 2025-07-20) beside RV60 and RV1440;
- in the tactical report style (TACTICAL.md "Report style"), as a section of the outlook report.

## Tests (seconds)
- the bank is causal (overwriting bars >= a leaves every feature at a unchanged);
- no bar 2025-07-21..2025-09-30 is read; training labels end before each month;
- with a fixed one-hot kernel, I_k equals ln of that EWMA exactly;
- R0 equals O1's R0 on a shared month (same rows, same predictions to 1e-9) once O1's code exists; otherwise S3 equals t01_lib.build.

## Outputs
`runs/tactical/outlook/O3_synth_volind/`: code, `results.jsonl`, `tables.txt`, `kernels.json`, `run.log`, `tests.log`, `REPORT.md`.
