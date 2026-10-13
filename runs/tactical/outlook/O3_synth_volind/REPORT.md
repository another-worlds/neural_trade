# O3 report: synthetic volatility indicators as learned memory kernels (SPEC 37b58af; code and results 5e4ab42)

Tactical, exploratory, CPU (1,167 s). Months are the unit; 95% t-intervals of paired differences; "k/n" = months with a
positive difference. Numbers: `tables.txt`, `summary.json`; kernels: `kernels.json`. Written by the outlook lead from the
implementer's results (the implementer could not write this file); every number below was checked against `tables.txt`.

## Verdict (SPEC criterion, word for word)
- **Sanity gate:** LKsh - R0 = -0.0006 on the selection months at H = 60 (limit +-0.003): ok, not "broken".
- **Verdict: "no".** At H = 60, LK - R0 Spearman on the look is +0.0004 [-0.0002, +0.0009] (8/13). That is below +0.005 and the
  lower bound is under 0. It also fails on selection (+0.0004) and forward (-0.0008 [-0.0023, +0.0007]).
- **Synthetic adds: no.** LK - BANK on the look is -0.0065 [-0.0086, -0.0043] Spearman (0/13) and +0.0078 QLIKE (worse).

## H = 60
| span | R0 | BANK | LK | LK - R0 | LK - BANK |
|---|---|---|---|---|---|
| selection (42) | 0.7305 | 0.7342 | 0.7309 | +0.0004 [-0.0004, +0.0011] 25/42 | -0.0033 [-0.0044, -0.0022] 8/42 |
| look (13) | 0.7984 | 0.8053 | 0.7988 | +0.0004 [-0.0002, +0.0009] 8/13 | -0.0065 [-0.0086, -0.0043] 0/13 |
| forward (12, frozen) | 0.7745 | 0.7817 | 0.7737 | -0.0008 [-0.0023, +0.0007] 5/12 | -0.0080 [-0.0101, -0.0060] 0/12 |

QLIKE (lower is better), LK - R0: selection -0.0027 [-0.0045, -0.0009], look -0.0014 [-0.0031, +0.0004], forward -0.0039
[-0.0093, +0.0015]. Secondary horizons: at H = 15 LK - R0 is +0.0031 (selection, 41/42) and +0.0026 (look, 13/13). That is real
but under the bar. At H = 240 it is negative.

## Why: the kernels did not learn a shape
- Frozen fits stopped after 1-2 epochs. Every H = 60 kernel weight sits at 0.06-0.10, against 0.083 for a uniform kernel.
- Monthly fits lean slightly to 1-8 minutes and to 2,048 minutes, with a dip at 64-512 minutes. That is HAR's own short-plus-long shape.
- A slower learning rate with more patience gives the same result (`diag_lr.json`: LK - R0 +0.0001 on the look). The stopping rule is not the cause.
- An arithmetic mixture is a worse readout of the bank than a ridge on its logs.

## The finding on the way (not pre-registered: the control, not the candidate)
**BANK beat the S3 base everywhere, at every horizon.** BANK is a ridge on S3 plus the logs of 12 EWMAs (tau 1..2048 min) of r^2,
range and volume.

| span | H = 15 | H = 60 | H = 240 |
|---|---|---|---|
| selection | +0.0038 (42/42) | +0.0037 (34/42) | +0.0044 (30/42) |
| look | +0.0049 (13/13) | **+0.0068 [+0.0047, +0.0089] (13/13)** | +0.0132 (12/13) |
| forward | +0.0054 (12/12) | **+0.0073 [+0.0049, +0.0096] (12/12)** | +0.0118 (11/12) |

QLIKE also improves at every span and horizon (H = 60: -0.0069 / -0.0091 / -0.0099).
- **Size:** on look and forward at H = 60 the gain would meet O1/O3's bar (>= +0.005, CI > 0, QLIKE not worse). On selection it would not (+0.0037).
- **Cost:** the extra features are 36 standard EWMAs. The gain is a denser HAR, not a synthetic indicator.
- **Status:** it was chosen after results were seen, and the forward months are spent. To count, it needs its own SPEC and an unseen test, which can be either:
  - early history 2017-09..2018-12, for the volatility LEVEL target;
  - prospective months from 2026-10.

## Caveats
- **Spent months:** the forward months were already used by T01 and T03.
- **Feed:** the forward volume comes from Binance spot, a different feed. BANK's gain is the same on look and forward, so nothing points to a feed problem.
- **Warm-up:** 2025-10 is scored from 10-05 (6,144 bars of warm-up).
- **Dropped rows:** 1,612 of 422,488 train anchors and 349 of 74,380 forward anchors had a zero-RV window (`dropped_rows.json`).
- **Seeds:** each month keeps the better of 2 seeds by hold loss.
- **Row counts:** LK fits on 85% of the year while R0 and BANK use all of it. Against the fit-only R0f, LK gains +0.0008 on the look. The verdict does not change.

## Deviations
1. a_k starts at N(0, 0.1), not 0. With exact zeros the two kernels of a channel stay identical and the two seeds coincide.
2. I_k is standardised with a fixed mean and sd from the uniform-kernel indicator on the fit rows.
3. Channels:
   - r^2 and (high - low) / close are used raw;
   - volume is exp(ln(v + 1) - its trailing 1,440-bar mean);
   - eps is 1e-10 / 1e-6 / 1e-3 for r^2 / range / volume;
   - the EWMA weight is 1/tau.
4. QLIKE = v / (c e^mu) - ln(.) - 1, with c from the hold rows, as in the SPEC.
5. S3, the rows and the ridge are the implementer's own code on `t01_lib.build`, because O1's `o1_lib.py` was not committed yet. `test_o3` cross-checks R0 against O1's ridge to 1e-9.
6. The R0 solution was not an epoch-0 candidate.
7. numpy instead of TensorFlow, which the SPEC permits.

Tests: `tests.log`, 7 passed. They cover causality, the forbidden bars, the one-hot kernel identity, a finite-difference gradient and the S3/R0 identity.
