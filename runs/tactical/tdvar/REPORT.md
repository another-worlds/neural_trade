# tdvar REPORT: TD estimation of the variance term structure (hypothesis #5)

SPEC: `SPEC.md` (committed in 770387f before any result). Code `tdvar_lib.py`, `tdvar_nn.py`, `run.py`, `analyse.py`; tests
`test_tdvar.py` (12 passed, `tests.log`). Tables `selection_tables.txt`, `look_tables.txt`; numbers `selection.json`, `look.json`;
curves `curve_sel.png`, `curve_look.png`. Hard cutoff: no bar after 2025-07-20 23:59 is loaded (tests
`test_cutoff_drops_every_later_bar`, `test_real_file_respects_cutoff_if_present`).

## Verdict (pre-registered headline: linear TD `td` against direct HAR, QLIKE in levels, curve over 5 windows)
**REFUTED on the linear model, on the selection months and on the one look.** The gain does not fall with the window; it is
about zero at 60 days and, at H = 1440, largest at the longest window.

Selection, 53 common months (2020-02..2024-06, every window complete). Mean monthly QLIKE gain %, 95% t-interval:

| H | W=30 | W=60 | W=180 | W=365 | W=1095 | Spearman(gain, log W) |
|---|---|---|---|---|---|---|
| 240 | -0.42 [-0.77,-0.06] | +0.24 [-0.45,0.93] | -0.06 [-0.21,0.09] | +0.02 [-0.14,0.19] | +0.15 [0.03,0.26] | +0.40 |
| 1440 | +1.16 [-1.95,4.26] | -1.26 [-4.35,1.82] | +0.32 [-0.09,0.72] | +0.35 [-0.08,0.78] | +0.89 [0.41,1.37] | 0.00 |

- C1 (W=60 gain >= 1.0% at both horizons, Holm p < 0.05): **not met** (0.24%, -1.26%; p 0.25, 0.79).
- C2 (W=1095 gain <= 0.3% at both): **not met** at H = 1440 (0.89%, the interval excludes 0.3 only barely: 0.41..1.37; the gain is
  positive and significant at the longest window, opposite to the prediction).
- C3 (Spearman < 0 at both): **not met** (+0.40 and 0.00).

One look 2024-07-01..2025-07-20 (13 months, contaminated by H53, H57 and volnet as stated in the SPEC): W=60 gain -0.28% (H=240),
-0.31% (H=1440); W=1095 -0.17%, -0.49%; Spearman +0.70 and +0.30. C1 and C3 not met, C2 met (trivially, the gain is negative).
Same verdict.

The linear `td` and `iterated` rows are identical (the stated identity; the test passes), `td_recent` (fresher anchors) is no
better (H=1440, W=30: -58% selection, huge intervals: 30-day windows are too short for a 1440-minute target, n_eff_train 30).
Against the other baselines: MIDAS is far better at W = 30 (QLIKE 0.32 against 1.17 for direct at H = 1440) and far worse at
long windows (a 20-point grid on the SSE in levels is not robust); the naive trailing rate beats every OLS-in-levels model at
H = 240 and 1440 for W = 30 and 60 (QLIKE 0.46 and 0.36 against 3.10 and 1.17 for direct at W = 30). So the short-window regime is won by a heavily regularised or fixed-form forecaster, not by a bootstrap.

## Secondary (neural tower; not in the pre-registered verdict, and the neural `direct` is a different, better model than OLS)
Numpy MLP with a log-linear skip, QLIKE loss, 27 selection months (every second common month) and all 13 look months.
`nn_td` (stage-1 `g_s`, frozen target network, bootstrap target) against `nn_direct` (same net, trained on the H-window):

| H | split | W=30 | W=60 | W=180 | W=365 | W=1095 | Spearman |
|---|---|---|---|---|---|---|---|
| 240 | selection | +5.63 [0.79,10.47] | +3.61 [-1.58,8.80] | +0.35 | +0.09 | -0.03 | -1.00 |
| 1440 | selection | +11.97 [2.78,21.16] | +8.09 [1.40,14.77] | +1.11 | -0.27 | +0.10 | -0.90 |
| 240 | look | +4.42 [-1.26,10.11] | +3.26 [0.13,6.40] | +0.11 | -0.09 | -0.14 | -1.00 |
| 1440 | look | +18.77 [8.64,28.90] | +7.07 [0.34,13.80] | +1.93 | -1.98 | -2.13 | -1.00 |

This is the predicted signature: a gain of a few to ten-plus percent at 30 to 60 days that vanishes (and turns slightly negative on
the look) at one to three years, falling monotonically with log W (Spearman -1.0 / -0.9 on selection, -1.0 / -1.0 on the look).
Applying the SPEC's C1-C3 arithmetic to it (labelled secondary): selection PARTIAL (C1 point met at both horizons, Holm fails at
H = 240: p 0.082), look SUPPORTED. Caveats: (1) the direct neural net itself is unstable at 30 days (-78% against OLS direct at
H = 1440), so part of the "gain" is the bootstrap target acting as a smoother for a noisy net, which is the mechanism but also means
it is a regularisation effect; (2) `nn_td` against linear direct HAR: +10 to +15% at W >= 60 for H = 240 and 60, almost entirely
the log-link QLIKE-trained net, not the tower (nn_direct gets the same); at W = 30, H = 1440 it is -40% (worse); (3) 27 and 13 months,
one seed, 200 fixed steps, no tuning; the neural variant was added to the SPEC before any result, but it is not part of the headline.

## Honest reading
- The tower/TD idea gives nothing for a linear HAR: linear TD is iterated HAR, and iterated is no better than direct, with no window
  dependence of the predicted kind. Ghysels et al.'s ranking (direct and MIDAS over iterated) is not overturned.
- The only place the predicted curve shows up is a small nonlinear net trained by QLIKE at 30 to 90 days: there bootstrapping
  through the tower removes a large share of the noisy direct net's error. Absolute levels on the same 27 selection months (QLIKE, H = 240 / 1440): at W = 30 MIDAS (0.395 / 0.291) and naive (0.480 / 0.380)
  still beat nn_td (0.513 / 0.463); at W = 60 nn_td wins (0.330 / 0.288 against MIDAS 0.379 / 0.487 and naive 0.481 / 0.380); from W = 180 on
  nn_td and nn_direct are equal (about 0.30 / 0.22). So the bootstrap helps a net at 60 days, but at 30 days a fixed-form forecaster is still better.
- No direction claim; this is risk-branch work (variance), no default changed.

## Run times
Linear selection (66 months x 5 windows x 3 horizons x 6 models): 80 s wall, 1 process (73 s summed per month; about 1.1 s per month).
Linear look: about 15 s. Neural selection (27 months): 2,885 CPU-s summed over 3 processes (about 16 min wall, 107 s per month
per process); neural look (13 months): 1,558 CPU-s over 3 processes (about 9 min wall). CPU only, below-normal priority, no GPU.
