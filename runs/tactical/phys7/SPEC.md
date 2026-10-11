# phys7: do the physics loss terms help the VARIANCE head? (tactical, shrunk; fixed before any run)

Source: hypothesis #7 of `runs/tactical/research/2026-10-11-unknown/README.md` (A4) with the verification notes. Owner,
2026-10-11: run it shrunk "at the expense of precision" instead of about 11 GPU-hours (96 runs x ~7 min). Branch
`nt-tactical-phys7` from `nt-tactical`; code pinned to the worktree `D:/nt/nt_phys7` (code head `3702d1b`; `src/` and
`tests/` are not touched, so the SPEC commit has the same code). Every run is launched with `PYTHONPATH=D:/nt/nt_phys7/src`.

## Question

D-003 keeps six physics terms (t_perp, Casimir, HD, IFE, vacuum, vacuum overflow). They were judged on direction and a
composite metric. The terms are about variance, so: with the terms on versus all six lambdas at 0, is the variance head
better, measured by what a variance forecast is for (ranking of |r| and QLIKE after re-scaling), and how far is it from
the standard HAR+tod forecast on the same data? Tactical and exploratory: no change of any default; the physics terms are
removed only by the owner (D-003).

## Design

The log-variance head named in the hypothesis does NOT exist as a Config switch (grep of `core/config.py` and the models:
the variance heads are softplus only; `log_var` in `gru_attention.py:151` is a gate input). Not added (no `src/` edits).
So the head axis of the 2 x 2 grid is replaced by the axis that decides which terms are alive:

* `PRICE_HEAD: none` removes the price outputs, and with them point, trend, coherence, **IFE, Casimir, vacuum** and vol
  (Config doc, and `loss_term_shares` of the run hc5_noprice1: only `t_perp` 0.008 and `vac_overflow` 0.041 are non-zero;
  `hd` and `casimir` are 0). So with `PRICE_HEAD: none` the contrast measures t_perp and vacuum overflow only. This is the
  owner's specified design (PRICE_HEAD none): it is the **primary** contrast.
* `PRICE_HEAD: on` (today's graph) keeps all five default-on terms alive (t_perp, Casimir, HD, IFE, vacuum overflow;
  vacuum's own lambda is 0 by default). The **secondary** contrast.

| cell | PRICE_HEAD | physics lambdas | config |
|---|---|---|---|
| none_on | none | default (0.1 each; vacuum 0) | `configs/tactical/phys7_none_on.yaml` |
| none_off | none | LAMBDA_T_PERP, CASIMIR, HD, IFE, VAC, VAC_OVERFLOW = 0 | `phys7_none_off.yaml` |
| price_on | on | default | `phys7_price_on.yaml` |
| price_off | on | all six = 0 | `phys7_price_off.yaml` |

Everything else = the hc4 long-block layout (`runs/tactical/hc4/make_hc4.py` WEEK): the static excerpt
`Bitcoin_BTCUSDT_tactical.csv`, 126,000 windows (7-day-type long block), N_FOLDS 2, FOLD_INDEX -2, batch 256, all three
horizons (10/15/20 bars), **4 epochs** (hc4: 8), calibration off, `save_predictions` on, `DATA_TAIL_TRIM`.
Slices: the 6 climb slices of hc4 (data ends 2017-05-02, 2018-11-16, 2020-03-30, 2021-10-13, 2023-04-29, 2024-11-11), **1 seed
(0)** each. 4 cells x 6 slices = 24 runs. The generator is `runs/tactical/phys7/make_phys7.py`. Each pair (on, off) is paired
over the same slice and the same seed.

**Hard cutoff 2025-07-20:** the latest data end is 2024-11-11; the analysis script asserts that no bar used (target bars
included) is after 2025-07-20. The test fold (the later part of each slice) is never scored: the metrics are computed on the
fold -2 out-of-sample (dev) block only, the 12,600 windows the screen saves.

## Metrics (analysis script `runs/tactical/phys7/analyze_phys7.py`, CPU)

Per slice, horizon h in {10, 15, 20} bars, r_h = close[t+h]/close[t] - 1 for the OOS windows (checked against the saved y):

1. **Spearman** of the head's variance with |r_h| over the whole OOS block (n 12,600, n_eff about 1,260). Mean over the
   three horizons is the headline.
2. **QLIKE** = mean(x - log x - 1), x = r_h^2 / (c s^2), over the second half B of the OOS block; the scale c = mean(r^2 / s^2)
   is fitted on the first half A (D-008 says re-scale on the cal block; the screen does not save that block, so A is the
   replacement, labelled; the same rule for the net and for HAR). Mean over horizons; lower is better.
3. **HAR+tod**, the same code as `runs/tactical/vol/volbench.py` (log future variance on log trailing variances over
   15/60/240/1440 bars plus 2 time-of-day harmonics, OLS per horizon, s^2 = exp(fit) x h), fitted on the bars of the same data
   segment that precede the OOS block by at least 81 bars (every 5th bar; the 95-day excerpt segment limits it to about 26,000
   bars for the first slice, so HAR gets roughly the net's own span), scaled on the same half A.

Unit of inference = slice (6). Paired differences are per slice (net on minus net off; net minus HAR+tod), 95% t-interval over
the 6 slices (t 2.571), plus the count of slices that go up.

## Verdict rule (before any run)

For the physics contrast (on minus off), on the mean-over-horizons Spearman:

* HELPS: the interval is above 0 AND the mean difference is at least +0.010.
* HURTS: the interval is below 0 AND the mean difference is at most -0.010.
* otherwise NO DETECTABLE EFFECT (reported with the minimum detectable effect below; this is not "no effect").

QLIKE is the secondary check with the same shape (negative = better; threshold 0.05); a verdict HELPS needs the QLIKE
difference not worse than +0.05. Each horizon is also listed; no multiple-testing correction (exploratory). Guard-rails: a
run with a non-finite step or a constant variance output (sd < 1e-9) is flagged; its slice is dropped from that pair and the
n is stated. Distance to HAR is reported (net minus HAR+tod, Spearman and QLIKE) for each cell; it is descriptive.

## Precision (stated up front)

Shrunk from 96 runs (2 x 2, 6 slices x 2 seeds x ... , 8 epochs) to 24 runs: 1 seed, 4 epochs (under-trained: the
variance head at 4 epochs may be more affected by the init than a converged one), 6 slices. Noise measured on existing runs
(`hc5_noprice3`, all heads, the same slices, seeds 0 and 1, 8 epochs; `scores_hc5_noprice3.json` from this analysis code): the
sd of a single-run paired difference of the mean Spearman is **0.035** (6 slices; one slice, 2018-11-16, contributes
0.087), of QLIKE **0.11**. With n = 6 the interval half-width is 2.571 x 0.035 / sqrt(6) = **0.037 Spearman** and
2.571 x 0.11 / sqrt(6) = **0.116 QLIKE**: only a physics effect of at least about 0.04 Spearman would be detected. For
scale, the gap net - HAR+tod in `hc5_noprice3` is -0.060 Spearman (8 epochs). So the study can detect a large effect and
rule out nothing small; a "NO DETECTABLE EFFECT" outcome is expected and is the weak answer. The 4-epoch noise may differ;
it is re-estimated from the runs themselves (the on/off spread across slices) and reported.

## GPU budget

Measured earlier (same layout, 8 epochs): `PRICE_HEAD none` about 360 s per run, price head on about 475 s (hc5_noprice1,
hc4_nophys / cand_c2_base mean wall_s), so 4 epochs about 190 s and 250 s (estimate). 12 x 190 + 12 x 250 = about 5,300 s =
**1.5 GPU-hours** if serial, less wall time with two processes (shards). Target about 2 GPU-hours. The measured wall time per
run after the first run is recorded in REPORT.md; if the projection exceeds 3 GPU-hours the study stops and reports what
exists. Order: none_on, none_off (primary) first, then price_on, price_off.
GPU rules: at most 2 heavy processes (2 shards per cell), below-normal priority, `NT_GPU_MEMORY_LIMIT_MB=4000`,
`TF_FORCE_GPU_ALLOW_GROWTH`; the owner's other project is never touched. Each run may exceed 2 minutes (owner approved this
shrunk study, 2026-10-11). Outputs: `runs/tactical/screens/phys7_*` in the worktree (results files, committed; weights and
prediction arrays stay untracked).
