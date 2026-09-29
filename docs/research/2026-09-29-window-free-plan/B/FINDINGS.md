
Everything ran on CPU from D:/nt_research/wfp/B/: `CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python <script>`. Nothing in D:/neural_trade changed. Measured = measured on CPU or computed from saved runs. Estimate = reasoning, basis given. CPU timings are medians of 7 interleaved repeats on a shared machine, so they are relative only. The q4b checks you asked about completed before the interruption; their results are in Q4.

## Answer
**Replacement:** alpha_{i,t} = sigmoid(logit_i + 0.5·tanh(ctx_t·W + b)_i).
- **Context:** ctx_t is today's two features computed at every bar over a trailing 60-bar span: the mean and max of (close_k − close_t)/scale. It should get its own wall-clock key, separate from LOOKBACK.
- **Same as today at the anchors:** it reproduces today's applied periods exactly there (max relative difference 0.0, Measured). So A/B-1 changes only the indicator memory.
- **Kernel:** a factored per-bar kernel. It costs about 1.7× a fixed period on the full 31-channel layer (CPU), with about the same op count (0.93×).
- **D-031:**
  - The report gives the global learned value plus a per-bar range.
  - The off switch sets the shift to 0 and routes to the fixed-period kernel.
  - The frozen twin is the textbook logits plus that switch.
- **Streaming:** K recurrence states plus a 60-close buffer.

**Warm-up rule:** M(eps) = ceil(ln eps / ln(1 − alpha_min)), with alpha_min = sigmoid(min logit − 0.5).
- It is an upper bound, because tanh caps the shift at ±0.5.
- The two-stage channels (MACD signal, Bollinger variance) use their convolution tail.
- Recompute it every step with a one-epoch margin.
- A burn-in loss mask after the data start and after time gaps is fixed at run start.
- A history bound derived from the data (not a configured ceiling) projects any logit whose M would exceed the available history.
- Fail loudly only if the textbook starting periods themselves cannot be warmed.
- The Predictor needs M + L − 1 bars of history or a carried state.

## Q1: today's adaptation, exactly
- **Context:** the input is (x − last_close)/scale (scaling.py:52; scale 257.52 USD). The two features are GlobalAveragePooling1D and GlobalMaxPooling1D of the window (gru_attention.py:47-51): the window mean and max relative to the last close, in scaled units.
- **Network:** `Dense(18, tanh)` (gru_attention.py:52-56), 54 parameters. It is trained by the **main** optimizer (custom_model.py:486-493), so ReduceLROnPlateau applies to it. The indicator optimizer stays at LR 5e-3 (optim.py:24; lr_indicator_used = 0.005 in every logged epoch).
- **Entry:** alpha = sigmoid(STE(logit) + 0.5·adj) (learnable_indicators.py:33, :110-115). The ×5 straight-through multiplier applies only to the logit's gradient.
  - The shift is in (−0.5, +0.5) logit: ×0.61 to ×1.65 of the base for long periods (base 60 → 36.4 to 98.0 bars), ×0.69 to ×1.52 at base 5.
  - The clip acts on the base logit only (learnable_indicators.py:300-314, custom_model.py:516-519), so applied periods can pass the 60-bar ceiling.
- **Measured on the 7 local runs:**
  - The realized shift spans −0.496 to +0.497 (p0.1 to p99.9), and 3.5% of pre-activations have |tanh| > 0.9. The bound is actually reached.
  - Newest run, macd_1_slow: base 58.9, applied p5/p50/p95 53.3/73.0/93.8, max 96.5.
  - The meta kernel is still mostly its initial random structure: it moved 9-10% (Frobenius, relative) from its seed-42 initialisation after 3 epochs and 27-32% after 20 (q1_meta_init.py).
- **How it is reported today:**
  - `applied_periods` (indicator_evolution.py:340-374) runs the meta sub-model and converts to periods in numpy. My re-implementation matches it to 1.4e-7 relative.
  - `_summary` (:238-289) adds p5/p50/p95.
  - The evolution figure (:463-640) shows a strip of median applied periods and the clip line marked "= lookback".
  - The applied figure (:769-862) shows 5-95% and IQR bars and a "lookback ... never warms up" line.
  - The notebooks use the test block (build.py:227, :434).

## Q2: candidates in series mode
Kernels are my own (kernels.py): a two-level chunked form with cumsums local to each chunk and a Hillis-Steele carry; a Toeplitz form for a fixed alpha; and a **factored** per-bar form, exp(cs_i − cs_j) = exp(cs_i)·exp(−cs_j), which needs no C×C matrix and uses O(KT) memory. All run in increment form, 18 logits → 31 channels, with the served weights of run 20260924T182915Z.

**Cost, N = 30,720 bars, forward + backward (Measured CPU; ops are graph nodes)**

| Variant | Full layer: ops / ms / × fixed | Bare K=31: × fixed | Bare K=87: × fixed (largest intermediate, calculated) |
|---|---|---|---|
| (c) fixed period, Toeplitz | 2,524 / 20-31 / 1.0 | 1.0 | 1.0 (10 MB) |
| (a) per-bar, factored C=32 | 2,352 / 52.8 / **1.71** | 2.09 | **2.42** (O(KT)) |
| (a) per-bar, matrix C=64 | 2,344 / 229-288 / 9.3-11.4 | 19 | 20-21 (652 MB) |
| (b) bank M=3 | 2,789 / 57-88 / 2.8 | 3.9 | 4.7 (31 MB) |
| (b) bank M=5 | 2,791 / 84 / 4.2 | 6.7 | 7.2 (51 MB) |

- **Ops:** every candidate is within 0.93-1.15× the fixed period's ops, and ops are what matters on a launch-bound GPU. The GPU cost difference is an estimate (probably a few ms); nothing was run on the GPU.
- **Factored kernel limit:** exp(−cs) must stay finite, so alpha ≤ 0.918 at C=32 or ≤ 0.99 at C=16. That means an alpha cap tied to the chunk size instead of today's 1 − 1e-6. Both edges tested finite (Measured). Choosing C is part A's call.
- **Precision against float64** (30,720 and 43,008 bars; the run's periods and periods ×20):
  - Every channel is ≤ 7e-5 absolute. The worst channel is tanh(10·hist), which amplifies errors in the histogram 10×. RSI is ≤ 5.5e-5 on its 0-100 scale.
  - State channels are ≤ 6.6e-7 relative to their largest absolute value.
  - This meets the Challenge's tolerance of 1e-4 absolute or 1e-5 × the state's largest absolute value.
- **Causality** (end to end from the raw closes):
  - Perturbing every close after t* leaves all outputs and context features up to t* bitwise unchanged. This holds for (a) with the rolling or the EW context, (b) M=3 and (c).
  - The gradient of the output at t* with respect to any later bar is exactly zero.
  - A NaN in a later bar poisons at most C−1 earlier bars in the same chunk (25 bars here). The finite-input contract is still needed.
- **Longest memory:** for macd_1_slow, the realized worst case to reach 1e-3 on the initial state is 314 bars. The tanh bound gives 334, and 204 with no shift. So adaptation lengthens the warm-up about 1.6×, and real data comes within 6% of the bound.
- **What each candidate computes** (q3_reporting.py):
  - The EW variant of the context correlates 0.98 (mean) and 0.85 (max) with today's features, so it would change the adaptation map.
  - Against a fixed-period EWMA at each bar's own period, (a) differs by 2.5-17% RMS because the memory mixes past alphas. The bank differs by 1.2-1.6% at M=3 and 0.3-0.4% at M=5.
  - Output roughness is within 0.96-1.01× of the fixed period for all candidates.

**Recommendation: (a) with the rolling context and the factored kernel.**
- It matches today's adaptation at the anchor, which gives A/B-1 one change per arm.
- It costs about 1.7-2.4× on CPU with about equal ops.
- Memory is O(KT), so it scales to K=87 and long passes, and it streams naturally.
- **Off switch:** the per-bar kernel with a zero shift matches the fixed path to 6.8e-5 (float32 rounding). The switch should route to the Toeplitz kernel so the two are identical.
- **Frozen twin:** textbook logits plus the switch come within 1.5e-5 of the float64 textbook EWMAs.
- **Context features:** computed per block from trailing data with the train-fitted scale, and never with a whole-block normaliser.
- **Bank (b):** keep it only as the fallback if the owner wants "each bar gets one fixed period" literally.
- **No adaptation (c):** it is the off switch, not a replacement (D-031 keeps adaptation).

## Q3: reporting
- **What to report per instance, over the test block:**
  - The global learned value, i.e. the base with no shift, in bars and in wall-clock time.
  - The instantaneous per-bar period p_t = 2/alpha_t − 1, as p5/p50/p95 and min/max. This is the direct successor of today's "applied per window".
  - The **effective period** p_eff = 2·m_t + 1, where m_t = (1 − a_t)(m_{t−1} + 1) is the exact mean lag of the weights. It equals p for a fixed alpha and is computed only at report time.
- **Example (Measured, newest run):**

| Instance | Textbook | Base | Instantaneous p5/p50/p95 | Effective p5/p50/p95 |
|---|---|---|---|---|
| macd_1_slow | 35 | 58.9 | 53.3 / 73.0 / 93.8 | 60.6 / 71.3 / 86.2 |
| bb_period_2 | 25 | 27.2 | 24.8 / 33.3 / 42.8 | 26.9 / 32.6 / 40.8 |

- **Textbook comparison:** in series mode, "textbook" means the configured period computed as a full-history fixed EWMA, i.e. the real indicator after warm-up.
- **On price:**
  - Draw textbook (dashed), learned adaptive (solid) and learned base without adaptation (dash-dot; not dotted, because D-014 reserves dotted for training). The two gaps separate "the period moved" from "the adaptation".
  - Example: macd_1_slow adaptive vs textbook differs 57% RMS relative, base vs textbook 39%.
- **Figures (D-014, one home each):**
  - A per-bar period panel with instantaneous and effective lines and base and textbook reference lines.
  - The 5-95%/IQR strip.
  - A table with textbook, base, instantaneous and effective percentiles, M(eps), and an "at history bound" flag.
  - In the training figure, the "lookback" line becomes the history-bound line.
  - The NT-048 HTML report shows the same numbers in minutes.

## Q4: warm-up with no ceiling
**The rule:**
- The rule is as in the Answer. The upper bound is exact for single EWMAs.
- Cascades use the tail of (1−b)^t + f·b(1−a)[(1−a)^t − (1−b)^t]/(b−a), with f = 1 for the MACD signal and 2 for the Bollinger variance.
- **Values at eps 1e-3:**
  - Newest run: 340 bars, set by the macd_1 signal cascade.
  - Textbook starting periods: 202 bars with the shift, 124 without.
  - Single EWMA with the shift: period 60 → 340 bars, 240 → 1,365, 1440 → 8,198, 10,080 → 57,396.
- **Empirical check** (q4b_checks.py; full 31-channel mode-(a) layer, cold vs warm start at 4 offsets):
  - Bars until every channel is within eps × its standard deviation: 151 / 250 / 347 at eps 1e-2 / 1e-3 / 1e-4.
  - The analytic rule gives 226 / 338 / 449.
  - The no-shift version (138 / 206 / 274) is too small. The shifted bound is required and sufficient.

**Where the history is needed:**
- **Training:** M + L − 1 bars before the first anchor. Each step's pass spans [min anchor − (L−1) − M, max anchor].
- **Evaluation:** the same, per block.
- **Serving:** history ≥ M + L − 1, or a carried state (the K states, the last L indicator rows and the context buffer). The Predictor refuses shorter histories.
- **Bundle metadata:** eps, meta_scale, context span, L, M at the served weights, and the kernel form. This needs a format version bump.
- **Margin within an epoch:** computing M at logit_min − 0.5 − S·lr multiplies it by 1.80 at 119 steps and lr 5e-3, 1.21 at 39 steps (a 7-day block), and 1.12 / 1.04 at lr 1e-3.

**Measured drift** (7 local + 84 ablation runs; per-epoch logit change divided by steps × 5e-3):
- Local: median 0.042, p99 0.20, max 0.30. Ablation: median 0.055, p99 0.28, max 0.76.
- So "Adam moves a logit by about LR × multiplier per step" is only the upper bound. Realized drift is about 5% of it because the gradient sign flips between steps.
- Slow periods lengthen in 67-68% of epochs.
- The run that capped after 3 epochs used batch 64 (473 steps per epoch).
- Cap hits: macd_1_slow in 3 of 7 local runs; macd_0_slow 8 times and macd_1_slow 5 times in 84 ablation runs.

**Projection without a ceiling:**
- **At measured rates** (worst instances, 119 steps per epoch):

| Worst instance | Period after 20 / 40 epochs | M(1e-3) after 20 / 40 epochs |
|---|---|---|
| Local runs | 109 / 343 bars | 618 / 1,953 bars |
| Ablation runs | 219 / 1,898 bars | 1,244 / 10,807 bars |

  On a 7-day block (39 steps per epoch) it stays ≤ 104 bars.
- **At the Adam bound (worst case):** starting at 35 bars, 4.9 million bars after 20 epochs at 119 steps and lr 5e-3; 1,681 at 39 steps; 368 / 75 at lr 1e-3.

**When M exceeds the history — recommended combination:**
1. Fail loudly at run start only if the textbook starting periods cannot be warmed.
2. Fix the mask at run start after the data start and after gaps, so the objective never changes during a run. (A mask that grows during training is rejected.)
3. After each step, project each logit so that M with its margin stays within the history the purge rule allows. Log the count per epoch and flag it in the report.
   - This is not a configured ceiling: on the 2017-2025 file the available history is years, and the real limit becomes the pass span per step.
   - I rejected fail-loudly alone, because drift would kill sweep trials at random.
   - I rejected a soft penalty, because it adds another lambda that interacts with the loss-weight calibration.
4. A log-timescale parametrisation adds nothing: d log(period)/d logit is already −0.97 at period 30 and −0.998 at 240.

**Curse of memory** (Measured):
- The relative sensitivity RMS(∂d/∂logit)/RMS(d) is flat at 0.61-0.71 from period 5 to 10,080. One Adam step changes the feature by about 0.35% of its RMS at any period, so sensitivity does not blow up under this parametrisation.
- What does grow is the channel's magnitude. RMS(d) follows the random-walk law σ₁·√((1−a)²/(a(2−a))) (measured-to-predicted ratio 0.89-0.99): 0.97 at period 60, 4.5 at 1440, 12.1 at 10,080 (scaled units).
- **Mitigation:**
  - Normalise each price-type channel by that closed-form scale (behind the A/B, since it changes numbers).
  - Add INDICATOR_LR_MULT = 1 as a pre-registered series-mode variant.
  - Add a named NT-038 harness case with slow periods starting at 1,440 and 10,080 bars.
  - Report M and the history-bound count in the NT-037 health numbers.

## Q5: the ceiling evidence (block bootstrap, 240-bar blocks × 300)
Measured on the newest run's training block (27,812 anchors):
- **Correlation with the 59-bar return (R59) at the learned applied periods:** the windowed MACD-1 line gives 0.920, the full-history ("warm") line 0.888. The gap is +0.032 [0.026, 0.039].
  - At the textbook period 35 the gap is only +0.001.
  - All local runs with slow > 55 show +0.027 to +0.034; the 13 capped ablation runs show +0.015 to +0.025.
- **As the slow period grows:** the windowed correlation rises steadily (0.785 at 35, 0.899 at 60, 0.956 at 98, 0.992 at 300). The warm one peaks at about 0.91 around 80-98 and falls to 0.72 at 300.
- **Direction of the drift:** the change in the line when the period lengthens correlates with R59 at 0.965 windowed vs 0.846 warm.
- **Across window positions:** the windowed line correlates 0.939 with the price path relative to the window's first bar (warm 0.793).
- **Predictive value:** correlation with the h=10/15/20 targets is the same for windowed and warm within 0.001 on train and 0.007 on test. All values sit inside the noise band (±0.037 to ±0.053 on train, n_eff = N // h).
- **Verdict: still ambiguous, but now quantified.**
  - The artefact is real, and lengthening the period pushes the windowed line toward R59.
  - At the learned periods, though, the windowed and warm lines correlate 0.97-0.99 and carry no measurably different signal.
  - The warm line's own R59 alignment also peaks near 80-98 bars, so "wanting longer memory" explains the drift just as well.
  - The deciding evidence is an A/B-1 secondary: whether series-mode slow periods still drift up, and how far.

## Proposed gates (checkable at QA time)
- **G-B1 (per-bar adaptation):**
  - Float64 agreement ≤ 1e-4 absolute or ≤ 1e-5 × the state's largest absolute value, per channel, at 43,008 bars, for the run's periods and ×20.
  - The rolling-context shift equals today's per-window shift at every anchor, relative difference ≤ 1e-6.
  - Past outputs bitwise unchanged end to end from raw closes; the gradient from any later bar exactly zero.
  - With the switch off, every per-bar period equals the base, and outputs are bitwise those of the fixed path.
  - The frozen twin is within 1e-4 of float64 textbook EWMAs.
- **G-B2 (warm-up):**
  - The analytic M is ≥ the empirical offset-invariance value at 4 offsets for eps 1e-2 to 1e-4.
  - The loss anchor set is identical at epoch 1 and epoch N.
  - The history-bound projection holds after every step under a forced drift, and is counted.
  - The run-start refusal names the instance.
- **G-B3 (Predictor):**
  - It refuses a history shorter than M + L − 1.
  - Streaming with a carried state matches batch prediction within 1e-4.
  - The bundle metadata round-trips.
- **G-B4 (reporting):**
  - Base, instantaneous and effective p5/50/95 are reported per instance.
  - The effective period equals p exactly for a fixed alpha (test).
  - No "lookback" line appears in series mode.
  - The D-014 figure tests pass.
- **A/B-1 secondaries to pre-register:**
  - Share of slow periods at the history bound.
  - Drift per epoch in each arm.
  - corr(slow line, R59) in each arm.

## Proposed backlog items
1. **Per-bar adaptive periods in series mode** — P1, implementer. Depends on the series engine and NT-046.
   - Acceptance: G-B1.
   - A context-span key.
   - Precomputed trailing context features with a leakage test.
2. **Warm-up rule, history bound and burn-in mask** — P1, implementer. Depends on 1, NT-041's gap policy and the purge-rule decision. Acceptance: G-B2, plus M and the bound count in the per-epoch logs (NT-037).
3. **Predictor series mode and bundle metadata** — P1, implementer. Depends on 1 and 2. Acceptance: G-B3.
4. **Reporting of per-bar periods** — P1, implementer. Touches indicator_evolution and the NT-043/NT-048 views. Acceptance: G-B4; notebooks rebuilt through build.py.
5. **"Long memory" stability-harness case** — P2, joins NT-038. Slow periods starting at 1,440 and 10,080 bars with lr multipliers 5 and 1, plus the channel-scale normalisation variant.
6. **A/B-1 SPEC additions** — experimenter. The secondaries above, plus an optional INDICATOR_LR_MULT = 1 variant.

## Owner question
None needed. "Unlimited" is respected: nothing configures a ceiling. The plan the owner approves should state the data-derived history bound plainly. If the owner rejects it, the only alternative is failing the run loudly.

## Risks
- **Factored kernel:** it imposes an alpha cap per chunk size, which limits the shortest period. Part A must choose C and test that edge.
- **Timings:** CPU timings came from a shared machine; the same variant measured 20-31 ms across runs. GPU cost is an estimate until the benchmark kit runs on the GPU.
- **History bound:** it is effectively a soft ceiling. If slow periods sit at it on the 30-day file, the correct reading is "needs more history".
- **Meta Dense:** it is still about 70% initialisation structure, so the value of adaptation is unproven. The switch-off arm is the cheap check.
- **Q5:** correlations cannot establish causation; only A/B-1 can.
- **Rolling context:** it keeps a 60-bar feature span (not an input window). The EW alternative changes the adaptation map.

## Files (D:/nt_research/wfp/B/)
- Shared modules: common.py, kernels.py, candidates.py
- q1_meta_init.py → q1_meta_init.json
- q1_adapt_today.py → q1_adapt_today.json
- q2_candidates.py → q2_candidates_prec_caus_mem.json, q2_candidates_cost.json (log q2_cost.log)
- q2b_factored.py → q2b_factored.json
- q3_reporting.py → q3_reporting.json
- q4_drift_warmup.py → q4_drift_warmup.json
- q4b_checks.py → q4b_checks.json
- q5_ceiling_artifact.py → q5_ceiling_artifact.json (log q5.log)