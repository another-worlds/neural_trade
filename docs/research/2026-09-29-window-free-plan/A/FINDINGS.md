# NT-053 part A: series kernel, window assembly, pass span, gaps (findings)

All measurements are CPU only (i9-13900F, TF 2.10.0, op determinism on, `import neural_trade` first). Every
script runs as `CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python <script>`
from D:/nt_research/wfp/A/. Timings are medians of 7-9 interleaved repeats (spread in the JSON files) and are
relative only. **Every GPU number is an estimate.** Nothing in D:/neural_trade was changed.

## Answer

- **Kernel:** a hierarchical chunked recurrence ("V1"). Its C x C decay matrices are built from segment sums: direct sums of consecutive log-decays, never a difference of cumsums. Every cumsum is local to at most C bars, or to at most G = 32 blocks at the higher levels.
  - The kernel takes the **log decay** `la = -softplus(logit) x dt` directly. It never takes 1 - alpha in float32.
  - It carries state in and out, and represents a reset as `la = -1e30` (decay exactly 0).
  - It contracts with an **elementwise multiply + reduce_sum, not a matmul**. TF 2.10 runs every matmul in TensorFloat-32 on this GPU by default, and in an emulation of TF32 that breaks the tolerances by 20-200x.
  - Chunk size: **C = 16** (it meets every tolerance and has the smallest memory and CPU cost; C = 64 and 128 also pass).
- **Window assembly (D6):** forward `tf.gather`, with a `tf.custom_gradient` whose backward is a "transpose by gather": a host-computed inverse map (bar to batch row; the anchors in a batch are distinct), one gather of the padded upstream gradient, then `reduce_sum` over L. It has no segment op, scatter op, sparse op or matmul.
- **Pass span:** one full-block pass per training step over [block start - M - (L-1), block end] on the 7-day block, with today's batch composition unchanged. Span-restricted batches cost effective samples. The eval block's series is computed once per validation pass.
- **Gaps:** hybrid:
  - gaps of 60 minutes or less: elapsed-time decay, with previous-tick semantics and a closed-form phantom term for the variance EWMA;
  - longer gaps and the data start: reset, plus an M-bar masked burn-in;
  - serving: the Predictor recomputes from history with the trainer's pass-start rule. It is stateless and needs timestamps.

## Q1 Series kernel (`kernel.py`, `q1_kernel_checks.py` -> `q1_results.json`, `q1_summary.txt`)

**Scale.** The model's input unit is the target scale S = $257.52: the StandardScaler.scale_ of fold -1's training deltas, which WindowNormalizer divides by (computed by `common.target_scale()`). The state `d = EMA - close` is in these units.

**Tolerances (pre-set):**
- forward error <= 1e-5 x max|state| per channel, or <= 1e-4 absolute;
- the same tolerance for split invariance;
- causality bitwise;
- gradients within 1e-3 of float64 finite differences.

**Test grid.** Real closes of the bundled file, T = 30,720 and 43,008. Periods 2, 5, 14, 30, 60, 240, 1440, 10,080, 40,000 and 1e6, plus logit -40 (alpha 4e-18, which is an exact integrator in float32). Alpha both constant and per bar (+-0.5 tanh shift). Channels:
- `d` (increment form);
- RSI gain and loss EWMAs, and the RSI value;
- Bollinger `EWMA(d^2)` (stage 2);
- MACD line and signal for (12,26,9) up to (10080,40000,10080).

| variant (C = 64 and 128) | worst rel err, all channels | tolerance |
|---|---|---|
| **V1 segsum, hierarchical, log-decay, mulsum** | 2.9e-6 (var p1440, C128) | **PASS everywhere** |
| V1e same with einsum | 2.7e-6 | PASS (CPU only, see TF32) |
| V2 local cumsum + subtraction, hierarchical | 2.7e-6 | PASS |
| V3 = first round's mat2 (global cumsum, input 1 - alpha) | up to 1.1e-4 (d, p1e6), 6.7e-4 (var) | **FAIL**, constant alpha p >= 1440 |
| V3b global cumsum, log-decay input | up to 3.0e-5 (var p10080, C64) | FAIL, var p1440-40000 |
| V1 with input 1 - alpha | up to 1.1e-4 | FAIL, p >= 1440 |
| V1e with TF32 emulated (RN / RZ) | 1.8e-4 - 6.8e-3 | FAIL on every channel |

**Findings:**
- **Periods >= 1,440 fail for two reasons.**
  - Rounding 1 - alpha to float32 is a systematic period error (V1 with input 1 - alpha fails).
  - The global cumsum cancels (V3b fails on the variance).
  - The log-decay input with local segment sums fixes both.
- **C = 16 and 32** (`q3_chunk_size.json`): at 43,008 bars the worst relative error is 1.25e-6 and 1.37e-6, and every channel passes.
- **RSI value** (V1): error 1-2e-5 points.
- **Split invariance** (V1, split at 15,360 and 15,001, carried state, both stages): max relative error 5.7e-7, PASS. It is **not bitwise**: 60-145k elements differ.
- **Causality:** perturbing dx and logits after t leaves every output up to t bitwise unchanged at 11 probes (chunk and group edges), for C = 64 and 128, mulsum and einsum. `d out[t]/d dx[t'>t]` and `d out[t]/d logit[t'>t]` are exactly 0, and the past gradients are non-zero, so the test is not vacuous. PASS.
- **Gradients** with respect to the base logits, against float64 central differences (the difference estimate converges to 3e-8, and float64 autodiff agrees to 3e-8):
  - increment channels and the BB/RSI pipeline, constant and per-bar alpha, C = 64 and 128: max 2.4e-5;
  - C = 16 and 32: max 9.7e-6 (`q3_layer_c16.json`);
  - PASS in all cases.
- **Clamps:**
  - logits +-13.8 (alpha 1e-6 and 1 - 1e-6), +-30 and -40: forward and gradients finite, and the gradient matches float64 autodiff within 4e-6 relative;
  - no alpha clamp is needed in the log-decay form;
  - as alpha tends to 0 the state becomes the exact negative cumulative price change since the start (unbounded, |d| up to 50 on 30 days), and the logit gradient vanishes like alpha: that logit freezes, but no NaN appears.
- **Non-finite input:** `check_numerics` raises InvalidArgumentError, both eager and inside tf.function. Without the check, one NaN or Inf at bar 20,000 makes 196,768 of 220,000 earlier outputs non-finite. The check belongs at data load plus the kernel's check flag.
- **Round-1 gate "cold 60-bar window = ewma_sequence_matrix_multi within 1e-6 absolute" FAILS as worded** (`q1_window_repro.json`, 500 windows):
  - max absolute error 2.9e-6 against |state| up to 8.3;
  - today's own layer differs from float64 by 1.8e-6;
  - the relative error is 4e-7, which passes a 1e-5 x max|state| criterion.

## Q2 Window assembly (`q2_assembly.py` -> `q2_results.json`)

Setup: B = 256, L = 60, plus the price-relative term.

**Evidence for what raises on the GPU under op determinism:**
- Binary strings: `det_strings_all.txt`.
- TF v2.10.0 sources:
  - GatherV2's gradient becomes dense through `unsorted_segment_sum` (indexed_slices.py:447);
  - GatherNd's gradient uses `scatter_nd`;
  - ExtractImagePatches' gradient uses `sparse_tensor_dense_matmul`;
  - `segment_reduction_ops_impl.h` sets `use_deterministic_kernels = false` under `PLATFORM_WINDOWS`, so sorted segment ops also raise on this build;
  - `scatter_nd_op.cc` runs `DoScatterNdOnCpu` plus `BlockHostUntilDone` when determinism is on.

Every variant's X equals `tf.gather`'s bit for bit. Every dF is within 1.3e-7 relative of the CPU gather gradient.

| design | fwd+bwd ops (compute) | GPU under determinism | TF32 | largest tensor | CPU ms, N=11,520: C31 / C87 | CPU ms, N=30,720: C31 / C87 |
|---|---|---|---|---|---|---|
| D0 tf.gather | 88 (30) | **raises** (UnsortedSegmentSum) | - | 2-10 MB | 2.0 / 4.9 | 3.1 / 8.5 |
| D0b gather_nd | 75 (27) | host round trip (ScatterNd) | - | same | 2.2 / 6.8 | 3.1 / 8.2 |
| D9 signal.frame | 162 | **raises** | - | 81 MB | 29 / - | - / - |
| D10 extract_patches | 166 | **raises** (SparseTensorDenseMatMul) | - | 81 MB | 51 / - | - / - |
| D1 L slices + one-hot select | 667 (147) | ok | forward | 81-611 MB | 119 / 325 | 305 / 875 |
| D1b L slices + gather backward | 704 (161) | ok | - | 81-611 MB | 88 / 246 | 235 / 670 |
| D2 banded one-hot matmul | 77 (28) | ok | forward and backward | 675-1,800 MB | 211 / 319 | 624 / 886 |
| D3 gather + one-hot backward | 84 (28) | ok | backward | 675-1,800 MB | 174 / 209 | 516 / 607 |
| D6a transpose by gather (map in graph) | 150 (54) | ok | - | 82-612 MB | 24 / 68 | 68 / 180 |
| **D6b transpose by gather (host map)** | **118 (41)** | **ok** | **-** | **82 / 229 / 218 / 612 MB** | **19.5 / 63** | **58 / 166** |
| D7 USS pinned to /CPU:0 | 81 (27) | device round trip | - | small | 2.3 / 5.4 | 3.0 / 9.3 |
| D8 scatter_nd backward | 81 (27) | host round trip plus sync | - | small | 2.3 / 6.0 | 3.0 / 8.4 |

**Recommendation: D6b.**
- It has about 40 launching ops and no matmul, no segment op, no scatter op and no sparse op.
- Its cost is the [N, L, C] gathered gradient: 82 MB at the 7-day block with C = 31, 229 MB with C = 87.
- If memory binds, sum the L gathers with AddN instead (about 120 more ops, [N, C] each).
- It requires distinct anchors per batch (true for today's shuffled epoch), asserted in the data pipeline, not in the graph: an in-graph Assert costs a host sync.

## Q3 Per-step pass span (`q3_span.py`, `q3_chunk_size.py`, `q3_layer_c16.py`, `q3_batches.py` -> JSON)

**Burn-in.** M = burn_in(240, eps 1e-3, -0.5 shift) = 1,365, so the 7-day pass is 10,080 + 1,365 + 59 = **11,504 bars**. As a function of the longest period P: M(60) = 340, M(1,440) = 8,198, M(10,080) is about 57k (estimate from the same formula: roughly 3.45 x P x 1.65).

**Bare kernel pass, fwd+bwd, per-bar alpha, mulsum** (compute ops / largest fp32 tensor / static bytes written / CPU ms):

| | C = 16 | C = 64 | C = 128 |
|---|---|---|---|
| 7-day, K = 31 | 185 / 22 MB / 0.6 GB / 54 | 195 / 87 MB / 2.0 GB / 188 | 194 / 174 MB / 3.8 GB / 467 |
| 7-day, K = 87 | 187 / 61 MB / 1.7 GB / 161 | 199 / 245 MB / 5.5 GB / 602 | 196 / 489 MB / 10.7 GB / 1,235 |
| 30,720, K = 31 | 241 / 58 MB / 1.6 GB / 150 | 175 / 232 MB / 5.2 GB / 553 | 185 / 465 MB / 10.2 GB / 1,191 |
| 30,720, K = 87 | 241 / 163 MB / 4.5 GB / 484 | 175 / 652 MB / 14.6 GB / 1,679 | 187 / 1,305 MB / 28.6 GB / 3,277 |

- Constant and per-bar alpha cost the same.
- The op count does not depend on T or K.
- einsum is 20-25% faster on the CPU but TF32-exposed on the GPU.

**The layer against today's** (B = 256, fwd+bwd; total ops / compute ops / static GB / CPU ms, from `q3_layer_c16.json`):
- today's LearnableIndicators with the meta Dense: 1,232 / 588 / 2.01 / 172;
- A2 at C = 16, 7-day block: 2,000 / 721 / 0.62 / **59**;
- A2 at C = 64, 7-day block: 2,036 / 739 / 1.67 / 175;
- A2 at C = 16, 30,720 bars: 2,354 / 833 / 1.62 / 173.

The A2 layer passes no raise or host-round-trip op. Most of its extra total ops are Const (1,110).

**Batch composition** (effective samples per 256-batch across the price, direction and variance proxies at h10/15/20; the first round's method):

| layout | 7-day block | 30,213 block |
|---|---|---|
| i.i.d. | 261-278 | 251-270 |
| **today (shuffle 2048)** | **165-237** | **150-207** |
| span 4,096 | 144-222 | 136-215 |
| span 2,048 | 95-174 | 90-173 |
| span 1,024 | 59-115 | 52-122 |
| span 256 | 14-38 | 12-39 |

- Today's real-pipeline batches span a median of 7,110 of 10,080 anchors (30,213 block: 11,110), so a span-restricted pass saves little unless the batch composition changes.
- Restricting to spans of 4,096 would roughly cut the pass (at C = 64, 66 ms against 154 ms on the CPU) for an n_eff loss of 5-15%. It is not needed at C = 16 on the 7-day block.

**Eval once.** The val block's series (4,290 bars, forward only) is 277 compute ops, 20 ms on the CPU, **1 pass per validation instead of 12**.

**GPU step-time estimate (estimate).**
- Launches: +133 compute ops (721 against 588) at 5-20 µs per op, about +0.7 to +2.7 ms.
- Traffic: static bytes fall from 2.0 to 0.6 GB, about -3 ms at about 450 GB/s if the bytes are the traffic.
- Net: **about -3 to +3 ms on a 98 ms step** at C = 16 on the 7-day block.
- Counting total ops as the Challenge did (+768) gives +4 to +15 ms; Const, Reshape and Identity do not launch kernels, so that is an upper bound.
- K = 87: add about +2 to +5 ms (1.7 GB for the kernel alone).
- At 30,720 bars with C = 64: traffic bound, about +25 to +60 ms. **Do not run that configuration.**

**Retracing.** M follows the learned periods, so the pass length changes as training moves them. The proposal: allocate M in multiples of 1,024 bars, recompute it at each epoch start from the current base logits minus 0.5, and retrace only when it grows. Refuse a fold whose M exceeds the history before the block (masking the shortfall is the alternative).

## Q4 Gaps and the data start (`q4_gaps.py`, `q4_ffill_runs.py`, `q4_serving.py` -> JSON)

**Bundled file.** 43,500 bars with **0 gaps** and no zero-volume bars. Its only cold start is the data start: M/30,213 of fold -1's training anchors (4.5% at M = 1,365).

**Long file (2017-2025).**
- There is **1 timestamp gap** (1,161 minutes, 2025; 4,598,198 bars). The file is otherwise forward-filled.
- It contains **149,245 flat, zero-volume bars equal to the previous close, in 118,923 runs:**
  - 99% of runs are 4 minutes or shorter;
  - 440 runs are longer than 5 minutes, 15 are longer than 60 minutes, and the longest is 287 minutes (2020-04-25);
  - they are indistinguishable from genuinely quiet minutes.
- **An NT-041-style "drop anchors whose window or label touches a gap" rule that counted these bars would drop 47% of anchors** (mean per 7-day block, p90 86%). NT-041 must treat short filled runs as elapsed time, not as gaps to drop.

**Masked share per 7-day block with reset + burn-in** (456 blocks):

| reset threshold | M = 340 | M = 1,365 | M = 8,198 |
|---|---|---|---|
| > 15 minutes | 0.27% (max 6%) | 1.1% (max 16%) | 6.2% (max 100%) |
| > 60 minutes | 0.11% | 0.45% (max 14%) | 2.7% (max 81%) |
| > 240 minutes | 0.007% | 0.03% | 0.18% |

Resetting at every short run would mask most of the history, so resets are only for long gaps.

**Synthetic series with holes** (36,602 bars: 60 gaps of 1-3 minutes, plus 30 minutes, 6 hours and 2 days; G_reset = 60, M = 1,365):

| check | result |
|---|---|
| E: elapsed-time decay against the kernel on the forward-filled grid | `d` 2.3e-7, RSI 7.2e-8, variance 3.5% without the phantom term and **2.8e-7 with the closed-form phantom term** `exp(la1(dt+1))(1-exp(la1(dt-1)))d_{t-1}^2` |
| after the 2-day outage, without a reset | `d` jumps to -1.7 to -2.5 (typical |d| 0.13-1.08): a stale-state artefact, which is why long gaps reset |
| A1: Predictor pass from the same origin | 29/30 bitwise, max rel 9.5e-9 |
| A2: pass from the last reset, start aligned to the C x G grid | 29/30 bitwise; aligned to C only 20/30; unaligned 17/30; max rel 9.5e-8 |
| A3: cold start M + L - 1 bars back | 1.2e-5 relative (truncation, bounded by eps x state) |
| B: carry-forward, float32 single steps over 18,300 bars | 1.5e-6 relative |

Choice: **recompute from history** (stateless).
- The trainer and the Predictor share one pass-start rule: the first anchor - (L-1) - M_run, clipped at the last reset or the data start, on a grid anchored at the pass start.
- Result: bitwise on the same block, and within eps (1e-3 relative bound, measured 1.2e-5) for live single-anchor serving.

**The bundle stores:**
- the logits and the meta-shift weights;
- the kernel specification (C, G, contraction, log-decay form);
- eps, the M rule and M_run;
- G_reset and the bar size;
- the phantom-term flag;
- the context normalisation constants.

**The Predictor API needs timestamps.** It refuses a history since the last reset shorter than M_run + L - 1. Carry-forward (state plus timestamp plus last close in the bundle) is the optional streaming mode, within 2e-6.

## Q5 What breaks a Challenge must-fix (plainly)

1. **TF32 (precision, new).** `tf.config.experimental.tensor_float_32_execution_enabled()` is True in the nt env, and the repo never disables it.
   - With TF32 emulated on the CPU, any **einsum/matmul form of the kernel fails the tolerances by 20-200x**: 1.8e-4 to 6.8e-3 relative, RSI errors of 0.004-0.03 points, MACD 4e-4 to 2e-3.
   - **Today's windowed EWMA is already affected on the GPU** (`q5_tf32_today.json`): median 2.5e-4 to 7.8e-4 relative, max 0.0055-0.011 units ($1.4-2.9).
   - The one-hot matmul assemblies D1, D2 and D3 are exposed too.
   - The CPU precision measurements do not transfer to a matmul kernel on the GPU. Use the mulsum contraction, or decide on TF32 globally (DECISIONS; it changes today's GPU numbers).
   - These are estimates, not GPU measurements.
2. **The first round's kernel form** (global cumsum, input 1 - alpha) fails the corrected relative tolerance for constant-alpha periods of 1,440 and longer. Its "about 1e-5" precision held only for the short periods measured.
3. **Segment-sum backward:** confirmed. Also, on this Windows build the *sorted* segment ops have no deterministic kernel either, ScatterNd falls back to the host with a blocking sync, and `tf.signal.frame` and `extract_patches` raise.
4. **Round-1 gate (3), 1e-6 absolute, is unmeetable:** today's own layer is 1.8e-6 from float64. Use the relative form.
5. **Speed arithmetic:** C = 64 at 30,720 x 87 materialises 652 MB tensors and about 14.6 GB of writes per step. The Challenge's 684 MB calculation is confirmed. C = 16 and the 7-day block keep it at about 0.6-1.7 GB.

## Proposed stage gates (this part)

- **G-A1 (CPU, QA):** the kernel and assembly tests below pass. The op census of the kernel, the layer and the assembly graphs (fwd+bwd) contains no op from the RAISE or host-round-trip lists (`common.py`) and **no MatMul, BatchMatMul or Einsum in the kernel**.
- **G-A2 (GPU, in the 0a kit run by the experimenter):**
  - the kernel plus D6 fwd+bwd run on the RTX 4070 Ti with op determinism on and without an error;
  - the kernel precision on the GPU meets 1e-5 x max|state| at 43,008 bars **with TF32 at its default**;
  - two identical runs are bitwise equal;
  - the A2 layer step time is measured against today's layer (median of 3 interleaved runs).
- **G-A3 (D-018):** sec_per_step in series mode is within +5% of window mode on a real run. If it is not, the owner decides.

## Proposed backlog items (this part)

1. **Series kernel**, implementer, P1.
   - (1) V1 as specified: log-decay input, segment sums, G = 32, mulsum, default C = 16, carried state in and out, reset by NEG, check_numerics flag.
   - (2) Precision on the bundled file at 30,720 and 43,008 bars:
     - periods {2, 5, 14, 30, 60, 240, 1440, 10080, 40000, 1e6} plus logit -40;
     - alpha constant and per bar;
     - channels `d`, RSI gain and loss, `EWMA(d^2)`, MACD signal;
     - C in {16, 64};
     - tolerance <= 1e-5 x max|state| or <= 1e-4 absolute (test).
   - (3) Two halves with a carried state, same tolerance (test).
   - (4) Causality bitwise at chunk and group edges, plus a zero future Jacobian (test).
   - (5) Gradients within 1e-3 of float64 finite differences; finite at logits +-13.8, +-30, -40 (test).
   - (6) Outputs after a reset are bitwise independent of all inputs before it (test).
   - (7) Non-finite input raises (test).
   - (8) Op census: no RAISE, host-round-trip or matmul op (test).
   - (9) utils/math.py unchanged; fast suite and ruff pass.
2. **Deterministic window assembly (D6b)**, implementer, P1, with the kernel item or after it.
   - X is bitwise equal to tf.gather, and dF is within 1e-6 relative of the CPU gather gradient at N in {11,504, 30,720} and C in {31, 87} (test).
   - The inverse map is built in the data pipeline, which asserts distinct anchors (test).
   - The op census is clean (test).
   - Peak intermediate is reported.
3. **Additions to the INDICATOR_MEMORY switch item** (round 1):
   - pass start = block start - M_run - (L-1), with M_run from the current logits minus 0.5 and eps 1e-3, allocated in multiples of 1,024 and retraced only on growth (test);
   - a fold is refused, or its shortfall masked, when the history before the block is shorter than M_run (test);
   - the eval series is computed once per validation, predict and calibration pass (test: 1 kernel call per pass);
   - the Predictor recomputes with the shared pass-start rule and is bitwise equal on the same block; it refuses a history shorter than M_run + L - 1 since the last reset; it takes timestamps (test).
4. **NT-041 amendment (gap policy):**
   - forward-filled flat zero-volume runs are detected;
   - runs and gaps of G_reset = 60 minutes or less become elapsed time (dt), including the phantom variance term;
   - longer ones reset and mask M_run bars identically in every arm, and the counts are recorded in meta.json;
   - synthetic-holes tests: check E within 1e-6 relative, A1 and A2 bitwise, carry-forward within 1e-5.
5. **TF32 decision** (lead, DECISIONS): either keep TF32 on and require no matmul in precision-critical indicator code (test by op census), or turn it off globally after a measured GPU speed check. The windowed EWMA is already affected.

## Risks

- **GPU:** nothing was measured on the GPU (TF32 effect, launch cost, traffic, CUB cumsum and reduction order). G-A2 exists for that. Determinism of reduce_sum and cumsum on the GPU is assumed from their absence in the exception list.
- **D6 constraints:** it needs distinct anchors per batch, and its [N, L, C] gradient buffer grows with the block and the catalogue.
- **Memory:** at 30-day blocks with K = 87 even C = 16 writes about 4.5 GB per step; the 7-day block is assumed.
- **Changing M:** it forces retraces and, for periods beyond about a day, a burn-in longer than the block. That needs pre-block history, which the bundled file lacks, so fold -1 masks the start.
- **Forward-filled bars** in the long file cannot be told apart from quiet minutes. G_reset = 60 is a judgement.
- **Train/serve differences:** they are bounded by eps (1e-3 relative) unless the pass start matches. The offset-invariance test should use that bound.
- **Inverse-map memory:** the host inverse map is N + L - 1 int32 per step, which is cheap.

## Files (D:/nt_research/wfp/A/)

- **Shared code:** common.py, kernel.py, a2layer.py, smoke_kernel.py, summarize_q1.py.
- **Q1:** q1_kernel_checks.py -> q1_results.json, q1_log.txt, q1_summary.txt; q1_window_repro.py -> q1_window_repro.json.
- **Q2:** q2_assembly.py -> q2_results.json, q2_log.txt.
- **Q3:**
  - q3_span.py -> q3_span.json, q3_span_log.txt;
  - q3_chunk_size.py -> q3_chunk_size.json;
  - q3_layer_c16.py -> q3_layer_c16.json;
  - q3_batches.py -> q3_batches.json.
- **Q4:** q4_gaps.py -> q4_gaps.json; q4_ffill_runs.py -> q4_ffill_runs.json; q4_serving.py -> q4_serving.json.
- **Q5:** q5_tf32_today.py -> q5_tf32_today.json.
- **Evidence:** det_strings_raw.txt, det_strings_all.txt (strings from the TF binary).
