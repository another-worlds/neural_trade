# Plan: removing the fixed input window (NT-053, 2026-09-29)

The second research round that D-032 asked for ("commit to this in another research and write it down
in a plan"). It turns the first round ([../2026-09-28-window-free/](../2026-09-28-window-free/README.md))
and its adversarial review into stages, gates and pre-registered A/B designs. **The owner approves this
plan before its first implementation item is picked** (D-032). Until an A/B adopts a replacement, the
window model stays the default and the golden-run baseline.

Evidence: three CPU investigations, each with its scripts and JSON outputs in this folder:
[A/](A/FINDINGS.md) the series kernel, the window assembly, the per-step pass and data gaps;
[B/](B/FINDINGS.md) the replacement for the per-window adaptive periods, reporting and the warm-up
rule; [C/](C/FINDINGS.md) the purge rule, measured noise, the A/B designs and the comparator's gaps.
Every claim there is labelled Measured (CPU, command given) or Estimate. **Every GPU number in this
plan is an estimate until the benchmark kit (stage 1) measures it.** The prototypes ran in a session
scratch folder (D:/nt_research/wfp/); their paths inside the scripts point there.

## Answer

Remove the window in two steps, each behind a gate:

1. **Indicators over the series (option A2).** The indicators are computed by one causal pass over
   the block's series each step; the network still reads the last 60 bars of them. This removes the
   period ceiling, the cold start at every window and the quadratic memory, and it is the engine the
   indicator catalogue needs. The measured CPU cost of the whole layer at the 7-day block is lower
   than today's (59 ms against 172 ms, relative only; A/ Q3); on the GPU it is estimated at -3 to
   +3 ms on a 98 ms step. A pre-registered A/B (A/B-1) decides whether it becomes the default.
2. **A per-bar model (option B)**, only if a GPU probe finds today's model epoch-bound and A/B-1
   adopted the series memory. Its own A/B (A/B-2) must show at least 2x faster training at
   non-inferior skill. This is a research track after MVP-4.

Four findings of this round change the first round's design:

- **TF32.** TensorFlow 2.10 runs matmul and einsum in TF32 on this GPU by default (measured:
  `tensor_float_32_execution_enabled()` is True). Emulated on the CPU, any matmul form of the
  recurrence misses the precision tolerance by 20-200x, and today's windowed EWMA is probably already
  affected on the GPU (median relative error 2.5e-4 to 7.8e-4, an estimate; A/ Q5). The kernel
  therefore uses elementwise multiply plus reduce_sum, and the benchmark kit measures TF32 on the
  GPU before the lead records a TF32 decision.
- **The window gather's backward** must avoid every op without a deterministic GPU kernel, and there
  are more of them than the first round said: on Windows the sorted segment ops raise as well,
  ScatterNd falls back to the host with a blocking sync, and `tf.signal.frame` and
  `extract_patches` raise (A/ Q2). The chosen assembly ("transpose by gather", D6b) has none.
- **The first round's kernel form fails** the corrected precision tolerance at periods of 1,440 bars
  and longer (A/ Q1). The new form (segment sums, log-decay input, hierarchical, chunk 16) passes
  every channel at 43,008 bars and periods up to 1e6.
- **The A/B designs need many folds, not seeds.** With arm x fold interaction present, a naive test
  on 1-3 folds passes a null 9-31% of the time instead of 5%, and a fold-clustered test on 3 folds
  has 15-42% power (C/ Q3). The designs below use 10-21 judgement folds with one seed each, which
  needs NT-041's long-history folds.

## Stages

Each stage has an objective gate checked at QA time. A stage starts only when the one before has
passed, except where "in parallel" is stated.

| Stage | What | Role | Depends on | Gate (summary; full criteria in the proposed items) |
|---|---|---|---|---|
| 0 | Per-run fixed costs and launches: NT-054 as it stands | implementer | NT-027, NT-046 | NT-054's criteria |
| 1 | Benchmark kit: series kernel V1, assembly D6b, today's layer and the A2 layer, forward and backward, CPU-tested, then run on the GPU | implementer, then experimenter | R1 exited | G-A1 on CPU; on the GPU (G-A2): runs with determinism on without error, precision within 1e-5 x max\|state\| at 43,008 bars with TF32 at its default, two runs bitwise identical, the A2 layer timed against today's layer |
| 1b | TF32 decision (lead, DECISIONS) | lead | stage 1's GPU numbers | a recorded decision: keep TF32 and forbid matmul in precision-critical indicator code (enforced by an op-census test), or turn it off after a measured speed check |
| 2 | GPU probe: is today's model epoch-bound or update-bound? | experimenter | stage 1; NT-026 with fixed lambdas and early stopping off; VAL_BATCH_SIZE (recommended) | the pre-registered classification of spec (0) below; it decides only whether stage 7 may start |
| 3 | Series kernel V1 and assembly D6b in the indicators package | implementer | NT-046 (where indicator code lives) | the kernel and assembly acceptance of A/ "Proposed backlog items" 1 and 2 |
| 4 | INDICATOR_MEMORY switch (window default, series option): per-bar adaptive periods, warm-up rule and history bound, burn-in mask, gap policy, purge-rule test, Predictor series mode, reporting | implementer (several items) | stage 3; NT-041 (gap policy, fold roles) | window mode: `golden_run.py verify` passes; series mode: the gates G-B1 to G-B4 and the A/ Q3-Q4 tests; one real GPU run with sec_per_step within +5% of window mode (D-018; otherwise the owner decides) |
| 5 | A/B-1: series against window indicator memory | experimenter | stage 4; NT-026 (+ amendments), NT-032 (+ amendments), NT-041 | the comparator's verdict under spec (1) |
| 6 | NT-047: OHLCV and the new families on the adopted engine | implementer | NT-046; the owner's answer on indicator forms (question 2 below); A/B-1's verdict for the series engine, or window mode if the owner chooses to build in window mode first | NT-047's criteria plus causality and offset-invariance tests |
| 7 | Per-bar model B as a Models registry entry, then A/B-2 | implementer, then experimenter | stage 2 says epoch-bound (or the owner overrides); A/B-1 ADOPT; NT-041, NT-042 | gates T1-T6 of the first round's ab_spec on CPU; one GPU sanity run at least 4x faster per epoch; then A/B-2's verdict under spec (2) |

Order and placement: stages 1-2 run once R1 has exited (it has, except the STATUS citation check)
and NT-026 exists; stage 0 is NT-054 where it already sits (Continuous, after NT-027 and NT-046);
stages 3-6 sit in MVP-6 between NT-046 and NT-048; stage 7 becomes a research track "R6 window-free
model" after MVP-4. Stage 1's code touches only a new script and the indicators package's future
home, so it is disjoint from NT-027 and NT-028 if it lives under `scripts/bench/`.

## Design

### Kernel (A/ Q1)

The causal recurrence y_t = (1 - a_t) y_{t-1} + a_t x_t, in increment form for price channels
(d = EMA - close, d_t = (1 - a_t)(d_{t-1} - dx_t)), computed as a hierarchical chunked recurrence:
- the decay matrices come from segment sums of the log-decays `la = -softplus(logit) x dt`, never
  from a difference of two cumsums and never from 1 - alpha rounded to float32;
- every cumsum is local to at most C = 16 bars, or to at most G = 32 blocks at the higher levels;
- the contraction is an elementwise multiply plus reduce_sum (no MatMul, BatchMatMul or Einsum, so
  TF32 cannot touch it);
- the state is carried in and out; a reset is `la = -1e30` (decay exactly 0);
- non-finite input raises (`check_numerics`); one NaN would otherwise corrupt most earlier outputs.

Measured on the CPU at 30,720 and 43,008 bars, periods 2 to 1e6, constant and per-bar alpha, every
channel (d, RSI gains and losses, Bollinger EWMA(d^2), MACD line and signal): worst relative error
2.9e-6 against float64 (tolerance 1e-5 x max|state| or 1e-4 absolute); split invariance 5.7e-7;
causality bitwise with a zero future Jacobian at chunk and group edges; logit gradients within
2.4e-5 of float64 finite differences and finite at logits +-13.8, +-30 and -40. The first round's
gate "a cold 60-bar window equals `ewma_sequence_matrix_multi` within 1e-6 absolute" cannot be met
by any implementation, today's layer included (it is 1.8e-6 from float64); the gate uses the relative
tolerance instead.

### Window assembly (A/ Q2)

Forward: a plain `tf.gather` of [B, L, C] windows at the anchors. Backward: a custom gradient
"transpose by gather" (D6b): a host-computed inverse map (bar to batch row; anchors are distinct
within a batch, asserted in the data pipeline), one gather of the padded upstream gradient and a
reduce_sum over L. It contains no segment, scatter, sparse or matmul op. X equals `tf.gather`'s
bitwise and the gradient is within 1.3e-7 of the CPU gather gradient; 118 ops forward and backward.
If memory binds, the one [N, L, C] gather becomes L gathers plus AddN.

### Per-step pass (A/ Q3)

One full pass per step over [block start - M - (L - 1), block end] of the 7-day training block; the
batch composition stays as today (today's batches already span a median 7,110 of 10,080 anchors, so
restricting the pass would need contiguous batches, which cost 5-15% of the effective samples). The
evaluation block's series is computed once per validation pass. At C = 16 on the 7-day block the
kernel writes about 0.6 GB per step against 2.0 GB for today's layer. A 30,720-bar pass at C = 64
would be traffic-bound (+25 to +60 ms, an estimate): avoid it.

### Replacement for the per-window adaptive periods (B/ Q1-Q2)

alpha_{i,t} = sigmoid(logit_i + 0.5 tanh(ctx_t W + b)_i), where ctx_t is today's two context
features (the mean and the max of (close_k - close_t) / scale) computed at every bar over a trailing
60-bar span, through today's small Dense network. At the anchor bars it reproduces today's applied
periods exactly (relative difference 0.0), so A/B-1 changes only the memory. Cost on the CPU: about
1.7x a fixed period on the full 31-channel layer with about the same op count (0.93x). The context
span gets its own wall-clock key, separate from LOOKBACK. D-031 is kept: the report gives the global
learned value plus the per-bar range; the off switch sets the shift to 0 and routes to the
fixed-period kernel (bitwise identical); the frozen twin (NT-033) is the textbook logits plus that
switch. Rejected: a bank of fixed periods (2.8-4.2x the cost; the fallback if the owner wants "one
fixed period per bar" literally) and no adaptation (that is the off switch, not a replacement).

### Reporting (B/ Q3)

Per instance over the evaluation block: the global learned value (in bars and wall-clock time), the
instantaneous per-bar period p5 / p50 / p95 and range, and the effective period (the exact mean lag
of the weights). On price: textbook (dashed), learned adaptive (solid) and learned base without
adaptation (dash-dot; dotted stays reserved for training, D-014). In series mode "textbook" means
the configured period as a full-history EWMA. The "lookback" line of today's figures becomes the
history-bound line. NT-043's view (notebook 07) and NT-048's report take these numbers once series
mode exists.

### Warm-up with no ceiling (B/ Q4, A/ Q3)

D-032 sets no period ceiling. The warm-up follows from the learned periods at run time:
- M(eps) = ceil(ln eps / ln(1 - alpha_min)), alpha_min = sigmoid(min logit - 0.5) (an upper bound,
  because tanh caps the shift; the two-stage channels use their convolution tail). eps = 1e-3.
  Measured: the analytic bound is at least the empirical offset-invariance value at eps 1e-2 to 1e-4
  on the full layer; the version without the shift is too small.
- M_run is recomputed each epoch with a one-epoch margin and allocated in multiples of 1,024 bars,
  so the graph is retraced only when it grows.
- The burn-in mask (after the data start and after long gaps) is fixed at run start, so the
  objective never changes during a run.
- **History bound, derived from the data, not configured:** after each step a logit whose M (with
  its margin) would exceed the history available before the block under the purge rule is
  projected back to that bound, and the count is logged per epoch and flagged in the report. On the
  2017-2025 file the available history is years, so in practice the per-step pass length binds
  first. A run fails loudly at start only if the textbook starting periods themselves cannot be
  warmed.
- The Predictor needs M + L - 1 bars of history since the last reset, or refuses.

Measured drift: realized logit drift is about 5% of Adam's per-step bound (the gradient sign flips);
the worst instance in 84 ablation runs would reach about 1,900 bars after 40 epochs, M about 10,800
bars. Long-period channels grow in magnitude like a random walk (RMS 12 scaled units at 10,080 bars)
while their relative sensitivity stays flat, so the stability harness (NT-038) gets a named
long-memory case, and a per-channel scale normalisation is a pre-registered A/B variant, not a
default.

### Data gaps and the data start (A/ Q4, C/ Q1)

The bundled 30-day file has no gaps. The 2017-2025 file has one timestamp gap (1,161 minutes) but is
forward-filled: 149,245 flat zero-volume bars equal to the previous close, in 118,923 runs (the
longest 287 minutes), which a missing-bar check cannot see. Dropping every anchor near such a run
would drop 47% of the anchors; resetting at every run would mask up to 16% of a block. So:
- gaps and flat runs of 60 minutes or less are elapsed time: the decay uses the elapsed minutes, plus
  a closed-form "phantom" term that keeps the variance EWMA equal to the forward-filled grid
  (measured 2.8e-7 relative);
- longer gaps and runs, and the data start, reset the state and mask an M_run-bar burn-in,
  identically in training, evaluation and the Predictor, with the counts in meta.json;
- the Predictor recomputes from history with the trainer's pass-start rule (bitwise equal on the
  same block; within eps for a live single anchor) and needs timestamps; the serving bundle stores
  the kernel spec, eps and the M rule, M_run, the reset threshold, the bar size and the
  normalisation constants (a bundle format version bump).

### The purge rule (C/ Q1): the lead's decision

**Rule (a+):** the gap between adjacent blocks is max(2 max(H), W + max(H)), where W is the longest
finite window any consumer reads (60 today: the network input, realized_vol, DIRECTION_SKIP,
logreg_lags); the reference gap therefore stays 80 bars and today's anchors are unchanged. An
indicator state reads every earlier bar, as it does live; it resets only at the data start and at
long gaps (above). Judgement folds come after every fold whose score makes a choice.

Why: a fit on a block sees bars only up to that block's last label bar, and a later block's label
increments start after the gap, so no later label reaches any gradient, epoch choice or calibrator,
provided every input path is causal (the recurrence, the context features and the normalisers, fit
on the training block or trailing). An earlier block's labelled bars in a later block's state are
that block's past, available live. Taking D-005 literally (reset and a burn-in inside every gap)
closes no further leak and would cost 12.5% / 43% / 246% / 1,711% of a 7-day training block at
learned periods of 60 / 240 / 1,440 / 10,080 bars (C/ Q1), which the unlimited periods make
infeasible. Recorded as D-034; stage 4's test `tests/test_purge_rule.py` (C/ Q1) pins it.

## Corrected A/B specifications

Full drafts, with every number's source, are in [C/FINDINGS.md](C/FINDINGS.md) Q3; the experimenter
commits each as a SPEC before any GPU time (OPERATING_MODEL). The corrections the first round's
review demanded, and how each is met:

| Review finding | Correction |
|---|---|
| Probe judged on losses recalibrated per arm | judged on val CRPS (`val_crps_loss`), which is grouping- and lambda-free (measured: under 2e-7 relative across batch 256, 1024 and the whole block, while val_loss moves 8-11%); lambdas calibrated once at 256 and frozen |
| Probe cap could not observe "update-bound" | A256 capped at 40 epochs with early stopping off (re-run once at 80 if censored); B1024 at 3x A's reach epoch; a run that never reaches the target counts as never |
| Margins let an arm lose most of the edge | quality margin = one third of arm A's own dev-fold edge over constant variance, measured before the judged runs (planning value 0.0065 in horizon-mean log CRPS) |
| Pair counts | 10-21 judgement folds x 1 seed (measured paired SD 0.0087 x 1.25 inflation for 7-day training, an estimate; 5-day out-of-sample blocks, since 1-day blocks need about 8x the pairs) |
| Speed primary producible by caps | A/B-2: time to a pre-registered quality level; a pair with a run capped before early stopping is inconclusive |
| Coverage band a block can fail | coverage judged paired (\|cov_B - 0.9\| not worse than \|cov_A - 0.9\| by more than 0.01); the per-run band is a diagnostic (0 of 42 v1 runs on fold -2 were inside it) |
| AUC guard-rail could not bind | dropped; AUC minus logreg_lags reported per horizon with paired intervals (arm A has no dev-fold direction skill: 0.489 against 0.492) |
| B1 changed three things | A/B-1 changes only the memory: same ceiling (60), same adaptation context at the anchors, same initial weights per seed (tested) |
| Speed P2 undefined, fails on contention | Hodges-Lehmann on the median epoch time of epochs 1 and later, a one-sided Wilcoxon bound at ln 1.05, one re-time of a run slower than 1.10x its arm's median, runs ordered ABBA; a P2 failure goes to the owner under D-018, not REJECT (simulated false failure 0.03 at 20 pairs) |
| Single 5-day block on the 30-day file | held-out folds of the long history, placed after the dev folds, after NT-041 |

- **(0) Probe `window_free_probe_v0`** (a measurement): A256 against B1024 (LR 2e-3 or 4e-3 picked on
  dev fold -3), 4 seeds each on dev fold -2; epoch-bound iff the seed-mean reach-epoch ratio is at
  most 1.5 and B is faster in wall-clock; update-bound iff at least 3; otherwise inconclusive
  (simulated misclassification under 0.1%). Budget about 1.1 GPU-hours, up to 2.2 with the re-run.
- **(1) A/B-1 `series_memory_v1`**: non-inferiority on horizon-mean log CRPS (primary), paired
  coverage, speed (to the owner if it fails); integrity: each arm beats constant variance at every
  horizon, no non-finite steps, identical anchors. 7-day training, 1-day val and cal, 5-day
  out-of-sample blocks; at least 3 dev folds (arm A only) fix the margin and the run time; then
  max(10, n80) judgement folds (21 at the planning margin, 10 if the dev edge is near 0.040).
  Budget 1.0-4.8 GPU-hours (an estimate); if the recomputed budget after the dev runs exceeds 3
  GPU-hours, the SPEC goes to the owner before any judged run.
- **(2) A/B-2 `window_free_v1`**: speed primary = ln(t_A / t_B) to the pre-registered quality, lower
  bound at least ln 2 (the owner may choose 1.5x or 3x); guard-rails as A/B-1; 20 judgement folds x 1
  seed. Budget 1.7-3.0 GPU-hours (an estimate).

Joint power when the arms are truly equivalent is about 0.77 at 20 pairs (C/ Q3): an inconclusive
result is a real possibility and is accepted up front.

## Proposed backlog items

Added to the backlog once the owner approves; IDs are assigned then. Full acceptance criteria are in
the findings files named.

| Item | Role, priority | Stage | Criteria |
|---|---|---|---|
| Benchmark kit: V1 kernel, D6b assembly, today's and the A2 layer, forward and backward, op census, TF32 check; CPU smoke test | implementer, P1 | 1 | A/ G-A1 and "Proposed gates"; runs on CPU in 2 minutes or less |
| GPU run of the kit and the probe (spec 0) | experimenter, P1 | 1-2 | A/ G-A2; C/ spec (0) |
| VAL_BATCH_SIZE key (default = BATCH_SIZE, golden run passes) | implementer, P2 | 2 | C/ item 6 |
| NT-026 follow-up: a `lambda_calibration: once` option, early stopping off or epochs per arm, a per-run contention record | implementer, P1 | 2, 5 | C/ item 4 |
| Series kernel V1 and D6b assembly in the indicators package | implementer, P1 | 3 | A/ items 1-2 |
| INDICATOR_MEMORY switch with the per-bar adaptation, the warm-up rule, the history bound and the burn-in mask | implementer, P1 | 4 | A/ item 3; B/ items 1-2 (G-B1, G-B2); C/ item 2 |
| Purge rule test and config guard (D-034) | implementer, P1 | 4 | C/ item 1 |
| Predictor series mode and bundle metadata | implementer, P1 | 4 | B/ item 3 (G-B3); A/ Q4 |
| Reporting of per-bar periods (indicator_evolution, NT-043's and NT-048's views) | implementer, P1 | 4 | B/ item 4 (G-B4) |
| NT-041 amendment: forward-filled flat runs, elapsed-time gaps up to 60 minutes, fold roles, 5-day out-of-sample blocks, the read range in meta.json | lead edits NT-041 | 4-5 | A/ item 4; C/ item 3 |
| NT-032 amendment: the 13 comparator gaps | lead edits NT-032 | 5 | C/ Q4 and item 5 |
| NT-038 amendment: a long-memory harness case | lead edits NT-038 | 4 | B/ item 5 |
| A/B-1 | experimenter, P1 | 5 | C/ spec (1); B/ item 6 (secondaries) |
| NT-047 amendment on the owner's answer to question 2 | lead edits NT-047 | 6 | - |
| Per-bar model B and A/B-2 (research track R6) | implementer, experimenter, P2 | 7 | first round's items; C/ spec (2) |

## Questions for the owner

1. **Approve this plan** (D-032)? Recommendation: yes. The staged path gets the indicator
   capabilities first at about today's step cost, and spends GPU time on the per-bar model only if a
   ~1-2 GPU-hour probe supports it.
2. **The new indicator families' forms** (NT-047). In series mode an indicator must be a causal
   recurrence. Textbook Donchian, Stochastic and Williams %R use box windows; textbook OBV is
   cumulative; VWAP is session-anchored. Options: (a) exponential and leaky forms (decayed soft
   max/min, leaky OBV, VWAP as a ratio of EWMAs): offset-invariant and streaming, but they differ
   from the textbook values; (b) finite causal box forms (a soft learnable box over the series):
   closest to the textbook, with a finite memory; (c) build NT-047 now in window mode against the
   registry interface, and add the series forms after A/B-1. Recommendation: (c) for the schedule,
   with (a) as the series forms, reported next to the textbook values (the D-027 view shows the
   difference).

## Risks

- Nothing was measured on the GPU: TF32, launch cost, memory traffic and the determinism of cumsum
  and reduce_sum (inferred from their absence from the exception list). Stage 1 measures them first.
- The noise behind the pair counts was measured on a small change (physics weights) with 23-30k
  training anchors; 7-day blocks and a bigger change may be noisier. The budget rule catches an
  overrun only after the dev runs.
- The edge over constant variance is fold-dependent (2.5-2.9x between folds), so a margin fixed on
  the dev folds is strict on weak folds and loose on strong ones.
- Joint power decays with the number of judged criteria; each needs power of at least 0.95.
- D6b needs distinct anchors per batch, and its buffer grows with the block and the catalogue.
- As learned periods grow, the per-step pass grows with M, and periods beyond about a day need history
  before the block that the bundled file does not have.
- The 60-minute reset threshold for flat runs is a judgement call: forward-filled bars cannot be
  told apart from quiet minutes.
- Evidence continuity: series mode changes numbers by design; the window path stays registered, the
  default and the golden-run baseline until A/B-1 adopts series.
- The ceiling evidence stays ambiguous (B/ Q5): the cold-start artefact is real, but at the learned
  periods the windowed and the warm lines carry the same predictive signal. A/B-1's secondaries
  (drift of the slow periods in each arm) are the deciding evidence.
