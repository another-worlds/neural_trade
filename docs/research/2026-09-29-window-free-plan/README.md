# Plan: removing the fixed input window (NT-053, revised 2026-09-29)

The second research round D-032 asked for ("commit to this in another research and write it down in a
plan"). It turns the first round ([../2026-09-28-window-free/](../2026-09-28-window-free/README.md)) into
stages, gates and pre-registered A/B designs. **The owner approves this plan before its first
implementation item is picked** (D-032). Until an A/B adopts a replacement, the window model stays the
default and the golden-run baseline.

This is the revised plan. Its draft failed an adversarial review ([REVIEW.md](REVIEW.md): 7 must-fix,
15 should-fix, 9 nits); the table "How the review was answered" at the end maps every finding to its
fix. Evidence, each with its scripts and JSON results in this folder:
- [A/](A/FINDINGS.md): the series kernel, the window assembly, the per-step pass, data gaps;
- [B/](B/FINDINGS.md): the replacement for the per-window adaptive periods, reporting, the warm-up rule;
- [C/](C/FINDINGS.md): the purge rule, measured noise, the first A/B drafts, the comparator's gaps;
- [review/](review/): the reviewer's measurements (one-change decomposition, the kernel at 262k bars,
  reach-rule misses);
- [rev/](rev/): the revision's numbers (reach rules, per-fold retention, epoch caps, two-look designs,
  budgets; the scripts' docstrings give their commands). Its numbers are quoted in this README.

Every claim in the evidence is labelled Measured (CPU, command given) or Estimate. **Every GPU number
in this plan is an estimate until the benchmark kit (stage 1) measures it.** Sizing and designs use
development fold -2 only; no test-fold number chose anything (D-020).

## Answer

Remove the window in steps, each behind a gate, and say plainly where a bound remains.

1. **The series engine (option A2):** the indicators are computed by one causal pass over the series
   each step, and the network still reads the last 60 bars of them. It ends the cold start at every
   window and the quadratic memory, and it is the engine the indicator catalogue needs. At the 7-day
   block it is estimated at -3 to +3 ms on a 98 ms GPU step (CPU layer time 59 ms against today's
   172 ms, relative only; A/ Q3). A/B-1 decides whether it becomes the default.
2. **Removing the 60-bar clip** is a separate, later step with its own A/B (A/B-1b), once the series
   engine is adopted. Only then do learned periods grow past 60 bars.
3. **"Unlimited" periods, and the bound that remains.** Nothing configures a ceiling. But a period's
   warm-up needs history and pass length, and both are finite: on the bundled 30-day file the history
   before the first training anchor is whatever burn-in is spent there (a 2,048-bar burn-in bounds the
   longest single-EWMA period at about 360 bars); on the 2017-2025 file the per-step pass length binds
   first. The plan makes that bound explicit, derived from the data and a stated compute budget,
   logged and reported, and asks the owner how to treat it (question 3).
4. **A per-bar model (option B)** only if a GPU probe, run on the setting B would use, finds the model
   epoch-bound, and A/B-1 adopted the series engine. A/B-2 must show at least 2x faster training at
   non-inferior skill. A research track after MVP-4.

Four findings of this round change the first round's design: **TF32** is enabled for matmul and
einsum in this TF 2.10 env (the flag is measured; its effect on this GPU is an estimate), so the whole
indicator layer avoids MatMul, BatchMatMul and Einsum; **the gather's backward** must avoid every op
without a deterministic GPU kernel, and there are more than the first round said (on Windows the
sorted segment ops raise too, ScatterNd falls back to the host, `tf.signal.frame` and
`extract_patches` raise; A/ Q2); **the first round's kernel form fails** the corrected precision
tolerance at periods of 1,440 bars and more, while the new form passes every channel up to 262,144
bars (A/ Q1; review/ r2); **the A/B designs need many folds and one seed each** (a naive test on 1-3
folds passes a null 9-31% of the time instead of 5%; C/ Q3).

## Stages and dependencies

Each stage has an objective gate checked at QA time. "Needs" lists every predecessor; a stage starts
when all of them are done. GPU work is the experimenter's (an implementer never runs a GPU job).

| Stage | What | Role | Needs | Gate (summary) |
|---|---|---|---|---|
| 1 | Benchmark kit in `scripts/bench/` only: kernel V1, assembly D6b, the whole A2 indicator layer (meta-shift included) and today's layer, forward and backward; op census; TF32 check. CPU smoke test | implementer | NT-026 (done) | G-A1: census of the whole layer contains no MatMul, BatchMatMul, Einsum, and nothing from the determinism RAISE or host-round-trip lists; CPU precision within 1e-5 x max\|state\| and 3e-5 x RMS per channel at 43,008 bars |
| 1g | The kit on the GPU | experimenter | stage 1 | G-A2: runs with determinism on without error; precision as G-A1 with TF32 at its default; two runs bitwise identical; the A2 layer's forward+backward time at most 1.10x today's layer's (median of at least 5 interleaved repeats after the GPU-free check) |
| 1b | TF32 decision (DECISIONS) | lead | stage 1g | recorded: keep TF32 with the layer's no-matmul rule enforced by the census test, or turn it off; if turning it off slows training, the owner decides (D-018) |
| 3 | Kernel V1 and assembly D6b in the indicators package | implementer | NT-046, stage 1b | A/ items 1-2 (precision, split invariance, causality bitwise, gradients, reset independence, non-finite refusal at data load, clean census) |
| 4a | INDICATOR_MEMORY switch (window default, series option) with the per-bar adaptation, the warm-up and history-bound rules, the data-start burn-in, the Predictor's series mode, reporting, and the purge-rule test (D-034) | implementer (4 items) | stage 3; NT-046 | window mode: `golden_run.py verify` passes; series mode: G-B1 to G-B4 and the A/ Q3-Q4 tests; the D-018 check below |
| 4b | Gap policy (elapsed-time gaps, flat runs, resets) and fold roles | implementer (NT-041 amendment) | NT-041's own dependencies; stage 4a | A/ item 4, C/ item 3 |
| 5 | A/B-1: the series engine against the window engine | experimenter | stages 4a, 4b; NT-032 with its amendment; NT-041 | the verdict of spec (1) |
| 5b | A/B-1b: removing the 60-bar clip in series mode | implementer (the switch), experimenter | stage 5 ADOPT | the verdict of spec (1b) |
| 6 | NT-047 on the owner's answer to question 2 | implementer | NT-046; question 2; for series forms, stage 4a | NT-047's criteria plus causality and offset-invariance tests, and each family's own M(eps) (below) |
| 7p | GPU probe: epoch- or update-bound, on 7-day blocks with the adopted engine | experimenter | stage 5 ADOPT; NT-041 | the classification of spec (0) |
| 7 | Per-bar model B, then A/B-2 | implementer, experimenter | stage 7p epoch-bound (or the owner overrides); NT-042 | gates T1-T6 (first round) on CPU; one GPU sanity run at least 4x faster per epoch; A/B-2's verdict |

Placement (ROADMAP): stage 1 and 1g may run now, in parallel with MVP-1's remaining items (its files
are only `scripts/bench/`). Stages 3 and 4a sit in MVP-6 after NT-046; NT-048 does not wait for them.
Stage 4a and NT-047 share `models/`, `indicators/` and `core/config.py`, so they run one after the
other: NT-047 first in window mode if the owner picks option (c) of question 2, stage 4a first
otherwise. Stage 4b goes with NT-041 (MVP-4). Stages 5, 5b, 7p and 7 form a research track "R6
window-free" after MVP-4. NT-054 (fixed costs and launches) stays where it is (Continuous, after
NT-027 and NT-046).

**D-018 check for stage 4a** (the review's should-fix 2): the experimenter runs at least 3 interleaved
real runs per mode (window, series) on the same commit, after the GPU-free check, and compares the
median epoch time of epochs 1 and later; series may be at most 5% slower (ratio of medians, with the
spread reported). Otherwise the owner decides under D-018.

## Design

### Kernel V1 (A/ Q1; review/ r2)

The causal recurrence y_t = (1 - a_t) y_{t-1} + a_t x_t, in increment form for price channels
(d = EMA - close, d_t = (1 - a_t)(d_{t-1} - dx_t)), as a hierarchical chunked recurrence:
- decay matrices from segment sums of the log-decays `la = -softplus(logit) x dt`, never from a
  difference of cumsums and never from 1 - alpha rounded to float32;
- every cumsum local to at most C = 16 bars, or at most G = 32 blocks at the higher levels;
- the contraction an elementwise multiply plus reduce_sum (no MatMul, BatchMatMul or Einsum);
- a carried state in and out; a reset is `la = -1e30` (decay exactly 0);
- non-finite input is refused at data load (a finite-input contract), not by a per-step check
  (D-018); one NaN would otherwise corrupt most earlier outputs.

Measured (CPU): worst error 2.9e-6 of max|state| at 43,008 bars and 1.1e-6 at 70,000 and 262,144 bars
of the long file (2.8e-5 of RMS), for periods 2 to 1e6, constant and per-bar alpha, every channel (d,
RSI gains and losses, Bollinger EWMA(d^2), MACD line and signal); split invariance 5.7e-7; causality
bitwise with a zero future Jacobian; logit gradients within 2.4e-5 of float64 finite differences and
finite at logits +-13.8, +-30 and -40. Constant and per-bar alpha cost the same (A/ Q3). The tolerance
is stated against both max|state| and RMS, because max alone is lax for spiky channels. The first
round's "cold window within 1e-6 absolute" gate cannot be met by any implementation (today's layer is
1.8e-6 from float64); it becomes the relative tolerance.

### Window assembly D6b (A/ Q2)

Forward: `tf.gather` of [B, L, C] windows at the anchors. Backward: a custom gradient that transposes
by gather: a host-computed inverse map (bar to batch row; anchors are distinct within a batch, asserted
in the data pipeline), one gather of the padded upstream gradient and a reduce_sum over L. No segment,
scatter, sparse or matmul op. X equals `tf.gather`'s bitwise, the gradient is within 1.3e-7 of the CPU
gather gradient, 118 ops. If memory binds, L gathers plus AddN.

### Per-step pass (A/ Q3)

One pass per step over [block start - M_run - (L - 1), block end]; today's batch composition stays
(its batches already span a median 7,110 of 10,080 anchors; contiguous batches would cost 5-15% of the
effective samples). The evaluation block's series is computed once per validation pass. At C = 16 on
the 7-day block the kernel writes about 0.6 GB per step against 2.0 GB for today's layer.

### Per-bar adaptive periods (B/ Q1-Q2; review/ r1)

alpha_{i,t} = sigmoid(logit_i + 0.5 tanh(ctx_t W + b)_i), where ctx_t is today's two context features
(the mean and the max of (close_k - close_t) / scale) over a trailing 60-bar span, through today's
small network written as an elementwise multiply-add (no MatMul, so TF32 cannot touch it). At the
anchor bars it reproduces today's applied periods exactly (relative difference 0.0). Inside the former
window it is not the same as today: today one alpha, set at the anchor, applies to all 60 bars; in
series mode alpha varies per bar. Measured at fold -1 val anchors with the served weights, that
granularity change alone moves the features by 1.9-6.2% RMS, more than the memory change for 5 of 6
instances (review/ r1). A/B-1 therefore compares engines, not memory alone (spec 1). With the switch
off, every bar uses the base period and the kernel is V1 with a zero shift (bitwise the fixed path by
construction); the frozen twin (NT-033) is the textbook logits plus that switch. The context span gets
its own wall-clock key. Rejected: B's factored kernel (it needs an alpha cap tied to the chunk size)
and a bank of fixed periods (2.8-4.2x the cost; the fallback if the owner wants "one fixed period per
bar" literally).

### Reporting (B/ Q3)

Per instance over the evaluation block: the global learned value (bars and wall-clock time), the
instantaneous per-bar period p5 / p50 / p95 and range, and the effective period (the exact mean lag of
the weights). On price: textbook (dashed), learned adaptive (solid), learned base without adaptation
(dash-dot; dotted stays for training, D-014). In series mode "textbook" is the configured period as a
full-history EWMA. The "lookback" line of today's figures becomes the history-bound line, and the
report counts the steps and instances held at the bound. NT-043's view and NT-048's report adopt these
numbers once series mode exists.

### Warm-up, history bound and pass budget (B/ Q4; A/ Q3; review M2, should-fix 5, 6, 11, 12, 13)

- **M(eps)** is part of every indicator family's registry contract (NT-046): the number of bars after
  which the state's dependence on its start is below eps = 1e-3, computed from the current logits after
  the maximal shift (for a single EWMA ceil(ln eps / ln(1 - alpha_min)), alpha_min = sigmoid(min logit
  - 0.5); cascades from their convolution tail; NT-047's families supply their own). A test per family
  checks it is at least the empirical offset-invariance value.
- **M_run** is the pass allocation: the current M with the margin of one epoch of the largest per-step
  logit change Adam allows (about 3 x lr per step, not lr), grown geometrically (x2) so the graph
  retraces at most a few times per run. After each step each logit is also projected so its M stays
  within the allocated M_run.
- **Burn-in mask:** M_run + L - 1 bars, identically in training, evaluation and the Predictor, after
  the data start and after every reset (long gaps, below). It is fixed at run start (the objective
  never changes during a run) with its length recorded.
- **History bound:** after each step a logit whose M would exceed the history available before its
  block (under the purge rule) is projected to that bound. On the bundled file this history is the
  burn-in spent at the data start, so the burn-in length is the period bound there: 1,024 bars bound
  the longest single-EWMA period at about 180 bars, 2,048 at about 360 (review M2). Default: 2,048 bars
  (about 7% of fold -1's training block on the 30-day file; on the long history the history before a
  block is years and costs nothing).
- **Pass budget:** the per-step pass is at most P_max bars (default: 4 x the training block, 40,320
  bars for a 7-day block: M up to about 30,000, a longest single-EWMA period of about 5,300 bars, 3.7
  days; an estimate from B's M formula). A logit that would need more is projected to it.
- Both bounds are counted per epoch and reported, and a run whose textbook starting periods cannot be
  warmed fails at start, naming the instance. Question 3 asks the owner whether these data- and
  compute-derived bounds are acceptable under "unlimited".
- **Serving:** the Predictor needs M_run + L - 1 bars since the last reset. After an outage longer than
  the reset threshold it refuses for that long (about 40 days at a learned period of 10,080 bars; minutes
  to hours at today's periods). Resets on the long file mask up to 16% of a 7-day block at M = 1,365 and
  up to 81% at M = 8,198 in the worst block (A/ Q4).

### Data gaps and the data start (A/ Q4; C/ Q1)

The bundled file has no gaps; the 2017-2025 file has one timestamp gap (1,161 minutes) and is
forward-filled: 149,245 flat zero-volume bars in 118,923 runs (longest 287 minutes), invisible to a
missing-bar check. Dropping every anchor near such a run would drop 47% of anchors. So: gaps and flat
runs of **60 minutes or less** are elapsed time (the decay uses the elapsed minutes, plus a closed-form
term that keeps the variance EWMA equal to the forward-filled grid; measured 2.8e-7); longer ones, and
the data start, reset the state and mask M_run + L - 1 bars, identically everywhere, counts in
meta.json. The Predictor recomputes from history with the trainer's pass-start rule (bitwise equal on
the same block) and needs timestamps; the serving bundle stores the kernel spec, eps, the M rule,
M_run, the reset threshold, the bar size and the normalisation constants (a format version bump).

### The purge rule: D-034

Gap max(2 max(H), W + max(H)), W the longest finite window any consumer reads (60 today: the network
input, realized_vol, DIRECTION_SKIP, logreg_lags, HD, and the adaptation's context span), so the
reference gap stays 80 bars and anchors are unchanged; the state reads every earlier bar, resetting
only at the data start and long gaps. Judgement folds' read ranges (block start minus the pass length)
must not overlap any choice run's blocks. Recorded as D-034 with its leakage argument; the stage-4a
item adds `tests/test_purge_rule.py` and the config guard.

## Pre-registered designs

Common to every study: lambdas calibrated once per study on the first dev fold and frozen in both arms
(needs the engine's `lambda_calibration: once` option); VAL_BATCH_SIZE = 256 in both arms, so val_loss
is the same function in both and D-011's best-val_loss checkpoint is comparable; one seed per judgement
fold; the RUNBOOK GPU-free check before every run, ABBA order, one re-time of a run slower than 1.10x
its arm's study median; no choice uses a judgement or test fold; A/B-1 is pinned to today's four
families. The numbers below are from dev fold -2 only ([rev/](rev/)).

**Epoch cap** (review should-fix 1): dev runs of both arms on 7-day blocks with early stopping and a
safety cap of 120; then EPOCHS = ceil(2 x the largest served epoch of the dev runs) + EARLY, written into
the SPEC before any judged run; any judged run that hits the cap is re-run with the same seed at
2 x EPOCHS (counted in the budget), so the cap cannot decide the verdict.

**Two looks** (should-fix 8): F judgement folds, then F more only if the first look is inconclusive,
pooled, with Pocock boundaries (one-sided nominal alpha about 0.030 per look for the t-test, 0.031 for
the exact Wilcoxon; total size 0.049-0.051, simulated). Inconclusive after the second look goes to the
owner.

### (0) Probe `window_free_probe_v0` (stage 7p; a measurement)

On 7-day blocks with the adopted engine. Arms A256 (defaults) and B1024 (LR 2e-3 or 4e-3 picked on one
dev fold); 4 seeds per arm on another dev fold; both to early stopping; A's cap from the epoch-cap rule,
B1024's cap 4x A's epochs. **Reach epoch E** = the interpolated epoch at which a run reaches 50% of its
own improvement span (J~(1) - min J~ of the grouping- and lambda-free val CRPS); rho = exp(mean ln E_B -
mean ln E_A); quality guard: B's seed-mean ln min J~ within delta of A's. Epoch-bound iff rho <= 1.5 and
B is faster in wall-clock; update-bound iff rho >= 3; otherwise inconclusive. Simulated on dev fold -2
curves: P(epoch-bound | rho = 1) = 0.95, P(update-bound | rho = 4) = 0.89, no cross-misclassification;
quality-guard false failure 0.002. (The draft's common-target rule classified an equivalent B1024
update-bound about 66% of the time.) Budget about 1.2-1.5 GPU-hours plus the LR pick (an estimate).

### (1) A/B-1 `series_engine_v1` (stage 5)

- **Arms:** A = the window engine (today's defaults); B = the series engine (series memory and per-bar
  adaptation, clip 60 unchanged, same initial weights per seed, tested). Optional attribution arm B0
  (series, adaptation off), run only after an ADOPT, reported, deciding nothing.
- **Blocks:** 7-day training, 1-day val and cal, 5-day out-of-sample; gap 80 (D-034); anchors
  identical, asserted. At least 3 dev folds, both arms (they fix EPOCHS and measure the edge and the
  run time).
- **Primary G1, retention per fold (M5):** r_f = d_f - E_A,f / 3, where d_f is B's paired loss in
  horizon-mean log CRPS and E_A,f is arm A's log edge over constant variance on that fold. B is
  non-inferior iff the one-sided upper bound of mean r_f is below 0; breach iff the lower bound is above
  0. Size at the boundary 0.048-0.051 whatever the fold edges (simulated). SD(r_f) about 0.0095 with
  shared initial weights (an estimate: dev fold -2 measured 0.0065-0.0070 within the fold, plus assumed
  fold terms). Folds for 80% power at true d = 0, by the judged folds' mean edge: 5 (0.040), 8 (0.030),
  15 (0.020), 24 (0.015). **F per look is fixed from the dev runs** (their noise, and the lowest dev-fold
  edge); planning value F = 10.
- **G2, coverage, paired:** |cov_B - 0.9| not worse than |cov_A - 0.9| by more than 0.01 (bounds as G1).
  The per-run band [0.88, 0.92] is a diagnostic only.
- **P2, speed (D-018):** Hodges-Lehmann on ln(median epoch time B / A), epochs 1 and later, one-sided
  Wilcoxon bound at ln 1.05; read on all pairs at the final look. A failure goes to the owner, not to
  REJECT.
- **Integrity:** each arm beats constant variance at every horizon (pooled one-sided bound), no
  non-finite steps, identical anchors.
- **Verdict (intersection-union):** ADOPT iff G1 and G2 non-inferior, integrity holds and P2 passes; if
  only P2 fails, the owner decides; REJECT iff G1 or G2 breaches or B fails integrity; otherwise
  INCONCLUSIVE (second look, then the owner). Power at F = 10 per look: 0.996 at a mean edge of 0.030,
  0.88 at 0.020 (expected folds 11.4 and 14.6).
- **Not judged:** net Sharpe (every arm-A run loses money after costs; pair SD about 20) and AUC (arm A
  has no dev-fold direction skill); both reported with paired intervals. The verdict is on probabilistic
  skill, not on the net-Sharpe yardstick (question 4).
- **Secondaries:** per-horizon CRPS, CRPSS, NLL, PIT, AUC minus logreg_lags, coverage; the share of slow
  periods at the clip and their drift per epoch in each arm; corr(slow MACD line, 59-bar return) per
  arm; per-bar period ranges; sec_per_step and peak memory.
- **Budget (whole SPEC, estimate):** calibration, 6 dev runs, 20 judged runs per look, re-times and cap
  extensions: look 1 only 1.4-3.3 GPU-hours; expected 1.5-4.5; worst case both looks 2.4-5.9. Above the
  3-GPU-hour limit, so question 5. After the dev runs the worst case is recomputed from their measured
  times; if it exceeds what the owner approved, the SPEC returns to the owner before any judged run.

### (1b) A/B-1b `series_no_clip_v1` (stage 5b)

One change: arm A = the adopted series engine with the 60-bar clip; arm B = the same without the clip
(the history bound and pass budget apply). Design, gates and budget as A/B-1. Deciding secondary:
whether slow periods drift past 60 and by how much. Only this study delivers D-032's "unlimited".

### (2) A/B-2 `per_bar_model_v1` (stage 7)

Arms: A = the default of the day; B = the per-bar model (at most 3 variants of chunk length, chunks per
step and LR, chosen on dev folds, recorded before judging). **Primary: fit wall-clock from start to the
end of the served (best-val_loss) epoch**, tracing included; symmetric in both arms, so no target miss
can occur (simulated size 0.048; pass probability at a true 2.5x 0.80-0.82, at 3x 0.99, at 20-24 pairs).
The one-sided Wilcoxon lower bound of ln(t_A / t_B) must reach ln 2 (the owner may choose 1.5x or 3x).
Guard-rails G1 and G2 as A/B-1. Two looks of F = 12. Integrity as A/B-1 plus gates T1-T6 and
`assert_no_lookahead`. Budget (whole SPEC, with the optional A' arm): worst case 2.8-5.6 GPU-hours
(question 5).

## Proposed backlog items

Added once the owner approves; IDs assigned then. Full criteria in the findings named.

| Item | Role, priority | Stage | Criteria |
|---|---|---|---|
| Benchmark kit in scripts/bench/ | implementer, P1 | 1 | A/ G-A1 as above; CPU in 2 minutes or less |
| GPU run of the kit | experimenter, P1 | 1g | G-A2 as above |
| TF32 decision | lead | 1b | a DECISIONS entry; owner if slower |
| VAL_BATCH_SIZE key (default = BATCH_SIZE; golden run passes) | implementer, P2 | 5 | C/ item 6 |
| NT-026 follow-up: `lambda_calibration: once`, per-arm EPOCHS and early stopping off, a per-run contention record, cap extension re-runs | implementer, P1 | 5 | C/ item 4; the epoch-cap rule above |
| Kernel V1 and D6b in the indicators package | implementer, P1 | 3 | A/ items 1-2 |
| INDICATOR_MEMORY switch with per-bar adaptation, M_run, burn-in, history bound and pass budget | implementer, P1 | 4a | A/ item 3; B/ items 1-2; the warm-up section above |
| Purge-rule test and config guard (D-034) | implementer, P1 | 4a | C/ item 1 |
| Predictor series mode and bundle metadata | implementer, P1 | 4a | B/ item 3; A/ Q4 |
| Reporting of per-bar periods and bound counts | implementer, P1 | 4a | B/ item 4 |
| M(eps) in the Indicators registry contract (NT-046 amendment) | lead edits NT-046 | 3-6 | a per-family test against empirical offset invariance |
| NT-041 amendment: flat runs, elapsed-time gaps up to 60 minutes, fold roles and read ranges, 5-day out-of-sample blocks | lead edits NT-041 | 4b | A/ item 4; C/ item 3 |
| NT-032 amendment: the comparator gaps (per-fold retention, one-sided bounds, log ratios, intersection-union, clustered checks, robust timing, anchor hashes, fold placement, two looks) | lead edits NT-032 | 5 | C/ Q4 and item 5; this plan's designs |
| NT-038 amendment: a long-memory harness case | lead edits NT-038 | 5b | B/ item 5 |
| A/B-1, then A/B-1b | experimenter, P1 | 5, 5b | specs (1), (1b) |
| Probe, per-bar model B, A/B-2 (research track R6) | experimenter, implementer, P2 | 7p, 7 | spec (0), first round's items, spec (2) |

## Questions for the owner

1. **Approve this plan** (D-032)? Recommendation: yes.
2. **The new families' forms (NT-047).** In series mode an indicator is a causal recurrence; textbook
   Donchian, Stochastic and Williams %R use box windows, OBV is cumulative, VWAP is session-anchored.
   Options: (a) exponential and leaky forms (decayed soft max/min, leaky OBV, VWAP as a ratio of
   EWMAs): streaming, offset-invariant, different from the textbook values; (b) finite causal box forms
   (a soft learnable box over the series): closest to the textbook, finite memory; (c) NT-047 now in
   window mode against the registry interface, series forms after A/B-1. Recommendation: (c), with (a)
   as the later series forms, drawn next to the textbook values.
3. **The period bound under "unlimited".** No ceiling is configured, but warm-up needs history and pass
   length. Options: (a) project a period at the data-derived history bound and the stated pass budget
   (defaults above: a 2,048-bar burn-in on the bundled file bounds periods at about 360 bars; P_max = 4
   training blocks bounds them at about 5,300 bars on the long history), counted and reported;
   (b) fail the run loudly when a period needs more; (c) truncate the state (a stop-gradient warm state
   cached across steps; biased towards short periods). Recommendation: (a).
4. **Verdict metric.** A/B-1's verdict is on probabilistic skill (per-fold retention of the CRPS edge,
   coverage), not on net Sharpe, which every arm-A run loses today and which is too noisy to judge at
   these fold counts. Accept? Recommendation: yes; net Sharpe is reported beside it.
5. **GPU budgets.** Worst cases (estimates): A/B-1 about 6 GPU-hours, A/B-1b about 6, A/B-2 about 5.6
   with the attribution arm, each above the 3-GPU-hour limit per study; the probe about 1.5. Approve
   these ceilings now, each re-checked from the dev runs' measured times? Recommendation: yes.

A/B-2's speed factor (default 2x) and turning TF32 off (only if stage 1b finds it needed and it slows
training) are asked when those stages come.

## Risks

- Nothing was measured on the GPU (TF32 effect, launch cost, memory traffic, determinism of cumsum and
  reduce_sum); stage 1g measures them first.
- Noise was measured on one dev fold with a small change (physics weights); fold-level terms are
  assumptions, and the 20-epoch cap censors convergence. The designs fix F and EPOCHS from their own dev
  runs, and the budget is re-checked after them.
- The edge over constant variance varies 2.5-2.9x between folds; per-fold retention keeps the size
  exact, but a low-edge regime needs many folds (24 at a mean edge of 0.015).
- D6b needs distinct anchors per batch; its buffer grows with the block and the catalogue.
- The 60-minute reset threshold is a judgement call: forward-filled bars look like quiet minutes.
- Series mode changes numbers by design; the window path stays registered, the default and the
  golden-run baseline until A/B-1 adopts the series engine.
- The ceiling evidence stays ambiguous (B/ Q5); A/B-1's and A/B-1b's secondaries decide it.

## How the review was answered

| Finding | Answer |
|---|---|
| M1 reach rule | probe: 50% of each run's own span, geometric mean; A/B-2: time to the served epoch, quality by the guard-rails (rev/ q1) |
| M2 hidden bound | "Warm-up, history bound and pass budget"; question 3; the A/B contradiction removed (projection, counted) |
| M3 no stage removes the clip | stage 5b and spec (1b) |
| M4 not one change | A/B-1 compares engines; granularity measured and stated; optional B0 attribution |
| M5 margin | per-fold retention r_f (rev/ q2) |
| M6 budget | whole-SPEC budgets with two looks and contingencies (rev/ q5); question 5 |
| M7 placement | dependency table; 4a/4b split; R6 after MVP-4; NT-048 free; 4a and NT-047 ordered |
| S1 epoch cap | the epoch-cap rule (rev/ q3) |
| S2 single-run D-018 gate | 3 interleaved runs per mode by the experimenter; G-A2 threshold 1.10x |
| S3 TF32 | census of the whole layer; shift network elementwise; off switch = V1 with zero shift |
| S4 1.7x claim | removed; V1 per-bar costs the same as constant alpha; factored kernel rejected |
| S5 Adam bound | margin about 3 x lr per step; projection to M_run each step |
| S6 mask and threshold | M_run + L - 1 everywhere; 60 minutes |
| S7 probe setting | stage 7p on 7-day blocks with the adopted engine |
| S8 re-run | Pocock two looks (rev/ q4) |
| S9 test-fold sizing | dev fold -2 only |
| S10 missing evidence | copied into A/ and C/ |
| S11 warm-up per family | M(eps) in the registry contract, tested per family |
| S12 serving and masking costs | stated in the warm-up section |
| S13 retracing | geometric growth |
| S14 owner questions | questions 3-5; the speed factor and TF32 when due |
| S15 family set | A/B-1 pinned to today's four families |
| Nits | preambles removed; HD and the context span in W; RMS tolerance added; read-range wording; M per epoch with per-step projection; finite-input check at load; stage 1 in scripts/bench/ only; joint power at F = 10 stated; A/B-2 uses the common contention rules and study-level lambdas |
