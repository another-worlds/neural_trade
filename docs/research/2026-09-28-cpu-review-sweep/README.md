# Review sweep: issues a CPU can check in minutes (2026-09-28)

**What this is.** A review sweep by a remote (cloud) session under D-033: fixes and findings outside the
backlog items. It looked for issues that a 4-core CPU without a GPU can check and fix in minutes, and
that the plans did not have yet. The owner asked for it on 2026-09-27: "focus on computing-instant
issues ... Target issues that aren't included in the already existing plans ... Compile an exhaustive
research and plan for this", and "gpu-free training/inference-free fast results for the repo health".
On 2026-09-28 the owner asked that every contribution serve the overhaul.

**Code checked.** `remediation/plan`. The search agents read ffb1b67 (the code in `src/` is the same from
7785ec9 to ffb1b67). The second-round verifiers and all re-checks used 330ba2f, which adds NT-001
(CI pins), NT-002 (the size-matched random null) and NT-025 (the 5 MB notebook limit). The findings
were checked against the backlog NT-001 to NT-057 and the decisions D-001 to D-033.

**Already delivered from this session.** PR #14 (NT-001): CI pins match the nt env, jinja2 added,
failed tests become annotations. Merged as 6d01d11; CI green on `remediation/plan` (run 36375717146),
the first green head since 609d19e. NT-055 and NT-056 are the lead's follow-ups to it.

## Answer

**Is it possible?** Yes, for most kinds of issue. Measured on this 4-core cloud CPU, without a GPU:

- The 697 fast tests of the CI unit job take about 10 minutes serially and 3 minutes 12 seconds
  with pytest-xdist on 4 cores (697 passed, 192 s); its coverage gates and CLI smoke pass as well.
- The sweep produced 155 findings. An independent verifier checked every one: 148
  confirmed (27 P1, 66 P2, 55 P3; no P0), 5 already in the backlog, 2
  refuted.
- After merging duplicates, they give **33 proposed new items** (CPU-01 to CPU-33,
  in [NEW_ITEMS.md](NEW_ITEMS.md)) and **evidence or corrections for 30
  existing items**.
- A CPU can check and fix: code bugs, statistics and formulas, look-ahead and leakage, test gaps,
  CI and packaging, config validation, import structure, stale code with D-029 evidence, loss terms
  and gradients on tiny tensors, CPU speed of the training-free paths, and the correctness of the
  numbers that figures and the leaderboard show.
- A CPU cannot reach: training on the real defaults (and every `sec_per_step` claim of D-018),
  notebook 01 on the shipped defaults (D-013), sweeps, stability-harness runs, the ablation grid,
  and the research items (NT-003 to NT-006, NT-035, NT-039, NT-050 to NT-052). Findings whose
  effect on a real run needs those are marked; the suspected issues of this kind are listed below.

**Are the runs optimised for the CPU?** They are now. The sweep itself ran without GPU, training or
inference after the owner's request (TensorFlow only for tiny tensor ops, one thread per process,
tests in parallel). For the repository, the next section gives a health tier that needs no GPU, no
training and no inference and takes about 1.5 minutes.

## A fast health tier: no GPU, no training, no inference

Measured here, in a venv built from `requirements-ci.txt` without TensorFlow (CPU-12; INFRA-1,
INFRA-2, INFRA-3, TEST-5):

| Piece | Time | Note |
|---|---|---|
| `ruff check src tests scripts` (ruff 0.6.9) | under 0.1 s | CI lints only `src scripts` today (NT-056) |
| Tests without TensorFlow: `-m "not gpu and not slow and not tf"` | 90 s with `-n 2`, 168 s serially | 541 tests pass; 27 more in that selection still import TensorFlow (3 train or build a model), because the `tf` marker only reads a module's own import lines |
| `scripts/notebooks/build.py --check` and `check.py` | 0.4 s and 1.4 s | already inside the tests |
| Bundled CSV integrity (sha256, 43,500 rows, no gaps, no duplicates, no NaN) | 0.2-0.5 s | no test pins the reference dataset today |
| Load every `configs/**/*.yaml`; `default.yaml` equals `Config()` | about 1 s | no fast test loads `default.yaml` today |
| Import-graph layering check | 0.45 s | joins with NT-027 |
| Scoring golden (report, signals, strategies, backtest) | about 10 s (estimate) | CPU-02 |

The proposal (CPU-12): a `health` CI job without TensorFlow (target at most 3 minutes end to end) and
one local command (at most 2 minutes with `-n 2`):

```bash
$PY -m ruff check src tests scripts && CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not gpu and not slow and not tf" -n 2
```

It needs three small changes first: the `tf` marker also looks at the fixtures a test uses, the
metrics.jsonl reader stops importing TensorFlow (1.8 s saved per figure that reads a run), and
pytest-xdist is pinned (in `requirements-ci.txt`, test infrastructure; in the nt env only with the
owner's approval). Also measured: CI runs its 697 fast tests serially (8.5 minutes); with `-n 4` they
took 192 s here.

## How the work should land

In the pick order of the roadmap (R1, then MVP-1 to MVP-6). The new items are in
[NEW_ITEMS.md](NEW_ITEMS.md); the evidence and corrections for existing items
are in [ITEM_ADDITIONS.md](ITEM_ADDITIONS.md). Each item gives its own timing.

1. **Before the first module move of MVP-1** (they protect every later move and cost little):
   - CPU-01 (P1): `golden_run.py verify` passes when a number turns NaN or an infinity changes.
     Eleven backlog items use it as their "no number changed" proof.
   - CPU-02 (P1): the golden run records only training outputs; the leaderboard's numbers (report,
     signals, strategies, backtest) are unguarded. A training-free scoring golden takes about 10 s.
   - CPU-12, CPU-13, CPU-14 (P2): the health tier, a coverage gate that fails when a gated package
     disappears, and Predictor tests that run in the fast tier.
   - CPU-09 (P1; during NT-026's scorer at the latest) and CPU-31 (P2): tests that pin the
     scorer's and the served numbers. Today 5 of 6 performance-metric mutants and 13 of 15
     backtest-engine and strategy mutants pass every test.
   - CPU-22 (P2): a deny list in `.claude/settings.json` for the git commands CLAUDE.md forbids.
   - Write the corrections for NT-027 into the item before it starts: its layering test misses a
     real 3-cycle (experiments, training, visualization), and any import-line change rebuilds the
     notebooks without outputs.
2. **During MVP-1** (NT-026 to NT-029): the ITEM_ADDITIONS.md blocks for NT-026 (a configuration key, run ids,
   run records, strict JSON, and the AUC definition that flips the M3 verdict), NT-027 (the edge
   inventory), NT-028 (stale candidates with D-029 evidence), NT-029 (validation ranges, fields
   without effect, the drift of `default.yaml`); CPU-16 (plugins in the shared startup) and CPU-17
   (a transactional `Config.override`).
3. **Before MVP-2's sweeps and leaderboard** (NT-030 to NT-034): CPU-03 (the look-ahead guard misses
   one-bar peeks; before NT-033), CPU-04 (temperature scaling stops near T = 1), CPU-05 (the
   loss-weight calibration moves weights it documents as fixed, and turns a 0 into 0.1), CPU-08
   (Diebold-Mariano cells give 26-42% false "significant" verdicts under volatility regimes),
   CPU-10, CPU-11, CPU-21 and CPU-24; NT-031's evidence that a configuration that never trades
   outranks every losing one.
4. **MVP-3, gradient stability:** CPU-07 (the vac_overflow gradient has the wrong sign) and CPU-19
   (the hyper-decoherence term reads price-level spread, not volatility), both maths fixes under
   D-003; NT-036's correction (a guarded step still moves every weight through Adam) and its list of
   16 missing invariants; NT-037's additions.
5. **MVP-4, generality:** CPU-06 (the period ceiling stays at 60 bars when LOOKBACK changes; before
   NT-041), CPU-15, CPU-23 and CPU-25; the corrections for NT-040 (RESAMPLE_MINUTES is not the bar
   size), NT-041 (runs are rebuilt from whatever file CSV_PATH names now) and NT-042 (at N = 4, nine
   functions silently keep three horizons; "h1" is the traded horizon in about 40 places).
6. **MVP-5 and MVP-6, visuals and indicators:** the correction for NT-046, NT-043, NT-033 and NT-048
   (the "textbook" RSI 9/14/21 are EMA spans, that is Wilder RSI 5/7.5/11), CPU-20 (RSI enters the
   network on a 0-100 scale; after NT-046, before NT-047), CPU-28 (before NT-048), CPU-32 and CPU-33.
7. **The window-free plan (NT-053):** additions for the chunk sampler (no partial batch, a full
   shuffle; CPU-18 part b) and for the window start.

## For the owner

- **CPU-30:** an owner decision on the local environment. tensorflow 2.10.1 (the same CUDA 11.2 /
  cuDNN 8.1 line) fixes 50 of the 94 published TensorFlow advisories, and 12 other pinned packages
  have fixes that resolve with the TF 2.10 stack. D-001 requires only TF 2.10.x.
- **CPU-07 and CPU-19** change the loss. D-003 lets the lead fix the physics terms' maths and record
  it; the verifier notes that for vac_overflow the layer's intent and the loss's target contradict
  each other, so the lead may want the owner's view before NT-006.
- Nothing else here changes a recorded decision or the default trading behaviour. Making liberal's
  unreachable low-quality tier trade (NT-016's block) would be such a change; it is not proposed.

## The existing backlog on this CPU

| Reach | Items |
|---|---|
| Fully (code, tests and the golden run on a CPU) | NT-002, NT-010, NT-013, NT-014, NT-016, NT-025, NT-027, NT-028, NT-029, NT-036, NT-040, NT-041, NT-046, NT-049, NT-055, NT-056 (NT-001 and NT-011 done) |
| Partly (code and tests on a CPU; the definition of done also needs a real GPU run or the notebook routine on the real defaults) | NT-015, NT-018 to NT-023, NT-026, NT-030 to NT-034, NT-037, NT-038, NT-042 to NT-045, NT-047, NT-048, NT-054, NT-057 |
| Not reachable (GPU research and measurements) | NT-003, NT-004, NT-005, NT-006, NT-035, NT-039, NT-050, NT-051, NT-052 |
| The lead's research (its CPU prototypes are reachable) | NT-053 |
| Owner decisions | NT-007, NT-008, NT-009, NT-017 |
| Dropped | NT-012, NT-024 |

## Results in numbers

| | Count |
|---|---|
| Findings | 155 |
| Confirmed by an independent verifier | 148 (P1 27, P2 66, P3 55) |
| Already in the backlog | 5 |
| Refuted | 2 |
| No confirmed verdict | 0 |
| Proposed new items after merging duplicates | 33 |
| Existing items with evidence or corrections | 30 (from 90 findings) |

## New items (proposed)

| ID | P | growth point | title | when | findings |
|---|---|---|---|---|---|
| [CPU-01](NEW_ITEMS.md#cpu-01) | P1 | 1 | golden_run.py verify must fail when a number turns NaN or an infinity changes | before the first module move of NT-026, NT-027 or NT-028, and before any golden record that later items verify against | [STRUCT-1](findings/overhaul-structure.md#struct-1), [TOOL-1](findings/static-tools-sweep.md#tool-1) |
| [CPU-02](NEW_ITEMS.md#cpu-02) | P1 | 1 and 2 | A training-free scoring golden: the leaderboard's numbers (report, signals, strategies, backtest, calibration outputs) under 'no number changed' | before the first module move of evaluation/ or strategy/ and before NT-026's scorer | [STRUCT-2](findings/overhaul-structure.md#struct-2), [GEN-11](findings/generality-inventory.md#gen-11), [TEST-3](findings/tests-quality.md#test-3) |
| [CPU-03](NEW_ITEMS.md#cpu-03) | P1 | 2 | assert_no_lookahead catches one-bar peeks in decide, exit_signal and the take-profit level | before NT-033 adds the classic TA strategies (their acceptance relies on this guard) | [STRAT-1](findings/strategy-backtest.md#strat-1) |
| [CPU-04](NEW_ITEMS.md#cpu-04) | P1 | 2 | Temperature scaling reaches the NLL minimum (bounded scalar search instead of fixed-step gradient descent) | before NT-030's first sweep and before the R2 studies read calibrated P(up) | [CALIB-1](findings/calibration-serving.md#calib-1), [FUZZ-1](findings/property-fuzz-sweep.md#fuzz-1) |
| [CPU-05](NEW_ITEMS.md#cpu-05) | P1 | 2 | Loss-weight calibration must not move the weights it documents as never rescaled, nor a weight set to 0 | before NT-030's first sweep and before the NT-006 and NT-039 SPECs | [STAB-14](findings/stability-invariants.md#stab-14), [LOSS-8](findings/losses-training.md#loss-8) |
| [CPU-06](NEW_ITEMS.md#cpu-06) | P1 | 4 | The learned-period ceiling follows LOOKBACK on every override path (interim fix for the window model) | before NT-041 converts the window, before any NT-030 axis over LOOKBACK, and before NT-053's A/B arms | [CORE-1](findings/core-config-cli.md#core-1), [FUZZ-2](findings/property-fuzz-sweep.md#fuzz-2), [TOOL-3](findings/static-tools-sweep.md#tool-3) |
| [CPU-07](NEW_ITEMS.md#cpu-07) | P1 | 3 | vac_overflow: the term is unsatisfiable and its gradient has the wrong sign (fix the maths under D-003) | before NT-006 (R4) and before NT-051 fixes its harness thresholds | [LOSS-4](findings/losses-training.md#loss-4) |
| [CPU-08](NEW_ITEMS.md#cpu-08) | P1 | 2 | Diebold-Mariano cells of the report: HAC long-run variance that survives persistent volatility regimes | before NT-032's comparator reuses the report's tests and before NT-013 reads the verdicts | [EVAL-1](findings/evaluation-metrics.md#eval-1) |
| [CPU-09](NEW_ITEMS.md#cpu-09) | P1 | 2 | Pin the scorer's numbers: the performance metrics, the backtest engine rules and the report's variance block | during NT-026 (4)'s scorer, before NT-031 ranks on net Sharpe and before NT-040; coordinate the Sharpe helper with NT-040 | [STRAT-9](findings/strategy-backtest.md#strat-9), [TEST-1](findings/tests-quality.md#test-1), [TEST-2](findings/tests-quality.md#test-2), [TEST-7](findings/tests-quality.md#test-7) |
| [CPU-10](NEW_ITEMS.md#cpu-10) | P2 | 2 | The gross book of the backtest does not depend on the cost setting | before NT-031 shows any gross column; NT-057 reads the result | [STRAT-2](findings/strategy-backtest.md#strat-2) |
| [CPU-11](NEW_ITEMS.md#cpu-11) | P2 | 2 | BacktestConfig.max_hold: a real override or no control (today the widget knob and the YAML key do nothing) | before NT-029 declares the backtest knobs' metadata and search spaces | [STRAT-3](findings/strategy-backtest.md#strat-3), [NB-7](findings/notebooks-tooling.md#nb-7) |
| [CPU-12](NEW_ITEMS.md#cpu-12) | P2 | 1 | A fast CPU health tier: a fixture-aware tf marker, a TF-free metrics.jsonl reader, one CI job and one local command | before the first module move of NT-027 or NT-026 | [INFRA-2](findings/infra-health-tier.md#infra-2), [INFRA-1](findings/infra-health-tier.md#infra-1), [TEST-5](findings/tests-quality.md#test-5), [INFRA-3](findings/infra-health-tier.md#infra-3) |
| [CPU-13](NEW_ITEMS.md#cpu-13) | P2 | 1 | check_coverage.py fails when a gated package disappears and gates every package | before NT-027's first module move | [INFRA-5](findings/infra-health-tier.md#infra-5) |
| [CPU-14](NEW_ITEMS.md#cpu-14) | P2 | 1 | Predictor logic in the fast tier (a stand-in model, no TensorFlow) and a serving coverage gate | before NT-027 breaks the serving-training import pair | [CALIB-4](findings/calibration-serving.md#calib-4), [TEST-11](findings/tests-quality.md#test-11) |
| [CPU-15](NEW_ITEMS.md#cpu-15) | P2 | 4 | predict --last goes through the same preprocessing as predict_frame | before or with NT-041 | [DATA-3](findings/data-pipeline.md#data-3), [CALIB-2](findings/calibration-serving.md#calib-2) |
| [CPU-16](NEW_ITEMS.md#cpu-16) | P2 | 1 | Plugins load in the shared startup of training and serving, not only in the CLI's train command | during MVP-1, before NT-046's Indicators registry | [CALIB-3](findings/calibration-serving.md#calib-3) |
| [CPU-17](NEW_ITEMS.md#cpu-17) | P2 | 1 | Config.override is transactional, and train_and_evaluate does not change the caller's Config | before NT-026's runner and NT-034's panel reuse a base Config | [CORE-10](findings/core-config-cli.md#core-10), [TOOL-4](findings/static-tools-sweep.md#tool-4) |
| [CPU-18](NEW_ITEMS.md#cpu-18) | P2 | 2 and 3 | Loss-weight calibration samples the whole training block (the training shuffle is near-chronological) | part (a) before NT-039 (its baseline is today's calibration); part (b) in NT-053's chunk sampler or as a pre-registered change | [LOSS-12](findings/losses-training.md#loss-12), [DATA-1](findings/data-pipeline.md#data-1) |
| [CPU-19](NEW_ITEMS.md#cpu-19) | P2 | 3 and 4 | The hyper-decoherence term orders sigma by the window's price-level spread, not by realised volatility (fix the maths under D-003) | before NT-006 (R4) and before NT-053's A/B specifications fix the loss | [LOSS-3](findings/losses-training.md#loss-3) |
| [CPU-20](NEW_ITEMS.md#cpu-20) | P2 | 4 | RSI channels enter the network on a 0-100 scale: scale them, and make every indicator family declare its output range | after NT-046 (whose acceptance keeps every number), before NT-047 adds ten more families; judged by NT-032 | [MODEL-1](findings/model-layers.md#model-1), [STAB-7](findings/stability-invariants.md#stab-7) |
| [CPU-21](NEW_ITEMS.md#cpu-21) | P2 | 2 | Backtest loop about 2x faster with bitwise-identical results | with or after NT-002, before NT-026's scorer and NT-033's TA baselines run many backtests | [INFRA-4](findings/infra-health-tier.md#infra-4) |
| [CPU-22](NEW_ITEMS.md#cpu-22) | P2 | 1 | .claude/settings.json: a deny list for the git commands CLAUDE.md forbids | now (it protects every later session) | [INFRA-7](findings/infra-health-tier.md#infra-7) |
| [CPU-23](NEW_ITEMS.md#cpu-23) | P3 | 2 and 4 | Stop and take-profit fills follow the bar's price sequence; close-mode exits fill at the next open | before NT-041 runs the long history with gaps, or before NT-031 exposes the close mode | [STRAT-5](findings/strategy-backtest.md#strat-5), [TOOL-6](findings/static-tools-sweep.md#tool-6) |
| [CPU-24](NEW_ITEMS.md#cpu-24) | P3 | 2 | Strategy parameter plumbing: from_file, refused derived thresholds, one strategy's YAML knobs, YAML random_seeds | during NT-029 or before NT-030 defines search spaces | [STRAT-7](findings/strategy-backtest.md#strat-7), [FUZZ-6](findings/property-fuzz-sweep.md#fuzz-6) |
| [CPU-25](NEW_ITEMS.md#cpu-25) | P3 | 2 and 4 | Calibration and serving edge cases: the conformal rank clamp, interval names that follow alpha, n/a coverage without a pipeline, one pipeline constructor | before or with NT-041 (short cal blocks at coarse bars) and with NT-026's scorer | [CALIB-5](findings/calibration-serving.md#calib-5), [CALIB-6](findings/calibration-serving.md#calib-6), [CALIB-7](findings/calibration-serving.md#calib-7), [CALIB-9](findings/calibration-serving.md#calib-9) |
| [CPU-26](NEW_ITEMS.md#cpu-26) | P3 | 3 | The scorer and the delta-shrinkage fit handle a non-finite head (today: a crash, or a silent beta = 0) | before NT-038's harness injects NaN into engine scenarios | [EVAL-3](findings/evaluation-metrics.md#eval-3) |
| [CPU-27](NEW_ITEMS.md#cpu-27) | P3 | 1 | Loss documentation numbers: the Casimir bound, the effective L2 coefficient, LAMBDA_DIR, soft ECE | before NT-029 generates the config reference | [LOSS-14](findings/losses-training.md#loss-14), [LOSS-1](findings/losses-training.md#loss-1) |
| [CPU-28](NEW_ITEMS.md#cpu-28) | P3 | 5 | The per-epoch indicator periods are logged before EarlyStopping restores the best weights | before NT-048 reads the per-epoch periods | [LOSS-7](findings/losses-training.md#loss-7) |
| [CPU-29](NEW_ITEMS.md#cpu-29) | P3 | 1 | CLI and logging polish: an invalid log level, user errors as tracebacks, the implicit plugins load that the READMEs deny | after the MVP-1 items | [CORE-11](findings/core-config-cli.md#core-11), [CORE-8](findings/core-config-cli.md#core-8) |
| [CPU-30](NEW_ITEMS.md#cpu-30) | P3 | 1 | Owner decision: local env security updates (tensorflow 2.10.1 locally fixes 50 of 94 TF advisories; 12 non-TF packages have fixes) | whenever the owner chooses; after the D: move (D-030) | [TOOL-9](findings/static-tools-sweep.md#tool-9) |
| [CPU-31](NEW_ITEMS.md#cpu-31) | P2 | 1 and 2 | Pin the served numbers: the calibration pipeline and the extended-trend features; repair a test that skips its own assertion | before NT-027 moves calibration/ and data/ | [TEST-8](findings/tests-quality.md#test-8), [TEST-9](findings/tests-quality.md#test-9), [TEST-10](findings/tests-quality.md#test-10) |
| [CPU-32](NEW_ITEMS.md#cpu-32) | P2 | 5 | Notebook tooling: check.py misses a vanished figure or an empty panel; TrainingSession can train twice into one run | before NT-034 and NT-045 rely on check.py | [NB-3](findings/notebooks-tooling.md#nb-3), [NB-8](findings/notebooks-tooling.md#nb-8), [NB-9](findings/notebooks-tooling.md#nb-9) |
| [CPU-33](NEW_ITEMS.md#cpu-33) | P2 | 5 | The variance figure crashes when a conformal interval covers every sample | any time; before NT-045 moves the figure | [VIZ-1](findings/visualization-correctness.md#viz-1) |

## Additions to existing items

| Item | findings | highest severity | kinds |
|---|---|---|---|
| [NT-003](ITEM_ADDITIONS.md#nt-003) Direction skill: pass the M3 direction clauses (AUC h1 > 0.52, val ... | 1 | P1 | correction |
| [NT-004](ITEM_ADDITIONS.md#nt-004) Price heads: a served delta with positive EV (M3 EV clause) | 2 | P2 | evidence |
| [NT-006](ITEM_ADDITIONS.md#nt-006) Physics-term ablation re-run on the current trainer, pre-registered ... | 1 | P3 | evidence |
| [NT-013](ITEM_ADDITIONS.md#nt-013) Evaluation report: realised-vol variance baseline and honest baseline ... | 1 | P2 | correction |
| [NT-014](ITEM_ADDITIONS.md#nt-014) Evaluation report: stored bootstrap intervals, n/a for constant ... | 1 | P2 | correction |
| [NT-016](ITEM_ADDITIONS.md#nt-016) Backtest data: trade info columns, entry tiers, per-side summary ... | 1 | P3 | correction |
| [NT-018](ITEM_ADDITIONS.md#nt-018) Notebook package UX: explorer window slider with presets, ... | 1 | P2 | evidence |
| [NT-020](ITEM_ADDITIONS.md#nt-020) Head-analytics figures polish at notebook widths ... | 2 | P3 | correction, evidence |
| [NT-026](ITEM_ADDITIONS.md#nt-026) Experiment engine: one scenario and sweep spec, a resumable runner, ... | 14 | P1 | correction, evidence |
| [NT-027](ITEM_ADDITIONS.md#nt-027) Layering: no circular subpackage imports, one metrics and statistics ... | 4 | P1 | correction, evidence |
| [NT-028](ITEM_ADDITIONS.md#nt-028) Stale removal under D-029: every deletion shows evidence of stale and ... | 13 | P1 | correction, evidence |
| [NT-029](ITEM_ADDITIONS.md#nt-029) Config metadata for the control panel and search spaces, and a ... | 13 | P1 | correction, evidence |
| [NT-030](ITEM_ADDITIONS.md#nt-030) Sweeps: quick mode (about 5 minutes) and Optuna mode (measured ... | 1 | P2 | evidence |
| [NT-031](ITEM_ADDITIONS.md#nt-031) Leaderboard ranked by dev-fold net Sharpe after costs, with ... | 5 | P1 | correction, evidence |
| [NT-033](ITEM_ADDITIONS.md#nt-033) Manual-search baselines: frozen-period twin and classic TA rules ... | 1 | P2 | evidence |
| [NT-034](ITEM_ADDITIONS.md#nt-034) Control-panel notebook 06 (ipywidgets + plotly) | 2 | P1 | correction, evidence |
| [NT-035](ITEM_ADDITIONS.md#nt-035) GPU measurements: concurrent-runs throughput and deterministic-mode ... | 1 | P3 | evidence |
| [NT-036](ITEM_ADDITIONS.md#nt-036) Stability invariants in CI (strict mode, masks off) | 7 | P2 | correction, evidence |
| [NT-037](ITEM_ADDITIONS.md#nt-037) Per-run gradient health at most 2% of sec_per_step, per-term probe ... | 6 | P2 | correction, evidence |
| [NT-038](ITEM_ADDITIONS.md#nt-038) Stability harness and config guard (refuse hyperparameter regions ... | 5 | P1 | evidence |
| [NT-040](ITEM_ADDITIONS.md#nt-040) Annualisation ignores the bar size (Sharpe and Sortino overstated by ... | 1 | P1 | correction |
| [NT-041](ITEM_ADDITIONS.md#nt-041) Dataset spec and wall-clock configuration (window, horizons, blocks, ... | 13 | P1 | correction, evidence |
| [NT-042](ITEM_ADDITIONS.md#nt-042) Variable number of horizons | 7 | P1 | correction, evidence |
| [NT-043](ITEM_ADDITIONS.md#nt-043) Learned indicators on price against the textbook defaults (notebook ... | 2 | P2 | evidence |
| [NT-046](ITEM_ADDITIONS.md#nt-046) Indicators package and registry with today's four families | 8 | P1 | correction, evidence |
| [NT-048](ITEM_ADDITIONS.md#nt-048) Discovered-indicators report: a self-contained interactive HTML ... | 1 | P2 | evidence |
| [NT-049](ITEM_ADDITIONS.md#nt-049) Training silently warm-starts from weights in the working directory | 2 | P1 | evidence |
| [NT-053](ITEM_ADDITIONS.md#nt-053) Window-free plan: a second research round that writes the path, gates ... | 3 | P3 | evidence |
| [NT-054](ITEM_ADDITIONS.md#nt-054) Per-run fixed costs and GPU launches (independent of the window) | 5 | P2 | correction, evidence |
| [NT-056](ITEM_ADDITIONS.md#nt-056) CI and packaging hygiene: CI lints tests, actions off Node 20, the ... | 2 | P3 | evidence |

## Findings that are not confirmed

None: every finding has a verdict.

## Refuted findings and findings already planned

The refuted ones are dropped. The ones already planned stay in [ITEM_ADDITIONS.md](ITEM_ADDITIONS.md) as evidence for the item that plans them.

| Finding | status | why |
|---|---|---|
| [DATA-4](findings/data-pipeline.md#data-4) The data pipeline never checks bar spacing or price sanity: windows and horizon targets silently ... | already in the backlog | The finding's backlog_check ('not covered') is wrong for gaps: NT-041(8) covers them. Two residual sub-cases are not in NT-041 and belong there as 'Evidence for NT-041'. (a) Price sanity: a zero or negative Close passes validate_ohlcv_frame (loaders.py:32-41) and inflates the TRAIN-fitted target ... |
| [LOSS-13](findings/losses-training.md#loss-13) Correction to NT-012: acceptance (1)'s contribution list omits the six physics terms (and ... | already in the backlog | Nothing to add beyond two details for NT-037's implementer: the total spans functions.py:771-789 (not 770-788); coherence enters as LAMBDA_COHERENCE x coherence_penalty and the regulariser as 0.1 x LAMBDA_INTER x reg_loss (see LOSS-1). |
| [MODEL-7](findings/model-layers.md#model-7) Config accepts invalid indicator periods: 0 gives all-NaN features, -1 crashes the build, and ... | already in the backlog | Covered by NT-029 (ranges enforced by Config.validate) and NT-046 (per-family parameter bounds). Keep it only as 'Evidence for NT-029: the indicator list/dict fields need element-wise and cross-field checks'. NT-029's test ('one out-of-range value per numeric type') would not exercise list ... |
| [STAB-8](findings/stability-invariants.md#stab-8) Evidence for NT-038/NT-051: EnergyGate feeds the raw window variance and max into a softmax Dense, ... | refuted | What remains is a regime-relative risk inside one block. The 1-day rolling std of 1-minute changes varies 4.6x (max/min) over the bundled 30 days, and the reporter measured 1.6% saturated gates at x1: small. NT-053's research already lists energy_gate.py:24-36 as window-dependent code to redesign ... |
| [STAB-12](findings/stability-invariants.md#stab-12) Evidence for NT-037: the per-step gradient diagnostics today cost as much as the clip itself and ... | already in the backlog | NT-054 was added at 97ad06b, after the 71a0fd2 cut-off. Its acceptance (2) is 'One global-norm computation per optimizer group and fused finite guards; the graph op count before and after is reported'. Its area and 'coordinate with NT-037 (health diagnostics in the same train step)' cover the ... |
| [VIZ-2](findings/visualization-correctness.md#viz-2) Evidence for NT-002: strategy_comparison draws a zero-width '5-95%' random-null whisker from the ... | refuted | The core claim (the engine null has no percentiles, so the registry path always draws a zero-width band) held at ffb1b67. NT-002 fixed it at 330ba2f, as the finding's own fix sketch expected. Residual (P3, latent): trade_analytics.py:709-710 still falls back to the mean when a percentile is ... |
| [VIZ-4](findings/visualization-correctness.md#viz-4) Evidence for NT-028: reachability map of the legacy figure modules (four stale, two live) | already in the backlog | NT-028 (P1, MVP-1) already lists 'figure modules without a production caller', the interactive_plot callback and compat.py. This finding is that item's inventory: paste it into NT-028's why/acceptance rather than filing a new item. Every line number holds at 330ba2f (none of these files changed). ... |

## Suspected issues that need a GPU or long runs

The finders listed these; none of them was checked.

- (strategy-backtest) Exact size of the STRAT-2 gross bias on the real gate runs: m6 calibrated_quantile reports gross_pnl $407.95 against net -23.4%, and my first-order estimate of the cost-free gross is about +4.6%. Checking it needs runs/gates/m6/predictions_test.npz and predictions_cal.npz, which are not in git (then it is CPU-instant: re-run scripts/backtest_gate.py on the patched engine), or a fresh GPU gate run.
- (strategy-backtest) Whether any registered strategy has a one-bar peek on the REAL test block, as opposed to the synthetic _frame fixtures. The strengthened assert_no_lookahead (STRAT-1) would need the saved test/cal PredictionFrames of a run (runs/<id>/artifacts plus predictions), which exist only on the owner's GPU machine; the check itself is CPU-cheap.
- (strategy-backtest) Effect of the STRAT-3 and STRAT-4 dead knobs on published notebook numbers (notebooks/02_backtest saved outputs) needs the notebook's run directory 20260924T182915Z-1aeff1c-dirty-af67ee43, which is local to the owner's machine.
- (data-pipeline) DATA-1: the effect of a full-buffer shuffle on trained results (test AUC and MCC h1, served and raw EV, CRPS and coverage, early-stopping epoch, delta-shrinkage betas) needs fold -1 x >= 3 seeds of 20-epoch GPU runs. Also whether the chronological epochs explain part of the 0.01-0.05 AUC spread between identical GPU runs.
- (data-pipeline) DATA-1: whether the about 33% lower calibrated lambda_vol (and +5% ref_loss) under the current shuffle changes the physics-term or variance verdicts. That needs the NT-006 ablation grid re-run after the fix.
- (data-pipeline) DATA-2 / DATA-4 exposure on the owner's other data: Bitcoin_BTCUSDT.csv (291 MB, on the owner's machine only) could not be scanned for gaps, duplicates or bad ticks here.
- (data-pipeline) Serving on live exchange feeds: whether the newest (in-progress) candle returned by an exchange API is fed to predict_frame / predict_last as if complete. That needs a live feed, not reachable here.
- (evaluation-metrics) Re-scoring the real reference run (runs/20260924T182915Z-1aeff1c-dirty-af67ee43, local to the owner's machine) with the fixed-b DM rule, to see whether the cited 'CRPS DM z 6.2 / 4.0 / 4.0' and 'boot z -2.31' (logreg AUC h1) survive. The predictions are not in this checkout, and a CPU 1-epoch smoke model is not representative.
- (evaluation-metrics) Measuring the actual persistence (autocorrelation) of a trained model's variance-head and P(up) outputs. The simulations assume persistence at or below the real 60-bar realized-vol process; confirming it needs a real GPU-trained run's predictions.
- (evaluation-metrics) Re-executing the notebooks (D-013) after the EVAL-1 and EVAL-2 caption changes in analytics_tables. Notebook 01 trains on the GPU with the full EPOCHS.
- (calibration-serving) Re-fitting the temperatures of the tracked gate runs (m3-m6) and of the reference run with a converged optimiser (CALIB-1). This needs their cal-block prediction npz files, which are gitignored and live on the owner's machine, or a full-length GPU retrain. Only with those can the change in calibrated ECE/Brier and in trade counts (threshold_spike, enhanced_multi_horizon, liberal; agreement votes ...
- (calibration-serving) Whether the realized-vol conformal coverage stays within [0.87, 0.93] on the walk-forward folds (-3, -2, -1) with the real overlapping targets: this needs full-size training per fold.
- (calibration-serving) Whether the no-skill heads seen in the 1-epoch smoke (h0/h1 temperature MLE at the upper bound) are also no-skill at full training. That would decide whether a bounded, converged T flattens the served P(up) to about 0.5 on real runs. It needs a full GPU run.
- (losses-training) LOSS-2 / LOSS-12 impact: whether shuffling the validation stream (or excluding batch-statistic terms from the monitored val loss) and a full-size train shuffle buffer change the served epoch and the fold -1 test AUC / EV. Needs full 20-epoch GPU runs x >= 3 seeds (NT-024 / NT-003 machinery).
- (losses-training) LOSS-3 / LOSS-4 impact on the physics verdicts: whether HD with a realised-vol target reaches VALUE, and what a reachable vac_overflow term does to the variance heads. Needs the NT-006 GPU grid (84 runs).
- (losses-training) LOSS-10: whether LAMBDA_VOL = 0 (or a one-sided vol penalty, or no magnitude ordering) improves raw or served EV h1 on the real network, where memorisation dominates the spread. Needs NT-004 GPU variants.
- (losses-training) LOSS-5: the sec_per_step gain from removing ~370 guard ops per step on the RTX 4070 Ti (D-018 measurement from a real run's status.json).
- (losses-training) LOSS-11: whether dropping the 5-window tail step changes val_loss trajectories measurably. Needs multi-epoch GPU runs.
- (losses-training) Price-head overfitting mechanism (m5/m6 test pred_std/true_std 0.79-0.93 on h1 with EV ~ -r^2): separating memorisation from the objective's pull needs multi-epoch training with and without the candidate terms.
- (losses-training) Whether the calibrated lambdas change materially when calibration samples the whole train block instead of its oldest 17% (LOSS-12 consequence 3). Cheap to measure once data prep is done, but meaningful only with a trained comparison.
- (model-layers) MODEL-1: whether the RSI-driven GRU gate saturation (60% at init and after the 5-step smoke) lasts through a full GPU training run (about 2,400 Adam steps at batch 256), and whether rescaling RSI to [-1, 1] changes direction skill (AUC h1, MCC h1) or the price heads. This needs NT-003-style multi-seed gate runs on dev folds.
- (model-layers) MODEL-2 / MODEL-3: whether switching to Wilder RSI or correcting the Bollinger std changes any gate metric. The CPU checks cover only the definitions.
- (model-layers) Vacuum-noise train/inference shift in the variance heads of a fully trained model. After the 1-epoch smoke the ratio is only 0.98-1.03; a trained model and the cal-fit var_scale are needed to judge the effect on calibration.
- (model-layers) Whether the per-sample meta_adjust period spread (about x0.6 to x1.6 of the base period) grows during full training enough to push applied periods well past LOOKBACK. It needs trained weights; none are committed.
- (overhaul-structure) Whether scripts/golden_run.py record + verify on the same commit are bit-identical (TF_DETERMINISTIC_OPS on CPU with the tf.data pipeline and the lambda-calibration pass): it needs a 2-epoch training run, which the owner's fast-CPU rules forbid for this audit.
- (overhaul-structure) Whether a golden recording made on the owner's Windows machine verifies in Linux CI (oneDNN / CPU instruction-set differences in TF 2.10 kernels): needs the Windows machine and a training run.
- (overhaul-structure) Whether a pure code move that reorders float operations stays within atol 1e-6 / rtol 1e-5 after 2 training epochs (false golden failures): needs training runs.
- (overhaul-structure) The size of the gate-vs-notebook backtest difference (STRUCT-7) on the real m6 run: needs runs/gates/m6/predictions_test.npz and predictions_cal.npz, which are untracked and exist only on the owner's machine (no GPU needed, only the files); m6 temperatures of 0.985-1.033 suggest a small effect for calibrated_quantile and a larger one for strategies that read deltas.
- (overhaul-structure) The per-move cost of re-running notebook 01 (a GPU training run) and 04 if MVP-1 module moves change notebook import lines without re-export shims (STRUCT-3).
- (generality-inventory) GEN-12: how much Casimir and the pooled scaler actually inflate short-horizon sigma with a wide horizon span (5/60/240 minutes) needs training runs, as stability-harness cases (NT-038).
- (generality-inventory) GEN-8: whether losing the 30-bar skip lag (or 3 of 8 skip features at LOOKBACK=12) changes direction AUC or MCC needs paired training runs.
- (generality-inventory) GEN-7: what the collapse of 8 of 18 textbook periods to the window length does to learned-indicator quality and to NT-033's frozen-twin comparison at 5-minute bars needs training.
- (generality-inventory) NT-042 (6): the N=2 and N=4 harness runs themselves (GPU, experimenter). The CPU probes here show only which pure functions crash or truncate.
- (generality-inventory) GEN-11: running scripts/golden_run.py record/verify (2 CPU training epochs) to show a scorer or backtest mutation passing verify was not done under the no-training rule. The claim rests on code reading.
- (stability-invariants) GPU cost of today's guard (about 370 small kernels per step: 123 is_finite + 123 reduce_all + stack + 123 tf.where, custom_model.py:474-484) against the fused version of STAB-4/STAB-12, relative to the ~100 ms step (12 s epochs at batch 256, README.md:54-59); needs sec_per_step on the RTX 4070 Ti.
- (stability-invariants) Whether the per-step tf.device('/CPU:0') floormod on optimizer.iterations and the tf.cond predicate in _update_epoch_metrics (custom_model.py:250-258) force a device-to-host sync every step on the GPU; needs a GPU profile.
- (stability-invariants) Whether trained runs actually push variance outputs below VAR_FLOOR (STAB-6); needs inference on saved runs or NT-037's counters (m6 analytics suggest var_mean 0.41-0.68 with dispersion 0.59-0.70, so probably rare on the reference setup).
- (stability-invariants) Effect on dev-fold metrics of the RSI rescale (STAB-7), the scale-free EnergyGate inputs (STAB-8), the soft variance floor (STAB-6) and the differentiable coherence penalty (STAB-9); each needs training runs and a pre-registered comparison (NT-032).
- (stability-invariants) How often the non-differentiable coherence parts changed the best-val epoch in past runs (STAB-9); coherence is not logged today (NT-037 will log it).
- (stability-invariants) Confirming that the vacuum overflow output collapses to about 0 in trained models (STAB-10; gate logs show the term at 0.096-0.0995, i.e. about its weight); needs the overflow output of a trained model.
- (stability-invariants) EnergyGate and GRU saturation with trained weights (STAB-7/STAB-8 were measured at random initialisations only).
- (static-tools-sweep) How often CalibrationPipeline or lambda-calibration fits actually fail on real (unstable or extreme-input) GPU runs, which is the frequency behind TOOL-7; it needs training plus inference.
- (static-tools-sweep) The effect on training outcomes of Adam still stepping on non-finite steps (TOOL-2) over a run with repeated NaN steps; it needs NT-038's fault-injection harness on the GPU.
- (static-tools-sweep) An end-to-end scripts/golden_run.py record/verify cycle (2 epochs on 3,000 sequences). It is excluded by the fast CPU rules, so TOOL-1 was verified with a monkeypatched run().
- (static-tools-sweep) How the learned periods behave with a stale MOMENTUM_CLIP_MAX when LOOKBACK != 60 (TOOL-3); it needs training.
- (static-tools-sweep) Run-id collisions under real concurrent GPU training processes (TOOL-5, NT-035); reproduced only with two in-process RunContext.create calls.
- (tests-quality) Whether scripts/golden_run.py verify would flag the TEST-1, TEST-2, TEST-7 and TEST-8 mutants in practice: by code reading it records no scorer numbers, but a verify run trains 2 epochs, which the fast CPU rules forbid.
- (tests-quality) The wall-clock cost of the three model-building tests inside the 'not tf' tier (test_contracts training test, 2 x test_calibration_pass): not timed, because running them trains or builds a model.
- (tests-quality) Whether an unmutated CI smoke (1-epoch train, then predict) yields finite lo90/hi90 and has_calibration true: shown only indirectly (the data path provides the raw windows); confirming it needs a training run.
- (tests-quality) Whether the slow tests (test_cli round trip, test_predictor_round_trip, test_notebook_ui training session, test_ablation_harness real cell) would kill any of the surviving backtest or loss mutants: code reading says they assert nothing on Sharpe sign, net or gross, or per-horizon loss wiring, but each run trains.
- (property-fuzz-sweep) FUZZ-1 on the real calibration blocks: how far the stored m3-m6 and current-default temperatures are from the NLL optimum needs each run's cal-block raw direction probabilities, i.e. model inference. The notebook-02 served test-block series (logit sd 0.078; optimum at T -> infinity; fitted 1.07) strongly suggests the stall regime. The effect of the fix on backtest trades and net Sharpe on real ...
- (property-fuzz-sweep) Size of report.dm_z on real persistent forecast errors: on synthetic AR(1) forecasts it is oversized (10.7% false 'beats' at a nominal 5% for phi 0.97, 18.8% for phi 0.99; Bartlett lag 2h = 30). The served P(up) from notebook 02 decorrelates within about 30 bars (acf 0.25 at lag 15, 0.07 at lag 30), so the direction cells look fine. The delta and variance heads' error persistence needs saved ...
- (property-fuzz-sweep) Whether the 60-bar ceiling kept under LOOKBACK overrides (FUZZ-2) changed any past window-length experiment: needs the local runs' config.yaml and indicator_params_history, which are not in the worktree.
- (infra-health-tier) Nightly workflow end to end: the slow suite (training, notebook execution) and the smoke ablation (3 CPU trainings) were not run under the FAST CPU RULES. Its wall time on the GitHub runner is unknown; the check was static plus a fake-runner emulation.
- (infra-health-tier) Wall time of the proposed `health` CI job on the GitHub 4-vCPU runner: estimated at 3 min or less (install without TF about 30 s, tests about 60-90 s at -n auto). Measured here only on a shared 4-core CPU: 91 s at -n 2, 106 s at -n 4 under load.
- (infra-health-tier) Optuna on the Windows nt env: sqlite storage with --parallel N processes on NTFS (0 errors for 4 processes x 50 trials only on Linux ext4), and the lead's real `pip install --dry-run optuna==5.0.0` in the nt env (only simulated with a Windows-platform uv compile of requirements.txt).
- (infra-health-tier) Whether plotly 7.1.0, which `pip install .[viz]` resolves today, renders the committed notebooks' saved figures identically (render.py needs a browser or kaleido; only the 541 non-TF tests were run under 7.1.0).
- (infra-health-tier) Whether the backtest speed-up changes GPU-machine numbers: none expected, since it is the same IEEE arithmetic and bitwise identity was shown on CPU; the golden run was not recorded (it trains).
- (core-config-cli) CORE-1: how much a ceiling that does not follow LOOKBACK changes the learned periods and skill at a window other than 60 bars (or at NT-041's wall-clock windows at other bar sizes) needs GPU training runs.
- (core-config-cli) CORE-2: whether Optuna --parallel batches actually start identical trials in the same second (run-id FileExistsError) depends on the engine and on real sweeps; the collision itself is reproduced on CPU.
- (core-config-cli) CORE-7(d): the effect on results of a plugin-overridden objective or component needs training; only the silent replacement is reproduced.
- (core-config-cli) CORE-4: whether the unused numpy-tier metric values would differ from evaluation/report.py's numbers on a real run needs a trained run's predictions (both call the same numpy_metrics functions, so they are expected to agree).
- (notebooks-tooling) NB-6: share of the 30 s gap between timed epochs that the live redraw takes on the owner's CPU; the in-epoch cost on the GPU of the session's train_epoch_logs() every 25 batches (a float() sync per metric). Needs a GPU notebook run and a CLI run of the same config.
- (notebooks-tooling) NB-1 end to end: explorers scoring a model on a block re-cut from the wrong file. Needs a model loaded with Predictor. Shown here on the split alone; the real overlap depends on the end date of the owner's local Bitcoin_BTCUSDT.csv.
- (notebooks-tooling) tests/test_notebooks_thin.py::test_all_notebooks_execute and the other slow notebook UI tests. They train, so they were not run; the time wasted on 100 null seeds (NB-9) was not measured.
- (notebooks-tooling) scripts/notebooks/render.py: Windows with headless Edge only, not exercised.
- (visualization-correctness) Figures on a real saved run: this worktree's runs/ holds only old gate CSV/JSON artefacts, with no metrics.jsonl, eval_report_*.json or artifacts/meta.json. So the served-epoch marker against a real meta.json, the chance bands with a real fold.val, and the loss-contribution 'other' residual on real logs were checked only on synthetic histories.
- (visualization-correctness) applied_periods and indicator_applied_periods need a trained model and model.predict (inference is forbidden under the fast CPU rules), so the applied-period diamonds and table columns were checked only by reading the code against models/layers/learnable_indicators.py.
- (visualization-correctness) Re-executing notebooks 00-05 end to end (they train or load bundles) to confirm VIZ-1 in a real quick-mode block, and the D-013 rendering of the changed figures.

## How this was checked

- **Areas.** Sixteen search agents, each on one area or one search method: backtest and strategies;
  the data pipeline; evaluation and metrics; calibration and serving; losses and training; model
  layers; structure for the overhaul (import graph, the four experiment paths, stale code, the golden
  run); the N-horizon and wall-clock inventory; gradient-stability invariants; a static-tool sweep
  (ruff with extended rules, pyright or mypy, bandit, deptry, pip-audit); test quality (mutation
  checks); property-based tests (hypothesis); CI, packaging and the health tier (with the Optuna
  check); config, registries and the CLI; notebook tooling; the correctness of the figures' numbers.
- **Verification.** One independent verifier per area rebuilt each reproduction and tried to refute
  each finding (intended behaviour, already planned, not reachable on a CPU, wrong severity). Thirteen
  verifiers read ffb1b67; the last three read 330ba2f. Their corrections (line numbers, severity, the
  growth point and the timing) are part of every item.
- **Rules.** No GPU, no training, no model inference. TensorFlow only for tiny tensor ops (a loss or a
  layer on small arrays, gradients with `tf.GradientTape`). One thread per process, tests in parallel.
  The first six areas (backtest, data, evaluation, calibration, losses, model) ran before the owner
  asked for training-free checks; some of their evidence includes a 1-epoch CPU smoke run.
- **Scope.** Only new issues: everything was checked against BACKLOG NT-001 to NT-057, ROADMAP,
  STATUS, DECISIONS D-001 to D-033 and the archived remediation plan. A finding that shows an existing
  item is wrong is kept as a correction to that item.

## Limits

- **One verifier per finding.** The planned second round (agents that try to disprove every P0 and
  P1 finding from scratch) and the completeness critic with its gap round did not run: the owner's
  spend limit stopped them twice. The verifiers were strict (they refuted 2, marked 5 as
  already planned, changed 17 severities (3 raised, 14 lowered) and corrected many details), but a P1 here has had one independent check, not two.
- **Line numbers.** They refer to ffb1b67 unless a verifier's correction gives 330ba2f. Between the two
  heads only the NT-001, NT-002 and NT-025 files changed; lines of `strategy/backtest.py` below line
  270 moved down by up to 35 lines.
- **Scratch scripts.** The reproductions ran in the session's scratch area (paths such as
  `agents/<area>/<script>.py` in the findings files). They are not committed: each finding records the command
  and the observed output, and QA re-checks every claim (D-033).
- **Not examined.** The owner's local run directories and the long 2017-2025 file (not on this
  machine); the effect of any finding on a real GPU run; the docs as a whole (the doc overhaul rewrote
  them while the sweep ran).
- **Moving target.** The sweep ran while the overhaul pushed 13 commits to `remediation/plan` (VISION,
  DECISIONS D-019 to D-033, the backlog NT-026 to NT-057, the window research, NT-001, NT-002, NT-025). I checked every
  finding's placement against NT-001 to NT-057 when I merged them.

## Files

- [README.md](README.md): this report.
- [NEW_ITEMS.md](NEW_ITEMS.md): the proposed new items in the BACKLOG format.
- [ITEM_ADDITIONS.md](ITEM_ADDITIONS.md): the evidence and corrections for existing items, one block per item.
- [findings/](findings/README.md): every finding with its description, failure scenario, evidence,
  verdict, corrections, fix sketch and proposed acceptance, one file per area.
