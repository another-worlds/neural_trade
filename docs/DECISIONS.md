# Decisions

Append-only log. An entry changes only through a new entry that cites new evidence ("Supersedes
D-0xx"). Owner decisions change only with the owner. Format: context, decision, consequence,
evidence. From D-019 on, an owner entry quotes the owner, points to the VISION section or the
`docs/qa/` round instead of restating it, adds only what VISION does not say, and gives the
consequence as backlog item IDs; the lead's readings are labelled. When you append an entry, add its
row to the index.

## Index

"By" is who decided: the owner, or the lead (the agent sessions). "See also" points to later entries
that extend or partly replace an entry.

| ID | Title | By | See also |
|---|---|---|---|
| D-001 | Stay on TensorFlow 2.10 / Keras 2 | owner | |
| D-002 | All nine component registries, wired into the training path | owner | D-027 |
| D-003 | Keep the six physics terms; fix the maths; judge them by pre-registered ablation | owner | D-025 |
| D-004 | Never rewrite git history | owner | D-017 |
| D-005 | Four-way purged split; calibration on its own block; test touched once | lead | D-020, D-022 |
| D-006 | Direction loss: binary cross-entropy | lead | |
| D-007 | Serve a shrunk price delta fitted on cal | lead | |
| D-008 | Conformal intervals scaled by the window's realised volatility | lead | |
| D-009 | Default strategy: calibrated_quantile | lead | |
| D-010 | Batch 256 and training diagnostics every 10 steps | lead | |
| D-011 | The served weights are the best-validation epoch | lead | |
| D-012 | Noise-aware statistics everywhere | lead | D-025 |
| D-013 | Notebooks are generated, executed on the real defaults, committed with outputs | owner | D-028 |
| D-014 | Rich figures, one visual system | owner | D-022, D-027 |
| D-015 | Plotly hover formats never start with '+' | lead | |
| D-016 | How the project is run: roles, loop limits, escalation | owner | D-024, D-029 |
| D-017 | Autonomous work through the backlog; push policy | owner (autonomy); lead (push rule, revocable) | |
| D-018 | Training speed is first-class; inference speed is not | owner | D-026 |
| D-019 | The vision: an indicator-learning predictor; BTC/USDT 1-minute is the reference setup | owner (NT-017 at P3: lead's) | |
| D-020 | Deliverable and yardstick: indicators plus predictions, a leaderboard on dev-fold net Sharpe | owner | |
| D-021 | MVP order: foundations first; the old roadmap folds in | owner (generality's and MVP-6's places, milestone names: lead's) | |
| D-022 | Generality designed in; only the reference setup tested in the MVP | owner (24/7 market, N = 3 scaling: lead's reading) | |
| D-023 | Experiment engine first; a control-panel notebook; quick and Optuna sweeps | owner | D-024 |
| D-024 | GPU use for sweeps | owner (A/B studies keep the old limit, one-night cap: lead's reading) | |
| D-025 | Verdict rule: paired test plus a pre-registered minimum effect | owner (verdict folds: lead's reading) | D-046 |
| D-026 | Gradient stability: invariants, per-run health, stress harness, config guard | owner | |
| D-027 | Indicators extend through a registry; learned indicators on price first; D-014 on every figure | owner | D-031 |
| D-028 | Notebooks persist and evolve | owner | |
| D-029 | Delete only what is stale and has no effect | owner | |
| D-030 | The project moves to D:/neural_trade | owner (NT-009 closed by the move: lead's reading) | |
| D-031 | The indicator catalogue | owner | D-032 |
| D-032 | The fixed input window goes; the path comes from a written plan | owner (purge rule: lead) | |
| D-033 | Remote sessions run review sweeps; the lead QA's and merges their pull requests | owner (fetch between items: lead's reading) | |
| D-034 | The purge rule for indicators with unbounded memory: label overlap, gap kept at 80 | lead (delegated by D-032) | |
| D-035 | Models and effort: Sonnet for implementer, QA and experimenter; the lead escalates | owner | D-036 |
| D-036 | A cheap tracker agent for task tracking; the other roles keep the strong model | owner | |
| D-037 | The window-free plan is approved; NT-047 in window mode first; periods held at the data bound; study budgets | owner | |
| D-038 | The old C: copy is ignored; pushing stays automatic | owner ("push auto": lead's reading) | |
| D-039 | The window-free build-up (NT-064 to NT-068) moves after the MVP, into R6 | owner | |
| D-040 | One 360-day training run on the long history, set up by the lead, tracked in a notebook | owner (setup choices: lead's) | |
| D-041 | The micro-scale loop: minutes-long runs drive hypothesis iteration toward predictive power and PnL | owner (protocol: lead's) | |
| D-042 | No pinging: the lead waits for completion notices; active polling goes to the Haiku tracker | owner | |
| D-043 | Model split: Opus for the lead, P0/P1 QA and research; Sonnet for implementer, experimenter and P2/P3 QA | owner (the split: lead's proposal) | |
| D-044 | Trading costs are 0; no tick order-book data | owner | |
| D-045 | The maths report's 22 recommendations are implemented by a staged plan: measure, prune, reweigh, rebuild, generalise | owner (the plan: lead's) | |
| D-046 | Verdicts are inferred over folds, at least 5 judgement folds; one fold x seeds is not enough | lead | |
| D-047 | NT-047's default: OHLCV input with all 14 indicator families; the 1.6x slower step is accepted | owner | |
| D-048 | Tiny first: maths and stability checks on the 6-hour screen layout; test-run discipline; pytest-xdist | owner | |
| D-049 | NT-099 may finish its two open cells past the 3-hour cap | owner | |
| D-050 | No new data source; the micro loop does not go looking for taker-buy volume | owner | |
| D-051 | Strategies: raw heads for coherence, served delta for size | owner | |
| D-052 | NT-006 may spend about 7-10 GPU-hours on the physics re-run | owner | |
| D-053 | Epoch selection may grow a switch; validation loss stays the default | owner | |
| D-054 | remediation/plan is not merged into master yet | owner | |
| D-055 | File a P3 item for Predictor.predict GPU latency | owner | |
| D-056 | The repository licence is MIT | owner | |

## D-001 Stay on TensorFlow 2.10 / Keras 2 (owner, 2026-09-22)
- **Context:** TF 2.10 is the last release with native Windows GPU support; the owner trains on a
  Windows RTX 4070 Ti.
- **Decision:** no Keras 3 / newer TF migration.
- **Consequence:** CUDA 11.2 / cuDNN 8.1 in the `nt` conda env; `import neural_trade` puts the
  env's CUDA DLLs on PATH; ptxas-missing log lines are harmless; XLA JIT ops must be pinned to CPU.

## D-002 All nine component registries, wired into the training path (owner, 2026-09-22)
- **Decision:** Models, Losses, Metrics, Optimizers, Callbacks, DataLoaders, Preprocessors,
  Layers, Visualizations - each strict, each queried by the trainer / predictor.
- **Evidence:** `tests/registries/test_contracts.py`.

## D-003 Keep the six physics terms; fix the maths; judge them by pre-registered ablation (owner, 2026-09-22)
- **Decision:** T-perp, Casimir, HD, IFE, vacuum and vacuum overflow stay. Their value is decided
  only by `configs/ablation_criteria.yaml` (written before the grid ran).
- **Status:** grid `runs/ablations/ablate_physics_v1-full` (84 runs): no term reaches VALUE; the
  family's variance VALUE is withdrawn by the h1 direction-AUC guard-rail (-0.0119 vs tolerance
  0.01). See BACKLOG for the follow-up.

## D-004 Never rewrite git history (owner, 2026-09-22)
- **Decision:** no force-push, no rebase of pushed commits, no history filtering. Work happens on
  `remediation/plan` (the working branch); merging into `master` is the owner's. The push policy
  is D-017.

## D-005 Four-way purged split; calibration on its own block; test touched once (2026-09-22)
- **Decision:** train / val / cal / test with an 80-sequence gap (lookback + longest horizon).
  Early stopping and LR schedule on val; temperature, conformal and delta shrinkage on cal; test
  only for the verdict. Walk-forward folds for experiments.
- **Evidence:** `tests/test_data_processor.py` (purged split, leak count), `runs/gates/REPORT.md` M4.

## D-006 Direction loss: binary cross-entropy (2026-09-23)
- **Context:** focal + dice has a constant extreme as its optimum (the old one-class collapse).
- **Decision:** `DIRECTION_LOSS: bce` (a proper scoring rule) plus `DIRECTION_SKIP` (a linear
  logit from trailing-return features).
- **Evidence:** `runs/gates/REPORT.md` (m5), `runs/experiments/direction_v1/REPORT.md`.

## D-007 Serve a shrunk price delta: beta = clip(E[yd] / E[d^2], 0, 1) fitted on cal (2026-09-23)
- **Context:** the raw price heads overfit (negative explained variance on test).
- **Decision:** `DELTA_SHRINKAGE: true`. beta = 0 is a legitimate outcome: the served delta is
  then exactly 0, and every "served delta" statistic must be reported as n/a (not as a measured
  0% / 100%), with the raw heads shown alongside.
- **Evidence:** gate m6; commits 1863d05, 9a0d070 (beta = 0 handling in figures and report).

## D-008 Conformal intervals scaled by the window's realised volatility (2026-09-23)
- **Decision:** `CONFORMAL_SCALE: realized_vol`. Coverage 0.90-0.91 at a 0.90 target on test.
- **Evidence:** `runs/gates/REPORT.md` M4.

## D-009 Default strategy: calibrated_quantile (2026-09-23)
- **Context:** the notebook strategies' fixed 0.55 / 0.45 lines are rarely crossed by calibrated
  probabilities.
- **Decision:** `Strategies.default = "calibrated_quantile"`: entry lines are quantiles of the
  weighted P(up) on the calibration block.

## D-010 Batch 256 and training diagnostics every 10 steps (2026-09-24)
- **Context:** training is kernel-launch bound; a step costs about the same at batch 64 or 256.
- **Decision:** `BATCH_SIZE: 256`, `TRAIN_METRICS_EVERY: 10` (training loss and all validation
  metrics stay exact). About 7.7x faster epochs; test AUC matches the batch-64 gate run m6.
- **Evidence:** commit 6a72613; README "Performance".

## D-011 The served weights are the best-validation epoch (2026-09-24)
- **Context:** Keras 2.10 `EarlyStopping(restore_best_weights=True)` restores only when it stops
  early, so a run that reached `EPOCHS` evaluated, calibrated and bundled its LAST epoch.
- **Decision:** after `fit`, the trainer restores the best-validation weights whenever early
  stopping did not fire, and records `weights_epoch` / `weights_val_loss` in the TrainResult,
  `artifacts/meta.json` and `status.json`.
- **Consequence:** runs before commit 53c0df2 (including the physics ablation grid) were scored
  on their last epoch.
- **Evidence:** `tests/test_served_epoch.py`; commit 53c0df2.

## D-012 Noise-aware statistics everywhere (2026-09-24)
- **Decision:** intervals and chance bands use effective samples (`n_eff = N // horizon bars`)
  or a block bootstrap / HAC where that is measurably closer; correlation bands are drawn in r
  units (`stats.corr_null_r`), not on the Fisher-z scale.
- **Evidence:** `src/neural_trade/visualization/stats.py`, `tests/test_viz_noise_bands.py`.

## D-013 Notebooks are generated, then executed on the real defaults and committed with outputs (owner, 2026-09-24)
- **Context:** the owner wants to open a notebook and see real outputs; toy runs missed real
  defects.
- **Decision:** notebooks are built by `scripts/notebooks/build.py` (never edited by hand),
  executed in place on the shipped defaults (01 trains the full `EPOCHS` on the GPU), checked
  (`scripts/notebooks/check.py`), every changed figure rendered and looked at, then committed
  with outputs. Each notebook stays under 5 MB.
- **Evidence:** `tests/test_notebook_tooling.py`, `tests/test_notebooks_thin.py`. The nbstripout
  filter and hook, which would strip the outputs, were removed (NT-011, a0edba7).

## D-014 Rich figures, one visual system (owner, 2026-09-24)
- **Context:** the owner judged the first rebuilt figures "oversimplified" versus the old
  `inference.ipynb` (pre-rewrite version: `git show cb53872^:inference.ipynb`).
- **Decision:** figures show every logged metric per horizon with noise references, plus the
  tables the old notebook printed. All use `visualization/theme.py`: horizons h0 #3987e5,
  h1 #d95926, h2 #199e70 (only for horizons); dotted = training; status colours only for
  good / bad outcomes, always with a shape.

## D-015 Plotly hover formats never start with '+' (2026-09-24)
- **Context:** plotly.js prefixes such a format with '~', d3 rejects it and prints the raw number.
- **Decision:** use `>+.3f` style formats. Enforced for every visualization module by
  `tests/test_viz_trading.py::test_every_visualization_module_writes_only_plotly_valid_number_formats`.

## D-016 How the project is run: roles, loop limits, escalation (owner, 2026-09-25)
- **Decision:** the operating model in [OPERATING_MODEL.md](OPERATING_MODEL.md): owner / lead /
  implementer / QA / experimenter roles; at most two repair rounds per item per session;
  findings triaged into the backlog; recorded decisions not re-litigated; owner questions
  parked in STATUS instead of guessed.

## D-017 Autonomous work through the backlog; push policy (owner request 2026-09-23 / 2026-09-25; push rule proposed by the lead)
- **Context:** the owner asked to "continue with the plan fully autonomously until completion"
  (2026-09-23) and for "long term autonomous multi-session development" without repeating
  instructions (2026-09-25).
- **Decision:** after an item is recorded, the lead takes the next actionable item in the same turn
  and does not wait for "continue". Owner questions go into STATUS "Waiting for the owner" instead of
  interrupting the run. The run stops only when no actionable item is left or the owner says stop.
  The lead pushes `remediation/plan` after each integrated item and at handoff, and implementers may
  push their own `nt-<id>` branches to run CI, without asking; never `master`, never `--force`.
  This project rule takes precedence over the global "ask before pushing" default. The push part is
  the lead's proposal that follows from the autonomy request: **the owner may revoke it**.

## D-018 Training speed is first-class; inference speed is not (owner, 2026-09-24)
- **Context:** "I want fast GPU training and solid optimization. Inference I believe is negligible."
- **Decision:** a change to the per-step training path must not slow training (`sec_per_step` in a
  real run's `status.json`, compared with the previous run), unless the owner accepts the cost.
  No work on inference speed unless the owner asks.
- **Evidence:** commit 6a72613 (about 7.7x faster epochs; D-010).

## D-019 The vision: an indicator-learning predictor; BTC/USDT 1-minute is the reference setup (owner, 2026-09-25 / 2026-09-28)
- **Owner (2026-09-25):** "the project IS NOT about just 60 minute BTCUSD, it shouldn't be constrained
  to a timeframe or a ticker" and "It is a substitute for manual technical indicator search and
  analysis."
- **Decision:** as VISION "Purpose", "The reference setup" and "Audience" (approved as written,
  2c58370). It replaces the vision the lead inferred on 2026-09-25 (STATUS question 1, answered).
- **Not in VISION:** the licence. The owner left the licence open (Round 9); the lead lowers NT-017
  to P3 because the MVP audience does not need one.
- **Consequence:** backlog text that framed BTC 1-minute as the purpose calls it the reference setup.
  NT-017 stays open at P3. The growth points toward the MVP: D-020 to D-031.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) owner statement of
  2026-09-25, Round 9, "Approval".

## D-020 Deliverable and yardstick: indicators plus predictions, a leaderboard on dev-fold net Sharpe (owner, 2026-09-28)
- **Owner:** "Searched the grid of configs for the best financial metrics" and "FOR MVP let's keep it
  as simple winner, but we want to counter it by leaderboarding only on validation dataset valid".
- **Decision:** as VISION "What every run delivers" and "The yardstick" (2c58370).
- **Not in VISION:** no multiple-testing correction in the MVP (Round 2). The owner accepted the risk
  of picking by eye from the test columns; the UI labels the ranking column (Round 3). D-005 stands:
  no choice and no ranking uses the test block.
- **Consequence:** NT-030 (seed re-runs), NT-031 (leaderboard), NT-033 (frozen twin, TA rules),
  NT-050 (the first real sweep and the learned-against-baseline verdicts under D-025).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 1-4.

## D-021 MVP order: foundations first; the old roadmap folds in (owner, 2026-09-28)
- **Owner:** chose "foundations first" (Round 1) and folding the old roadmap into the MVP (Round 2).
- **Decision:** structure and the experiment engine; then the control panel and model comparison as
  one framework (growth points 4 and 5); then gradient stability, inside the same framework; then
  visual comprehension on top, in parallel where its files are disjoint (NT-043). R1 first; R2-R4
  become scenarios run through the engine; R5 unchanged. "Unified testing framework" means model
  comparison; the pytest suite stays for code correctness. VISION "The MVP" lists the points.
- **Consequence:** ROADMAP "Order". R5 whenever the owner is ready, after R1 (unchanged). NT-003 to
  NT-006 wait for the engine (NT-026) and the comparator (NT-032). Generality's place (after
  stability, before visuals) is the lead's, approved with VISION's MVP list (2c58370). The milestone
  names and MVP-6 right after MVP-1 are the lead's reading.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 1 (order), Round 2
  (roadmap, testing framework).

## D-022 Generality designed in; only the reference setup tested in the MVP (owner, 2026-09-28)
- **Owner:** chose "design now, add later" (Round 1). On the target: "I'd advise for deltas, because
  this logic is already entrenched in the repo for math reasons". On the data span: "that would be
  configurable, but let's keep 7 days even, not 30".
- **Decision:** as VISION "The reference setup", "The MVP" point 4 and "Not in the MVP" (2c58370).
- **Not in VISION:** the pairwise physics terms run over every neighbouring pair (h_i, h_i+1) and are
  averaged, so N = 3 reproduces today; horizon colours beyond h2 follow the validated categorical
  palette of `visualization/theme.py` in a fixed order (Round 5b).
- **Consequence:** NT-040 (bar size in annualisation), NT-041 (dataset spec, wall-clock
  configuration), NT-042 (N horizons), NT-052 (harness runs for N = 2 and N = 4). The MVP assumes a
  24/7 market (lead's reading, approved with VISION). Casimir (`losses/functions.py:328`) and IFE
  (`:442-443`) sum their (h0,h1) and (h1,h2) terms today; to keep "N = 3 reproduces today", the
  per-pair mean is scaled so that N = 3 gives today's value (lead's reading; NT-042).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 1, 5 and 5b.

## D-023 Experiment engine first; a control-panel notebook; quick and Optuna sweeps (owner, 2026-09-28)
- **Owner:** "The system should have option to do quick sweeps or optuna sweeps. In quick we try to do
  everything as fast as by 5 minutes, but optuna is measured, but can allow for overnight".
- **Decision:** as VISION "The MVP" points 1 and 2 (2c58370).
- **Not in VISION:** the restructure is incremental, engine first; every module move is checked by
  `scripts/golden_run.py verify` (Round 5b). Adding `optuna` (a version that suits Python 3.10 and
  numpy 1.23.x: 1.23.0 in the nt env, 1.23.5 in CI) to the local `nt` env and the requirements is owner-approved (Round 3); how: RUNBOOK
  "Installing optuna". Any other change to the `nt` env still needs the owner.
- **Consequence:** the frozen set is `scripts/gate_run.py`, `scripts/check_gates.py`,
  `scripts/backtest_gate.py`, `scripts/direction_experiments.py`, `scripts/ablate.py` and
  `src/neural_trade/experiments/ablation.py` (kept importable until NT-026 subsumes it); they stay
  runnable as history, and nothing new builds on them. NT-026 (engine; supersedes NT-024), NT-027
  (layering), NT-029 (config metadata), NT-030 (sweeps, the optuna pin), NT-034 (notebook 06).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 3 (panel, Optuna),
  Round 5b (restructure), Round 6 (sweep modes).

## D-024 GPU use for sweeps (owner, 2026-09-28)
- **Owner:** "optuna is measured, but can allow for overnight" (D-023); Round 7's recommended options.
- **Decision:** as VISION "The MVP" point 2 (2c58370): budget measured and stated first; overnight
  while the GPU is idle.
- **Not in VISION (Round 7):** one seed per trial per dev fold; the top 5 re-run with 3 seeds; order
  and winner by the seed mean. Several training processes at once only while the owner's other
  project leaves the GPU idle, N from one measured throughput test; otherwise one at a time.
- **Consequence:** NT-030 (`--max-hours` defaults to 12; `--parallel` above 1 needs NT-035's result),
  NT-035, NT-050; the rules: OPERATING_MODEL "Sweeps and pre-registered studies". Lead's reading: this
  replaces the 3-GPU-hour limit for sweeps only (A/B studies keep it and the research budget); one
  launch runs at most one night, about 12 GPU-hours, and a larger budget goes to the owner (STATUS
  "Waiting for the owner"). NT-006's GPU time stays a separate owner question (asked 2026-09-25).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 6 (GPU budget),
  Round 7 (seeds, parallel GPU).

## D-025 Verdict rule: paired test plus a pre-registered minimum effect (owner, 2026-09-28)
- **Owner:** Round 7's recommended options. Context: the v1 physics ablation (D-003) judged its
  guard-rails by a point tolerance (h1 direction AUC -0.0119 against 0.01).
- **Decision:** as VISION "The yardstick", "A beats B" (2c58370).
- **Not in VISION:** guard-rails are judged by the same paired test. A deterministic TF mode is opt-in
  for comparison studies, after a measured speed test; sweeps stay fast. The mode is
  `seed_everything(seed, deterministic=True)` (`utils/seeding.py:32`), not the
  `TF_DETERMINISTIC_OPS=1` that every run already sets (`__init__.py:26`).
- **Consequence:** supersedes `configs/ablation_criteria.yaml` for new studies; the v1 grid
  `runs/ablations/ablate_physics_v1-full` stays the record under its own criteria. NT-032
  (comparator; error rates simulated with block noise, D-012), NT-035 (deterministic-mode speed),
  NT-006 (pre-registered again; GPU time asked 2026-09-25), NT-039, NT-050. Lead's reading: a
  verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them
  before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history
  once NT-041 exists); at least 5 pairs.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 7.

## D-026 Gradient stability: invariants, per-run health, stress harness, config guard (owner, 2026-09-28)
- **Owner:** "Mathematical gradient stability. unified gradient stability testing framework"
  (2026-09-25); on unstable runs: "attribute and fail loudly, but this should be configured to not
  allow a configuration of hyperparameters that would fail the system".
- **Decision:** as VISION "The MVP" point 3 (2c58370).
- **Not in VISION:** the harness runs inside the experiment engine (Round 1). Strict mode (the loss
  masks off) in CI and in the harness (Round 8). The harness sweeps scale and volatility x0.1 to x10,
  with pre-registered thresholds. Optuna prunes an unstable run; the leaderboard shows it as failed.
  The per-term probe costs about 10%.
- **Consequence:** NT-036 (CI invariants, `stability` marker), NT-037 (per-run health; absorbs
  NT-012), NT-038 (harness cases as engine scenarios, config guard), NT-039 (the A/B, under D-025),
  NT-051 (first harness run), NT-052. The 2% cost is held to D-018. Once NT-036 exists, the
  definition of done runs the stability marker (OPERATING_MODEL).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 1 and 8.

## D-027 Indicators extend through a registry; learned indicators on price first; D-014 on every figure (owner, 2026-09-28)
- **Owner:** "Lets elaborate on that in a separate Q&A session. In general I want the list of
  indicators to be extendable and easily integrated via registry". On combinations: "Lets elaborate
  on that in a separate Q&A".
- **Decision:** as VISION "The MVP" point 5 and its "Also in the MVP" paragraph (2c58370). The
  registry's design, and where indicator combinations come from, are settled in a separate owner Q&A
  held right after this overhaul (2026-09-28, before the move to D:, D-030).
- **Not in VISION:** the first view draws moving averages and Bollinger lines on price and RSI and
  MACD in panels, against the textbook defaults, with how the periods moved over training and per
  window (Round 4). D-014 applies to every figure, the comprehension views included: there is no
  simplified tier (Round 9).
- **Consequence:** an Indicators registry extends the nine of D-002. NT-046 stays a placeholder
  until the indicator Q&A; NT-043 builds the first view and notebook 07 (D-028).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 4 and 9.

## D-028 Notebooks persist and evolve (owner, 2026-09-25 / 2026-09-28)
- **Owner (2026-09-25):** "I want the notebook structure to persist and even be updated during the
  development towards MVP".
- **Decision:** as VISION "The MVP" point 5 and "Principles" (2c58370): 00-05 keep their numbers and
  roles; new views get new notebooks (06 control panel, 07 discovered indicators); all generated and
  executed per D-013.
- **Not in VISION:** each figure has one home, so overlaps (01 / 04) are trimmed (Round 6). An item
  that changes what a notebook shows updates it through `scripts/notebooks/build.py` in the same item
  (OPERATING_MODEL definition of done).
- **Consequence:** NT-034 (06), NT-043 (07), NT-045 (the overlap).
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) owner statement after
  Round 3 (dated 2026-09-25 there), Round 6.

## D-029 Delete only what is stale and has no effect (owner, 2026-09-28)
- **Owner:** "only delete something which is both stale and doesnt affect current system in any
  sense".
- **Decision:** as VISION "The MVP" point 1 (2c58370). Every deletion shows evidence of both: it is
  stale, and it has no effect (no production, notebook, script, doc or test use, and no default
  re-creates it). Anything with any effect stays.
- **Not in VISION:** deleting runs, data, models or remote branches still needs the owner
  (OPERATING_MODEL "Escalate to the owner"). Changing a behaviour is not a deletion.
- **Consequence:** NT-028 (the stale candidates of the 2026-09-28 survey). The silent warm start from
  weights in the working directory is a behaviour, not a deletion: NT-049.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 6.

## D-030 The project moves to D:/neural_trade (owner, 2026-09-28)
- **Owner:** asked where the run store should live: "move the project to disk D" (Round 5b).
- **Decision (Round 6; VISION does not cover it):** in the 2026-09-28 session, after the doc overhaul
  is committed and pushed and the indicator Q&A is held, the project moves to D:/neural_trade: the
  repo with its untracked runs, both sibling worktrees (then `git worktree repair`), the editable
  install re-pointed with `pip install --no-deps -e .` from D: (an owner-approved env change), and the
  Claude memory folder. Tests and the notebook check are verified on D:; the owner reopens VS Code
  there. The C: copy is deleted by the lead only on the owner's explicit go-ahead, after the owner confirms
  the D: copy works (OPERATING_MODEL "Escalate to the owner"). Runs, the Optuna
  studies and the run index live in the repo on D:.
- **Consequence:** the steps: RUNBOOK "Planned move to D:". Until the move, paths stay
  C:/Users/Step/Documents/neural_trade. Lead's reading: NT-009 is closed by the move, once verified.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 5b (run store),
  Round 6 (move), Round 9 (indicator Q&A before the move).

## D-031 The indicator catalogue (owner, 2026-09-28)
- **Owner:** "All should be learnable"; on combinations: "the combination are what the network does
  automatically, it's the core of it's mechanic. I don't think it needs any extra mechanism"; on
  delivery: "html report with extensive rich interactive visualizations".
- **Decision (Rounds A-C; VISION says only "an extendable indicator catalogue"):** all four groups
  of families: today's MA/EMA, MACD, RSI, Bollinger; ATR, Stochastic, Williams %R, Keltner; OBV,
  VWAP, MFI; ADX/DMI, CCI, Donchian; so the model input becomes OHLCV. Every indicator learnable
  (smooth versions where needed); combinations left to the network; adaptive per-window periods
  kept, reported as a range, with a switch. One registry entry per family (inputs, learnable
  parameters with textbook defaults and bounds, output channels, drawing); all families on by
  default, 3 instances each. Grouped permutation importance (read-out only). A self-contained
  interactive HTML report per run.
- **Open:** the owner's window question ("I think we can get rid off it completely. Research this.")
  is under research (2026-09-28); its decision follows as a new entry.
- **Consequence:** NT-046 (package and registry, today's four families), NT-047 (OHLCV and the new
  families; waits for the window decision; D-018 applies), NT-048 (the HTML report, notebook 07). Supersedes D-027's "NT-046 stays a placeholder" (the
  indicator Q&A is held; NT-046 is defined).
- **Evidence:** [qa/2026-09-28-indicators.md](qa/2026-09-28-indicators.md) Rounds A-C.

## D-032 The fixed input window goes; the path comes from a written plan (owner, 2026-09-28)
- **Owner:** on the research's staged-path question: "commit to this in another research and write
  it down in a plan."; the period ceiling: "unlimited"; the replacement for the per-window adaptive
  periods: "research"; the purge rule: "research yourself".
- **Decision:** removing the fixed input window is committed to. The concrete path (stages, order,
  gates), the replacement for the per-window adaptive periods and the corrections the adversarial
  review demanded are settled by a second research round that writes a plan (NT-053), approved by
  the owner before its first implementation item is picked. No configured period ceiling: a learned
  period may grow without limit (the warm-up it needs is the plan's problem, not a config cap). The
  purge rule for indicators with unbounded memory is the lead's to research and decide, recorded as
  a new entry with a test.
- **Not in VISION:** all of it. VISION's "60-minute input window" wording stays until the plan is
  approved: the window model is the default until an A/B adopts a replacement (D-025).
- **Consequence:** NT-053 (the plan); NT-047 waits for NT-053 (the input path and the indicator
  forms may change); NT-054 (per-run fixed costs and GPU launches; window-independent).
- **Evidence:** [qa/2026-09-28-indicators.md](qa/2026-09-28-indicators.md) Round D;
  [research/2026-09-28-window-free/README.md](research/2026-09-28-window-free/README.md).

## D-033 Remote sessions run review sweeps; the lead QA's and merges their pull requests (owner, 2026-09-28)
- **Owner:** "i opened PR on a remote session, which runs a parallel code sweep"; then Round 1 of
  [qa/2026-09-28-remote-sessions.md](qa/2026-09-28-remote-sessions.md): the sweep is a review sweep
  that takes no backlog items, and the lead QA's and merges its pull requests; then "continue
  autonomously until the plan completion. check for updates to repo".
- **Decision:** a remote (cloud) session the owner runs reviews and fixes code outside the backlog
  items. Its pull requests into `remediation/plan` are treated like an implementer branch: QA
  verifies them (against the item's criteria, or the PR's own claims plus the definition of done's
  general clauses when the PR is not a backlog item), the lead merges them, checks CI and records
  the result. A PR that would change a recorded decision, default trading behaviour or `master` goes
  to the owner instead (OPERATING_MODEL "Escalate to the owner"). Local sessions keep the backlog
  order; they do not wait for the remote session.
- **Lead's reading:** "check for updates to repo" means `git fetch origin` and the open-PR list at
  session start and between items (RUNBOOK "CI" has the API call), so remote work is integrated as
  it arrives. The rule is written into CLAUDE.md, OPERATING_MODEL and the `/next` and `/handoff`
  skills.
- **Consequence:** PR #14 (NT-001) was QA'd and merged this way (6d01d11). Findings of a remote
  review that are not fixed in its PR become backlog items, triaged as usual.
- **Evidence:** [qa/2026-09-28-remote-sessions.md](qa/2026-09-28-remote-sessions.md).

## D-034 The purge rule for indicators with unbounded memory: label overlap, gap kept at 80 (lead, 2026-09-29)
- **Context:** D-032 left the purge rule for indicators whose state reads every earlier bar to the
  lead ("research yourself"). D-005's invariant (no training-label bar is an evaluation input) holds
  literally only with a state reset and a burn-in inside every gap.
- **Decision (rule a+):** the gap between adjacent blocks is max(2 max(H), W + max(H)), W being the
  longest finite window any consumer reads (60 today), so the reference gap stays 80 bars and today's
  anchors are unchanged. An indicator state reads every earlier bar, as it does live, and resets
  (with a masked burn-in) only at the data start and at long gaps. Judgement folds come after every
  fold whose score makes a choice. Every input path must be causal and every normaliser fit on the
  training block or trailing.
- **Why:** no later label can reach a gradient, an epoch choice or a calibrator when the gap covers
  the label increments; earlier labelled bars in a later block's state are its past, available live.
  D-005 taken literally closes no further leak and would cost 12.5% / 43% / 246% / 1,711% of a 7-day
  training block at learned periods of 60 / 240 / 1,440 / 10,080 bars.
- **Consequence:** the plan's stage-4 item "purge rule test and config guard" (`tests/test_purge_rule.py`,
  Config.validate refuses a smaller gap). D-005 stands for the window model.
- **Evidence:** [research/2026-09-29-window-free-plan/](research/2026-09-29-window-free-plan/README.md)
  "The purge rule"; C/FINDINGS.md Q1.

## D-035 Models and effort: Sonnet for implementer, QA and experimenter; the lead escalates (owner, 2026-09-29)
- **Owner:** asked how model and effort use could be optimized "autonomously without my involvement"
  and answered "yes" to the lead's proposal (this session).
- **Decision:** as OPERATING_MODEL "Models and effort": Sonnet 5 by default for implementer, QA and
  experimenter; Opus 5.5 for QA of P0 and statistics or verdict items and for a second repair round;
  Fable 5.1 or Opus 5.5 for the lead, research and adversarial reviews; lighter QA for P2 and P3; no
  duplicate integration suite runs; at most two agents at once; reports under about 800 words. The
  lead chooses per call without asking.
- **Context:** the session of 2026-09-28 / 29 ran seven agents at once and hit the monthly spend limit
  and then the weekly limit, stalling all work for about 21 hours.

## D-036 A cheap tracker agent for task tracking; the other roles keep the strong model (owner, 2026-09-29)
- **Owner (verbatim):** "drop agnt limit requirments and 800-word reports. Let's revisit this. I'd like
  to use weaker models specifically for TASK TRACKING, so that you don't have to use fable 5, for
  example, to poke the running agent, a running process or fix minor issues."
- **Decision:** supersedes D-035's model and limit parts. A `tracker` agent on Haiku
  (`.claude/agents/tracker.md`) waits for CI, processes, suites and agents and makes mechanical fixes
  the lead names; implementer, QA and experimenter set no model (the session's strong model); no cap
  on the number of agents or on report length. D-035's effort rules stay (lighter QA for P2/P3, no
  duplicate integration suite runs, batching). OPERATING_MODEL "Models and task tracking".
- **Lead's reading:** "specifically for task tracking" means the weaker model is not used for
  implementation or QA; the owner may correct this.

## D-037 The window-free plan is approved; NT-047 in window mode first; periods held at the data bound; study budgets (owner, 2026-09-29)
- **Owner:** chose every recommended option of [qa/2026-09-29-window-free-plan.md](qa/2026-09-29-window-free-plan.md).
- **Decision:** the plan [research/2026-09-29-window-free-plan/](research/2026-09-29-window-free-plan/README.md)
  is the path of D-032: stages 1 (benchmark kit) to 7 with their gates; A/B-1 (series engine against
  window engine), A/B-1b (removing the 60-bar clip) and, if the probe finds the model epoch-bound,
  A/B-2 (per-bar model), all judged on per-fold retention of the CRPS edge and paired coverage under
  D-025 with Pocock two looks; net Sharpe reported, not judged. NT-047 is built now in window mode
  against the Indicators registry; its series forms (exponential and leaky) come after A/B-1. A learned
  period that needs more history or pass length than available is projected at the data-derived bound
  (a 2,048-bar burn-in on the bundled file, a pass budget of 4 training blocks), counted and reported.
  GPU ceilings: A/B-1 about 6, A/B-1b about 6, A/B-2 about 5.6, the probe about 1.5 GPU-hours, each
  re-checked from its dev runs.
- **Not in VISION:** VISION's "60-minute input window" wording stays until A/B-1 adopts the series
  engine (D-032).
- **Consequence:** NT-053 done; NT-059 to NT-073; amendments to NT-032, NT-038, NT-041, NT-046, NT-047;
  ROADMAP research track R6.
- **Evidence:** the plan, its REVIEW.md and the Q&A record.

## D-038 The old C: copy is ignored; pushing stays automatic (owner, 2026-09-29)
- **Owner (verbatim):** "1) ignore c copy 2) revoke.push auto" ([qa/2026-09-29-standing-questions.md](qa/2026-09-29-standing-questions.md)).
- **Decision:** the C: copy (C:/Users/Step/Documents/neural_trade and its two worktrees) is left as it is; no
  session deletes it or asks about it (this replaces D-030's "deleted on the owner's go-ahead"). NT-009 is done.
  Pushing `remediation/plan` and `nt-*` without asking stays (D-017's push rule, now owner-confirmed; never
  `master`, never `--force`).
- **Lead's reading:** "revoke.push auto" read as withdrawing the question and keeping automatic pushes; if the
  owner meant the opposite, a new entry records it.

## D-039 The window-free build-up moves after the MVP, into research track R6 (owner, 2026-09-29)
- **Owner (verbatim):** "Yeah move and log the window retirement work past MVP."
- **Decision:** NT-064 (kernel and assembly), NT-065 (INDICATOR_MEMORY switch), NT-066 (purge-rule test),
  NT-067 (Predictor series mode) and NT-068 (per-bar reporting) leave MVP-6's exit and join research track
  R6 with the A/Bs (NT-069 to NT-073). MVP-6's exit is NT-046 to NT-048 (the catalogue in window mode,
  D-037) plus NT-060 and NT-061 (the kit on the GPU and the TF32 decision, which also concern today's
  model). The MVP ships with the window model as the default, as VISION defines it; D-032's commitment
  and D-037's plan stand, only later.
- **Consequence:** ROADMAP MVP-6 and R6; about 2 days off the MVP's critical path (an estimate).

## D-040 One 360-day training run on the long history, set up by the lead, tracked in a notebook (owner, 2026-09-29)
- **Owner:** "launch 1 360 days training sample. Ensure GPU is running at full effectiveness", "use as high batch
  size as possible" and "after the benchmark, work out yourself how to set up the 360-day training correctly and
  launch it yourself, autonomously, preferably through a notebook" ([qa/2026-09-29-long-training.md](qa/2026-09-29-long-training.md)).
- **Decision:** one run trains on a 360-day block of Bitcoin_BTCUSDT.csv (the local 2017-2025 file). The lead chooses the
  layout, batch size, shuffling and learning rate from a measured benchmark (old and new code), shown to the owner
  first, and launches the run itself; its progress is tracked in a generated notebook. The owner's request is the GPU
  approval for this one run.
- **Lead's reading:** the run trains on dev fold -2 (fold -1, the newest ~32 days, stays the untouched test fold, D-020),
  so its result may inform later choices such as the training length. Defaults (the 7-day reference setup, D-022) do
  not change.

## D-041 The micro-scale loop: minutes-long runs drive hypothesis iteration toward predictive power and PnL (owner, 2026-09-29)
- **Owner:** the /goal of 2026-09-29, verbatim in [qa/2026-09-29-long-training.md](qa/2026-09-29-long-training.md)
  Follow-up 2: raise the model's predictive power and the strategy PnL "at micro-scales: hours instead of days",
  using micro-scales for lightning-fast training, inference and hypothesis development.
- **Decision:** hypothesis iteration runs on micro setups: small training blocks on the long file (about 10 days,
  batch 2048: a full train + score cycle in roughly 3 minutes), strategy work on stored predictions (CPU rescore,
  no retraining). Quick sweeps under the existing sweep rules (OPERATING_MODEL); anything claiming "A beats B"
  still goes through D-025. The reference setup and defaults do not change without a recorded decision.
- **Lead's protocol (first iteration):** micro layout on Bitcoin_BTCUSDT.csv: N_FOLDS 2, total 129,600 windows,
  fold -2 = train about 10 days + val/cal about 10 days each + a 30-day dev out-of-sample block; fold -1 (the
  newest 30 days) stays the untouched test fold. Hypotheses H1 (longer horizons: 1-4 h, where the move is
  several times the 26 bps cost) and H2 (fewer, more selective trades of the existing signal) run first.
- **Consequence:** NT-085 (the micro loop and its scenarios).

## D-042 No pinging: the lead waits for completion notices; active polling goes to the Haiku tracker (owner, 2026-09-29)
- **Owner (verbatim):** "why a are you [inging it constanly? i dont think its a good idea. If you kkep doing then
  emply haiku 4.5".
- **Context:** during the micro loop the lead answered every interim "still running" notification and checked
  running agents and runs by hand, on the strong model.
- **Decision:** the lead never polls or checks running agents, runs or suites itself, and never replies to interim
  notifications; it waits for the completion notice. Active polling (CI, an external process) goes to the
  `tracker` agent on Haiku 4.5 (D-036). CLAUDE.md "Project rules" and OPERATING_MODEL "Models and task tracking".

## D-043 Model split: Opus for the lead, P0/P1 QA and research; Sonnet for implementer, experimenter and P2/P3 QA (owner, 2026-09-29)
- **Owner (verbatim):** asked "which model would be optimal to complet this task? sonnet 5 medium effort or opus 5.5
  medium effort", then, after the lead proposed the split below: "Try to optimize the tasks and close them asap".
- **Decision (lead's reading of the second message as approval of the proposal; the owner may correct it):**
  lead (the session), QA of P0/P1 items and research run on Opus 5.5 at medium effort; implementer, experimenter and
  QA of P2/P3 items run on Sonnet 5 at medium effort (the lead passes the model per call); the tracker stays on Haiku
  4.5 (D-036, D-042). Supersedes D-036's "implementer, QA and experimenter keep the strong model" for those roles.

## D-044 Trading costs are 0; no tick order-book data (owner, 2026-09-30)
- **Owner (verbatim):** "1 нет / 2 издержки делай 0 / дай статус по текущим моделям с издержкой 0" (the lead's four
  options after the micro loop's report: 1 tick order-book data - no; 2 lower costs - make them 0).
- **Decision:** backtests and scoring use zero trading costs (fee, half-spread, slippage all 0; the P&L objective's
  and the cost-aware strategies' cost 0). Tick order-book data is not collected. This supersedes VISION "The
  yardstick" and "Principles" on costs (the owner's document; the lead updates VISION's wording in the same change).
- **Consequence:** NT-094 (defaults to zero costs); the zero-cost status of every stored model (rescore, CPU).

## D-045 The maths report's 22 recommendations are implemented by a staged plan (owner, 2026-09-30)
- **Owner:** asked for a plan to implement all recommendations of the maths report ("Напиши план по внедрению всех
  рекоммендаций") and approved the lead's plan.
- **Decision:** the recommendations of [research/2026-09-30-math-report/](research/2026-09-30-math-report/A_losses.md)
  (A_losses.md, B_model_indicators.md; presentations/4_math_report.html) are implemented in five phases: (0) foundations
  on CPU: NT-036, NT-037 (with the D-045 amendment), NT-032, NT-096, NT-074; (1) measure the per-term gradient shares on
  real trainings (NT-098); (2) prune what the measurements condemn (NT-099 soft ECE and vol, NT-100 NLL tail); (3) reweigh
  (NT-101 -> NT-039, then NT-102 clip); (4) rebuild (NT-104 capacity, NT-105 attention; NT-097 -> NT-106 indicators);
  (5) generalise (NT-040/041/042 -> NT-107). Every change that moves numbers is a Config switch with today's default
  (golden run unchanged) until a pre-registered A/B judged by NT-032 adopts it (D-025); the per-step path keeps D-018;
  the physics terms stay (D-003; NT-006 decides them).
- **Gates for the owner:** epoch selection on proper scores (NT-103) changes the rule of D-011; any study above 3
  GPU-hours, including the final 360-day confirmation (about 3.5 GPU-hours, an estimate).
- **Consequence:** NT-096 to NT-107; amendments to NT-037, NT-038, NT-041.

## D-046 Verdicts are inferred over folds: at least 5 judgement folds (lead, 2026-09-30)
- **Context:** D-025's verdict-fold reading (lead's) was "fold -1 x at least 5 seeds today, at least 5 pairs". QA of
  NT-032 simulated that design with the block noise shared by every seed on the fold: the false-'beats' rate was
  0.069 / 0.146 / 0.314 at minimum effects 0.01 / 0.005 / 0, against the nominal 0.05; with 5 folds x 1 seed it was
  0.000 (QA report on nt-032 2aef6ca; scripts D:/nt_qa/nt032_sim.py).
- **Decision:** supersedes the lead's reading in D-025. The fold is the unit of inference: the paired differences are
  averaged over each fold's seeds and the test runs over at least 5 distinct judgement folds that no choice used, named
  in the SPEC before any GPU time. Seeds still reduce the within-fold noise. The long history has 13 usable folds
  (long_360d_stab meta, n_usable_folds 13); a study places its judgement folds after every fold its choices used (D-034).
- **Consequence:** NT-032 (repair round 1), OPERATING_MODEL "Sweeps and pre-registered studies"; every A/B of D-045
  (NT-097, NT-099-NT-107, NT-039) is designed with at least 5 judgement folds.

## D-047 NT-047's default: OHLCV input with all 14 indicator families; the slower step is accepted (owner, 2026-10-01)
- **Owner (verbatim):** "2" (option 2 of question 7: all 14 families on by default, as D-031 says; accept the 1.6x
  slowdown), [qa/2026-10-01-nt047-default.md](qa/2026-10-01-nt047-default.md).
- **Decision:** D-031's default stands: OHLCV input and all 14 families with 3 instances each. The per-step cost
  (0.1735 against 0.1066 s on the GPU, 1.63x) is accepted under D-018. The lead's recommendation (close-only default)
  is declined; the duel showed no directional gain for either input.
- **Consequence:** NT-047 is merged as built (72d3838). The D-045 studies (NT-098 onward) and the golden record use the
  new default; the golden run is re-recorded on the merged head.

## D-048 Tiny first: maths and stability on the 6-hour layout; test-run discipline; pytest-xdist (owner, 2026-10-01)
- **Owner:** "I expect this session and whole testing regarding maths and architecture to take place nearly
  instantly", then approved the lead's three rules and the install: "да, внедряй правила и ставь xdist".
- **Context:** on 2026-09-30 / 10-01 no GPU training of the D-045 plan had run yet; the wall time went to test suites
  run 3-5 times per item (fast 6-9 min, slow 17-38 min, inflated by agents running suites at the same time on one CPU).
- **Decision:** (1) maths, stability and architecture checks (gradient shares, non-finite steps, clipping, loss
  terms, shapes, memory) run first on the 6-hour screen layout (NT-088, seconds per trial); quality verdicts use the
  micro layout with at least 5 judgement folds (D-046); 360-day runs only for final confirmations. (2) Test runs: QA
  runs the tests an item touches plus ONE full suite; no two full suites run at the same time in different
  checkouts; slow tests are profiled and shrunk or moved. (3) pytest-xdist 3.8.0 (with execnet 2.1.2) is installed in
  the nt env (owner-approved env change) and the suites run with `-n 8` (measured: fast suite 1,156 passed in 3 min 26
  s at 8 workers against 7 min 20 s serial; 16 workers 3 min 40 s, bound by six ~30-55 s training tests).
- **Consequence:** OPERATING_MODEL "Tiny first", CLAUDE.md commands, RUNBOOK, the agent files; NT-098 runs on the
  screen and micro layouts; NT-109 (shrink the slowest tests); requirements-ci.txt and CI use xdist.

## D-049 NT-099 may finish its two open cells past the 3-hour cap (owner, 2026-10-03)
- **Owner (verbatim):** "Q-9 - okay." ([qa/2026-10-03-nt099-gpu.md](qa/2026-10-03-nt099-gpu.md)).
- **Context:** `loss_prune_v1` stopped over its stated 3 GPU-hour cap (about 3.2 hours used). Verdict 1 (soft ECE
  off) is provisional because `ece0` fold −39 trained during a 27-second code-tree mix-up. Verdict 2 (soft ECE and
  vol off) has 4 of 5 folds; `ece0_vol0` fold −35 never logged a step. Both gaps were estimated at 30–35 GPU minutes.
- **Decision:** run both. The original suspect cell stays. Its replacement is a new run on the same spec and code,
  scored in place of it for verdict 1. No loss-weight default changes from the provisional verdict.
- **Consequence:** NT-099 (the two cells ran the same day; the verdicts are in `runs/experiments/loss_prune_v1/REPORT.md`).

## D-050 No new data source; the micro loop does not go looking for taker-buy volume (owner, 2026-10-03)
- **Owner (verbatim):** "No new source" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** question 8, asked 2026-09-30. The micro loop had not reached the owner's hit-rate and drawdown
  goal, and the research note said that goal needs a different information source. The recommendation with the
  question was a CPU logistic check of Binance taker-buy volume on 2024-2025, then basis and funding. A new
  source is outside the MVP.
- **Decision:** do not add taker-buy volume, basis, or funding. Close the micro loop on its journal and return
  to the MVP backlog.
- **Consequence:** NT-085.

## D-051 Strategies: raw heads for coherence, served delta for size (owner, 2026-10-03)
- **Owner (verbatim):** "Split raw and served (Recommended)" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** NT-007. Served delta is beta times the raw head. On served deltas, magnitude agreement is true on
  7.2% of test bars against 56.2% on the raw heads, and it drove most of the enhanced_multi_horizon exits. With
  beta 0 the same strategy cannot enter, because entry still requires served d1 > 0.
- **Decision:** `magnitude_coherent` and `direction_aligned` use the raw heads. Take-profit sizing stays on the
  served delta. The enhanced_multi_horizon d1 > 0 entry rule was not in the option, so it stays on the served
  delta. A zero beta still blocks that entry.
- **Consequence:** NT-007 (the decision), NT-115 (the code).

## D-052 NT-006 may spend about 7-10 GPU-hours on the physics re-run (owner, 2026-10-03)
- **Owner (verbatim):** "Approve the re-run (Recommended)" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** the v1 physics grid was scored on the last epoch, before D-011, and its family verdict is
  inconclusive. A re-run under D-025 was estimated at 7-10 GPU-hours on 2026-09-25, over the 3-hour cap that
  pre-registered studies keep unless the owner approves (D-024). That estimate has not been remeasured.
- **Decision:** the re-run is approved. It still needs its own SPEC before any GPU time. This answer does not
  start it.
- **Consequence:** NT-006.

## D-053 Epoch selection may grow a switch; validation loss stays the default (owner, 2026-10-03)
- **Owner (verbatim):** "Add the switch (Recommended)" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** D-011 serves the best validation-loss epoch. NT-103 was not to be picked until the owner
  answered, because that rule is D-011's. Validation loss moves 0.1-0.3 per epoch, far more than the direction
  signal.
- **Decision:** an implementer may add `EPOCH_SELECT_METRIC`. The default stays validation loss, and the golden
  run stays put, until a paired test chooses otherwise. The other registered choice is direction BCE plus CRPS,
  each divided by its epoch-1 value.
- **Consequence:** NT-103.

## D-054 remediation/plan is not merged into master yet (owner, 2026-10-03)
- **Owner (verbatim):** "Not yet (Recommended)" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** NT-008. D-004 leaves the merge into `master` to the owner. `master` still holds the
  pre-remediation code, so the nightly workflow has never run. `nt-099` is not on `remediation/plan`.
- **Decision:** do not merge. `master` stays untouched. NT-008 stays open until a later yes.
- **Consequence:** NT-008.

## D-055 File a P3 item for Predictor.predict GPU latency (owner, 2026-10-03)
- **Owner (verbatim):** "File the P3 item (Recommended)" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** asked 2026-09-29. One single window and one batch, a few GPU-minutes. Inference speed is not a
  yardstick (D-018).
- **Decision:** file the item at P3. Do not run it ahead of P1 work, and do not start it with this entry.
- **Consequence:** NT-116.

## D-056 The repository licence is MIT (owner, 2026-10-03)
- **Owner (verbatim):** "MIT" ([qa/2026-10-03-owner-quiz.md](qa/2026-10-03-owner-quiz.md)).
- **Context:** NT-017. The repository is public and had no LICENSE file. The owner had left the choice open
  (2026-09-28, round 9); D-019 lowered it to P3.
- **Decision:** MIT, copyright 2026 another-world. `LICENSE` is that text. The README points at it.
- **Consequence:** NT-017. QA has not passed the file.

