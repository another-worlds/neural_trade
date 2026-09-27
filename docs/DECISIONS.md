# Decisions

Append-only log. An entry changes only through a new entry that cites new evidence ("Supersedes
D-0xx"). Owner decisions change only with the owner. Format: context, decision, consequence,
evidence. When you append an entry, add its row to the index.

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
| D-019 | The vision: an indicator-learning predictor; BTC/USDT 1-minute is the reference setup | owner | |
| D-020 | Deliverable and yardstick: indicators plus predictions, a leaderboard on dev-fold net Sharpe | owner | |
| D-021 | MVP order: foundations first; the old roadmap folds in | owner | |
| D-022 | Generality designed in; only the reference setup tested in the MVP | owner (24/7 market: lead's reading) | |
| D-023 | Experiment engine first; a control-panel notebook; quick and Optuna sweeps | owner | D-024 |
| D-024 | GPU use for sweeps | owner (A/B studies keep the old limit: lead's reading) | |
| D-025 | Verdict rule: paired test plus a pre-registered minimum effect | owner | |
| D-026 | Gradient stability: invariants, per-run health, stress harness, config guard | owner | |
| D-027 | Indicators extend through a registry; learned indicators on price first; D-014 on every figure | owner | |
| D-028 | Notebooks persist and evolve | owner | |
| D-029 | Delete only what is stale and has no effect | owner | |
| D-030 | The project moves to D:/neural_trade | owner | |

## D-001 Stay on TensorFlow 2.10 / Keras 2 (owner, 2026-09-22)
- **Context:** TF 2.10 is the last release with native Windows GPU support; the owner trains on a
  Windows RTX 4070 Ti.
- **Decision:** no Keras 3 / newer TF migration.
- **Consequence:** CUDA 11.2 / cuDNN 8.1 in the `nt` conda env; `import neural_trade` puts the
  env's CUDA DLLs on PATH; ptxas-missing log lines are harmless; XLA JIT ops must be pinned to CPU.

## D-002 All nine component registries, wired into the training path (owner, 2026-09-22)
- **Decision:** Models, Losses, Metrics, Optimizers, Callbacks, DataLoaders, Preprocessors,
  Layers, Visualizations - each strict, each queried by the trainer / predictor.
  Further registries extend the nine and do not replace them (Strategies exists; Indicators per D-027).
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
  with outputs. Each notebook stays under 5 MB. See D-028.
- **Evidence:** `tests/test_notebook_tooling.py`, `tests/test_notebooks_thin.py`. The nbstripout
  filter and hook, which would strip the outputs, were removed (NT-011, a0edba7).

## D-014 Rich figures, one visual system (owner, 2026-09-24)
- **Context:** the owner judged the first rebuilt figures "oversimplified" versus the old
  `inference.ipynb` (pre-rewrite version: `git show cb53872^:inference.ipynb`).
- **Decision:** figures show every logged metric per horizon with noise references, plus the
  tables the old notebook printed. All use `visualization/theme.py`: horizons h0 #3987e5,
  h1 #d95926, h2 #199e70 (only for horizons); dotted = training; status colours only for
  good / bad outcomes, always with a shape. See D-027.

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
- **Context:** the VISION of 2026-09-25 was inferred by the lead and framed the project as BTC/USDT
  1-minute trading (STATUS question 1). The owner (2026-09-25): "the project IS NOT about just 60
  minute BTCUSD, it shouldn't be constrained to a timeframe or a ticker" and "It is a substitute for
  manual technical indicator search and analysis."
- **Decision:** `neural_trade` is a neural network that predicts complex financial time series from
  technical indicators whose parameters and combinations are learned by gradient descent, a
  substitute for manual indicator search. It is not tied to a ticker or a timeframe. BTC/USDT
  1-minute bars with a 60-minute window and 10 / 15 / 20-minute horizons is the reference setup,
  chosen for its effectiveness, noise level and complexity. Audience: the owner and a few reviewers
  (2026-09-28). The docs are in English.
- **Consequence:** replaces the inferred vision; STATUS question 1 is answered. Every result names
  the setup it was measured on. Backlog text that framed BTC 1-minute as the purpose now calls it
  the reference setup. A licence is not needed for this audience: NT-017 stays open at P3. The
  growth points toward the MVP are D-020 to D-030.
- **Evidence:** [VISION.md](VISION.md) "Purpose", "The reference setup", "Audience" (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) owner statement of 2026-09-25, Round 9.

## D-020 Deliverable and yardstick: indicators plus predictions, a leaderboard on dev-fold net Sharpe (owner, 2026-09-28)
- **Context:** the owner's yardstick: "Searched the grid of configs for the best financial metrics".
  On selection: "FOR MVP let's keep it as simple winner, but we want to counter it by leaderboarding
  only on validation dataset valid".
- **Decision:**
  - Each run delivers both, judged together: the discovered indicators (the product) and the
    predictions and strategy built on them (the evidence that the indicators are good).
  - A searched leaderboard ranks configurations by net Sharpe after costs on the development folds
    (the out-of-sample blocks of the earlier walk-forward folds; the last fold never ranks).
    Guard-rail columns can disqualify a row: maximum drawdown, the number of trades, beating
    buy-and-hold, beating random entries at the same frequency.
  - Every row also shows its test-fold numbers; they never rank. The owner accepted the risk of
    picking by eye; the UI makes the ranking column explicit.
  - MVP winner: the top row after the top 5 are re-run with 3 seeds and ranked by the seed mean. No
    multiple-testing correction in the MVP.
  - Manual-search baselines, under the same search budget and the same dev-fold net Sharpe: the
    frozen twin (the same network with the periods frozen at the textbook values) and classic TA
    rules (moving-average cross, RSI threshold, Bollinger breakout) whose parameters the same search
    tunes.
- **Consequence:** test numbers are displayed on every row, but D-005's rule stands: choices and
  ranking never use the test block. Built by NT-030 (seed re-runs), NT-031 (leaderboard) and NT-033
  (baselines).
- **Evidence:** [VISION.md](VISION.md) "What every run delivers", "The yardstick" (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 1-4.

## D-021 MVP order: foundations first; the old roadmap folds in (owner, 2026-09-28)
- **Context:** the owner named five growth points toward the MVP (D-019 statement) and chose
  "foundations first" for their order.
- **Decision:**
  - Order: structure and the experiment engine, then the control panel and model comparison (one
    framework for growth points 4 and 5), then gradient stability, then generality, then visual
    comprehension. Visual work may run in parallel where its files are disjoint (the first view,
    NT-043).
  - The old roadmap R1-R5 folds in: R1 (trustworthy pipeline) first; R2-R4 become scenarios run
    through the engine; R5 (merge and publish) is unchanged.
  - "Unified testing framework" means model comparison (configurations across folds and seeds,
    statistical verdicts, the leaderboard). The pytest suite stays for code correctness.
- **Consequence:** ROADMAP runs R1, then MVP-1 to MVP-6, the research tracks R2-R4 after MVP-2, and
  R5 last (the milestone names and the indicator catalogue's place after MVP-1 are the lead's). The
  MVP exit is VISION "The MVP". The research items NT-003 to NT-006 wait for the engine (NT-026) and
  the comparator (NT-032).
- **Evidence:** [VISION.md](VISION.md) "The MVP" (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 1 (order), Round 2 (roadmap,
  testing framework).

## D-022 Generality designed in; only the reference setup tested in the MVP (owner, 2026-09-28)
- **Context:** the owner's choice "design now, add later". On the target: "I'd advise for deltas,
  because this logic is already entrenched in the repo for math reasons". On the data span: "that
  would be configurable, but let's keep 7 days even, not 30".
- **Decision:**
  - Instrument, bar size, window and horizons are configuration. Only BTC/USDT 1-minute is tested in
    the MVP; a second instrument or timeframe comes after it.
  - The window and horizons are configured in wall-clock time and converted to bars from the bar
    size.
  - A variable number of horizons now (towers, loss components, calibration, serving, figures,
    tests). The pairwise physics terms run over every neighbouring pair (h_i, h_i+1) and are
    averaged, so N = 3 reproduces today. Colours: h0 blue, h1 orange, h2 green, then the validated
    categorical palette of `visualization/theme.py` in a fixed order.
  - The target stays the price change in quote currency; no returns A/B in the MVP.
  - The training block is 7 days; the validation, calibration and out-of-sample blocks follow with
    their own configured lengths. The data file is configurable; the local 2017-2025 BTC/USDT file
    stays, for walk-forward folds over different months.
  - The MVP assumes a 24/7 market (the lead's reading, approved with VISION); trading sessions and
    calendars are outside the MVP.
- **Consequence:** D-014's horizon colours extend beyond h2 in a fixed order. Every run records the
  dataset it used. Built by NT-040 (bar size in annualisation), NT-041 (dataset spec and wall-clock
  configuration) and NT-042 (N horizons).
- **Evidence:** [VISION.md](VISION.md) "The reference setup", "The MVP" point 4, "Not in the MVP"
  (2c58370); [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds 1, 5 and 5b.

## D-023 Experiment engine first; a control-panel notebook; quick and Optuna sweeps (owner, 2026-09-28)
- **Context:** experiments run through four separate paths today: `scripts/gate_run.py`,
  `scripts/check_gates.py`, `scripts/direction_experiments.py` and the physics ablation
  (`scripts/ablate.py`, `experiments/ablation.py`). The owner on sweeps: "The system should have
  option to do quick sweeps or optuna sweeps. In quick we try to do everything as fast as by 5
  minutes, but optuna is measured, but can allow for overnight".
- **Decision:**
  - Restructure incrementally, engine first: one scenario and sweep specification, a resumable
    runner, one run store with an index, one scorer. The four old paths are frozen as history.
  - Every module move is checked by `scripts/golden_run.py verify`; the notebooks work at every step.
  - The control panel is a Jupyter notebook (ipywidgets and plotly). The same engine runs behind a
    CLI for long unattended sweeps, into the same run store. No web framework.
  - Two sweep modes: quick (the whole sweep in about 5 minutes) and Optuna (Bayesian search; GPU use
    in D-024).
  - Adding `optuna` to the local `nt` env and to the requirements is approved by the owner, who
    chose that option knowing it needs an install. The version must suit Python 3.10 and numpy
    1.23.5.
- **Consequence:** built by NT-026 (engine; supersedes NT-024), NT-027 (layering), NT-029 (config
  metadata), NT-030 (sweeps) and NT-034 (control-panel notebook 06). Any other change to the `nt`
  env still needs the owner.
- **Evidence:** [VISION.md](VISION.md) "The MVP" points 1 and 2 (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 3 (panel, Optuna), Round 5b
  (restructure), Round 6 (sweep modes).

## D-024 GPU use for sweeps (owner, 2026-09-28)
- **Context:** an Optuna sweep over dev folds can exceed the limit of about 3 GPU hours per backlog
  item (OPERATING_MODEL "Escalate to the owner"), and the owner's other project shares the GPU. The
  owner's words on the Optuna budget are quoted in D-023.
- **Decision:**
  - An Optuna sweep's GPU budget is measured and stated before it starts. It may run overnight while
    the owner's other project leaves the GPU idle.
  - Several training processes run at once only while the GPU is otherwise idle, with N set by one
    measured throughput test; otherwise one at a time.
  - One seed per trial per dev fold; the top 5 are re-run with 3 seeds; the final order and the
    winner use the seed mean (D-020).
- **Consequence:** for sweeps, this replaces the 3-GPU-hour limit. Pre-registered A/B studies keep
  that limit and the research budget (the lead's reading: the owner's answer was about sweeps). The
  GPU-free check (RUNBOOK) runs before each trial. NT-035 runs the throughput test. The GPU time for
  NT-006 is still a separate owner question.
- **Evidence:** [VISION.md](VISION.md) "The MVP" point 2 (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 6 (GPU budget), Round 7 (seeds,
  parallel GPU).

## D-025 Verdict rule: paired test plus a pre-registered minimum effect (owner, 2026-09-28)
- **Context:** the v1 physics ablation (D-003) judged its guard-rails by a point tolerance (h1
  direction AUC -0.0119 against 0.01).
- **Decision:** "A beats B" (learned against frozen indicators, a loss term on against off, any two
  scenarios) rests on a paired test over (seed, fold) pairs on the same blocks plus a minimum
  practical effect fixed before the run. Guard-rails are judged by the same test, not by a point
  tolerance. A deterministic TF mode is opt-in for comparison studies, after a measured speed test;
  sweeps stay fast.
- **Consequence:** supersedes the ablation verdict rule (`configs/ablation_criteria.yaml`) for new
  studies; the v1 ablation `runs/ablations/ablate_physics_v1-full` stays the record under its own
  criteria. NT-032 builds the comparator, with its error rates checked by simulation including block
  noise (D-012). NT-035 measures the deterministic mode's speed. NT-006 is pre-registered again
  under this rule; its GPU time still needs the owner (asked 2026-09-25).
- **Evidence:** [VISION.md](VISION.md) "The yardstick" (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 7.

## D-026 Gradient stability: invariants, per-run health, stress harness, config guard (owner, 2026-09-28)
- **Context:** the owner on unstable runs: "attribute and fail loudly, but this should be configured
  to not allow a configuration of hyperparameters that would fail the system".
- **Decision:**
  - Hard invariants run in CI in strict mode (the loss masks off).
  - Every run records gradient health at no more than 2% of `sec_per_step`; a per-loss-term probe
    (about 10%) is behind a flag.
  - An on-demand stress harness (scale and volatility sweep x0.1 to x10, extreme-input fuzzing,
    fault injection, 3 seeds; thresholds pre-registered) must be passed by every new setup (ticker,
    timeframe or horizons).
  - An unstable run is attributed to its loss term and fails loudly: Optuna prunes it and the
    leaderboard shows it as failed.
  - Config validation and sweep search spaces refuse hyperparameter regions the harness shows to
    fail, before a run starts.
  - Measure first, then one pre-registered A/B of gradient-based loss weighting against today's
    value-based calibration.
- **Consequence:** built by NT-036 (CI invariants, a "stability" pytest marker), NT-037 (per-run
  health; absorbs NT-012), NT-038 (harness and config guard) and NT-039 (the A/B, judged under
  D-025). The 2% cost is held to D-018. Once NT-036 exists, the definition of done runs the
  stability marker when loss, model, indicator or train-step code changes (OPERATING_MODEL).
- **Evidence:** [VISION.md](VISION.md) "The MVP" point 3 (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 8.

## D-027 Indicators extend through a registry; learned indicators on price first; D-014 on every figure (owner, 2026-09-28)
- **Context:** the owner on the indicators package: "Lets elaborate on that in a separate Q&A
  session. In general I want the list of indicators to be extendable and easily integrated via
  registry". On combinations: "Lets elaborate on that in a separate Q&A".
- **Decision:**
  - The indicator list is extendable and integrated through a registry. Its design, and where
    indicator combinations come from, are settled in a separate owner Q&A held right after this
    overhaul (2026-09-28, before the move to D:, D-030).
  - The first comprehension view: the learned indicators drawn on price (moving averages and
    Bollinger lines on price, RSI and MACD panels) against the textbook defaults, with how the
    periods moved over training and per window.
  - D-014 applies to every figure, including the comprehension views: there is no simplified tier.
- **Consequence:** an Indicators registry extends the nine of D-002. NT-046 stays a placeholder
  until the indicator Q&A; NT-043 builds the first view and notebook 07 (D-028).
- **Evidence:** [VISION.md](VISION.md) "What every run delivers", "The MVP" point 5 and the
  indicator catalogue (2c58370); [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Rounds
  4 and 9.

## D-028 Notebooks persist and evolve (owner, 2026-09-25 / 2026-09-28)
- **Context:** the owner: "I want the notebook structure to persist and even be updated during the
  development towards MVP".
- **Decision:** notebooks 00-05 keep their numbers and roles and are updated as the code changes;
  they stay the primary interface through the MVP (generated and executed per D-013). New views get
  new notebooks (06 control panel, 07 discovered indicators). Each figure has one home (the 01 / 04
  overlap is trimmed).
- **Consequence:** an item that changes what a notebook shows updates it through
  `scripts/notebooks/build.py` in the same item (OPERATING_MODEL definition of done). NT-034 builds
  06, NT-043 builds 07, NT-045 trims the overlap.
- **Evidence:** [VISION.md](VISION.md) "The MVP" point 5, "Principles" (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) owner statement after Round 3 (dated
  2026-09-25 there), Round 6.

## D-029 Delete only what is stale and has no effect (owner, 2026-09-28)
- **Context:** the restructure (D-023) removes old code. The owner: "only delete something which is
  both stale and doesnt affect current system in any sense".
- **Decision:** every deletion shows evidence of both: it is stale, and it has no effect (no
  production, notebook, script, doc or test use, and no default re-creates it). Anything with any
  effect stays. Deleting runs, data or remote branches still needs the owner (OPERATING_MODEL
  "Escalate to the owner").
- **Consequence:** NT-028 applies the rule to the stale candidates found by the 2026-09-28 survey.
- **Evidence:** [VISION.md](VISION.md) "The MVP" point 1 (2c58370);
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 6.

## D-030 The project moves to D:/neural_trade (owner, 2026-09-28)
- **Context:** C: is nearly full (the other project's Docker WSL image; NT-009), and runs, Optuna
  studies and the run index need room. Asked where the run store should live, the owner: "move the
  project to disk D".
- **Decision:** in this session, after the doc overhaul is committed and pushed and the indicator
  Q&A (D-027) is held, the project moves to D:/neural_trade: the repo with its untracked runs, both
  sibling worktrees (then `git worktree repair`), the editable install re-pointed with
  `pip install --no-deps -e .` from D: (an owner-approved env change), and the Claude memory folder.
  Tests and the notebook check are verified on D:, and the owner reopens VS Code there. The C: copy
  is deleted only after the owner confirms the D: copy works. Runs, the Optuna studies and the run
  index live in the repo on D:.
- **Consequence:** until the move, paths stay C:/Users/Step/Documents/neural_trade. NT-009 is
  resolved by the move and by VISION keeping the 2017-2025 file; it is done when the move is
  verified.
- **Evidence:** [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) Round 5b (run store),
  Round 6 (move), Round 9 (indicator Q&A before the move).
