# Decisions

Append-only log. An entry changes only through a new entry that cites new evidence ("Supersedes
D-0xx"). Owner decisions change only with the owner. Format: context, decision, consequence,
evidence.

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

## D-004 Never rewrite git history; master changes only when the owner merges (owner, 2026-09-22)
- **Decision:** no force-push, no rebase of pushed commits, no history filtering. Work happens on
  `remediation/plan` (the working branch). The lead commits and pushes the working branch at the
  end of each session; merging into `master` is an owner decision.

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
- **Evidence:** `tests/test_notebook_tooling.py`, `tests/test_notebooks_thin.py`.

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
