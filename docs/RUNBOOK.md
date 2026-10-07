# Runbook

How to run everything, and the traps of this machine. Commands were verified on 2026-09-25 unless
marked otherwise. Run them from the repository root. The repository is at `D:/nt/neural_trade` since
2026-09-28 (D-030, see the last section); commands here use relative paths.

```bash
PY=C:/Users/Step/miniforge3/envs/nt/python     # conda env nt: Python 3.10, TF 2.10 GPU. Do not use `conda run` (plugin crash).
export PYTHONIOENCODING=utf-8                   # the console code page is cp1251
```

## Environment

| Task | Command |
|---|---|
| Rebuild the env from nothing (Windows, GPU; not re-verified) | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`, then `conda activate nt`, `pip install -e ".[viz,dev]"` |
| Linux / CPU only (exactly what CI's `unit` job does) | `python -m pip install --upgrade pip && pip install -r requirements-ci.txt && pip install --no-deps -e .`, then `pytest -m "not gpu and not slow" --timeout=600 --cov=neural_trade ...` (see `.github/workflows/ci.yml`) |
| Environment fingerprint | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli env --no-devices` |

Changing the local env (install / upgrade / remove packages) needs the owner. Two changes are
already approved (owner, 2026-09-28): adding `optuna` for the sweeps (D-023) and re-pointing the
editable install at the D: copy (D-030, step 4 of the last section). Implementers and the
experimenter never change the nt env. `requirements-ci.txt` and the CI workflow pins are test
infrastructure: a backlog item may change them (TF stays 2.10.x).

**Installing optuna** (the lead, once; done 2026-10-06 before NT-030: dry run showed only new packages; installed optuna 5.0.0, sqlalchemy 2.0.54, alembic 1.20.0, mako 1.4.3, colorlog 6.12.0, greenlet 3.5.6; check line `5.0.0 1.23.0 2.10.0`):

1. `$PY -m pip install --dry-run optuna==<version>`, with the version NT-030 pins in
   `requirements*.txt` (pip 26.2 in the env supports `--dry-run`).
2. If any already-installed package would change version, stop and ask the owner (STATUS
   "Waiting"). numpy must stay at the installed 1.23.0 (checked 2026-09-28; `requirements-ci.txt`
   pins 1.23.5 for CI) and TensorFlow at 2.10.x.
3. Otherwise run the same command without `--dry-run`, then
   `CUDA_VISIBLE_DEVICES=-1 $PY -c "import neural_trade, optuna, numpy, tensorflow as tf; print(optuna.__version__, numpy.__version__, tf.__version__)"`
   and record the line in STATUS.

## Data (the reference setup)

BTC/USDT one-minute bars, a 60-minute window, horizons of 10, 15 and 20 minutes (VISION "The
reference setup"). It is the default and the only setup tested in the MVP; it is not what the
project is about. Every result names the setup it was measured on.

| Dataset | File | Used for |
|---|---|---|
| Reference dataset: BTC/USDT 1-minute, 30 days (2025-10-11 to 2025-11-10) | `binance_btcusdt_1min_ccxt.csv` (tracked, repo root) | the default `CSV_PATH`: training, tests, CI, notebooks |
| Long history: BTC/USDT 1-minute, 2017-01-01 to 2025-09-29 | `Bitcoin_BTCUSDT.csv` (291 MB, repo root, gitignored, the owner's) | walk-forward folds over different months (D-022). No run, test or notebook uses it yet; NT-041 adds the 7-day training block, walk-forward folds over the long history and a dataset fingerprint in every run. |

Today the data file is `CSV_PATH`, and the window (`LOOKBACK`), the horizons (`HORIZON_STEPS`,
exactly three) and the bar size (`RESAMPLE_MINUTES`) are Config keys in bars. NT-041 moves the
window and horizons to wall-clock time and NT-042 allows any number of horizons (D-022). Backtest
Sharpe and Sortino are annualised from the run's own bar size through one function,
`strategy.performance.periods_per_year(bar_minutes, calendar="24/7")` (NT-040): every live path (the
CLI backtest, the notebook explorer, the engine's scorer) sets `BacktestConfig.bar_minutes` from
`RESAMPLE_MINUTES`, so `RESAMPLE_MINUTES` is tunable (NT-029's metadata; a sweep may search it once
NT-030 exists).

## Tests, lint, coverage, docs

| Task | Command | Time |
|---|---|---|
| Tests for what changed (tier 1, D-060) | `$PY scripts/test_changed.py --run` (list only without `--run`; `--base <ref>`) | under a minute |
| Fast suite | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow" -n 8` | 3 min 26 s at 8 workers (2026-10-01; 7 min 20 s serial; 16 workers are not faster: six ~30-55 s training tests bound it, NT-109). Never two full suites at once (D-048) |
| Slow suite (training, CLI and predictor round trips, reproducibility, notebook execution on synthetic bars) | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m slow -n 8` | ~4 min serial on an idle machine; 38 min seen when three suites ran at once (2026-10-01) |
| Stability invariants (NT-036) | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m stability` | not measured |
| Lint | `$PY -m ruff check src tests scripts` (CI lints `src scripts` with ruff 0.6.9; locally 0.16.8) | seconds |
| Coverage and per-package gates | `COVERAGE_FILE=<scratch>/.coverage CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow" --cov=neural_trade --cov-report=json:<scratch>/coverage.json` then `$PY scripts/check_coverage.py <scratch>/coverage.json` | ~6 min |
| Refactor guard: a small deterministic CPU run (D-023: every module move) | `$PY scripts/golden_run.py record OUT.npz` before, `$PY scripts/golden_run.py verify OUT.npz` after | not measured |
| Regenerate TESTING_DOCUMENTATION.md | `CUDA_VISIBLE_DEVICES=-1 $PY scripts/test_inventory.py` | seconds |

**CI.** GitHub Actions `ci` (lint + unit on Linux CPU, `requirements-ci.txt`). CI pins the local
env's versions of plotly, narwhals, pandas, scikit-learn, scipy and jinja2 (NT-001: CI tests what
the owner runs; the size budgets stay version-specific). The other pins still differ from the local
env (for example numpy 1.23.5, matplotlib 3.7.5, ruff 0.6.9), so green locally does not guarantee
green on CI. After every push, check the branch head (the GitHub CLI is not installed; job logs
need auth, the run list does not):

```bash
curl -s "https://api.github.com/repos/another-worlds/neural_trade/actions/runs?branch=remediation/plan&per_page=5" \
  | $PY -c "import json,sys; [print(r['head_sha'][:7], r['name'], r['status'], r['conclusion']) for r in json.load(sys.stdin)['workflow_runs']]"
```

For one run's jobs (public, no auth): `curl -s https://api.github.com/repos/another-worlds/neural_trade/actions/runs/<run id>/jobs`.
Job logs need auth. Failing test names do not: the `unit` job writes each failed test as an
annotation (NT-001), readable at `curl -s https://api.github.com/repos/another-worlds/neural_trade/check-runs/<job id>/annotations`
(the job id from the jobs call above).
Wait at most 30 minutes for a run (OPERATING_MODEL, definition of done).

Open pull requests (remote sessions open them into `remediation/plan`; D-033):
`curl -s "https://api.github.com/repos/another-worlds/neural_trade/pulls?state=open&per_page=20" | $PY -c "import json,sys; [print(p['number'], p['head']['ref'], p['head']['sha'][:7], '->', p['base']['ref'], p['title'][:80]) for p in json.load(sys.stdin)]"`.

**Rate limit (trap, 2026-09-28).** The unauthenticated API allows 60 requests per hour per IP, and every
session and agent on this machine shares them: a 30-second poll loop plus a few agents' checks used them
all within the hour, and every call then returns `403 rate limit exceeded` until the `X-RateLimit-Reset`
time (`curl -s -i ... | grep -i x-ratelimit`). Poll a run at most every 3 minutes, and prefer one check
after the run's usual duration (about 10 minutes for `ci`). The web page
`https://github.com/another-worlds/neural_trade/actions/runs/<run id>` is not under that limit and says
"In progress" while a run is going, but its job results are not reliably readable without a browser.

The nightly workflow (`.github/workflows/nightly.yml`) never runs: GitHub registers scheduled
workflows only from the default branch, and `master` has no `.github/` yet (NT-008).

## The CLI

| Task | Command |
|---|---|
| Overview | `$PY -m neural_trade.cli --help` (or `neural-trade --help`) |
| Train, evaluate on test, save a serving bundle (GPU; not re-verified) | `$PY -m neural_trade.cli train --epochs 20` or `--config configs/default.yaml --set LR=5e-4 --set EPOCHS=10` |
| Forecast with a saved bundle, on the reference dataset | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli predict --artifacts runs/<id>/artifacts --csv binance_btcusdt_1min_ccxt.csv --last` |
| Backtest with a saved bundle, on the reference dataset (in-sample, see below) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli backtest --artifacts runs/<id>/artifacts --csv binance_btcusdt_1min_ccxt.csv --out <dir> [--strategy NAME] [--plot]` |
| Registry contents | `$PY -m neural_trade.cli registry list`, `registry info Optimizers adamw`, `registry search QUERY` |

**`neural-trade backtest` scores every window of the data file, the training blocks included: its
numbers are in-sample.** Out-of-sample backtests: notebook `02_backtest` (the test block) or
`scripts/backtest_gate.py`.

## Experiments and gates

| Task | Command |
|---|---|
| Physics-term ablation: plan / pending cells | `$PY scripts/ablate.py --scale full --out runs/ablations/<name> --dry-run` |
| Run the ablation (resumable) | `$PY scripts/ablate.py --scale full --out runs/ablations/<name>` (7-10 GPU hours: needs the owner) |
| Judge the M1-M4 gates clause by clause | `CUDA_VISIBLE_DEVICES=-1 $PY scripts/check_gates.py runs/gates` (exits 1 while M3 fails: by design) |
| Produce a gate run, then its backtest | `$PY scripts/gate_run.py --name m7 --epochs 20`, then `$PY scripts/backtest_gate.py runs/gates/m7` |
| Direction experiments on the dev folds -3 / -2 | `$PY scripts/direction_experiments.py --out runs/experiments/<name> [--only NAME ...]` |
| Compare scored runs | `$PY -c "from neural_trade.experiments.compare import compare_runs; print(compare_runs('runs/*', metrics=['h1/direction/auc'], skip_unscored=True))"` |

Pre-registered studies follow `.claude/agents/experimenter.md` and OPERATING_MODEL "Sweeps and
pre-registered studies" (which also covers sweeps): a SPEC.md (QA-checked before GPU time), a
pinned worktree (`git worktree add --detach D:/nt/nt_exp_<name> <sha>`, `PYTHONPATH` set to its
`src`), a verdict judged once, a REPORT with run ids. A verdict's pairs are (seed, fold) over judgement folds that no
choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more
held-out folds from the long history once NT-041 exists); at least 5 pairs. Never reuse a
`scripts/gate_run.py` `--name`: it deletes an existing run directory (`scripts/gate_run.py:157-158`).
NT-024, which would have made it refuse, was dropped for NT-026; the script is in the frozen set
(D-023), so this rule stays.

### Run directories in git

Every number a doc, report or notebook cites links to a tracked run (NT-010). A run directory is a
directory under `runs/`, at any depth, whose name starts with a run id
`YYYYMMDDTHHMMSSZ-<sha7>[-dirty]-<hash8>` (`runs/<id>/`, `runs/ablations/<name>/runs/<id>-<cell>/`,
`runs/experiments/<name>/runs/<id>-<name>/`). Every file in it is tracked except the heavy artefacts,
which `.gitignore` ignores: weights `*.h5`, `*.joblib`, `*.pkl`, `*.npz`, `*.parquet` and `tb/`. The
experiment log folders `runs/ablations/*/cells/`, `runs/ablations/*/logs/` and
`runs/experiments/*/logs/` are ignored too. Tracked: `config.yaml`, `meta.json`, `status.json`,
`env.json`, `metrics.jsonl`, `eval_report_*.json` and `.md`, `training_log.csv`,
`indicator_params_history.csv`, `period_init.json`, `artifacts/meta.json`, `artifacts/config.yaml`,
`artifacts/calibration/*.json`. An experiment's `result.json` (in `runs/experiments/<name>/<variant>/`,
outside the run directories) is tracked too.

`scripts/check_run_evidence.py` finds the run ids cited in `docs/**/*.md`, `README.md`,
`runs/**/REPORT.md`, `report.md`, `summary.md` and the saved notebooks, and exits 1 when a cited run
has no directory, git does not track its `config.yaml` or its `meta.json`, or a light file in it is
untracked. A fast-suite test runs it, so CI enforces it. Stage a new run's light files by explicit
path, in the commit that first cites the run (for notebook 01, with the executed notebooks):

```bash
$PY scripts/check_run_evidence.py --list-untracked | git add --pathspec-from-file=-
```

A run made on another machine (for example the remote review session's, D-033) has no directory here:
list its id in `runs/EXTERNAL_RUNS.md` (id, where it ran, why it is not here, the citing record). The
check then passes it without a directory and counts it as external; a listed id that does have a
directory is checked normally, and the file itself is not a citing file.

For a run nothing cites yet, `git ls-files --others --exclude-standard -- runs/<run dir>` lists the
same files. Never `git add runs` (CLAUDE.md start step 2).

### GPU rules

**Before any GPU job, check that the GPU is free:** `nvidia-smi dmon -s um -c 10` (10 one-second
samples; `sm` = utilisation %, `fb` = memory used MB). Idle on this machine (measured 2026-09-25):
fb about 700 MB (the Windows desktop), sm mostly under 30%. The GPU is **busy** if fb is above
2000 MB or the median sm is above 30% (the owner's Docker/WSL project shows up this way, often as
pid 0). Per-process memory `N/A` is normal under WDDM. If busy: do not start; do CPU work and check
again later; after about 2 hours of waiting, record it in STATUS. Disk: `df -h /c /d`; at least 5 GB
free on the target drive, else write to D:.

How much GPU a job may use is ruled in one place: OPERATING_MODEL "Sweeps and pre-registered
studies" (studies, sweep budgets, and the one-night cap on one sweep launch, which is the lead's
reading of D-024 and NT-030's default `--max-hours`) and "Escalate to the owner". What this machine
adds:

- **Per-process memory cap (opt-in):** set `NT_GPU_MEMORY_LIMIT_MB=<MB>` (e.g. 5000) before `python -m neural_trade.cli ...` to cap that process's GPU memory so two screen trainings can share the 12 GB card; unset or 0 changes nothing.
- **Default: one GPU job at a time.** The lead's notebook routine (01 trains about 5 minutes) is one
  GPU job like any other.
- **The tactical session (D-063):** runs its ultra-short screen trials (at most 2 minutes each) in
  parallel with the MVP session's GPU job; neither waits for the other. A slower MVP `sec_per_step`
  measured while tactical trials ran is not evidence for D-018: re-measure with the GPU otherwise idle.
  [TACTICAL.md](TACTICAL.md) "GPU".
- **The budget's `sec_per_step`** comes from the `status.json` of the latest real run of the same
  setup (NT-030 (3)).
- **Parallel sweep trials** (`--parallel N` above 1, NT-030 (4)). The check above cannot see which
  process uses the GPU (per-process memory is `N/A` under WDDM), so while the sweep's own trials run
  it would always read "busy". Therefore:
  1. The GPU-free check runs only when none of the sweep's own trials is running.
  2. Trials launch in batches of N, and a batch starts only after that check passes.
  3. While a batch runs, the sweep watches utilisation. If it rises above the level NT-035 recorded
     for N of the sweep's own processes (someone else is on the GPU), the sweep stops launching.
  4. `--parallel` above 1 is refused unless NT-035's recorded result allows that N (a test with a
     stubbed record). Until NT-035 has run, N = 1.

### Experiment engine

One scenario spec, one resumable runner, one run store with an sqlite index, one scorer (NT-026;
code in `src/neural_trade/experiments/`: `scenario.py`, `runner.py`, `store.py`, `scorer.py`,
`dataset.py`; strategy studies on stored cells: `rescore.py`, NT-076). New experiments go here, not
into the frozen set.

| Task | Command |
|---|---|
| Check a spec and list its cells with their state (no training, writes nothing but the index) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli scenario plan configs/scenarios/reference.yaml` |
| Run or resume a scenario (GPU unless `CUDA_VISIBLE_DEVICES=-1`; GPU rules above) | `$PY -m neural_trade.cli scenario run configs/scenarios/reference.yaml [--max-cells N] [--retry-failed]` |
| Rebuild the index from the run directories | `$PY -m neural_trade.cli scenario reindex [--store runs]` |
| Re-score a strategy study on the scenario's stored cells (CPU, no training) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli scenario rescore configs/scenarios/reference.yaml --study configs/strategy_studies/example.yaml [--store runs] [--random-seeds N]` |

- **Spec** (YAML, `schema_version: 1`): `name`, `base_config` (a flat Config YAML, relative to
  the spec), `overrides`, `variants` (named Config overrides), `sweep: {mode: grid, axes: {FIELD:
  [values]}}`, `folds` (FOLD_INDEX values), `seeds`, `strategy: {name, params}` (Strategies registry,
  default calibrated_quantile), `backtest` (BacktestConfig fields; default costs 0, D-044),
  `run: {calibrate, save_artifacts, indicator_report}`. `indicator_report` writes
  `indicator_report.html` from `artifacts/` and requires `save_artifacts`. Unknown keys,
  unknown or invalid Config values, unregistered components, folds the data does not have
  and engine-owned fields (FOLD_INDEX, SEED, MODEL_PATH, SCALER_PATH, ARTIFACTS_DIR,
  bar_minutes) are refused before anything trains. Run from the repository root: a relative
  `CSV_PATH` resolves against the working directory.
- **Cells and run store.** Each (variant x grid point, fold, seed) cell trains into its own
  directory `runs/scenarios/<name>/<run id>-<configuration>__f<fold>__s<seed>/` with the usual run
  files plus `meta.json` sections `engine`, `dataset` (file sha256, first and last timestamp, bar
  count), `setup` (bar minutes, LOOKBACK, HORIZON_STEPS) and `blocks`, the scorer's
  `eval_report_<dev|test>.json/.md`, the stored predictions (below) and `result.json` (status,
  error, scores). The runner never writes into an existing directory and deletes nothing.
- **Stored predictions (NT-076).** Every scored cell also writes `predictions_oos.npz` (its
  out-of-sample block) and `predictions_cal.npz` (its calibration block): the block's
  PredictionFrame (served and raw head deltas, the delta-shrinkage betas, P(up) raw and calibrated,
  variances, conformal intervals, pred_scale / pred_mean, HORIZON_STEPS; not the input windows) plus
  the block's OHLC bars at its anchor bars, the anchor timestamps and the bar size
  (`PredictionFrame.save_npz` / `load_npz`, `experiments.scorer.save_predictions` / `load_block`).
  They are **heavy and machine-local**: about 1.2 MB per cell on the reference setup, `*.npz` under
  `runs/` is git-ignored, so a clone has the light files only. Cells scored before NT-076 have none.
- **Scoring.** The out-of-sample block of each cell's fold is scored with evaluation/report.py and
  backtested with the scenario's strategy (knobs fitted on the fold's cal block, next-open fills,
  buy-and-hold, always-flat and the size-matched random null). The latest usable fold is `test`
  (shown, never ranks, D-020); the earlier folds are `dev`.
- **Index.** `runs/index.sqlite` (tables `runs` and `scores`; `--index` moves it) is derived from
  the run directories' light JSON files; delete it freely, `scenario reindex` rebuilds it. Resume:
  a cell counts as finished when a `done` (or, without `--retry-failed`, `failed`) run with the same
  config hash and scoring settings exists; an interrupted cell (no `result.json`) stays on disk as
  `incomplete` and trains again into a new directory.
- **Strategy studies (`scenario rescore`, NT-076).** A study spec (YAML, `schema_version: 1`,
  `configs/strategy_studies/`): `name`, `description`, `entries: [{id, strategy, params, backtest,
  grid}]`; `grid: {knob: [values]}` expands into one configuration per combination with ids like
  `cq[entry_quantile=0.8]`. Unknown keys, unregistered strategies, unknown strategy or backtest
  knobs, knobs a calibrated strategy fits on cal (long_above, short_below, median), engine-owned
  backtest fields and repeated ids are refused before anything runs (exit 2). Every configuration is
  backtested on every `done` cell of the scenario spec (same config hash and run.calibrate) that has
  stored predictions, exactly as the scorer does it (the same `scorer.fit_and_backtest`: knobs
  fitted on the cell's cal block only, next-open fills, the scenario's `backtest:` costs with the
  entry's `backtest` on top, bar size from the run's config, buy-and-hold, always-flat and the
  size-matched random null, `--random-seeds` overriding its seed count); the scenario's own strategy
  reproduces each cell's `result.json` scores exactly. Cells without stored predictions, failed or
  incomplete runs, runs of an older spec and older duplicates of a cell are skipped and listed
  (stdout, stderr log, meta.json); with no dev cell left the command exits 1 and writes nothing.
  Output: a new directory `runs/scenarios/<name>/rescore/<study>-<UTC stamp>/` (never a cell
  directory; light files, tracked like any run's): `cells.csv` (configuration x cell: fold, seed,
  role, the backtest summary and the baselines), `leaderboard.csv` / `.md` (one row per
  configuration, **ranked by the mean dev-cell net Sharpe**; sd, mean net return, max drawdown,
  trades, the share of dev cells beating buy-and-hold, the mean random-null percentile; the test
  cells in `test_` columns and a separate table, never ranking, D-020), `study.yaml` (the normalised
  spec) and `meta.json` (git sha, scenario spec hash, cells used and skipped, configurations). Cost:
  about 2 s per configuration and cell with 100 random-null seeds on the reference setup (measured
  once on fake cells, CPU; an estimate for other machines).
- **Long runs:** launch detached like any long job (next section) and resume with the same command.
- Notebooks 02-05 never pick an engine run by default (`pick_run` skips `runs/scenarios/` and any
  run whose meta.json has an `engine` section).

### Paired comparator ("A beats B", NT-032, D-025, D-046)

`src/neural_trade/experiments/comparator.py`: a pre-registered paired test over two engine scenarios'
runs, for every "A beats B" verdict D-025 asks for (learned against frozen, a loss term on against
off, any two scenarios). It never trains, and it never touches the sqlite index: it reads the run
directories `scenario run` already wrote directly (meta.json / result.json), so `runs/index.sqlite`
is unchanged by a compare.

**The unit of inference is the judgement fold, not the (seed, fold) pair (D-046).** Seeds trained on
the same fold share that fold's block noise (the same bars, the same realised path); treating them as
independent pairs inflates the false-"beats" rate well past 5% (measured: 0.069-0.314 for one fold x
5 seeds, depending on the minimum effect). A comparison therefore needs `min_folds` (>= 5, never
lower) distinct judgement folds with a usable pair; within each fold, its seeds' paired differences
are averaged first, and the paired test runs over those fold means.

| Task | Command |
|---|---|
| Compare two scenarios by a pre-registered spec | `$PY -m neural_trade.cli compare configs/compares/<name>.yaml` |
| ...and write `result.json` / `report.md` | `$PY -m neural_trade.cli compare configs/compares/<name>.yaml --out runs/compares/<name>` |
| ...and add the calibrated null/power simulation | `$PY -m neural_trade.cli compare configs/compares/<name>.yaml --simulate [--n-sim N]` |

Exit codes: `0` a verdict was reached (any of "A beats B" / "B beats A" / "inconclusive"); `1` the
comparison was refused (too few judgement folds, a pre-registration violation, an ambiguous
configuration, a pair-count mismatch: see `refusal_reason` and `excluded_pairs`); `2` the spec itself
is malformed (unknown key, invalid value) before any comparison is attempted.

**Writing a spec** (YAML; `neural_trade.experiments.comparator.CompareSpec`):

```yaml
name: learned_vs_frozen_h1_auc          # also the registration sidecar's name (below): pick a new name for
                                         # a materially different comparison, never reuse one for an edited spec
scenario_a: reference_learned           # runs/scenarios/<scenario_a>/ (must already exist when you compare)
scenario_b: reference_frozen            # runs/scenarios/<scenario_b>/
configuration_a: default                # optional: which scenario configuration to use, when a scenario has
configuration_b: default                #  more than one (a spec's variants/grid, NT-026); required whenever a
                                         #  (seed, fold) is shared by more than one configuration (refused
                                         #  otherwise -- never guessed, D-046)
duplicate_policy: refuse                # refuse (default; lists the run ids), latest, or average: how to
                                         #  handle more than one done run for the SAME (seed, fold, configuration)
metric: h1/direction/auc                # a key of result.json's "scores" (the same names the index's
                                         # scores table and notebooks use, e.g. h1/variance/crpss,
                                         # backtest/sharpe_net, a coverage metric -- any of them work the same)
direction: higher_better                # or lower_better
metric_kind: diff                       # or log_ratio: ln(A/B) (or ln(B/A) under lower_better)
min_effect: 0.01                        # the minimum practical effect, fixed before any GPU time
judgment_folds: [-5, -4, -3, -2, -1]    # FOLD_INDEX values no earlier choice used (D-025); a pair on
                                         # any other fold is excluded, not silently dropped
min_folds: 5                            # >= 5, never lower (D-046): the number of DISTINCT judgement
                                         # folds needed for a verdict, not the pair count
min_pairs: 5                            # a secondary floor on the total (seed, fold) pair count
pairs_planned: 5                        # optional: refuses a comparison with any other pair count
                                         # (no peeking); a list (one per look) when looks > 1
alpha: 0.05
registered_utc: "2026-09-30T12:00:00Z"  # REQUIRED: there is no "now" default (a spec that does not
                                         # declare this is not pre-registered). Refused if any compared
                                         # run started before it. Accepts this ISO form or the run
                                         # store's compact form (20260930T120000Z). If the spec file is
                                         # committed to git AND the working copy matches that commit
                                         # exactly, its last commit time is used instead (more
                                         # trustworthy than a string nothing stops you editing) -- see
                                         # "registered_utc_source" in the output. An uncommitted edit
                                         # (the working copy differs from HEAD) never borrows the old
                                         # commit's time, nor this declared value (an edit keeps it):
                                         # it counts as registered at compare time, source "compare
                                         # time (working tree differs from HEAD)", so any run that
                                         # already exists makes it post hoc and it is refused. Commit
                                         # the spec before the runs start.
estimator: mean                         # or hodges_lehmann (+ its exact Wilcoxon interval): robust to
                                         # one bad fold, D-037. No exact 95% Wilcoxon interval exists
                                         # below n = 6 folds (n_folds = 5 always falls back); the
                                         # comparator falls back to the t interval rather than
                                         # mislabelling a lower attained confidence as 95% (D-046)
non_inferiority_margin: 0.02            # optional: adds a pass/breach/undecided non-inferiority read
looks: 1                                # > 1: a Pocock two-look design (D-037); the per-look alpha
                                         # tightens automatically (neural_trade.metrics.statistics.pocock_alpha);
                                         # pairs_planned is still enforced at every look
guard_rails:                            # judged by the SAME paired test over folds, never a point tolerance
  - metric: h1/variance/crpss
    direction: higher_better
    max_degradation: 0.01               # breach iff the CI is confidently worse than this
noise_sd: 0.02                          # for --simulate: the null/power check (see below), when the fold and
                                         # seed noise are not modelled separately; or give
seed_sd: 0.015                          #  seed_sd (between seeds sharing a fold) and
block_sd: 0.01                          #  block_sd (the fold's own, shared-by-every-seed noise)
root: runs                              # the run store root (default "runs")
```

**Registration is two separate, both-enforced checks (D-046).** (1) `registered_utc` (or the spec
file's git commit time, when it has one AND the working copy matches that commit -- an uncommitted
edit falls back to the declared value instead) must predate every compared run's `created_utc`.
(2) The spec's *content* is locked the first time a `name` is compared: `<root>/compares/<name>/registration.json`
records its `spec_hash`, and a later call under the same name whose hash differs (anything edited,
even with `registered_utc` untouched) is refused. Practically: write the spec, run `compare` on it
once (even before enough runs exist -- it still records the hash) to lock it in, and give a genuinely
revised comparison a new `name`.

A pair is excluded (listed, with its reason and BOTH sides' values, in `excluded_pairs`), not
silently dropped, when its fold is not in `judgment_folds`; its dataset, setup or **judged
out-of-sample block** fingerprint (`dataset_sha256`, `bar_minutes`, `HORIZON_STEPS`, `LOOKBACK`, and
all four of the block's own `start`, `stop`, `first_timestamp`, `last_timestamp` when present -- an
anchor hash per block, D-037; two blocks sharing the same bar indices but different timestamps, e.g.
a slice of a different file, still count as a mismatch) differs between A and B; its metric is
missing; a `log_ratio` metric is non-positive; more than one configuration shares a (seed, fold) and
none was named (`configuration_a` / `configuration_b`); or more than one done run exists for the same
(seed, fold, configuration) and `duplicate_policy` is `refuse` (the default). The whole comparison is
refused (`verdict: "refused"`, exit code 1) when fewer than `min_folds` judgement folds have a usable
pair, fewer than `min_pairs` total pairs survive, either registration check fails, or the pair count
does not match a pre-registered `pairs_planned`. The verdict ("A beats B" / "B beats A" /
"inconclusive" / "refused") comes from the paired interval, over fold means, against `min_effect`;
every guard-rail gets its own pass/breach/undecided the same way.

`--simulate` adds a fast (well under a second for 1,000+ simulations), seeded null/power check
(D-025's acceptance criterion, D-037's "calibrated to the measured variance components") that models
the real fold x seed structure: each simulated fold draws one shared `block_sd` value (or `noise_sd`
when seed/block are not modelled separately), and each of its seeds adds independent `seed_sd` noise
on top, then the fold mean is what the paired test actually runs over -- the false "beats" rate under
a true null effect (must be <= 0.05 + its own Monte Carlo error) and the power at twice `min_effect`.
The CLI calibrates the simulated design (`n_folds`, `seeds_per_fold`, shown in the output) to the
comparison's OWN actual pairing (`comparator.observed_design`: the number of judgement folds that
paired at least one run, and the pair count divided by that), not a guessed default -- a simulation
run before any pairs exist falls back to `(spec.min_folds, 1)`.

Two building blocks the D-037 amendment asks for are library functions, not spec keys (a study wires
them into its own spec / reporting): `comparator.per_fold_retention` (the generic
`r_f = mean(diff) - baseline / denom` one-sided check A/B-1 will use for its CRPS-edge retention
margin) and `comparator.intersection_union_verdict` (ADOPT only when every named component passes;
a breach on a name in `owner_route` routes to the owner instead of rejecting, for a D-018 speed
guard-rail). What is not implemented (contention/re-time metadata, the literal A/B-1 primary metric,
infinite pairs kept in the rank statistic rather than excluded) is documented in `comparator.py`'s
module docstring and the NT-032 backlog entry.

### Screen mode

Mass, sub-30-second CPU/GPU trials over a grid and/or a random/LHS sample of Config fields (NT-088,
`docs/research/2026-09-29-screen-plan.md`): level 1 of the plan finds broken math and unstable
hyperparameter regions on hundreds of tiny (reference size: a 6-hour training block) configurations;
it ranks nothing (level 2 re-runs survivors through `scenario run` on a real block, where quality is
measurable). Code: `src/neural_trade/experiments/screen.py`, a separate path from the experiment
engine above (`scenario run`'s scoring is untouched; a change here never touches it).

| Task | Command |
|---|---|
| Run or resume a screen (CPU: `CUDA_VISIBLE_DEVICES=-1`) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli screen configs/screens/example_6h.yaml [--store runs] [--max-trials N]` |
| Split a screen across N processes (disjoint shards, their union is every trial) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli screen configs/screens/example_6h.yaml --shard 0/3` (repeat with `1/3`, `2/3`; NT-035's 3-process ceiling) |

- **Spec** (YAML, `schema_version: 1`, `configs/screens/`): `name`, `base_config` (a flat Config
  YAML, relative to the spec), `overrides` (applied to every trial), `grid: {axes: {FIELD:
  [values]}}` (a Cartesian grid, like a scenario's `sweep.axes`), `sample: {n, method: random|lhs,
  seed, space: {FIELD: {low, high, log}}}` (drawn in addition to, not crossed with, the grid;
  every field needs explicit `low`/`high` bounds, since most Config fields have no upper bound),
  `slices` (a non-empty list of `DATA_END` timestamps, or `null` for today's newest-bars behaviour;
  a screen trains the SAME configuration on different points in history, e.g. quiet vs. volatile
  weeks), `seeds`, `run: {calibrate, epochs}` (`epochs`, if given, sets `EPOCHS` for every trial),
  and `rules` (below). Every `(grid point or sample point) x slice x seed` is one trial. Unknown
  keys and unknown or invalid Config fields (in `overrides`, `grid.axes` or `sample.space`) are
  refused before anything trains, the same way a scenario spec is (`Config.field_names()`).
- **DATA_END** (Config field, `core/config.py`): ends the prepared data at this timestamp instead
  of the file's newest bar (`None`, the default: today's behaviour everywhere else, including
  `scenario run`). The slice is taken in `DataProcessor.load_and_prepare_data`, BEFORE
  `MAX_SEQUENCE_COUNT` trims from the end of the slice, so a trial's training block sits anywhere
  in history, not only at the file's tail. **Protected span (D-020):** a `DATA_END` that falls
  within the last `DATA_END_PROTECTED_DAYS` days of the FULL file (default 64) is refused with a
  clear error — a screen trial must never be able to slice into the long file's held-out dev/test
  period. This is checked for EVERY trial before ANY trial trains
  (`experiments.screen._preflight_data_end`, a preflight over the whole spec: one violating slice
  refuses the whole run, not only itself), and it covers the implicit `DATA_END: null` ("use the
  newest data") case too — the newest bar is trivially inside the file's own protected span for any
  `DATA_END_PROTECTED_DAYS >= 0`, so a screen spec always needs an explicit, sufficiently old
  `DATA_END`; there is no config that lets a screen trial use "whatever's newest". A screen spec
  also cannot lower `DATA_END_PROTECTED_DAYS` below the Config default against a file that actually
  spans at least that default (closing the obvious way to sneak a `DATA_END` past the guard) — but
  MAY lower it against a file SHORTER than the default to begin with, which is why the bundled
  30-day CSV (tests, `configs/screens/example_6h.yaml`) needs the exception: the real 64-day default
  would refuse every `DATA_END` outright on a file that short. A real campaign against the local
  long 2017-2025 file (which does span more than 64 days) always keeps the 64-day floor.
- **The light training path.** A trial trains through `experiments.screen._run_trial_light`, not
  `train_and_evaluate`: no baselines, backtest, random null, stored predictions (`*.npz`),
  checkpoints or serving bundle, and — by default — no per-trial run directory at all (only the
  JSONL row below). `run.calibrate` switches the pre-training loss-weight calibration pass on or
  off, same meaning as a scenario's `run.calibrate`. Data (load, preprocess, the `DATA_END` slice)
  AND its sliding windows (`DataProcessor.build_windows`, the part that loops every bar — 10-18s on
  the long file) are each cached **once per data key per process**
  (`experiments.dataset.data_key`, which already covers every Config field that can change the
  prepared bars or the windows, `DATA_END` included): many trials that only vary a hyperparameter
  like `LR` share one load AND one windowing pass. Only the fold split, target scaling and window
  normalisation (`DataProcessor.prepare_datasets_from_windows`, cheap: no per-bar loop) and the
  model itself are built fresh per trial, since fields outside the data key (`N_FOLDS`,
  `VAL_FRACTION`, `CAL_FRACTION`, `WINDOW_NORMALIZER`, `FOLD_INDEX`, `BATCH_SIZE`, ...) may still
  vary trial to trial.
- **Health numbers and rules.** Every trial's JSONL row (`<store>/screens/<name>/results.jsonl`,
  one line per trial) carries: whether every logged value was finite, `nonfinite_grad_steps`, the
  max and mean of the per-logged-step `grad_global_norm` and the share of logged steps at or above
  `GRAD_CLIP_NORM` (a small per-batch sampler reads the exact epoch accumulator
  `CustomTrainModel` already keeps; "logged steps" respects `TRAIN_METRICS_EVERY` — screens usually
  set it to 1 for exact per-step numbers, since the cost is negligible at screen sizes), the
  training loss's first-to-last-epoch drop, the final validation loss, each loss term's share of
  the final total loss (WEIGHTED by that term's actually-applied lambda — read from the trained
  model itself (post-`run.calibrate`, post-`ABLATE_LAMBDAS`), not the pre-run `cfg.LAMBDA_*` — see
  `experiments.screen._term_multiplier` — so a huge `LAMBDA_*`, or one calibration rescaled, still
  shows up as its true share and `max_term_share` can catch it), per-horizon direction AUC on the validation block (labelled
  with its noise level, D-012: `n` and `n_eff = n // horizon bars`), the trial's config diff,
  `DATA_END` and seed, and a timing breakdown (below). The spec's `rules:` block
  (`finite`, `max_nonfinite_grad_steps`, `max_clipped_share`, `min_train_loss_drop`,
  `max_term_share`) turns the health numbers into `passed: true/false` with `reasons`. A non-finite
  health number (an extreme `LAMBDA_*` blowing up training) is written to the row as `null`, never
  as a raw NaN/Infinity (which strict JSON, and this row's own `json.dumps(allow_nan=False)`, both
  refuse — the previous behaviour left the trial permanently unrecorded and unresumable): such a
  trial is always `passed: false`, with a reason naming the non-finite field(s), regardless of the
  spec's `rules:`.
  `clipped_share` is an APPROXIMATION: `training/custom_model.py`'s `train_step` clips gradients in
  TWO SEPARATE groups (network weights, indicator logit variables), each against its own group
  norm, but the sampler reads the PRE-CLIP norm of the COMBINED gradients (computed once, before the
  split) — see `_GradNormSampler`'s docstring for exactly what this over/under-counts.
- **All nine heads (`head_metrics`, additive; `direction_auc` is unchanged).** Every row also has
  `head_metrics[h0|h1|h2]` on the same validation block, raw heads (a screen has no calibration), each group
  with `n` and `n_eff = n // horizon bars`: `delta` {`corr`, `skill_vs_zero` = 1 - MSE/MSE of zero}, `direction`
  {`auc`, `brier`, `log_loss`, `hit_rate`, `mean_abs_p_dev`} (non-deadband bars, the AUC mask), `variance` {`crps`
  of N(predicted delta, predicted variance), `crpss` against a constant variance fitted on the training block,
  `nll`, `coverage90`, `width90`, `corr_var_err2_spearman`} (delta and variance on every bar). Non-finite is
  `null`. **`run: {save_predictions: true}`** (default false) also writes
  `<store>/screens/<name>/preds/<trial_key>.npz` (float32, compressed): `y` [n, H] realised raw deltas,
  `last_close`, and per horizon `delta_hX`, `p_up_hX`, `var_hX` (raw head outputs; `var` in scaled units, sigma =
  sqrt(var) x `pred_scale`), plus `pred_scale`, `pred_mean`, `deadband_bps`, `horizon_steps`. No anchor timestamps
  (the validation block carries none). Code: `screen._direction_auc` / `_head_metrics_one`.
- **Timing breakdown.** `load_s` (the data load, cached-or-not), `prep_s` (windowing, cached-or-not,
  plus the per-trial split/scale/normalise), `build_s` (`Models.build` + optimizers + compile only),
  `train_s` (`fit`), `score_s` (health + direction AUC) and `epoch_s` (a list of per-epoch
  wall-clock seconds). For the phase-2 tracing decision (whether the first epoch's one-off
  `tf.function` tracing cost is worth avoiding at screen sizes), estimate
  `trace_time ~= epoch_s[0] - median(epoch_s[1:])` from `epoch_s`.
- **Resumable, shardable.** A trial's key is a hash of its exact Config values
  (`experiments.scenario.config_hash`); a key already recorded is skipped, so the same command
  resumes (`--max-trials` stops early on purpose). `--shard i/N`: trial index `j` runs in this
  process only when `j % N == i` — the shards are disjoint and their union is every trial. Each
  shard writes its OWN file, `results.shard-{i}-of-{N}.jsonl` (0-indexed, in
  `<store>/screens/<name>/`), never the shared `results.jsonl` (concurrent processes used to append
  to the same file, risking interleaved/corrupted lines); `experiments.screen.merge_results` reads
  every shard file (and a plain `results.jsonl` from an unsharded run, if present) back together for
  resumability checks and for reporting total progress across shards. Resuming one shard only ever
  merges/considers shard files for the SAME total shard count `N`: shard files left over from a run
  with a different `N` are ignored, never merged in.
- **Level 2** (survivors, real quality): re-run through `scenario run` on a real block (a
  `configs/scenarios/micro_*.yaml`-style spec), not through screen mode again.
- **Phase 2: reused-graph trials** (NT-092; `run: {reuse_graph: true}`, the default). GPU measurement
  (`runs/experiments/micro_loop_v1/LOG.md`, 2026-09-30) found tracing at 73% of a trial's wall time
  (12.2s of 16.7s): phase 1's fresh-model-per-trial path retraces `train_step`/`test_step` from
  scratch on every trial's first `fit()` call, even when only a continuous hyperparameter changed.
  Trials are grouped by `experiments.screen.structural_key` (every Config field EXCEPT `SEED`,
  `DATA_END`, `EPOCHS`, `SEEDED_STOCHASTIC_LAYERS` (below) and `CONTINUOUS_FIELDS` — see that
  constant's docstring in `screen.py` for the exact list and, for each, WHERE it is read from a live
  `tf.Variable`/Keras optimizer hyper at run time instead of a Python constant baked into the graph);
  trials of one group are run contiguous (`_group_order`; only reordering, never changing which trial
  produces which row) through one `_TrialGroup`, which builds the model, both optimizers and the
  compiled train/test step ONCE (from the group's first trial) and resets, before every later trial:
  the model's WEIGHTS (a throwaway `Models.build` at the trial's seed, cheap and not traced, its
  weights copied in), both optimizers' variables to zero, every RESETTABLE stochastic layer's random
  state (below) from the trial's seed, and every continuous field to the trial's value.
  `run: {reuse_graph: false}` disables grouping (every trial fresh, phase 1's path) for a direct
  comparison or to reproduce old numbers exactly; a `trainer=` override (tests only) always runs
  ungrouped, since grouping only matters for a real, traced TF graph.
  - **What makes a reused trial reproduce a fresh one (`Config.SEEDED_STOCHASTIC_LAYERS`, default
    `False`, forced `True` by screen.py itself on every trial it builds — never by the normal
    training path, so `scenario run` and the golden run are bit-for-bit unaffected).** Round 1 of
    this item reset only `training/lambdas.py`'s `tf.Variable`-backed loss weights and
    `training/custom_model.py`'s new `grad_clip_norm`/`pred_scale`/`pred_mean` Variables — real, but
    not the whole picture: Keras 2.10's `layers.Dropout` and `layers.MultiHeadAttention`'s internal
    attention dropout (both used in `models/gru_attention.py`, rate 0.1) default to
    `rng_type='legacy_stateful'`, a plain `tf.nn.dropout` backed by TF's LEGACY stateful random ops,
    which have NO Python-visible state at all — nothing to reset — and
    `models/layers/vacuum_saturation_noise.py`'s `VacuumSaturationNoise` (active whenever
    `LAMBDA_T_PERP > 0`, the default) called unseeded `tf.random.normal`, same problem. A reused
    trial's dropout/noise draws kept advancing the PREVIOUS trial's stream instead of starting from
    its own seed, so round 1 only matched a fresh run when the model happened to be noise-free
    (dropout 0, `LAMBDA_T_PERP` 0) — QA repair round 1 caught this on real data (trial 3 of a 4-trial
    group: val loss 6.5250 reused vs 6.0214 fresh). The fix: `training/reset.seeded_stochastic_layers()`
    is a context manager around every `Models.build(...)` screen mode makes; inside it, Keras's global
    `tf.keras.backend.experimental.enable_tf_random_generator()` is on, so a Dropout/MultiHeadAttention
    layer CONSTRUCTED WHILE IT IS ON (read once, at construction, never at call time — a layer built
    outside this context, i.e. every layer the normal training path builds, is untouched) gets a real,
    resettable `tf.random.Generator` instead; `VacuumSaturationNoise` takes an explicit `seeded=`
    argument (`Config.SEEDED_STOCHASTIC_LAYERS`) and, when true, draws from its own
    `tf.random.Generator` (a plain, untracked attribute — not a layer weight) instead of the unseeded
    call. `training/reset.reset_stateful_rngs` (extended) finds and resets BOTH shapes of generator
    from the trial's seed; called on both the fresh and reused paths, so they draw the identical
    stream (`tests/test_screen.py`'s `test_reused_later_trial_matches_an_independent_fresh_run_...`
    reproduces QA's exact scenario — real BTC-shaped data, calibrate on, LR/a LAMBDA/GRAD_CLIP_NORM/
    DATA_END all changed, default dropout and noise ACTIVE — bit-for-bit). A grep of `tf.random\.`,
    `Dropout`, `GaussianNoise` and `MultiHeadAttention` across `src/neural_trade/models/` found no
    other stateful random op.
  - `rules.clip_skip_epochs` (default 1) excludes the FIRST `clip_skip_epochs` epochs' logged steps
    from `clipped_share` (and the norm max/mean) only — `n_steps` still counts them — because the
    initial, pre-any-update gradient norm routinely exceeds the clip on a fresh model's very first
    steps regardless of LR, which is not the "is this config unstable" signal the rule exists for
    (the `docs/BACKLOG.md` NT-092 "why": every 2-epoch trial at LR 1e-4 failed `max_clipped_share` on
    this transient alone). The min_epochs guard refuses a spec whose EXPLICIT
    `rules.clip_skip_epochs >= EPOCHS` (nothing would ever be scored); the unset default is clamped to
    `EPOCHS - 1` instead, so an ordinary 1-epoch smoke screen is unaffected. `EPOCHS` is NOT part of
    `structural_key` (a grid/sample axis may set it per trial), so this guard, and the effective
    `clip_skip_epochs` value passed to a trial, are both computed from THAT TRIAL's own `EPOCHS`, not
    the spec-wide base (an `EPOCHS=1` trial under a 2-epoch base is refused, not silently scored on
    zero steps).
  - **Measured speed-up (CPU, this machine, one 8-trial continuous-only group, 1 epoch each):** trial
    0 (builds + traces) wall 15.97s (`train_s` 14.13s); trials 2-8 median wall 6.24s (`train_s` 5.02s,
    `build_s` 0.66s — the throwaway shadow model, not the trace — `score_s` 1.14s). About 2.1-2.6x,
    short of the ≤5s-per-trial target on total wall (QA measured a similar ratio on its machine:
    trial 0 19.11s, trials 2-8 median 9.08s). **GPU estimate (not measured — no GPU access from this
    item):** the original single-trial GPU measurement was 16.7s total with a 12.2s trace
    (`runs/experiments/micro_loop_v1/LOG.md`); subtracting that trace from a non-first trial's wall
    (16.7 - 12.2 ≈ 4.5s of load/prep/train/score) suggests trials after the first should land near or
    under the 5s target on GPU, since GPU trains and predicts faster than CPU while `build_s`'s
    throwaway-model cost stays roughly fixed; this is an estimate to confirm on a real GPU run, not a
    measured number.

### Sweeps

What exists today (one GPU job at a time; `ablate.py` and `direction_experiments.py` resume,
`gate_run.py` overwrites):

- `scripts/ablate.py`: a fixed grid over the physics terms, judged by `configs/ablation_criteria.yaml`
  (D-003; the v1 grid stays the record under those criteria).
- `scripts/direction_experiments.py`: named direction variants on the dev folds -3 / -2.
- `scripts/gate_run.py` with `scripts/check_gates.py` and `scripts/backtest_gate.py`: single named
  runs judged against the M1-M4 gates.

None of them is a search: there is no quick mode, no Optuna study and no leaderboard yet. They
belong to the frozen set (D-023): they stay runnable as history and are replaced by the experiment
engine (NT-026: scenario and sweep specs, a resumable runner, one run store with an sqlite index, one
scorer), the sweeps (NT-030: quick mode, about 5 minutes for the whole sweep, and Optuna mode;
`neural-trade sweep`) and the leaderboard
(NT-031: dev-fold net Sharpe after costs, guard-rails, test columns shown but never ranking;
`neural-trade leaderboard`). Their commands are documented here when they land. The Optuna studies
and the run index live in the repo (on D: after the move, D-030).

### Long jobs (longer than the 10-minute tool timeout)

- Use only resumable harnesses: the experiment engine (`neural-trade scenario run SPEC` resumes per
  cell; "Experiment engine" in this file) for new work; the frozen `scripts/ablate.py` and
  `scripts/direction_experiments.py` only to reproduce history (D-023).
- Launch detached so the job survives the session, with its log next to its outputs, e.g. from the
  pinned worktree: `nohup env PYTHONPATH=D:/nt/nt_exp_<name>/src $PY scripts/ablate.py ... > <out>/logs/run.log 2>&1 &`
  (Git Bash), and record in STATUS: what runs, where its log is, how to check it (`ablate.py --dry-run`
  lists pending cells), and how to resume.
- The next session checks the job first (STATUS), resumes it if it died, and never starts another
  GPU job next to it beyond what the GPU rules above allow.

### Long runs (notebook 08, NT-082)

One long engine run launched and tracked from a notebook; today the 360-day run of D-040,
`configs/scenarios/long_360d.yaml` (fold -2 on `Bitcoin_BTCUSDT.csv`, the machine-local long history;
`SHUFFLE_BUFFER: 0`, a full reshuffle of the training block every epoch). Code:
`src/neural_trade/notebook/longrun.py`.

- **Launch** (GPU rules above first): open `notebooks/08_long_run.ipynb`, set `LAUNCH = True`, run the
  launch cell once. Or from Python: `launch("configs/scenarios/long_360d.yaml", "runs", root=".")`. It
  starts `python -m neural_trade.cli scenario run <spec> --store runs` detached (Windows:
  DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP; it survives the kernel and VS Code closing) in the
  repository root, with this checkout's `src` first on PYTHONPATH and without CUDA_VISIBLE_DEVICES. It
  refuses while a cell of the scenario is running (no result.json and a progress file changed in the
  last 15 minutes, or the last launch's process alive) and once the cell is done (failed too, unless
  `retry_failed=True`). The CLI works as well: `$PY -m neural_trade.cli scenario run
  configs/scenarios/long_360d.yaml`, launched detached as in "Long jobs".
- **Where things are:** the log and the pid file of each launch:
  `runs/scenarios/<name>/launch/<UTC>.log` and `<UTC>.pid.json` (pid, command, log); the cell:
  `runs/scenarios/<name>/<run id>-<cell>/` (status.json, metrics.jsonl, training_log.csv, then
  result.json and eval_report_dev.md).
- **Watch:** re-run notebook 08's monitor cell (`show_progress(progress(SPEC, STORE, root=ROOT))`): state
  (not started / running / stalled: no update for max(3 x the median epoch, 30 min) / done / failed),
  epochs, elapsed, an ETA estimate, learning rate, best validation loss, figures, a GPU snapshot and the
  log's last 20 lines.
- **Stop:** `taskkill /PID <pid> /T /F` (the pid from the pid file or the monitor cell), or
  `longrun.stop(pid)`. The cell stays without result.json (`incomplete`); launching again trains it anew
  into a new directory.

## Notebooks

Workflow and rules: [scripts/notebooks/README.md](../scripts/notebooks/README.md). In short:

```bash
$PY scripts/notebooks/build.py [NN]        # regenerate from the generator (edit notebooks only there)
$PY scripts/notebooks/build.py --check     # drift check: committed notebooks == generator
$PY scripts/notebooks/execute.py [NN]      # execute in place; 01 trains a NEW run on the GPU (~5 min)
$PY scripts/notebooks/check.py             # must print "all clean"
$PY scripts/notebooks/render.py [NN]       # PNGs (Windows + Edge); then open and LOOK at every one
```

Notebooks 02-05 read the newest notebook/CLI run, never an engine cell: `pick_run`
(`src/neural_trade/notebook/runs.py`) takes the newest directory under `runs/` that has
`artifacts/weights.h5`, skipping `runs/scenarios/` and any run whose meta.json has an `engine`
section (NT-026). Weights are not committed (a
run's light files are: "Run directories in git" above), so in a fresh clone 02-04 fail with a clear
message until 01 has trained. The run a new execution of 01 creates is committed with the executed
notebooks (`check_run_evidence.py --list-untracked`). Notebook 06
(NT-034) launches nothing when executed: it reads the run store.

The notebooks persist and evolve (D-028): 00-05 keep their numbers and roles and are updated through
`build.py` in the same item that changes what they show; new views get new notebooks (planned: 06
control panel, NT-034; 07 discovered indicators, NT-043). Every figure follows D-014, the new views
included.

## Traps on this machine

- **Concurrent training (NT-035, 2026-09-29):** at most 3 training processes at once (runs/experiments/gpu_measurements_v1/parallel_n.json); with 4, one crashed (0xC00000FD) and the rest ran at half speed.
- **Determinism:** `TF_DETERMINISTIC_OPS=1`, set by `import neural_trade`, already enables op determinism in TF 2.10 (`enable_op_determinism()` adds nothing); same-seed GPU runs still differ (NT-074). A GPU-free check can read a median sm of about 40% from the desktop alone: judge by fb (memory).

- **CUDA on Windows.** TF 2.10 finds CUDA 11.2 / cuDNN 8.1 only through PATH. `import
  neural_trade` before `tensorflow` adds the env's DLL folders; a script that imports TensorFlow
  first trains on the CPU with only a log line. Opt out: `NEURAL_TRADE_NO_DLL_PATH=1`.
- **`CUDA_VISIBLE_DEVICES=-1`** for tests and CPU scripts (TF then logs `CUDA_ERROR_NO_DEVICE`:
  expected). Never for real training or for executing notebook 01.
- **`ptxas.exe ... CreateProcess failed`** lines are harmless. Ops that need XLA JIT on the GPU
  (e.g. int64 FloorMod) fail: pin them to `/CPU:0`.
- **Concurrent training processes.** Training is kernel-launch bound (~0.1 s/step); two TF processes
  were seen to give no extra throughput and can page to system RAM. This is not yet measured
  systematically: NT-035's throughput test decides whether N > 1 ever helps (GPU rules above). Never
  touch the owner's Docker/WSL GPU work.
- **GPU runs are not bit-reproducible** (`TF_DETERMINISTIC_OPS=1` is set, still): a same-config
  re-run moved per-horizon AUC by 0.01-0.06 (`runs/experiments/direction_v1/REPORT.md`). Compare
  settings over several seeds, never on one run. `import neural_trade` sets `TF_DETERMINISTIC_OPS=1`
  for every run (`src/neural_trade/__init__.py:26`). The opt-in deterministic mode of D-025 is
  `seed_everything(seed, deterministic=True)`, which calls `enable_op_determinism()`
  (`src/neural_trade/utils/seeding.py:32`); comparison studies use it only after NT-035's speed test.
- **Disk.** C: is nearly full (the owner's Docker WSL image, ~116 GB; never touch it). Large
  scratch and renders go to D:. Delete your own scratch when done.
- **Editable install.** `neural_trade` imports from the main checkout's `src/` (after the move, the
  D: copy's). In a worktree (including `../neural_trade_gates` and `../neural_trade_ablation`, both at
  6dec27a), put the worktree's `src` first on `PYTHONPATH` for ad-hoc scripts; pytest and
  `scripts/notebooks/*` do this themselves.
- **Stale root files.** `MODEL_PATH` / `SCALER_PATH` default to repo-root files; old local copies
  exist there (gitignored). Real runs write into `runs/<id>/` through RunContext. NT-028 removes,
  under D-029, the defaults that write new files there; the existing root files are listed for the
  owner and not deleted (models and data need the owner). A run without RunContext loads
  `MODEL_PATH` from the working directory when the file exists, and without `force` it skips
  training (`src/neural_trade/training/trainer.py:365-377`); NT-049 fixes this silent warm start.
- **Served epoch.** The trainer serves the best-validation epoch (D-011). Gate runs m1a-m6 and
  the 84 ablation runs predate this and were scored on their last epoch.
- **beta = 0.** When a horizon's delta-shrinkage beta is 0, its served delta is 0 on every bar:
  judge the price heads on the raw deltas (`raw_delta=...`); reports print n/a for served stats.
- **`compare_runs('runs/*')`** raises on an unscored run: pass `skip_unscored=True`.
- **Bash heredocs** with nested quotes break easily in this shell: write scripts with the Write
  tool and run them.
- **Another session may be working** in the same checkout. Check `git status -sb` and
  `git log -3` before editing; give parallel implementers separate worktrees.
- **Deleting anything** follows D-029: only what is both stale and without any effect, with the
  evidence. Runs, data and remote branches need the owner.
- **`LOSS_NAME=pnl_utility` (NT-087) demeans its volatility-scaled return per BATCH, not per
  training block.** This equals block demeaning only under a full reshuffle (`SHUFFLE_BUFFER: 0`,
  the P&L micro loop's E2 scenario setting). With the default `SHUFFLE_BUFFER` (2048, a partial,
  order-preserving shuffle) a training batch tracks local drift instead of the block mean, and
  validation/test batches are never shuffled at all, so `val_loss`'s demeaning basis differs from
  training's under that objective. Set `SHUFFLE_BUFFER: 0` for any run or study that uses
  `pnl_utility`.

## Move to D: (D-030; done 2026-09-28, steps 1-6; step 7 waits for the owner)

The project moved from `C:/Users/Step/Documents/neural_trade` to `D:/neural_trade` on 2026-09-28
(robocopy: 2,161 files, 754 MB, 0 failed; the two worktrees 277 and 276 files; `git worktree repair`
done; editable install re-pointed; 697 fast tests passed in 3:28 on D:; `build.py --check` and
`check.py` clean; the Claude memory copied to `~/.claude/projects/d--neural-trade/memory/`). The conda
env stays where it is (`$PY` does not change). The steps, for the record and for a future move:

1. **Quiet.** No job writes into `runs/` (STATUS, GPU check above); `git status -sb` shows a clean
   tree (a run's light files are tracked, "Run directories in git"); the branch is pushed.
2. **Copy, do not clone.** The gitignored run files (weights, scalers, experiment logs) and the
   gitignored data (`Bitcoin_BTCUSDT.csv`) must come along. In PowerShell:
   `robocopy C:\Users\Step\Documents\neural_trade D:\neural_trade /E /COPY:DAT /R:1 /W:1`, and the
   same for `neural_trade_gates` and `neural_trade_ablation` to `D:\neural_trade_gates` and
   `D:\neural_trade_ablation` (robocopy exit codes below 8 mean success). Compare file counts and
   total size of source and copy.
3. **Repair the worktree links.** From `D:/neural_trade`:
   `git worktree repair D:/neural_trade_gates D:/neural_trade_ablation`, then `git worktree list`
   shows the three D: paths (the main checkout on `remediation/plan`, both worktrees at 6dec27a).
4. **Re-point the editable install** (owner-approved env change): from `D:/neural_trade`,
   `$PY -m pip install --no-deps -e .`; then
   `CUDA_VISIBLE_DEVICES=-1 $PY -c "import neural_trade; print(neural_trade.__file__)"` prints a
   `D:/neural_trade/src` path.
5. **Carry the Claude memory.** Copy `C:/Users/Step/.claude/projects/c--Users-Step-Documents-neural-trade/memory/`
   into the project folder Claude Code uses for `D:/neural_trade` (the folder name is the path with
   `:`, `/`, `\` and `_` replaced by `-`, e.g. `d--neural-trade`; the drive letter's case follows how
   the folder is opened, so check which folder the first session at D: creates). Update memory
   entries that name the C: path.
6. **Verify on D:.** `git status -sb` and `git log --oneline -3` match the C: copy; the fast suite
   passes; `$PY -m ruff check src tests scripts` is clean; `$PY scripts/notebooks/build.py --check`
   and `$PY scripts/notebooks/check.py` pass. Record the result lines in STATUS.
7. **Owner confirms** (open as of 2026-09-28). The owner reopens VS Code at `D:/neural_trade`. After
   the owner confirms the D: copy works, the lead deletes the C: copy (repo and both worktrees) only
   on the owner's explicit go-ahead. NT-009 closes then. Until then, never work in the C: copy
   (CLAUDE.md start step 2).

After the move, the header of this file and the machine-local memory name `D:/neural_trade`; the
relative worktree paths (`../neural_trade_gates`) and `$PY` keep working unchanged. Scratch, QA and
experiment worktrees stay where they are (`D:/nt_qa/`, `D:/nt/nt_exp_<name>`). The C: disk trap stays:
the Docker image is still there.
