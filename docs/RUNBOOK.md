# Runbook

How to run everything, and the traps of this machine. Commands were verified on 2026-09-25 unless
marked otherwise. Run them from the repository root. The repository is at `D:/neural_trade` since
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

**Installing optuna** (the lead, once, as part of NT-030's integration; not yet run):

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
window and horizons to wall-clock time and NT-042 allows any number of horizons (D-022). Until NT-040
is fixed, backtest Sharpe and Sortino are right only for one-minute bars (and assume a 24/7 market),
so `RESAMPLE_MINUTES` is not tunable and a sweep refuses any other bar size (NT-029, NT-030).

## Tests, lint, coverage, docs

| Task | Command | Time |
|---|---|---|
| Fast suite | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow"` | 5-6 min idle; 14 min seen while the machine was busy (2026-09-25): run it in the background |
| Slow suite (training, CLI and predictor round trips, reproducibility, notebook execution on synthetic bars) | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m slow` | ~4 min |
| Stability invariants (from NT-036; the marker does not exist yet) | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m stability` | not measured |
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
pinned worktree (`git worktree add --detach D:/nt_exp_<name> <sha>`, `PYTHONPATH` set to its
`src`), a verdict judged once, a REPORT with run ids. A verdict's pairs are (seed, fold) over judgement folds that no
choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more
held-out folds from the long history once NT-041 exists); at least 5 pairs. Never reuse a
`scripts/gate_run.py` `--name`: it deletes an existing run directory (`scripts/gate_run.py:157-158`).
NT-024, which would have made it refuse, was dropped for NT-026; the script is in the frozen set
(D-023), so this rule stays.

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

- **Default: one GPU job at a time.** The lead's notebook routine (01 trains about 5 minutes) is one
  GPU job like any other.
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

- Use only resumable harnesses: today `scripts/ablate.py` (resumes per cell) and
  `scripts/direction_experiments.py` (skips experiments that have a `result.json`); once NT-026
  exists, the engine's runner.
- Launch detached so the job survives the session, with its log next to its outputs, e.g. from the
  pinned worktree: `nohup env PYTHONPATH=D:/nt_exp_<name>/src $PY scripts/ablate.py ... > <out>/logs/run.log 2>&1 &`
  (Git Bash), and record in STATUS: what runs, where its log is, how to check it (`ablate.py --dry-run`
  lists pending cells), and how to resume.
- The next session checks the job first (STATUS), resumes it if it died, and never starts another
  GPU job next to it beyond what the GPU rules above allow.

## Notebooks

Workflow and rules: [scripts/notebooks/README.md](../scripts/notebooks/README.md). In short:

```bash
$PY scripts/notebooks/build.py [NN]        # regenerate from the generator (edit notebooks only there)
$PY scripts/notebooks/build.py --check     # drift check: committed notebooks == generator
$PY scripts/notebooks/execute.py [NN]      # execute in place; 01 trains a NEW run on the GPU (~5 min)
$PY scripts/notebooks/check.py             # must print "all clean"
$PY scripts/notebooks/render.py [NN]       # PNGs (Windows + Edge); then open and LOOK at every one
```

Notebooks 02-05 read the newest notebook/CLI run, never an engine cell. Today `pick_run`
(`src/neural_trade/notebook/runs.py:19`) takes the newest directory anywhere under `runs/` that has
`artifacts/weights.h5`; NT-026 keeps engine runs out of that default. Run directories are not
committed, so in a fresh clone 02-04 fail with a clear message until 01 has trained. Notebook 06
(NT-034) launches nothing when executed: it reads the run store.

The notebooks persist and evolve (D-028): 00-05 keep their numbers and roles and are updated through
`build.py` in the same item that changes what they show; new views get new notebooks (planned: 06
control panel, NT-034; 07 discovered indicators, NT-043). Every figure follows D-014, the new views
included.

## Traps on this machine

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

## Move to D: (D-030; done 2026-09-28, steps 1-6; step 7 waits for the owner)

The project moved from `C:/Users/Step/Documents/neural_trade` to `D:/neural_trade` on 2026-09-28
(robocopy: 2,161 files, 754 MB, 0 failed; the two worktrees 277 and 276 files; `git worktree repair`
done; editable install re-pointed; 697 fast tests passed in 3:28 on D:; `build.py --check` and
`check.py` clean; the Claude memory copied to `~/.claude/projects/d--neural-trade/memory/`). The conda
env stays where it is (`$PY` does not change). The steps, for the record and for a future move:

1. **Quiet.** No job writes into `runs/` (STATUS, GPU check above); `git status -sb` shows only the
   expected untracked run directories; the branch is pushed.
2. **Copy, do not clone.** The untracked `runs/` directories and the gitignored data
   (`Bitcoin_BTCUSDT.csv`) must come along. In PowerShell:
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
experiment worktrees stay where they are (`D:/nt_qa/`, `D:/nt_exp_<name>`). The C: disk trap stays:
the Docker image is still there.
