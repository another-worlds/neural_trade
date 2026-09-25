# Runbook

How to run everything, and the traps of this machine. Commands were verified on 2026-09-25 unless
marked otherwise. Run them from the repository root.

```bash
PY=C:/Users/Step/miniforge3/envs/nt/python     # conda env nt: Python 3.10, TF 2.10 GPU. Do not use `conda run` (plugin crash).
export PYTHONIOENCODING=utf-8                   # the console code page is cp1251
```

## Environment

| Task | Command |
|---|---|
| Rebuild the env from nothing (Windows, GPU; not re-verified) | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`, then `conda activate nt`, `pip install -e ".[viz,dev]"` |
| Linux / CPU only (what CI does) | `pip install -r requirements-ci.txt && pip install -e ".[viz,dev]"` |
| Environment fingerprint | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli env --no-devices` |

Changing the env (install / upgrade packages) needs the owner.

## Tests, lint, coverage, docs

| Task | Command | Time |
|---|---|---|
| Fast suite | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow"` | 5-6 min |
| Slow suite (training, CLI and predictor round trips, reproducibility, notebook execution on synthetic bars) | `CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m slow` | ~4 min |
| Lint | `$PY -m ruff check src tests scripts` (CI lints `src scripts` with ruff 0.6.9; locally 0.16.8) | seconds |
| Coverage and per-package gates | `COVERAGE_FILE=<scratch>/.coverage CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow" --cov=neural_trade --cov-report=json:<scratch>/coverage.json` then `$PY scripts/check_coverage.py <scratch>/coverage.json` | ~6 min |
| Regenerate TESTING_DOCUMENTATION.md | `CUDA_VISIBLE_DEVICES=-1 $PY scripts/test_inventory.py` | seconds |

**CI.** GitHub Actions `ci` (lint + unit on Linux CPU, `requirements-ci.txt`). Green locally does
not mean green on CI: its pins differ from the local env (see NT-001). After every push, check the
branch head (the GitHub CLI is not installed; job logs need auth, the run list does not):

```bash
curl -s "https://api.github.com/repos/another-worlds/neural_trade/actions/runs?branch=remediation/plan&per_page=5" \
  | $PY -c "import json,sys; [print(r['head_sha'][:7], r['name'], r['status'], r['conclusion']) for r in json.load(sys.stdin)['workflow_runs']]"
```

The nightly workflow (`.github/workflows/nightly.yml`) never runs: GitHub registers scheduled
workflows only from the default branch, and `master` has no `.github/` yet (NT-008).

## The CLI

| Task | Command |
|---|---|
| Overview | `$PY -m neural_trade.cli --help` (or `neural-trade --help`) |
| Train, evaluate on test, save a serving bundle (GPU; not re-verified) | `$PY -m neural_trade.cli train --epochs 20` or `--config configs/default.yaml --set LR=5e-4 --set EPOCHS=10` |
| Forecast with a saved bundle | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli predict --artifacts runs/<id>/artifacts --csv binance_btcusdt_1min_ccxt.csv --last` |
| Backtest with a saved bundle (in-sample, see below) | `CUDA_VISIBLE_DEVICES=-1 $PY -m neural_trade.cli backtest --artifacts runs/<id>/artifacts --csv binance_btcusdt_1min_ccxt.csv --out <dir> [--strategy NAME] [--plot]` |
| Registry contents | `$PY -m neural_trade.cli registry list`, `registry info Optimizers adamw`, `registry search QUERY` |

**`neural-trade backtest` scores every window of the CSV, the training blocks included: its
numbers are in-sample.** Out-of-sample backtests: notebook `02_backtest` (the test block) or
`scripts/backtest_gate.py`.

## Experiments and gates (GPU: one job at a time)

| Task | Command |
|---|---|
| Physics-term ablation: plan / pending cells | `$PY scripts/ablate.py --scale full --out runs/ablations/<name> --dry-run` |
| Run the ablation (resumable) | `$PY scripts/ablate.py --scale full --out runs/ablations/<name>` (7-10 GPU hours: needs the owner) |
| Judge the M1-M4 gates clause by clause | `CUDA_VISIBLE_DEVICES=-1 $PY scripts/check_gates.py runs/gates` (exits 1 while M3 fails: by design) |
| Produce a gate run, then its backtest | `$PY scripts/gate_run.py --name m7 --epochs 20`, then `$PY scripts/backtest_gate.py runs/gates/m7` |
| Direction experiments on the dev folds -3 / -2 | `$PY scripts/direction_experiments.py --out runs/experiments/<name> [--only NAME ...]` |
| Refactor guard (small deterministic run) | `$PY scripts/golden_run.py record OUT.npz`, later `$PY scripts/golden_run.py verify OUT.npz` |
| Compare scored runs | `$PY -c "from neural_trade.experiments.compare import compare_runs; print(compare_runs('runs/*', metrics=['h1/direction/auc'], skip_unscored=True))"` |

Before any GPU job: `nvidia-smi` must show the GPU idle (the owner's other project runs in
Docker/WSL and shows as pid 0 or unexplained memory use: then wait), and `df -h /c /d` must show
room. Experiments follow `.claude/agents/experimenter.md`: pre-registered SPEC.md, verdict on the
test block once, report with run ids.

## Notebooks

Workflow and rules: [scripts/notebooks/README.md](../scripts/notebooks/README.md). In short:

```bash
$PY scripts/notebooks/build.py [NN]        # regenerate from the generator (edit notebooks only there)
$PY scripts/notebooks/build.py --check     # drift check: committed notebooks == generator
$PY scripts/notebooks/execute.py [NN]      # execute in place; 01 trains a NEW run on the GPU (~5 min)
$PY scripts/notebooks/check.py             # must print "all clean"
$PY scripts/notebooks/render.py [NN]       # PNGs (Windows + Edge); then open and LOOK at every one
```

Notebooks 02-05 read the newest run under `runs/`; run directories are not committed, so in a
fresh clone 02-04 fail with a clear message until 01 has trained.

## Traps on this machine

- **CUDA on Windows.** TF 2.10 finds CUDA 11.2 / cuDNN 8.1 only through PATH. `import
  neural_trade` before `tensorflow` adds the env's DLL folders; a script that imports TensorFlow
  first trains on the CPU with only a log line. Opt out: `NEURAL_TRADE_NO_DLL_PATH=1`.
- **`CUDA_VISIBLE_DEVICES=-1`** for tests and CPU scripts (TF then logs `CUDA_ERROR_NO_DEVICE`:
  expected). Never for real training or for executing notebook 01.
- **`ptxas.exe ... CreateProcess failed`** lines are harmless. Ops that need XLA JIT on the GPU
  (e.g. int64 FloorMod) fail: pin them to `/CPU:0`.
- **One GPU job at a time.** Training is kernel-launch bound (~0.1 s/step); two TF processes give
  no extra throughput and can page to system RAM. Never touch the owner's Docker/WSL GPU work.
- **GPU runs are not bit-reproducible** (`TF_DETERMINISTIC_OPS=1` is set, still): same seed and
  config, h1 AUC 0.478-0.500 over four runs. Compare over seeds.
- **Disk.** C: is nearly full (the owner's Docker WSL image, ~116 GB; never touch it). Large
  scratch and renders go to D:. Delete your own scratch when done.
- **Editable install.** `neural_trade` imports from the main checkout's `src/`. In a worktree
  (including `../neural_trade_gates` and `../neural_trade_ablation`, both at 6dec27a), put the
  worktree's `src` first on `PYTHONPATH` for ad-hoc scripts; pytest and `scripts/notebooks/*` do
  this themselves.
- **Stale root files.** `MODEL_PATH` / `SCALER_PATH` default to repo-root files; old local copies
  exist there (gitignored). Real runs write into `runs/<id>/` through RunContext.
- **Served epoch.** The trainer serves the best-validation epoch (D-011). Gate runs m1a-m6 and
  the 84 ablation runs predate this and were scored on their last epoch.
- **beta = 0.** When a horizon's delta-shrinkage beta is 0, its served delta is 0 on every bar:
  judge the price heads on the raw deltas (`raw_delta=...`); reports print n/a for served stats.
- **`compare_runs('runs/*')`** raises on an unscored run: pass `skip_unscored=True`.
- **Bash heredocs** with nested quotes break easily in this shell: write scripts with the Write
  tool and run them.
- **Another session may be working** in the same checkout. Check `git status -sb` and
  `git log -3` before editing; give parallel implementers separate worktrees.
