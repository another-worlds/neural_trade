# Experiments: scenarios, sweeps, the panel, the leaderboard and verdicts

How to run a comparison and read its answer. One engine does all of it (D-023): a **scenario** says what
to train, a **sweep** searches over it, the **leaderboard** ranks the result, the **comparator** gives an
"A beats B" verdict, and notebook `06_control_panel` is the same code behind widgets. Terms such as dev
fold, test fold and net Sharpe are in [Concepts](concepts.md). The full command reference is
[docs/RUNBOOK.md](../RUNBOOK.md) ("Experiment engine", "Sweeps (NT-030)", "Stability harness").

On this machine run commands with the `nt` environment's Python, from the repository root:
`C:/Users/Step/miniforge3/envs/nt/python -m neural_trade.cli <command>` is the same as
`neural-trade <command>` below. Commands that train use the GPU unless `CUDA_VISIBLE_DEVICES=-1`.

## Scenarios

A scenario is a YAML file (`configs/scenarios/*.yaml`, schema in
`src/neural_trade/experiments/scenario.py`): a base config, overrides, named **variants** (each a set of
Config overrides), an optional grid, the **folds** and **seeds** to run, the strategy and its backtest
settings. Every (variant, fold, seed) triple is a **cell**, trained and scored into its own directory
`runs/scenarios/<name>/<run id>-.../`. Each cell is scored on its fold's out-of-sample block; a fold's
role is *dev* (an earlier fold that the ranking may use) or *test* (fold -1).

- `neural-trade scenario plan SPEC` validates the spec and lists every cell with its state. It trains
  nothing.
- `neural-trade scenario run SPEC` trains and scores every pending cell. It is **resumable**: finished
  cells are skipped, `--max-cells N` stops after N, `--retry-failed` trains failed cells again.
- Everything lands in one **run store** (`--store`, default `runs/`): the run directories and an sqlite
  index (`runs/index.sqlite`, rebuilt from the directories by `scenario reindex`). The run directories
  are the record; the index is a cache.

`configs/scenarios/reference.yaml` is the reference setup: the default model on folds -3, -2 and -1 with
seeds 0-2 (9 cells; about 5 GPU minutes each is an estimate). The baselines of the yardstick are scenarios
too: `nt033_learned`, `nt033_frozen_twin` (periods frozen at the textbook values) and
`nt033_ta_ma_cross`, `nt033_ta_rsi`, `nt033_ta_bollinger` (technical-analysis rules with no network and no
GPU). The same search budget and the same dev folds make them comparable.

## Sweeps and their GPU budget

A sweep searches Config fields on the **dev folds only**, one seed per trial, and ranks trials by the
dev-fold mean net Sharpe. The search space is the scenario's `search:` block; only fields the Config
metadata marks *tunable* can be searched ([config reference](config-reference.md)). Without a block the
space is LR, BATCH_SIZE, LAMBDA_DIR and LAMBDA_CRPS. The test fold is never run for a trial and never ranks.

| mode | what it does | size |
|---|---|---|
| **quick** | trials, epochs (at most 3) and dev folds are sized so that the printed estimate is at most `--quick-minutes` (5); every result is labelled `quick`: reduced epochs, one seed, **a leader, not a winner** | about 5 minutes |
| **optuna** | a resumable TPE study in sqlite; after the search the top `--top-k` (5) trials are re-run with several seeds (3) and the winner is the best dev-fold seed mean | measured, may run overnight |

The budget is **stated before anything trains** (D-023, D-024):

- Both modes need a **measured `sec_per_step` for exactly this setup**: the latest finished run of the same
  dataset, batch size, input layout, window, horizons, bar size and model in the index, or an explicit
  `--sec-per-step`. With neither the sweep refuses; it never borrows another setup's speed.
- `--dry-run` prints the estimate (quick) or the GPU budget (optuna: an upper bound and an expected value,
  trials times dev folds times steps times `sec_per_step`, plus the re-run) and stops.
- **Optuna refuses above `--max-hours`, default 12 (one night)**. A larger budget goes to the owner first
  (OPERATING_MODEL "Sweeps and pre-registered studies"). Only run a sweep while the GPU is free of other
  jobs (RUNBOOK "GPU rules"); `--parallel N` runs N trials at once only up to a recorded, measured limit.
- The optuna mode needs the `sweep` extra: `pip install -e ".[sweep]"`.

A sweep **picks a winner; it is not a verdict that A beats B** (below).

## The control panel (notebook 06)

`notebooks/06_control_panel.ipynb` is the same engine behind widgets: choose a scenario, a search space and
a mode, press **Estimate** to see what `sweep --dry-run` prints, then **Launch** (or **Resume**). A budget
above the `ControlPanel(confirm_gpu_hours=1.0)` argument also needs **Confirm budget**. A refusal of the CLI (budget over the cap, a bar
size the sweep does not support, an existing sweep without resume) appears as the same refusal with nothing
started. **Executing the notebook top to bottom starts nothing**: only the Launch button does. The board
refreshes while a sweep runs, and a second table compares the rows you select. The command line does the
same for unattended runs.

## The leaderboard

`neural-trade leaderboard SCENARIO [SCENARIO ...] [--out DIR]` (or the panel's board) prints one row per
configuration. Several scenarios put their configurations on one board, which is how the learned model, the
frozen twin and the technical-analysis rules are compared.

- **The ranking column is the dev-fold net Sharpe after costs:** the mean over dev folds of each fold's seed
  mean. Its spread is the sd *between fold means* (the fold is the unit of inference, D-046), shown beside
  the seed sd.
- **The test-fold columns are labelled "test, not used for ranking"**. They are shown for every row and
  never decide the order or the winner (D-020).
- **Guard-rails** can disqualify a row from winning: trades (at least 1 on the mean and on every dev fold, so a
  row that never trades cannot win), beating buy-and-hold, beating the random null at the same trade
  frequency (the dev percentile must reach 50 by default), a scored cell on every dev fold, a maximum
  drawdown if you set one (`--max-drawdown`), and the **cost profile**. A row whose Sharpe was stored with
  another cost profile is marked *not comparable* and cannot win; it is never silently re-ranked.
- Defaults come from the spec's `leaderboard:` block; flags override it.

An empty store prints an empty board, not an error.

## Verdicts: A beats B

"Learned beats frozen", "a loss term on beats off" and any two-scenario question are answered only by the
**paired comparator** (D-025, D-046), with a spec written **before** the runs:

```yaml
# configs/compares/example.yaml (abridged)
scenario_a: reference_learned
scenario_b: reference_frozen
metric: h1/direction/auc
min_effect: 0.01                       # the smallest difference that matters, fixed in advance
judgment_folds: [-5, -4, -3, -2, -1]   # folds no earlier choice used
min_folds: 5                           # never below 5 (D-046)
registered_utc: "2026-09-30T00:00:00Z" # before the compared runs started
guard_rails: [{metric: h1/variance/crpss, direction: higher_better, max_degradation: 0.01}]
```

`neural-trade compare SPEC --out DIR` pairs the two scenarios' cells by (seed, fold), averages each
fold's seeds first, and runs the paired test over the judgement folds. It writes `result.json` and
`report.md`. **A beats B only when the whole confidence interval of the mean paired difference lies at or
beyond `min_effect`**; anything else, including a significant difference smaller than `min_effect`, is
*inconclusive*. It **excludes** pairs whose fold is not named in the spec or whose two sides were trained on different data
or setups, and **refuses** the whole comparison when fewer than 5 judgement folds (`min_folds`) have a
usable pair, when a compared run started
before the spec's `registered_utc` (the spec must predate the runs), or the spec's content changed after its
first comparison. Identical GPU runs differ by 0.01-0.05 AUC, which is why one run, or one fold with many
seeds, never decides. The panel shows a stored verdict as it was written and never runs one; without a pre-registered
spec it shows per-fold differences labelled *exploratory*.

## Worked example: the reference setup

The commands below are the whole path, in order. A test (`tests/test_docs_guides.py`) runs them on a
tiny CPU scenario with the same name and keys (synthetic-sized bars, one epoch, two folds, one seed) instead
of the full reference scenario, with the `--store` redirected to a temporary directory. The `compare` line
is not run by the test: it needs five judgement folds of two finished scenarios, which no tiny scenario
has; the test only checks that the example spec parses and that the command's arguments are accepted.

```bash worked-example
# 1. check the scenario and list its cells (no training; works on the CPU)
neural-trade scenario plan configs/scenarios/reference.yaml
# 2. train and score every cell (GPU; resumable, so a stop is safe)
neural-trade scenario run configs/scenarios/reference.yaml
# 3. rank the configurations on the dev folds; test columns are shown, never ranked
neural-trade leaderboard reference_default
# 4. size a quick sweep before running it (needs a measured sec_per_step or an explicit one)
neural-trade sweep configs/scenarios/reference.yaml --mode quick --dry-run --sec-per-step 0.17
# 5. state the GPU budget of an Optuna sweep before it starts
neural-trade sweep configs/scenarios/reference.yaml --mode optuna --n-trials 10 --dry-run --sec-per-step 0.17
# 6. a pre-registered verdict (edit the placeholder scenario names first)
neural-trade compare configs/compares/example.yaml --out runs/compares/example
```

What to expect, measured on this repository's reference spec (the numbers in step 4 and 5 are estimates for
an assumed 0.17 s per step; use your own measured value):

- Step 1 lists 9 cells, all `pending`, three dev-fold cells for each of folds -3 and -2 and three test-fold
  cells for fold -1 (seeds 0, 1, 2).
- Step 4 sizes 4 trials on dev fold -2 with one epoch each, about 4 minutes of the 5 allowed.
- Step 5 with 10 trials prints an upper bound of about 9.5 GPU-hours and about 4.3 expected, under the
  12-hour cap. With `--n-trials 30` the same command **refuses** (upper bound about 15.6 hours) and starts
  nothing.
- Step 3 on a store with no scored cell prints an empty board.

After step 2 open `notebooks/06_control_panel.ipynb` (or `05_compare_runs`) to see the leaderboard and the
per-metric comparison. Read a row's Sharpe next to its spread and its guard-rails; a rank-1 row with a
spread larger than the gap to rank 2 is not a clear winner.

## Cheaper tools before a long run

- `neural-trade screen SPEC` (`configs/screens/`): many sub-30-second trials on a 6-hour block to find
  broken mathematics and unstable regions. It ranks nothing.
- `neural-trade stability` is the stability harness ([Your own data](own-data.md) "Stability").
- `neural-trade scenario rescore SPEC --study STUDY` re-scores stored predictions with another strategy on
  the CPU, without retraining.

Tests and short checks run on the CPU with `CUDA_VISIBLE_DEVICES=-1`; do not use it for real training.
