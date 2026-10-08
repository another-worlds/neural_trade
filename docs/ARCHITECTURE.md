# Architecture

A map of `src/neural_trade/`: what each subpackage is for, which way imports are allowed to point, how
components are registered, and where the experiment engine keeps its runs. It describes the code as it is.
Why the project exists is in [VISION](VISION.md); the settled choices are in [DECISIONS](DECISIONS.md); how to
run things is in [RUNBOOK](RUNBOOK.md).

## Module map

One line per subpackage (`tests/test_docs_guides.py` checks that every subpackage is listed here).
`cli.py` is the `neural-trade` command and `__init__.py` puts the CUDA DLLs on `PATH` (import
`neural_trade` before `tensorflow`).

| subpackage | what it holds |
|---|---|
| `core/` | the typed flat `Config` (YAML round trip, metadata per field, validation), `BaseRegistry`, the dataset spec (instrument, bar size, wall-clock lengths), cost profile, the configuration guard, exceptions and logging |
| `utils/` | small shared helpers: EWMA maths, seeding (including deterministic mode), atomic file writes with retry, environment fingerprint |
| `indicators/` | the learnable indicator families and their `Indicators` registry entries: inputs, learnable parameters with textbook defaults and bounds, output channels, drawing spec |
| `metrics/` | numpy and graph-safe TensorFlow metrics, the one direction-label rule (deadband), effective-sample statistics (`n_eff`, intervals, bootstraps) |
| `losses/` | the loss terms of the training objective (direction, CRPS, point, and the six physics terms) and their registry |
| `data/` | loaders, preprocessors, windowing, the purged four-way split and folds, gap handling, scaling fitted on the training block, `DataProcessor` |
| `models/` | the architectures (`gru_attention`, `gru_small`, `linear_indicators`), the three horizon heads, and the custom layers (learnable indicators, positional encoding, noise, energy gate) |
| `telemetry/` | append-only per-epoch JSONL logging of a run |
| `calibration/` | temperature scaling for P(up), conformal intervals scaled by realised volatility, delta shrinkage, the calibration pipeline and the "no usable direction signal" state |
| `evaluation/` | `PredictionFrame`, the evaluation report with its baselines, walk-forward, applied-period and permutation-importance read-outs |
| `strategy/` | `SignalFrame`, the strategies (including the technical-analysis baselines), the backtest engine (next-open fills, costs, stops on high/low, random null), performance statistics |
| `training/` | the Keras model wrapper with the custom train step, loss weights and their calibration, optimizers, callbacks, the stability guard, the trainer and artifact bundles |
| `serving/` | `Predictor`: raw OHLCV bars in, forecasts out, from a saved artifact bundle; the post-processing and the HTML indicator report |
| `visualization/` | every figure: training, analytics per head, calibration, trading, indicators, comparison, leaderboard; one theme (`theme.py`) |
| `experiments/` | the experiment engine: scenario spec, resumable runner, run store and sqlite index, scorer, sweeps, leaderboard, paired comparator, screen mode, stability harness; and the frozen ablation harness |
| `registries/` | the ten component registries, the plugin loader entry point (`load_all`), the registrations of every built-in component |
| `notebook/` | the notebook front-ends: widgets and logic (training session, backtest and calibration explorers, control panel, long-run monitor) so the notebooks stay thin |

## Layering

The import graph is built from the source with the standard library `ast` module, so an import written
inside a function counts like one at the top of a file, and a dynamic import (`importlib.import_module`,
used for plugin discovery) does not count.

**Enforced** (`tests/test_layering.py`, runs in the fast suite):

1. no two subpackages import each other, directly, anywhere in a file;
2. `evaluation/` does not import `training/` or `experiments/`;
3. `data/` does not import `visualization/`.

**Not enforced.** Longer cycles are not forbidden by the test. At the time of writing seven subpackages
(`registries`, `experiments`, `evaluation`, `serving`, `strategy`, `training`, `visualization`) sit on at
least one cycle of three or more packages (for example `registries` to `visualization` to `evaluation` to
`registries`). Treat the layers below as the intended direction and the cycle list as known debt.

Intended direction, from the bottom (a package may import from the ones above it in this list, never
from the ones below):

```
core, utils                                   foundations: config, registry base, helpers
metrics, indicators, losses, telemetry        pure maths and components, no pipeline knowledge
data, models, calibration                     the pipeline's parts
evaluation                                    scoring predictions
strategy                                      trading on predictions
training, serving, visualization              orchestration and presentation
registries                                    wires the built-in components to their registry
experiments                                   the engine: runs, scores and compares trained cells
notebook                                      front-ends for the notebooks (top)
```

Observed imports on the merged code, for orientation (generated from the same graph the test reads):

| package | imports |
|---|---|
| `core`, `utils` | nothing inside the package |
| `telemetry` | core |
| `data` | core |
| `metrics` | core, utils |
| `indicators` | core, utils |
| `losses` | core, metrics, utils |
| `models` | core, indicators, utils |
| `calibration` | metrics |
| `strategy` | core, evaluation |
| `evaluation` | data, indicators, metrics, models, registries |
| `registries` | core, data, indicators, losses, metrics, models, visualization |
| `training` | calibration, core, data, evaluation, losses, metrics, registries, strategy, telemetry, utils, visualization |
| `serving` | core, data, evaluation, metrics, registries, training, utils, visualization |
| `visualization` | calibration, core, data, evaluation, experiments, indicators, metrics, strategy, telemetry, utils |
| `experiments` | core, data, evaluation, losses, metrics, registries, serving, strategy, telemetry, training, utils |
| `notebook` | calibration, core, data, evaluation, experiments, metrics, registries, serving, strategy, telemetry, training, visualization |

Some of these edges cut against the intended direction (for example `evaluation` to `registries`, or
`visualization` to `experiments`); they are what makes the cycles. The frozen ablation harness
(`experiments/ablation.py`, D-023) is history and is not part of the layering being fixed.

## Registries

A registry maps a name to a component plus metadata. `core/registry.py` holds `BaseRegistry`; each registry
is a subclass, strict (a bad component or a duplicate name raises), with its own dictionary. A component
registers itself with a decorator and the config selects it by name; the pipeline never edits to add one.

| registry | selects | default | module |
|---|---|---|---|
| `Models` | the architecture | `gru_attention` | `registries/models.py` |
| `Optimizers` | the optimizer | `adam` | `registries/optimizers.py` |
| `Metrics` | metrics | `rmse` | `registries/metrics.py` |
| `Callbacks` | training callbacks | `early_stopping` | `registries/callbacks.py` |
| `DataLoaders` | the file reader (`csv`, `parquet`, `dataframe`) | `csv` | `registries/data_loaders.py` |
| `Visualizations` | figures | `plotly_interactive` | `registries/visualizations.py` |
| `Layers` | model layers | `learnable_indicators` | `registries/layers.py` |
| `Preprocessors` | data-frame steps | `standardize_ohlcv` | `registries/preprocessors.py` |
| `Losses` | loss terms | `custom_loss` | `registries/losses.py` |
| `Indicators` | indicator families | none (all families are listed in the config) | `registries/indicators.py` |
| `Strategies` | trading strategies (a separate registry, defined in `strategy/strategies.py`, outside the ten) | `calibrated_quantile` | |

`registries.load_all(config)` imports every registry module, loads plugins from `Config.PLUGINS_DIR` (unset
by default; `load_all(config, plugins_dir="plugins")` or `registry list --plugins plugins` load the repository's
`plugins/`, which holds templates and a tested example) and checks that every component the config names
exists; call it once at an entry point. `neural-trade registry list` prints everything registered;
`registry info <Registry> <name>` shows one entry. To add a component: register it in a module under
`plugins/` (or in the registry's own module), and name it in the config; for an indicator, one registry entry
and a config line, with no edit to the model or the training pipeline.

## The experiment engine and the run store

`experiments/` is the one path for experiments (D-023); the older scripts are a frozen set kept runnable as
history.

- `scenario.py`: the YAML scenario and sweep spec; a scenario expands to **cells** (variant, fold, seed).
- `runner.py`: trains and scores every pending cell, resumable; it never writes into an existing run
  directory and deletes nothing. A cell's run id carries the scenario hash (D-065).
- `store.py`: the **run store**. Each cell is a directory under `runs/scenarios/<scenario>/` with the usual
  run files (`config.yaml`, `metrics.jsonl`, `meta.json`, `status.json`, `result.json`, the evaluation
  reports and the stored predictions), plus an sqlite index `runs/index.sqlite`. The directories are the
  record and the index is a cache (`scenario reindex` rebuilds it).
- `scorer.py`: the one scorer: the evaluation report of a fold's out-of-sample block plus the backtest of the
  scenario's strategy at its cost profile.
- `sweep.py`, `claims.py`: quick and Optuna sweeps; `leaderboard.py`: the ranking on dev folds;
  `comparator.py`: the pre-registered paired verdict; `screen.py`: short mass trials; `stability.py`: the
  stability harness; `rescore.py`: strategy studies on stored predictions without retraining.

Heavy files in run directories (weights, stored predictions) are ignored by git; their light files are
tracked (RUNBOOK "Run directories in git").

## Data flow of one run

```
CSV -> data/ (preprocess, window, purged split) -> models/ + indicators/ + losses/ (training/ fits it)
    -> calibration/ (fitted on the calibration block) -> evaluation/ (frame, report, baselines)
    -> strategy/ (signals, backtest) -> experiments/ (score, store, rank, compare)
    -> visualization/ and notebook/ (the figures and the widgets that show them)
```

`serving/` loads a saved artifact bundle (weights, scalers, the calibration pipeline, the config) and runs
the same path forward on new bars.
