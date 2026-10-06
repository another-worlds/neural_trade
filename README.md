# neural-trade

A neural network that predicts financial time series from technical indicators whose parameters
and combinations are learned by gradient descent. It is meant as a substitute for manual indicator
search: instead of a person trying RSI 14 against RSI 21, the network learns the indicator periods
that predict best, shows what it learned, and is judged by net financial metrics after trading
costs. The design goal is that the instrument, bar size, window and horizons are configuration;
today the code is built and tested on one reference setup.

**Reference setup: BTC/USDT one-minute bars.** From the last 60 bars (open, high, low, close,
volume; `Config.INPUT_SERIES`, `["close"]` reproduces the pre-NT-047 close-only input), one
network predicts, for 10, 15 and 20 minutes ahead:

- the price change in quote currency (`delta`; USDT here),
- the probability that the price goes up (`direction`),
- the variance of the price change (`sigma`),

and learns a set of technical indicators, their periods trained by gradient descent: EMA, MACD,
RSI and Bollinger on the close, and - since NT-047 - ATR, Stochastic, Williams %R, Keltner, OBV,
VWAP, MFI, ADX/DMI, CCI and Donchian on the full OHLCV bars (14 families, 3 instances each, 54
learned periods, 82 indicator channels; rolling max/min and sign branches use smooth
differentiable forms). Six "physics-inspired" regularisers act on these heads. An ablation harness tests whether
each one earns its place.

The package lives in `src/neural_trade/`. Around the model it provides a typed config, nine
component registries, a purged evaluation protocol with baselines, post-hoc calibration, a
serving API, an honest backtest engine, run tracking and a CLI.

> **Status.** What works, what does not, and the evidence for each: [`docs/STATUS.md`](docs/STATUS.md).
> The goal and the MVP: [`docs/VISION.md`](docs/VISION.md).

## Install

Python 3.10, TensorFlow 2.10 (the last release with native Windows GPU support).

```powershell
# Windows, GPU: creates the `nt` conda env (CUDA 11.2 / cuDNN 8.1 + pinned pip stack)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
conda activate nt
pip install -e ".[viz,dev]"
```

```bash
# Linux / CPU only
pip install -r requirements-ci.txt
pip install -e ".[viz,dev]"
```

Importing `neural_trade` does not import TensorFlow. Training modules load it on demand.

## Performance

On Windows, TensorFlow 2.10 finds CUDA/cuDNN only through `PATH`. `import neural_trade` adds the
conda env's DLL folders when the env has the CUDA runtime, so **import `neural_trade` before
`tensorflow`** (or start from an activated env). Training logs which device it uses and warns if
no GPU is visible. Set `NEURAL_TRADE_NO_DLL_PATH=1` to opt out.

Training time, full dataset (~30k training sequences), RTX 4070 Ti:

| | before | now |
|---|---|---|
| epoch at batch 64 | ~90 s | 43 s |
| epoch at batch 256 (the default) | ~20 s | 12 s |
| 20-epoch run with the default settings | ~32 min | ~4-5 min |

What made the difference:

- **Training metrics.** The step returns only the running loss. The ~150 epoch-level training
  diagnostics are computed once per epoch and updated every `TRAIN_METRICS_EVERY` steps (10). The
  training loss and all validation metrics stay exact.
- **Soft-ECE loss.** It is computed for all bins in one operation.
- **Learnable indicators.** Every EWMA of every family runs inside 2 batched matrix products
  (57 first-stage and 12 second-stage averages at the OHLCV default; 24 in close-only mode).
- **Batch size.** A step costs about the same at 64 or 256 because it is kernel-launch-bound.

For bulk prediction, pass a larger `batch_size` to `Predictor.predict` / `predict_frame`, or
`--batch-size` in the CLI: about 16,700 windows/s at 1024 on the GPU. The default, the training
batch, reproduces training output bit for bit.

## Quick start (CLI)

```bash
# train on the bundled CSV, evaluate on the held-out test block, save a serving bundle
neural-trade train --epochs 20                        # prints runs/<run id>
neural-trade train --config configs/default.yaml --set LR=5e-4 --set EPOCHS=10

# forecast with a saved bundle
neural-trade predict  --artifacts runs/<run id>/artifacts --csv bars.csv --last
neural-trade predict  --artifacts runs/<run id>/artifacts --csv bars.csv --out forecasts.csv

# backtest a strategy (fees, spread, slippage, next-open fills, stops on high/low)
neural-trade backtest --artifacts runs/<run id>/artifacts --csv bars.csv --out bt/ --plot   # calibrated_quantile

# experiment engine: every (variant, fold, seed) cell of a scenario, scored on dev / test folds (resumable)
neural-trade scenario plan configs/scenarios/reference.yaml   # validate, list the cells (no training)
neural-trade scenario run  configs/scenarios/reference.yaml   # runs/scenarios/<name>/, index runs/index.sqlite

neural-trade registry list            # every registered component
neural-trade registry info Optimizers adamw
neural-trade env                      # versions, CUDA build, devices, git state
```

Every Config key with its default, unit, valid range and tunable / deprecated flags:
[docs/guide/config-reference.md](docs/guide/config-reference.md) (generated by `scripts/gen_config_reference.py`).

The CSV needs a timestamp column plus open/high/low/close/volume, as in
`binance_btcusdt_1min_ccxt.csv`.

## Python API

```python
from neural_trade.core.config import Config
from neural_trade.training.trainer import train_and_evaluate
from neural_trade.serving.predictor import Predictor

cfg = Config.from_yaml("configs/default.yaml").override(EPOCHS=10)
result = train_and_evaluate(config=cfg, force=True)       # TrainResult: model, predictions, calibration

p = Predictor.from_artifacts("runs/<run id>/artifacts")
p.predict_last(ohlcv_dataframe)  # {"h0": {"delta", "p_up", "p_up_calibrated", "sigma", "lo90", "hi90", ...}, ...}
                                 # (a bare close series suffices for close-only bundles)
```

Evaluation and backtesting:

```python
from neural_trade.evaluation.frame import PredictionFrame
from neural_trade.evaluation.report import evaluate
from neural_trade.strategy import Bars, SignalFrame, backtest, build_strategy, var_scale_from

test, cal = PredictionFrame.from_result(result, "test"), PredictionFrame.from_result(result, "cal")
report = evaluate(test, result.config, cal_frame=cal)      # EvalReport: to_json / to_markdown / flat
signals = SignalFrame.build(test, var_scale_from(cal))     # confidence scale from the CAL block
bt = backtest(signals, bars, build_strategy("liberal"))    # bars: Bars.from_frame(df, anchor rows)
```

## Notebooks

The notebooks in `notebooks/` contain no functions. The logic and widgets live in
`neural_trade.notebook` and are tested headlessly.

| Notebook | What you can do |
|---|---|
| `00_data_and_splits` | See the bars, the purged train / val / cal / test blocks of any walk-forward fold, and the label balance per block |
| `01_train_and_monitor` | Train in the background with **Pause / Resume / Stop**. The live dashboard shows training-health tiles (convergence, patience, class collapse per horizon, non-finite gradients), the loss batch by batch, and 12 per-epoch panels: losses and components, direction MCC / balanced accuracy / ECE per horizon for train and validation, the price head's Gaussian readout, PIT-KS, predicted vs true up-rate, physics terms, learning rates, gradient norm. Then the test report with baselines, analytics for every head per horizon, and the learned indicator periods |
| `02_backtest` | Strategy dropdown, per-strategy settings and cost controls with a **Run** button. The trading dashboard stacks price and trades (entries, exits, take-profit / stop levels), P(up) per horizon against the entry lines, confidence, predicted sigma, equity and drawdown on one time axis with shared hover and zoom. Per-trade analytics cover P&L, cost drag, exit reasons and favourable / adverse excursion. Every strategy is compared on the same block |
| `03_signals_and_trades` | The trading dashboard for a window of bars (`START`, `BARS`), the signal features, and the trades |
| `04_diagnostics` | Everything about a saved run: its training dashboard, the head analytics, the learned periods, and the **calibration explorer**, which refits on the calibration block (interval scale, delta shrinkage, miscoverage) and shows test coverage, width, temperature, reliability diagram and coverage over time |
| `05_compare_runs` | Compare scored runs side by side, and read the ablation verdicts with their paired deltas |

The buttons work because training runs in a background thread: Jupyter processes widget
clicks only while no cell is running.

The figures live in `neural_trade.visualization` and are registered in `Visualizations`
(`training_dashboard`, `direction_analytics`, `delta_analytics`, `variance_analytics`,
`confidence_analytics`, `coherence_analytics`, `trading_dashboard`, `trade_analytics`,
`strategy_comparison`, `indicator_evolution` ...). They share one dark theme in which each
horizon keeps its colour in every figure (`visualization/theme.py`).

## How it is evaluated

- **Purged four-way split.** Train | val | cal | test in time order, with an 80-sequence gap
  between blocks (lookback + longest horizon), so no bar is both a training label and an
  evaluation input. Scalers are fitted on train; early stopping uses val; temperature scaling,
  conformal intervals and the strategies' confidence scale use cal; test is scored once.
  `FOLD_INDEX` selects a walk-forward fold.
- **Baselines in every report.** Zero change, mean change, class prior, logistic regression on
  trailing returns, and constant variance. Each report says, per metric, whether the model
  beats each baseline. Explained variance is reported on the price *change* only: on price
  levels, "no change" already scores 0.999.
- **Direction metrics** use a 5 bps deadband (moves smaller than that are not labelled).
  They include MCC, AUC, Brier, positive-class ECE, and a confidence gap with a
  block-bootstrap interval.
- **Variance metrics:** CRPS and CRPSS against constant variance, NLL, PIT-KS,
  variance/error-squared Spearman correlation, and conformal coverage and width. Conformal
  intervals are scaled by each window's realised volatility (`CONFORMAL_SCALE`), so they
  keep their coverage when volatility changes between the cal and test blocks.

## Backtests

`neural_trade.strategy` places orders at a bar's close and fills them at the next bar's
open. Each side pays a fee, a half-spread and a slippage cost, `BacktestConfig` fields that
default to 0 bps each (D-044: no trading costs assumed by default; set them explicitly to
backtest at a cost). Take-profit and
stop-loss are checked against each bar's high and low; if both are hit in the same bar, the
stop is assumed to fill first, and a gap through the stop fills at the open. Trades are
capped at 30 bars, and any open position is marked to market at the end. Every result
carries three baselines: buy-and-hold, always-flat, and random entries at the same trade
frequency, holding time and mean position size (the strategy's percentile among 100 random
seeds, after and before costs). A null at full size would pay more costs than a strategy that
sizes down; the CLI, `scripts/backtest_gate.py` and the notebooks use this one null.
`assert_no_lookahead` perturbs all predictions and bars after bar *t* and checks that nothing up to *t*
changes: the `decisions`, every `exit_signal` answer and each order's `tp`, `sl` and `max_hold`
(recorded in a private trace, since the `decisions` dicts carry none of them), and the equity
(for exposure strategies: the fills and every evaluated target). A one-bar peek differs only at
its own bar, so the default probes are up to 24 bars where the strategy was asked something
(orders, exit requests, targets) plus a grid of 4. The tests run it on every registered strategy
and on deliberately leaky strategies (a peek in `decide`, in `exit_signal`, in the TP level).

The three notebook strategies are ported as `threshold_spike`, `enhanced_multi_horizon` and
`liberal`, with their knobs as dataclass fields (see `configs/strategies/`). The port fixed
their look-ahead and sizing bugs; the module docstrings list each fix. They use fixed
probability lines (a horizon "votes" beyond 0.55 / 0.45), which a calibrated weak-edge model
almost never crosses. The default, `calibrated_quantile`, therefore sets its entry lines on the
calibration block instead: long above the 90th percentile of its confidence-weighted P(up),
short below the 10th. The serving bundle stores those quantiles.

## Physics-term ablation

```bash
python scripts/ablate.py --scale smoke --dry-run      # pending cells, projected hours
python scripts/ablate.py --scale full                 # 14 conditions x 3 seeds x 2 folds = 84 runs, resumable
```

The grid runs all-on, all-off, each term alone, and all-but-one. Each cell trains in its own
process with frozen, pre-calibrated loss weights, then evaluates against the baselines and
runs a backtest. Verdicts (VALUE / HARMFUL / NEUTRAL / INCONCLUSIVE) come from paired deltas
over (seed, fold) under criteria fixed in advance in `configs/ablation_criteria.yaml`.
Results go to `runs/ablations/<name>/report.md`.

## Layout

```
src/neural_trade/
  core/           Config (typed, flat, YAML round-trip), BaseRegistry, exceptions, logging, plugin loader
  registries/     the ten registries: Models, Losses, Optimizers, Metrics, Callbacks,
                  DataLoaders, Preprocessors, Layers, Visualizations, Indicators
  indicators/     the learnable indicator families (registry entries: inputs, learnable
                  parameters with bounds, output channels, drawing spec, M(eps))
  data/           loaders, preprocessors, windowing, purged splits, scaling, DataProcessor
  models/         gru_attention + layers (learnable indicators, positional encoding, noise, energy gate)
  losses/         the loss terms and the custom objective
  training/       CustomTrainModel, loss weights, optimizers, lambda calibration, callbacks, trainer, artifacts
  metrics/        numpy and graph-safe TF metrics, direction labels
  calibration/    temperature scaling, (normalised) conformal intervals, online calibrator
  evaluation/     PredictionFrame, baselines, evaluate(), walk-forward, report plots
  serving/        Predictor (raw OHLCV bars in, forecasts out)
  strategy/       signals, strategies, backtest engine, performance statistics
  experiments/    RunContext (one directory per run), ablation harness
  telemetry/      JSONL epoch logger
  visualization/  training dashboard, indicator evolution, trading and evaluation figures
  cli.py          the `neural-trade` command
configs/          default.yaml, ci.yaml, ablation specs, strategy parameter files
plugins/          drop-in components (templates/ and a tested example)
scripts/          gate runs, golden-run oracle, ablation, backtests of saved runs, env setup
tests/            pytest suite (markers: tf, slow, gpu, data, notebook)
```

To add a component, register it with its registry, for example
`@Metrics.register(name="my_metric")`, in a module under `plugins/`. `registries.load_all`
loads the plugins, and the config then selects the component by name. See `plugins/README.md`.

## Tests

```bash
pytest -m "not slow"          # 5-6 min on CPU
pytest -m slow                # end-to-end training, CLI round trip, reproducibility
python scripts/golden_run.py verify <oracle.npz>   # a refactor changed no numbers
```

## Working on this project

Development runs as a multi-session loop with Claude Code agents. Start with
[CLAUDE.md](CLAUDE.md) (loaded by every session). It points to the vision
([docs/VISION.md](docs/VISION.md)), the roadmap and backlog ([docs/ROADMAP.md](docs/ROADMAP.md),
[docs/BACKLOG.md](docs/BACKLOG.md)), the current state ([docs/STATUS.md](docs/STATUS.md)), settled
decisions ([docs/DECISIONS.md](docs/DECISIONS.md)), how work is done
([docs/OPERATING_MODEL.md](docs/OPERATING_MODEL.md)) and how to run everything
([docs/RUNBOOK.md](docs/RUNBOOK.md)). The notebooks are generated and executed by
`scripts/notebooks/` ([README](scripts/notebooks/README.md)).

## Licence

MIT. See [LICENSE](LICENSE).
