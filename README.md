# neural-trade

Multi-horizon forecasting for 1-minute BTC/USDT bars. From the last 60 closes, one network
predicts, for 10, 15 and 20 minutes ahead:

- the price change in dollars (`delta`),
- the probability that the price goes up (`direction`),
- the variance of the price change (`sigma`),

and a set of learnable technical indicators (EMA, MACD, RSI, Bollinger periods trained by
gradient descent). Six "physics-inspired" regularisers act on these heads. An ablation harness
tests whether each one earns its place.

The package lives in `src/neural_trade/`. Around the model it provides a typed config, nine
component registries, a purged evaluation protocol with baselines, post-hoc calibration, a
serving API, an honest backtest engine, run tracking and a CLI.

> **Status.** The model trains: finite gradients, calibrated variance heads and conformal
> intervals with the target coverage (milestones M1, M2 and M4 in
> [`runs/gates/REPORT.md`](runs/gates/REPORT.md)). It does **not yet have directional skill**
> (milestone M3). Its direction heads stay near 0.5, while a logistic regression on trailing
> returns from the same window reaches out-of-sample AUC ≈ 0.52–0.53. Read every backtest in
> that light.

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

## Quick start (CLI)

```bash
# train on the bundled CSV, evaluate on the held-out test block, save a serving bundle
neural-trade train --epochs 20                        # prints runs/<run id>
neural-trade train --config configs/default.yaml --set LR=5e-4 --set EPOCHS=10

# forecast with a saved bundle
neural-trade predict  --artifacts runs/<run id>/artifacts --csv bars.csv --last
neural-trade predict  --artifacts runs/<run id>/artifacts --csv bars.csv --out forecasts.csv

# backtest a strategy (fees, spread, slippage, next-open fills, stops on high/low)
neural-trade backtest --artifacts runs/<run id>/artifacts --csv bars.csv \
                      --strategy enhanced_multi_horizon --out bt/ --plot

neural-trade registry list            # every registered component
neural-trade registry info Optimizers adamw
neural-trade env                      # versions, CUDA build, devices, git state
```

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
p.predict_last(close_series)   # {"h0": {"delta", "p_up", "p_up_calibrated", "sigma", "lo90", "hi90", ...}, ...}
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
open. Each side pays a 10 bps fee, 1 bps half-spread and 2 bps slippage. Take-profit and
stop-loss are checked against each bar's high and low; if both are hit in the same bar, the
stop is assumed to fill first, and a gap through the stop fills at the open. Trades are
capped at 30 bars, and any open position is marked to market at the end. Every result
carries three baselines: buy-and-hold, always-flat, and random entries at the same trade
frequency (the strategy's percentile among 100 random seeds).
`assert_no_lookahead` perturbs all data after bar *t* and checks that nothing up to *t*
changes. The tests run it on every registered strategy.

The three notebook strategies are ported as `threshold_spike`, `enhanced_multi_horizon` and
`liberal`, with their knobs as dataclass fields (see `configs/strategies/`). The port fixed
their look-ahead and sizing bugs; the module docstrings list each fix.

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
  registries/     the nine registries: Models, Losses, Optimizers, Metrics, Callbacks,
                  DataLoaders, Preprocessors, Layers, Visualizations
  data/           loaders, preprocessors, windowing, purged splits, scaling, DataProcessor
  models/         gru_attention + layers (learnable indicators, positional encoding, noise, energy gate)
  losses/         the loss terms and the custom objective
  training/       CustomTrainModel, loss weights, optimizers, lambda calibration, callbacks, trainer, artifacts
  metrics/        numpy and graph-safe TF metrics, direction labels
  calibration/    temperature scaling, (normalised) conformal intervals, online calibrator
  evaluation/     PredictionFrame, baselines, evaluate(), walk-forward, report plots
  serving/        Predictor (raw closes in, forecasts out)
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
pytest -m "not slow"          # ~2 min on CPU
pytest -m slow                # end-to-end training, CLI round trip, reproducibility
python scripts/golden_run.py verify <oracle.npz>   # a refactor changed no numbers
```

## Licence

No licence file has been chosen yet. That decision is the repository owner's.
