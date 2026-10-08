# neural-trade

A neural network that predicts financial time series from technical indicators whose **parameters and
combinations it learns by gradient descent**. It is meant as a substitute for manual indicator search:
instead of a person trying RSI 14 against RSI 21, the network learns the indicator periods that predict
best, shows what it learned, and is judged by the financial metrics a manual search would use (dev-fold net
Sharpe after costs, against manual-search baselines). The instrument, the bar size, the window and the
horizons are configuration; the goal and the yardstick are in [docs/VISION.md](docs/VISION.md).

**Reference setup, the only one tested: BTC/USDT one-minute bars.** From the last 60 bars (open, high, low,
close, volume), one network predicts for 10, 15 and 20 minutes ahead the price change in quote currency
(`delta`), the probability that the price goes up (`direction`) and the variance of the change (`sigma`).
It learns 14 indicator families, 3 instances each (54 periods), and six "physics-inspired" regularisers act
on its heads. The code supports exactly three horizons today.

## What works and what does not

Read this before the rest. The live state is [docs/STATUS.md](docs/STATUS.md); the evidence is
[runs/gates/REPORT.md](runs/gates/REPORT.md) and
[runs/experiments/capacity_v1/REPORT.md](runs/experiments/capacity_v1/REPORT.md). The numbers below are a
snapshot of the latter (2026-10-06) and may be out of date when you read this.

- **The direction head has no skill worth trading.** On five judgement folds of the long history the
  default model's direction AUC is about 0.50 to 0.53 per horizon, and a logistic regression on trailing
  returns scores higher on the same blocks (h1: 0.535 against 0.552). The calibrated P(up) is not better
  than a constant 0.5 out of sample. When the calibration finds no usable signal for a horizon the run says
  so explicitly ("no usable direction signal") and the strategies that read P(up) stay flat on it.
- **The variance heads and the conformal intervals are the edge that exists.** The variance forecast beats a
  constant-variance forecast by a small margin (CRPSS about 0.01 to 0.02), and the 90% conformal intervals
  hold their coverage (0.89 to 0.94 in the same runs).
- **The owner's trading goal (hit rate above 60%, drawdown under 5%) is not reached.**
- **The learned indicators are the product, and they are only partly evidenced.** They can be drawn on price
  next to their textbook defaults (notebook 07). The comparison that would show them beating the same network
  with frozen periods and technical-analysis rules under the paired test of VISION is built, but its first
  real sweep has not run yet.
- **What is solid:** the purged split, the baselines in every report, noise-aware statistics, the next-open
  backtest with a random null, the resumable experiment engine, sweeps, the leaderboard, the paired
  comparator and the stability harness, all covered by tests.

A clear negative answer is a valid outcome of this project (VISION "The yardstick"): it is recorded, not
tuned away.

## Quick start

Python 3.10 and TensorFlow 2.10 (the last release with native Windows GPU support).

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

Extras: `.[notebooks]` for the notebooks, `.[sweep]` for Optuna sweeps. Importing `neural_trade` does not
import TensorFlow; on Windows, **import `neural_trade` before `tensorflow`** so the CUDA DLLs are found.

```bash
# train on the bundled CSV, evaluate on the held-out test block, save a serving bundle
neural-trade train --epochs 20                        # prints runs/<run id>
neural-trade train --config configs/default.yaml --set LR=5e-4 --set EPOCHS=10

# forecast with a saved bundle
neural-trade predict  --artifacts runs/<run id>/artifacts --csv bars.csv --last

# backtest a strategy (next-open fills, stops on high/low, a random null)
neural-trade backtest --artifacts runs/<run id>/artifacts --csv bars.csv --out bt/ --plot

# the experiment engine: every (variant, fold, seed) cell of a scenario, scored on dev and test folds
neural-trade scenario plan configs/scenarios/reference.yaml   # validate, list the cells (no training)
neural-trade scenario run  configs/scenarios/reference.yaml   # runs/scenarios/<name>/, index runs/index.sqlite
neural-trade leaderboard reference_default                    # ranked on the dev folds; test columns never rank

# sweeps: size the run first, then launch
neural-trade sweep configs/scenarios/reference.yaml --mode quick --dry-run --sec-per-step 0.17

neural-trade registry list            # every registered component
neural-trade env                      # versions, CUDA build, devices, git state
```

The CSV needs a `timestamp` (or `datetime`) column plus `open`, `high`, `low`, `close`, `volume` (lower case),
as in `binance_btcusdt_1min_ccxt.csv`. Every Config key with its default, unit and range:
[docs/guide/config-reference.md](docs/guide/config-reference.md). Tests, notebooks, GPU rules and the traps
of the development machine: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Guides

| Read | For |
|---|---|
| [Concepts](docs/guide/concepts.md) | learned indicators, horizons and heads, costs, dev folds against the test fold, the yardstick |
| [Reading the figures](docs/guide/reading-figures.md) | every figure the notebooks draw: what it shows, its noise band, how to read it |
| [Experiments](docs/guide/experiments.md) | scenarios, quick and Optuna sweeps and their GPU budget, the control panel, the leaderboard, A/B verdicts, a worked example |
| [Your own data](docs/guide/own-data.md) | the CSV format, the dataset spec, wall-clock windows and horizons, costs, the stability harness |
| [Architecture](docs/ARCHITECTURE.md) | module map, layering rules, the registries, the experiment engine and the run store |
| [Status](docs/STATUS.md) | what works, what is open, what waits for the owner (the place for current numbers) |

## Notebooks

The notebooks in `notebooks/` contain no functions: the logic and widgets live in `neural_trade.notebook`
and are tested headlessly. They are generated by `scripts/notebooks/build.py`, executed on the real
defaults and committed with their outputs ([scripts/notebooks/README.md](scripts/notebooks/README.md)).

| Notebook | What you can do |
|---|---|
| `00_data_and_splits` | See the bars, the purged train / val / cal / test blocks of any walk-forward fold, and the label balance per block |
| `01_train_and_monitor` | Train in the background with Pause / Resume / Stop, watch the live dashboard and health tiles, then read the test report with baselines and the analytics of every head |
| `02_backtest` | Pick a strategy, settings and costs and press Run; trading dashboard, per-trade analytics, every strategy against a random null |
| `03_signals_and_trades` | The trading dashboard for a window of bars, the signal features, and the trades |
| `04_diagnostics` | Everything about a saved run: training dashboard, head analytics, learned periods, and the calibration explorer |
| `05_compare_runs` | Compare scored runs side by side, and read the ablation verdicts with their paired deltas |
| `06_control_panel` | Choose a scenario, a search space and a mode; launch or resume a sweep; watch the leaderboard; compare rows with a paired verdict. Executing it starts nothing; only the Launch button does |
| `07_discovered_indicators` | The learned indicators drawn on price next to their textbook periods, how the periods moved, and permutation importance |
| `08_long_run` | Launch and monitor the 360-day training run (`LAUNCH = False` by default: nothing starts) |
| `09_candidate_run` | One saved run end to end from its own files: summary, training record, fit evaluation, backtest |

## Python API

```python
from neural_trade.core.config import Config
from neural_trade.training.trainer import train_and_evaluate
from neural_trade.serving.predictor import Predictor

cfg = Config.from_yaml("configs/default.yaml").override(EPOCHS=10)
result = train_and_evaluate(config=cfg, force=True)       # TrainResult: model, predictions, calibration

p = Predictor.from_artifacts("runs/<run id>/artifacts")
p.predict_last(ohlcv_dataframe)  # {"h0": {"delta", "p_up", "p_up_calibrated", "sigma", "lo90", "hi90", ...}, ...}
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

To add a component (a metric, a loss, an indicator family, a strategy), register it with its registry; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) "Registries" and [plugins/README.md](plugins/README.md).

## How it is evaluated

- **Purged four-way split.** Train, validation, calibration and test in time order, with a gap between
  blocks (80 bars on the reference setup) so no bar is both a training label and an evaluation input.
  Scalers are fitted on train; early stopping uses validation; temperature scaling, conformal intervals and
  the strategies' confidence scale use calibration; the test block is scored once and never ranks.
  `FOLD_INDEX` selects a walk-forward fold.
- **Baselines in every report.** Zero change, mean change, class prior, a logistic regression on trailing
  returns, and constant variance; each report says, per metric, whether the model beats each.
- **Noise-aware statistics.** Consecutive bars share target bars, so intervals and chance bands use
  effective samples (about `N // horizon`) or a block bootstrap.
- **Direction metrics** use a 5 bps deadband (smaller moves are not labelled): MCC, AUC, Brier, ECE, and a
  confidence gap with a block-bootstrap interval. **Variance metrics:** CRPS and CRPSS against constant
  variance, NLL, PIT-KS, and conformal coverage and width, with intervals scaled by each window's realised
  volatility.

## Backtests

`neural_trade.strategy` places orders at a bar's close and fills them at the next bar's open. Fee,
half-spread and slippage are `BacktestConfig` fields that **default to 0 bps** (D-044); set them explicitly to
backtest at a cost. Take-profit and stop-loss are checked against each bar's high and low (the stop fills
first if both are hit in one bar, and a gap through the stop fills at the open). Trades are capped at 30
bars and any open position is marked to market at the end. Every result carries three baselines:
buy-and-hold, always-flat, and random entries at the same trade frequency, holding time and mean position
size. `assert_no_lookahead` perturbs everything after bar *t* and checks that nothing up to *t* changes; the
tests run it on every registered strategy and on deliberately leaky ones.

The default strategy, `calibrated_quantile`, sets its entry lines on the calibration block (long above the
90th percentile of its confidence-weighted P(up), short below the 10th) because the fixed 0.55 / 0.45 lines
of the notebook strategies are almost never crossed by calibrated probabilities. The serving bundle stores
those quantiles.

## Performance

Training is kernel-launch bound; the default batch is 256 and the per-epoch training diagnostics are
computed once per epoch (`TRAIN_METRICS_EVERY`). Training speed is a first-class requirement and inference
speed is not (D-018). Measurements and the trade-offs: [docs/RUNBOOK.md](docs/RUNBOOK.md) and
[docs/DECISIONS.md](docs/DECISIONS.md) D-010, D-018, D-047.

## Physics-term ablation (frozen history)

```bash
python scripts/ablate.py --scale smoke --dry-run      # pending cells, projected hours
```

The v1 grid (`runs/ablations/ablate_physics_v1-full`) found no term that earns its place under its own
criteria (D-003). The script belongs to the frozen set of D-023: new experiments go through the experiment
engine.

## Tests

```bash
pytest -m "not slow" -n 8     # the fast suite (pytest-xdist; a few minutes on CPU)
pytest -m slow -n 8           # end-to-end training, CLI round trip, reproducibility
python scripts/golden_run.py verify <oracle.npz>   # a refactor changed no numbers
```

Tests run on the CPU with `CUDA_VISIBLE_DEVICES=-1`.

## Working on this project

Development runs as a multi-session loop with Claude Code agents. Start with [CLAUDE.md](CLAUDE.md) (loaded
by every session). It points to the vision ([docs/VISION.md](docs/VISION.md)), the roadmap and backlog
([docs/ROADMAP.md](docs/ROADMAP.md), [docs/BACKLOG.md](docs/BACKLOG.md)), the current state
([docs/STATUS.md](docs/STATUS.md)), settled decisions ([docs/DECISIONS.md](docs/DECISIONS.md)), how work is
done ([docs/OPERATING_MODEL.md](docs/OPERATING_MODEL.md)) and how to run everything
([docs/RUNBOOK.md](docs/RUNBOOK.md)).

## Licence

MIT. See [LICENSE](LICENSE).
