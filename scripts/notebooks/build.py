"""Generate the notebooks in notebooks/ from the cell lists below.

EDIT THE NOTEBOOKS HERE, NEVER BY HAND. Every .ipynb in notebooks/ is written by this file.
tests/test_notebook_tooling.py fails when a committed notebook's cells (type, source, tags) differ
from what this file generates, so a change made in Jupyter is lost at the next build or fails the test.

    python scripts/notebooks/build.py                           # all of them, into notebooks/
    python scripts/notebooks/build.py 01_train_and_monitor 04   # some: full name or number prefix
    python scripts/notebooks/build.py --out D:/tmp/nb           # write somewhere else (e.g. to diff)
    python scripts/notebooks/build.py --check                   # write nothing; exit 1 on drift

A build overwrites the notebook WITHOUT outputs. Execute, check and render it next
(scripts/notebooks/README.md). Build only the notebooks you changed: executing 01 trains a new run.

The notebooks stay thin (tests/test_notebooks_thin.py): no def, class or lambda in a cell, because
all logic lives in neural_trade. The first code cell of each notebook has the tag "parameters".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "notebooks"
KERNELSPEC = {"display_name": "Python 3 (nt)", "language": "python", "name": "python3"}

RUN_PICK = '''run_dir = pick_run(RUN_DIR, RUNS_DIR)   # newest run with a serving bundle, or a clear error
print("run:", run_dir)'''

# ---------------------------------------------------------------------------- 00
data = [
    ("md", """
# 00 - Data and splits

The raw bars, the purged four-way split for a walk-forward fold (train | val | cal | test, with an
80-sequence gap between blocks so no bar is both a training label and an evaluation input), and
what the labels look like in each block. Change `FOLD_INDEX` to see another fold; the same
override in notebook 01 trains on it.
"""),
    ("code", """
# Parameters
CONFIG_PATH = "../configs/default.yaml"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
OVERRIDES = {"FOLD_INDEX": -1}   # -1 = the latest fold; -2, -3 ... earlier ones
""", "parameters"),
    ("code", """
from IPython.display import display

from neural_trade.core.config import Config
from neural_trade.data.processor import split_arrays
from neural_trade.visualization.data_overview import split_overview_figure, split_table

cfg = Config.from_yaml(CONFIG_PATH).override(CSV_PATH=CSV_PATH, **OVERRIDES)
blocks = split_arrays(cfg)
print(f"{len(blocks['df']):,} bars, fold {blocks['fold'].fold}, gap {blocks['fold'].gap} sequences")
split_overview_figure(blocks, cfg).show()
"""),
    ("md", "## The blocks\n\nShare of labels inside the 5 bps deadband (not trained or scored on direction), up-rate outside it, and realised volatility."),
    ("code", """
split_table(blocks, cfg).round(4)
"""),
]

# ---------------------------------------------------------------------------- 01
train = [
    ("md", """
# 01 - Train and monitor

Trains in the background with **Pause / Resume / Stop** buttons and a live dashboard, then scores the held-out
TEST block against baselines and analyses every head per horizon. The notebook stays responsive during training
(the buttons only work while no cell is running). Stop ends training after the current batch; the run is still
evaluated, calibrated and saved.

The weights that are evaluated, calibrated and saved are the best-validation epoch's (the *served* epoch), also
when training ran all `EPOCHS`. Every verdict below is graded against noise: consecutive 1-minute samples share
most of their target bars, so the chance bands count fewer effective samples than bars (about `samples // bars ahead`). Horizons keep
their colours in every figure (h0 blue, h1 orange, h2 green); solid lines are validation, dotted lines training.
"""),
    ("code", """
# Parameters
CONFIG_PATH = "../configs/default.yaml"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
RUNS_DIR = "../runs"
OVERRIDES = {}            # e.g. {"EPOCHS": 5, "LR": 5e-4, "FOLD_INDEX": -2, "BATCH_SIZE": 64}
EPOCHS = None             # None -> Config.EPOCHS
CALIBRATE_LOSS_WEIGHTS = True
""", "parameters"),
    ("code", """
from IPython.display import HTML, Markdown, display

import neural_trade  # first: on Windows it puts the CUDA DLLs on PATH before TensorFlow loads
from neural_trade.core.config import Config
from neural_trade.data.processor import split_arrays
from neural_trade.evaluation.baselines import BaselineSet
from neural_trade.evaluation.frame import PredictionFrame
from neural_trade.evaluation.report import evaluate
from neural_trade.experiments.run_context import RunContext
from neural_trade.notebook import TrainingSession
from neural_trade.registries.visualizations import Visualizations
from neural_trade.strategy import var_scale_from
from neural_trade.visualization import analytics_tables as AT
from neural_trade.visualization.indicator_evolution import applied_periods, indicator_applied_periods, indicator_summary
import tensorflow as tf

print("GPU:", tf.config.list_physical_devices("GPU") or "none - training will run on the CPU")
cfg = Config.from_yaml(CONFIG_PATH).override(CSV_PATH=CSV_PATH, **OVERRIDES)
ctx = RunContext.create(cfg, root=RUNS_DIR, tags=["notebook"])
print("run:", ctx.run_dir)
"""),
    ("md", "## Train\n\nThis cell returns immediately; the dashboard below keeps updating. The next cell waits for the run to finish."),
    ("code", """
session = TrainingSession(ctx.config, run_context=ctx, epochs=EPOCHS, calibrate=CALIBRATE_LOSS_WEIGHTS)
display(session.widget())
session.start()
"""),
    ("code", """
result = session.wait()   # blocks until training, evaluation and calibration are done
print(session.status, "-", len(session.history), "epochs; served weights: epoch", result.weights_epoch,
      f"(val loss {result.weights_val_loss:.4f}, {result.weights_source})")
"""),
    ("md", """
### Training record

A static copy of the live dashboard, readable in the saved notebook without a running kernel. The health tiles
(hover one for what it checks) and the epoch table come first; then every per-epoch metric: total loss with the
best and served epochs, the validation loss by term as each term enters the total, MCC and balanced accuracy of
the direction head and the price head with 95% chance bands, Brier skill against the base rate, ECE and PIT-KS
with the level a calibrated head stays under, the up-call bias, the physics terms, learning rates and gradient
norm. Then the per-horizon direction metrics against their no-skill references, the loss terms per horizon, and
the loss batch by batch.
"""),
    ("code", """
display(HTML(session.health_html()))
session.curves_figure().show()
session.direction_figure().show()
session.loss_terms_figure().show()
session.batch_figure().show()
"""),
    ("md", """
## Evaluate on the TEST block

Baselines are fitted on the train block; the confidence threshold and calibration come from the cal block. The
report scores the served heads and the raw price heads (before delta shrinkage) side by side; the baseline table
gives the model and baseline values, the margin, a noise test (HAC Diebold-Mariano or block bootstrap) and the
verdict.
"""),
    ("code", """
blocks = split_arrays(ctx.config)
baselines = BaselineSet.fit(blocks["train"]["X"], blocks["train"]["y"], blocks["train"]["last_close"],
                            ctx.config.DIR_DEADBAND_BPS)
test = PredictionFrame.from_result(result, "test")
cal = PredictionFrame.from_result(result, "cal")
raw_test = result.predictions["delta"]
betas = result.calibration_pipeline.delta_scale if result.calibration_pipeline is not None else None
report = evaluate(test, ctx.config, baselines=baselines, cal_frame=cal, run_id=ctx.run_id,
                  raw_delta=raw_test, delta_scale=betas)
report.to_json(ctx.path("eval_report_test.json"))
display(Markdown(report.to_markdown(ctx.path("eval_report_test.md"))))
display(AT.styled(AT.baseline_table(report)))
"""),
    ("md", "### The numbers per horizon (test block)\n\nClassification per horizon (outside the deadband), and the price heads' errors in dollars: raw, served (beta x raw) and predicting zero."),
    ("code", """
display(AT.styled(AT.classification_table(test, ctx.config)))
display(AT.styled(AT.delta_quality_table(test, ctx.config, raw_delta=raw_test, delta_scale=betas)))
"""),
    ("md", """
## Model analytics on the TEST block

**Direction heads**: does the served P(up) differ between bars that went up and bars that went down? Row 2 draws
the ROC as lift over chance (TPR - FPR; the area is AUC - 0.5) for the direction head and the price head's
Gaussian readout, inside the band a no-skill head stays in. Row 3 shows the reliability of raw (open) vs
calibrated (filled) P(up) in 10 equal-count bins with block-clustered 95% intervals. The scorecard repeats the
evaluation report's numbers against a constant 0.5.
"""),
    ("code", """
Visualizations.build("direction_analytics", test, ctx.config, raw_delta=raw_test).show()
"""),
    ("md", "**Price heads**: the per-horizon error table (raw vs served vs predicting 0, with intervals), every sample's predicted vs realised move, the binned calibration curve, and how the correlation and the served skill drift along the block against their no-skill bands."),
    ("code", """
Visualizations.build("delta_analytics", test, ctx.config, raw_delta=raw_test).show()
"""),
    ("md", "**Variance heads**: does the predicted sigma size and rank the realised error, against a free baseline (the trailing 60-bar realised vol the conformal intervals use)? Is the error shape Gaussian (PIT, tail rates)? Do the 90% intervals hold their coverage and width along the block?"),
    ("code", """
Visualizations.build("variance_analytics", test, ctx.config, raw_delta=raw_test).show()
"""),
    ("md", """
**Direction confidence.** Row 1: accuracy by decile of |P(up) - 0.5| with 95% block-bootstrap intervals, the
accuracy a calibrated P(up) would reach (diamonds) and the no-skill level of each decile's call mix (ticks).
Row 2: selective accuracy, keeping only the most decided x%. Row 3: the same for the strategies' confidence
exp(-var / var_scale), which gates and sizes their trades. Row 4: confusion matrices with recall, precision,
balanced accuracy and MCC.
"""),
    ("code", """
Visualizations.build("confidence_analytics", test, ctx.config, var_scale=var_scale_from(cal), report=report).show()
"""),
    ("md", """
**Cross-horizon coherence.** P(up) correlation between horizons; the eight up/down vote patterns against what
independent votes would give; the realised up-rate by the number of horizons voting up; direction-head vs
price-head sign agreement; whether |delta| grows with the horizon on the raw heads vs the served (shrunk) deltas;
and the strategies' vote agreement at their vote lines.
"""),
    ("code", """
Visualizations.build("coherence_analytics", test, ctx.config, raw_delta=raw_test).show()
display(AT.styled(AT.magnitude_ordering_table(test, raw_delta=raw_test)))
display(AT.styled(AT.alignment_table(test, ctx.config, raw_delta=raw_test)))
"""),
    ("md", "Trailing move (the extended-trend feature and the momentum prior) vs the realised move; a correlation whose sign changes between blocks is not a stable edge."),
    ("code", """
display(AT.styled(AT.trailing_move_table(
    {"cal": cal, "test": test}, ctx.config,
    raw_deltas={"cal": result.predictions_cal["delta"], "test": raw_test},
    trends={"cal": blocks["cal"]["extended_trends"], "test": blocks["test"]["extended_trends"]})))
"""),
    ("md", """
## Learned indicator periods

The lines are the base periods per epoch; the model applies a per-window shifted period, whose medians on the
test windows (served weights) are the diamonds right of the last epoch. Change % is measured from the configured
start. The correlation bars are epoch-to-epoch changes against a noise band.
A second figure draws every registered family, one panel per family.
"""),
    ("code", """
metrics_path = ctx.path("metrics.jsonl")
applied = applied_periods(result, blocks["test"]["X_model"], block="test")   # the model-input windows
Visualizations.build("indicator_evolution", metrics_path, ctx.config, applied=applied).show()
Visualizations.build("indicator_family_periods", metrics_path, ctx.config, applied=applied).show()
indicator_applied_periods(applied, ctx.config, metrics=metrics_path).show()
display(AT.styled(indicator_summary(metrics_path, ctx.config, applied=applied)))
"""),
]

# ---------------------------------------------------------------------------- 02
backtest = [
    ("md", """
# 02 - Backtest a trained run

Pick a strategy, edit its knobs and the costs, press **Run backtest**. Everything is replayed on the run's own
TEST block (out of sample: the purged split is rebuilt from the run's config) with next-open fills, stops on the
bar's high/low, fees + spread + slippage, and three baselines (buy-and-hold, always-flat, random entries at the
same frequency, holding time and size).

The dashboard stacks price and trades, the per-horizon P(up) against the strategy's entry lines, confidence and
signal strength, predicted sigma, equity and drawdown on one time axis; hover shows every panel at that bar. The
whole block shows the equity story; stops, holding periods and entry-to-exit lines are drawn on views of 800 bars
or fewer, so a detail window follows (its numbers are the window's own). Then the per-trade analytics.
"""),
    ("code", """
# Parameters
RUN_DIR = None                  # a runs/<id> directory; None -> the newest run under RUNS_DIR
RUNS_DIR = "../runs"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
DETAIL_BARS = 600
DETAIL_AROUND = "steepest_fall" # "steepest_fall", "worst_trade" or "last"
""", "parameters"),
    ("code", """
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from neural_trade.notebook import BacktestExplorer, pick_run
from neural_trade.visualization.trading_dashboard import detail_window

""" + RUN_PICK + """
explorer = BacktestExplorer.from_run(run_dir, csv_path=CSV_PATH)
display(explorer.widget())
explorer.click_run()   # render the default strategy once; then use the controls
"""),
    ("md", "Static copy of that first result (the widget above holds the live one): the summary with the random null, the whole block, a detail window, and every trade."),
    ("code", """
display(explorer.summary_frame(styled=True))
explorer.dashboard().show()
start, end = detail_window(explorer.last, DETAIL_BARS, around=DETAIL_AROUND)
explorer.dashboard(start=start, end=end).show()
explorer.trade_analytics().show()
"""),
    ("md", """
## Every strategy with its default knobs

Each strategy is compared with random entries at its own trade rate, holding time and position size (the
diamonds; the same null as the summary table above). After costs the return is mostly cost x trade count, so only
the rank against that null tells skill from chance.
"""),
    ("code", """
runs, comparison = explorer.compare_strategies()
comparison.show()
explorer.comparison_table(runs, styled=True)
"""),
    ("md", """
A strategy with 0 trades is usually blocked by one of two things. Fixed probability lines
(`threshold_spike` enters above 0.65 / below 0.35) are rarely crossed by calibrated probabilities.
Strategies that need the predicted move to agree with the side (`enhanced_multi_horizon`) cannot trade
when delta shrinkage serves a zero delta. Delta shrinkage sets beta = 0 when the raw price head's
moves pointed the wrong way on the calibration block (negative correlation with the realised move).
This run's values:
"""),
    ("code", """
p_up = explorer.signals.p
pd.DataFrame({"delta beta (served delta = beta x raw)": pd.Series(explorer.blocks["predictor"].bundle.calibration_pipeline.delta_scale),
              "P(up) 1st percentile": np.percentile(p_up, 1, axis=0),
              "P(up) 99th percentile": np.percentile(p_up, 99, axis=0)}, index=["h0", "h1", "h2"]).round(4)
"""),
]

# ---------------------------------------------------------------------------- 03 (signals)
signals_nb = [
    ("md", """
# 03 - Signals and trades, up close

What a strategy sees and does in a window of the TEST block: price with every entry, exit, take-profit and stop;
the three horizons' P(up) against the entry lines; confidence and strength; predicted sigma with the variance
spikes; equity and drawdown, all as the window's own numbers. Set `START` to a bar number, or to
"steepest_fall" / "worst_trade" / "last", and `BARS` to the window length.
"""),
    ("code", """
# Parameters
RUN_DIR = None
RUNS_DIR = "../runs"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
STRATEGY = "calibrated_quantile"
START = "worst_trade"           # a bar number, or "steepest_fall" / "worst_trade" / "last"
BARS = 600
""", "parameters"),
    ("code", """
import os
from pathlib import Path

from IPython.display import display

from neural_trade.notebook import BacktestExplorer, pick_run
from neural_trade.visualization.trading_dashboard import detail_window

""" + RUN_PICK + """
explorer = BacktestExplorer.from_run(run_dir, csv_path=CSV_PATH)
res = explorer.run(STRATEGY, costs={"random_seeds": 0}, baselines=False)
start = detail_window(res, BARS, around=START)[0] if isinstance(START, str) else int(START)
explorer.dashboard(res, start=start, end=start + BARS).show()
"""),
    ("md", "## The signal features over the whole block\n\nShares of bars for the boolean flags, consensus and horizon votes, and the distribution of the numeric features."),
    ("code", """
for name, table in explorer.signal_summary().items():
    print(name)
    display(table.round(4))
"""),
    ("md", "## Trades"),
    ("code", """
explorer.trade_analytics(res).show()
trades = res.trades_frame()
if len(trades):
    display(trades.groupby("exit_reason")[["net_pnl", "gross_pnl", "bars_held"]].agg(["count", "mean", "sum"]).round(2))
    display(trades.groupby("side")[["net_pnl", "gross_pnl"]].agg(["count", "sum", "mean"]).round(2))
trades.tail(30).round(2)
"""),
]

# ---------------------------------------------------------------------------- 04 diagnostics + calibration
diag = [
    ("md", """
# 04 - Diagnostics and calibration

Everything about a saved run: its settings, its training record (health tiles and every per-epoch metric), the
test block re-scored with the baselines' noise tests, every head analysed per horizon, the learned indicator
periods, and an interactive **calibration explorer** that refits the calibration pipeline on the calibration
block with another interval scale, with or without delta shrinkage, at another miscoverage level. Nothing here
writes to the run.
"""),
    ("code", """
# Parameters
RUN_DIR = None
RUNS_DIR = "../runs"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
""", "parameters"),
    ("code", """
import os
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

from neural_trade.evaluation.baselines import BaselineSet
from neural_trade.evaluation.report import evaluate
from neural_trade.notebook import CalibrationExplorer, load_run_blocks, pick_run
from neural_trade.registries.visualizations import Visualizations
from neural_trade.visualization import analytics_tables as AT
from neural_trade.visualization.indicator_evolution import applied_periods, indicator_applied_periods, indicator_summary
from neural_trade.visualization.training_dashboard import training_health_html

""" + RUN_PICK + """
blocks = load_run_blocks(run_dir, csv_path=CSV_PATH)
test, cfg = blocks["test"], blocks["config"]
metrics = run_dir / "metrics.jsonl"
display(AT.styled(AT.run_settings_table(run_dir, cfg)))
"""),
    ("md", "## Training record"),
    ("code", """
display(HTML(training_health_html(metrics, cfg)))
Visualizations.build("training_dashboard", metrics, cfg).show()
Visualizations.build("training_direction_detail", metrics, cfg).show()
Visualizations.build("training_loss_terms", metrics, cfg).show()
"""),
    ("md", "## The TEST block, re-scored\n\nBaselines refit on the train block, so the noise tests appear even for runs saved before they existed. Served heads and raw price heads side by side."),
    ("code", """
pipe = blocks["predictor"].bundle.calibration_pipeline
betas = pipe.delta_scale if pipe is not None else None
raw = blocks["test_raw"].delta
tr = blocks["blocks"]["train"]
report = evaluate(test, cfg, baselines=BaselineSet.fit(tr["X"], tr["y"], tr["last_close"], cfg.DIR_DEADBAND_BPS),
                  cal_frame=blocks["cal"], run_id=run_dir.name, raw_delta=raw, delta_scale=betas)
display(Markdown(report.to_markdown()))
display(AT.styled(AT.baseline_table(report)))
display(AT.styled(AT.classification_table(test, cfg)))
display(AT.styled(AT.delta_quality_table(test, cfg, raw_delta=raw, delta_scale=betas)))
"""),
    ("md", "## The heads on the TEST block"),
    ("code", """
Visualizations.build("direction_analytics", test, cfg, raw_delta=raw).show()
Visualizations.build("delta_analytics", test, cfg, raw_delta=raw).show()
Visualizations.build("variance_analytics", test, cfg, raw_delta=raw).show()
Visualizations.build("confidence_analytics", test, cfg, var_scale=blocks["predictor"].bundle.meta.get("var_scale"),
                     report=report).show()
Visualizations.build("coherence_analytics", test, cfg, raw_delta=blocks["test_raw"]).show()
"""),
    ("code", """
display(AT.styled(AT.magnitude_ordering_table(test, raw_delta=raw)))
display(AT.styled(AT.alignment_table(test, cfg, raw_delta=raw)))
display(AT.styled(AT.trailing_move_table({"cal": blocks["cal_raw"], "test": blocks["test_raw"]}, cfg)))
"""),
    ("md", "## Learned indicator periods\n\n"
           "Base periods per epoch; the diamonds are the median applied (per-window shifted) periods of the "
           "served weights on the test windows. A second figure draws every registered family, one panel per family."),
    ("code", """
applied = applied_periods(blocks["predictor"], blocks["blocks"]["test"]["X_model"], block="test")
Visualizations.build("indicator_evolution", metrics, cfg, applied=applied).show()
Visualizations.build("indicator_family_periods", metrics, cfg, applied=applied).show()
indicator_applied_periods(applied, cfg, metrics=metrics).show()
AT.styled(indicator_summary(metrics, cfg, applied=applied))
"""),
    ("md", "## Calibration explorer\n\nFit on the calibration block, scored on the test block; the run itself is not changed."),
    ("code", """
calib = CalibrationExplorer(blocks)
display(calib.widget())
calib.click_refit()   # the run's saved settings first; then change them and press Refit
"""),
    ("md", "Static copy of that refit (the explorer above stays interactive): the refit next to the run's saved pipeline, per horizon."),
    ("code", """
display(calib.comparison_table().round(4))
reliability, coverage = calib.figures("h1")
reliability.show()
coverage.show()
"""),
]

# ---------------------------------------------------------------------------- 05 compare runs + ablation
compare = [
    ("md", """
# 05 - Compare runs and the ablation

Every scored run under `RUNS_GLOB` side by side (metrics come from each run's `eval_report_test.json`; unscored
runs are listed, not silently dropped), with intervals and the baseline where the report has them, and the
physics-term ablation's pre-registered verdicts with their paired deltas.
"""),
    ("code", """
# Parameters
RUNS_GLOB = "../runs/*"
METRICS = [f"{h}/{k}" for k in ("direction/auc", "direction/mcc", "delta/ev", "variance/crpss", "variance/coverage90")
           for h in ("h0", "h1", "h2")]
ABLATION_DIR = "../runs/ablations/ablate_physics_v1-full"
""", "parameters"),
    ("code", """
import json
from pathlib import Path

from IPython.display import Markdown, display

from neural_trade.experiments.compare import compare_runs
from neural_trade.visualization.comparison import ablation_deltas_figure, runs_comparison_figure

runs = compare_runs(RUNS_GLOB, metrics=METRICS, skip_unscored=True)
print(len(runs), "scored runs; unscored:", runs.attrs["unscored"])
runs
"""),
    ("code", """
runs_comparison_figure(runs, METRICS, run_dirs=RUNS_GLOB).show() if len(runs) else print("no scored runs yet")
"""),
    ("md", "## Physics-term ablation"),
    ("code", """
ablation = Path(ABLATION_DIR)
if (ablation / "analysis.json").exists():
    ablation_deltas_figure(json.loads((ablation / "analysis.json").read_text())).show()
    display(Markdown((ablation / "report.md").read_text()))
else:
    print("no ablation report at", ablation, "- run scripts/ablate.py")
"""),
]

# ---------------------------------------------------------------------------- 07 discovered indicators
discovered = [
    ("md", """
# 07 - Discovered indicators

The indicators the network learned, drawn on the price of one window next to the same indicators at their
textbook periods: the discovered indicators are what a run delivers (VISION), and the prediction and trading
quality in 01-04 are the evidence that they are good. Each copy of every registered family is drawn twice on the
same window: solid at the period the served model applied to that window, dashed at its configured textbook
period. The families are moving average, MACD, RSI, Bollinger, ATR, stochastic, Williams %R, Keltner, OBV,
VWAP, MFI, ADX, CCI and Donchian. The original four use the numpy formulas on the raw close; the other ten are
drawn from each family's outputs on the model window. Lines that smooth use an EWMA with alpha = 2 / (period + 1),
started at the window's first bar.

Set `WINDOW` to a window index, or to "typical" (closest to the block's median periods), "longest" / "shortest"
(where the per-window adjustment stretches or shrinks the periods most) or "last"; `BLOCK` to another block of
the run's fold. Nothing here writes to the run.
"""),
    ("code", """
# Parameters
RUN_DIR = None                  # a runs/<id> directory; None -> the newest run under RUNS_DIR
RUNS_DIR = "../runs"
CSV_PATH = "../binance_btcusdt_1min_ccxt.csv"
BLOCK = "test"                  # "train", "val", "cal" or "test"
WINDOW = "typical"              # a window index, or "typical" / "longest" / "shortest" / "last"
""", "parameters"),
    ("code", """
from neural_trade.data.processor import split_arrays
from neural_trade.notebook import pick_run
from neural_trade.registries.visualizations import Visualizations
from neural_trade.serving.predictor import Predictor
from neural_trade.visualization import analytics_tables as AT
from neural_trade.visualization.discovered_indicators import discovered_table
from neural_trade.visualization.indicator_evolution import applied_periods

""" + RUN_PICK + """
predictor = Predictor.from_artifacts(run_dir / "artifacts")
cfg = predictor.config.copy(CSV_PATH=CSV_PATH)
blocks = split_arrays(cfg)
block = blocks[BLOCK]
times = blocks["df"]["timestamp"].to_numpy()[block["anchor_bar"]]
metrics = run_dir / "metrics.jsonl"
applied = applied_periods(predictor, block["X_model"], block=BLOCK)   # the period each window gets (served weights)
print(f"{len(block['X']):,} {BLOCK} windows of {cfg.LOOKBACK} bars; served weights: epoch",
      predictor.bundle.meta.get("weights_epoch"))
"""),
    ("md", """
## The learned indicators on price

Top: the block's close, one point per window, with the drawn window shaded. The grid: one row per family, one
column per copy; each panel's heading gives the learned and the textbook period (MACD: fast / slow / signal).
Below it, each base period over training (from the recorded start, the dashed line is its textbook value) and,
right of the last epoch, how the period the model applies varies across the block's windows: 5-95% and the middle
50% of the windows, their median, this window's period and the served base period. A period longer than the
window has not warmed up by the window's end (the EWMA starts at the window's first bar); the table's
"applied > lookback" column counts those windows.
"""),
    ("code", """
Visualizations.build("discovered_indicators", block["X"], cfg, applied=applied, metrics=metrics, window=WINDOW,
                     times=times, ohlcv=block["X_model"]).show()
"""),
    ("md", """
## Learned against textbook periods

The table behind the figure, in bars: the served base period (the trained logit alone) and the median applied
period, each with its change against the textbook period; the applied range over the block's windows; this
window's period; the range the base period took during training; the share of windows whose applied period is
longer than the window; and whether the base period sits at a clip bound.
"""),
    ("code", """
AT.styled(discovered_table(applied, cfg, metrics=metrics, window=WINDOW), digits=2)
"""),
    ("md", """
## Every learned period over training

One panel per family: each learned period (a MACD's fast, slow and signal, a Keltner's period and ATR period, a
stochastic's K and D) against the epochs, from its textbook start (dashed). Families whose periods are not in the
run's log are named in the subtitle. This is the second figure of the run's `indicator_report.html`.
"""),
    ("code", """
Visualizations.build("indicator_family_periods", metrics, cfg, applied=applied).show()
"""),
    ("md", """
## Which indicators the model uses: grouped permutation importance

For each family instance, the channels of that instance are shuffled across the **validation** block's windows and
the model is scored again: the bar is how much the per-window loss rises, and one panel per horizon gives how much
the direction AUC falls (the hover also gives the hit-rate drop). Whiskers are the 2.5 and 97.5 percentiles of a
moving-block bootstrap (block of at least the longest horizon) that recomputes the loss and the AUC on every
resample (D-012); a bar whose whiskers cross zero is not distinguishable from an unused indicator. This is a
read-out after training: nothing in training reads it. It is the third figure of `indicator_report.html`.
"""),
    ("code", """
from neural_trade.serving.indicator_report import indicator_importance

Visualizations.build("permutation_importance", indicator_importance(predictor, blocks["val"]), cfg).show()
"""),
]

# ---------------------------------------------------------------------------- 08 long run
long_run = [
    ("md", """
# 08 - Long run: 360 days of training

One training run of the default model on a 360-day block of the long BTC/USDT history (`Bitcoin_BTCUSDT.csv`,
2017-01 to 2025-09, one-minute bars; machine-local), launched and tracked from here (D-040). The scenario
`configs/scenarios/long_360d.yaml` trains walk-forward fold -2: 518,432 training windows (360 days), 30-day
validation and calibration blocks, and a ~32-day out-of-sample block that is a **dev** block. Fold -1, the newest
~32 days, stays untouched as the test fold (D-020). The run goes through the experiment engine
(`neural-trade scenario run`) into `runs/scenarios/long_360d/`.

**How to use it.** Open this notebook in VS Code and run all cells. With `LAUNCH = False` (the default) nothing
starts: the monitor cell shows the state. To start the run, set `LAUNCH = True` and run the launch cell once: it
starts a detached process on the GPU that keeps running when the kernel or VS Code closes, and refuses to start
a second one while a cell is running or once it is done. Re-run the **monitor cell** whenever you want to see the
progress: state, epochs, elapsed time, an ETA estimate (early stopping can end the run sooner), the learning rate,
the best validation loss, the progress figure, notebook 01's training dashboards for the finished epochs, a GPU
snapshot and the tail of the process log. The **results cell** shows the scored out-of-sample (dev) block once
the run has written its `result.json`. To stop the run: `taskkill /PID <pid> /T /F` with the pid the monitor cell
prints (the cell stays without `result.json`; launching again trains it anew). docs/RUNBOOK.md "Long runs".
"""),
    ("code", """
# Parameters
ROOT = ".."                                   # the repository root (this notebook runs in notebooks/)
SPEC = "configs/scenarios/long_360d.yaml"     # relative to ROOT
STORE = "runs"                                # the run store, relative to ROOT
LAUNCH = False                                # True: start the run (once); False: only show its state
""", "parameters"),
    ("code", """
from neural_trade.notebook.longrun import launch, progress, results, show_progress, show_results

if LAUNCH:
    print(launch(SPEC, STORE, root=ROOT))
else:
    print("LAUNCH is False: nothing started. Set LAUNCH = True and re-run this cell to start the run.")
"""),
    ("md", """
## Progress

Re-run this cell to refresh. Top: the state and the numbers; then six panels (training and validation loss with
the best epoch, the learning rates, seconds per epoch and per step, elapsed hours with the ETA projection, and the
validation CRPS per horizon), the per-epoch table, notebook 01's training dashboard (every logged validation
metric per horizon with its 95% chance range on the 43,200-window validation block) and the loss terms per
horizon, and the last 20 lines of the process log. Solid = validation, dotted = training.
"""),
    ("code", """
show_progress(progress(SPEC, STORE, root=ROOT))
"""),
    ("md", """
## Results

Once the cell has written `result.json`: the key numbers of the evaluation report per horizon (direction,
price and variance heads), the backtest of the default strategy (calibrated_quantile, knobs fitted on the
calibration block, next-open fills, default costs) against buy-and-hold, always-flat and random entries at the
same frequency, and the full `eval_report_dev.md`. These are dev-block numbers (fold -2's out-of-sample block).
"""),
    ("code", """
show_results(results(SPEC, STORE, root=ROOT))
"""),
]

# ---------------------------------------------------------------------------- 09 candidate run
candidate_run = [
    ("md", """
# 09 - A candidate run: training, fit and backtest

One saved run, end to end: what it is, its training record, how its heads score on the out-of-sample
(dev) block against the baselines, and a backtest at the notebook's own strategy and costs. Everything
here reads the run's saved files (config, metrics, the evaluation report, the saved prediction blocks)
so it needs neither the training CSV nor a fresh inference pass (`neural_trade.notebook.run_report`).

The default `RUN_DIR` is candidate **C1** (owner "сохрани 1-3", `configs/candidates/README.md`): the
360-day model's fold -2, seed 0 cell, `calibrated_quantile` at `entry_quantile` 0.9, full size, at
**zero trading costs** (D-044). Its dev (out-of-sample) block is 2025-07-27 to 2025-08-28.
"""),
    ("code", """
# Parameters
ROOT = ".."
RUN_DIR = "../runs/scenarios/long_360d_stab/20260930T094257Z-dce15ed-e3669618-default__f-2__s0"
CANDIDATE_ID = "C1"
MANIFEST_PATH = "../configs/candidates/manifest.json"
STRATEGY = "calibrated_quantile"
STRATEGY_PARAMS = {"entry_quantile": 0.9}
BACKTEST_PARAMS = {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0, "random_seeds": 20}
MAX_DASHBOARD_POINTS = 6000      # thins the trading dashboard's per-bar lines on this long a block (D-013 size)
MAX_ROC_POINTS = 300             # direction_analytics' own ROC-curve resolution
""", "parameters"),
    ("md", "## Summary"),
    ("code", """
from pathlib import Path

from IPython.display import HTML, Markdown, display

from neural_trade.notebook import BacktestExplorer, run_report
from neural_trade.visualization import analytics_tables as AT

run_dir = Path(RUN_DIR)
print("run:", run_dir)
display(Markdown(run_report.overview_markdown(run_dir, candidate_id=CANDIDATE_ID, manifest_path=MANIFEST_PATH)))
run_report.blocks_table(run_dir)
"""),
    ("md", "### Key numbers (dev block): direction AUC against the logistic-regression-on-lags baseline, variance CRPSS against constant variance, and 0.90-target conformal coverage."),
    ("code", """
display(AT.styled(run_report.key_numbers_table(run_dir)))
"""),
    ("md", "### The candidate's six cells, for context (`configs/candidates/manifest.json`; already verified, not recomputed here)"),
    ("code", """
run_report.candidate_cells_table(CANDIDATE_ID, MANIFEST_PATH).round(4)
"""),
    ("md", """
## Training

The training dashboard (every logged validation metric per horizon with its chance range) and the loss
terms, from the run's `metrics.jsonl`; then the learned indicator periods over training (the per-window
applied range needs the training CSV and is not shown here; see notebook 07 on a run with it at hand).
"""),
    ("code", """
from neural_trade.registries.visualizations import Visualizations
from neural_trade.visualization.indicator_evolution import indicator_summary
from neural_trade.visualization.training_dashboard import training_health_html

cfg = run_report.load_config(run_dir)
metrics_path = run_dir / "metrics.jsonl"
display(HTML(training_health_html(metrics_path, cfg)))
Visualizations.build("training_dashboard", metrics_path, cfg).show()
Visualizations.build("training_loss_terms", metrics_path, cfg).show()
"""),
    ("code", """
Visualizations.build("indicator_evolution", metrics_path, cfg).show()
display(AT.styled(indicator_summary(metrics_path, cfg)))
"""),
    ("md", """
## Fit evaluation on the out-of-sample block

The evaluation report already scored on the run (`eval_report_dev.md`/`.json`) against every baseline;
then the per-horizon direction, price and variance diagnostics notebook 04 draws, and the reliability /
coverage tables, all from the saved `predictions_oos.npz`.
"""),
    ("code", """
from neural_trade.evaluation.report import EvalReport
from neural_trade.experiments.scorer import load_block

report = EvalReport.from_json(run_dir / "eval_report_dev.json")
display(Markdown(report.to_markdown()))
display(AT.styled(AT.baseline_table(report)))
oos_frame, oos_bars, oos_extra = load_block(run_dir / "predictions_oos.npz")
raw_delta = oos_frame.meta.get("delta_raw")
display(AT.styled(AT.classification_table(oos_frame, cfg)))
display(AT.styled(AT.delta_quality_table(oos_frame, cfg, raw_delta=raw_delta)))
"""),
    ("code", """
Visualizations.build("direction_analytics", oos_frame, cfg, raw_delta=raw_delta, max_points=MAX_ROC_POINTS).show()
Visualizations.build("delta_analytics", oos_frame, cfg, raw_delta=raw_delta).show()
Visualizations.build("variance_analytics", oos_frame, cfg, raw_delta=raw_delta).show()
"""),
    ("md", """
## Backtest

`calibrated_quantile` at `entry_quantile` 0.9, fitted on the calibration block, at zero trading costs
(D-044): the summary against buy-and-hold, always-flat and the size-matched random null, the trading
dashboard (equity vs buy-and-hold, drawdown, positions; its per-bar lines thinned to `MAX_DASHBOARD_POINTS`
on a block this long, every trade and decision marker still at its own bar) and the per-trade analytics.
"""),
    ("code", """
explorer = BacktestExplorer(run_report.load_saved_blocks(run_dir))
res = explorer.run(STRATEGY, STRATEGY_PARAMS, BACKTEST_PARAMS)
display(explorer.summary_frame(res, styled=True))
explorer.dashboard(res, max_line_points=MAX_DASHBOARD_POINTS).show()
explorer.trade_analytics(res).show()
"""),
]

# ---------------------------------------------------------------------------- the notebooks, in run order
NOTEBOOKS = {
    "00_data_and_splits": data,
    "01_train_and_monitor": train,
    "02_backtest": backtest,
    "03_signals_and_trades": signals_nb,
    "04_diagnostics": diag,
    "05_compare_runs": compare,
    "07_discovered_indicators": discovered,
    "08_long_run": long_run,
    "09_candidate_run": candidate_run,
}


def notebook(cells) -> nbf.NotebookNode:
    """One notebook from ``(kind, source, *tags)`` tuples. A cell's id is its position, so a rebuild
    changes only the cells whose content changed (nbformat's default ids are random)."""
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": dict(KERNELSPEC), "language_info": {"name": "python"}}
    for i, (kind, src, *tags) in enumerate(cells):
        maker = nbf.v4.new_markdown_cell if kind == "md" else nbf.v4.new_code_cell
        cell = maker(src.strip("\n"))
        cell["id"] = f"cell-{i:02d}"
        if tags:
            cell.metadata["tags"] = list(tags)
        book.cells.append(cell)
    return book


def resolve(names) -> list[str]:
    """Notebook names, in run order, from full names, number prefixes ("01") or file names; all when empty."""
    if not names:
        return list(NOTEBOOKS)
    picked = set()
    for arg in names:
        stem = Path(arg).stem
        hits = [n for n in NOTEBOOKS if n == stem] or [n for n in NOTEBOOKS if n.startswith(stem)]
        if len(hits) != 1:
            raise SystemExit(f"unknown or ambiguous notebook {arg!r}; one of: {', '.join(NOTEBOOKS)}")
        picked.update(hits)
    return [n for n in NOTEBOOKS if n in picked]


def build(names=None) -> dict:
    """{name: notebook} for ``names`` (all when empty), in run order."""
    return {name: notebook(NOTEBOOKS[name]) for name in resolve(names)}


def write(book, path: Path) -> None:
    """Write a notebook with LF line endings (the repository's; nbformat's own write gives CRLF on Windows)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        nbf.write(book, fh)


def cell_sources(book) -> list:
    """What this file owns in a notebook: each cell's type, source and tags. Outputs, execution counts,
    ids and the metadata that execution adds are ignored."""
    out = []
    for c in book["cells"]:
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        out.append((c["cell_type"], src, tuple(c.get("metadata", {}).get("tags", []))))
    return out


def drift(nb_dir=NB_DIR, names=None) -> list[str]:
    """One line per notebook in ``nb_dir`` whose cells differ from what this file generates; a missing
    notebook counts, and so does (when every notebook is compared) an .ipynb this file does not generate."""
    nb_dir = Path(nb_dir)
    problems = []
    for name, book in build(names).items():
        path = nb_dir / f"{name}.ipynb"
        if not path.exists():
            problems.append(f"{name}: missing from {nb_dir}")
            continue
        want, have = cell_sources(book), cell_sources(json.loads(path.read_text(encoding="utf-8")))
        if want != have:
            first = next((i for i, (w, h) in enumerate(zip(want, have)) if w != h), min(len(want), len(have)))
            problems.append(f"{name}: {len(have)} cells saved, {len(want)} generated; first difference at cell {first}")
    if not names:
        extra = sorted(p.stem for p in nb_dir.glob("*.ipynb") if p.stem not in NOTEBOOKS)
        problems += [f"{stem}: not generated by build.py" for stem in extra]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="notebooks (full name or number prefix); default: all")
    ap.add_argument("--out", default=str(NB_DIR), help="folder to write to (default: notebooks/)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; compare the notebooks in --out with this file and exit 1 on any difference")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.check:
        problems = drift(out, args.names)
        for line in problems:
            print("DRIFT", line)
        print("notebooks match build.py" if not problems else
              f"{len(problems)} notebook(s) differ from build.py: rebuild them, or move the hand edit into build.py")
        return 1 if problems else 0
    out.mkdir(parents=True, exist_ok=True)
    for name, book in build(args.names).items():
        write(book, out / f"{name}.ipynb")
        print(f"wrote {out / f'{name}.ipynb'} (no outputs yet: execute it next)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
