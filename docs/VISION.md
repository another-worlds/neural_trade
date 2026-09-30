# Vision

Stable document, owned by the owner. It changes only when the owner changes direction. Plans live in
[ROADMAP.md](ROADMAP.md) and [BACKLOG.md](BACKLOG.md); the current state lives in [STATUS.md](STATUS.md).
Set by the owner on 2026-09-25 / 28 (D-019 to D-030); it replaces the BTC-trading framing of 2026-09-25.

## Purpose

`neural_trade` is a neural network that predicts complex financial time series from **technical
indicators whose parameters and combinations are learned by gradient descent**. It is a substitute for
manual technical-indicator search and analysis: instead of a person trying RSI 14 against RSI 21, or
one moving-average cross against another, the network learns the indicator set that predicts best,
shows what it discovered, and is judged by the same financial metrics a manual search would use.

It is not tied to a ticker or a timeframe. The instrument, the bar size, the input window and the
forecast horizons are configuration.

## The reference setup

BTC/USDT one-minute bars, a 60-minute input window, horizons of 10, 15 and 20 minutes. It was chosen
for its effectiveness, noise level and complexity, not because the project is about it. Every result
names the setup it was measured on. Data: `binance_btcusdt_1min_ccxt.csv` (30 days, in the repo; tests
and CI) and the local 2017-2025 BTC/USDT file (fingerprinted, not in git) for walk-forward folds spread
over different months. The training block is 7 days; the validation, calibration and out-of-sample
blocks follow it with their own configured lengths.

## What every run delivers

Both, judged together:

1. **The discovered indicators:** the indicator families, periods and combinations the network learned,
   readable, and drawn on the price chart next to the textbook defaults.
2. **The predictions built on them:** for each horizon, the price change in quote currency, P(up) and
   the variance, with calibrated probabilities and intervals, and a trading strategy on top.

The indicators are the product. The prediction and trading quality are the evidence that they are good.

## The yardstick

A search over configurations finds the one with the best financial metrics.

- **Leaderboard:** ranked by the net Sharpe ratio after trading costs (next-open fills, fees, spread,
  slippage; these default to 0, D-044: "издержки делай 0" - trading costs are zero unless a study sets
  them explicitly). Guard-rails shown beside it and able to disqualify a row: maximum drawdown, the
  number of trades, beating buy-and-hold and random entries at the same frequency.
- **Honest ranking:** rows are ranked on the development folds only (the out-of-sample blocks of the
  earlier walk-forward folds). Every row also shows its numbers on the held-out test fold, but those
  never rank.
- **Winner (MVP):** the top row, after the top five are re-run with three seeds and ranked by their mean.
- **The manual-search baseline:** the learned indicators must beat, under the same search budget and
  the same dev-fold net Sharpe, (a) the same network with the periods frozen at the textbook values and
  (b) classic technical-analysis rules (moving-average cross, RSI threshold, Bollinger breakout) whose
  parameters the same search tunes.
- **"A beats B" verdicts** (learned against frozen, a loss term on against off, any two scenarios): a
  paired test over (seed, fold) pairs on the same blocks, plus a minimum practical effect fixed before
  the run.
- A clear, evidenced negative answer is a valid outcome. It is recorded, not tuned away.

## The MVP

The owner's five growth points, taken foundations first (D-021). The MVP is done when all of these hold,
on the reference setup, with the evidence (runs, tests, executed notebooks) in the repo.

1. **Structure.** One experiment engine: a scenario and sweep specification, a resumable runner, one
   run store with an index, and one scorer, replacing today's four experiment paths. Packages layered
   without circular imports. Code is removed only when it is stale and has no effect on the current
   system. The notebooks work at every step.
2. **Configuration control panel and model comparison** (one framework). A control-panel notebook
   (ipywidgets and plotly), with the same engine behind a CLI for long unattended runs. Two sweep modes:
   **quick** (the whole sweep in about 5 minutes) and **Optuna** (Bayesian search, GPU budget measured
   and stated before it starts, may run overnight while the GPU is idle). The leaderboard and the
   verdict rules of the yardstick above.
3. **Gradient stability.** Hard invariants in CI; health numbers in every run (no more than 2% of the
   training step's time; a detailed per-loss-term probe behind a flag); an on-demand stress harness
   (scale and volatility sweeps, extreme inputs, fault injection, three seeds) that every new setup must
   pass. An unstable run is attributed to its loss term and fails loudly, and configuration validation
   and search spaces refuse hyperparameter regions known to fail. One pre-registered comparison of
   gradient-based loss weighting against today's calibration.
4. **Generality designed in.** Instrument, bar size, window and horizons are configured, the window and
   horizons in wall-clock time. Any number of horizons (the pairwise physics terms apply to each
   neighbouring pair). Every run records the dataset it used. The target stays the price change in quote
   currency. BTC/USDT one-minute stays the reference and the only setup tested in the MVP.
5. **Visual comprehension of learning, inference and the backtest.** First view: the learned indicators
   drawn on price against the textbook defaults, and how their periods moved. Every figure stays rich
   (D-014). Notebooks 00-05 keep their numbers and roles and are updated as the code evolves; new
   notebooks are added for new views (for example a control panel and the discovered indicators).

Also in the MVP: an **extendable indicator catalogue**: new indicators are added and integrated through a
registry (the design comes from the owner's indicator Q&A).

## Principles

- **Evidence, not claims.** A result counts only if it links to a run directory, a test or an executed
  notebook. "It works" means a real run was executed and its outputs were inspected.
- **Real runs over toy runs.** Tests on synthetic data are necessary but not sufficient. Figures and
  notebooks are verified on the shipped defaults.
- **Noise-aware statistics.** Consecutive bars share most of their target bars. Every interval and
  chance band uses effective samples (about `N // horizon bars`) or a block bootstrap. A number
  without its noise level is not reported as a finding.
- **Choices never use test data.** Sweeps rank on the development folds; A/B verdicts are pre-registered
  and judged once.
- **Honest trading numbers.** Next-open fills, fees, spread and slippage, stops on high/low, baselines
  in every report, and a random null at the same trade frequency. The default cost profile is 0 (D-044,
  owner decision: no tick order-book data, costs set to 0); a study states any non-zero cost it assumes.
- **Fast training.** Fast GPU training and solid optimisation are first-class requirements; inference
  speed is negligible (D-018).
- **Extendable by registries.** Components (models, losses, metrics, indicators, strategies, ...) are
  added through registries, not by editing the pipeline.
- **One visual system, rich figures.** Every figure uses `visualization/theme.py` and keeps every
  metric, band and table (D-014). Horizons keep their colours (h0 blue, h1 orange, h2 green; more
  horizons extend the palette in a fixed order); dotted lines mean training.
- **Notebooks are the living interface.** They are generated, executed on real runs and committed with
  their outputs (D-013), and they grow with the project.

## Fixed decisions (owner)

Settled; see [DECISIONS.md](DECISIONS.md). They change only with the owner.

- TensorFlow 2.10 / Keras 2 (the last release with native Windows GPU support). No Keras 3 migration.
- The component registries stay and stay wired into the training path.
- The physics-inspired loss terms stay. Fix their mathematics and iterate until they demonstrably provide
  value under pre-registered criteria, or until the evidence says they cannot.
- Git history is never rewritten. `master` changes only when the owner merges.

## Audience

The owner and a few reviewers. The docs are in English.

## Not in the MVP (designed for, not built)

- A second instrument or timeframe actually tested (the configuration supports it; the MVP tests only
  the reference setup).
- Markets with trading sessions (calendars, overnight gaps); the MVP assumes a 24/7 market.
- Return-based targets; one model trained on several instruments at once.
- New data sources (exchange APIs, order book); Keras 3 / newer TensorFlow; a web application or
  MLflow (the control panel is a notebook).
