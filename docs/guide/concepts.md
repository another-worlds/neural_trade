# Concepts

The ideas you need before reading a figure or a leaderboard. Short on purpose: the goal and the
yardstick are the owner's words in [VISION](../VISION.md), every settled choice is in
[DECISIONS](../DECISIONS.md) (cited below as D-0xx), and the state of the work is in
[STATUS](../STATUS.md). This page explains; it does not repeat them.

## What the project is for

A person who looks for a trading signal in technical indicators tries many settings by hand: RSI 14
against RSI 21, one moving-average cross against another. `neural_trade` lets a neural network learn
those settings by gradient descent, shows the ones it learned, and judges them by the same financial
yardstick a manual search would use (D-019, D-020). The instrument, the bar size, the input window and
the horizons are configuration. One setup is tested: **BTC/USDT one-minute bars, a 60-minute window,
horizons of 10, 15 and 20 minutes** (the reference setup, D-022). Every result names the setup it was
measured on.

## Learned indicators

The model does not read a fixed indicator set. A layer (`models/layers/learnable_indicators.py`) holds
the **periods** of 14 indicator families as trainable numbers, 3 instances per family by default (54
periods; D-031, D-047):

- moving average / EMA, MACD, RSI and Bollinger bands, on the close;
- ATR, Stochastic, Williams %R, Keltner, OBV, VWAP, MFI, ADX/DMI, CCI and Donchian, on the full
  open-high-low-close-volume bars (`Config.INPUT_SERIES`).

Each family is one entry in the `Indicators` registry (`indicators/`): its inputs, its learnable
parameters with their textbook defaults and bounds, its output channels and how it is drawn. Where the
textbook form is not differentiable (a rolling maximum, a sign), the layer uses a smooth version.
Combinations of indicators are not built by hand: the network combines the channels itself (D-031).

Two period numbers appear in the figures:

- the **base period**: the trained number, logged every epoch as `period/<name>` in `metrics.jsonl`;
- the **applied period**: by default the model shifts each window's periods a little, from that window's
  own mean and maximum (`meta_adjust`, up to roughly x0.6 to x1.6 on long periods). The applied period
  is what the model actually used on that window. `Config.ADAPTIVE_INDICATORS` switches the shift off;
  then applied equals base. `Config.FREEZE_INDICATOR_PERIODS` goes further and keeps every period at its
  textbook value (the "frozen twin" of the yardstick).

The **textbook period** is the value a copy starts from (`MA_SPANS`, `MACD_SETTINGS`, `RSI_PERIODS`,
`BB_PERIODS` and the `INDICATOR_FAMILIES` table). "Learned against textbook" in a figure means: the
period after training against the period it started from. The discovered indicators are the product;
the predictions and trading quality are the evidence that they are good (VISION "What every run
delivers").

## Horizons and heads

For every horizon h (h0, h1, h2 = 10, 15, 20 bars on the reference setup) the model has three heads:

| head | what it predicts | how it is used |
|---|---|---|
| `delta` | the price change over h bars, in quote currency (USDT here) | the target stays a price change, not a return (D-022) |
| `direction` | P(up): the probability that the change is positive | trained with binary cross-entropy (D-006) |
| `sigma` | the variance of the price change | gives a Gaussian readout and conformal intervals |

After training, three post-hoc steps are fitted on the **calibration block**, not on the training data:
a temperature for P(up), a conformal interval scaled by each window's realised volatility (D-008,
target 90% coverage), and a shrinkage factor beta for the price change (`beta = clip(E[y d] / E[d^2],
0, 1)`, D-007). Two consequences matter when you read a figure:

- **beta = 0 is a legitimate result.** The served delta is then exactly 0, so any statistic of it is
  not a measurement and the figures print `n/a` instead of 0% or 100%. The raw head is shown beside it.
- **"No usable direction signal" is an explicit state** (D-066). When the temperature fit has no interior
  minimum (it hit a bound), that horizon's calibrated P(up) is flat, the run records the state in
  `direction_signal`, figures and tables print `n/a` for the calibrated numbers, and strategies that
  read P(up) stay flat on it instead of trading on a degenerate probability.

The code supports **exactly three horizons today**; a variable number of horizons is open work (NT-042,
D-022). The six "physics-inspired" loss terms (D-003) act on the neighbouring pairs of horizons; whether
they earn their place is decided by pre-registered ablation, not by this page.

## Costs

Backtests place an order at a bar's close and fill it at the **next bar's open**. Stops and targets are
checked against each bar's high and low, and a stop is assumed to fill first when both are touched in one
bar. The fee, half-spread and slippage per side are `BacktestConfig` fields (and the `FEE_BPS`,
`HALF_SPREAD_BPS`, `SLIPPAGE_BPS` entries of the Config, the instrument's cost profile). **They default
to 0 basis points** (D-044: the owner has no order-book data and chose zero costs). A study that assumes
a cost states it. Two things follow:

- every number labelled "net" is net of whatever cost profile that row used; the leaderboard states the
  profile per row, marks a row scored at another profile "not comparable" and bars it from winning;
- runs scored before 2026-09-30 used 13 basis points per side and are not comparable with zero-cost runs.

## Folds: dev against test

The data is split in time into four purged blocks: **train, validation, calibration and out-of-sample
(test)**, with a gap between them so no bar is both a training label and an evaluation input (D-005,
D-034; the gap is 80 bars on the reference setup). Early stopping uses the validation block; the
temperature, the conformal scale and beta use the calibration block. A **fold** is one placement of the
four blocks on the series. `FOLD_INDEX` picks the fold: -1 is the newest, -2 the one before it, and so
on (`N_FOLDS` of them).

- **Dev folds** are every fold except -1. Their out-of-sample blocks are where choices and rankings are
  made.
- **The test fold** is -1, the newest. It is shown beside every result and **never ranks or chooses**
  (D-020): picking by eye from the test columns is the one thing the project forbids.

An A/B verdict is judged on **judgement folds** that no earlier choice used, named before any compute is
spent (D-025, D-046).

## The yardstick

A run is ranked by its **dev-fold net Sharpe ratio after costs** (VISION "The yardstick"):

- Guard-rails beside the Sharpe can disqualify a row: maximum drawdown, number of trades, beating
  buy-and-hold, beating random entries at the same frequency.
- The learned indicators must beat, under the same search budget and the same Sharpe: the same network
  with periods frozen at the textbook values, and classic technical-analysis rules (moving-average cross,
  RSI threshold, Bollinger breakout) whose parameters are tuned by the same search.
- **"A beats B" is a paired test over judgement folds plus a minimum effect fixed beforehand**, never a
  single run's difference. Identical GPU runs differ by 0.01-0.05 AUC, so single runs mean little (D-025,
  D-046). A clear, evidenced negative answer is a valid outcome and is recorded, not tuned away.

## Noise

Consecutive bars share most of their target bars: a 20-bar target overlaps its neighbour's by 19 bars. So
a sample is not an independent draw, and every interval, chance band and verdict uses **effective
samples** (about `N // horizon` bars) or a block bootstrap (D-012). A number without its noise level is
not a finding. [Reading the figures](reading-figures.md) names the band each panel uses.

## Where to go next

- [Reading the figures](reading-figures.md) for what each plot shows;
- [Experiments](experiments.md) for scenarios, sweeps, the control panel, the leaderboard and verdicts;
- [Your own data](own-data.md) for another instrument or bar size;
- [Config reference](config-reference.md) for every setting with its unit and range.
