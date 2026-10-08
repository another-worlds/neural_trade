# Your own data

The instrument, the bar size, the input window and the horizons are configuration (D-019, D-022). The only
setup that has been tested is BTC/USDT one-minute bars (the reference setup); everything below is what the
code accepts and checks, not a claim that another market works well. Two limits to know first:

- **Exactly three horizons today.** `Config.validate` refuses any other number of entries in
  `HORIZON_STEPS`. A variable number of horizons is open work (NT-042).
- **A 24/7 market is assumed.** Trading sessions, calendars and overnight gaps are not handled (VISION
  "Not in the MVP").

## The CSV

One file, one instrument, rows in time order:

| column | notes |
|---|---|
| `timestamp` or `datetime` | text (any format pandas parses, with or without a time zone) or a number; a numeric column is read as epoch time and its unit (seconds, milliseconds, microseconds, nanoseconds) is taken from its size. The first of the two names that exists is used |
| `open`, `high`, `low`, `close`, `volume` | numeric; lower-case as in the example (the capitalised `Open`, `High`, `Low`, `Close`, `Volume` are read too, any other spelling is not). `close` is required and must be positive. The model input carries all five by default (`INPUT_SERIES`); `["close"]` gives the close-only input |

`binance_btcusdt_1min_ccxt.csv` is the example. The loader sorts the rows, keeps the last row of a repeated
timestamp, and drops rows with no close. Point a run at the file with
`CSV_PATH` (`--csv FILE` on the CLI, `CSV_PATH` in a scenario's `overrides`). A relative path resolves
against the working directory, then the repository root.

**Holes.** A gap of more than one bar in the timestamps is not silently bridged. `GAP_POLICY` (default
`drop`) builds no input window, past-delta lag or target that spans a hole and records how many windows that
removed; `refuse` makes any hole an error; `ignore` does not look.

## The dataset spec

`neural_trade.core.dataset_spec.DatasetSpec.from_config(config)` is the read-only view of the setup that
figures, backtests, run metadata and the leaderboard share: the instrument (`SYMBOL`, `QUOTE_CURRENCY`),
the bar size, the data file, the window and horizons in bars and in minutes, and the cost profile. Set the
instrument and quote currency so figure titles and currency labels name your market. Every run records the
dataset it used in its `meta.json` (`dataset`: file sha256, first and last timestamp, bar count) and the
leaderboard shows the fingerprint, bar size, instrument, window and horizons per row; rows trained on
different data are visible as such, and the paired comparator refuses to pair them.

## Bar size, window and horizons in wall-clock time

Lengths are configured in **minutes** and converted to bars by the bar size:

```bash
neural-trade train --csv my_5min_bars.csv --set RESAMPLE_MINUTES=5 --set WINDOW_MINUTES=60 \
    --set "HORIZON_MINUTES=[10, 15, 20]" --set "EXTENDED_TREND_MINUTES=[10, 15, 20]"
```

- `RESAMPLE_MINUTES` is the bar size of the model. A finer file is aggregated to it (open first, high max,
  low min, close last, volume sum). **It must equal the data's measured median bar spacing after
  aggregation**, otherwise the run stops with "the declared bar size is ... but the data's median bar
  spacing is ...". A 5-minute file with `RESAMPLE_MINUTES=1` is refused rather than annualised or read on
  the wrong clock.
- `WINDOW_MINUTES`, `HORIZON_MINUTES` and `EXTENDED_TREND_MINUTES` (the past-delta lags, one per horizon)
  override `LOOKBACK`, `HORIZON_STEPS` and `EXTENDED_TREND_PERIODS`. **A length that is not a whole number
  of bars is refused, never rounded** (`WINDOW_MINUTES=62` on 5-minute bars stops with an error). On 5-minute
  bars the example above gives a 12-bar window and horizons of 2, 3 and 4 bars.
- The horizons must be strictly ascending, and there must be three of them.
- Sharpe and Sortino are annualised from the bar size (`RESAMPLE_MINUTES`), so a 5-minute run is not
  annualised as a one-minute one.
- The block layout. `FOLD_LAYOUT: tscv` (the default) places `N_FOLDS` folds over the newest
  `MAX_SEQUENCE_COUNT` windows with `VAL_FRACTION` and `CAL_FRACTION` blocks. `FOLD_LAYOUT: timed` sets the
  train, validation, calibration and test blocks in minutes (`TRAIN_MINUTES`, `VAL_MINUTES`,
  `CAL_MINUTES`, `TEST_MINUTES`) and places the folds `FOLD_SPACING_DAYS` apart or at `FOLD_STARTS`; it is
  opt-in. Either way there is a purge gap between blocks.

**Indicator periods are still in bars.** `MA_SPANS`, `MACD_SETTINGS`, `RSI_PERIODS`, `BB_PERIODS` and the
`INDICATOR_FAMILIES` table are bar counts, and the textbook values are the ones tuned for a minute-bar
reference. Config validation warns when a period lies above the period ceiling of the window ("the clip will
move it"): on a 12-bar window almost every textbook period does. Choose starting periods that fit your
window; the network then moves them from there.

## The cost profile

Trading costs are an input, set per instrument (`FEE_BPS`, `HALF_SPREAD_BPS`, `SLIPPAGE_BPS`, basis points
per side) or per study (a scenario's `backtest:` block, which overrides the Config). **They default to 0**
(D-044). Backtests fill at the next bar's open and check stops against each bar's high and low. If you set a
cost, say so in every result that uses it: the leaderboard states the cost profile per row and marks a row scored at another
profile "not comparable", so it cannot win.

## Stability: what a new setup has to pass

VISION and D-026 require that a setup survive the stability harness before it is trusted:

```bash
neural-trade stability --profile tiny --dry-run --csv my_bars.csv   # plan the cases, train nothing
neural-trade stability --profile tiny --csv my_bars.csv             # CPU, about 30 s per cell
```

It stresses the setup with price-scale and volatility sweeps (x0.1 and x10), extreme inputs (a long constant
block, spikes, a level jump, prices x1e4 and x1e-4), fault injection (a NaN in the input, in a loss term and
in one gradient, each of which must stop the run and name the term) and a few named configurations. Each case
runs with 3 seeds in strict mode. The verdict per case and the loss term blamed for a failure are written to
`runs/stability/<id>/REPORT.md`, together with the sha256 of the pre-registered thresholds file used
(`configs/stability_thresholds.yaml`, v1, the default; `--thresholds v2` is the file for the reference
profile). A configuration that failed is added to `configs/stability_failing_regions.json`, and
`Config.validate` then refuses it, with the report that showed the failure. Exit code 1 means a case
failed, 2 that only non-verdict cells (a resource error, a crash) are left, 64 that the arguments were
refused.

**What the harness covers today.** It builds its cases from the Config defaults (the 60-bar window, the
10/15/20 horizons, one-minute bars) and your CSV's bars; it does not take your window, horizons or bar size.
So it checks that **your data** do not break the reference model, and it refuses a file whose bar size is not
one minute ("the declared bar size is 1 minutes ... but the data's median bar spacing is 5 minutes"). A
harness run on a different bar size or window is not available yet. Under thresholds v2 the tiny profile judges no variance
check (too few effective samples); the reference profile is a GPU run (NT-051) and has not been done.

## Before you trust a run on new data

1. `DatasetSpec` shows the bar size, window and horizons you meant.
2. The stability harness passes on your bars (above, with its limit).
3. The training health card shows no class collapse and no non-finite gradients
   ([Reading the figures](reading-figures.md)).
4. The evaluation report compares the model with its baselines (zero change, class prior, a logistic
   regression on trailing returns, constant variance). A model that does not beat them has not learned
   anything from your market, whatever its backtest says.
5. Rankings and verdicts use dev folds and judgement folds only ([Experiments](experiments.md)).
