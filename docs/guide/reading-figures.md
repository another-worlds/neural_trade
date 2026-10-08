# Reading the figures

Every figure the notebooks draw, in notebook order: what it shows, the noise band or reference line it
carries, and how to read it. The figures are Plotly (a few are matplotlib or HTML) and share one theme
(`visualization/theme.py`, D-014):

- **Horizons keep their colour in every figure:** h0 blue, h1 orange, h2 green. More horizons would
  extend the palette in a fixed order. Colours that are not horizons (copies of an indicator, status,
  eligibility) never reuse these three.
- **Dotted means training, solid means validation or test.** No other dash style means training. Grey
  dashed lines are references.
- **Status colours (good / bad) always come with a shape** (a star, a hatch, a cross), so no meaning
  rests on colour alone.
- **Every figure keeps every metric, horizon, band and table.** There is no simplified tier (D-014).

Most numbers have a noise level. Consecutive bars share target bars, so the "number of independent
samples" is about `N // horizon`, not `N` ([Concepts](concepts.md) "Noise"). A difference smaller than the
drawn band is not a finding. Anything from the newest fold (the test fold) is shown for information and
never ranks ([Concepts](concepts.md) "Folds").

A horizon whose served price delta is exactly zero (shrinkage beta = 0, D-007) or whose temperature fit
reached a bound (D-066, "no usable direction signal") prints `n/a` in the cells that would be a
meaningless constant. `n/a` means "not measured", not "0%".

Registered figures are built through the `Visualizations` registry
(`Visualizations.build("<key>", data, config, ...)`); the keys are listed in
`neural-trade registry list Visualizations` (27 keys). Each is named below in backticks. The
notebooks' own helpers are named too.

## 00 Data and splits

**`split_overview` / `split_overview_figure`**: the close price with the train, validation, calibration
and test blocks of the chosen fold shaded and named. The thin unshaded strips between blocks are the
purge gaps. Read it to see which period each block covers and that the test block is the newest; `split_table`
beside it gives each block's bars and label balance. **`matplotlib_splits`** is the older matplotlib walk-forward plot (registry entry): the price with the
scikit-learn `TimeSeriesSplit` train and test spans alternating in yellow and green. It does not show the
purge gaps or the four-way blocks; use `split_overview` for those.

## 01 Train and monitor, and 04 Diagnostics: the training record

These read a run's `metrics.jsonl` (the number of values per epoch depends on the model and the enabled terms).

**`training_dashboard`** (the default training figure; `plotly_interactive` is its registry alias and
`session.curves_figure` is the notebook's call): 12 panels. Total loss with the best epoch and the
**served** epoch marked (the served epoch is the one whose weights were evaluated and bundled, D-011);
what the validation loss is made of, term by term; direction-head and price-head MCC and the direction head's balanced
accuracy; Brier skill against the base rate; ECE and PIT-KS of the raw heads; up-rate bias; the physics
terms; learning rates; the gradient norm with its clip level. *Noise:* the 95% chance band of each metric
on the validation block, with `n_eff = validation samples // horizon`; withheld below 10 effective
samples. A metric inside its band has not left chance.

**`training_direction_detail`** (`direction_figure`): accuracy, sensitivity and specificity, F1, Brier and
mean P(up) per horizon, each against its no-skill reference and that reference's 95% range. Read the mean
P(up) panel for class collapse (a head that always says "up").

**`training_loss_terms`** (`loss_terms_figure`): every loss term per horizon on its own linear axis.
Read it for a term that dominates or goes non-finite. **`batch_figure`** (the batch-loss figure) is the
within-epoch loss and, when available, the running share of up calls per horizon, which is the
class-collapse signal while training.

**`training_health_html`** (the health card): the verdicts a person scans first (convergence, patience,
class collapse per horizon, non-finite gradients), each with its meaning in plain text, then the epoch
table at the served epoch (every loss term with its weight and share of the validation loss, every
direction metric per horizon with its reference and chance range). It is an HTML table, not a figure.

**`qbox_dashboard_html`**: the HTML block of the physics (T-perp / QBOX) loss terms for the per-epoch
dashboard; empty when every term is about zero.

## 01 and 04: the heads on a held-out block

All five analytics figures take the served predictions of the out-of-sample block (calibrated P(up), shrunk
delta, conformal intervals) and draw one column per horizon. Direction numbers use only moves outside the
`DIR_DEADBAND_BPS` band (5 bps).

**`direction_analytics`**: P(up) split by the realised class; the ROC drawn as *lift over chance* (TPR
minus FPR, whose area is AUC minus 0.5, so differences of a few hundredths are visible) for the direction
head and for the price head's Gaussian readout; reliability of raw and calibrated P(up) on 10 equal-count
bins; a scorecard against a constant 0.5 (n, up-rates, accuracy, MCC, Brier, ECE, sign agreement).
*Noise:* the ROC sits inside the 95% band a no-skill head stays in; AUC and accuracy use `n / h` effective
samples; reliability bins carry block-clustered intervals. A curve inside the band is no skill.

**`delta_analytics`**: a per-horizon table (served beta; RMSE and MAE of the raw head, the served delta and
predicting 0; skill against predicting 0 with HAC intervals); every sample, realised against predicted,
with out-of-range points as diamonds in shaded margins; the mean realised move per predicted decile
against the served line `y = beta x`; rolling correlation and rolling served skill with their no-skill
bands. With beta = 0 the served rows say `n/a` and the raw head is shown. Read the correlation row first:
a correlation inside its band is no relationship.

**`variance_analytics`**: five rows per horizon. RMS error per equal-count bin of the predicted sigma
against the diagonal (a calibrated sigma has RMS error equal to sigma) and against a free baseline, the
trailing realised volatility; the PIT histogram with the range a calibrated sigma would still show;
tail rates as a multiple of the Gaussian rate on a log axis; rolling 90% coverage with its chance range;
rolling interval width. *Noise:* block-bootstrap intervals and the long-run variance of the inside/outside
indicator. This is where the variance heads and the conformal intervals earn their keep.

**`confidence_analytics`**: accuracy by decile of `|P(up) - 0.5|` and of the strategies' variance
confidence `exp(-var / var_scale)`; selective accuracy (keep only the most decided x%); confusion
matrices with recall, precision, balanced accuracy and MCC. *References:* random calls with the model's
own up/down mix and the majority class; 95% moving-block bootstrap (80-bar blocks).

**`coherence_analytics`**: how the horizons agree: correlation of P(up) across horizons, the eight vote
patterns, the realised up-rate by number of up votes, sign agreement of the direction head with the price
head, `|delta|` ordering (raw against served), and the strategies' vote agreement. Served-delta cells are
`n/a` where beta = 0.

**`eval_report`** (`eval_report_figure`): reliability diagram, PIT histogram and interval coverage for one
prediction frame, the compact view behind the saved evaluation report.

## 04 Diagnostics: calibration

**`reliability`** (`reliability_figure`): raw, calibrated and saved P(up) on equal-count bins against the
diagonal. *Noise:* a Bartlett (HAC) 95% band per bin with lag equal to the horizon. A horizon in the
"no usable direction signal" state shows the note instead of a calibrated ECE.

**`interval_coverage`** (`coverage_over_time_figure`, returned by the calibration explorer's `figures`):
trailing coverage of the conformal interval against its target, with the 95% range one window shows by
chance when coverage is exactly on target. The band is centred on the target, so a line outside it is a
real miss.

The calibration explorer (`CalibrationExplorer`, the widget) refits temperature, interval scale, delta
shrinkage and miscoverage on the calibration block and shows the test result next to the saved pipeline.

## 02 Backtest and 03 Signals and trades

**`trading_dashboard`** (`explorer.dashboard`): seven panels on one time axis with one shared hover:
price with entries and exits (and, on a short view, holds and stop / target levels); P(up) per horizon and
the weighted P(up) with the entry lines; confidence and signal strength; predicted sigma at h1; net P&L
after costs; P&L before costs against buy-and-hold; drawdown. The decision triangle sits one bar before the
entry triangle (decide at the close, fill at the next open). Every number is for the bars shown.

**`trade_analytics`** (`explorer.trade_analytics`): nine panels, each with a one-line readout: return per
trade before and after costs, cost drag, net return by exit reason, cumulative P&L, holding time, best
and worst move while open (MFE / MAE), gross return by conviction, long against short, predicted against
realised move.

**`strategy_comparison`** (`explorer.compare_strategies`): every strategy's equity curve and return before
and after costs, each beside random entries with the same trade rate, holding time and size. *Reference:*
the random null. Costs default to 0 (D-044), so net equals gross unless a study sets costs; either way only the
rank against the null tells skill from chance. **`plotly_trading`** is the registry entry for one `BacktestResult`: price
with entries and exits, equity (net and gross) and the position held.

## 04 and 07: the learned indicators

**`indicator_evolution`**: base periods per family over the epochs on a log scale, the change of every
period from its start, and each period's correlation with the validation loss (epoch-to-epoch changes
against a noise band; the level correlation is a hollow marker). With `applied` given it adds each period's
median applied period as a diamond right of the last epoch.

**`indicator_family_periods`**: one panel per family whose periods are in the log, including the ten
families beyond the original four. A second parameter of one copy (MACD fast, slow, signal; Keltner period
and ATR period) is a second dash.

**`indicator_applied_periods`**: the base period against the periods the model applied across the block's
windows, one row per learned period, with the configured start and the lookback and clip bounds as
reference lines. A wide spread means the per-window shift is doing work.

**`discovered_indicators`** (notebook 07): the learned indicators on the price of one window, solid at the
period the model applied to that window and dashed at the textbook period. One row per family and one
column per copy; on price for averages and Bollinger bands, in their own panels for RSI and MACD. A strip
shows how each period moved over training and the spread of the applied period across windows, and a
table lists learned against textbook periods. Colour is the copy, not a horizon. `WINDOW` picks the
window: a number, `typical`, `longest`, `shortest` or `last`.

**`permutation_importance`** (notebook 07): one panel of loss importance per indicator instance, then one
panel per horizon of the direction-AUC drop when that instance is shuffled. *Noise:* 2.5 and 97.5
percentile whiskers of a block bootstrap. A bar whose whisker crosses zero is not distinguishable from an
irrelevant indicator.

## 05 Compare runs

**`runs_comparison`** (`runs_comparison_figure`): one panel per metric kind, one row per run, one dot per
run and horizon in the horizon's colour. *References:* the metric's no-skill or target value (AUC 0.5,
MCC 0, CRPSS 0, coverage 0.90), dashed; the best baseline of the run's own report as a grey open diamond.
*Noise:* a 95% interval on effective samples where one has a closed form. Runs scored on one block share
noise, so their intervals overlap by construction; the subtitle tells which runs share a block.

**`ablation_deltas`** (`ablation_deltas_figure`): per physics term and mode, the mean paired delta divided
by that comparison's decision threshold, so metrics of different units share one axis. A bar past the
dashed plus or minus 1 line passed its threshold; the grey zone is the neutral zone; crosses mark
guard-rail breaches. The verdict names (VALUE / HARMFUL / NEUTRAL / INCONCLUSIVE) are in the row headings.

## 06 Control panel

**`leaderboard`**: configurations ranked by the dev-fold net Sharpe after costs, rank 1 at the top, with
the table of every column below. The bar is the ranking column. The winner has a star; a disqualified or
"not comparable" row is hatched and labelled. Two whiskers where they exist: the *fold sd* (between dev-fold
means, the unit of inference) and the *seed sd* (within a fold). An open diamond just below each bar is the
**test-fold Sharpe, labelled "test, not used for ranking"**. Each label carries the cost profile of its
stored Sharpe. A row with no spread says "no spread (1 cell)".

The panel's **`board`** draws the leaderboard and refreshes while a sweep runs; its **`comparison`** draws
every logged metric of the selected rows per horizon (a dot per configuration, a whisker for the fold sd,
a thin grey line for the seed sd, the no-skill reference), a table of every score key, and the paired
verdict where a pre-registered comparison names the pair. Without one the per-fold differences are shown
as **exploratory**, not a verdict.

## 08 Long run and 09 Candidate run

**`show_progress`** (notebook 08) draws six panels: training and validation loss with the best epoch,
learning rates, seconds per epoch and per step, elapsed hours with the ETA projection, and validation CRPS
per horizon; then the per-epoch table, the training dashboard and loss terms, and the log tail.
**`show_results`** prints the key numbers per horizon, the backtest of the default strategy against
buy-and-hold, always-flat and random entries, and the evaluation report. Notebook 09 shows one saved run's
numbers, training record and backtest from its own saved files, on its dev (out-of-sample) block.

## Where each number comes from

The evaluation report (`eval_report_*.md` / `.json` in a run directory) holds the same numbers the figures
draw, per horizon, with the baselines (zero change, mean change, class prior, a logistic regression on
trailing returns, constant variance) and whether the model beats each. A figure and the report must
agree; if they do not, the figure is wrong.
