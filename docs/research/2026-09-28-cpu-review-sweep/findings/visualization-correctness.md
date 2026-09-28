# Area: visualization-correctness (7 findings)

Back to the [index](README.md) and the [report](../README.md).

<a id="viz-1"></a>

## VIZ-1: Variance figure crashes (ZeroDivisionError) when a conformal interval covers every sample of the block

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-33.
- **Files:** `src/neural_trade/visualization/analytics_variance.py`

**Description.** Row 3 of variance_analytics_figure adds the conformal miss ratio `miss / 0.10` to `tail_lo` without a positivity check (analytics_variance.py:512-514). The Wilson bounds next to it are filtered to v > 0 (:507-508). When a horizon's conformal coverage is 1.0, miss = 0, so `lo_r = min([0.5] + [0.85 * v ...])` becomes 0 (:581). `_log_ticks(lo_r, ...)` (:584) then computes `np.log10(hi / lo)` on Python floats (:302) and raises ZeroDivisionError, and `np.log10(lo_r)` at :586 would be -inf anyway. The whole figure is lost; notebooks 01 and 04 call it at scripts/notebooks/build.py:188 and :422. Related divisions in the row-5 titles, `c1 / c0` and `g1 / g0` at :386 and :389, raise the same error when the first half of the block has zero mean width. Empty halves when n < 2 also make these ratios meaningless.

**Failure scenario.** (a) A 3000-sample test frame whose h2 conformal intervals cover every sample (an over-wide conformal scale, e.g. fitted in a calmer regime) gives ZeroDivisionError at analytics_variance.py:302 and the notebook cell aborts. (b) Tiny blocks, such as quick-mode sweeps (NT-030) or MAX_SEQUENCE_COUNT slices: frames of 1, 2 or 7 samples crash the same way, and so does a frame whose moves all sit inside the intervals. Even with perfectly calibrated 90% intervals at h = 20 bars, a 130-sample block covers every sample in 6.7% of blocks (simulated) and a 333-sample block in 0.13%.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/probe_var_crash.py: 'coverage h2: 1.0' then '(1) CRASH ZeroDivisionError float division by zero 302'. Ran edge_harness.py: 'FAIL n=1 / n=2 / n=7 / all_in_deadband variance_analytics ZeroDivisionError: float division by zero @ [analytics_variance.py:584, analytics_variance.py:302]'; the other four analytics figures built on the same frames. Ran sim_fullcov.py (numpy random walk, interval +/-1.645 sigma sqrt(h)): 'n=130 h=20: P(... covers every sample) = 0.0673', 'n=333 h=20: 0.0013'. Code: analytics_variance.py:507-508 filter `if v > 0` for the Wilson bounds but :513-514 append miss/0.10 unconditionally.

**Backlog check (finder).** Not covered. NT-015 (PIT band per bin), NT-020 (layout polish of direction/delta/confidence) and NT-023 (theme) do not mention the variance figure's row-3 range or any crash. No crash of this figure is listed in STATUS, ROADMAP or the remediation archive.

**Fix sketch.** In row 3, treat a zero miss rate like the Wilson bounds: append miss/0.10 only when > 0, and draw a 0-miss conformal marker at the axis floor with the hover 'no miss (0 of n)'. Compute lo_r over positive values only, so it can never reach 0. Make the half-to-half width change n/a when a half is empty or its mean width is 0. Growth point 5 (visuals). Land before NT-030's quick mode starts re-executing notebooks on small blocks.

**Acceptance (proposed).** In tests/test_viz_variance.py: (1) a 3000-sample frame whose h2 intervals cover every sample builds, and row 3's y range is finite with a positive lower end. (2) Frames of 1, 2 and 7 samples build, or return one short 'too few samples' figure, without raising. (3) A frame whose first-half interval widths are 0 prints n/a for the half-to-half change. (4) The existing variance tests pass.

<a id="viz-2"></a>

## VIZ-2: Evidence for NT-002: strategy_comparison draws a zero-width '5-95%' random-null whisker from the engine's null

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-057.
- **Files:** `src/neural_trade/visualization/trade_analytics.py`, `src/neural_trade/strategy/backtest.py`, `src/neural_trade/notebook/backtest_ui.py`

**Description.** strategy_comparison_figure reads random_p05_total_return and random_p95_total_return and falls back to the mean when either is missing (trade_analytics.py:709-710). It then draws the whisker as p95 - mean and mean - p05 (:711-712), prints '5-95% {p05} to {p95}' in the hover (:716) and 'size 1.00' from a default (:715). The subtitle says 'whisker 5-95%; a bar inside the whisker is no better than chance' (:763-764). The engine's random_same_frequency (strategy/backtest.py:271-293) returns neither percentile and no size_frac. So the registry path Visualizations.build('strategy_comparison', {name: backtest(...)}) always draws a zero-width band and prints a false 5-95% range. The notebooks are not affected: BacktestExplorer replaces the null with matched_random_null (notebook/backtest_ui.py:335, :390), which records p05 and p95. NT-002 acceptance (2) adds the percentiles to random_same_frequency but does not name this consumer. Nothing guards the figure against a null without them (older backtest.json files, other callers), and no test goes through the registry path.

**Failure scenario.** A 3000-bar synthetic block, calibrated_quantile strategy, engine null with 30 seeds, drawn via the registry. The figure draws the null at -66.68% with whisker +0.0/-0.0, and the hover reads '5-95% -66.68% to -66.68%'. The same null from matched_random_null has p05 -68.43% and p95 -64.54%. The strategy's -66.77% lies inside the true band, so it is no better than chance, but it sits outside the drawn zero-width whisker. The leaderboard guard-rail 'beats random entries at the same frequency' (NT-031) is the same judgement.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/probe_null_whisker.py. Output: engine null keys ['hold_bars', 'n_seeds', 'percentile_sharpe_net', 'percentile_total_return', 'random_mean_sharpe_net', 'random_mean_total_return', 'trade_rate']. Then 'drawn null mean x: (-66.676,) whisker +: (0.0,) whisker -: (0.0,)', the hover '... mean -66.68%, 5-95% -66.68% to -66.68% (30 seeds) | strategy -66.77%: beats 53% of them', and 'matched_random_null: mean -66.68%, p05 -68.43%, p95 -64.54%'.

**Backlog check (finder).** NT-002 covers the engine null (size and the p05/p95 fields) but not this figure's silent fallback or the registry path. NT-021 (trading figures polish) does not mention strategy_comparison's null whisker.

**Fix sketch.** In strategy_comparison_figure, draw error_x only when both percentiles are present. Otherwise draw the marker alone and say 'spread not recorded' in the hover and the subtitle; print the size only when size_frac is present. With NT-002, random_same_frequency carries p05, p95 and size_frac, so the registry and notebook paths draw the same band. Growth point 2 (comparison). Land with NT-002, before NT-031's leaderboard guard-rail uses the same null.

**Acceptance (proposed).** (1) After NT-002, a test through Visualizations.build('strategy_comparison', {name: backtest(...)}) checks that the whisker equals the null's p05-p95 range, identical to the matched_random_null path on the same seeds. (2) A test with a null dict lacking random_p05/p95 checks that no error_x array is drawn and that the hover and subtitle say the spread was not recorded. (3) The existing trading tests pass.

<a id="viz-3"></a>

## VIZ-3: Evidence for NT-041: every overlap-adjusted interval and several time labels assume one sample per bar (WINDOW_STEP = 1)

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** short. **Placement:** addition to NT-041.
- **Files:** `src/neural_trade/visualization/stats.py`, `src/neural_trade/evaluation/frame.py`, `src/neural_trade/evaluation/report.py`, `src/neural_trade/visualization/analytics_variance.py`, `src/neural_trade/visualization/analytics_delta.py`, `src/neural_trade/visualization/data_overview.py`, `src/neural_trade/visualization/calibration_plots.py`, `src/neural_trade/visualization/trading_dashboard.py`

**Description.** stats.n_eff(n, steps) and every figure and table that calls it treat the horizon in BARS as the number of overlapping SAMPLES. That horizon comes from stats.horizon_steps (stats.py:90-94), read from PredictionFrame.horizon_steps (frame.py:30) or Config.HORIZON_STEPS. The evaluation report does the same (report.py:295, n_eff = len // horizon_steps). The data path supports a stride: make_sequences_with_extended_trends steps by WINDOW_STEP (data/windowing.py:70, :82), and the purged splits convert their gap to sequences for WINDOW_STEP > 1 (data/splits.py:30-38). With a stride s, neighbouring samples are s bars apart and share h - s bars, so the overlap is ceil(h / s) samples, not h. Time labels have the same hole. analytics_variance.py:336-338 turns the trailing window into hours with RESAMPLE_MINUTES only, while analytics_delta.py:326 uses RESAMPLE_MINUTES x WINDOW_STEP. data_overview.py:73 prints 'one-minute bars' and calibration_plots.py:292 prints '1-minute bars' whatever the bar size. trading_dashboard.py:600-603 labels bars with BacktestConfig.bar_minutes, which is never set (NT-040). NT-041's acceptance (5) scans only for 'BTC' and '$', so all of these pass it, and NT-041 never mentions the stride.

**Failure scenario.** Config(WINDOW_STEP=5) with horizons 10/15/20, for example a quick sweep that thins the windows. Every 95% interval and chance band is too wide: direction AUC and accuracy, the delta mean, the reliability fallbacks, Wilson intervals and the training chance bands. Simulated: stride 5, h = 20 gives a drawn half-width 2.2x the true one with 100% coverage, and stride 10 gives 3.2x. Real effects then read as noise. The variance row-4 heading says '500 samples (1-min bars, ~8 h)' for about 42 h of data.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/sim_stride.py (numpy random walk, stats.mean_ci with steps = horizon bars). Output: 'stride 1: drawn 95% half-width 1.004, true 1.013, ratio 0.99; coverage 0.947', 'stride 5: drawn 1.011, true 0.457, ratio 2.21; coverage 1.000', 'stride 10: drawn 1.012, true 0.319, ratio 3.17; coverage 1.000'. grep: analytics_variance.py:336 `bar_min = int(getattr(config, "RESAMPLE_MINUTES", 1) ...)` against analytics_delta.py:326 `RESAMPLE_MINUTES ... * WINDOW_STEP`; data_overview.py:73 'one-minute bars'; calibration_plots.py:292 '(1-minute bars, time order)'.

**Backlog check (finder).** NT-041 covers wall-clock units and labels (acceptance 5 tests 'BTC' and '$' only) but not the stride. NT-015 and NT-027 move the interval helpers into one module but keep the N / horizon-bars convention. Neither mentions WINDOW_STEP.

**Fix sketch.** Carry the sample stride in PredictionFrame, from the dataset spec, as window_step or as the horizon in samples. stats.horizon_steps and n_eff then return ceil(h_bars / stride) overlapping samples, and the report does the same through the one statistics module of NT-027. Build time labels from bar size x stride through one helper, and extend NT-041 (5)'s label test to 'one-minute' and '1-minute' literals. Growth point 4 (generality). Land during NT-041, together with NT-027's statistics module.

**Acceptance (proposed).** (1) Test: n_eff of a frame with window_step 5 and a 20-bar horizon is N / 4. A numpy random walk with stride 5 gives 95% +/- 2% coverage for stats.mean_ci. (2) evaluate() and the figures print the same n_eff for a strided frame (test). (3) The variance row-4 heading gives hours = w x bar minutes x stride / 60 (test). (4) A test finds no 'one-minute' or '1-minute' literal in the figure text of visualization/.

<a id="viz-4"></a>

## VIZ-4: Evidence for NT-028: reachability map of the legacy figure modules (four stale, two live)

- **Severity:** P2. **Status:** not verified. **Type:** hygiene. **CPU cost:** short. **Placement:** addition to NT-028.
- **Files:** `src/neural_trade/visualization/aliases.py`, `src/neural_trade/visualization/plotly_training.py`, `src/neural_trade/visualization/qbox_dashboard.py`, `src/neural_trade/visualization/matplotlib_splits.py`, `src/neural_trade/visualization/plotly_trading.py`, `src/neural_trade/registries/visualizations.py`, `src/neural_trade/data/processor.py`, `src/neural_trade/compat.py`, `src/neural_trade/training/callbacks.py`, `tests/registries/test_visualizations.py`

**Description.** NT-028 names 'figure modules without a production caller' but gives no inventory. Searched src, scripts, scripts/notebooks/build.py, notebooks, tests, configs and docs.
(1) STALE aliases.py (293 lines, 5% covered): imported only by plotly_training.py:18 (used at :179 and :270 inside make_interactive_plot_callback) and the compat re-export at compat.py:34.
(2) STALE plotly_training.training_curves_figure (:29) and make_interactive_plot_callback (:127): reached only from training/callbacks.py:394-398 (the 'interactive_plot' callback, registries/callbacks.py:60), which is not in CALLBACKS (core/config.py:205, configs/default.yaml:128) and is used by no notebook or script, and from compat.py:35.
(3) STALE qbox_dashboard.py: `_qbox_dashboard_html` is used by that callback (plotly_training.py:19, :457) and compat.py:36. Its registry key 'qbox_dashboard_html' (registries/visualizations.py:37) is used only by tests/registries/test_visualizations.py:17 and :35-37.
(4) STALE matplotlib_splits.py: reached via DataProcessor.plot_splits (data/processor.py:108-111), which has no caller, and via the registry key 'matplotlib_splits' (registries/visualizations.py:38), which only that test uses. It needs a sklearn TimeSeriesSplit `tscv` that today's split code never produces, draws a Train/Test alternation that misstates the purged train|val|cal|test split, hard-codes 'BTC', and is the data -> visualization import that NT-027 lists (processor.py:109). Notebook 00 draws data_overview.split_overview_figure instead.
(5) LIVE plotly_training.plotly_interactive (:114): it is the Config.VISUALIZATION default (core/config.py:208, configs/default.yaml:129), is validated at registry build (registries/__init__.py:56), is the registry's discovery module (registries/visualizations.py:19) and is tested at tests/test_viz_training.py:79.
(6) LIVE plotly_trading.py: `neural-trade backtest --plot` builds it at cli.py:132-134.

**Failure scenario.** A deletion by module name breaks live paths. Deleting plotly_training.py removes the default VISUALIZATION component, and build_all validation fails. Deleting plotly_trading.py breaks `neural-trade backtest --plot`. Conversely, matplotlib_splits cannot run on today's split output at all: passing make_purged_splits' FoldIndices raises AttributeError.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/probe_legacy.py. Output: 'plotly_trading OK calibrated_quantile 86 | calibrated_quantile: 86 trades, net return -19.25%, ...', 'plotly_trading OK always_flat 0', 'matplotlib_splits with today's FoldIndices FAIL: AttributeError 'FoldIndices' object has no attribute 'split''. The grep results are listed in the description, e.g. `grep -rn interactive_plot notebooks scripts configs tests` returns only configs/default.yaml:128 (not listed) and no uses. `grep -rn 'plot_splits\|TimeSeriesSplit\|tscv' src scripts` finds no producer of `tscv` outside data/splits.py:42, which is internal.

**Backlog check (finder).** NT-028 lists the interactive_plot callback and compat.py, and 'figure modules without a production caller' in general. This finding adds the exact reachability, the two live modules that must stay and the matplotlib_splits breakage. NT-027 lists the processor.py:109 import.

**Fix sketch.** Under D-029, remove aliases.py, training_curves_figure and make_interactive_plot_callback (with the interactive_plot callback and its registry entry), qbox_dashboard.py and its registry key, and matplotlib_splits.py with DataProcessor.plot_splits and its registry key. Remove their compat re-exports and the test lines that exist only for them. Keep plotly_interactive, moved into training_dashboard.py with discovery_modules pointed there, and keep plotly_trading. Mention in the D-029 report that the `neural-trade registry` listing loses three keys. Growth point 1 (structure). Land during NT-028, after NT-027's layering test exists (it then also passes on data -> visualization).

**Acceptance (proposed).** (1) The implementer's D-029 report gives the searches above for each removed item. (2) A test checks that Visualizations.list() still contains plotly_interactive, plotly_trading and every key that scripts/notebooks/build.py builds. (3) A test runs `neural-trade backtest --plot` on the bundled CSV and a fixture bundle and checks that backtest.html is written, or runs cli.cmd_backtest with a stub predictor. (4) The layering test finds no data -> visualization import. (5) The fast suite passes.

<a id="viz-5"></a>

## VIZ-5: Confidence and coherence figures label an uncalibrated P(up) 'calibrated'

- **Severity:** P3. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-020.
- **Files:** `src/neural_trade/visualization/analytics_confidence.py`

**Description.** Both subtitles hard-code 'calibrated P(up)': confidence_analytics_figure at analytics_confidence.py:554 and coherence_analytics_figure at :853. Both draw frame.prob(h, True), which is the RAW head when the frame has no fitted calibration: direction_prob_calibrated is None, or it equals the raw head when the Predictor serves calibrated=False. The direction figure in the same notebook checks this with analytics_direction.calibration_applied (:168-178) and says 'no calibration applied', so the two figures disagree about what is shown.

**Failure scenario.** A PredictionFrame with direction_prob_calibrated=None (a run without a fitted pipeline, or served with calibrated=False). The direction figure says 'raw P(up) (no calibration applied)', while the confidence and coherence figures on the same frame say 'calibrated P(up)'.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/probe_labels.py. Output: 'direction_analytics uncalibrated frame; title mentions 'calibrated P(up)': False | mentions 'no calibration'/'raw P(up)': True', 'confidence_analytics ... 'calibrated P(up)': True | ... False', 'coherence_analytics ... True | ... False'.

**Backlog check (finder).** NT-020 lists confidence-figure polish (contrast, swatches, majority label, clipping) but not this label. NT-021 (3) covers the trading dashboard's P(up) title only.

**Fix sketch.** Use calibration_applied(frame) from analytics_direction, or move it to the shared module of NT-027, to write 'calibrated P(up)' or 'raw P(up) (no calibration applied)' in both subtitles. Growth point 5 (visuals). Land with the NT-020 polish batch (after).

**Acceptance (proposed).** Test: with direction_prob_calibrated=None, neither subtitle contains 'calibrated P(up)' and both say 'raw P(up)' with 'no calibration applied'. With a fitted calibration the text is unchanged (test on viz_frame).

<a id="viz-6"></a>

## VIZ-6: Correction to NT-020: the frame.meta['delta_raw'] fallback is missing in the delta and variance figures, not only in analytics_tables._raw_or

- **Severity:** P3. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-020.
- **Files:** `src/neural_trade/visualization/analytics_delta.py`, `src/neural_trade/visualization/analytics_variance.py`, `src/neural_trade/visualization/analytics_tables.py`

**Description.** Two figures already find the raw price heads the way evaluate() does (report._resolve_raw, report.py:911-919): the direction figure (analytics_direction.py:199-208, :343) and the coherence figure (analytics_confidence.py:625-630). They use the argument, else frame.meta['delta_raw'] (which PredictionFrame.from_result sets, frame.py:93), else served / beta. delta_analytics_figure uses only its argument (`served_known = raw_delta is not None`, analytics_delta.py:320-321). variance_analytics_figure does the same (:228, :442, :599); its _zero_beta_horizons reads meta['delta_scale'] but not meta['delta_raw']. NT-020 (4) asks for the fallback only in analytics_tables._raw_or (:107-108), which delta_quality_table, magnitude_ordering_table and alignment_table use. The notebooks pass raw_delta= explicitly (build.py:184, :188, :421-422), so today only registry and other callers are affected. The experiment engine and the control panel (NT-026, NT-034) will call the registry.

**Failure scenario.** Visualizations.build('delta_analytics', frame, cfg) on a from_result-style frame with beta = 0 on every horizon. The raw heads are in frame.meta, yet the figure says 'raw heads not supplied' and draws only 'not drawn: the served delta is 0' strips in rows 2-5. The direction and coherence figures on the same call read the raw heads. The variance figure drops its 'RMS around the raw head' series.

**Evidence (finder, reproduced).** Ran agents/visualization-correctness/probe_labels.py. Output: 'direction_analytics beta=0 frame with meta['delta_raw']: reads the raw heads: True', 'coherence_analytics ... True', 'delta_analytics ... reads the raw heads: False; says 'raw heads not supplied': True', 'variance_analytics ... reads the raw heads: False ... | RMS-around-raw-head trace: False'.

**Backlog check (finder).** NT-020 (4) covers analytics_tables._raw_or only. NT-027 (4) says 'figures only draw' but does not name the raw-head resolution.

**Fix sketch.** Route every figure and table through one resolver (report._resolve_raw, moved into NT-027's shared module) for raw_delta and delta_scale, and widen NT-020 (4)'s acceptance to the two figures. Growth point 1 (structure, the one resolver during NT-027) or growth point 5 with the NT-020 batch.

**Acceptance (proposed).** Test: for a frame with meta['delta_raw'] and meta['delta_scale'] (beta = 0 on all horizons, and beta = 0 on h1 only), delta_analytics, variance_analytics, delta_quality_table, magnitude_ordering_table and alignment_table built without raw_delta= give the same numbers and texts as with raw_delta=frame.meta['delta_raw'].

<a id="viz-7"></a>

## VIZ-7: Evidence for NT-042: file:line inventory of the three-horizon assumptions in visualization/, and the hidden 'h1 is the primary horizon' sub-case

- **Severity:** P2. **Status:** not verified. **Type:** design-risk. **CPU cost:** short. **Placement:** addition to NT-042.
- **Files:** `src/neural_trade/visualization/theme.py`, `src/neural_trade/visualization/stats.py`, `src/neural_trade/visualization/analytics_common.py`, `src/neural_trade/visualization/analytics_direction.py`, `src/neural_trade/visualization/analytics_delta.py`, `src/neural_trade/visualization/analytics_confidence.py`, `src/neural_trade/visualization/analytics_tables.py`, `src/neural_trade/visualization/training_dashboard.py`, `src/neural_trade/visualization/trade_analytics.py`, `src/neural_trade/visualization/trading_dashboard.py`, `src/neural_trade/strategy/strategies.py`

**Description.** NT-042 cites only theme.py:28-30 for the figures. The inventory in visualization/:
(1) Horizon names: theme.py:29-30 (HORIZONS, HORIZON_COLORS = SERIES[:3]); stats.py:93 and theme.horizon_label (:115-122) index into the fixed tuple, so 'h3' raises.
(2) Layouts of exactly three columns: analytics_common.py:85 (_grid cols=3, used by the variance, confidence and coherence figures); analytics_direction.py:423-425 and :675 (a 4-entry columnwidth); analytics_delta.py:378-379 and :808-813 (range(3), 4-entry columnwidth); training_dashboard.py:1632.
(3) Cross-horizon statistics written for three horizons: the 2^3 vote patterns at analytics_confidence.py:40, :620-622, :643-645 and :722; the 3x3 correlation at :711; 'all 3' at :776 and analytics_tables.py:380; the pairs (0,1), (1,2) and the full chain at analytics_confidence.py:649, analytics_delta.py:302 and :823, and analytics_tables.py:313; extended_h0..2 at training_dashboard.py:418.
(4) Hidden sub-case, 'index 1 is the primary horizon': trade_analytics.py:193, :200-202, :205, :446 and :462; trading_dashboard.py:47-50, :277, :326, :468 and :490-494. These mirror strategy/strategies.py:85, :98, :145, :150, :162, :211 and :271-272, and the report's coherence_primary (report.py:263). With N = 2, index 1 is the LONGEST horizon; with N = 4 it is the second shortest; with N = 1 it raises IndexError. NT-042 does not define which horizon the strategies and these panels read. The vote-pattern panel also grows as 2^N (16 bars at N = 4).

**Failure scenario.** A Config with four horizons (NT-042's N = 4 test): stats.horizon_steps(config, 'h3') and theme.horizon_label('h3', config) raise ValueError, so every figure fails. With N = 2 the strategies and the trade figures silently switch their 'h1' reading to the longest horizon, which changes backtests and their labels without an error.

**Evidence (finder, reproduced).** Ran a probe. `S.horizon_steps(c, 'h3')` with HORIZON_STEPS=[10, 15, 20, 30] gives 'ValueError tuple.index(x): x not in tuple'; theme.HORIZONS is ('h0', 'h1', 'h2'); `theme.horizon_label('h3')` gives the same ValueError. The file:line list above comes from grep over visualization/ and strategy/strategies.py (e.g. strategies.py:85 `p, conf = s.p[t, 1], 1.0 / (1.0 + s.var_scaled[t, 1])`).

**Backlog check (finder).** NT-042 covers N horizons in 'the figures' and 'the strategies' signal frame' but names only theme.py:28-30 in visualization/. It does not mention strategy/strategies.py or the primary-horizon choice.

**Fix sketch.** Add a primary-horizon setting to the horizon or dataset spec, defaulting to the middle horizon, which is h1 at N = 3 and reproduces today's numbers. Strategies, SignalFrame, the report's coherence_primary and the trading figures read it. Figures loop over the configured horizons and the neighbouring pairs (i, i+1). Beyond N = 3, the vote-pattern panel becomes the 'number of up votes' view. Growth point 4 (generality). Decide the primary horizon before NT-042's implementation, then land during NT-042.

**Acceptance (proposed).** (1) NT-042 acceptance names a primary-horizon setting, and at N = 3 `scripts/golden_run.py verify` passes. (2) Tests build every registered figure and table for synthetic N = 2 and N = 4 frames without an exception or an empty panel. (3) A grep test finds no literal ('h0', 'h1', 'h2') tuple and no `[:, 1]` / `[t, 1]` primary-horizon index in visualization/ or strategy/strategies.py outside the horizon spec.

