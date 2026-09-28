# Area: notebooks-tooling (10 findings)

Back to the [index](README.md) and the [report](../README.md).

<a id="nb-1"></a>

## NB-1: Correction to NT-041 (and NT-026 (5)): notebooks 02-04 rebuild 'the run's TEST block' from the bundled CSV, whatever file the run trained on, so the block can be inside the run's training block

- **Severity:** P1. **Status:** not verified. **Type:** bug. **CPU cost:** short. **Placement:** addition to NT-041.
- **Files:** `src/neural_trade/notebook/backtest_ui.py`, `src/neural_trade/notebook/calibration_ui.py`, `scripts/notebooks/build.py`, `src/neural_trade/experiments/run_context.py`

**Description.** load_run_blocks copies the bundle's config and, whenever csv_path is given, overrides CSV_PATH (backtest_ui.py:247-250) before split_arrays rebuilds the purged split. Notebooks 02, 03 and 04 always pass their CSV_PATH parameter, which defaults to '../binance_btcusdt_1min_ccxt.csv' (build.py:253/269, 323/338, 375/393). CalibrationExplorer.from_run goes through the same loader. The override exists because a run's CSV_PATH is relative to the working directory it trained in, so a CLI run's path does not resolve from notebooks/. No run records the data file it used: run_context.py:44-62 writes only run_id, seed, tags and created_utc, and there is no check at all. NT-026 (5) and NT-041 (3) add a dataset fingerprint to meta.json, but no acceptance criterion makes the notebook loader use or verify it. After NT-041 (7) (walk-forward folds on the long local file), 02-04 would keep scoring every run on the bundled 30-day file and label the result as that run's out-of-sample test block.

**Failure scenario.** Train a run on another file: notebook 01 with CSV_PATH set to the long local file, or `neural-trade train --csv Bitcoin_BTCUSDT.csv`. Then execute 02-04 with their defaults. The explorers cut the bundled file with the run's FOLD_INDEX, N_FOLDS and MAX_SEQUENCE_COUNT and score the model on 2025-11-05..11-10. That is not the run's test block. When the bundled period falls inside the run's training block, 02's backtest, 04's 'TEST block, re-scored' report and the calibration explorer show in-sample numbers labelled out of sample. Notebook 04's report then disagrees with the run's own eval_report_test.json, with no warning.

**Evidence (finder, reproduced).** Reproduced with split_arrays only (no model): scratchpad/agents/notebooks-tooling/repro_csv_override.py, run from the scratch copy of the code. Setup: a 'long' file made of the bundled 30 days plus 20 later synthetic days, and configs/default.yaml. The run's own split: train 2025-10-24 07:10 .. 2025-11-19 01:57, val 11-19..11-21, cal 11-21..11-24, test 2025-11-24 03:10 .. 11-30 07:09. What notebooks 02-04 rebuild as its test block: 2025-11-05 06:34 .. 2025-11-10 07:09. Output: "notebook 'test' block inside the run's TRAIN block: True".

**Backlog check (finder).** Checked NT-026 (5), NT-041 (3)/(7), NT-010, NT-018 and NT-028. The fingerprint is recorded but never verified by the loaders; no item mentions load_run_blocks or the CSV_PATH parameter of notebooks 02-04. New as a correction to NT-041 (and NT-026 (5)).

**Fix sketch.** Record the data file (absolute path plus NT-026's sha256, first and last timestamp and bar count) in meta.json at RunContext.create. By default, load_run_blocks loads the run's own recorded file (resolved independently of the working directory). When csv_path is given, it refuses with a clear error that names both fingerprints unless they match. build.py 02-04: CSV_PATH = None, meaning the run's own data. Older runs without a fingerprint load only with an explicit csv_path and print that the data could not be verified. Growth point 4 (dataset spec). Land together with NT-026 (5), before NT-041 (7) produces runs on other files.

**Acceptance (proposed).** (1) load_run_blocks (or the data-resolution helper factored out of it) raises an error naming both fingerprints when csv_path's fingerprint differs from the run's recorded one. Test with a fake run dir and no model. (2) With csv_path=None it loads the run's recorded file, also from a different working directory (test). (3) The CSV_PATH default of notebooks 02-04 is None, and build.py --check passes after rebuilding. (4) A run without a recorded fingerprint loads only with an explicit csv_path, and a note says the data was not verified (test).

<a id="nb-2"></a>

## NB-2: Correction to NT-034 (5): check.py and the notebook tests see neither the package's warnings nor anything saved in widget state, so a notebook 01 run whose calibration failed passes the notebook routine

- **Severity:** P1. **Status:** not verified. **Type:** infra. **CPU cost:** short. **Placement:** addition to NT-034.
- **Files:** `scripts/notebooks/check.py`, `src/neural_trade/core/logging.py`, `src/neural_trade/notebook/session.py`, `src/neural_trade/notebook/backtest_ui.py`, `tests/test_notebooks_thin.py`, `tests/test_notebook_tooling.py`

**Description.** check.py has three blind spots.
(a) The package logger's only handler writes every level, WARNING and ERROR included, to the current stdout with the format '%(message)s' (core/logging.py:20-33, 46-50). check.py flags only 'stderr' streams (check.py:104-114).
(b) TrainingSession routes every record of the training thread into the widget's Log Output via append_stdout and filters those records out of every other handler (session.py:31-53, 117-127, _log). train_and_evaluate's warnings therefore exist only in metadata.widgets: 'CalibrationPipeline fit FAILED (continuing without calibration)' (trainer.py:485), '[calib] Calibration pass failed — restored configured lambdas' (lambda_calibration.py:322) and 'No GPU is visible' (trainer.py:95).
(c) check.py and test_parameters_come_first_and_no_saved_output_is_an_error (test_notebooks_thin.py:40-46) read only cell outputs, never metadata.widgets. An exception inside an ipywidgets Output context is swallowed into the widget's state (Output.__exit__ returns True under IPython). BacktestExplorer.click_run, which notebook 02 calls in a cell, writes a backtest error into the status HTML and returns normally (backtest_ui.py:531-539, 566-569).
NT-034 (5) accepts the control panel on 'check.py finds no error, stderr or empty panel'. A widget-based panel satisfies that criterion vacuously.

**Failure scenario.** Notebook 01 is executed and the CalibrationPipeline fit raises. Training continues uncalibrated, cell 9 handles calibration_pipeline None, execute.py prints OK and check.py prints 'all clean'. The only trace is one line in the saved widget log, and the notebooks are committed as a real run. In notebook 02, an invalid knob makes click_run write the real error into the status HTML. The next cell then fails with AttributeError: 'NoneType' object has no attribute 'config', and that is the error check.py reports.

**Evidence (finder, reproduced).** probe_warnings.py executes a notebook with nbclient, as execute.py does. It runs TrainingSession with a stubbed trainer (no TF, no training) that logs the trainer's two real warning texts in its thread, plus a package WARNING in the main thread. Result: 'check.ok = True stderr = [] errors = []'. Cell 2 has 'stream stdout: the artifacts carry no calibration-split var_scale...'. The widget log in metadata.widgets holds 'CalibrationPipeline fit FAILED ...' and '[calib] Calibration pass failed ...', and the status reads 'finished'.
probe_check.py: a cell that runs 'with out: raise ValueError' gives check.ok=True, errors_in_widget_state=[('ValueError', 'boom inside the panel')].
Explorer probe: click_run with entry_quantile=1.5 leaves the status HTML 'Quantiles must be in the range [0, 1]', and the next call raises AttributeError.
Prototype fix in the scratch copy: WARNING and above go to stderr in the package handler; the session's thread warnings go through append_stderr; check.py also reads the OutputModel outputs in metadata.widgets. With it, the six committed notebooks are still 'all clean' (38 figures, 6 of them from widget state), the three probes FAIL as they should, and 78 notebook/config/telemetry/hygiene tests pass.

**Backlog check (finder).** NT-025 covers only the 5 MB limit, NT-018 covers only the widget-state size and reopen policy, and NT-034 (5) relies on check.py. No item mentions warnings going to stdout or errors kept in widget state. New as a correction to NT-034 (5).

**Fix sketch.** As prototyped:
(1) core/logging: records at WARNING and above go to the current stderr (the CLI already writes to stderr).
(2) TrainingSession._ThreadLogHandler writes WARNING and above with append_stderr, and wait() re-emits the thread's WARNING-and-above records to the calling cell's stderr.
(3) check.py reads the OutputModel outputs in metadata.widgets (errors, stderr, figures and their empty panels) and reports them as [widget].
(4) click_run re-raises after writing the status; the button callback keeps catching.
(5) The saved-error test in test_notebooks_thin.py also reads widget state.
Growth points 1 (the notebooks as the per-step gate) and 2 (NT-034 depends on it). Land before MVP-1's first module move, at the latest before NT-034.

**Acceptance (proposed).** (1) A synthetic notebook executed with nbclient, in which a stubbed TrainingSession's thread logs a WARNING, fails check.py, which prints the warning's text (test). (2) A notebook with an exception inside an ipywidgets Output context fails check.py (test). (3) A package logger.warning in a cell produces a stderr stream (test). (4) click_run with an invalid knob raises in the cell (test). (5) The committed notebooks still pass check.py. (6) NT-034 (5) says that widget-state outputs are checked.

<a id="nb-3"></a>

## NB-3: check.py passes a notebook whose figure vanished or whose panel holds no data: notebook 05 with no scored runs shows 1 of its 2 figures and is 'all clean'

- **Severity:** P2. **Status:** not verified. **Type:** test-gap. **CPU cost:** short. **Placement:** new item CPU-32.
- **Files:** `scripts/notebooks/check.py`, `src/neural_trade/visualization/theme.py`, `scripts/notebooks/build.py`

**Description.** check.py inspects only the figures that were saved (check.py:115-126). theme.empty_panels flags only an axis that holds no trace (theme.py:131-144). Two kinds of broken output therefore pass. First, a cell whose figure was replaced by a fallback print (build.py:484 'no scored runs yet', :488-493 'no ablation report') or was never produced. Second, a panel whose traces carry no finite point (all-NaN or zero-length arrays). NT-043 (4) and the figure tests use empty_panels as their 'no empty panel' criterion. The coming module moves (NT-027, NT-028) are exactly the kind of change that makes figures disappear silently.

**Failure scenario.** Notebook 05 is executed in a fresh worktree, or on any machine without scored runs (runs/ is untracked, NT-010). The run-comparison figure is replaced by 'no scored runs yet', and check.py prints 'figures=1 ... OK' and 'all clean'. Committed as it is, the notebook loses its main figure without any gate failing. Likewise, a refactor that draws a metric as an all-NaN trace passes check.py and every empty_panels-based test.

**Evidence (finder, reproduced).** Executed notebooks 00 and 05 with nbclient into the scratch dir (2.8 s and 1.7 s). exec_05 prints '0 scored runs' and 'no scored runs yet'. check.py on it: 'exec_05_compare_runs.ipynb ... figures=1 errors=0 ... OK' and '2 notebook(s), 2 figures: all clean' (the committed 05 holds 2 figures).
probe_check.py: all_nan_panel gives check.ok=True, traces=[2], empty=[[]]; empty_array_panel gives check.ok=True.
The rule 'every executed cell with N .show() calls holds at least N saved plotly figures' (show_rule.py) flags only exec_05 cell 3 and nothing in the six committed notebooks.
A scan of every saved figure for panels with no finite point (nan_panels.py) finds only legend-only dummy traces and the labelled 'nothing to draw' strip of 'Price heads (delta)' (analytics_delta.py:653).
The .show() rule was prototyped in the scratch copy of check.py: the committed notebooks still pass, and exec_05 fails with '1 .show() call(s) but 0 saved figure(s)'.

**Backlog check (finder).** Checked NT-025 (size), NT-018, NT-043 (4) (uses empty_panels), NT-045 (figure homes) and the NT-001 correction (the plotly 5 load failure). None covers vanished figures or data-less panels. New.

**Fix sketch.** check.py: (1) apply the .show()-count rule (prototyped; the committed notebooks pass). (2) Treat a panel whose non-legend traces have no finite y as empty unless it carries a note (the note_on_empty / 'nothing to draw' convention), and extend theme.empty_panels the same way so the figure tests inherit it. (3) Optionally compare a per-notebook manifest of figure titles with the last commit. Growth points 1 (the notebooks work at every step) and 5. Land before NT-027 and NT-028 start moving figure code.

**Acceptance (proposed).** (1) A synthetic executed notebook whose cell calls .show() but saves no figure fails check.py (test). (2) A panel with only all-NaN or zero-length traces and no note fails both check.py and theme.empty_panels, while legend-only traces and a panel with a note pass (tests). (3) The committed notebooks pass check.py. (4) Notebook 05 executed with an empty RUNS_GLOB fails check.py.

<a id="nb-4"></a>

## NB-4: A run trained with an epochs argument records Config.EPOCHS as its planned epoch count: notebook 04 says 'stopped before the planned epochs' where notebook 01 says 5 / 5

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** short. **Placement:** addition to NT-026.
- **Files:** `src/neural_trade/training/trainer.py`, `src/neural_trade/experiments/run_context.py`, `src/neural_trade/training/artifacts.py`, `src/neural_trade/visualization/analytics_tables.py`, `src/neural_trade/visualization/training_dashboard.py`, `src/neural_trade/notebook/session.py`

**Description.** train_and_evaluate trains actual_epochs = epochs or cfg.EPOCHS (trainer.py:354) but never records it. config.yaml was already written at RunContext.create (run_context.py:56), and the bundle meta holds only epochs_run (artifacts.py:52). The readers take cfg.EPOCHS as the plan. run_settings_table writes 'stopped before the planned epochs' when epochs_run < EPOCHS (analytics_tables.py:613-615), and the health tile shows 'n / EPOCHS' (training_dashboard.py:603-604). Notebook 01 hides the problem because it overrides EPOCHS for display only (session.py:287-291). Paths that pass an epochs argument: notebook 01's EPOCHS parameter (build.py:89, 116), `neural-trade train --epochs` (cli.py:64) and TrainingSession(epochs=). An engine quick mode that shortens runs the same way would inherit the problem.

**Failure scenario.** Notebook 01 is run with EPOCHS = 5, its documented parameter, and trains 5 of 5 epochs. Notebook 04 on that run shows 'EPOCHS config 20, effective 5', the note 'stopped before the planned epochs' and the tile '5 / 20', although nothing stopped the run. The same run shows '5 / 5' in 01.

**Evidence (finder, reproduced).** repro_epochs.py builds a run dir as the trainer writes it after epochs=5: config.yaml has EPOCHS 20, artifacts/meta.json has epochs_run 5 and weights_epoch 5. run_settings_table gives 'run EPOCHS 20 5 stopped before the planned epochs'. The training_health epochs tile gives '5 / 20' as notebook 04 reads it, and '5 / 5' with notebook 01's display override.

**Backlog check (finder).** Searched BACKLOG, ROADMAP, STATUS and the archive for 'planned epoch', epochs_run and --epochs: no match. NT-026 (5) records the dataset and setup but not the epochs. New.

**Fix sketch.** Record the planned epoch count with the run. The trainer sets cfg.EPOCHS = actual_epochs before training, and config.yaml, status.json and the artifacts meta carry epochs_planned. Readers use it, with cfg.EPOCHS as the fallback for older runs. Growth point 1 (the run store records what ran) and point 2 (leaderboard rows). Land with NT-026's run records.

**Acceptance (proposed).** (1) A run trained with epochs=k and Config.EPOCHS different from k records k as planned (test on the written files, with a stubbed or 1-epoch CPU TrainResult). (2) run_settings_table and training_health on such a run show 'k / k' and no 'stopped' note (test with a fake run dir, as in the repro). (3) A run cut short by early stopping still gets the note (test).

<a id="nb-5"></a>

## NB-5: The backtest and calibration explorers offer knob and calibration choices only on the TEST block, with no dev block, against CLAUDE.md 'Evidence'; the control panel would inherit this

- **Severity:** P2. **Status:** not verified. **Type:** design-risk. **CPU cost:** short. **Placement:** addition to NT-034, NT-018.
- **Files:** `src/neural_trade/notebook/backtest_ui.py`, `src/neural_trade/notebook/calibration_ui.py`, `scripts/notebooks/build.py`, `docs/STATUS.md`

**Description.** BacktestExplorer builds its signals and bars from the test block only (backtest_ui.py:3-5, 301-309, 332). Its widget is a knob-tuning interface: strategy knobs, costs and a Run button (:491-564). compare_strategies compares every strategy on that same block (notebook 02 cell 6, build.py:281-291). CalibrationExplorer fits on cal and scores only on test (calibration_ui.py:127-141); its button reads 'Refit on cal, score on test' (:231; build.py:439-443). Neither takes a block argument, so every interactive choice notebooks 02 and 04 offer is judged on test numbers. CLAUDE.md 'Evidence' forbids this ('no choice uses test-block numbers, the notebooks' and the leaderboard's test columns included'), and so does VISION ('Choices never use test data'). The saved outputs are already read as a ranking: STATUS.md:36 calls calibrated_quantile the 'best strategy' based on notebook 02's test table. By the yardstick, net Sharpe after costs, it ranks below liberal and below the two strategies that made no trade.

**Failure scenario.** A user tunes liberal's min_agreement in notebook 02, or the calibration explorer's miscoverage or interval scale in notebook 04, until the test net return or coverage looks best. They then carry that setting into a config or a sweep's search space. The test block has now been used for the choice, and its numbers are no longer an honest verdict. NT-034's panel would reuse these explorers for its comparisons.

**Evidence (finder, code reading).** Code reading at the lines above. Saved notebook 02, cell 6 (comparison table), Sharpe after costs: calibrated_quantile -84.23, liberal -68.27, enhanced_multi_horizon 0.00 (0 trades), threshold_spike 0.00 (0 trades), buy_and_hold +6.65. Net return: -29.86% for calibrated_quantile against -14.25% for liberal. STATUS.md:36: 'best strategy `calibrated_quantile` +2.3% gross, -29.9% net'.

**Backlog check (finder).** NT-031 labels the leaderboard's test columns, and the owner accepted the pick-by-eye risk for the leaderboard only. NT-018 is UX, and NT-034 covers sweeps and the leaderboard. No item gives the explorers a dev block. New.

**Fix sketch.** Give load_run_blocks and both explorers a block argument: 'val' for a single run, and the dev folds once NT-026 exists. The interactive widgets default to that block and say 'dev block: for choices'. Test views carry NT-031's wording ('test, not used for ranking' or 'never used for choices'), and compare_strategies on test becomes a read-only table. Reword STATUS's 'best strategy'. Growth point 2 (control panel and comparison). Land before NT-034.

**Acceptance (proposed).** (1) BacktestExplorer and CalibrationExplorer accept block='val' or block='test' (dev folds later), and the widgets' default is not 'test' (headless test). (2) Every explorer figure and table built on test carries the 'never used for choices' label (test on titles and captions). (3) Notebooks 02 and 04 are regenerated, the drift test passes, and 02's static test tables keep their numbers.

<a id="nb-6"></a>

## NB-6: Correction to NT-054: its reference run is a notebook run, whose gaps between timed epochs include TrainingSession's live redraw (2.0 s per epoch on this CPU) and whose timed epochs include the live batch strip

- **Severity:** P2. **Status:** not verified. **Type:** perf. **CPU cost:** short. **Placement:** addition to NT-054.
- **Files:** `src/neural_trade/notebook/session.py`, `src/neural_trade/training/trainer.py`, `src/neural_trade/telemetry/epoch_logger.py`, `src/neural_trade/training/custom_model.py`, `docs/BACKLOG.md`

**Description.** NT-054 attributes the 'about 30 s between the timed epochs' of runs/20260924T182915Z-1aeff1c-dirty-af67ee43 to epoch-end callback work (research README:53 says this is an inference) and targets callbacks.py and epoch_logger.py. That run is notebook 01's: notebook 01's cell 2 output prints 'run: ..\runs\20260924T182915Z-...', tags ['notebook'].
The session's Keras callback is appended after every configured callback (trainer.py:350-352), so it runs after jsonl_epoch_logger (inserted before reduce_lr_on_plateau, trainer.py:215-226). The logger's epoch_seconds stop at its own on_epoch_end (epoch_logger.py:105-127). The session's on_epoch_end then rebuilds the whole training dashboard, the health HTML and the batch strip in the training thread (session.py:198-207, 354-374), outside the timed epoch.
Inside the timed epoch, the session calls model.train_epoch_logs() every 25 batches (session.py:168-181). That function is a tf.function whose results are each converted with float(), and it is documented as 'called once per epoch' (custom_model.py:225-231). The session also rebuilds the batch strip at most every 2 s.
CLI and engine runs pay none of this. The notebook run is therefore the wrong reference for per-run fixed costs, for sec_per_step comparisons (D-018) and for NT-037's 2% budget.

**Failure scenario.** NT-054's implementer profiles callbacks.py and epoch_logger.py on a CLI run and finds far less than 30 s between epochs. Or the implementer compares before and after on notebook runs, where the redraw dominates the gap. Either way, the 'fixed cost' being removed is partly the notebook's own UI. The reference sec_per_step of 0.0984 (research README:52, :274) also includes the live strip.

**Evidence (finder, reproduced).** time_redraw.py and time_redraw2.py on this CPU (nproc 1) time the per-epoch _redraw: training_dashboard_figure build 1.97-2.23 s, to_json 0.01-0.02 s, health HTML 0.03 s. The cost does not depend on the epoch count (2.08 / 2.14 / 2.04 s at 5 / 10 / 20 epochs). The batch strip takes 0.05 s per draw. Over 20 epochs that is about 41 s here. The owner's CPU is faster, and the share of the 30 s gap on it was not measured (plausibly most of it).

**Backlog check (finder).** NT-054's area lists callbacks.py and epoch_logger.py, not notebook/session.py, and it does not say which run path to measure. NT-037 sets the 2% budget without naming the path. New as a correction to NT-054, with evidence for NT-037.

**Fix sketch.** (1) NT-054 acceptance (1) and (4): profile and compare CLI or engine runs, and name the run path; add notebook/session.py to the area. (2) In the session, build the full dashboard off the training thread (a refresher thread that snapshots the rows), or throttle it (every k epochs, or at least 10 s apart) and keep only a cheap loss strip live. Call train_epoch_logs at most once per epoch, reusing _EpochTrainLogs' values. Growth points 3 (per-run health of at most 2%, measured on the right path) and 2 (the control panel's live views must not slow sweeps). Land before the measurements of NT-054 and NT-037.

**Acceptance (proposed).** (1) NT-054 names the run path of its reference run and of its before/after runs (CLI or engine). (2) A unit test with a stubbed model shows that the session callback's on_epoch_end returns in under 50 ms, because the heavy redraw is elsewhere or throttled, and that the session calls train_epoch_logs at most once per epoch. (3) A GPU notebook run and a CLI run on the same config report sec_per_step within their noise (owner-machine measurement, recorded in the item).

<a id="nb-7"></a>

## NB-7: The explorer's 'max hold (bars)' cost control, BacktestConfig.max_hold, has no effect on any registered strategy

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-11.
- **Files:** `src/neural_trade/notebook/backtest_ui.py`, `src/neural_trade/strategy/backtest.py`, `src/neural_trade/strategy/strategies.py`, `src/neural_trade/strategy/trades.py`, `configs/strategies/enhanced_default.yaml`

**Description.** The engine uses BacktestConfig.max_hold only for an Order whose max_hold is None (backtest.py:217; trades.py:18). Every registered strategy sets Order.max_hold from its own field (strategies.py:92, 153, 216, 273, 293, 326). Two controls are therefore dead: 'max hold (bars)' in the explorer's 'Costs and execution' panel (backtest_ui.py:505), and backtest.max_hold in the shipped params file (configs/strategies/enhanced_default.yaml:12). A strategy knob with the same name in the 'Strategy knobs' panel does work. A search space built from BacktestConfig fields (NT-029, NT-030) would carry a dimension with no effect.

**Failure scenario.** In notebook 02, a user sets 'max hold (bars)' to 2 and presses Run. The result is identical: the same trades, held up to 8-19 bars. The user concludes that a hold limit does not matter. Editing backtest.max_hold in the params YAML changes nothing either.

**Evidence (finder, reproduced).** Headless probe (test_zz_probe_notebooks.py, using the viz fixtures and the widget callbacks), results as (trades, net return, longest hold in bars). With the cost control at 2, 30 and 500, every strategy returns the same result: calibrated_quantile (427, -0.667681, 8), liberal (548, -0.744085, 11), enhanced_multi_horizon (208, -0.291738, 5), threshold_spike (22, -0.070813, 19). The strategy knob max_hold=2 gives calibrated_quantile (460, -0.699715, 2).

**Backlog check (finder).** Checked NT-016, NT-018, NT-028 (stale config fields: not listed), NT-029, NT-030 and NT-041 (4) (cost profile). Not covered. New.

**Fix sketch.** Either remove max_hold from the explorer's cost panel and from any BacktestConfig-derived search space, or make a set BacktestConfig.max_hold an explicit cap, min(order, config), and document which one wins. Drop it from the YAML's backtest section or move it under params. Growth point 2 (search spaces in NT-029, the panel in NT-034). Land before NT-029.

**Acceptance (proposed).** (1) Either the cost panel has no max-hold control, or changing it changes the longest hold of calibrated_quantile's trades (headless test). (2) A params file whose backtest.max_hold no strategy honours is refused or warned about, or the value caps holds (test). (3) The existing backtest tests pass, including assert_no_lookahead.

<a id="nb-8"></a>

## NB-8: TrainingSession can train twice into one run: a second start() makes wait() return the previous result at once and begins already 'stopped'; re-running notebook 01's train cell appends a second training to the run's metrics.jsonl

- **Severity:** P2. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-32.
- **Files:** `src/neural_trade/notebook/session.py`, `scripts/notebooks/build.py`, `src/neural_trade/telemetry/epoch_logger.py`, `src/neural_trade/visualization/training_dashboard.py`

**Description.** start() refuses only while its own thread is alive (session.py:103-108). It does not reset _done, _stop, _pause, result, error or history. wait() (110-115) therefore returns the old result immediately, and the new run begins with _stop already set, so it stops after its first batch.
Notebook 01 creates the RunContext in cell 2 and the session in cell 4 (build.py:111, 116-118). Re-running cell 4, for example after Stop, creates a new session on the same ctx. The per-instance guard does not see a first session that is still running. The epoch logger appends to metrics.jsonl (epoch_logger.py:130-132), and history_frame keeps the last row per epoch (training_dashboard.py:159). Notebook 04 then draws a mix of two trainings as one run.

**Failure scenario.** (a) Control-panel or user code calls session.start() again after a Stop. wait() returns training #1's result while #2 runs; #2 stops after one batch and reports 'stopped early - evaluated'.
(b) In notebook 01, the user presses Stop and re-runs the train cell. The run dir now holds two trainings, concurrently if the first is still finishing. Notebook 04 shows epochs 1-3 from training 2 and epochs 4-6 from training 1. It reports 'best 5.5000 @ epoch 6' from training 1 and 'served epoch 3 ≠ best (6)', as if the served-epoch logic were wrong.

**Evidence (finder, reproduced).** probe_restart.py (stubbed trainer, no TF): 'run 2: wait() returned after 0.00 s with 'result of training #1' while run 2 is still running: True', and the second call records 'stop_already_set_at_start': True.
repro_dup_epochs.py builds a run dir with two appended trainings of 6 and 3 epochs, with the bundle's weights_epoch 3. Notebook 04 reads 6 rows, and the tiles say 'best 5.5000 @ epoch 6', 'served epoch 3 ≠ best (6) | val loss 8.8000' and 'last epoch val 5.5000'.

**Backlog check (finder).** NT-018 (notebook UX) does not mention session lifecycle or reuse of a run context, and NT-034 has only 'launch or resume'. NT-049 is about warm starts from MODEL_PATH, which is different. New.

**Fix sketch.** start() refuses a session that has already run, or fully resets its events and state for a fresh RunContext. TrainingSession refuses a run_context whose metrics.jsonl already exists or whose run is in progress (a lock file). Notebook 01 creates the RunContext in the train cell. Growth point 2 (the panel launches and resumes runs). Land before NT-034.

**Acceptance (proposed).** (1) A second start() on a finished session raises, or trains fresh, and wait() never returns a previous result (test with a stubbed trainer). (2) A TrainingSession on a RunContext that already has metrics.jsonl raises with a clear message (test). (3) Re-running notebook 01's train cell creates a new run dir (build.py change; the drift test passes).

<a id="nb-9"></a>

## NB-9: The slow notebook execution test passes parameters the notebooks do not have (RANDOM_SEEDS to 02, WINDOW to 03) and never notices

- **Severity:** P3. **Status:** not verified. **Type:** test-gap. **CPU cost:** instant. **Placement:** new item CPU-32.
- **Files:** `tests/test_notebooks_thin.py`, `scripts/notebooks/build.py`

**Description.** _run appends 'NAME = value' lines to the first code cell without checking the names (test_notebooks_thin.py:49-57). test_all_notebooks_execute passes RANDOM_SEEDS=3 to 02 and WINDOW=300 to 03 (:77-78). Neither is a parameter, and neither is read anywhere. 02's parameters are RUN_DIR, RUNS_DIR, CSV_PATH, DETAIL_BARS and DETAIL_AROUND (build.py:249-256); 03 has BARS, not WINDOW (build.py:319-327). So 02 runs the matched null with BacktestConfig's 100 seeds, for the widget run and for each strategy in compare_strategies, instead of 3. 03 runs with BARS=600. NT-034 (5) and (6) will reuse this harness for the 'tiny CPU scenario' execution.

**Failure scenario.** A notebook parameter is renamed in build.py, as WINDOW became BARS. The slow test stays green while it executes the default configuration. For notebook 06, a mistyped SCENARIO parameter would make the headless test run the full default scenario.

**Evidence (finder, reproduced).** dead_params.py parses the test and the committed notebooks with ast. Output: "02_backtest.ipynb injected=['RANDOM_SEEDS'] parameters=['CSV_PATH', 'DETAIL_AROUND', 'DETAIL_BARS', 'RUNS_DIR', 'RUN_DIR'] NEVER READ=['RANDOM_SEEDS']" and "03_signals_and_trades.ipynb injected=['WINDOW'] ... NEVER READ=['WINDOW']". BacktestConfig.random_seeds is 100 (strategy/backtest.py:47). The slow test itself was not run, because it trains.

**Backlog check (finder).** Covered by no item (NT-034 (5)/(6) would reuse the harness). New.

**Fix sketch.** _run asserts that each passed name is assigned in the notebook's parameters cell, as papermill warns, and fails otherwise. Rename the test's parameters to real ones (03: BARS), or add RANDOM_SEEDS to 02's parameters and use it in click_run and compare_strategies. Growth point 2 (NT-034's headless harness). Land before NT-034.

**Acceptance (proposed).** (1) A fast test fails when _run's name check is given a name that is not in the parameters cell. Factor the check out so no execution is needed. (2) A static test confirms that test_all_notebooks_execute passes only real parameter names.

<a id="nb-10"></a>

## NB-10: Evidence for NT-026 (9): pick_run orders runs by directory mtime, so writing any file into an old run makes it 'the newest', and equal mtimes are ordered by hash order

- **Severity:** P3. **Status:** not verified. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-026.
- **Files:** `src/neural_trade/notebook/runs.py`

**Description.** servable_runs sorts a set of run dirs by the directory's os.path.getmtime (runs.py:15-16). A directory's mtime changes whenever an entry in it is created, removed or renamed. Any later write into an old run therefore makes notebooks 02-05 load that run: a backtest --out into it, a re-score, or, once NT-010 tracks run light files, a git checkout of those files. Equal mtimes are ordered by the set's iteration order, which depends on PYTHONHASHSEED. Notebooks 02, 03 and 04 run in separate kernels and can each load a different run. NT-026 (9) only asks that engine runs be ignored.

**Failure scenario.** After NT-010 tracks run light files, `git pull` updates eval_report_test.json in an older run dir. The next execute.py of 02-04 shows that older run's numbers as the latest run's.

**Evidence (finder, reproduced).** pickrun probe with three servable runs. With equal directory mtimes, the pick over PYTHONHASHSEED 0-7 was 3x 20260920T100000Z-aaa, 3x 20260922T120000Z-ccc and 2x 20260924T182915Z-bbb. With distinct mtimes: 'before: 20260924T182915Z-bbb'; after writing backtest.json into the oldest run: 'after : 20260920T100000Z-aaa'.

**Backlog check (finder).** NT-026 (9) covers engine runs only, and the RUNBOOK describes 'newest'. The mtime ordering and the hash-order ties are not mentioned. Filed as evidence for NT-026 (9).

**Fix sketch.** Fold into the engine. Order runs by their recorded creation time (meta.json created_utc, or the UTC stamp in the run id), newest first, with ties broken by run id; the run store index (NT-026) supplies this. Growth point 1. Land during NT-026.

**Acceptance (proposed).** (1) pick_run returns the run with the latest created_utc regardless of directory mtimes (test that touches an older run). (2) The result is the same across PYTHONHASHSEED values (test with equal mtimes). (3) NT-026 (9)'s exclusion of engine runs still holds.

