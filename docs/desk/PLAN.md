# Trading desk on neural_trade: architecture plan

_Repository record of the plan approved for execution; contracts are repeated in [CONTRACTS.md](CONTRACTS.md), the task tables in [TASKS.md](TASKS.md), the entry point for an implementer session is [README.md](README.md). Source: the lead's plan file of 2026-10-15/16._

_Written by Fable 5.1 on 2026-10-15 for execution by Sonnet-class implementers through the project's
work loop. Evidence: two read-only research workflows (a repository map by 9 readers; a framework-reuse study by
9 agents with an adversarial refuter), then this plan, then a two-lens verification pass whose findings were
applied. Full reports: the workflow journals of this session; the digests are below._

## Context

The owner asked (2026-10-15, in Russian) for a trading system with an HTML front end "like TradingView":
full manual control, automation, backtesting, plug-in development and application of indicators, and the
full pipeline for developing neural networks and indicators; a detailed architecture plan written by the
strong model so that simpler models can execute it.

neural_trade already holds the ML half: ten strict registries plus a `Strategies` registry, a plugin
loader (`core/plugin_loader.py`, `plugins/` at the repo root), the experiment engine with an sqlite run
store, a backtest engine with next-open fills and a random null, `serving.Predictor`, a self-contained HTML
indicator report, and a notebook control panel. Missing for a desk: a web server, an exchange adapter, an
order/position/account model with persistence, an event-driven engine shared by backtest, paper and live,
a chart front end, and a plugin contract for classic (non-TF) indicators and strategies.

## Owner decisions taken for this plan (2026-10-15, to be recorded in DECISIONS as D-067)

| question | answer | consequence |
|---|---|---|
| research scale | the owner chose "medium (about 1M tokens)"; the owner then added the framework-reuse study. Actual spend: repository map 9 readers 1.95M tokens (29 min), framework study 9 agents 1.78M (47 min), verification 2 lenses 0.55M (16 min): about 4.3M subagent tokens, four times the chosen budget; the readers cost about four times the lead's estimate, and the lead did not stop to re-ask | this plan's evidence base; the overspend is reported in STATUS "Agent ledger" |
| where it lives | **a subpackage `src/neural_trade/desk/`** | it is under the layering test (top layer, nothing imports it), the ARCHITECTURE module map, the docs tests, the golden run (must stay unchanged), the MVP definition of done; package-data for static files |
| backend stack | **FastAPI + uvicorn + pydantic v2 + ccxt** (pinned; installed by the lead after a dry run in a scratch venv, as optuna was); after the framework research: **NiceGUI 3.17.1 on the same FastAPI app for the panels** (question 0) | typed contracts, an OpenAPI document as a QA artifact, WebSocket, tests through httpx `TestClient` and NiceGUI's `user` fixture |
| trading scope v1 | **Binance spot through ccxt: paper account by default, live behind an explicit flag, confirmation and risk limits; keys only from environment variables** | one exchange adapter; the execution model is shared by backtest, paper and live |

**Owner correction (2026-10-15, later the same session):** before designing, research whether existing
stable frameworks (trading frameworks, backtest platforms, UI frameworks and window managers, charting,
storage, job runners) can carry most of the desk; the goal is an application built maximally on stable
frameworks for simplicity of modification, development and testing. The stack choices above (FastAPI +
vanilla JS) therefore become one candidate among others; the research's recommendation decides, and a
different outcome goes back to the owner as a decision.

**Research outcome (workflow wf_27eb9e1c-91c, 9 agents, 1.78M tokens; 30 of 36 decisive claims confirmed by the
refuter, 3 refuted and corrected below):** no whole-product trading framework (freqtrade, OctoBot, Jesse,
Hummingbot, vn.py, NautilusTrader, LEAN) and no backtest engine can be imported next to TensorFlow 2.10: they need
Python >= 3.11/3.12 or numpy 2, or are copyleft (backtrader GPL-3, backtesting.py AGPL-3; vectorbt 1.0.0 installs
but is vectorised and Commons-Clause). None delivers drawing tools, bar replay, chart order lines, a stop-order
ticket, the plugin contract or the Model Lab. What stable frameworks do carry, inside one MIT-clean process in the
`nt` env: **NiceGUI 3.17.1 mounted on the FastAPI app** (every panel a Python widget, socket.io push, an in-process
`user` test fixture with no browser), **lightweight-charts 5.2.1** vendored as one custom element, **dockview-core
8.5.1** vendored as the window manager, **ccxt 4.5.88**, **sqlite3 + SQLAlchemy 2.0.54** (installed), **ta 0.11.0**
(MIT, pure Python) for classic indicators with **talipp 2.7.0** optional for streaming, **TensorBoard 2.10.1**
(installed) and **optuna-dashboard 0.21.0** linked as viewers, **Playwright 1.64** for a few smoke tests. What stays
custom in every option: the event-driven engine (about 1k lines, vn.py's MIT crossing rules as the written spec),
the ccxt broker, the plugin contract on the existing registries, the Model Lab pages, and the chart element's
drawing and order-line primitives. Corrected sizes: approach B (this plan) about 14-20k lines including tests, of
which 4-8k is the chart element; the vanilla-JS control C about 20-30k. **This plan adopts B**; the owner's earlier
pick (FastAPI + vanilla JS) is the control, and the choice is question 0 below.

Fixed decisions that still apply: D-001 (TF 2.10, Python 3.10), D-002 (registries), D-018 (the per-step
training path never gets slower), D-029 (delete nothing with an effect), D-044 (the research yardstick uses
zero costs; the desk's paper and live accounts use the exchange fee schedule, explicitly), D-050 (no new
model input source; a live OHLCV feed is the same Binance OHLCV source, not a new model input), D-062/063
(GPU sharing rules), VISION "Not in the MVP" (a web application): this plan is a new track after the MVP's
points, which the owner's request opens; VISION's "Not in the MVP" wording is the owner's to amend.

## Requirements (the owner's six, made checkable)

R1 **Chart front end "like TradingView"** (HTML, served by the desk): candlestick chart of any stored
symbol and timeframe, with zoom and pan over at least one year of 1-minute bars without freezing; several
panes (price plus indicator panes); indicators added from a catalogue with a parameter dialog generated
from the plugin's parameter schema; drawing tools (at least horizontal line, trend line, rectangle, text);
saved layouts (symbol, timeframe, indicators, drawings) that reload; a watchlist of symbols; bar replay
(step forward from a chosen time with the same engine the backtester uses).

R2 **Full manual control**: market, limit and stop orders placed from an order ticket and from the chart
(order and position lines, drag to modify, cancel), a positions and orders panel with live P&L, an account
view, a kill switch that cancels every order and stops every bot, a paper account by default and a live
account only after an explicit arming step; every action journalled.

R3 **Automation**: bots = strategy plugins running on bar closes for a symbol and timeframe, started and
stopped from the UI, with risk limits enforced before every order (max order notional, max position, daily
loss limit, max orders per minute) in every mode; alerts on price, indicator and prediction conditions,
delivered in the UI and logged.

R4 **Backtest**: an event-driven backtester that runs the same strategy code as the bots, with next-open
fills, stop and take-profit on the bar's high and low, fees, spread and slippage from an explicit cost
profile, a report with the Strategy-Tester fields (net profit, trades, win rate, profit factor, max
drawdown, Sharpe, Sortino, buy-and-hold, average trade, largest win and loss) plus the project's noise-aware
statistics (n_eff bands, the random null at the same trade frequency, buy-and-hold and always-flat
baselines); trades drawn on the chart and the equity curve in a pane; a parity test that reproduces the
existing `strategy/backtest.py` trades for a wrapped model-signal strategy on the same inputs.

R5 **Plug-in development and application of indicators**: a Python plugin contract for classic indicators
(parameter schema with defaults and bounds, inputs, outputs, draw spec: overlay or pane, line, band,
histogram, marker) and for strategies, discovered by the existing plugin loader from `plugins/`, with
templates, a golden-fixture test harness, reload from the UI, and a plugin page that lists every plugin
with its schema and test status; the learnable indicator families of neural_trade shown on the chart at
their textbook parameters and at a run's learned parameters.

R6 **Full NN and indicator development pipeline** (the Model Lab): datasets (import bars from the repo's
CSV files and from Binance, export a range as a training CSV, dataset specs), scenarios (list, view, edit a
copy), training runs launched as `neural-trade` subprocesses under the GPU lock with live progress (epoch
metrics, status, logs) and the run's figures (training, evaluation, calibration, trading, discovered
indicators), sweeps (quick and Optuna with the stated budget), the leaderboard, pre-registered compare
verdicts, a model registry (promote a run's bundle; a promoted model serves predictions on the chart and to
bots through the existing `Predictor`), and the indicator-development path (plugin template → screen →
train → learned periods against textbook → chart).

Cross-cutting: single user on localhost; every timestamp UTC; the desk's own state in a gitignored
directory; the desk server process never uses the GPU (all GPU work is a subprocess under the GPU lock);
no change to the training path, the golden run or the notebooks' numbers; every task sized for a
Sonnet-class implementer (a contract to implement against, a test to pass, at most L = 800 lines including
tests; larger is split by the lead).

## What the research established (digest; full reader reports in the workflow journals)

**Reuse map (exists, verified by readers on 2026-10-15):**

| need | what exists | where | how the desk uses it |
|---|---|---|---|
| bars in, standardised frame out | `standardize_ohlcv`, `sort_dedupe`, `resample_bars` (all `(df, config)`, in `data/preprocessors.py`); `validate_ohlcv_frame` (`data/loaders.py`); `run_preprocessors(df, Config())` (`data/preprocessors_registry.py`); `DataLoaders` ('csv', 'parquet' (pyarrow absent), 'dataframe') | `data/` | every bar frame entering the desk is validated by `run_preprocessors` + `validate_ohlcv_frame`; training input is a CSV in this contract (the engine refuses in-memory loaders) |
| bar size, wall-clock lengths, annualisation | `DatasetSpec`, `minutes_to_bars`, `periods_per_year(bar_minutes)` (24/7 only) | `core/dataset_spec.py`, `strategy/performance.py` | the desk's Sharpe uses the same function; timeframes are integer minutes |
| inference | `Predictor.from_artifacts(dir)`; `predict_last(df)` needs >= 60 closed bars (LOOKBACK, no burn-in), returns per horizon delta, price, p_up, p_up_calibrated, gauss_p_up, sigma, lo90, hi90, horizon_bars, direction_signal ('ok'/'none'/'n/a'); `predict_windows_frame(df)` -> `PredictionBatch.to_prediction_frame(pred_scale, pred_mean)` | `serving/predictor.py` | prediction overlays and model-signal bots; CPU is enough (`CUDA_VISIBLE_DEVICES=-1` in the server) |
| vectorised backtester (the parity oracle) | `run_backtest(signals, bars, strategy, BacktestConfig)`: decide at close t, fill at open t+1 only; SL/TP on high/low, gap-through-stop fills at the open, SL-first tie; `max_hold` -> 'TIME' exit; `size_frac` of realised equity resolved at the fill open, no leverage; costs per side fee + (half_spread + slippage), both on the mid notional, the slipped price recorded as the fill; `Bars(open, high, low, close, breaks)`; `BacktestResult(equity[N+1], trades, decisions, position, summary)`; `ExposureStrategy` + `run_exposure_backtest`; `random_same_frequency`, `circular_shift_null`; `assert_no_lookahead` | `strategy/backtest.py`, `strategy/strategies.py`, `strategy/trades.py` | the desk engine reproduces these fill rules (A5); a parity test runs the same strategy through both engines |
| model-signal strategies | 15 registered `Strategies` (dataclasses, `decide(s, t)`, `exit_signal(...)`, `FittedOnCalibration.from_calibration(cal_signalframe)`, `search_field` ranges; two of them, `vol_target` and `net_edge_kelly`, are `ExposureStrategy` entries with `target(s, t, current)`); `SignalFrame.build(PredictionFrame, var_scale)`; `ta_rules.price_only_frame(close)` for price-only strategies; `ta_rules.sma/rolling_std/wilder_rsi`; `scorer.fit_and_backtest` | `strategy/`, `experiments/scorer.py` | wrapped by an adapter so the discrete ones run as desk bots and in the desk backtester unchanged (exposure ones: DESK-1.12) |
| statistics | `sharpe`, `sortino`, `max_drawdown`, `profit_factor`, `summarize`; `n_eff`, `mean_ci`, `wilson`, `block_bootstrap_counts`, `paired_t_ci` | `strategy/performance.py`, `metrics/statistics.py` | the desk report calls these, never re-implements them |
| learnable indicators | `Indicators` registry (TF-only; 15 entries: the 14 D-031 families plus `macd_ratio`; `indicators.base.ParamSpec`, `ChannelSpec(draw 'price'|'panel')`); numpy path only for ma/macd/rsi/bb (`discovered_indicators.indicator_lines`); generic eager path `compute_reference(family, close, periods, series=...)`; learned periods in `metrics.jsonl` `period/<name>` and `period_init.json`; `indicator_evolution.applied_periods(predictor.model, windows, normalizer=predictor.bundle.normalizer)` | `indicators/`, `visualization/discovered_indicators.py`, `visualization/indicator_evolution.py` | the "learned indicator lens": one LOOKBACK window drawn on the chart at textbook and learned periods |
| plugins | `load_plugins(dir)` (rglob *.py, skips `templates/` and `_*.py`, module `neural_trade_plugins.<path>`, idempotent, no unload); `registries.load_all(config, plugins_dir=)`; `BaseRegistry` (strict, duplicate raises, `remove`, `list` with metadata) | `core/plugin_loader.py`, `core/registry.py` | desk plugins are ordinary plugin files registering into two new desk registries; reload = restart the plugin host process |
| experiment engine | `Scenario.from_yaml/from_dict/validate/cells`; `Runner.from_spec(path, store).plan()`; CLI `scenario run|plan|rescore|reindex`, `sweep --dry-run`, `compare`, `leaderboard`, `screen`, `stability`, `train`, `predict`, `backtest`, `indicators`; `RunStore(root).index.rows(scenario)`, `.scores(run_id)`; sqlite schema (tables meta, runs, scores); `result.json`, `eval_report_<role>.json`, `predictions_{cal,oos}.npz` (`scorer.load_block`) | `experiments/` | the Model Lab is a thin HTTP layer over these calls; writers go through the engine (D-065) |
| headless panel logic | `panel_data.discover_scenarios/list_sweeps/field_options/strategy_options/fold_choices/validate_search/build_board/store_fingerprint`; `panel_compare.aggregate_selection/paired_by_fold/find_compare_spec/stored_verdict`; `ControlPanel` headless methods; `longrun.launch/progress/stop` (detached subprocess, pid.json, tolerant readers, stall detection); `run_report.load_saved_blocks/run_overview/key_numbers_table` | `notebook/` | called directly by the Lab API; `longrun` is the pattern for every desk job |
| figures | 27 `Visualizations` entries, all plotly, serialise with `pio.to_json(fig, validate=False)` (typed arrays need plotly.js 3.x: serve `plotly.offline.get_plotlyjs()`); theme tokens (PAPER #121211, SURFACE #1a1a19, INK #f2f1ec, INK_2 #c3c2b7, MUTED #898781, GRID #2c2c2a, AXIS #383835; horizons #3987e5/#d95926/#199e70; GOOD #0ca30c, WARNING #fab219, CRITICAL #d03b3b; LONG/UP #9085e9, SHORT/DOWN #c98500) | `visualization/theme.py`, `notebook/_display.py` | run-detail pages embed the existing figures unchanged; the desk's CSS variables come from the theme tokens; candles get their own two tokens (not green/red status colours) |
| process | item template (status, priority/type/role, area, depends on, why, acceptance, source, evidence); roles and models (implementer Sonnet 5.5 medium, qa Opus 5.5 medium, qa-deep high); test tiers; CI `unit` job runs all of `tests/` with `requirements-ci.txt`; `test_layering.py` scans `src/neural_trade` subpackages; `test_docs_guides.py` checks ARCHITECTURE's module map and backticked paths/ALL_CAPS tokens in README and `docs/guide/*.md`; `check_run_evidence.py` treats any `runs/` dir named like a run id as a run | `docs/`, `tests/`, `.github/` | desk docs live in `docs/desk/`; desk state lives outside `runs/`; a `desk` CI job keeps `unit` flat |

**Missing (the desk builds it):** exchange adapter and live feed (no ccxt anywhere); order, fill, position, account model and persistence; an event-driven engine shared by backtest, replay, paper and live (the existing backtester is whole-array, single position, bar-indexed, no timestamps, no limit or stop entries); a classic (non-TF) indicator contract and catalogue (only `sma`, `rolling_std`, `wilder_rsi` exist); a candlestick chart (no OHLC figure anywhere), drawing tools, layouts; a web server and API (none); a job registry (only `longrun` pid files); JSON serialisers for `LeaderboardRow`; a GPU mutual-exclusion rule (D-063 withdrew the lock file; only `nvidia_smi_gpu_check` exists).

**External facts that fix choices (sources in the external reader's report):** lightweight-charts 5.2.1 (Apache-2.0, standalone build 193 KB, keep `attributionLogo`; v5 API: `chart.addSeries(CandlestickSeries, opts, paneIndex)`, `createSeriesMarkers`, `series.createPriceLine`, pane API; no built-in drawing tools, official rectangle/trend-line primitive examples to copy); KLineCharts 10.1.0 is the fallback with built-in drawings; plotly.js candlesticks stall far below 100k bars. ccxt 4.5.88 (MIT, Python >= 3.10, Pro websockets free, Binance spot testnet via `set_sandbox_mode(True)`, klines 1000 per request, the closed flag is dropped by `watch_ohlcv`; exact dependency pins must be dry-run against the env). FastAPI 0.143.2 needs `starlette==1.7.0` (1.8 dropped Python 3.10) and pydantic 2.14.1; never `fastapi[standard]` (opentelemetry-proto needs protobuf >= 5, TF 2.10 needs < 3.20); uvicorn 0.54.0 + websockets. NautilusTrader (Python >= 3.12), freqtrade (>= 3.11), vn.py (numpy >= 2.2) cannot share the TF 2.10 process: design references only (their shape: kernel + bus + data/risk/execution engines + portfolio; three contexts differing only in clock, data client and execution client; explicit order state machine). Playwright 1.64.0 lists Windows 11+ officially: E2E runs in CI on Linux; there is no Node on the machine and no JS unit test.

## Architecture

### A1. Principles (what makes this executable by Sonnet-class implementers)

1. **Every task implements a written contract** (section C) and is accepted by a named test. No task asks the implementer to design an interface.
2. **One domain model, three contexts.** Backtest, replay, paper and live run the same `Engine` with the same `Strategy` code; they differ only in `Clock`, `DataSource` and `Broker`. The fill rules are written down (A5) and tested against the existing vectorised backtester.
3. **Reuse, never re-implement:** statistics, preprocessors, the Predictor, the experiment engine, the figures and the theme are called, not copied (table above). A desk module that re-implements one of them fails review.
4. **The server process never uses the GPU** (`CUDA_VISIBLE_DEVICES=-1`); every GPU job is a subprocess of the existing CLI, started only after the desk's own GPU gate says free: the JobRunner never calls `nvidia_smi_gpu_check()` (it short-circuits to "free" under `CUDA_VISIBLE_DEVICES=-1`, `experiments/sweep.py` `_cpu_only`); it runs `nvidia-smi dmon -s um -c 10` itself (`subprocess.run`, timeout 40 s) and judges the text with `experiments.sweep.gpu_status_from_dmon` (RUNBOOK thresholds: median fb <= 2000 MB and sm <= 30%). Nothing the desk adds runs inside a training step (D-018).
5. **Files are the record, sqlite is the index** (as in the run store): desk state in `var/desk/` (gitignored), run-store writes only through the engine (D-065), atomic writes through `utils/atomic.py`, tolerant readers.
6. **Honest numbers:** the direction-signal state is shown as such (D-066), served-delta n/a when beta = 0 (D-007), every report carries buy-and-hold, always-flat and the matched random null, n_eff bands (D-012), test columns never rank (D-020). The desk's accounts carry an explicit cost profile; the research yardstick keeps D-044 (0).
7. **Small, flat modules**, dataclasses over classes with behaviour, pure functions for indicators and fills, explicit state machines for orders and jobs.

### A2. Package layout (`src/neural_trade/desk/`, the top layer; nothing in neural_trade imports it)

```
desk/
  __init__.py            no TensorFlow import at import time (test); exports version and DESK_HOME
  config.py              DeskSettings (pydantic-settings style but hand-written dataclass + YAML + env): home dir, server host/port, exchange, accounts, risk limits, cost profiles, plugin dirs
  core/                  domain model and pure logic (no I/O, no web)
    model.py             Instrument, Bar, Order, Fill, Position, AccountState, Alert, events (A4)
    params.py            ParamSpec, params_schema, validate_params (C1; shared by strategies and indicators)
    journal.py           Journal (jsonl writer with atomic finalisation), EngineRun (C1)
    clock.py             Clock protocol: SimClock, LiveClock
    bus.py               in-process EventBus (publish/subscribe by topic; sync; deterministic order)
    fills.py             pure fill functions (A5): market_next_open, limit_in_bar, stop_in_bar, bracket_check
    costs.py             CostProfile (fee_bps taker/maker, half_spread_bps, slippage_bps) -> per-side costs; default profiles: research_zero (D-044), binance_spot_taker
    risk.py              RiskGuard: pre-trade checks (precision/lot rounding, min notional, max order notional, max position, max orders per minute, daily loss limit, trading state ACTIVE|REDUCING|HALTED)
    portfolio.py         Portfolio: positions (NETTING per instrument+strategy), cash, equity mark, realised/unrealised P&L, drawdown
    execution.py         ExecutionEngine: order state machine, routes commands to a Broker, applies fills to the Portfolio, emits events
    strategy.py          DeskStrategy base (on_start/on_bar/on_order_event/on_stop), StrategyContext (history access, indicators, predictions, order helpers), ParamSpec for UI forms
    engine.py            Engine: the loop (A3) for backtest/replay/paper/live; EngineConfig; EngineRun record
    report.py            BacktestReport from the journal: Strategy-Tester fields + neural_trade performance/statistics calls; baselines
  data/
    store.py             MarketStore (sqlite): bars(symbol, tf, ts) read/write, ranges, gaps; resample(tf) via pandas; export_csv (training contract)
    sources.py           DataSource protocol: HistoricalSource(store), ReplaySource(store, start, step), LiveSource(store watcher)
    importers.py         import from repo CSVs (standardise via neural_trade preprocessors), from ccxt REST (paginated backfill)
    collector.py         the live collector process: ccxt REST polling (v1) / websocket (v2) -> MarketStore; closed-bar rule
    exchange.py          ExchangeClient protocol + CcxtBinanceSpot (markets, precision, balances, orders) + FakeExchange (tests)
  broker/
    base.py              Broker protocol: submit/modify/cancel, poll fills, sync (reconcile)
    sim.py               SimBroker: fills from bars with core/fills.py and a CostProfile (backtest, replay, paper)
    ccxt_broker.py       CcxtBroker: live orders on Binance spot (sandbox or production), reconciliation by trade id, idempotent client ids
  indicators/
    api.py               the classic indicator contract (C3): IndicatorMeta, ParamSpec, DrawSpec, ChartIndicators registry; compute(df, **params) -> DataFrame; causal/warm-up tests helper
    builtin/*.py         one file per indicator (sma, ema, wma, rsi, macd, bollinger, atr, stochastic, willr, keltner, obv, vwap, mfi, adx, cci, donchian, supertrend, ichimoku, psar, volume, heikin_ashi, vwma)
    learned_bridge.py    draws a registered learnable family on one LOOKBACK window at textbook or learned periods (compute_reference / indicator_lines)
    predictions.py       prediction overlays from a promoted model: per-horizon price path, lo90/hi90 band, P(up) pane with the direction-signal state
  plugins_api.py         what a plugin file imports: register_indicator, register_strategy, DeskParamSpec, templates' base classes, reload(); discovery = neural_trade.core.plugin_loader.load_plugins(d) for d in settings.plugin_dirs (default plugins_desk/)
  strategies/            built-ins ma_cross, rsi_threshold, bollinger_breakout, buy_and_hold, always_flat, random_entries; model_signal.py (C2); exposure adapter (DESK-1.12)
  trading/
    manual.py            manual orders through the ExecutionEngine on the paper/live accounts (DESK-2.6)
    replay.py            replay sessions: a server-side cursor feeding chart and engine bar by bar (DESK-3.7)
  testing/
    causality.py         the look-ahead harness for desk strategies (DESK-1.11)
  bots/
    runner.py            BotRunner: one Engine per bot in paper/live mode fed by the LiveSource; start/stop/kill; heartbeat; status
    alerts.py            AlertEngine: conditions (price, indicator, prediction, strategy event) with once/every-bar-close semantics
  jobs/
    model.py             Job(kind, params, state PENDING|RUNNING|DONE|FAILED|CANCELLED, pid, log, started, finished, result)
    runner.py            JobRunner: subprocess launcher (`python -m neural_trade.desk.jobs <job_id>`), log capture, pid liveness (reuse longrun.pid_alive), cancel (taskkill)
    kinds/*.py           import_bars, backfill, backtest, train (wraps `neural-trade train`/`scenario run` through longrun.launch), sweep, compare, indicator_report, screen, stability, export_dataset
    __main__.py          entry point for the subprocess
  lab/
    datasets.py          dataset specs for training (symbol, tf, range) -> CSV export + fingerprint; list repo CSVs
    scenarios.py         list/read/validate/copy scenario YAMLs (Scenario.from_dict; writes only to configs/scenarios/desk/)
    runs.py              run store queries (RunStore, RunIndex rows/scores), run detail (meta, status, metrics, eval_report, result), progress (longrun.progress readers)
    figures.py           figure endpoints: Visualizations.build(key, ...) -> plotly JSON; theme tokens -> CSS
    leaderboard.py       build_leaderboard rows -> JSON serialiser (LeaderboardRow/RoleAggregate)
    models.py            model registry (promote a run's artifacts/ bundle; list; the "active model" per symbol/tf); Predictor cache (one per bundle, CPU)
  store/
    db.py                DeskDB (sqlite, WAL): schema migrations (C4), repositories: accounts, orders, fills, positions, bots, alerts, layouts, drawings, models, jobs, journal
  api/
    app.py               FastAPI app factory; routers; lifespan (start/stop store watcher, bot runner, job runner); NiceGUI mounted with ui.run_with
    schemas.py           pydantic models of the API (C5): one module, mirrors core/model.py
    routers/*.py         bars, indicators, plugins, orders, positions, accounts, bots, alerts, backtests, replay, layouts, drawings, lab, jobs, system
    ws.py                WebSocket hub for scripts and the chart element: topics (C5), fan-out from the EventBus and store watchers
  ui/                    NiceGUI (A9): pages and panels in Python, two custom elements in JS
    conventions.md       the pinned NiceGUI 3.17 idioms every UI task cites (ui.run_with, element skeleton, storage, timers, bulk data rule)
    theme.py             CSS variables from visualization/theme.py tokens plus candle/order tokens
    pages/{desk,lab,plugins,backtests}.py
    panels/*.py          ticket, orders, positions, fills, account, watchlist, bots, alerts, jobs, replay controls, indicator dialog, arming/kill dialogs, journal
    elements/lwc_chart.js + chart.py     the chart element (lightweight-charts 5.2.1 vendored; props/methods/events documented in the file header)
    elements/dock.js + dock.py           the window-manager element (dockview-core 8.5.1 vendored; toJSON/fromJSON layouts)
    static/vendor/       lightweight-charts.standalone.production.js + LICENSE + NOTICE, dockview-core.min.js + licence note, THIRD_PARTY_NOTICES.md
  cli.py                 `neural-trade desk init|serve|collector|job|import-csv|backtest` (registered in neural_trade/cli.py as one subparser)
```

Tests: `tests/desk/` (unit tests per module; API tests through httpx `TestClient`; panel flows through NiceGUI's `user` fixture, no browser; parity tests; plugin contract tests), `tests/desk/e2e/` (5-10 Playwright smoke tests, marker `e2e`, run in CI on Linux; local Windows 10 best-effort).

### A3. Process model

- `neural-trade desk serve`: one process, CPU only (`CUDA_VISIBLE_DEVICES=-1` set in the entry point before the first TensorFlow import, `sys.stdout.reconfigure(encoding="utf-8")` for the cp1251 console; a startup assertion checks `tf.config.list_physical_devices('GPU') == []` after the first Predictor load), hosts the FastAPI API, NiceGUI (socket.io push to the browser, pages and panels), the WebSocket hub for scripts and the chart element, the BotRunner (paper/live engines on a bar-close scheduler, one asyncio task per bot), the AlertEngine, the JobRunner and a store watcher that broadcasts new bars. Blocking work (ccxt REST calls, backtests, sqlite reads, Predictor batches) runs in a ThreadPoolExecutor. Bulk series (bar windows of at most 20,000 bars, indicator arrays, trade lists) go through FastAPI JSON endpoints that the chart element fetches itself; never through socket.io props (a test asserts the cap). TensorBoard (`--logdir runs`) and optuna-dashboard are started on demand as separate processes and shown in an iframe.
- `neural-trade desk collector --symbol BTC/USDT --tf 1`: a separate process that keeps `market.sqlite` current (REST polling every 5 s in v1; a candle is closed when the next open time appears or `ts_open + tf <= server time`; reconcile the last two candles against REST after reconnects). The store is the boundary, so the collector may run in its own venv if ccxt's pins cannot be installed next to TF 2.10.
- Jobs run as subprocesses (`python -m neural_trade.desk.jobs <id>`); GPU kinds call the existing CLI (`neural-trade train`, `scenario run`, `sweep`, `compare`, `screen`, `stability`, `indicators`) with the launch pattern of `notebook/longrun.launch` (`launch_env(root)`: `CUDA_VISIBLE_DEVICES` popped, `PYTHONIOENCODING=utf-8`; `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`; repo root cwd; stdout to the job log); `longrun.launch` itself only for the `scenario run` kind (it takes a spec), `train` as `[sys.executable, "-m", "neural_trade.cli", "train", ...]` with the run dir from the last stdout line (as CI's smoke does); progress from the run directory files through `longrun.read_jsonl`, `longrun.tail`, `longrun.pid_alive` and `longrun.progress(spec, store)` for scenarios; the desk's GPU gate (A1.4) runs before every GPU kind; one GPU job at a time is enforced by the JobRunner (a `gpu` queue of size 1). The entry point sets `CUDA_VISIBLE_DEVICES=-1` before the first TensorFlow import, calls `sys.stdout.reconfigure(encoding="utf-8")` (setting `PYTHONIOENCODING` inside a running process has no effect) and passes `PYTHONIOENCODING=utf-8` to every subprocess.
- Backtests run as CPU jobs (subprocess) so a long backtest never blocks the server; replay runs in-process (interactive, one bar per request or per timer tick).
- The front end is static files served by the same process (same origin, no CORS). Single user on 127.0.0.1; a random session token in `var/desk/token` is required on every mutating request and on the WebSocket.

### A4. Domain model (dataclasses in `desk/core/model.py`; the field lists are the contract)

- `Instrument(id='BINANCE:BTCUSDT', symbol='BTC/USDT', base, quote, price_increment, size_increment, min_qty, min_notional, max_qty, is_spot=True)`; fees live in `CostProfile`, not on the instrument.
- `Bar(instrument_id, tf_minutes, ts_open_ms, open, high, low, close, volume, is_closed)`; a strategy sees only closed bars; the forming candle is display-only. `ts_open_ms` is the Binance open time (UTC ms); the bar closes at `ts_open_ms + tf*60_000`.
- `Order(client_id, instrument_id, account_id, strategy_id|None, bot_id|None, side BUY|SELL, type MARKET|LIMIT|STOP_MARKET|STOP_LIMIT, qty|None, size_frac|None, price|None, trigger_price|None, tif GTC|DAY|IOC|FOK, reduce_only, bracket: {take_profit, stop_loss, tp_is_offset: bool, max_hold: int|None}|None, status, filled_qty, avg_fill_price, reason, venue_id|None, ts_created, ts_updated, tags)`. Exactly one of `qty` and `size_frac` is set: `size_frac` is a fraction of realised cash equity resolved by the broker at the fill open (the oracle's `size_frac`, A5.1); with `tp_is_offset` the bracket levels are quote-currency offsets from the fill bar's open mid, resolved at the fill (the oracle's `tp_is_offset`, which keys both tp and sl). **Status edges (the complete list; any other transition raises `IllegalTransition`):** INITIALIZED->DENIED, INITIALIZED->SUBMITTED, SUBMITTED->REJECTED, SUBMITTED->ACCEPTED, ACCEPTED->TRIGGERED, ACCEPTED->PARTIALLY_FILLED, ACCEPTED->FILLED, ACCEPTED->CANCELED, ACCEPTED->EXPIRED, TRIGGERED->PARTIALLY_FILLED, TRIGGERED->FILLED, TRIGGERED->CANCELED, TRIGGERED->EXPIRED, PARTIALLY_FILLED->PARTIALLY_FILLED, PARTIALLY_FILLED->FILLED, PARTIALLY_FILLED->CANCELED, PARTIALLY_FILLED->EXPIRED. `is_closed = status in {FILLED, CANCELED, EXPIRED, REJECTED, DENIED}`. One exception edge `late_fill`: a Fill reported for a CANCELED order (the venue filled before the cancel landed) is applied to the Portfolio and moves the order CANCELED->PARTIALLY_FILLED or FILLED with journal kind `late_fill`; a Fill for a FILLED order with a known `trade_id` is ignored, with an unknown one raises `OverfillError`. Transitions only through `ExecutionEngine.apply(event)`.
- `Fill(trade_id, client_id, instrument_id, side, qty, price, mid, fee, fee_currency, liquidity MAKER|TAKER, ts)`; `price` is the slipped fill price, `mid` the reference price before spread and slippage (equal to `price` for venue fills), so the report computes gross P&L on mids and costs = qty x |price - mid| + fee, as the oracle does; partial fills are several Fill events.
- `Position(position_id='{instrument}-{strategy}', instrument_id, account_id, strategy_id, side LONG|SHORT|FLAT, qty, avg_price, realised_pnl, fees, ts_opened, ts_closed|None)` (NETTING; Binance spot cannot short: `RiskGuard` denies SELL beyond the held base balance on spot accounts).
- `AccountState(account_id, kind BACKTEST|PAPER|LIVE, quote_currency, cash, equity, locked, realised_pnl, unrealised_pnl, ts)`.
- Events (all with `ts_event`, `ts_init`; `ts_init` is `clock.now_ms()` of the engine's Clock, never `time.time()`, so backtests are deterministic): `BarClosed`, `BarForming`, `OrderEvent(kind, order)`, `FillEvent`, `PositionEvent`, `AccountEvent`, `BotEvent(status|log)`, `AlertFired`, `JobEvent`, `TimeEvent`. Commands: `SubmitOrder`, `ModifyOrder`, `CancelOrder`, `CancelAll`, `ClosePosition`, `Halt`.

### A5. Fill rules (written before coding; the parity test pins them)

1. A strategy decides on the closed bar `t` (after `BarClosed`). A MARKET order created then fills at bar `t+1`'s open. **Sizing at the fill:** a `size_frac` order's quantity is `clip(size_frac, 0, 1) x realised cash equity (net of costs paid; the oracle's equity_cash, after every exit filled at the same open) / open`; a `qty` order fills its quantity. **Offset brackets** (`tp_is_offset`) resolve to `open + offset` at the fill (both tp and sl). RiskGuard checks a `size_frac` order twice: at submit on the last close (an estimate) and at the fill on the resolved quantity (the binding check). **Cost arithmetic (oracle, bit for bit):** per side, slip cost = `qty x mid x (half_spread_bps + slippage_bps)/1e4` and fee = `fee_bps/1e4 x qty x mid`, where `mid` is the bar OPEN for a market, TIME or strategy-exit fill and the bracket or limit level for an intrabar fill; `Fill.price = mid x (1 + side x slip)`, `Fill.mid = mid`, `Fill.fee` is the fee on the mid notional (never on the slipped price); cash accounting `cash -= side x qty x Fill.price + Fill.fee`. This is exactly `strategy/backtest.py` (`fill='next_open'`), so the wrapped model-signal strategies reproduce the existing trades bit for bit at the same cost profile.
2. LIMIT: fills inside bar `t+1..` when the bar's range touches the price, at `min(limit, open)` for buys / `max(limit, open)` for sells (gap through the level fills at the open); a touch counts as a fill (documented optimistic assumption; a `touch_fill_prob` knob is an item for later). STOP_MARKET: triggers when high >= trigger (buy) / low <= trigger (sell); fills at `max(trigger, open)` / `min(trigger, open)` plus slippage. STOP_LIMIT: trigger then a limit.
3. Bracket (take-profit and stop-loss on a position): checked on every closed bar's high/low from the fill bar onward (the fill bar included, as the oracle does); both hit in one bar -> stop first (`same_bar_tiebreak='sl_first'`, the oracle's default; `tp_first` configurable); gap through the stop fills at the open. **Bracket ownership:** a bracket belongs to the position it opens or enlarges; a fill that adds to an open position replaces the position's bracket with the new order's (last fill wins; a fill without a bracket keeps the existing one); a fill that reduces the position keeps the bracket on the remainder; a bracket exit closes the whole position with one market exit (reason TP or SL); `ModifyOrder` cannot change a bracket in v1 (cancel and resubmit). The Portfolio stores the active bracket per position and the journal records every replacement.
4. `max_hold`: the **Engine** (never the broker) counts `held = bar_index - fill_bar_index + 1` per position and, at the close where `held >= max_hold` (the order's, else `EngineConfig.max_hold_default`, else none), submits a reduce-only market exit tagged TIME for the next open; the same code path in backtest, replay, paper and live.
5. Data hole after bar `t` (`breaks[t]`): close at `t`'s close with reason EOW, no entry at `t` (oracle).
6. Paper (SimBroker on live closed bars): a market order with `ts_created` inside bar k (`ts_open(k) <= ts_created < ts_open(k) + tf`) fills at bar k+1's OPEN, applied when the store delivers bar k+1 as closed (the collector writes closed candles only), at `open(k+1) x (1 + side x slip)`; limit and stop orders are evaluated against bar k+1 onward by rule 2; the UI labels fills "bar-resolution paper fill (open of the next bar)". Replay: the same rule, k being the bar last stepped.
7. Live (CcxtBroker): the venue fills; the desk records what the venue reports; the parity of paper against live is reported, not assumed.
8. Equity mark: cash + position quantity x bar close, net of costs paid (oracle definition); per-bar returns include flat bars; `periods_per_year(tf_minutes)` for annualisation.
9. **Per closed bar, in this order:** (1) open: the SimBroker fills pending exits (TIME, strategy exits, `close_position`), the engine applies them to the Portfolio; (2) open: the SimBroker fills pending entries (`size_frac` orders sized on the cash after step 1); (3) intrabar: bracket checks on this bar's high/low (fill bar included), limit and stop fills; (4) close: the engine closes any position at this bar's close with reason EOW when `breaks` follows the bar or (backtest) the source is exhausted, and asks no entry on those bars; else when `held >= max_hold` it queues a TIME exit for the next open and does not ask the strategy; else `strategy.on_bar` runs (for the adapter: `exit_signal` when in a position, `decide` when flat); (5) mark equity at the close and journal.
10. Decisions: the strategy is asked for an entry only when flat, after its warm-up bars, never on the last bar of a backtest block and never on a bar flagged as a break (oracle: `t >= warmup and t < n-1 and not broken`).
11. The adapter hands the oracle's `exit_signal` the entry MID (the fill bar's open), `held` (the fill bar counts as 1) and the original oracle `Order`.
12. A backtest closes any open position at the last bar's close with reason EOW (`mark_to_market_at_end`).
13. **Parity settings:** the parity test runs the BACKTEST account on a margin-like instrument (`is_spot=False`, `price_increment=0`, `size_increment=0`, `min_qty=0`, `min_notional=0`), `RiskLimits.unlimited()` (every limit None, no per-minute cap; shorting allowed) and `EngineConfig(max_hold_default=30)` (= `BacktestConfig.max_hold`); the spot no-short rule and rounding are exercised by `test_risk.py`, never by the parity test.

### A6. Data layer

- `market.sqlite`: table `bars(symbol TEXT, tf INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, src TEXT, PRIMARY KEY(symbol, tf, ts))`, table `sources(symbol, tf, src, first_ts, last_ts, n_bars, updated)`; WAL mode; one writer per process; readers use `read once` semantics. Base timeframe 1 minute; other timeframes are resampled on read (pandas `resample('{tf}min', label='left', closed='left')`: open first, high max, low min, close last, volume sum, drop empty) and cached per (symbol, tf) in memory with an invalidation stamp.
- Importers: (a) repo CSVs (`binance_btcusdt_1min_ccxt.csv`, the local long file; each imported under its own `src` with a provenance text the lead supplies in the item: the tactical journal (`D:/nt/nt_tactical/runs/tactical/LOG.md`, hypothesis H91) reports the long file as Bitstamp BTC/USD with shifted timestamps until 2024-09-13, which is not yet recorded in this repo's docs); (b) ccxt REST backfill `fetch_ohlcv(symbol, '1m', since, limit=1000)` paginated by open time with the built-in rate limiter; (c) the collector's live updates.
- Export for training: `export_csv(symbol, tf, start, end, path)` writes `timestamp,open,high,low,close,volume` (the neural_trade CSV contract) under `var/desk/datasets/<name>.csv` and returns `dataset_fingerprint`-compatible facts (sha256, rows, first/last); scenario YAML copies reference that path through `CSV_PATH`.
- Gaps: kept as missing rows (never forward-filled); `gap_report`-style summary per source; the chart shows gap markers.

### A7. Plugin SDK (R5)

- Two new strict registries, defined in `desk/indicators/api.py` and `desk/core/strategy.py`: `ChartIndicators` and `DeskStrategies` (subclasses of `core/registry.BaseRegistry`, so `list()`, metadata, duplicates and `remove` behave like the ten). Discovery: `load_plugins(d)` for each `settings.plugin_dirs`, default **`plugins_desk/` at the repo root, a sibling of `plugins/` and never inside it**: `load_plugins` rglobs a whole tree and the ten registries load `Config.PLUGINS_DIR` (default None) inside the training process, so desk plugin files under `plugins/` would be executed by a training run that sets `PLUGINS_DIR: plugins`. Desk plugins live in `plugins_desk/indicators/*.py` and `plugins_desk/strategies/*.py`; templates stay in `plugins/templates/desk_*_template.py` (skipped by the loader) and are copied into `plugins_desk/` by the UI. Reload (`desk/plugins_api.reload()`): remove every plugin-registered name from the two registries, drop the `neural_trade_plugins.*` modules from `sys.modules` (the loader treats a module in `sys.modules` as loaded), re-run `load_plugins`; errors reported per file; a restart is the fallback.
- Indicator contract (C3): `META`, `PARAMS` (schema with bounds -> the dialog and sweep spaces), `compute(df, **params) -> DataFrame` (causal, NaN warm-up, aligned to `df.index`), `DRAW` (pane overlay or new; figures line/band/histogram/marker/hline). Acceptance tests every plugin must pass (provided as a pytest helper): causality (compute on a prefix equals the prefix of compute on the full frame), warm-up length, bounds rejection, golden fixture when one is shipped.
- Strategy contract (C2): a `DeskStrategy` subclass with `PARAMS`, `on_bar(ctx)` and optional `on_order_event`, `on_start`, `on_stop`; `ctx` offers `bars(n)`, `indicator(name, **params)` (cached per bar), `predictions()` (when a model is attached), `position`, `account`, `submit_market/limit/stop`, `cancel`, `close`, `log`, `clock.now`. The adapter `ModelSignalStrategy(name, params)` wraps any registered neural_trade `Strategies` entry.
- Templates in `plugins/templates/desk_indicator_template.py` and `desk_strategy_template.py` (skipped by the loader, copied by the UI's "new plugin from template" action into `plugins_desk/indicators/` or `plugins_desk/strategies/`).
- Learnable families are not duplicated as classic plugins; the lens (A2 `learned_bridge.py`) shows them on the chart; a classic twin (for example `rsi_wilder` next to the EWMA-form `rsi` family) is a documented difference, not a bug.

### A8. Model Lab (R6) as a thin layer over the engine

| Lab page | calls |
|---|---|
| Datasets | `desk/lab/datasets.py` (export_csv, repo CSV list with `dataset_fingerprint`), `DatasetSpec.from_config` |
| Scenarios | `panel_data.discover_scenarios`, `Scenario.from_yaml/from_dict/validate/cells`, `Runner.from_spec(...).plan()` (plan_table), `panel_data.field_options/strategy_options/fold_choices/validate_search`; edits are saved as copies under `configs/scenarios/desk/` |
| Runs | `RunStore.index.rows`, `.scores`, `result.json`, `eval_report_<role>.json`, `status.json`, `metrics.jsonl` (`longrun` readers), `run_report.*`; launch = job kind `train` (`neural-trade train` or `scenario run`), progress = `longrun.progress` fields (state, epochs, eta, log tail) |
| Run detail figures | `Visualizations.build(key, data, config, **kw)` for training_dashboard, training_direction_detail, training_loss_terms, eval_report, direction/delta/variance/confidence/coherence analytics, trading_dashboard, trade_analytics, discovered_indicators, indicator_family_periods, permutation_importance -> plotly JSON; inputs built by `run_report.load_saved_blocks` and `scorer.load_block` (npz) |
| Sweeps | job kind `sweep` (`neural-trade sweep ... --dry-run` for the budget, then the run), `panel_data.list_sweeps`, `sweep.json` readers; the confirm-above-1-GPU-hour rule of the panel is kept |
| Leaderboard | `leaderboard_for_scenario(store, scenario, spec=, sync=False)` / `panel_data.build_board` -> `desk/lab/leaderboard.py` JSON; test columns labelled "test, not used for ranking" |
| Compare | `panel_compare.find_compare_spec/stored_verdict/paired_by_fold` (read-only); job kind `compare` for `neural-trade compare SPEC --out` |
| Models | `desk/lab/models.py`: promote `runs/<id>/artifacts` (bundle format 1) into the `models` table with its config facts (symbol, tf, LOOKBACK, horizons, fingerprint) and dev scores; the active model per (symbol, tf) feeds `indicators/predictions.py` and bots |
| Indicator development | the chain: "new plugin from template" -> plugin tests -> (learnable family) screen job -> train job -> learned periods (`period/<name>`) vs textbook -> the lens on the chart |
| Stability, screens | job kinds `stability` (`--dry-run` first) and `screen`; results from `<store>/stability/<id>/REPORT.md` and `screens/<name>/results.jsonl` |

### A9. Web layer and front end (approach B)

- **Transport and panels: NiceGUI 3.17.1 mounted on the FastAPI app** (`ui.run_with(app)`); 3.18.0 needs anyio >= 4.14.2 (the env has 4.13.0), so 3.17.1 is pinned unless the dry run bumps anyio. Every non-chart panel is a Python widget (`ui.table`/`ui.aggrid`, `ui.dialog`, `ui.timer`, `ui.plotly` for the existing figures, `ui.notify` for alerts); the indicator parameter dialog is generated from the plugin's `PARAMS`; the arming and kill-switch dialogs, the order ticket, the bot manager and the Lab pages are panels. Rationale (research): the owner's backend pick is NiceGUI's own stack (fastapi + starlette + uvicorn + pydantic-core), it needs no Node or bundler, custom elements are plain `.js` files with props, events and `run_method`, and the pytest `user` fixture drives pages in-process without a browser, which is what makes panel tasks verifiable for Sonnet-class implementers. Rejected: Panel (second server model on bokeh/tornado), Dash (Flask stack, drawing and chart-side dragging impossible from Python, Selenium tests), Streamlit/Reflex (Python >= 3.11), vanilla JS (control C: 3-5k more hand-written lines, browser-only tests).
- **Two JS elements only.** `lwc_chart.js` wraps the vendored lightweight-charts 5.2.1 standalone build: candles and volume, v5 panes for indicators, `createPriceLine` for order and position lines, `createSeriesMarkers` for trades and bot signals, visible-range control for replay, crosshair and click events emitted to Python, windowed history fetched by the element from `/api/v1/bars` (20k bars per request, "infinite history" on scroll-left), and the drawing primitives ported from the Apache-2.0 plugin examples (trend line, rectangle, horizontal and vertical line, a draggable user price line that emits the new price on drop). `dock.js` wraps dockview-core 8.5.1 (tabs, splits, floating and popout panels, `toJSON`/`fromJSON` persisted in `layouts`). Each element has a short written API (props, methods, events) in its header; logic stays in Python (the server sends ready-to-draw arrays). Element work goes to the escalated implementer (Opus) with Playwright evidence, because the v5 API is newer than most training data.
- **Chart choice by a pre-registered one-day spike each** (DESK-4.0): lightweight-charts 5 + ported primitives against KLineCharts 10.1.0 (Apache-2.0, including its Pro edition; 27 indicators and 16 drawing overlays built in; v10 is three months old and single-author), judged on a draggable order line persisted to Python, a persisted trend line and a 525,600-bar scroll without freezing. The default is lightweight-charts; the spike can only switch the element's implementation, not its API.
- **Conventions page** (`desk/ui/conventions.md`, cited by every UI task): NiceGUI 3.17 snippets (mount, element skeleton, per-client storage, timers), the bulk-data rule, the theme tokens, the "test columns never rank" rule for Lab tables, the account-context banner rule, and the `user`-fixture test skeleton.
- **API (C5) stays** as the contract for the chart element, scripts, QA and the kill switch from curl; panels call the same service functions directly (no REST round trip inside the process). The OpenAPI document remains the committed source of truth for the bulk endpoints.
- **Testing:** panel flows with the `user` fixture; endpoints and the WebSocket with `TestClient`; the two elements and the full pages with 5-10 Playwright smoke tests (chart renders a 20k-bar window and scrolls further, an order line drag lands in sqlite, a drawing persists across reload, the dock layout round-trips, the Lab shows a running job's progress), mainly in CI on Linux; whether the `user` fixture observes `run_method` calls on custom elements is unverified and checked in DESK-4.1.
- **Licences:** everything imported is MIT/BSD/Apache-2.0; lightweight-charts' NOTICE text and a TradingView creator mention go into `THIRD_PARTY_NOTICES.md` with `attributionLogo` left on; dockview's MIT is attested by npm, README and the file header (GitHub reports NOASSERTION) and is recorded there too.

### A10. Persistence, files, secrets

- `var/desk/` (gitignored; `NT_DESK_HOME` env or `--home` overrides): `desk.sqlite`, `market.sqlite`, `datasets/`, `jobs/<id>.log`, `backtests/<id>/` (journal.jsonl, report.json, trades.csv), `token`.
- Secrets only from environment variables (`NT_BINANCE_API_KEY`, `NT_BINANCE_API_SECRET`, `NT_BINANCE_SANDBOX=1`); never in YAML, never logged; `.gitignore` gains `var/` and `.env`.
- Live arming: `NT_DESK_LIVE=1` in the server's environment AND an `arm` request with the typed account id, valid for a session; the UI shows the account context (BACKTEST / PAPER / LIVE) permanently; the kill switch cancels every open order and halts every bot in one call.

### A11. Repository integration (the subpackage decision)

- `src/neural_trade/desk/` is listed in `docs/ARCHITECTURE.md`'s module map (the docs test; the row `| `desk/` | ... |` names no path that does not exist at that commit) as the top layer; a new layering test `test_nothing_imports_desk` is added to `tests/test_layering.py` and scans every subpackage plus `cli.py`, where the only allowed form is an import inside a function (`cli.py` imports `neural_trade.desk.cli` lazily inside the `desk` subparser builder, and `desk/cli.py` builds its parser without importing fastapi, nicegui, ccxt or pandas-heavy modules: the CI `unit` job runs `build_parser()` without the desk extra); `tests/test_hygiene.py` rules apply (no `print`, no bare `except`).
- Test selection: `tests/desk/e2e/conftest.py` and `tests/desk/network/conftest.py` skip their directory unless `NT_DESK_E2E=1` / `NT_DESK_NETWORK=1` is set (and, for e2e, `pytest.importorskip("playwright")` succeeds), so the project's standard fast-suite command, `scripts/test_changed.py` and nightly need no new flags; the CI `desk` job sets the variables. The NiceGUI `user` fixture needs its pytest plugin and async tests: the root `tests/conftest.py` gains `if importlib.util.find_spec("nicegui"): pytest_plugins = ["nicegui.testing.user_plugin"]`, pyproject gains `asyncio_mode = "auto"`, and the `desk-test` extra pins `pytest-asyncio`.
- `pyproject.toml`: extra `desk` = nicegui==3.17.1 (with its fastapi 0.143.2, starlette==1.7.0, uvicorn 0.54.0, pydantic 2.14.1 / pydantic-core 2.50.1, python-socketio, python-engineio, aiohttp 3.14.x, orjson), ccxt==4.5.88 (plus its 15 exact pins as the dry run resolves them), ta==0.11.0, httpx==0.28.1 (installed); optional extras `desk-stream` (talipp==2.7.0), `desk-columnar` (duckdb==1.5.6), `desk-test` (playwright==1.64.0, pytest-playwright==0.10.0); never `fastapi[standard]`; package-data `neural_trade = ["py.typed", "configs/*.yaml", "desk/ui/static/**/*", "desk/ui/elements/*.js"]`; markers `desk`, `e2e`, `network`.
- CI: a new job `desk` that runs on every CI trigger but exits early through a changed-files step (`dorny/paths-filter` or a `git diff --name-only` guard) unless `src/neural_trade/desk/**`, `tests/desk/**`, `pyproject.toml` or `requirements-desk-ci.txt` changed (GitHub path filters exist only at the workflow trigger level); it installs `requirements-ci.txt` + `requirements-desk-ci.txt`, runs `pytest tests/desk -n auto --cov=neural_trade.desk --cov-report=json:coverage_desk.json` (unit, `TestClient` and `user`-fixture tests) then `python scripts/check_coverage.py coverage_desk.json --only desk` (gate 70), then from DESK-4.2 on `python -m playwright install --with-deps chromium` and `NT_DESK_E2E=1 pytest tests/desk/e2e` (traces and screenshots uploaded); the `unit` job gets `--ignore=tests/desk` and `check_coverage.py coverage.json --skip desk` (`check_coverage.py` gains `--only` and `--skip`; otherwise the unit job's coverage.json would show `desk` near 0% and fail the gate). `scripts/test_changed.py` needs no new rule (it matches `test_<module>.py` names and dotted paths in test text). No Node.js anywhere: there are no JS unit tests; the two elements are covered by Playwright.
- The local `nt` env: the lead installs the pinned desk extra after `pip install --dry-run` in a scratch env cloned from `nt` (RUNBOOK "Installing optuna" pattern; disk per D-066) and a smoke import of `neural_trade, nicegui, ccxt, tensorflow`, then `scripts/golden_run.py verify` and the fast suite. The dry run decides three open points: whether ccxt's exact pins and its certifi/cryptography/idna/urllib3 bumps resolve next to TF 2.10 (fallback: the collector in its own venv `D:/nt/venv_desk_collector`, the server stays in `nt`), whether anyio moves to >= 4.14.2 (then nicegui 3.18.0), and whether optional NumPy-2-built wheels (duckdb, TA-Lib) import under numpy 1.23.0 (fallback: numpy 1.23.5, CI's value, with a golden verify; or do without them).
- Owner-level records needed before implementers start: D-067 (the desk track: VISION "Not in the MVP" amended by the owner for "a web application" and "exchange APIs"; D-050 clarified: a live OHLCV feed is the same Binance source, not a new model input; costs: desk accounts explicit, research yardstick stays 0), a ROADMAP track `DESK-0..DESK-6` placed by the owner relative to the MVP milestones, the env install approval (given today for the stack; the exact pin list is confirmed by the dry run).

## Contracts (interfaces, schemas, API)

### C1. Python interfaces (signatures the implementer writes verbatim)

```python
# desk/core/clock.py
class Clock(Protocol):
    def now_ms(self) -> int: ...
class SimClock:            # set by the engine to each closed bar's close time
    def set(self, ts_ms: int) -> None: ...
class LiveClock: ...       # time.time() based; strategies never call time.time()

# desk/core/bus.py
class EventBus:
    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> Callable[[], None]: ...   # returns unsubscribe
    def publish(self, topic: str, event: Any) -> None: ...   # synchronous, handlers in subscription order

# desk/core/fills.py (pure; all prices floats; side +1 buy / -1 sell)
def market_fill(open_: float, side: int, cost: CostProfile, qty: float) -> Fill: ...
def limit_fill(bar: Bar, side: int, limit: float, cost: CostProfile, qty: float) -> Optional[Fill]: ...
def stop_fill(bar: Bar, side: int, trigger: float, cost: CostProfile, qty: float) -> Optional[Fill]: ...
def bracket_exit(bar: Bar, side: int, tp: Optional[float], sl: Optional[float], tiebreak: str) -> Optional[tuple[str, float]]: ...  # ('TP'|'SL', price)

# desk/core/params.py (written in DESK-0.4; shared by strategies and indicators; re-exported by plugins_api as DeskParamSpec to avoid confusion with indicators.base.ParamSpec)
@dataclass(frozen=True)
class ParamSpec: name: str; kind: str; default: Any; minimum: Optional[float] = None; maximum: Optional[float] = None; step: Optional[float] = None; choices: Optional[list] = None; doc: str = ""
def params_schema(specs: tuple[ParamSpec, ...]) -> list[dict]: ...
def validate_params(specs: tuple[ParamSpec, ...], params: dict) -> dict: ...   # raises ValueError naming the bound; fills defaults

# desk/core/journal.py
class Journal:
    def __init__(self, path: Optional[Path], clock: Clock): ...
    def append(self, kind: str, payload: dict) -> None: ...   # one JSON line {seq, ts_event, ts_init, kind, payload}; ts_init = clock.now_ms()
    def close(self) -> None: ...                                # atomic finalisation through utils.atomic.replace_with_retry
    @staticmethod
    def read(path) -> Iterator[dict]: ...

# desk/core/strategy.py
class DeskStrategy:
    name: ClassVar[str]; PARAMS: ClassVar[tuple[ParamSpec, ...]] = (); warmup_bars: ClassVar[int] = 0
    def __init__(self, **params): ...
    def on_start(self, ctx: "StrategyContext") -> None: ...
    def on_bar(self, ctx: "StrategyContext") -> None: ...          # called once per CLOSED bar
    def on_order_event(self, ctx, event: OrderEvent) -> None: ...
    def on_stop(self, ctx) -> None: ...
class StrategyContext:
    bar: Bar; bars: Callable[[int], pd.DataFrame]   # last n closed bars (columns ts, open, high, low, close, volume)
    def indicator(self, name: str, **params) -> pd.DataFrame: ...   # computed on the history, cached per bar
    def predictions(self) -> Optional[dict]: ...                      # Predictor.predict_last-shaped dict or None
    position: Position; account: AccountState; clock: Clock
    def submit_market(self, side: str, *, qty: Optional[float] = None, size_frac: Optional[float] = None, tp=None, sl=None, tp_is_offset: bool = False, max_hold: Optional[int] = None, tag="") -> str: ...   # exactly one of qty / size_frac
    def submit_limit(self, side, price, *, qty=None, size_frac=None, tif="GTC", tp=None, sl=None, tp_is_offset=False, tag="") -> str: ...
    def submit_stop(self, side, trigger, *, qty=None, size_frac=None, limit=None, tag="") -> str: ...
    def cancel(self, client_id: str) -> None: ...
    def close_position(self, tag="") -> None: ...
    def log(self, msg: str) -> None: ...

# desk/core/engine.py
@dataclass
class EngineConfig: mode: str  # 'backtest'|'replay'|'paper'|'live'
    instrument: Instrument; tf_minutes: int; cost: CostProfile; initial_cash: float = 10_000.0
    tiebreak: str = 'sl_first'; max_hold_default: Optional[int] = None; risk: RiskLimits = field(default_factory=RiskLimits)
class Engine:
    def __init__(self, cfg: EngineConfig, strategy: DeskStrategy, source: DataSource, broker: Broker, bus: EventBus, clock: Clock, journal: Journal): ...
    def run(self) -> EngineRun: ...          # backtest: until the source is exhausted; per bar exactly the A5.9 order
    def step(self) -> Optional[Bar]: ...     # replay/paper/live: process one closed bar; None when nothing new
    def halt(self, reason: str) -> None: ...
    # the Engine schedules TIME and EOW exits itself (A5.4, A5.9) so SimBroker and CcxtBroker behave the same

# desk/data/sources.py
class DataSource(Protocol):
    def next_closed_bar(self) -> Optional[Bar]: ...
    def history(self, n: int) -> pd.DataFrame: ...

# desk/broker/base.py
class Broker(Protocol):
    def submit(self, order: Order) -> OrderEvent: ...
    def cancel(self, client_id: str) -> OrderEvent: ...
    def fill_exits(self, bar: Bar) -> list[Fill | OrderEvent]: ...                 # A5.9 step 1 (SimBroker: pending exits at the open; CcxtBroker: venue reports)
    def fill_entries(self, bar: Bar, cash: float) -> list[Fill | OrderEvent]: ...  # A5.9 step 2 (size_frac resolved on `cash`, the post-exit realised equity)
    def on_bar_intrabar(self, bar: Bar) -> list[Fill | OrderEvent]: ...            # A5.9 step 3 (brackets, limits, stops on high/low)
    def sync(self) -> list[Fill | OrderEvent]: ...                                  # reconciliation (live)

# desk/indicators/api.py
@dataclass(frozen=True)
class IndicatorMeta: id: str; name: str; version: str; category: str; inputs: tuple[str, ...]; warmup: int; doc: str = ""
@dataclass(frozen=True)
class Figure: key: str; type: str   # 'line'|'band'|'histogram'|'marker'|'area'|'hline'
    color: Optional[str] = None; width: float = 1.5; style: str = 'solid'; fill_to: Optional[str] = None; levels: tuple[float, ...] = ()
@dataclass(frozen=True)
class DrawSpec: pane: str  # 'overlay'|'new'
    scale: str = 'auto'; precision: Optional[int] = None; figures: tuple[Figure, ...] = ()
class ChartIndicators(BaseRegistry): ...   # component = a module-like object with META, PARAMS, DRAW, compute
def register_indicator(meta: IndicatorMeta, params: tuple[ParamSpec, ...], draw: DrawSpec): ...  # decorator for compute(df, **params) -> DataFrame
def check_indicator(component, df: pd.DataFrame) -> None: ...   # the contract tests used by pytest and by the plugin page

# desk/jobs/model.py
@dataclass
class Job: id: str; kind: str; params: dict; state: str; pid: Optional[int]; log_path: str; created_ms: int; started_ms: Optional[int]; finished_ms: Optional[int]; result: Optional[dict]; error: Optional[str]; gpu: bool
```

### C2. Model-signal adapter (parity with the existing engine)

`ModelSignalStrategy(strategy_name, params, model_id=None)`: in `on_start` it builds the neural_trade strategy with `build_strategy(name, params, calibration=cal_signalframe)` where the calibration SignalFrame comes from the promoted model's stored `predictions_cal.npz` (`scorer.load_block`) or from `price_only_frame` for price-only strategies; it accepts **discrete strategies only** and raises `UnsupportedStrategy("exposure-mode strategies (vol_target, net_edge_kelly) need the rebalance path, DESK-1.12")` for an `ExposureStrategy`. In backtest it builds the `SignalFrame` once over the whole block and calls `decide(s, t)` / `exit_signal(s, t, side, held, entry_mid, order)` with the block-absolute index `t` (RandomSignal draws its schedule from `len(s)` and indexes by `t`); in paper/live it rebuilds a trailing SignalFrame from `Predictor.predict_windows_frame` over the last `max(warmup, 300)` bars. It maps the oracle `Order` field by field: `ctx.submit_market(side, size_frac=order.size_frac, tp=order.tp, sl=order.sl, tp_is_offset=order.tp_is_offset, max_hold=order.max_hold, tag=order.reason)` and keeps the oracle `Order` in the desk order's `tags` to hand it back to `exit_signal`. **Parity test (acceptance of DESK-1.9, settings A5.13):** for `ta_ma_cross`, `ta_rsi`, `ta_bollinger`, `calibrated_quantile` (with a stored calibration block) and `random_signal` on `tests/conftest.py::synthetic_bars` and on a stored engine cell, the desk engine's trades (entry bar, exit bar, prices, reasons) and final equity equal `strategy.backtest.run_backtest`'s within 1e-9 at the same `BacktestConfig` cost profile.

### C3. Indicator plugin file (what a plugin author writes)

```python
# plugins_desk/indicators/my_rsi.py   (a sibling of plugins/, never inside it: A7)
from neural_trade.desk.plugins_api import register_indicator, IndicatorMeta, ParamSpec, DrawSpec, Figure
import numpy as np, pandas as pd

@register_indicator(
    IndicatorMeta(id="my_rsi", name="RSI (Wilder)", version="1.0.0", category="momentum", inputs=("close",), warmup=14),
    params=(ParamSpec("length", "int", 14, 2, 500, 1),),
    draw=DrawSpec(pane="new", scale="fixed_0_100", figures=(Figure("rsi", "line", levels=(30, 70)),)),
)
def compute(df: pd.DataFrame, *, length: int) -> pd.DataFrame:
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    rsi.iloc[: length] = np.nan
    return pd.DataFrame({"rsi": rsi}, index=df.index)
```

Rules checked by `check_indicator`: output columns equal the figure keys; index equals `df.index`; first `warmup` rows NaN; causality (prefix invariance on three cut points); params outside bounds raise `ValueError`; compute is deterministic; when a golden fixture `<plugin file stem>.golden.npz` exists next to the file (arrays `t`, `inputs`, one array per figure key, `params` as JSON; made by `scripts/desk/make_indicator_fixtures.py --plugin <file>`), outputs match it within 1e-8.

### C4. Desk database schema (`var/desk/desk.sqlite`, versioned migrations in `store/db.py`)

```
accounts(id TEXT PK, kind TEXT, quote TEXT, cash REAL, cost_profile TEXT(JSON), created_ms INTEGER, armed_until_ms INTEGER)
orders(client_id TEXT PK, account_id, instrument_id, strategy_id, bot_id, side, type, qty REAL, size_frac REAL, price REAL, trigger_price REAL, tif, reduce_only INTEGER, bracket TEXT(JSON), status, filled_qty REAL, avg_fill_price REAL, reason, venue_id, ts_created INTEGER, ts_updated INTEGER, tags TEXT(JSON))
fills(trade_id TEXT PK, client_id, instrument_id, account_id, side, qty REAL, price REAL, mid REAL, fee REAL, fee_currency, liquidity, ts INTEGER)
positions(position_id TEXT PK, account_id, instrument_id, strategy_id, side, qty REAL, avg_price REAL, realised_pnl REAL, fees REAL, ts_opened INTEGER, ts_closed INTEGER)
journal(id INTEGER PK, ts INTEGER, account_id, kind, payload TEXT(JSON))              -- every event, append-only
bots(id TEXT PK, name, strategy, params TEXT(JSON), instrument_id, tf INTEGER, account_id, model_id, state, risk TEXT(JSON), created_ms, last_bar_ms, error)
alerts(id TEXT PK, name, condition TEXT(JSON), frequency, state, created_ms, last_fired_ms, message)
alert_log(id INTEGER PK, alert_id, ts INTEGER, payload TEXT(JSON))
layouts(id TEXT PK, name, payload TEXT(JSON), updated_ms)                               -- symbol, tf, panes, indicators, chart settings
drawings(id TEXT PK, layout_id, instrument_id, tool, points TEXT(JSON), style TEXT(JSON), locked INTEGER, updated_ms)
models(id TEXT PK, run_dir, bundle_dir, symbol, tf INTEGER, lookback INTEGER, horizons TEXT(JSON), fingerprint TEXT, dev_scores TEXT(JSON), promoted_ms, active_for TEXT(JSON))
jobs(id TEXT PK, kind, params TEXT(JSON), state, pid INTEGER, gpu INTEGER, log_path, created_ms, started_ms, finished_ms, result TEXT(JSON), error)
backtests(id TEXT PK, name, strategy, params TEXT(JSON), instrument_id, tf, start_ms, end_ms, cost_profile TEXT(JSON), job_id, report_path, created_ms)
settings(key TEXT PK, value TEXT(JSON))
```

### C5. HTTP and WebSocket API (`/api/v1`, JSON; errors `{error: {code, message}}`; mutating calls need `X-Desk-Token`)

| area | endpoints |
|---|---|
| system | `GET /system/health` {version, home, accounts, gpu_free, collector}; `GET /system/theme` (CSS tokens); `POST /system/kill` |
| bars | `GET /bars?symbol&tf&start&end&limit` -> arrays {t[], o[], h[], l[], c[], v[]} (typed, UTC ms); `GET /symbols`; `POST /bars/import` {source} -> job; `POST /bars/backfill` {symbol, since} -> job |
| indicators | `GET /indicators` (catalogue with META/PARAMS/DRAW); `POST /indicators/compute` {id, params, symbol, tf, start, end} -> {figures: {key: values[]}, t[]}; `GET /indicators/learned?model_id&window=last` (the lens) |
| plugins | `GET /plugins` (name, kind, file, status, test result); `POST /plugins/reload`; `POST /plugins/new` {kind, name} (from template); `POST /plugins/test` {id} |
| accounts | `GET /accounts`; `POST /accounts/{id}/arm` {confirm: id}; `GET /accounts/{id}/summary` |
| orders | `GET /orders?account&status`; `POST /orders` body `SubmitOrderRequest {account_id, instrument_id, side, type, qty | size_frac, price?, trigger_price?, tif=GTC, reduce_only=false, bracket?, tags?}` (the server assigns client_id (uuid4), status, filled_qty, ts_created/ts_updated and returns the Order); `PATCH /orders/{client_id}` {price, trigger_price, qty}; `DELETE /orders/{client_id}`; `GET /fills?account` |
| positions | `GET /positions?account`; `POST /positions/{id}/close` |
| bots | `GET /bots`; `POST /bots` {name, strategy, params, symbol, tf, account_id, model_id, risk}; `POST /bots/{id}/start|stop`; `GET /bots/{id}/log` |
| alerts | `GET/POST /alerts`; `DELETE /alerts/{id}`; `GET /alerts/log` |
| backtests | `POST /backtests` {strategy, params, symbol, tf, start, end, cost_profile, initial_cash} -> job; `GET /backtests`; `GET /backtests/{id}` (report); `GET /backtests/{id}/trades`; `GET /backtests/{id}/equity` |
| replay | `POST /replay/start` {symbol, tf, start_ms, strategy?}; `POST /replay/{id}/step` {n}; `POST /replay/{id}/order` (manual order inside the replay); `GET /replay/{id}` |
| layouts | `GET/POST /layouts`, `PUT/DELETE /layouts/{id}`; `GET/PUT /layouts/{id}/drawings` |
| lab | `GET /lab/datasets`, `POST /lab/datasets/export`; `GET /lab/scenarios`, `GET /lab/scenarios/{name}` (spec + plan table), `POST /lab/scenarios` (copy with edits, validated); `GET /lab/runs?scenario`, `GET /lab/runs/{id}` (meta, status, result, eval report), `GET /lab/runs/{id}/progress`, `GET /lab/runs/{id}/figure/{key}` (plotly JSON); `POST /lab/train` {scenario|config overrides, csv} -> job; `POST /lab/sweeps/estimate`, `POST /lab/sweeps` -> job, `GET /lab/sweeps`; `GET /lab/leaderboard?scenarios=`; `GET /lab/compare/{spec}`; `POST /lab/compare` -> job; `GET /lab/models`, `POST /lab/models/promote` {run_dir}, `POST /lab/models/{id}/activate` {symbol, tf}; `GET /lab/predictions?model_id&symbol&tf&n` (overlay arrays) |
| jobs | `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/log?tail`, `POST /jobs/{id}/cancel` |
| ws | `GET /ws?token` ; client sends `{op: 'sub', topics: [...]}`; server sends `{topic, ts, data}` for `bars.<symbol>.<tf>` (closed and forming), `orders`, `fills`, `positions`, `account.<id>`, `bots`, `alerts`, `jobs`, `lab.runs` |

Every endpoint has a pydantic request and response model in `desk/api/schemas.py`; the OpenAPI document is committed to `docs/desk/openapi.json` and a test fails when it drifts (so QA and the front end have one source of truth).

## Task plan for Sonnet-class implementers

Conventions: every task is one backlog item (NT-215 onward, filed by the lead at step 0 with the milestone tag below; IDs here are stable task keys inside this plan), one branch `nt-<id>`, one implementer run (Sonnet 5.5 medium), QA by risk (P1: `qa` agent; P2 mechanical: lead-verified). **Size:** S under 150 lines of code plus tests, M 150-400, L 400-800 (an implementer that finds a task larger than L stops and reports; the lead splits it). **Files** lists what may change. **Acceptance** is what QA checks; every task also needs: tests under `tests/desk/` passing with `-m "not e2e and not network"`, ruff clean, no TensorFlow import at `import neural_trade.desk` (the DESK-0.2 test), fast suite unaffected, CI `desk` job green. Each milestone ends in a runnable increment the owner can open.

### DESK-0 Foundations (the lead's step 0 plus seven implementer tasks)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 0.1 | **Land the plan** (lead): copy this plan to `docs/desk/PLAN.md`, the contracts to `docs/desk/CONTRACTS.md`, the API to `docs/desk/API.md`; file the items in BACKLOG with milestone tags DESK-0..6; add the ROADMAP track; write D-067 (the desk track, the VISION amendment request, the cost-profile rule, the live-feed clarification of D-050) and the open owner questions into STATUS | - | owner go-ahead | docs only | the owner can open `docs/desk/PLAN.md`; BACKLOG rows exist; STATUS lists the questions |
| 0.2 | **Package skeleton and guards**: `desk/__init__.py`, `desk/config.py` (`DeskSettings`: home dir from `--home`/`NT_DESK_HOME`/default `var/desk`, host/port, plugin dirs (default `plugins_desk/`), default accounts and cost profiles, loaded from `configs/desk.yaml` + env), `var/` and `.env` in `.gitignore`, the ARCHITECTURE module-map row (`| `desk/` | ... |`, naming no path that does not exist yet), `tests/test_layering.py::test_nothing_imports_desk` (subpackages plus `cli.py`; function-local imports allowed in `cli.py`), markers `desk`, `e2e`, `network` and `asyncio_mode = "auto"` in pyproject, the root `tests/conftest.py` line `if importlib.util.find_spec("nicegui"): pytest_plugins = ["nicegui.testing.user_plugin"]`, `tests/desk/conftest.py` (tmp home fixture), `tests/desk/e2e/conftest.py` and `tests/desk/network/conftest.py` (self-skip unless `NT_DESK_E2E=1` / `NT_DESK_NETWORK=1`), pyproject extra `desk` (pins from 0.3) and package-data `desk/ui/static/**/*`, `desk/ui/elements/*.js` | M | 0.1 | the files named; `pyproject.toml`; `.gitignore`; `docs/ARCHITECTURE.md`; `tests/test_layering.py`; `tests/conftest.py` | (1) `python -c "import neural_trade.desk"` with `sys.modules` checked: no `tensorflow`, `fastapi`, `nicegui`, `ccxt` loaded (test); (2) `DeskSettings.load()` precedence env > yaml > defaults (test); (3) `test_nothing_imports_desk` passes and fails on a planted import (mutation shown in the report); (4) `test_docs_guides` passes with the new row; (5) the standard fast-suite command and `scripts/test_changed.py --run` stay green with the e2e and network directories skipped; (6) fast suite count unchanged except the new tests |
| 0.3 | **Environment and CI** (blocked until questions 0 and 4 are answered; lead installs, implementer writes the files): `requirements-desk.txt` = the A11 `desk` extra verbatim with the exact transitive versions the dry run resolves (nicegui==3.17.1, fastapi==0.143.2, starlette==1.7.0, uvicorn[standard]==0.54.0, pydantic==2.14.1, pydantic-core==2.50.1, python-socketio, python-engineio, aiohttp 3.14.x, orjson, ccxt==4.5.88 and its pins, ta==0.11.0, httpx==0.28.1; `desk-test`: playwright==1.64.0, pytest-playwright==0.10.0, pytest-asyncio) and `requirements-desk-ci.txt` (Linux CI pins); `.github/workflows/ci.yml` job `desk` as in A11 (changed-files guard, pytest with `--cov=neural_trade.desk`, `check_coverage.py --only desk`, Playwright step from DESK-4.2 on with `NT_DESK_E2E=1`); `unit` gets `--ignore=tests/desk` and `check_coverage.py coverage.json --skip desk`; `scripts/check_coverage.py` gains `--only`/`--skip` and `GATES["desk"] = 70`; RUNBOOK "Desk" section (install, dry run, run, traps) | M | 0.2 | the files named | (1) `pip install --dry-run -r requirements-desk.txt` output committed to `docs/desk/env_dry_run.txt` shows no change to numpy, protobuf, tensorflow, keras, h5py; (2) `python -c "import neural_trade, nicegui, fastapi, ccxt, ta, tensorflow"` succeeds in `nt` and `scripts/golden_run.py verify` against the current golden record passes (lead's log in the item); (3) CI `desk` and `unit` jobs green on the pushed head; `unit` wall time within noise of the previous run |
| 0.4 | **Domain model, bus, clock, params**: `desk/core/model.py` (A4 dataclasses, enums, the complete status-edge list, `is_closed`, JSON (de)serialisation helpers), `desk/core/params.py` (C1 `ParamSpec`, `params_schema`, `validate_params`), `desk/core/bus.py`, `desk/core/clock.py` | M | 0.2 | `desk/core/{model,params,bus,clock}.py`, `tests/desk/test_core_model.py`, `test_params.py`, `test_bus.py` | (1) the A4 status-edge list tested exhaustively (every listed edge accepted, every other edge raises `IllegalTransition`, the `late_fill` exception edge included); (2) `Order/Fill/Position/AccountState` round-trip through `to_json/from_json`; (3) bus delivers in subscription order and unsubscribe works; (4) `SimClock.set` monotonic check raises on going back; (5) `validate_params` fills defaults and names the violated bound in its `ValueError` |
| 0.5 | **Cost profiles and pure fill functions**: `desk/core/costs.py` (`CostProfile` with `research_zero` and `binance_spot_taker` presets, per-side maths identical to `BacktestConfig.slip_rate/fee_rate`), `desk/core/fills.py` (C1) | M | 0.4 | `desk/core/{costs,fills}.py`, `tests/desk/test_fills.py` | (1) market fill price and costs equal `strategy/backtest.py` single-trade arithmetic for long and short at 0 and at 10/1/2 bps (numbers in the test); (2) limit touch/gap cases, stop trigger/gap cases, bracket tie `sl_first`/`tp_first`, all with hand-computed expected prices; (3) property test: fills never outside the bar's [low, high] except gap fills at the open |
| 0.6 | **Desk database**: `desk/store/db.py` (sqlite, WAL, versioned migrations, repositories for every C4 table, read-once helpers) | L | 0.4 | `desk/store/db.py`, `tests/desk/test_db.py` | (1) schema version 1 creates every C4 table; a second open is a no-op; (2) each repository round-trips its dataclass; (3) two processes (a subprocess in the test) write orders concurrently without error (WAL + busy timeout); (4) journal is append-only (update raises) |
| 0.7 | **Market store**: `desk/data/store.py` (A6: `write_bars`, `read_bars(symbol, tf, start, end, limit)`, `last_ts`, `ranges`, `resample`, `export_csv`, `gaps`) | M | 0.6 | `desk/data/store.py`, `tests/desk/test_market_store.py` | (1) 5-minute resample of `synthetic_bars` equals `resample_bars(standardize_ohlcv(synthetic_bars, cfg), cfg)` with `cfg = Config(); cfg.RESAMPLE_MINUTES = 5` (same OHLCV values, same timestamps); (2) `export_csv` output passes `run_preprocessors(df, Config())` + `validate_ohlcv_frame(bar_minutes=1)` and `dataset_fingerprint` facts match (rows, first, last); (3) duplicate writes are idempotent; (4) gap detection matches `data.gaps.gap_flags` on a frame with holes |
| 0.8 | **Importers and CLI**: `desk/data/importers.py` (repo CSV import through the neural_trade preprocessors, labelled source), `desk/cli.py` with `init`, `import-csv`, registered as `neural-trade desk` in `cli.py` through a lazy import inside the subparser builder | M | 0.7 | `desk/data/importers.py`, `desk/cli.py`, `src/neural_trade/cli.py` (one subparser hook), `tests/desk/test_importers.py`, `tests/desk/test_cli.py` | (1) importing the bundled CSV yields 43,500 bars with `src='repo:binance_btcusdt_1min_ccxt.csv'` (marker `data`); (2) `neural-trade desk init --home TMP` creates both sqlite files and the token; (3) `cli.build_parser` parses `neural-trade desk import-csv FILE` (docs test contract); (4) the `sources` row of an imported file carries the provenance text given in the item (default: the file name and span); (5) a subprocess test runs `build_parser()` with `sys.modules` guards for fastapi, nicegui and ccxt and asserts none loaded (the CI `unit` job has no desk extra) |

### DESK-1 Engine and backtest (R4)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 1.1 | **Portfolio**: `desk/core/portfolio.py` (NETTING positions, cash, fees, realised/unrealised, equity mark per bar, drawdown; spot rule: no negative base balance) | M | 0.5 | `desk/core/portfolio.py`, `tests/desk/test_portfolio.py` | (1) a long round trip at 10/1/2 bps reproduces `Trade.gross_pnl/costs/net_pnl` of the oracle for the same prices; (2) a fill crossing zero closes and opens positions (two `PositionEvent`s); (3) equity series equals the oracle's `equity` on a scripted 20-bar scenario |
| 1.2 | **Risk guard**: `desk/core/risk.py` (`RiskLimits`, rounding to `price_increment`/`size_increment`, min/max notional, max position, max orders per minute, daily loss limit, trading state ACTIVE/REDUCING/HALTED, spot no-short) | M | 1.1 | `desk/core/risk.py`, `tests/desk/test_risk.py` | (1) each limit has a passing and a denying test with the `reason` string; (2) REDUCING allows only position-reducing orders; (3) HALTED denies everything and `halt` is idempotent |
| 1.3 | **Execution engine**: `desk/core/execution.py` (commands -> RiskGuard -> Broker; events -> state machine -> Portfolio -> bus; idempotent client ids; overfill check) | L | 1.2 | `desk/core/execution.py`, `tests/desk/test_execution.py` | (1) scripted event sequences (accept, partial, fill, cancel, reject, late fill after cancel) end in the states of the A4 machine; (2) a duplicate `trade_id` is ignored; (3) every state change is journalled with `ts_event` and `ts_init` |
| 1.4 | **Simulated broker**: `desk/broker/sim.py` (A5 rules on closed bars: next-open market, limit/stop in bar, brackets with tiebreak, `max_hold` TIME, EOW on breaks) | L | 1.3 | `desk/broker/{base,sim}.py`, `tests/desk/test_sim_broker.py` | (1) each A5 rule has a scripted test with expected fill bar and price; (2) same-bar entry-fill then bracket check on that bar (oracle order of operations) tested; (3) no fill ever precedes the order's bar |
| 1.5 | **Data sources**: `desk/data/sources.py` (`HistoricalSource`, `ReplaySource` with step and seek, `history(n)` as a DataFrame; `seek(ts)` rebuilds the source position and the replay session (3.7) creates a new Engine and SimClock on a backward seek) | S | 0.7 | `desk/data/sources.py`, `tests/desk/test_sources.py` | (1) iteration yields closed bars in order with `is_closed=True`; (2) `history(n)` never includes a bar after the current one (causality) |
| 1.6 | **Strategy API and built-ins**: `desk/core/strategy.py` (C1 `DeskStrategy`, `StrategyContext`, `DeskStrategies` registry; `ParamSpec` imported from `core/params.py`; `ctx.indicator(name, **params)` raises `IndicatorsUnavailable` until 7.1 wires it to `ChartIndicators`), built-ins `desk/strategies/{ma_cross,rsi_threshold,bollinger_breakout,buy_and_hold,always_flat,random_entries}.py`; the three TA built-ins compute their arrays with `neural_trade.strategy.ta_rules.sma`, `rolling_std` and `wilder_rsi` on `ctx.bars(n)["close"]` (reuse, not re-implementation) and copy the entry and exit rules of `ta_rules.py` (enter while flat on the sign, exit on CROSS / RSI_MID / BB_MID, `allow_short`, `max_hold` 1440, warm-up = slow or period) | L | 1.4, 1.5 | the files named, `tests/desk/test_strategy_api.py` | (1) `params_schema()` lists every `ParamSpec` with bounds; constructing with an out-of-range param raises; (2) each built-in runs on `synthetic_bars` and the three TA rules place the same entry bars as neural_trade's `ta_ma_cross/ta_rsi/ta_bollinger` with the same parameters (bar indices compared); (3) `ctx.indicator` raises `IndicatorsUnavailable` with a message naming 7.1 |
| 1.7 | **Engine loop and journal**: `desk/core/engine.py` (backtest `run`, `step`, halt; exactly the A5.9 order per bar: fill_exits, fill_entries on the post-exit cash, intrabar, close-of-bar precedence EOW > TIME > strategy, mark and journal; the Engine schedules TIME and EOW); `desk/core/journal.py` (C1 `Journal` with `ts_init` from the engine's Clock, atomic finalisation; `EngineRun` summary) | L | 1.6 | `desk/core/engine.py`, `desk/core/journal.py`, `tests/desk/test_engine.py`, `test_journal.py` | (1) two runs of the same backtest produce byte-identical journals (SimClock timestamps); (2) a strategy exception halts the run with reason and leaves the journal readable; (3) events per bar appear in the A5.9 order (test reads the journal); (4) a `size_frac` entry at an open where an exit also fills is sized on the post-exit cash (scripted test) |
| 1.8 | **Backtest report**: `desk/core/report.py` (Strategy-Tester fields: net/gross profit and loss, commissions, buy-and-hold return, max run-up/drawdown with durations, total/winning/losing trades, percent profitable, avg trade, avg win/loss and ratio, largest win/loss, avg bars, max consecutive wins/losses, profit factor, Sharpe/Sortino via `strategy/performance.py`, exposure, turnover; gross on `Fill.mid`, costs = qty x |price - mid| + fee; baselines buy-and-hold, always-flat, matched random entries (engine re-run with `random_entries` at the strategy's rate/hold/size; seeds default 100 up to 100k bars, 20 above, configurable; the report records `n_seeds` and the null's wall time); n_eff bands via `metrics.statistics`) | L | 1.7 | `desk/core/report.py`, `tests/desk/test_report.py` | (1) on a scripted journal every field equals a hand-computed value; (2) Sharpe equals `performance.sharpe(returns, periods_per_year(tf))`; (3) the random-null percentile fields follow `random_same_frequency`'s key names; (4) report JSON schema committed and snapshot-tested; (5) the null on the bundled 43,500 bars with 100 seeds finishes under 60 s on CPU |
| 1.9 | **Model-signal adapter and the parity test** (P1, `qa`): `desk/strategies/model_signal.py` (C2, settings A5.13) | L | 1.8 | `desk/strategies/model_signal.py`, `tests/desk/test_parity.py` | (1) parity within 1e-9 on `synthetic_bars` for `ta_ma_cross`, `ta_rsi`, `ta_bollinger`, `random_signal` (trades, reasons, equity) against `strategy.backtest.run_backtest` at zero cost and at 10/1/2 bps; (2) parity for `calibrated_quantile` on a stored engine cell's `predictions_cal/oos.npz` (marker `data`, skipped when absent); (3) `assert_no_lookahead`-style perturbation test passes for the adapter; (4) an `ExposureStrategy` raises `UnsupportedStrategy` naming DESK-1.12 |
| 1.10 | **Backtest CLI**: `neural-trade desk backtest --strategy --symbol --tf --start --end --cost` runs the engine in-process and writes `var/desk/backtests/<id>/{journal.jsonl,report.json,trades.csv,equity.csv}`; the subprocess job kind `backtest` is added in 3.9's kinds list | M | 1.8 | `desk/cli.py`, `desk/core/backtest_run.py`, `tests/desk/test_backtest_cli.py` | (1) the CLI run on the bundled CSV produces the four files and exit 0; (2) the four files are written atomically (no half-written report after an injected failure); (3) the DESK-1 milestone command in 'Verification' works as written |
| 1.11 | **Causality harness for desk strategies**: `desk/testing/causality.py` (perturb bars after probe t, re-run, compare decisions up to t) applied to every registered built-in (1.6) | M | 1.7 | `desk/testing/causality.py`, `tests/desk/test_causality.py` | (1) a planted look-ahead strategy (reads `bars(n+1)`) is caught; (2) every built-in passes with 12 probes |
| 1.12 | **Exposure-strategy adapter** (after DESK-1): `target(s, t, current)` -> `ctx.rebalance_to(exposure)` with the band and `decide_every` rules of `run_exposure_backtest`; parity test against `run_exposure_backtest` on `synthetic_bars` for `vol_target` with an EWMA sigma | M | 1.9 | `desk/strategies/model_signal.py`, `desk/core/engine.py` (rebalance path), tests | (1) rebalances, costs and equity equal the oracle within 1e-9; (2) the discrete parity test (1.9) is unchanged |

### DESK-2 Market data live and paper trading (backend of R2 and R3)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 2.1 | **Exchange client**: `desk/data/exchange.py` (`ExchangeClient` protocol; `FakeExchange` for tests: scripted bars, orders, fills, balances; `CcxtBinanceSpot`: `load_markets` -> `Instrument`, `fetch_ohlcv` paging, balances, create/cancel/fetch orders, `set_sandbox_mode` first, rate limiter on) | L | 0.4 | `desk/data/exchange.py`, `tests/desk/test_exchange.py` (+ `tests/desk/network/test_binance_sandbox.py` marked `network`) | (1) `Instrument` fields from a recorded `load_markets` fixture (precision, min notional); (2) paging stops at `since` and never duplicates; (3) the network test (sandbox keys from env, skipped otherwise) places and cancels one limit order far from the market |
| 2.2 | **Backfill**: `desk/data/backfill.py` with `backfill(exchange, store, symbol, since, progress: Callable[[int, int], None]) -> int` (paginated REST into the store, a resumable cursor in `sources`); the 10-line job kind wrapper comes with 3.9 | S | 2.1 | the file, `tests/desk/test_backfill.py` | (1) 3,500 fake bars arrive in 4 pages, store has 3,500, `sources` updated; (2) resumable after an interruption (cursor test) |
| 2.3 | **Collector**: `desk/data/collector.py` + `neural-trade desk collector` (REST polling every N s; closed-bar rule A3; reconcile the last two candles after errors; heartbeat file) | M | 2.1 | `desk/data/collector.py`, `desk/cli.py`, `tests/desk/test_collector.py` | (1) with `FakeExchange` and a fake clock the forming candle is never written as closed; (2) a candle is written exactly once when the next open time appears; (3) after a simulated outage the gap is backfilled; (4) heartbeat file updated each loop |
| 2.4 | **Live source and bar-close scheduler**: `desk/data/sources.py::LiveSource` (store watcher by `last_ts`), `desk/bots/scheduler.py` (ticks on each new closed bar per (symbol, tf)) | M | 2.3 | the files, `tests/desk/test_live_source.py` | (1) a bar inserted into the store surfaces exactly once; (2) 5-minute consumers tick only when a 5-minute bar completes |
| 2.5 | **Bot runner**: `desk/bots/runner.py` (one `Engine` per bot in paper or live mode; start/stop/kill; status, heartbeat, last bar, error; persisted in `bots`; restart recovery re-creates running bots) | L | 2.4, 1.7 | `desk/bots/runner.py`, `tests/desk/test_bot_runner.py` | (1) a bot on `FakeExchange` bars trades on paper and its orders/fills appear in the DB; (2) `kill_all` cancels open orders and halts every bot within one tick; (3) after a simulated restart the bot resumes from its last bar without duplicate orders (client ids) |
| 2.6 | **Manual paper trading**: `desk/trading/manual.py` (manual orders through the ExecutionEngine with the paper account's SimBroker on live closed bars; modify/cancel; account summary) | M | 2.5 | `desk/trading/manual.py`, `tests/desk/test_manual_trading.py` | (1) a market order placed between bars fills at the next closed bar's open (A5.6) and the position/P&L update; (2) modify of a resting limit changes price without a new client id; (3) summary fields equal the Portfolio's |
| 2.7 | **Alerts**: `desk/bots/alerts.py` (conditions price/indicator/prediction/strategy-event with operators crossing up/down, above, below; frequency once, every bar close, once per bar; log; WS event) | M | 2.4 | `desk/bots/alerts.py`, `tests/desk/test_alerts.py` | (1) each operator and frequency has a scripted test; (2) alerts evaluate only on closed bars; (3) the log row carries the triggering values |

### DESK-3 Web API (bulk data for the chart element, scripts, QA and the kill switch; panels call the same service functions in-process)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 3.1 | **App factory**: `desk/api/app.py` (FastAPI, lifespan starting/stopping the store watcher, bot runner, job runner; token middleware; static mount; `/system/health`, `/system/theme` from `visualization/theme.py` tokens) + `neural-trade desk serve` | M | 0.6 | `desk/api/{app,auth,system}.py`, `desk/cli.py`, `tests/desk/test_app.py` | (1) `TestClient(app).get('/api/v1/system/health')` 200 with the documented keys; (2) a mutating call without the token is 401; (3) `serve` binds 127.0.0.1 only (config test); (4) the server process has `CUDA_VISIBLE_DEVICES=-1` set before any import (test on the env of the CLI entry) |
| 3.2 | **Schemas and OpenAPI snapshot**: `desk/api/schemas.py` (every C5 request/response), `docs/desk/openapi.json`, drift test | M | 3.1 | the files, `tests/desk/test_openapi.py` | (1) `app.openapi()` equals the committed file (test prints the diff); (2) every schema round-trips the matching `core/model.py` dataclass |
| 3.3 | **Bars and symbols routers** | S | 3.2, 0.7 | `desk/api/routers/bars.py`, tests | (1) typed arrays with UTC ms; `limit` and ranges honoured; (2) `tf` not in the allowed list is 422 |
| 3.4 | **Indicators and plugins routers** (catalogue from `ChartIndicators`, compute with caching key (id, version, params, symbol, tf, range), lens endpoint, plugin list/reload/new/test) | M | 3.2, 7.x (plugins milestone) | `desk/api/routers/{indicators,plugins}.py`, tests | (1) compute output keys equal the DRAW figure keys; (2) bad params 422 with the bound in the message; (3) reload picks up a newly written plugin file in the tmp plugin dir; (4) `new` writes from the template and `test` returns `check_indicator` results |
| 3.5 | **Accounts, orders, positions, fills routers** (paper by default; LIVE requires armed account) | M | 3.2, 2.6 | `desk/api/routers/{accounts,orders,positions}.py`, tests | (1) full manual cycle through the API on `FakeExchange` bars (place, modify, cancel, fill, close); (2) LIVE account order without arming is 403 with reason; (3) `/system/kill` cancels and halts (DB state asserted) |
| 3.6 | **Bots and alerts routers** | S | 3.2, 2.5, 2.7 | routers, tests | (1) create/start/stop/log cycle; (2) alert create/fire/log cycle with a scripted bar |
| 3.7 | **Backtests and replay routers** (replay session in-process: start, step n, manual order inside replay, state) | M | 3.2, 1.10 | routers, `desk/trading/replay.py`, tests | (1) a backtest job through the API ends DONE and the report endpoint returns the C-report schema; (2) replay step returns the next closed bar and applies A5 fills to replay orders |
| 3.8 | **Layouts and drawings routers** | S | 3.2 | routers, tests | (1) CRUD round-trips; (2) drawings are per layout and instrument; points carry `{time_ms, price}` |
| 3.9 | **Job runner and jobs router**: `desk/jobs/{model,runner,gpu_gate,__main__}.py` (subprocess launcher with the `longrun.launch` pattern and `longrun.pid_alive`, log files, cancel via taskkill/`terminate`, a `gpu` queue of size 1 gated by the desk's own gate: `nvidia-smi dmon -s um -c 10` run by `subprocess.run` and judged by `experiments.sweep.gpu_status_from_dmon`, never `nvidia_smi_gpu_check()`), kinds registry including `backtest` (1.10's runner as a subprocess) and `backfill` (2.2) | L | 0.6, 1.10, 2.2 | the files, `desk/api/routers/jobs.py`, `tests/desk/test_jobs.py` | (1) a `sleep` test kind runs, logs, finishes DONE; cancel ends CANCELLED and the pid is dead; (2) two GPU kinds queue (second waits); the gate test feeds a recorded busy `dmon` text while the test process has `CUDA_VISIBLE_DEVICES=-1` and asserts PENDING with reason (so the short-circuit cannot hide behind a mock); (3) the server restart re-attaches to a running job by pid; (4) the `backtest` kind produces 1.10's four files and the job row's `result.report_path` |
| 3.10 | **WebSocket hub**: `desk/api/ws.py` (topics C5; fan-out from the bus and the store watcher; forming-candle updates from the collector's heartbeat) | M | 3.1, 2.4 | the file, `tests/desk/test_ws.py` | (1) `websocket_connect` subscribe/receive for each topic with a scripted event; (2) unsubscribed topics are not delivered; (3) a slow client is dropped, not blocking others (queue bound test) |

### DESK-4 Front end (R1, R2 UI; approach B: NiceGUI panels, two JS elements)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 4.0 | **Chart spikes** (pre-registered, one day each, experimenter or escalated implementer): a throwaway page with (a) lightweight-charts 5.2.1 + the ported rectangle/trend-line/user-price-line primitives and (b) KLineCharts 10.1.0 built-ins, each loading 525,600 bars in 20k windows | M | 3.3 | `D:/nt/scratch` only; a report in `docs/desk/spikes/chart_spike.md` | (1) criteria fixed before the spike: a draggable order line whose drop price reaches Python, a trend line persisted and restored, a one-year scroll with no freeze over 200 ms per window; (2) the report states the winner and the measured numbers; (3) the default (lightweight-charts) stands unless KLineCharts wins all three |
| 4.1 | **App shell, conventions and the dock element**: `desk/ui/conventions.md`, `desk/ui/theme.py` (CSS variables from theme tokens plus candle up/down and order side tokens), `pages/desk.py` skeleton, `elements/dock.js` + `dock.py` (dockview-core 8.5.1 vendored from `https://unpkg.com/dockview-core@8.5.1/dist/dockview-core.min.js` and `dist/styles/dockview.css`, sha256 recorded and tested, `toJSON`/`fromJSON`), the `user`-fixture test skeleton (`async def test_...(user: User)`), `THIRD_PARTY_NOTICES.md` | L | 3.1 | the files, `tests/desk/test_ui_shell.py` | (1) `user` fixture opens `/` and sees the account-context banner and an empty dock; (2) a dock layout round-trips through `layouts`; (3) the test proves or disproves that `run_method` calls on a custom element are observable in the `user` simulation and records the answer in conventions.md (the panels' acceptance relies on the Python-side call log of 4.2, not on this) |
| 4.2 | **Chart element** (escalated implementer): `elements/lwc_chart.js` + `chart.py` (`ChartElement`: every outgoing call recorded as `(method, args)` in `self.calls` when `settings.testing` is true, with `last_call(method)`, so panel tests assert the call log without a browser): candles and volume, v5 panes, timeframe switch keeping the visible range, windowed history fetched from `/api/v1/bars` on scroll-left, forming candle `series.update`, last-price line, crosshair legend, UTC axis, markers API, `createPriceLine` API, click/crosshair events to Python; v5 cheat sheet in the header; vendored from `https://unpkg.com/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js` with LICENSE and NOTICE, sha256 recorded in `THIRD_PARTY_NOTICES.md` and checked by a test, `attributionLogo` on | L | 4.1, 3.3, 3.10, 4.0 | the files, `tests/desk/e2e/test_chart.py` | (1) Playwright: one year of 1-minute bars scrolls in windows with no single window blocking longer than 200 ms (measured); (2) the forming candle updates in place; (3) the element's documented props/methods/events all have a Playwright assertion; evidence = the CI run id of the `desk` job with the e2e step green and the uploaded trace and screenshot (a local Windows 10 run is additional, never the only evidence) |
| 4.3 | **Indicator panels** (NiceGUI): catalogue with search, parameter dialog generated from `PARAMS` with bounds and the server's refusal text, style tab from `DRAW`, overlay vs new pane via the element's pane API, remove and move, indicator set in the layout | M | 4.2, 3.4 | `desk/ui/panels/indicators.py`, tests | (1) `user` fixture: adding `my_rsi` leaves `(add_pane, {... levels (30, 70) ...})` in the ChartElement call log; (2) an out-of-bound value shows the server's message and sends nothing; (3) the indicator set persists in the layout |
| 4.4 | **Trading panels** (NiceGUI + chart lines): order ticket (market/limit/stop, qty or size fraction presets, TP/SL), orders/positions/fills/account tables, account-context banner, arming dialog (typed account id), kill-switch button; order and position lines through the element's price-line API with drag-to-modify confirm | L | 4.2, 3.5 | `desk/ui/panels/{ticket,orders,positions,account,arming}.py`, tests | (1) `user` fixture on paper: place, see the order row, modify, cancel, fill after a scripted bar, close the position; (2) kill switch empties open orders and halts bots; (3) LIVE requires the typed id and shows a red banner; (4) Playwright: dragging an order line lands the new price in sqlite (evidence = CI run id as in 4.2) |
| 4.5 | **Drawing tools** (escalated implementer): horizontal line, trend line, ray, rectangle, text as series primitives in `lwc_chart.js`; select, drag handles, delete, lock; persistence through `/layouts/{id}/drawings` with `{time_ms, price}` points | L | 4.2, 3.8 | `elements/lwc_chart.js`, `desk/ui/panels/drawings.py`, e2e tests | (1) each tool creates, moves and persists (Playwright); (2) drawings survive a timeframe switch; (3) Esc cancels, Delete removes; evidence = CI run id as in 4.2 |
| 4.6 | **Layouts, watchlist, symbol search** (NiceGUI) | M | 4.3 | panels, tests | (1) save/load/apply restores symbol, tf, dock layout, indicators, drawings; (2) the watchlist shows last prices pushed by the store watcher |
| 4.7 | **Bots, alerts and jobs panels** (NiceGUI) | M | 4.1, 3.6, 3.9 | panels, tests | (1) create/start/stop a bot, see its log and its trade markers on the chart; (2) an alert fires as `ui.notify` and a log row; (3) the job list shows a live log tail and cancels |
| 4.8 | **Backtest page**: form from the strategy's `params_schema()`, cost profile, range; run as a job; report with every C-report field; equity and drawdown panes; trades on the chart; trades table; baselines and n_eff band; the existing plotly figures through `ui.plotly` | L | 4.2, 3.7 | `desk/ui/pages/backtests.py`, tests | (1) `user` fixture runs `ma_cross` on the bundled data (job in-process for the test) and the report fields equal `report.json`; (2) selecting a trade leaves `(center_on, trade.ts)` in the ChartElement call log |
| 4.9 | **Replay controls** (start at bar or date, play/pause, step, speed, manual orders during replay with A5.6 fills, jump to live) | M | 4.4, 3.7 | panel, tests | (1) stepping advances one bar and the ChartElement call log shows the indicator recompute; (2) a replay order fills at the next stepped bar's open by A5.6 |
| 4.10 | **Lab pages** (NiceGUI): datasets, scenarios (spec, plan table, copy with edits), runs (list, progress, log tail, figures via `ui.plotly`), sweeps (estimate, confirm above one GPU-hour, launch), leaderboard (test columns labelled, never sortable), compare (stored verdicts), models (promote, activate), predictions overlay toggle, TensorBoard and optuna-dashboard links; plugin page (list, reload, new from template, test) | L | 4.1, 5.x, 3.4 | `desk/ui/pages/{lab,plugins}.py`, tests | (1) every page opens in the `user` fixture against fixture runs; (2) a figure renders (element present with the figure JSON); (3) the leaderboard table has no sortable test columns (test on the column definitions) |
| 4.11 | **Smoke suite and the look routine**: `tests/desk/e2e/` (5-10 Playwright tests: chart window and scroll, order-line drag, drawing persistence, dock round-trip, Lab progress), the "open it and look" checklist in `docs/desk/GUIDE.md`, the CI step enabled, screenshots as artifacts | M | 4.1-4.10 | tests, CI | (1) CI `desk` job green with the suite; (2) a screenshot per page uploaded; (3) the local run on Windows 10 is recorded as working or not (Playwright lists Windows 11+) |

### DESK-5 Lab backend (R6)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 5.1 | **Datasets**: `desk/lab/datasets.py` (list repo CSVs; fingerprints computed on demand per file through `dataset_fingerprint(Config(CSV_PATH=path))`, cached in `settings` by (path, size, mtime), the list returning the cached value or `fingerprint_pending`; export from the store; specs with `DatasetSpec` facts) | S | 0.7 | file, tests | (1) an export passes `Runner.from_spec(...).plan()` on a copied tiny scenario pointing at it (CPU, marker `slow` if needed); (2) the list endpoint never computes a fingerprint synchronously for a file over 50 MB |
| 5.2 | **Scenarios**: `desk/lab/scenarios.py` (discover, read, validate, plan table, copy-with-edits into `configs/scenarios/desk/`) | M | 5.1 | file, tests | (1) `discover_scenarios` parity; (2) an invalid edit returns the `ScenarioError` text, nothing written; (3) a copy is a valid spec with a new name |
| 5.3 | **Runs**: `desk/lab/runs.py` (index rows, scores, run detail from the light files, progress via `longrun`-style readers, servable run list) | M | 0.2 | file, tests on `tests/fixtures` run dirs | (1) detail of a fixture run equals its files; (2) progress states not started/running/stalled/done/failed reproduced with mtimes |
| 5.4 | **Figures and theme**: `desk/lab/figures.py` (`Visualizations.build(key, ...)` -> `pio.to_json(validate=False)` dict with `config.responsive`; inputs from `run_report.load_saved_blocks`/`scorer.load_block`; plotly.js served from `plotly.offline.get_plotlyjs()`), `/system/theme` CSS | M | 5.3 | file, router, tests | (1) each listed key returns JSON for the fixture run or a documented "needs artifacts/npz" 409; (2) the served plotly.js version string equals the python package's |
| 5.5 | **Leaderboard JSON**: `desk/lab/leaderboard.py` (serialiser for `LeaderboardRow` trees; `build_board` read-only path; ETag from `store_fingerprint`) | S | 5.3 | file, tests | (1) every `LeaderboardRow` field present; int-keyed `fold_values` become lists; (2) test columns flagged `rank_eligible=false` |
| 5.6 | **Model registry and predictions**: `desk/lab/models.py` (promote `runs/<id>/artifacts`, config facts, dev scores from the index, activate per (symbol, tf)); `desk/indicators/predictions.py` (overlay arrays from `Predictor.predict_frame` over the last n bars: per-horizon price path, lo90/hi90, P(up) with `direction_signal`); Predictor cache | L | 5.3, 0.7 | files, tests (a tiny bundle built once in a `slow` test with `tiny_config`) | (1) promote refuses a dir whose `artifacts/meta.json` lacks `format_version == 1`; (2) overlay arrays have one row per bar and `direction_signal` per horizon; (3) with `direction_signal == 'none'` the P(up) series is flagged and not drawn as a forecast (test on the payload) |
| 5.7 | **GPU and CLI job kinds**: `desk/jobs/kinds/{train,scenario,sweep,compare,screen,stability,indicator_report,export_dataset}.py` (`scenario`: `longrun.launch(spec, store)` verbatim with `longrun.progress`; `train` and the others: the launch pattern of A3 with `launch_env(root)`, the run dir from the last stdout line; progress for a bare run through `longrun.read_jsonl`, `tail`, `pid_alive` and the stall rule; the desk's GPU gate (3.9); budget announce for sweeps with the 1-GPU-hour confirm rule) | L | 3.9, 5.2 | files, tests with a fake `neural-trade` executable | (1) each kind builds the documented command line (snapshot test); (2) `train` returns the run dir from the last stdout line; (3) a sweep over 1 GPU-hour stays PENDING until confirmed; (4) GPU kinds refuse when the gate reads a recorded busy `dmon` text |
| 5.8 | **Indicator-development chain**: `docs/desk/GUIDE.md` section plus `desk/lab/indicator_dev.py` (learned periods table from `metrics.jsonl` and `period_init.json` vs textbook; lens inputs) and the `/indicators/learned` endpoint | M | 5.6, 7.3 | files, tests | (1) the lens output for a fixture run reproduces `discovered_table` numbers for ma/rsi/bb/macd; (2) a plugin family listed in `INDICATOR_FAMILIES` appears in the table |

### DESK-6 Live trading (guarded) and hardening

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 6.1 | **ccxt broker** (P1, `qa`): `desk/broker/ccxt_broker.py` (submit/modify/cancel with idempotent client ids, poll fills by trade id, `sync` reconciliation at start and periodically, sandbox first) | L | 2.1, 1.3 | file, tests on `FakeExchange`, `network` test on the sandbox | (1) scripted venue sequences (partial fills, late fill, reject) produce the A4 states; (2) restart reconciliation adopts open venue orders and positions without duplicates; (3) sandbox round trip (skipped without keys) |
| 6.2 | **Live arming and audit** (P1): `NT_DESK_LIVE=1` + arm request with typed id + session expiry; risk limits enforced in LIVE; every live command journalled with the token's session id; kill switch covers live orders | M | 6.1, 3.5 | files, tests | (1) without the env var the LIVE account cannot be armed (403); (2) an expired arming denies orders; (3) audit rows for every live command |
| 6.4 | **Chart performance at one year of 1-minute bars** (windowed loading, data conflation setting, server-side downsampling for the overview) | M | 4.2 | js, router | (1) the DESK-4.2 timing test passes with the long file; memory of the tab stays under 1 GB (measured in the E2E log); evidence = CI run id as in 4.2 |
| 6.5 | **Docs and inventory**: `docs/desk/GUIDE.md` (open it and look routine, pages, plugins, live safety), README pointer, RUNBOOK "Desk", `TESTING_DOCUMENTATION.md` regenerated; desk environment variables named in README or a guide are added to `_NOT_CONFIG_KEYS` in `tests/test_docs_guides.py` in the same item | S | all | docs | (1) `test_docs_guides` passes; (2) the inventory lists `tests/desk` |
| 6.6 | **Security pass** (lead + `qa`): token handling, secrets never logged, arming, CORS/origin check on the WebSocket, static path traversal | S | 6.2 | review | (1) a checklist in the item with evidence per line |

### DESK-7 Plugin SDK and built-in indicators (R5; can run in parallel with DESK-1 after DESK-0)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| 7.1 | **Indicator contract and registry**: `desk/indicators/api.py` (C1/C3: `IndicatorMeta`, `Figure`, `DrawSpec`, `ChartIndicators`, `register_indicator`, `check_indicator` with the golden-fixture convention; `ParamSpec` from `core/params.py`), `desk/plugins_api.py` (re-exports, `DeskParamSpec`), discovery through `load_plugins(d)` for `settings.plugin_dirs` (default `plugins_desk/`), the indicator template `plugins/templates/desk_indicator_template.py`, and the wiring of `StrategyContext.indicator` to `ChartIndicators.get(name).compute` (replacing 1.6's `IndicatorsUnavailable`) | M | 0.4, 1.6 | the files, `tests/desk/test_indicator_api.py` | (1) `check_indicator` catches a planted non-causal indicator, a wrong warm-up and an output key mismatch (three mutation tests); (2) the template is skipped by the loader and loads when copied to `plugins_desk/indicators/`; (3) duplicate id raises `DuplicateRegistrationError`; (4) `ctx.indicator` is cached per bar (call count test) |
| 7.2 | **Built-in indicators, batch 1** (trend and bands) as thin adapters over `ta` 0.11.0 where it has the indicator (sma, ema, wma, bollinger, keltner, donchian, ichimoku, psar) plus hand-written vwma, supertrend, heikin_ashi; one file per indicator with META/PARAMS/DRAW | L | 7.1 | `desk/indicators/builtin/*.py`, `tests/desk/test_builtin_indicators.py`, `tests/desk/fixtures/indicators/*.npz` | (1) every indicator passes `check_indicator`; (2) regression fixtures (values on `synthetic_bars`, committed by `scripts/desk/make_indicator_fixtures.py`) match within 1e-8; (3) each DRAW spec renders in the chart (DESK-4.3 smoke) |
| 7.3 | **Built-in indicators, batch 2** (momentum and volume): `rsi` equals `neural_trade.strategy.ta_rules.wilder_rsi` within 1e-12 (SMA-seeded Wilder; `ta`'s EWM-seeded RSI is registered as `rsi_ewm` if kept), macd, stochastic, willr, cci, adx, atr, obv, mfi, vwap (ta's volume-weighted plus a rolling variant), volume over `ta` where it matches; and `learned_bridge.py` (the lens: `compute_reference`/`indicator_lines` on one LOOKBACK window at textbook vs learned periods) | L | 7.1 | files, tests | (1) as 7.2; (2) the lens reproduces `discovered_indicators.indicator_lines` values for ma/rsi/bb/macd on a fixture window; (3) the documented difference "classic RSI (Wilder) vs learnable RSI (EWMA span)" is shown in the lens legend text; (4) the `rsi` equality test against `wilder_rsi` |
| 7.4 | **Strategy plugins**: `DeskStrategies` discovery from `plugins_desk/strategies/`, the strategy template `plugins/templates/desk_strategy_template.py` (subclasses `DeskStrategy` from 1.6), `check_strategy` (causality harness + params schema) and the plugin page's test action | M | 7.1, 1.11 | files, tests | (1) the template strategy loads, passes `check_strategy` and runs a backtest; (2) a planted look-ahead strategy fails `check_strategy` |
| 7.5 | **Plugin reload**: `desk/plugins_api.reload()` (remove every plugin-registered name from `ChartIndicators`/`DeskStrategies`, drop `neural_trade_plugins.*` modules from `sys.modules`, re-run `load_plugins`; errors reported per file) | S | 7.4 | file, tests | (1) editing a plugin file and reloading changes the computed output; (2) a broken file reports its error and the others load |

### Later (not required by R1-R6; filed as P3 items, not in the milestones)

| id | title | size | depends | files | acceptance |
|---|---|---|---|---|---|
| L.1 | **Websocket collector (v2)**: ccxt.pro `watch_ohlcv` with the closed-bar rule and REST reconciliation; falls back to polling | M | 2.3 | file, tests with a fake stream | (1) identical store contents from the stream and from polling on a recorded session |
| L.2 | **Streaming indicators for bots** (`desk-stream` extra): `talipp` adapters behind the same contract with `update(state, bar)`; batch-vs-stream equality test | M | 7.3, 2.5 | files, tests | (1) streaming output equals batch output on `synthetic_bars` for every adapted indicator; (2) a bot using a streaming indicator ticks under 5 ms per bar (measured) |
| L.3 | **Predictor subprocess** over a pipe (about 200 lines) if TensorFlow in the server process costs too much memory or latency | M | 5.6 | file, tests | (1) same outputs as the in-process path on a fixture bundle |

Order of execution (parallel lanes, each an implementer): lane A `0.2 -> 0.4 -> 0.5 -> 1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.6 -> 1.7 -> 1.8 -> 1.9 -> 1.10 -> 1.11 (-> 1.12 later)`; lane B `0.6 -> 0.7 -> 0.8 -> 1.5 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6 -> 2.7` (2.5 waits for lane A's 1.7); lane C `7.1 (after A's 1.6) -> 7.2 -> 7.3 -> 7.4 (after A's 1.11) -> 7.5`; then `3.1 -> 3.2 -> 3.9 -> 3.3..3.8, 3.10` and `5.1..5.8`; then DESK-4 (4.0 may start as soon as 3.3 exists); then DESK-6. Two implementers at once only on disjoint files (OPERATING_MODEL); 0.3 is the lead's and runs as soon as questions 0 and 4 are answered.

## Verification

**Per task:** the acceptance column, run by QA by risk (`qa` for P1 items: the fill rules, the parity test, risk guard, execution engine, ccxt broker, arming; lead-verified for mechanical P2 items) plus the standing checks: `tests/desk` with `-m "not e2e and not network" -n 8` (unit, `TestClient` and NiceGUI `user`-fixture tests), ruff, the no-TF-import test, the fast suite unchanged, CI `desk` and `unit` green; UI tasks cite `desk/ui/conventions.md` and the two JS elements carry Playwright evidence.

**Per milestone (the owner can open it):**

- DESK-0: `neural-trade desk init && neural-trade desk import-csv binance_btcusdt_1min_ccxt.csv` populates the stores; `pytest tests/desk` green; CI `desk` job exists.
- DESK-1: `neural-trade desk backtest --strategy ma_cross --symbol BTC/USDT --tf 1 --cost research_zero` writes a report whose fields are reproduced by `tests/desk/test_report.py`; `tests/desk/test_parity.py` proves the engine equals `strategy/backtest.py` on four strategies.
- DESK-2: with the collector running on the Binance spot testnet, a paper bot (`ma_cross`) trades on live bars and its orders, fills and positions are in `desk.sqlite`; `kill` halts it.
- DESK-3: `docs/desk/openapi.json` is current; every router has `TestClient` tests; the WebSocket test suite passes.
- DESK-4: the "open it and look" routine in `docs/desk/GUIDE.md`: chart of a year of 1-minute bars, add RSI and Bollinger, draw a trend line, save a dock layout, place a paper order from the chart, run a backtest and read its report, step a replay; every panel flow has a `user`-fixture test; the Playwright smoke suite is green in CI and screenshots are attached.
- DESK-5: from the Lab, launch `neural-trade train` on the bundled CSV as a job (GPU, after the owner lifts D-066's coding-only rule; CPU with `--set EPOCHS=1` until then), watch progress, open its figures, promote the bundle, see predictions on the chart and run a `model_signal` backtest with it.
- DESK-6: a sandbox live order placed from the UI is accepted by Binance testnet and reconciled after a server restart; the security checklist is complete.

**End-to-end (DESK-4.11):** the Playwright suite drives the real server on a temporary home with the bundled CSV imported; CI uploads traces and screenshots; locally the same suite is best-effort on Windows 10 (Playwright lists Windows 11+).

**Golden and speed guards:** `scripts/golden_run.py verify` against the current golden record (STATUS names it; `tests/fixtures/golden_nt124.npz` is the newest) after any task that touches `src/neural_trade/cli.py`, `strategy/`, `data/`, `tests/conftest.py`, `pyproject.toml` or `visualization/` (the desk only adds; a changed number fails the item); `sec_per_step` untouched because no desk code runs in training (checked by the layering test: nothing in `training/` imports `desk`, and desk plugins live outside `plugins/`).

## Risks and open questions

| risk | mitigation in the plan |
|---|---|
| ccxt 4.5.88 bumps certifi (2026.4.22 -> >= 2026.6.17), cryptography (47 -> 50.x), idna (3.13 -> 3.18) and urllib3 (2.6.3 -> 2.8.0) and adds winloop, coincurve, zlib-ng and aiohttp-fast-zlib (typing_extensions, attrs and cffi are already at its pins); nicegui adds `uvicorn[standard]`, python-socketio/engineio and about 20 transitive packages; none touch numpy, protobuf or TensorFlow | dry run first (0.3) with the full transitive list committed as `requirements-desk.txt`; the collector is a separate process by design (A3), so it can live in its own venv with only `ccxt` + `pandas` + stdlib sqlite if the pins fight |
| FastAPI/Starlette versions move past Python 3.10 | pin `starlette==1.7.0`, `fastapi==0.143.2`; the alternative stack (Tornado 6.5.10, already installed) keeps the same API surface |
| lightweight-charts v5 vs v4 confusion in generated code | the cheat sheet in the chart module header (DESK-4.2) lists the exact v5 calls; E2E catches regressions |
| an implementer "improves" the fill rules | A5 is the contract; the parity test (1.9) and `tests/desk/test_fills.py` pin them; changing a rule is an owner decision (default trading behaviour) |
| the Model Lab writes into the run store in a new way | every write goes through the existing CLI/engine (A8); the Lab's own files live in `var/desk/` and `configs/scenarios/desk/` |
| GPU contention with the owner's other project and the tactical session | `nvidia_smi_gpu_check` before every GPU job, a queue of one, and the owner's D-066 "coding only" rule: no GPU job until lifted |
| disk D: nearly full | `var/desk/` bounded: logs rotated at 50 MB, backtest journals compressed, the long file imported once into sqlite (about 300 MB) with the option to point `NT_DESK_HOME` elsewhere |
| the direction head has no skill (AUC about 0.5) | the desk shows the `direction_signal` state and the baselines; bots on model signals stay flat when the state is 'none' (D-066); the desk's value is manual control, data, backtests and the lab, not a signal |
| NiceGUI lock-in and API churn (3.x majors changed APIs; 3.18 raised anyio); implementers remember 1.x/2.x idioms | pinned 3.17.1; framework-specific code confined to `desk/ui/` (about 1k lines of panels, the app mount, two element wrappers); the conventions page with 3.17 snippets cited by every UI task; upgrades only as items gated by the `user` suite |
| TensorFlow on CPU inside the server (memory about 0.5-1 GB, slow first call, a code path could grab the GPU) | `CUDA_VISIBLE_DEVICES=-1` set in the entry point and asserted at startup; fallback item: a predictor subprocess over a pipe (about 200 lines) |
| Python 3.10 is end-of-life (2026-10-01) and the web stack is frozen at its last-3.10 versions, while the desk opens a port and holds exchange keys | bind 127.0.0.1 only; keys from environment variables or the OS keyring, never sqlite or YAML; trade-only API keys without withdrawal rights; no remote access; recorded in RUNBOOK; question 5 for the owner |
| the chart element is the largest hand-written piece (4-8k lines estimated) | the DESK-4.0 spike settles the library; the element's API is small and documented; escalated implementer with Playwright evidence; KLineCharts' built-in overlays are the fallback |
| total size: about 14-20k lines including tests (the refuter corrected every earlier estimate upward by 2-10x against comparable projects) | milestones ship runnable increments; the owner can stop after any of them |
| scope creep in the front end | the parity list (external reader, section 5) is the ladder; DESK-4 ships the eight most-used drawing tools and the TradingView report fields, nothing else until the owner asks |

**Questions for the owner (recorded in STATUS at step 0, with the recommendation first):**

0. **Stack: approach B (NiceGUI on FastAPI, two JS elements) or the earlier pick C (FastAPI + vanilla JS)?** Recommended: B. It is the maximum "stable frameworks" reachable without breaking the TF 2.10 boundary or the MIT licence, saves an estimated 5-8k hand-written front-end lines, and makes every panel testable without a browser; the cost is one more framework (pinned 3.17.1) confined to about 1k lines of panels and two element wrappers. The trading core, exchange, store, jobs, plugins and Lab logic are identical in both.
1. Amend VISION "Not in the MVP" for the desk track (a web application, exchange APIs) and confirm that the desk's live OHLCV feed is not a "new data source" in the D-050 sense. Recommended: yes, as D-067, the desk being a post-MVP track whose items the owner may interleave with the MVP.
2. Where the DESK milestones sit in ROADMAP "Order": after MVP-5 (recommended, so the MVP finishes first) or interleaved (DESK-0/1/7 are CPU-only and can run while the MVP waits for GPU time under D-066).
3. The desk's default cost profile for paper and live accounts: Binance spot taker 10 bps per side (recommended, explicit) while the research yardstick stays at 0 (D-044).
4. The environment change: approve one pinned install into `nt` after the dry run (nicegui 3.17.1 and its stack, ccxt 4.5.88 with its exact pins, ta 0.11.0, playwright 1.64.0 + pytest-playwright 0.10.0; optional talipp, duckdb), including a possible numpy move from 1.23.0 to 1.23.5 (CI's value) verified by the golden record, and the collector in its own venv if ccxt's pins fight (recommended: yes to all).
5. Security posture: accept a localhost-only desk holding exchange keys on the frozen Python-3.10 web stack with the mitigations in the risk table (recommended), or require a separate-service desk on a supported interpreter talking to a thin `nt` prediction service over HTTP (more machinery, deferred).
6. Chart spikes: fund the two pre-registered one-day spikes (DESK-4.0) or take lightweight-charts by default (recommended: fund them; the chart element is the largest custom piece).
7. Drawing-tool scope for the first version: horizontal and vertical line, trend line, ray, rectangle, text (recommended); Fibonacci, measure and position tools later as items. Bar replay in, websocket feed later (recommended).

## Step 0 for the lead (after the owner approves this plan)

1. Record D-067 and the four questions; add `docs/desk/{PLAN,CONTRACTS,API}.md` from this file; file the items (NT-215 onward) with milestone tags; add the ROADMAP track.
2. Run the environment dry run (0.3) in a scratch venv and install the approved pins in `nt`; write the RUNBOOK "Desk" section.
3. Start lanes A, B and C (DESK-0.2 first, then 0.4/0.6/7.1 in parallel on disjoint files).
4. The first owner-visible increment is DESK-0 + DESK-1 (a backtest report from the CLI) and the first chart (DESK-3.3 + 4.2) right after.
