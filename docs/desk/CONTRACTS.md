# Trading desk: contracts

_Extracted from [PLAN.md](PLAN.md); an implementer writes these signatures, fields and rules verbatim. If a task needs something this file does not define, the task stops and reports (OPERATING_MODEL)._

### A4. Domain model (dataclasses in `desk/core/model.py`; the field lists are the contract)

- `Instrument(id='BINANCE:BTCUSDT', symbol='BTC/USDT', base, quote, price_increment, size_increment, min_qty, min_notional, max_qty, is_spot=True)`; fees live in `CostProfile`, not on the instrument.
- `Bar(instrument_id, tf_minutes, ts_open_ms, open, high, low, close, volume, is_closed)`; a strategy sees only closed bars; the forming candle is display-only. `ts_open_ms` is the Binance open time (UTC ms); the bar closes at `ts_open_ms + tf*60_000`.
- `Order(client_id, instrument_id, account_id, strategy_id|None, bot_id|None, side BUY|SELL, type MARKET|LIMIT|STOP_MARKET|STOP_LIMIT, qty|None, size_frac|None, price|None, trigger_price|None, tif GTC|DAY|IOC|FOK, reduce_only, bracket: {take_profit, stop_loss, tp_is_offset: bool, max_hold: int|None}|None, status, filled_qty, avg_fill_price, reason, venue_id|None, ts_created, ts_updated, tags)`. Exactly one of `qty` and `size_frac` is set: `size_frac` is a fraction of realised cash equity resolved by the broker at the fill open (the oracle's `size_frac`, A5.1); with `tp_is_offset` the bracket levels are quote-currency offsets from the fill bar's open mid, resolved at the fill (the oracle's `tp_is_offset`, which keys both tp and sl). **Status edges (the complete list; any other transition raises `IllegalTransition`):** INITIALIZED->DENIED, INITIALIZED->SUBMITTED, SUBMITTED->REJECTED, SUBMITTED->ACCEPTED, ACCEPTED->TRIGGERED, ACCEPTED->PARTIALLY_FILLED, ACCEPTED->FILLED, ACCEPTED->CANCELED, ACCEPTED->EXPIRED, TRIGGERED->PARTIALLY_FILLED, TRIGGERED->FILLED, TRIGGERED->CANCELED, TRIGGERED->EXPIRED, PARTIALLY_FILLED->PARTIALLY_FILLED, PARTIALLY_FILLED->FILLED, PARTIALLY_FILLED->CANCELED, PARTIALLY_FILLED->EXPIRED. `is_closed = status in {FILLED, CANCELED, EXPIRED, REJECTED, DENIED}`. One exception edge `late_fill`: a Fill reported for a CANCELED order (the venue filled before the cancel landed) is applied to the Portfolio and moves the order CANCELED->PARTIALLY_FILLED or FILLED with journal kind `late_fill`; a Fill for a FILLED order with a known `trade_id` is ignored, with an unknown one raises `OverfillError`. Transitions only through `ExecutionEngine.apply(event)`.
- `Fill(trade_id, client_id, instrument_id, side, qty, price, mid, fee, fee_currency, liquidity MAKER|TAKER, ts)`; `price` is the slipped fill price, `mid` the reference price before spread and slippage (equal to `price` for venue fills), so the report computes gross P&L on mids and costs = qty x |price - mid| + fee, as the oracle does; partial fills are several Fill events.
- `Position(position_id='{instrument}-{strategy}', instrument_id, account_id, strategy_id, side LONG|SHORT|FLAT, qty, avg_price, realised_pnl, fees, ts_opened, ts_closed|None)` (NETTING; Binance spot cannot short: `RiskGuard` denies SELL beyond the held base balance on spot accounts).
- `AccountState(account_id, kind BACKTEST|PAPER|LIVE, quote_currency, cash, equity, locked, realised_pnl, unrealised_pnl, ts)`.
- Events (all with `ts_event`, `ts_init`; `ts_init` is `clock.now_ms()` of the engine's Clock, never `time.time()`, so backtests are deterministic): `BarClosed`, `BarForming`, `OrderEvent(kind, order)`, `FillEvent`, `PositionEvent`, `AccountEvent`, `BotEvent(status|log)`, `AlertFired`, `JobEvent`, `TimeEvent`. Commands: `SubmitOrder`, `ModifyOrder`, `CancelOrder`, `CancelAll`, `ClosePosition`, `Halt`.

### A5. Fill rules (written before coding; the parity test pins them)

1. A strategy decides on the closed bar `t` (after `BarClosed`). A MARKET order created then fills at bar `t+1`'s open. **Sizing at the fill:** a `size_frac` order's quantity is `clip(size_frac, 0, 1) x realised equity (net of costs paid; `Portfolio.realised_equity()` = cash + sum of signed_qty x avg_price over all open positions, which equals the oracle's equity_cash when the account is flat; after every exit filled at the same open; lead's decision 2026-10-16 from QA of NT-223: cash alone over-sizes when another position is open and sizes a margin short at 1.5x leverage) / open`; a `qty` order fills its quantity. **Offset brackets** (`tp_is_offset`) resolve to `open + offset` at the fill (both tp and sl). RiskGuard checks a `size_frac` order twice: at submit on the last close (an estimate) and at the fill on the resolved quantity (the binding check). **The binding check at the fill (lead 2026-10-16, NT-295):** the broker calls `ExecutionEngine.precheck_fill(client_id, mid, qty=None)` before it reports a fill; `mid` is the fill's reference price before spread and slippage (RiskGuard adds the slip). Only when `mid` is above the price the order was valued at when it was checked (stop gaps above the trigger, a market order opening above the last close; tolerance 1e-6 relative) the engine re-runs RiskGuard on the fill as a MARKET order at `mid` (`record=False`, the order's own commitments excluded). Allowed: the fill goes ahead, a `size_frac` order at the returned (possibly clipped) quantity. Denied: the fill is REFUSED, the engine journals `fill_refused` and the order ends REJECTED with reason `fill_price_check_failed` (CANCELED with that reason if part of it was already filled; this REJECTED edge from ACCEPTED or TRIGGERED is an exception edge next to `late_fill`), and the broker must not fill it. A fill that reaches `apply` and fails the same check is refused the same way on BACKTEST and PAPER accounts; on a LIVE account the venue fact stands: it is applied, journalled `fill_refused` (`applied: true`) and RiskGuard is halted. **Cost arithmetic (oracle, bit for bit):** per side, slip cost = `qty x mid x (half_spread_bps + slippage_bps)/1e4` and fee = `fee_bps/1e4 x qty x mid`, where `mid` is the bar OPEN for a market, TIME or strategy-exit fill and the bracket or limit level for an intrabar fill; `Fill.price = mid x (1 + side x slip)`, `Fill.mid = mid`, `Fill.fee` is the fee on the mid notional (never on the slipped price); cash accounting `cash -= side x qty x Fill.price + Fill.fee`. This is exactly `strategy/backtest.py` (`fill='next_open'`), so the wrapped model-signal strategies reproduce the existing trades bit for bit at the same cost profile.
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


### A7. Plugin SDK (R5)

- Two new strict registries, defined in `desk/indicators/api.py` and `desk/core/strategy.py`: `ChartIndicators` and `DeskStrategies` (subclasses of `core/registry.BaseRegistry`, so `list()`, metadata, duplicates and `remove` behave like the ten). Discovery: `load_plugins(d)` for each `settings.plugin_dirs`, default **`plugins_desk/` at the repo root, a sibling of `plugins/` and never inside it**: `load_plugins` rglobs a whole tree and the ten registries load `Config.PLUGINS_DIR` (default None) inside the training process, so desk plugin files under `plugins/` would be executed by a training run that sets `PLUGINS_DIR: plugins`. Desk plugins live in `plugins_desk/indicators/*.py` and `plugins_desk/strategies/*.py`; templates stay in `plugins/templates/desk_*_template.py` (skipped by the loader) and are copied into `plugins_desk/` by the UI. Reload (`desk/plugins_api.reload()`): remove every plugin-registered name from the two registries, drop the `neural_trade_plugins.*` modules from `sys.modules` (the loader treats a module in `sys.modules` as loaded), re-run `load_plugins`; errors reported per file; a restart is the fallback.
- Indicator contract (C3): `META`, `PARAMS` (schema with bounds -> the dialog and sweep spaces), `compute(df, **params) -> DataFrame` (causal, NaN warm-up, aligned to `df.index`), `DRAW` (pane overlay or new; figures line/band/histogram/marker/hline). Acceptance tests every plugin must pass (provided as a pytest helper): causality (compute on a prefix equals the prefix of compute on the full frame), warm-up length, bounds rejection, golden fixture when one is shipped.
- Strategy contract (C2): a `DeskStrategy` subclass with `PARAMS`, `on_bar(ctx)` and optional `on_order_event`, `on_start`, `on_stop`; `ctx` offers `bars(n)`, `indicator(name, **params)` (cached per bar), `predictions()` (when a model is attached), `position`, `account`, `submit_market/limit/stop`, `cancel`, `close`, `log`, `clock.now`. The adapter `ModelSignalStrategy(name, params)` wraps any registered neural_trade `Strategies` entry.
- Templates in `plugins/templates/desk_indicator_template.py` and `desk_strategy_template.py` (skipped by the loader, copied by the UI's "new plugin from template" action into `plugins_desk/indicators/` or `plugins_desk/strategies/`).
- Learnable families are not duplicated as classic plugins; the lens (A2 `learned_bridge.py`) shows them on the chart; a classic twin (for example `rsi_wilder` next to the EWMA-form `rsi` family) is a documented difference, not a bug.


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
    def fill_entries(self, bar: Bar, cash: float) -> list[Fill | OrderEvent]: ...  # A5.9 step 2 (size_frac resolved on the post-exit realised equity (`Portfolio.realised_equity()`, not raw `cash`))
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

