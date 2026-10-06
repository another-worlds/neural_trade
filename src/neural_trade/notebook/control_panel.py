"""The control panel (notebook 06, NT-034, D-023): choose a scenario, a search space and a mode, launch or
resume a sweep, watch its leaderboard, compare rows.

    panel = ControlPanel(store="../runs", specs_dir="../configs/scenarios")
    display(panel.launcher())      # widgets only: executing a cell never starts a sweep
    display(panel.board())         # the newest sweep's leaderboard now; start_polling() keeps it fresh
    display(panel.comparison())    # two or more rows: every metric per horizon, spread over folds and seeds

**One engine path.** A launch builds :class:`~neural_trade.experiments.sweep.Sweep` from the scenario, the
store and :class:`~neural_trade.experiments.sweep.SweepOptions` exactly as ``neural-trade sweep`` does
(``cli.cmd_sweep``) and calls ``Sweep.run()``: the same refusals (a budget above ``max_hours``, a bar size
other than 1 minute, ``parallel`` above the GPU record's allowed N, an existing sweep without ``resume``)
come from the sweep itself and are shown as refusals, nothing started. ``sweep_factory`` and
``sweep_kwargs`` (a trainer, a GPU check) are the injection points the headless tests drive.

**Nothing launches by itself.** Constructing the panel and displaying its widgets only read the run store.
A sweep starts only from the *Launch* button's callback, after the printed estimate / GPU budget of a dry run
(``SweepOptions.dry_run``) was shown; a budget above ``confirm_gpu_hours`` also needs the *Confirm budget*
button. The sweep runs in a background thread (as :class:`TrainingSession` trains), so the kernel stays free
and the board keeps refreshing (:meth:`ControlPanel.start_polling`).

**The board** is NT-031's: ranked on the dev folds' net Sharpe after costs, guard-rails beside it, the test
columns shown and labelled "not used for ranking" (D-020). Several sweeps (the yardstick's learned / frozen
twin / TA rules arms) go on one board through ``build_leaderboard``'s multi-scenario input.

**A failure is visible.** A sweep that raises (not a refusal) writes an ``error`` output into the panel's log
widget, which ``scripts/notebooks/check.py`` reads from the saved widget state; a thread's WARNING records go
into the same log.
"""
from __future__ import annotations

import dataclasses
import html
import json
import logging
import threading
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.store import RunStore
from neural_trade.experiments.sweep import DEFAULT_PARALLEL_RECORD, MODES, QUICK, Sweep, SweepOptions
from neural_trade.notebook import panel_compare as PC
from neural_trade.notebook import panel_data as PD
from neural_trade.notebook._display import show
from neural_trade.notebook.session import ThreadWarnings

logger = logging.getLogger(__name__)

NEW_SWEEP = "(new sweep)"
DEFAULT_METRICS = ("backtest/sharpe_net", "backtest/total_return", "backtest/max_drawdown", "h0/variance/crpss",
                   "h1/variance/crpss", "h2/variance/crpss", "h0/direction/auc", "h1/direction/auc", "h2/direction/auc")


def build_sweep(scenario: Scenario, store: RunStore, options: SweepOptions, announce: Callable[[str], None], **kwargs) -> Sweep:
    """The sweep ``neural-trade sweep`` would build (cli.cmd_sweep): the panel's one way to make one."""
    return Sweep(scenario, store, options, announce=announce, **kwargs)


@dataclasses.dataclass
class Estimate:
    key: str
    hours: float                  # the GPU budget's upper bound (optuna) or the quick estimate, in hours
    text: str                     # what the sweep printed before it would start
    budget: Dict[str, Any]
    state: str


def _stream(text: str, name: str = "stdout") -> Dict[str, Any]:
    return {"output_type": "stream", "name": name, "text": text}


def _error_output(exc: BaseException) -> Dict[str, Any]:
    return {"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc)[:500],
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:]}


class ControlPanel:
    """See the module docstring. The widgets are thin: every callback calls a method of this class, which the
    headless tests call too."""

    def __init__(self, store="runs", specs_dir="configs/scenarios", *, compares_dir="configs/compares",
                 index_path=None, scenario: Optional[str] = None, mode: str = QUICK, poll_seconds: float = 10.0,
                 confirm_gpu_hours: float = 1.0, sweep_factory: Optional[Callable[..., Any]] = None,
                 sweep_kwargs: Optional[Dict[str, Any]] = None, parallel_record: Optional[str] = DEFAULT_PARALLEL_RECORD,
                 root=None):
        self.store = store if isinstance(store, RunStore) else RunStore(store, index_path)
        self.specs_dir, self.compares_dir = specs_dir, compares_dir
        # the repository root: the notebook runs with notebooks/ as its working directory, while the GPU record's default
        # path is relative to the root, where the CLI is run from (a scenario's relative CSV_PATH: see build_scenario)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self._lock = threading.RLock()
        self.poll_seconds = float(poll_seconds)
        self.confirm_gpu_hours = float(confirm_gpu_hours)
        self._factory = sweep_factory
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.choices = PD.discover_scenarios(specs_dir)
        self.options_by_name = {o.name: o for o in PD.field_options()}
        # ---- the model the widgets edit
        self.scenario_name: Optional[str] = None
        self.base: Optional[Scenario] = None
        self.mode = mode if mode in MODES else QUICK
        self.name = ""
        self.folds: List[int] = []
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.resume = False
        self.opts: Dict[str, Any] = {"n_trials": 30, "max_hours": 12.0, "parallel": 1, "parallel_record": parallel_record,
                                     "sec_per_step": None, "quick_minutes": 5.0, "overhead_s": 30.0, "top_k": 5,
                                     "rerun_seeds": 3, "when_busy": "stop", "stop_after": None, "sampler_seed": 0}
        # ---- the run
        self.estimate: Optional[Estimate] = None
        self._confirmed: Optional[str] = None
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.refusal: Optional[str] = None
        self.status = "idle: nothing started"
        self.announced: List[str] = []
        self.launches: List[Dict[str, Any]] = []     # every real (non dry-run) sweep this panel started
        self._thread: Optional[threading.Thread] = None
        self._warnings = ThreadWarnings(lambda: self._thread.ident if self._thread else None)
        self._log_outputs: List[Dict[str, Any]] = []
        # ---- the board and the comparison
        self.board_ids: Optional[List[str]] = None     # None: follow the default (newest sweep / yardstick arms)
        self.board_data: Optional[PD.BoardData] = None
        self.sweeps: List[PD.SweepInfo] = []
        self.role = "dev"
        self.metric = "backtest/sharpe_net"
        self.selection: List[Tuple[str, str]] = []
        self._fingerprint: Optional[tuple] = None
        self._poller: Optional[threading.Thread] = None
        self._stop_poll = threading.Event()
        self._w: Dict[str, Any] = {}
        self._tl = threading.local()
        self.select_scenario(scenario or self._default_scenario())

    @property
    def _syncing(self) -> bool:
        """True while THIS thread sets widget values from the model: a widget observer runs in the thread that set the
        value, so the flag is per thread (a refresh in the poller must not mute a click handled in the UI thread)."""
        return bool(getattr(self._tl, "syncing", False))

    @_syncing.setter
    def _syncing(self, value: bool) -> None:
        self._tl.syncing = bool(value)

    # ------------------------------------------------------------------ the model (headless API)
    def _default_scenario(self) -> Optional[str]:
        usable = [c.name for c in self.choices if c.scenario is not None]
        for pref in ("reference_default", "nt033_learned"):
            if pref in usable:
                return pref
        return usable[0] if usable else None

    def _choice(self, name: Optional[str]) -> Optional[PD.ScenarioChoice]:
        return next((c for c in self.choices if c.name == name and c.scenario is not None), None)

    def select_scenario(self, name: Optional[str]) -> None:
        """Pick a scenario spec: its folds, name and ``search:`` block (or the sweep's default space) become the form."""
        choice = self._choice(name)
        self.scenario_name = name if choice else None
        self.base = choice.scenario if choice else None
        if self.base is not None:
            self.name = self.base.name
            self.folds = list(self.base.folds)
            self.rules = PD.rules_from_scenario(self.base)
        else:
            self.name, self.folds, self.rules = "", [], {}
        self.explicit = bool(self.base is not None and self.base.search)
        self.resume = False
        self._invalidate("scenario", rebuild=True)

    def select_sweep(self, sweep_id: Optional[str]) -> None:
        """Resume an existing sweep of the store: its scenario, mode and space; ``resume`` is ticked."""
        info = next((s for s in PD.list_sweeps(self.store.root) if s.sweep_id == sweep_id), None)
        if info is None:
            self.resume = False
            self._invalidate("sweep", rebuild=True)
            return
        self.select_scenario(info.scenario)
        self.mode = info.mode if info.mode in MODES else self.mode
        if info.space:
            self.rules = {p["name"]: ({"choices": p["choices"]} if p["kind"] == "cat" else
                                      {k: p[k] for k in ("low", "high", "log", "step") if p.get(k) is not None})
                          for p in info.space}
        self.resume = True
        self.explicit = True
        self._invalidate("sweep", rebuild=True)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        self.mode = mode
        self._invalidate("mode")

    def set_option(self, key: str, value: Any) -> None:
        if key not in self.opts and key not in ("resume", "name"):
            raise KeyError(key)
        if key == "resume":
            self.resume = bool(value)
        elif key == "name":
            self.name = str(value)
        else:
            self.opts[key] = value
        self._invalidate(key)

    def set_folds(self, folds: Sequence[int]) -> None:
        self.folds = sorted(int(f) for f in folds)
        self._invalidate("folds")

    def add_field(self, field_name: str) -> None:
        """Add a field to the space with the range the Config metadata (or the strategy) gives it."""
        opt = self.options_by_name.get(field_name)
        if opt is None and self.base is not None and field_name.startswith("strategy."):
            opt = next((o for o in PD.strategy_options(self.base.strategy.name) if o.name == field_name), None)
        if opt is None:
            raise KeyError(f"{field_name} is not a field a sweep may search")
        self.rules[field_name] = opt.default_rule()
        self.explicit = True
        self._invalidate("space", rebuild=True)

    def remove_field(self, field_name: str) -> None:
        self.rules.pop(field_name, None)
        self.explicit = True
        self._invalidate("space", rebuild=True)

    def set_rule(self, field_name: str, **rule: Any) -> None:
        cur = dict(self.rules.get(field_name) or {})
        for k, v in rule.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        self.rules[field_name] = cur
        self.explicit = True
        self._invalidate("space")

    def search(self) -> Dict[str, Any]:
        """The ``search:`` block of the scenario a launch builds: empty while the scenario has none and the form was not
        edited (the sweep then applies its own default space, so the scenario keeps the spec file's hash)."""
        return PD.search_block(self.rules) if self.explicit else {}

    def search_text(self) -> str:
        """The ``search:`` block the form builds, as YAML."""
        return PD.search_yaml(self.search())

    def build_scenario(self) -> Scenario:
        if self.base is None:
            raise InvalidConfigurationError("no scenario is selected")
        # the overrides stay the spec's: a relative CSV_PATH is part of every cell's identity, and the data layer opens it
        # from the project root when the working directory (notebooks/) lacks it (data.loaders.resolve_data_path), so
        # a sweep the CLI started resumes here, and back, without retraining a finished cell
        return dataclasses.replace(self.base, name=self.name or self.base.name, folds=list(self.folds or self.base.folds),
                                   search=self.search())

    def resolve_path(self, path: Optional[str]) -> Optional[str]:
        """``path`` as the CLI would find it from the repo root: unchanged when absolute or when it exists from the
        working directory, else against the root when it exists there (else unchanged, and the sweep says it is missing)."""
        if not path:
            return path
        p = Path(path)
        if p.is_absolute() or p.exists() or not (self.root / p).exists():
            return path
        return str(self.root / p)

    def build_options(self, *, dry_run: bool = False) -> SweepOptions:
        o = self.opts
        sps = o["sec_per_step"]
        return SweepOptions(mode=self.mode, n_trials=int(o["n_trials"]), stop_after=o["stop_after"] or None,
                            max_hours=float(o["max_hours"]), parallel=int(o["parallel"]),
                            parallel_record=self.resolve_path(o["parallel_record"]) or None, sec_per_step=float(sps) if sps else None,
                            quick_minutes=float(o["quick_minutes"]), overhead_s=float(o["overhead_s"]),
                            top_k=int(o["top_k"]), rerun_seeds=int(o["rerun_seeds"]), sampler_seed=int(o["sampler_seed"]),
                            resume=bool(self.resume), when_busy=o["when_busy"], dry_run=dry_run)

    def key(self) -> str:
        """Identity of what a launch would start: the scenario, the options and the resume flag."""
        try:
            sc = self.build_scenario()
            body = {"scenario": sc.spec_hash, "name": sc.name, "folds": sc.folds,
                    "options": dataclasses.asdict(self.build_options())}
        except InvalidConfigurationError as exc:
            body = {"invalid": str(exc)}
        return json.dumps(body, sort_keys=True, default=str)

    def _invalidate(self, why: str, *, rebuild: bool = False) -> None:
        """A change of the form: an earlier estimate and confirmation no longer describe what a launch would start.
        ``rebuild`` redraws the search-space rows (a field added or removed, another scenario), not on a mere edit."""
        self._confirmed = None
        if self.estimate is not None and self.estimate.key != self.key():
            self.estimate = None
        self._sync_widgets(rebuild_rules=rebuild)

    # ------------------------------------------------------------------ estimate, confirm, launch
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def needs_confirmation(self) -> bool:
        """True when the estimate shown for the current form is above ``confirm_gpu_hours`` and not yet confirmed."""
        e = self.estimate
        return e is not None and e.key == self.key() and e.hours > self.confirm_gpu_hours and self._confirmed != e.key

    def estimate_now(self) -> bool:
        """The *Estimate* button: a dry run in the background thread; returns False when refused to start."""
        return self._start("estimate")

    def confirm(self) -> bool:
        """The *Confirm budget* button: accept the shown estimate for this exact form (a change of the form revokes it)."""
        if self.estimate is None or self.estimate.key != self.key():
            self._set_status("nothing to confirm: estimate the current form first")
            return False
        self._confirmed = self.estimate.key
        self._set_status(f"budget confirmed ({self.estimate.hours:.2f} h): Launch now starts the sweep")
        self._sync_widgets()
        return True

    def launch_now(self) -> bool:
        """The *Launch* / *Resume* button: estimate (printed first), then start the sweep unless the estimate is
        above ``confirm_gpu_hours`` and not confirmed. Runs in a background thread."""
        return self._start("launch")

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the background job ended (tests, scripts); its WARNING+ records are re-emitted here."""
        if self._thread is not None:
            self._thread.join(timeout)
            if not self._thread.is_alive():
                self._warnings.replay()

    def _start(self, kind: str) -> bool:
        if self.running:
            self._set_status("a sweep job is already running in this panel")
            return False
        if self.base is None:
            self._set_status("refused, nothing started: no scenario is selected")
            return False
        key = self.key()
        self.error, self.refusal = None, None
        self._warnings.records.clear()
        self._thread = threading.Thread(target=self._job, args=(kind, key), name="neural_trade-sweep", daemon=True)
        self._thread.start()
        return True

    def _announce(self, text: str) -> None:
        self.announced.append(text)

    def _make(self, scenario: Scenario, dry_run: bool):
        factory = self._factory or build_sweep
        return factory(scenario, self.store, self.build_options(dry_run=dry_run), self._announce, **self.sweep_kwargs)

    def _job(self, kind: str, key: str) -> None:
        pkg = logging.getLogger("neural_trade")
        pkg.addHandler(self._warnings)
        try:
            scenario = self.build_scenario()
            self._set_status("estimating (a dry run: nothing is trained)")
            self.announced.clear()
            res = self._make(scenario, True).run()
            budget = dict(getattr(res, "budget", {}) or {})
            hours = (float(budget["gpu_hours"]) if budget.get("gpu_hours") is not None
                     else float(budget.get("estimated_minutes", 0.0)) / 60.0)
            self.estimate = Estimate(key, hours, "\n".join(self.announced), budget, str(getattr(res, "state", "")))
            self._show_estimate()
            if kind == "estimate":
                self._set_status(f"estimate shown: {hours:.2f} h" + (" (above the confirmation level: Confirm budget, then Launch)"
                                                                      if hours > self.confirm_gpu_hours else ""))
                return
            if hours > self.confirm_gpu_hours and self._confirmed != key:
                self._set_status(f"not launched: the budget {hours:.2f} h is above {self.confirm_gpu_hours:g} h: "
                                 "press Confirm budget, then Launch")
                self._sync_widgets()
                return
            self._set_status("running" + (" (resuming)" if self.resume else ""))
            self.launches.append({"key": key, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "scenario": scenario.name, "mode": self.mode, "resume": bool(self.resume)})
            self.result = self._make(scenario, False).run()
            state = getattr(self.result, "state", "?")
            reason = getattr(self.result, "stop_reason", None)
            self._set_status(f"sweep {getattr(self.result, 'sweep_id', scenario.name)}: {state}" + (f" ({reason})" if reason else ""))
            self.refresh_board(force=True)
        except InvalidConfigurationError as exc:   # what the CLI refuses: shown, nothing started, not a failure
            self.refusal = str(exc)
            self._set_status(f"refused, nothing was started: {exc}")
        except BaseException as exc:   # surfaced in the log widget as an error output, which check.py reads
            self.error = exc
            self._set_status(f"failed: {exc!r}"[:300])
            logger.exception("sweep job failed")
            self._log_outputs.append(_error_output(exc))
            self._push_log()
        finally:
            pkg.removeHandler(self._warnings)
            for rec in list(self._warnings.records):
                self._log_outputs.append(_stream(f"{rec.levelname}: {rec.getMessage()}\n"))
            self._push_log()
            self._sync_widgets()

    # ------------------------------------------------------------------ status, log, estimate views
    def _set_status(self, text: str) -> None:
        self.status = text
        w = self._w.get("status")
        if w is not None:
            color = "#b91c1c" if text.startswith(("failed", "refused")) else "#15803d" if text.startswith("running") else "#6b7280"
            w.value = f"<b style='color:{color}'>{html.escape(text)}</b>"

    def _push_log(self) -> None:
        out = self._w.get("log")
        if out is not None:
            out.outputs = tuple({**o, "metadata": {}} if o["output_type"] != "stream" else dict(o) for o in self._log_outputs)

    def _show_estimate(self) -> None:
        w = self._w.get("estimate")
        if w is not None:
            e = self.estimate
            w.value = ("<b>Estimate (a dry run; this is what the CLI prints before a sweep starts)</b>"
                       f"<pre style='white-space:pre-wrap'>{html.escape(e.text)}</pre>" if e else "")

    def summary_text(self) -> str:
        sweeps = PD.list_sweeps(self.store.root)
        lines = [f"run store: {self.store.root} (index {self.store.index_path})",
                 f"{len(self.choices)} scenario spec(s) in {self.specs_dir}; {len(sweeps)} sweep(s) in the store"]
        lines += [f"  {s.text}" for s in sweeps[:8]]
        lines.append("executing this notebook starts nothing: only the Launch button does")
        return "\n".join(lines)

    # ------------------------------------------------------------------ the board
    def refresh_board(self, *, force: bool = False) -> bool:
        """Re-read the store and redraw the board when it changed (or ``force``); returns True when it redrew.
        Read-only: no index sync, no write; safe to call from the polling thread while a sweep writes."""
        with self._lock:
            return self._refresh_board(force)

    def _refresh_board(self, force: bool) -> bool:
        self.sweeps = PD.list_sweeps(self.store.root)
        ids = self.board_ids if self.board_ids is not None else PD.default_board_sweeps(self.sweeps)
        known = {s.sweep_id for s in self.sweeps}
        ids = [i for i in ids if i in known] if self.board_ids is not None else ids
        fp = PD.store_fingerprint(self.store, ids)
        if not force and fp == self._fingerprint:
            return False
        self._fingerprint = fp
        self.board_data = PD.build_board(self.store, self.sweeps, ids, specs_dir=self.specs_dir) if ids else None
        self._draw_board(ids)
        return True

    def board_rows(self) -> List[Any]:
        return list(self.board_data.rows) if self.board_data else []

    def _draw_board(self, ids: Sequence[str]) -> None:
        w = self._w
        if "board_box" not in w:
            return
        info = [s for s in self.sweeps if s.sweep_id in set(ids)]
        sel = w["sweeps"]
        self._syncing = True
        try:
            sel.options = [s.sweep_id for s in self.sweeps]
            sel.value = tuple(i for i in ids if i in sel.options)
        finally:
            self._syncing = False
        if self.board_data is None or not self.board_data.rows:
            w["board_status"].value = self._no_sweep_html(info)
            w["board_fig"].outputs = ()
            w["board_fig"].layout.display = "none"
            return
        from neural_trade.visualization.leaderboard_fig import leaderboard_figure

        bd = self.board_data
        lines = [f"<b>{html.escape(s.text)}</b>" + (f" &mdash; {html.escape(s.stop_reason)}" if s.stop_reason else "")
                 + (" (quick: a leader, not a winner)" if s.label == QUICK else "") for s in info]
        lines.append(f"{bd.n_cells} cell(s) in the index, {bd.n_failed} failed, {bd.n_incomplete} incomplete; "
                     f"guard-rails from {html.escape(bd.guard_rail_source)}; refreshed {time.strftime('%H:%M:%S')}")
        lines += [html.escape(n) for n in bd.notes]
        lines.append("Ranking column: dev-fold net Sharpe after costs. Test-fold columns are shown for reading and "
                     "never rank (D-020).")
        w["board_status"].value = "<br>".join(lines)
        title = "Leaderboard: " + ", ".join(ids)
        w["board_fig"].layout.display = ""
        show(w["board_fig"], leaderboard_figure(bd.rows, title=title))
        self._sync_selection_options()

    def _no_sweep_html(self, info: Sequence[PD.SweepInfo]) -> str:
        spec_rows = PD.spec_summary(self.choices)
        body = "".join(f"<tr><td>{html.escape(str(r['scenario']))}</td><td>{r['trains']}</td><td>{html.escape(str(r['folds']))}</td>"
                       f"<td>{html.escape(str(r['search']))}</td></tr>" for r in spec_rows)
        why = ("the selected sweep has no cell in the run index yet" if info else
               f"the run store {html.escape(str(self.store.root))} holds no sweep yet")
        return (f"<b>No sweep yet:</b> {why}. Nothing was started by opening this notebook; launch one with the widgets "
                f"above (quick first). The scenario specs a sweep can take:"
                f"<table style='margin:6px 0'><tr><th>scenario</th><th>trains a network</th><th>folds</th><th>search space</th></tr>"
                f"{body}</table>")

    def board(self):
        """The leaderboard view (build once, display anywhere): sweep selector, status and the NT-031 figure."""
        import ipywidgets as w

        if "board_box" in self._w:
            return self._w["board_box"]
        sweeps = w.SelectMultiple(options=[], description="Sweeps", rows=4, layout=w.Layout(width="520px"))
        refresh = w.Button(description="Refresh now", icon="refresh", layout=w.Layout(width="140px"))
        auto = w.Checkbox(value=False, description=f"auto-refresh every {self.poll_seconds:g} s", indent=False,
                          layout=w.Layout(width="250px"))
        status = w.HTML()
        fig = w.Output(layout=w.Layout(min_height="120px"))
        sweeps.observe(lambda ch: None if self._syncing else self.set_board_ids(list(ch["new"])), names="value")
        refresh.on_click(lambda _: self.refresh_board(force=True))
        auto.observe(lambda ch: self.start_polling() if ch["new"] else self.stop_polling(), names="value")
        box = w.VBox([w.HBox([sweeps, w.VBox([refresh, auto])]), status, fig])
        self._w.update(board_box=box, sweeps=sweeps, board_status=status, board_fig=fig, auto=auto)
        self.refresh_board(force=True)
        return box

    def set_board_ids(self, ids: Sequence[str]) -> None:
        """Choose the sweeps on the board (empty: back to the default, the newest sweep)."""
        self.board_ids = list(ids) or None
        self.refresh_board(force=True)

    def start_polling(self, interval: Optional[float] = None) -> None:
        """Re-read the store every ``interval`` seconds in a background thread and redraw the board when it changed."""
        if self._poller is not None and self._poller.is_alive():
            return
        every = float(interval if interval is not None else self.poll_seconds)
        self._stop_poll = threading.Event()
        stop = self._stop_poll
        self._poller = threading.Thread(target=self._poll, args=(stop, every), name="neural_trade-board-poll", daemon=True)
        self._poller.start()
        self._tick_auto(True)

    def stop_polling(self) -> None:
        self._stop_poll.set()
        self._tick_auto(False)

    def _tick_auto(self, on: bool) -> None:
        box = self._w.get("auto")
        if box is not None and box.value != on:
            self._syncing = True
            try:
                box.value = on
            finally:
                self._syncing = False

    def _poll(self, stop: threading.Event, every: float) -> None:
        while not stop.wait(every):
            try:
                self.refresh_board()
            except Exception:   # a half-written file must not end the refresh for good
                logger.warning("board refresh failed", exc_info=True)

    # ------------------------------------------------------------------ the comparison
    def _sync_selection_options(self) -> None:
        sel = self._w.get("compare_rows")
        if sel is None:
            return
        opts = [(f"#{r.rank} {r.scenario} / {r.configuration}", (r.scenario, r.configuration)) for r in self.board_rows()]
        self._syncing = True
        try:
            sel.options = opts
            keep = [v for v in self.selection if v in {o[1] for o in opts}]
            sel.value = tuple(keep or [o[1] for o in opts[:2]])
        finally:
            self._syncing = False

    def compare(self, selection: Optional[Sequence[Tuple[str, str]]] = None, *, role: Optional[str] = None,
                metric: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Compare two or more configurations (default: the board's top two): figures per metric group, the table of
        every score key and the paired verdict. Returns what it drew ({} entries) or None with fewer than two rows."""
        rows = self.board_rows()
        if selection is None:
            selection = self.selection or [(r.scenario, r.configuration) for r in rows[:2]]
        self.selection = [tuple(s) for s in selection]
        self.role = role or self.role
        self.metric = metric or self.metric
        w = self._w
        if len(self.selection) < 2:
            if "compare_box" in w:
                w["compare_note"].value = ("<b>Pick two or more configurations</b> (the board has "
                                           f"{len(rows)} row(s); a comparison needs at least two).")
                w["compare_figs"].children = ()
            return None
        configs = PC.aggregate_selection(self.store, self.selection, role=self.role)
        steps = next((r.horizon_steps for r in rows if r.horizon_steps), None)
        figs = PC.comparison_figures(configs, horizon_steps=steps)
        verdict = PC.verdict_html(configs[0], configs[1], self.store, metric=self.metric, compares_dir=self.compares_dir) \
            if len(configs) == 2 else (f"<b>{len(configs)} configurations selected:</b> a paired verdict is for two; pick two "
                                       "to see NT-032's result or the exploratory per-fold differences.")
        table = PC.table_html(configs)
        if w and "compare_box" in w:
            import ipywidgets as ipw

            outs = []
            for fig in figs.values():
                out = ipw.Output(layout=ipw.Layout(min_height="200px"))
                show(out, fig)
                outs.append(out)
            tbl = ipw.HTML(f"<details><summary><b>Every score key of the selection</b> ({len(PC.metric_table(configs))} rows: "
                           f"value, sd between folds, sd across seeds)</summary><div style='max-height:520px;overflow:auto'>"
                           f"{table}</div></details>")
            w["compare_note"].value = (f"<b>{html.escape(self.role)} cells</b>: " + "; ".join(
                f"{html.escape(c.label)} ({c.n_cells} cell(s), folds {list(c.folds)})" for c in configs))
            w["compare_verdict"].value = verdict
            w["compare_figs"].children = tuple(outs) + (tbl,)
        return {"figures": figs, "configs": configs, "verdict": verdict, "table": table}

    def comparison(self):
        """The comparison view: row selector, role, metric for the exploratory pairs, the verdict, the figures."""
        import ipywidgets as w

        if "compare_box" in self._w:
            return self._w["compare_box"]
        if "board_box" not in self._w:
            self.board()
        rows = w.SelectMultiple(options=[], description="Rows", rows=6, layout=w.Layout(width="620px"))
        role = w.Dropdown(options=[("dev (the ranking's folds)", "dev"), ("test (shown, never ranks)", "test")], value="dev",
                          description="Cells")
        metric = w.Dropdown(options=list(DEFAULT_METRICS), value=self.metric, description="Paired on")
        go = w.Button(description="Compare", icon="bar-chart", button_style="primary")
        note, verdict = w.HTML(), w.HTML()
        figs = w.VBox([])
        rows.observe(lambda ch: None if self._syncing else setattr(self, "selection", list(ch["new"])), names="value")
        go.on_click(lambda _: self.compare(list(rows.value), role=role.value, metric=metric.value))
        box = w.VBox([w.HBox([rows, w.VBox([role, metric, go])]), note, verdict, figs])
        self._w.update(compare_box=box, compare_rows=rows, compare_note=note, compare_verdict=verdict, compare_figs=figs)
        self._sync_selection_options()
        self.compare()
        return box

    # ------------------------------------------------------------------ the launcher widgets
    def launcher(self):
        """Scenario, search space, mode, options and the Estimate / Confirm / Launch buttons (build once)."""
        import ipywidgets as w

        if "launcher_box" in self._w:
            return self._w["launcher_box"]
        lay = lambda px: w.Layout(width=f"{px}px")  # noqa: E731
        scenario = w.Dropdown(options=[(c.label, c.name) for c in self.choices if c.scenario is not None] or [("(no spec)", None)],
                              description="Scenario", layout=lay(420))
        resume = w.Dropdown(options=[NEW_SWEEP], value=NEW_SWEEP, description="Sweep", layout=lay(420))
        mode = w.ToggleButtons(options=list(MODES), value=self.mode, description="Mode")
        name = w.Text(description="Run name", layout=lay(420))
        folds = w.SelectMultiple(options=PD.fold_choices(), description="Folds", rows=6, layout=lay(200))
        add = w.Dropdown(options=[""] + [o.name for o in self.options_by_name.values()], description="Add field", layout=lay(380))
        rules_box = w.VBox([])
        yaml_view = w.HTML()
        o = self.opts
        numbers = {
            "n_trials": w.IntText(value=o["n_trials"], description="optuna trials", layout=lay(220)),
            "max_hours": w.FloatText(value=o["max_hours"], description="max hours", layout=lay(220)),
            "parallel": w.IntText(value=o["parallel"], description="parallel", layout=lay(220)),
            "quick_minutes": w.FloatText(value=o["quick_minutes"], description="quick min.", layout=lay(220)),
            "overhead_s": w.FloatText(value=o["overhead_s"], description="overhead s", layout=lay(220)),
            "sec_per_step": w.FloatText(value=o["sec_per_step"] or 0.0, description="sec/step (0: index)", layout=lay(260)),
            "top_k": w.IntText(value=o["top_k"], description="re-run top", layout=lay(220)),
            "rerun_seeds": w.IntText(value=o["rerun_seeds"], description="re-run seeds", layout=lay(220)),
            "stop_after": w.IntText(value=o["stop_after"] or 0, description="stop after (0: no)", layout=lay(260)),
        }
        record = w.Text(value=o["parallel_record"] or "", description="GPU record", layout=lay(520))
        resume_box = w.Checkbox(value=self.resume, description="resume the existing sweep of this name and mode", indent=False)
        estimate_b = w.Button(description="Estimate (dry run)", icon="calculator", layout=lay(190))
        confirm_b = w.Button(description="Confirm budget", icon="check", button_style="warning", layout=lay(170))
        launch_b = w.Button(description="Launch", icon="play", button_style="success", layout=lay(150))
        status, estimate, log = w.HTML(), w.HTML(), w.Output(layout=w.Layout(max_height="260px", overflow="auto"))
        self._w.update(scenario=scenario, resume_dd=resume, mode=mode, name=name, folds=folds, add=add, rules_box=rules_box,
                       yaml=yaml_view, numbers=numbers, record=record, resume_box=resume_box, estimate_b=estimate_b,
                       confirm_b=confirm_b, launch_b=launch_b, status=status, estimate=estimate, log=log)

        def on(widget, handler, names="value"):
            widget.observe(lambda ch: None if self._syncing else handler(ch["new"]), names=names)

        on(scenario, self.select_scenario)
        on(resume, lambda v: self.select_sweep(None if v == NEW_SWEEP else v))
        on(mode, self.set_mode)
        on(name, lambda v: self.set_option("name", v))
        on(folds, lambda v: self.set_folds(v))
        on(add, lambda v: (self.add_field(v) if v else None))
        on(resume_box, lambda v: self.set_option("resume", v))
        on(record, lambda v: self.set_option("parallel_record", v))
        for key, widget in numbers.items():
            on(widget, lambda v, k=key: self.set_option(k, (v or None) if k in ("sec_per_step", "stop_after") else v))
        estimate_b.on_click(lambda _: self.estimate_now())
        confirm_b.on_click(lambda _: self.confirm())
        launch_b.on_click(lambda _: self.launch_now())
        advanced = w.Accordion(children=[w.VBox([w.HBox([numbers["n_trials"], numbers["max_hours"], numbers["parallel"]]),
                                                 w.HBox([numbers["quick_minutes"], numbers["overhead_s"], numbers["sec_per_step"]]),
                                                 w.HBox([numbers["top_k"], numbers["rerun_seeds"], numbers["stop_after"]]),
                                                 record, resume_box])], titles=("Budget and options (the CLI's flags)",))
        box = w.VBox([w.HBox([scenario, mode]), w.HBox([resume, name]), w.HBox([folds, w.VBox([add, rules_box])]),
                      w.HTML("<b>The <code>search:</code> block this builds</b> (what a scenario file would hold)"), yaml_view,
                      advanced, w.HBox([estimate_b, confirm_b, launch_b, status]), estimate,
                      w.Accordion(children=[log], titles=("Log (warnings and errors of the sweep thread)",))])
        self._w["launcher_box"] = box
        self._sync_widgets(rebuild_rules=True)
        self._set_status(self.status)
        return box

    def _rule_row(self, name: str):
        import ipywidgets as w

        opt = self.options_by_name.get(name)
        rule = self.rules.get(name) or {}
        label = w.HTML(f"<code>{html.escape(name)}</code>" + (f" <small>{html.escape(opt.unit or '')}</small>" if opt else ""),
                       layout=w.Layout(width="190px"))
        drop = w.Button(icon="trash", tooltip=f"remove {name}", layout=w.Layout(width="40px"))
        drop.on_click(lambda _: self.remove_field(name))
        if opt is not None and opt.kind == "choice" or "choices" in rule:
            pool = list(opt.choices) if opt is not None and opt.choices else list(rule.get("choices") or [])
            pick = w.SelectMultiple(options=pool, value=tuple(c for c in (rule.get("choices") or pool) if c in pool), rows=min(4, len(pool) or 1))
            pick.observe(lambda ch: None if self._syncing else self.set_rule(name, choices=list(ch["new"])), names="value")
            return w.HBox([label, pick, drop])
        cast = int if (opt is not None and opt.kind == "int") else float
        low = w.FloatText(value=float(rule.get("low", 0.0)), description="low", layout=w.Layout(width="190px"))
        high = w.FloatText(value=float(rule.get("high", 0.0)), description="high", layout=w.Layout(width="190px"))
        log = w.Checkbox(value=bool(rule.get("log", False)), description="log", indent=False, layout=w.Layout(width="70px"))
        low.observe(lambda ch: None if self._syncing else self.set_rule(name, low=cast(ch["new"])), names="value")
        high.observe(lambda ch: None if self._syncing else self.set_rule(name, high=cast(ch["new"])), names="value")
        log.observe(lambda ch: None if self._syncing else self.set_rule(name, log=True if ch["new"] else None), names="value")
        return w.HBox([label, low, high, log, drop])

    def _sync_widgets(self, rebuild_rules: bool = True) -> None:
        """Make the launcher's widgets show the model (after any change of it); a no-op before the launcher exists."""
        with self._lock:
            self._sync_widgets_locked(rebuild_rules)

    def _sync_widgets_locked(self, rebuild_rules: bool) -> None:
        w = self._w
        if "launcher_box" not in w:
            return
        self._syncing = True
        try:
            if w["scenario"].value != self.scenario_name and self.scenario_name is not None:
                w["scenario"].value = self.scenario_name
            sweeps = PD.list_sweeps(self.store.root)
            w["resume_dd"].options = [NEW_SWEEP] + [s.sweep_id for s in sweeps]
            w["mode"].value = self.mode
            w["add"].value = ""
            w["name"].value = self.name
            w["folds"].value = tuple(f for f in self.folds if f in w["folds"].options)
            w["resume_box"].value = self.resume
            if rebuild_rules:
                w["rules_box"].children = tuple(self._rule_row(n) for n in self.rules)
            problem = PD.validate_search(self.base, self.search()) if self.base is not None else None
            yaml_text = html.escape(self.search_text())
            w["yaml"].value = (f"<pre style='margin:2px 0'>{yaml_text}</pre>"
                               + ("" if self.explicit else "<i>the scenario file has no <code>search:</code> block: the sweep's default "
                                  "space (shown) applies; edit a field to write it into the scenario</i><br>")
                               + (f"<b style='color:#b91c1c'>the sweep would refuse this space: {html.escape(problem)}</b>"
                                  if problem else ""))
            w["confirm_b"].disabled = not self.needs_confirmation
            w["launch_b"].description = "Resume" if self.resume else "Launch"
            w["launch_b"].icon = "step-forward" if self.resume else "play"
            if self.estimate is not None:
                self._show_estimate()
            else:
                w["estimate"].value = ""
        finally:
            self._syncing = False


__all__ = ["ControlPanel", "Estimate", "build_sweep"]
