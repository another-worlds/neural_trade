"""The control panel (notebook 06, NT-034): the search-space choices from the Config metadata, the widgets'
callbacks driven headless against the REAL sweep engine with a fake trainer (as test_sweep.py does), the CLI's
refusals, the explicit confirmation of a big budget, the leaderboard refresh, the comparison figures and the
paired verdict, and the guarantee that executing the notebook top to bottom launches nothing. No test trains a
network."""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from neural_trade.core.config import Config
from neural_trade.experiments.store import RunStore
from tests.test_sweep import FakeTrainer, Monitor, free_gpu, scenario_dict

REPO = Path(__file__).resolve().parent.parent
ipywidgets = pytest.importorskip("ipywidgets")
from neural_trade.notebook import control_panel as CP  # noqa: E402
from neural_trade.notebook import panel_compare as PC  # noqa: E402
from neural_trade.notebook import panel_data as PD  # noqa: E402


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("panel_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def make_specs(tmp_path, csv, **changes):
    specs = tmp_path / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    (specs / "panel_tiny.yaml").write_text(yaml.safe_dump(scenario_dict(csv, name="panel_tiny", **changes)), encoding="utf-8")
    return specs


def make_panel(tmp_path, csv, trainer=None, *, changes=None, **kw):
    specs = make_specs(tmp_path, csv, **(changes or {}))
    trainer = trainer if trainer is not None else FakeTrainer()
    kwargs = dict(trainer=trainer, gpu_check=free_gpu, monitor_factory=Monitor(), sleep=lambda s: None)
    panel = CP.ControlPanel(store=tmp_path / "runs", specs_dir=specs, compares_dir=tmp_path / "compares",
                            sweep_kwargs=kwargs, parallel_record=str(tmp_path / "no_such_record.json"), **kw)
    panel.set_option("sec_per_step", 0.001)
    panel.set_option("overhead_s", 70.0)
    return panel, trainer


# ------------------------------------------------------------------ the choices come from the metadata
def test_the_search_space_options_are_the_tunable_fields_with_their_metadata_ranges():
    opts = {o.name: o for o in PD.field_options()}
    specs = Config.field_specs()
    assert set(opts) == {n for n, s in specs.items() if s.tunable and not s.deprecated} - {"RESAMPLE_MINUTES"}
    lr = opts["LR"]
    assert (lr.kind, lr.log, lr.high) == ("float", True, specs["LR"].maximum)
    assert lr.low is None and lr.needs_bounds                    # the metadata's lower bound 0 is exclusive: not a log range
    rule = lr.default_rule()                                     # so the form starts from a guess around the default
    assert 0 < rule["low"] < specs["LR"].default < rule["high"] and rule["log"] is True
    assert opts["BATCH_SIZE"].kind == "int" and (opts["BATCH_SIZE"].low, opts["BATCH_SIZE"].high) == (16, 2048)
    assert opts["LAMBDA_DIR"].needs_bounds                       # no finite range in the metadata: low and high are asked for
    assert "FOLD_INDEX" not in opts and "EARLY" not in opts      # reserved by the engine / not tunable


def test_the_fold_range_is_dynamic_and_registry_keys_come_from_the_registries():
    assert PD.fold_choices(Config(N_FOLDS=3)) == [-3, -2, -1, 0, 1, 2]
    assert PD.fold_choices(Config(N_FOLDS=6))[0] == -6
    assert "gru_attention" in PD.registry_choices("MODEL_NAME")
    assert PD.registry_choices("LR") == ()


def test_the_search_block_the_form_builds_is_the_one_the_sweep_validates(tmp_path, bars_csv):
    panel, _ = make_panel(tmp_path, bars_csv)
    panel.add_field("RHO_MAX")
    panel.set_rule("LR", low=1e-3, high=1e-2)
    block = panel.search()
    assert block["RHO_MAX"] == {"low": 0.0, "high": 1.0} and block["LR"]["low"] == 1e-3
    assert yaml.safe_load(panel.search_text())["search"]["RHO_MAX"] == {"low": 0.0, "high": 1.0}
    assert PD.validate_search(panel.base, block) is None
    assert "not tunable" in PD.validate_search(panel.base, {"EARLY": None})
    with pytest.raises(KeyError):
        panel.add_field("EARLY")


# ------------------------------------------------------------------ the callbacks, against the real sweep engine
def press(button, panel):
    button.click()
    panel.wait(60)


def test_the_widgets_estimate_first_then_launch_a_quick_sweep_through_the_sweep_engine(tmp_path, bars_csv):
    panel, trainer = make_panel(tmp_path, bars_csv)
    box = panel.launcher()
    assert box is panel.launcher()                                # built once
    w = panel._w
    assert w["scenario"].value == "panel_tiny" and w["mode"].value == "quick"
    assert "LR" in w["yaml"].value and "LAMBDA_DIR" in w["yaml"].value          # the search: block it builds is shown
    # a widget edit reaches the form's model
    w["numbers"]["quick_minutes"].value = 5.0
    w["mode"].value = "quick"
    press(w["estimate_b"], panel)
    assert trainer.calls == [] and panel.launches == []           # an estimate trains nothing
    assert panel.estimate is not None and "estimated" in panel.estimate.text and "quick" in panel.estimate.text
    assert "estimated" in w["estimate"].value                      # printed estimate is on screen before any launch
    press(w["launch_b"], panel)
    assert panel.error is None and panel.refusal is None, (panel.error, panel.refusal)
    assert len(trainer.calls) == 4 and len(panel.launches) == 1 and panel.result.state == "quick_complete"
    # the same engine path as the CLI: the sweep summary and the run index are where `neural-trade sweep` puts them
    store = RunStore(tmp_path / "runs")
    summary = json.loads((store.root / "sweeps" / "panel_tiny-quick" / "sweep.json").read_text(encoding="utf-8"))
    assert summary["label"] == "quick" and summary["scenario"] == "panel_tiny"
    assert {r["role"] for r in store.index.rows("panel_tiny-quick")} == {"dev"}


def test_the_form_edits_reach_the_scenario_and_options_a_launch_would_use(tmp_path, bars_csv):
    panel, _ = make_panel(tmp_path, bars_csv)
    panel.launcher()
    w = panel._w
    w["folds"].value = (-2, -1)
    w["name"].value = "panel_edit"
    w["numbers"]["n_trials"].value = 3
    w["numbers"]["max_hours"].value = 2.5
    w["numbers"]["parallel"].value = 1
    w["mode"].value = "optuna"
    sc, opts = panel.build_scenario(), panel.build_options()
    assert sc.name == "panel_edit" and sc.folds == [-2, -1] and "LR" in sc.search
    assert (opts.mode, opts.n_trials, opts.max_hours, opts.parallel, opts.dry_run) == ("optuna", 3, 2.5, 1, False)
    assert panel.build_options(dry_run=True).dry_run is True
    panel.add_field("RHO_MAX")
    assert "RHO_MAX" in w["yaml"].value and len(w["rules_box"].children) == 3
    panel.remove_field("RHO_MAX")
    assert "RHO_MAX" not in w["yaml"].value


def test_a_budget_above_the_threshold_needs_the_confirm_button_before_launch_starts_anything(tmp_path, bars_csv):
    panel, trainer = make_panel(tmp_path, bars_csv, confirm_gpu_hours=1e-4)
    panel.launcher()
    w = panel._w
    press(w["launch_b"], panel)                                   # estimate shown, not launched
    assert trainer.calls == [] and panel.launches == [] and panel.needs_confirmation
    assert "Confirm budget" in panel.status and not w["confirm_b"].disabled
    press(w["launch_b"], panel)                                   # still not confirmed: still nothing
    assert trainer.calls == [] and panel.launches == []
    w["confirm_b"].click()
    assert not panel.needs_confirmation
    press(w["launch_b"], panel)
    assert len(panel.launches) == 1 and len(trainer.calls) == 4


def test_changing_the_form_revokes_an_estimate_and_its_confirmation(tmp_path, bars_csv):
    panel, trainer = make_panel(tmp_path, bars_csv, confirm_gpu_hours=1e-4)
    panel.launcher()
    press(panel._w["estimate_b"], panel)
    assert panel.confirm() and panel._confirmed
    panel.set_option("overhead_s", 71.0)
    assert panel.estimate is None and panel._confirmed is None
    assert not panel.confirm()                                    # nothing to confirm until it is estimated again
    assert trainer.calls == []


def test_the_panel_refuses_what_the_cli_refuses_and_starts_nothing(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    panel, trainer = make_panel(tmp_path, bars_csv)
    panel.launcher()
    w = panel._w
    # a budget above max hours (optuna mode)
    w["mode"].value = "optuna"
    w["numbers"]["max_hours"].value = 0.0001
    w["numbers"]["overhead_s"].value = 70.0
    press(w["launch_b"], panel)
    assert panel.refusal and "max-hours" in panel.refusal and "nothing was started" in panel.refusal
    assert "refused" in panel.status and trainer.calls == [] and panel.launches == []
    assert not (tmp_path / "runs" / "sweeps").exists()
    # parallel above the GPU record's allowed N (no record means N = 1)
    w["numbers"]["max_hours"].value = 12.0
    w["numbers"]["parallel"].value = 3
    press(w["launch_b"], panel)
    assert panel.refusal and "--parallel 3 is refused" in panel.refusal and panel.launches == []
    # a bar size other than 1 minute
    w["numbers"]["parallel"].value = 1
    panel2, trainer2 = make_panel(tmp_path / "b", bars_csv, changes={"overrides": {**scenario_dict(bars_csv)["overrides"],
                                                                                    "RESAMPLE_MINUTES": 5}})
    panel2.launcher()
    press(panel2._w["launch_b"], panel2)
    assert panel2.refusal and "RESAMPLE_MINUTES" in panel2.refusal and trainer2.calls == [] and panel2.launches == []


def test_an_existing_sweep_needs_resume_and_a_resume_continues_it(tmp_path, bars_csv):
    panel, trainer = make_panel(tmp_path, bars_csv)
    panel.launcher()
    press(panel._w["launch_b"], panel)
    n = len(trainer.calls)
    assert n == 4 and panel.launches
    press(panel._w["launch_b"], panel)                            # the CLI's own refusal: it exists, pass --resume
    assert panel.refusal and "resume" in panel.refusal and len(trainer.calls) == n
    panel._w["resume_dd"].value = "panel_tiny-quick"              # pick the sweep: resume is ticked, the button says Resume
    assert panel.resume and panel._w["launch_b"].description == "Resume"
    press(panel._w["launch_b"], panel)
    assert panel.error is None and len(trainer.calls) == n        # finished cells are never trained again


def test_a_sweep_that_raises_is_an_error_output_in_the_log_widget(tmp_path, bars_csv):
    class Boom:
        def __init__(self, *a, **k):
            pass

        def run(self):
            raise RuntimeError("engine exploded")

    panel, _ = make_panel(tmp_path, bars_csv)
    panel._factory = lambda *a, **k: Boom()
    panel.launcher()
    press(panel._w["estimate_b"], panel)
    assert isinstance(panel.error, RuntimeError) and panel.status.startswith("failed")
    outs = panel._w["log"].outputs
    assert outs[0]["output_type"] == "error" and outs[0]["ename"] == "RuntimeError"


def test_a_stubbed_factory_records_only_real_launches_not_estimates(tmp_path, bars_csv):
    calls = []

    class Stub:
        def __init__(self, dry):
            self.dry = dry

        def run(self):
            calls.append(self.dry)
            from types import SimpleNamespace

            return SimpleNamespace(sweep_id="stub", state="dry_run" if self.dry else "quick_complete", stop_reason=None,
                                   budget={"estimated_minutes": 1.0, "n_trials": 4})

    panel, _ = make_panel(tmp_path, bars_csv)
    panel._factory = lambda sc, store, opts, announce, **kw: Stub(opts.dry_run)
    panel.estimate_now()
    panel.wait(30)
    assert calls == [True] and panel.launches == []
    panel.launch_now()
    panel.wait(30)
    assert calls == [True, True, False] and len(panel.launches) == 1


# ------------------------------------------------------------------ the board and the comparison
def run_quick(tmp_path, bars_csv, **kw):
    panel, trainer = make_panel(tmp_path, bars_csv, **kw)
    panel.launcher()
    press(panel._w["launch_b"], panel)
    assert panel.error is None and panel.refusal is None, (panel.error, panel.refusal)
    return panel, trainer


def test_the_board_shows_the_ranking_and_test_columns_as_the_leaderboard_labels_them(tmp_path, bars_csv):
    panel, _ = run_quick(tmp_path, bars_csv)
    rows = panel.board_rows()
    assert len(rows) == 4 and [r.rank for r in rows] == [1, 2, 3, 4]
    box = panel.board()
    fig_json = panel._w["board_fig"].outputs[0]["data"]["application/vnd.plotly.v1+json"]
    text = json.dumps(fig_json)
    assert "not used for ranking" in text and "Dev-fold net Sharpe" in text
    status = panel._w["board_status"].value
    assert "quick" in status and "never rank" in status and box is panel.board()


def test_the_board_refreshes_while_the_run_index_changes_without_blocking_the_caller(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    panel, trainer = make_panel(tmp_path, bars_csv)
    panel.launcher()
    panel.board()
    assert "No sweep yet" in panel._w["board_status"].value        # the empty store: an explicit state, not a blank panel
    assert panel.refresh_board() is False                          # nothing changed: no redraw
    seen = []
    gate = threading.Event()
    real = panel.refresh_board

    def spy(**kw):
        drew = real(**kw)
        if drew:
            seen.append(len(panel.board_rows()))
        return drew

    panel.refresh_board = spy
    panel.start_polling(0.05)
    assert panel._w["auto"].value is True
    gate.set()
    press(panel._w["launch_b"], panel)
    deadline = time.time() + 20
    while time.time() < deadline and not (seen and seen[-1] == 4):
        time.sleep(0.05)
    panel.stop_polling()
    assert seen and seen[-1] == 4, seen
    assert panel._w["auto"].value is False


def test_the_board_puts_several_sweeps_on_one_board(tmp_path, bars_csv):
    panel, _ = run_quick(tmp_path, bars_csv)
    panel.select_scenario("panel_tiny")
    panel.set_option("name", "panel_other")
    panel.set_option("resume", False)
    press(panel._w["launch_b"], panel)
    assert PD.default_board_sweeps(panel.sweeps or PD.list_sweeps(panel.store.root))[0] in {"panel_tiny-quick", "panel_other-quick"}
    panel.set_board_ids(["panel_tiny-quick", "panel_other-quick"])
    scenarios = {r.scenario for r in panel.board_rows()}
    assert scenarios == {"panel_tiny-quick", "panel_other-quick"} and len(panel.board_rows()) == 8
    assert any("2 sweeps on one board" in n for n in panel.board_data.notes)


def test_comparing_two_rows_draws_every_metric_per_horizon_with_both_spreads(tmp_path, bars_csv):
    panel, _ = run_quick(tmp_path, bars_csv)
    rows = panel.board_rows()
    sel = [(rows[0].scenario, rows[0].configuration), (rows[1].scenario, rows[1].configuration)]
    out = panel.compare(sel)
    figs, configs = out["figures"], out["configs"]
    assert {"direction", "variance", "delta", "backtest"} <= set(figs)
    from neural_trade.visualization.theme import HORIZON_COLORS, empty_panels

    for group, fig in figs.items():
        assert not empty_panels(fig), group
    # one trace per horizon in the horizon's colour, in a per-horizon group
    colors = {t.marker.color if isinstance(t.marker.color, str) else t.marker.color[0] for t in figs["direction"].data if t.showlegend}
    assert colors == set(HORIZON_COLORS.values())
    # every score key of the configurations is in the table; every per-horizon metric is a panel
    table = PC.metric_table(configs)
    keys = {k for c in configs for k in c.aggs}
    assert set(table.index) == keys and len(keys) > 100
    titles = {a.text for f in figs.values() for a in f.layout.annotations}
    per_horizon = {PC.key_parts(k)[2] for k in keys if PC.key_parts(k) and PC.key_parts(k)[0] in ("direction", "variance", "delta")}
    assert per_horizon <= titles
    # the spreads: the fold sd is between dev folds; the dot is the mean of fold means
    a = configs[0].aggs["backtest/sharpe_net"]
    assert a.n_folds >= 1 and a.value == pytest.approx(sum(v for _, v in a.fold_values) / len(a.fold_values))
    assert "not a verdict" in out["verdict"] and "exploratory" in out["verdict"].lower()
    assert "dev" in panel._w["compare_note"].value if "compare_note" in panel._w else True


def test_the_comparison_view_is_built_with_the_top_two_rows_and_a_compare_button(tmp_path, bars_csv):
    panel, _ = run_quick(tmp_path, bars_csv)
    box = panel.comparison()
    w = panel._w
    assert len(w["compare_rows"].value) == 2 and len(w["compare_figs"].children) >= 3
    w["compare_rows"].value = tuple(v for _, v in w["compare_rows"].options[:3])
    w["compare_verdict"].value = ""
    box.children[0].children[1].children[-1].click()               # the Compare button
    assert "paired" in w["compare_verdict"].value.lower() or "selected" in w["compare_verdict"].value.lower()
    w["compare_rows"].value = tuple(v for _, v in w["compare_rows"].options[:1])
    box.children[0].children[1].children[-1].click()
    assert "Pick two or more" in w["compare_note"].value


def test_a_pre_registered_comparison_gives_nt032s_verdict_otherwise_the_pair_is_exploratory(tmp_path, bars_csv):
    panel, _ = run_quick(tmp_path, bars_csv)
    rows = panel.board_rows()
    a, b = rows[0], rows[1]
    configs = PC.aggregate_selection(panel.store, [(a.scenario, a.configuration), (b.scenario, b.configuration)])
    html_ = PC.verdict_html(configs[0], configs[1], panel.store, compares_dir=tmp_path / "compares")
    assert "not a verdict" in html_ and "D-025" in html_ and "mean difference" in html_
    cmp_dir = tmp_path / "compares"
    cmp_dir.mkdir()
    spec = {"name": "panel_pair", "scenario_a": a.scenario, "scenario_b": b.scenario, "configuration_a": a.configuration,
            "configuration_b": b.configuration, "metric": "backtest/sharpe_net", "min_effect": 0.1,
            "judgment_folds": [-2], "registered_utc": "2099-01-01T00:00:00Z", "root": str(panel.store.root)}
    (cmp_dir / "panel_pair.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    found, flipped = PC.find_compare_spec(configs[0], configs[1], cmp_dir)
    assert found is not None and found.name == "panel_pair.yaml" and not flipped
    assert PC.find_compare_spec(configs[1], configs[0], cmp_dir)[1] is True
    verdict = PC.verdict_html(configs[0], configs[1], panel.store, compares_dir=cmp_dir)
    assert "Paired verdict (NT-032" in verdict and "no verdict yet" in verdict   # viewing never runs compare()
    assert not (panel.store.root / "compares").exists()


# ------------------------------------------------------------------ the notebook launches nothing
def _build_module():
    spec = importlib.util.spec_from_file_location("nb_build_panel", REPO / "scripts" / "notebooks" / "build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_notebook_06_is_generated_by_build_py_and_listed():
    build = _build_module()
    assert "06_control_panel" in build.NOTEBOOKS
    assert (REPO / "notebooks" / "06_control_panel.ipynb").exists()
    assert "06_control_panel" in (REPO / "scripts" / "notebooks" / "README.md").read_text(encoding="utf-8")


def _exec_notebook(tmp_path, params, monkeypatch):
    """Run 06's code cells top to bottom in this process (as the notebook routine does, minus the kernel)."""
    book = _build_module().build(["06"])["06_control_panel"]
    ns: dict = {}
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "parameters" in cell.metadata.get("tags", []):
            src += "\n" + "\n".join(f"{k} = {v!r}" for k, v in params.items())
        exec(compile(src, "<06>", "exec"), ns)  # noqa: S102
    return ns


def test_executing_the_notebook_top_to_bottom_launches_nothing_and_says_no_sweep_yet(tmp_path, bars_csv, monkeypatch):
    made = []
    real = CP.build_sweep

    def spy(scenario, store, options, announce, **kw):
        made.append(options.dry_run)
        return real(scenario, store, options, announce, **kw)

    monkeypatch.setattr(CP, "build_sweep", spy)
    specs = make_specs(tmp_path, bars_csv)
    ns = _exec_notebook(tmp_path, {"STORE": str(tmp_path / "runs"), "SPECS_DIR": str(specs),
                                   "COMPARES_DIR": str(tmp_path / "compares")}, monkeypatch)
    panel = ns["panel"]
    try:
        assert made == [] and panel.launches == [] and not panel.running           # no sweep built, let alone run
        assert not (tmp_path / "runs" / "sweeps").exists() and not (tmp_path / "runs" / "scenarios").exists()
        text = panel._w["board_status"].value
        assert "No sweep yet" in text and "panel_tiny" in text                     # the specs available are listed
        assert panel._w["board_fig"].outputs == ()
        assert "Nothing was started" in text
        assert panel._poller is not None and panel._poller.is_alive()              # the board polls, read-only
    finally:
        panel.stop_polling()


def test_executing_the_notebook_on_a_store_with_a_sweep_shows_it_and_still_launches_nothing(tmp_path, bars_csv, monkeypatch):
    first, _ = run_quick(tmp_path, bars_csv)
    made = []
    monkeypatch.setattr(CP, "build_sweep", lambda *a, **k: made.append(1))
    ns = _exec_notebook(tmp_path, {"STORE": str(tmp_path / "runs"), "SPECS_DIR": str(tmp_path / "specs"),
                                   "COMPARES_DIR": str(tmp_path / "compares")}, monkeypatch)
    panel = ns["panel"]
    try:
        assert made == [] and panel.launches == []
        assert len(panel.board_rows()) == 4
        assert panel._w["board_fig"].outputs and len(panel._w["compare_figs"].children) >= 3
    finally:
        panel.stop_polling()


# ------------------------------------------------------------------ the session's warnings reach the waiting cell
def test_training_session_wait_re_emits_the_threads_warnings_in_the_waiting_thread(monkeypatch):
    pytest.importorskip("tensorflow")
    import neural_trade.training.trainer as trainer_mod
    from neural_trade.notebook import TrainingSession

    log = logging.getLogger("neural_trade.training.trainer")

    def fake_train(**kw):
        log.warning("CalibrationPipeline fit FAILED (continuing without calibration): boom")
        log.info("an info record stays in the log widget only")
        return "result"

    monkeypatch.setattr(trainer_mod, "train_and_evaluate", fake_train)
    where = []

    class Spy(logging.Handler):
        def emit(self, record):
            where.append((record.levelno, threading.current_thread() is threading.main_thread()))

    spy = Spy(level=logging.DEBUG)
    pkg = logging.getLogger("neural_trade")
    pkg.addHandler(spy)
    old = pkg.level
    pkg.setLevel(logging.DEBUG)
    try:
        s = TrainingSession(Config(EPOCHS=1), epochs=1, calibrate=False)
        s.start()
        time.sleep(0.01)
        assert s.wait(30) == "result"
    finally:
        pkg.removeHandler(spy)
        pkg.setLevel(old)
    # the session mutes other handlers for its thread's records (they go to its log widget), so the only
    # emission a handler of the package logger sees is the re-emission in the cell that waited
    assert where.count((logging.WARNING, True)) == 1
    assert (logging.INFO, True) not in where                  # only WARNING and above are re-emitted
    assert s._warnings.records == []                          # once
