"""Control panel, repair round 1 of NT-034 (QA of 027be49): the notebook's working directory, the n/a of a zero beta
(D-007), read-only verdicts, the unchanged spec hash and concurrent refreshes."""
from __future__ import annotations

import json
import sqlite3
import threading
import traceback

import yaml

from neural_trade.notebook import control_panel as CP
from neural_trade.notebook import panel_compare as PC
from tests.test_control_panel import REPO, _exec_notebook, bars_csv, run_quick  # noqa: F401
from tests.test_sweep import FakeTrainer, Monitor, free_gpu


def real_panel(tmp_path, trainer, **kw):
    kwargs = dict(trainer=trainer, gpu_check=free_gpu, monitor_factory=Monitor(), sleep=lambda s: None)
    return CP.ControlPanel(store=tmp_path / "runs", specs_dir=REPO / "configs" / "scenarios",
                           compares_dir=REPO / "configs" / "compares", sweep_kwargs=kwargs, **kw)


def test_estimate_and_parallel_work_from_the_notebooks_working_directory_on_the_real_scenarios(tmp_path, monkeypatch):
    """The notebook runs with notebooks/ as its working directory: reference.yaml's relative CSV_PATH and the GPU
    record's relative default path are the repository root's (where the CLI runs), so the panel resolves them there."""
    monkeypatch.chdir(REPO / "notebooks")
    trainer = FakeTrainer()
    panel = real_panel(tmp_path, trainer)
    assert panel.scenario_name == "reference_default"
    panel.set_option("sec_per_step", 0.01)
    panel.estimate_now()
    panel.wait(120)
    assert panel.refusal is None and panel.error is None, (panel.refusal, panel.error)
    assert panel.estimate is not None and "estimated" in panel.estimate.text and trainer.calls == []
    from neural_trade.data.loaders import resolve_data_path

    assert resolve_data_path(panel.build_scenario().base().CSV_PATH).is_file()   # opened from the root, text unchanged
    record = json.loads((REPO / "runs" / "experiments" / "gpu_measurements_v1" / "parallel_n.json").read_text(encoding="utf-8"))
    panel.set_option("parallel", int(record["allowed_n"]))                 # what the recorded measurement allows
    panel.estimate_now()
    panel.wait(120)
    assert panel.refusal is None and panel.error is None, (panel.refusal, panel.error)
    panel.set_option("parallel", int(record["allowed_n"]) + 1)             # above it: the CLI's own refusal
    panel.estimate_now()
    panel.wait(120)
    assert panel.refusal and "refused" in panel.refusal


def test_a_scenario_without_a_search_block_keeps_its_spec_hash_until_the_form_is_edited(tmp_path, monkeypatch):
    from neural_trade.experiments.scenario import Scenario

    monkeypatch.chdir(REPO)
    panel = real_panel(tmp_path, FakeTrainer())
    file_hash = Scenario.from_yaml(REPO / "configs" / "scenarios" / "reference.yaml").spec_hash
    assert panel.scenario_name == "reference_default" and not panel.base.search
    assert panel.search() == {} and panel.build_scenario().spec_hash == file_hash
    assert panel.rules                                                     # the default space is still shown
    panel.launcher()
    assert "no <code>search:</code> block" in panel._w["yaml"].value
    panel.set_rule("LR", low=1e-3)
    assert panel.search() and panel.build_scenario().spec_hash != file_hash


def _zero_beta(panel, scenario, configuration, horizons=("h0",)):
    """Make a configuration's dev cells look as the scorer stores a constant served delta: beta 0 in the cell's report
    and measured-looking zeros in the index."""
    store = panel.store
    for r in store.index.rows(scenario):
        if r["configuration"] != configuration or r["role"] != "dev":
            continue
        path = store.root / r["run_dir"] / "eval_report_dev.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("meta", {})["delta_scale"] = {h: 0.0 for h in horizons}
        path.write_text(json.dumps(doc), encoding="utf-8")
        with sqlite3.connect(store.index_path) as con:
            for h in horizons:
                for m in ("corr", "corr_spearman", "mean_pred", "share_pred_up"):
                    con.execute("UPDATE scores SET value = 0.0 WHERE run_id = ? AND name = ?", (r["run_id"], f"{h}/delta/{m}"))


def test_served_delta_statistics_of_a_zero_beta_are_na_not_zero_dots(tmp_path, bars_csv):  # noqa: F811
    panel, _ = run_quick(tmp_path, bars_csv)
    rows = panel.board_rows()
    a, b = rows[0], rows[1]
    _zero_beta(panel, a.scenario, a.configuration, horizons=("h0",))
    ca, cb = PC.aggregate_selection(panel.store, [(a.scenario, a.configuration), (b.scenario, b.configuration)])
    assert "h0/delta/corr" not in ca.aggs and ca.na["h0/delta/corr"] >= 1          # n/a, not a measured 0
    assert "h1/delta/corr" in ca.aggs and "h0/delta/corr" in cb.aggs               # other horizons, configurations untouched
    assert "h0/delta/mae" in ca.aggs and "h0/delta_raw/corr" in ca.aggs            # the raw heads stay alongside
    fig = PC.comparison_figures([ca, cb])["delta"]
    assert PC.NA_TEXT in [x.text for x in fig.layout.annotations]
    assert "not a measured 0" in fig.layout.title.text
    from neural_trade.visualization.theme import empty_panels

    assert not empty_panels(fig)
    table = PC.metric_table([ca, cb])
    assert table.loc["h0/delta/corr", (ca.label, "value")] == PC.NA_TEXT                                       # n/a, said why


def test_viewing_a_verdict_and_executing_the_notebook_write_nothing_into_the_store(tmp_path, bars_csv, monkeypatch):  # noqa: F811
    panel, _ = run_quick(tmp_path, bars_csv)
    rows = panel.board_rows()
    a, b = rows[0], rows[1]
    cmp_dir = tmp_path / "compares"
    cmp_dir.mkdir()
    spec = {"name": "panel_pair", "scenario_a": a.scenario, "scenario_b": b.scenario, "configuration_a": a.configuration,
            "configuration_b": b.configuration, "metric": "backtest/sharpe_net", "min_effect": 0.1,
            "judgment_folds": [-2], "registered_utc": "2099-01-01T00:00:00Z"}
    (cmp_dir / "panel_pair.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")

    def tree():
        return sorted((str(p.relative_to(panel.store.root)), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in panel.store.root.rglob("*") if p.is_file())

    before = tree()
    out = panel.compare([(a.scenario, a.configuration), (b.scenario, b.configuration)])
    assert "no verdict yet" in out["verdict"] and "neural-trade compare" in out["verdict"]
    ns = _exec_notebook(tmp_path, {"STORE": str(panel.store.root), "SPECS_DIR": str(tmp_path / "specs"),
                                   "COMPARES_DIR": str(cmp_dir)}, monkeypatch)
    ns["panel"].stop_polling()
    assert tree() == before and not (panel.store.root / "compares").exists()
    # a stored result of the same spec is shown as it is; a result of another content is refused as stale
    from neural_trade.experiments.comparator import CompareSpec

    cs = CompareSpec.from_yaml(cmp_dir / "panel_pair.yaml")
    res = panel.store.root / "compares" / "panel_pair"
    res.mkdir(parents=True)
    doc = {"spec_hash": cs.spec_hash, "verdict": "inconclusive", "estimate": {"estimate": 0.5, "ci_lo": -1.0, "ci_hi": 2.0},
           "n_folds": 6, "n_pairs": 6, "generated_utc": "2026-10-06T00:00:00Z"}
    (res / "result.json").write_text(json.dumps(doc), encoding="utf-8")
    ca, cb = PC.aggregate_selection(panel.store, [(a.scenario, a.configuration), (b.scenario, b.configuration)])
    assert "inconclusive" in PC.verdict_html(ca, cb, panel.store, compares_dir=cmp_dir)
    (res / "result.json").write_text(json.dumps({**doc, "spec_hash": "0" * 12}), encoding="utf-8")
    assert "stale" in PC.verdict_html(ca, cb, panel.store, compares_dir=cmp_dir)


def test_concurrent_refreshes_from_several_threads_do_not_collide_and_the_sync_flag_is_per_thread(tmp_path, bars_csv):  # noqa: F811
    panel, _ = run_quick(tmp_path, bars_csv)
    panel.board()
    panel.comparison()
    n_threads, rounds = 6, 8
    errors: list = []
    start = threading.Barrier(n_threads, timeout=60)      # every thread refreshes at once: the collision is forced

    def hammer(i):
        try:
            start.wait()
            for _ in range(rounds):
                panel.refresh_board(force=True)
                if i % 2:
                    panel.set_option("n_trials", 30 + i)
                assert not panel._syncing
        except BaseException:  # noqa: BLE001 - the full trace is the evidence when this fails
            errors.append(f"thread {i}:\n{traceback.format_exc()}")

    threads = [threading.Thread(target=hammer, args=(i,), name=f"hammer-{i}") for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"threads still running after 300 s: {alive}"
    assert errors == [], "\n".join(errors)
    assert len(panel.board_rows()) == 4
    # a click handled in this thread while another thread is mid-refresh is not muted by that thread's flag
    entered, release = threading.Event(), threading.Event()

    def other():
        panel._syncing = True
        entered.set()
        release.wait(60)
        panel._syncing = False

    t = threading.Thread(target=other)
    t.start()
    assert entered.wait(60)
    try:
        assert panel._syncing is False
    finally:
        release.set()
        t.join(60)
    assert not t.is_alive()
