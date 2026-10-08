"""Control panel, repair round 2 of NT-034 (re-QA of eef0fd6): a sweep the CLI ran from the repo root resumes in the
panel from notebooks/ (and back) with zero retraining and the same cell identities; stored compare results are found
where ``neural-trade compare --out runs/compares/<file stem>`` writes them, and shown in full (D-014); the panel's
n/a rule at beta = 0 is the report's (D-007)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest import mock

import pytest
import yaml

from neural_trade.data import loaders as L
from neural_trade.experiments import sweep as SW
from neural_trade.experiments.store import RunStore
from neural_trade.notebook import control_panel as CP
from neural_trade.notebook import panel_compare as PC
from tests.test_control_panel import REPO, bars_csv  # noqa: F401
from tests.test_sweep import FakeTrainer, Monitor, free_gpu, scenario_dict

KW = dict(gpu_check=free_gpu, sleep=lambda s: None)
OPTS = dict(sec_per_step=0.001, overhead_s=5.0, n_trials=3, top_k=2, rerun_seeds=2)


# ------------------------------------------------------------------ P1-A: resume across the CLI and the panel
@pytest.fixture
def project(tmp_path, bars_csv, monkeypatch):  # noqa: F811
    """A project root with a bar file and a scenario whose CSV_PATH is relative to the root (as every committed
    scenario's is), plus a notebooks/ folder: the data layer's project root points at it."""
    root = tmp_path / "proj"
    (root / "notebooks").mkdir(parents=True)
    (root / "specs").mkdir()
    shutil.copy(bars_csv, root / "bars.csv")
    spec = scenario_dict("bars.csv", name="rel")
    (root / "specs" / "rel.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(L, "PROJECT_ROOT", root, raising=False)
    return root


def cli_sweep(root, store, trainer, mode, monkeypatch, *extra):
    from neural_trade import cli

    monkeypatch.chdir(root)
    real = SW.Sweep.__init__

    def init(self, scenario, store_, options=None, **kw):
        kw.update(trainer=trainer, monitor_factory=Monitor(), **KW)
        real(self, scenario, store_, options, **kw)

    flags = ["--sec-per-step", "0.001", "--overhead-s", "5", "--parallel-record", "none.json"]
    if mode == "optuna":
        flags += ["--n-trials", "3", "--top-k", "2", "--rerun-seeds", "2"]
    with mock.patch.object(SW.Sweep, "__init__", init):
        args = cli.build_parser().parse_args(["sweep", "specs/rel.yaml", "--mode", mode, "--store", str(store), *flags,
                                              *extra])
        return cli.cmd_sweep(args)


def panel_sweep(root, store, trainer, mode, monkeypatch, *, resume_id=None, stop_after=None):
    monkeypatch.chdir(root / "notebooks")
    p = CP.ControlPanel(store=store, specs_dir="../specs", compares_dir="../compares", mode=mode, root=root,
                        sweep_kwargs=dict(trainer=trainer, monitor_factory=Monitor(), **KW), parallel_record="none.json")
    p.launcher()
    for k, v in OPTS.items():
        p.set_option(k, v)
    if stop_after:
        p.set_option("stop_after", stop_after)
    if resume_id:
        p._w["resume_dd"].value = resume_id
    p._w["launch_b"].click()
    p.wait(300)
    assert p.error is None and p.refusal is None, (p.error, p.refusal)
    return p


def identities(store, sweep_id):
    rows = RunStore(store).index.rows(sweep_id)
    keys = [(r["configuration"], r["fold"], r["seed"], r["role"]) for r in rows]
    return rows, keys


@pytest.mark.parametrize("mode", ["quick", "optuna"])
def test_a_cli_sweep_resumes_in_the_panel_from_notebooks_without_retraining(project, tmp_path, mode, monkeypatch):
    if mode == "optuna":
        pytest.importorskip("optuna")
    store = tmp_path / "store"
    t1, t2 = FakeTrainer(), FakeTrainer()
    assert cli_sweep(project, store, t1, mode, monkeypatch, *(["--stop-after", "3"] if mode == "optuna" else [])) == 0
    rows_before, _ = identities(store, f"rel-{mode}")
    p = panel_sweep(project, store, t2, mode, monkeypatch, resume_id=f"rel-{mode}")
    assert t2.calls == [], f"the panel retrained cells the CLI finished: {t2.calls}"
    rows, keys = identities(store, f"rel-{mode}")
    assert len(keys) == len(set(keys)) and len(rows) == len(rows_before)
    assert p.build_scenario().overrides["CSV_PATH"] == "bars.csv"             # the spec's text: the cells' identity


@pytest.mark.parametrize("mode", ["quick", "optuna"])
def test_a_panel_sweep_from_notebooks_resumes_in_the_cli_without_retraining(project, tmp_path, mode, monkeypatch):
    if mode == "optuna":
        pytest.importorskip("optuna")
    store = tmp_path / "store"
    t1, t2 = FakeTrainer(), FakeTrainer()
    p = panel_sweep(project, store, t1, mode, monkeypatch, stop_after=3 if mode == "optuna" else None)
    assert t1.calls and p.build_scenario().overrides["CSV_PATH"] == "bars.csv"
    rows_before, _ = identities(store, f"rel-{mode}")
    assert cli_sweep(project, store, t2, mode, monkeypatch, "--resume") == 0
    rows, keys = identities(store, f"rel-{mode}")
    done = {r["cell_key"] if "cell_key" in r else r["run_id"] for r in rows_before}
    retrained = [c for c in t2.calls if any(c in str(d) for d in done)]
    assert retrained == [], f"the CLI retrained cells the panel finished: {retrained}"
    assert len(keys) == len(set(keys))
    if mode == "quick":
        assert t2.calls == [] and len(rows) == len(rows_before)


def test_the_panel_writes_no_csv_override_and_keeps_every_committed_scenarios_hash(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO / "notebooks")
    panel = CP.ControlPanel(store=tmp_path / "runs", specs_dir=REPO / "configs" / "scenarios",
                            compares_dir=REPO / "configs" / "compares")
    for choice in panel.choices:
        if choice.scenario is None:
            continue
        panel.select_scenario(choice.name)
        built = panel.build_scenario()
        assert built.overrides == choice.scenario.overrides, choice.name
        if not choice.scenario.search:
            assert built.spec_hash == choice.scenario.spec_hash, choice.name


def test_a_relative_data_path_resolves_from_the_project_root_when_the_working_directory_lacks_it(project, monkeypatch):
    monkeypatch.chdir(project / "notebooks")
    assert L.resolve_data_path("bars.csv") == project / "bars.csv"
    monkeypatch.chdir(project)
    assert L.resolve_data_path("bars.csv") == Path("bars.csv")              # as given when it exists from the cwd
    assert L.resolve_data_path("missing.csv") == Path("missing.csv")        # unchanged when it exists nowhere


# ------------------------------------------------------------------ P1-B / P2-C: the real stored verdicts
@pytest.fixture
def real_compares(tmp_path):
    """A copy of the committed compare specs and stored results (loss_prune_v1 and capacity_v1)."""
    store = tmp_path / "runs"
    shutil.copytree(REPO / "runs" / "compares", store / "compares")
    specs = tmp_path / "compares_specs"
    shutil.copytree(REPO / "configs" / "compares", specs)
    return RunStore(store), specs


def _pair(scenario, a, b):
    return (PC.ConfigScores(f"{scenario} / {a}", scenario, a, "dev", 0, ()),
            PC.ConfigScores(f"{scenario} / {b}", scenario, b, "dev", 0, ()))


@pytest.mark.parametrize("stem,variant", [("loss_prune_v1_ece0", "ece0"), ("loss_prune_v1_ece0_vol0", "ece0_vol0")])
def test_the_stored_result_of_a_real_spec_is_found_under_the_file_stem_and_shown_in_full(real_compares, stem, variant):
    store, specs = real_compares
    a, b = _pair("loss_prune_v1", variant, "control")
    path, flipped = PC.find_compare_spec(a, b, specs)
    assert Path(path).name == f"{stem}.yaml" and not flipped
    out = PC.verdict_html(a, b, store, compares_dir=specs)
    assert "no verdict yet" not in out and "inconclusive" in out
    assert "no pre-registered comparison names this pair" not in out
    doc = json.loads((store.root / "compares" / stem / "result.json").read_text(encoding="utf-8"))
    # P2-C: the non-inferiority result (what D-057 adopted the variants on), the guard-rails, the per-fold and per-pair
    # tables and the stored report.md
    assert "Non-inferiority" in out and "margin 0.005" in out and "pass" in out
    for g in doc["guard_rails"]:
        assert g["metric"] in out
    for row in doc["fold_rows"]:
        assert f"<td>{row['fold']}</td>" in out
    for pair in doc["pairs"]:
        assert pair["a_run_id"] in out and pair["b_run_id"] in out
    report = (store.root / "compares" / stem / "report.md").read_text(encoding="utf-8")
    assert "report.md" in out and "## Guard-rails" in out and report.splitlines()[0][2:] in out
    # the reverse selection order is the same stored result
    out2 = PC.verdict_html(b, a, store, compares_dir=specs)
    assert "reverse order" in out2 and "inconclusive" in out2


def test_no_stored_result_names_the_real_spec_file_in_the_suggested_command(real_compares):
    store, specs = real_compares
    shutil.rmtree(store.root / "compares" / "loss_prune_v1_ece0")
    a, b = _pair("loss_prune_v1", "ece0", "control")
    out = PC.verdict_html(a, b, store, compares_dir=specs)
    assert "no verdict yet" in out
    assert str(Path(specs) / "loss_prune_v1_ece0.yaml") in out
    assert "_vs_control.yaml" not in out
    assert str(store.root / "compares" / "loss_prune_v1_ece0") in out
    assert "no pre-registered comparison names this pair" not in out and "Exploratory" in out


def test_a_stored_result_for_another_spec_content_is_refused_as_stale(real_compares):
    store, specs = real_compares
    path = store.root / "compares" / "loss_prune_v1_ece0" / "result.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**doc, "spec_hash": "0" * 12}), encoding="utf-8")
    a, b = _pair("loss_prune_v1", "ece0", "control")
    assert "stale" in PC.verdict_html(a, b, store, compares_dir=specs)


# ------------------------------------------------------------------ P2-D: n/a at beta = 0 is the report's rule
REAL_ZERO_BETA = "capacity_v1_timing"


def test_the_panel_n_a_rule_is_the_reports_and_covers_the_constant_readout():
    from neural_trade.evaluation import report as R

    assert set(R.SERVED_DELTA_NA_KEYS) <= set(PC.SERVED_NA_METRICS["delta"])
    assert set(R.GAUSS_CONST_NA_KEYS) <= set(PC.SERVED_NA_METRICS["gauss_direction"])
    assert {"recall", "specificity", "f1", "tp", "fp"} <= set(PC.SERVED_NA_METRICS["gauss_direction"])


def test_a_real_zero_beta_cell_shows_n_a_for_every_constant_statistic(tmp_path):
    src = REPO / "runs" / "scenarios" / REAL_ZERO_BETA
    if not any(src.glob("*control__*/eval_report_dev.json")):
        pytest.skip("the real beta-0 cells are not on this machine")
    store_root = tmp_path / "runs"
    dst = store_root / "scenarios" / REAL_ZERO_BETA
    dst.mkdir(parents=True)
    for d in src.iterdir():
        if d.is_dir() and d.name != "specs":
            shutil.copytree(d, dst / d.name, ignore=shutil.ignore_patterns("*.keras", "*.h5", "*.npz", "artifacts"))
    store = RunStore(store_root)
    store.sync(REAL_ZERO_BETA)
    ctrl, lin = PC.aggregate_selection(store, [(REAL_ZERO_BETA, "control"), (REAL_ZERO_BETA, "linear_indicators")])
    assert ctrl.n_cells >= 1
    for h in ("h0", "h1", "h2"):
        for key in ("auc", "mcc", "brier", "ece_pos", "pred_up_rate", "recall", "specificity", "f1", "tp", "fp"):
            assert f"{h}/gauss_direction/{key}" not in ctrl.aggs and f"{h}/gauss_direction/{key}" in ctrl.na, (h, key)
        for key in ("corr", "corr_spearman", "share_pred_up", "ev", "skill_vs_zero"):
            assert f"{h}/delta/{key}" in ctrl.na, (h, key)
        assert f"{h}/delta_raw/corr" in ctrl.aggs and f"{h}/gauss_direction/true_up_rate" in ctrl.aggs
        assert f"{h}/gauss_direction/auc" in lin.aggs                        # beta > 0: measured
    table = PC.metric_table([ctrl, lin])
    assert table.loc["h0/gauss_direction/auc", (ctrl.label, "value")] == PC.NA_TEXT
    assert table.loc["h0/delta/corr", (ctrl.label, "value")] == PC.NA_TEXT
    html_table = PC.table_html([ctrl, lin])
    assert PC.NA_TEXT in html_table
