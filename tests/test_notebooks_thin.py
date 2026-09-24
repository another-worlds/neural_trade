"""Notebooks are thin (plan B17): no def/class, outputs stripped, and all four execute end to end."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))
REPO = NB_DIR.parent


def _code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb, [("".join(c["source"]), c) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_six_notebooks_exist_and_the_old_ones_are_gone():
    assert [p.name for p in NOTEBOOKS] == ["00_data_and_splits.ipynb", "01_train_and_monitor.ipynb",
                                           "02_backtest.ipynb", "03_signals_and_trades.ipynb",
                                           "04_diagnostics.ipynb", "05_compare_runs.ipynb"]
    for old in ("inference.ipynb", "trade.ipynb", "diagnostics.ipynb", "cfg.ipynb"):
        assert not (REPO / old).exists(), old


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_function_or_class_definitions(path):
    _, cells = _code_cells(path)
    for src, _ in cells:
        tree = ast.parse(src)
        defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                                             ast.Lambda))]
        assert not defs, f"{path.name}: move {type(defs[0]).__name__} into neural_trade"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_parameters_come_first_and_no_saved_output_is_an_error(path):
    """Notebooks are committed with the outputs of their last real run (so they can be read without
    running them); none of those outputs may be an error."""
    nb, cells = _code_cells(path)
    assert "parameters" in cells[0][1].get("metadata", {}).get("tags", [])
    errors = [o for _, c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{path.name} was saved with an error output: {errors[0].get('ename')}"


def _run(path, params, tmp_path):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(path), as_version=4)
    first = next(c for c in nb.cells if c.cell_type == "code")
    first.source += "\n" + "\n".join(f"{k} = {v!r}" for k, v in params.items())
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    return nb


@pytest.mark.slow
@pytest.mark.notebook
def test_all_notebooks_execute(tmp_path, synthetic_bars, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    runs = tmp_path / "runs"
    common = {"CSV_PATH": str(csv), "RUNS_DIR": str(runs)}
    small = {"MAX_SEQUENCE_COUNT": 1500, "BATCH_SIZE": 32}
    _run(NB_DIR / "00_data_and_splits.ipynb",
         {"CSV_PATH": str(csv), "CONFIG_PATH": str(REPO / "configs" / "default.yaml"), "OVERRIDES": small}, tmp_path)
    _run(NB_DIR / "01_train_and_monitor.ipynb",
         {**common, "CONFIG_PATH": str(REPO / "configs" / "default.yaml"), "EPOCHS": 1,
          "CALIBRATE_LOSS_WEIGHTS": False,
          "OVERRIDES": {**small, "CALLBACKS": ["early_stopping", "model_checkpoint"]}}, tmp_path)
    (run_dir,) = [p for p in runs.iterdir() if (p / "artifacts").is_dir()]
    assert (run_dir / "eval_report_test.json").exists()
    _run(NB_DIR / "02_backtest.ipynb", {**common, "RANDOM_SEEDS": 3}, tmp_path)
    _run(NB_DIR / "03_signals_and_trades.ipynb", {**common, "WINDOW": 300}, tmp_path)
    _run(NB_DIR / "04_diagnostics.ipynb", common, tmp_path)
    _run(NB_DIR / "05_compare_runs.ipynb", {"RUNS_GLOB": str(runs / "*"), "ABLATION_DIR": str(tmp_path / "none")},
         tmp_path)


def _saved_training_dashboards():
    """(notebook, cell index, figure JSON) of every saved 'Training dashboard' output."""
    for path in (NB_DIR / "01_train_and_monitor.ipynb", NB_DIR / "04_diagnostics.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            for out in cell.get("outputs", []):
                fig = out.get("data", {}).get("application/vnd.plotly.v1+json")
                if fig and fig["layout"].get("title", {}).get("text", "").startswith("<b>Training dashboard</b>"):
                    yield path.name, i, fig


def test_saved_training_dashboards_mark_each_chance_band_edge_in_its_horizon_colour():
    """Final check, round 2: the saved outputs of 01 cell 7 and 04 cell 4 still showed the old dashboard.
    Its MCC and balanced-accuracy panels shaded three neutral tiers with no horizon edges, and its subtitle
    broke 'dashed = each / horizon's limit' across two lines. In every saved dashboard, each edge of a
    shaded chance range must now be a dashed line in a horizon colour (a one-sided range's 0 excepted)."""
    from neural_trade.visualization import theme as T

    horizon_hex = set(T.HORIZON_COLORS.values())
    found = {}
    for name, cell, fig in _saved_training_dashboards():
        found.setdefault(name, []).append(cell)
        shapes = fig["layout"].get("shapes", [])
        rects = [s for s in shapes if s.get("type") == "rect"]
        dashed = {(s["yref"], round(s["y0"], 6)) for s in shapes
                  if s.get("type") == "line" and s.get("line", {}).get("color") in horizon_hex
                  and s.get("line", {}).get("dash") == "dash" and s["y0"] == s["y1"]}
        edges = {(s["yref"], round(y, 6)) for s in rects for y in (s["y0"], s["y1"]) if y != 0}
        assert edges <= dashed, (name, cell, sorted(edges - dashed))
        lines = fig["layout"]["title"]["text"].replace("</span>", "").split("<br>")
        if rects:   # a run with too small a validation block has no chance range and no such phrase
            assert any("dashed = each horizon's limit, in its colour" in ln for ln in lines), (name, cell, lines[-2:])
    assert set(found) == {"01_train_and_monitor.ipynb", "04_diagnostics.ipynb"}, found
