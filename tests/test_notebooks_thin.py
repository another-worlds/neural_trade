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
