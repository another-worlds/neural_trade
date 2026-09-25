"""Notebook tooling (scripts/notebooks): the committed notebooks are exactly what build.py generates, and
check.py flags each problem it exists for. These tests only read and write small files, and execute nothing."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "scripts" / "notebooks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"notebook_tooling_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build():
    return _load("build")


@pytest.fixture(scope="module")
def check():
    pytest.importorskip("plotly")
    return _load("check")


# --------------------------------------------------------------------------- build.py: anti-drift


def test_committed_notebooks_are_what_build_py_generates(build):
    """Cell types, sources and tags of every committed notebook equal build.py's (outputs are not
    compared). A failure means a notebook was edited by hand: move the edit into
    scripts/notebooks/build.py and rebuild."""
    assert build.drift() == []


def test_build_out_writes_valid_notebooks_that_match_the_committed_sources(build, tmp_path, capsys):
    assert build.main(["--out", str(tmp_path), "00", "05_compare_runs"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00_data_and_splits.ipynb", "05_compare_runs.ipynb"]
    for path in tmp_path.iterdir():
        book = nbformat.read(str(path), as_version=4)
        nbformat.validate(book)
        committed = json.loads((REPO / "notebooks" / path.name).read_text(encoding="utf-8"))
        assert build.cell_sources(book) == build.cell_sources(committed)
        assert b"\r\n" not in path.read_bytes()   # LF, like the repository
        assert [c["id"] for c in book.cells] == [f"cell-{i:02d}" for i in range(len(book.cells))]
    assert build.main(["--check", "--out", str(tmp_path), "00", "05"]) == 0
    assert "notebooks match build.py" in capsys.readouterr().out


def test_drift_names_a_hand_edited_cell_and_a_missing_notebook(build, tmp_path, capsys):
    build.main(["--out", str(tmp_path), "05"])
    path = tmp_path / "05_compare_runs.ipynb"
    book = json.loads(path.read_text(encoding="utf-8"))
    book["cells"][1]["source"] = "RUNS_GLOB = 'edited in Jupyter'"
    path.write_text(json.dumps(book), encoding="utf-8")
    assert build.drift(tmp_path, ["05"]) == ["05_compare_runs: 6 cells saved, 6 generated; first difference at cell 1"]
    assert build.drift(tmp_path, ["00"]) == [f"00_data_and_splits: missing from {tmp_path}"]
    assert build.main(["--check", "--out", str(tmp_path), "05"]) == 1
    assert "DRIFT 05_compare_runs" in capsys.readouterr().out


def test_unknown_or_ambiguous_names_are_refused_before_anything_is_written(build, tmp_path):
    for bad in ("0", "99_nope"):
        with pytest.raises(SystemExit):
            build.main(["--out", str(tmp_path), bad])
    assert not any(tmp_path.iterdir())


# --------------------------------------------------------------------------- check.py


def _synthetic(tmp_path, *, error=False, stderr=False, empty=False, unexecuted=False) -> Path:
    """A one-figure notebook saved as if executed. Each flag adds one problem check.py must flag."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
    if not empty:
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0]), row=1, col=2)
    fig.update_layout(title="<b>Synthetic</b>")
    cell = nbformat.v4.new_code_cell("fig.show()", execution_count=None if unexecuted else 1)
    cell.outputs = [nbformat.v4.new_output("display_data", data={"application/vnd.plotly.v1+json":
                                                                 json.loads(fig.to_json())})]
    if stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text="UserWarning: something\n"))
    if error:
        cell.outputs.append(nbformat.v4.new_output("error", ename="ValueError", evalue="boom", traceback=[]))
    book = nbformat.v4.new_notebook()
    book.cells = [nbformat.v4.new_markdown_cell("# synthetic"), cell]
    path = tmp_path / "synthetic.ipynb"
    nbformat.write(book, str(path))
    return path


def test_check_passes_a_clean_notebook(check, tmp_path, capsys):
    path = _synthetic(tmp_path)
    report = check.check_notebook(path)
    assert report.ok and len(report.figures) == 1 and report.figures[0].title == "Synthetic"
    assert check.main([str(path)]) == 0
    assert "all clean" in capsys.readouterr().out


def test_check_flags_an_error_output_and_an_empty_panel(check, tmp_path, capsys):
    path = _synthetic(tmp_path, error=True, empty=True)
    report = check.check_notebook(path)
    assert report.errors == [(1, "ValueError: boom")]
    assert [f.empty for f in report.empty] == [["y2"]]
    assert not report.ok
    assert check.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "[1] ERROR ValueError: boom" in out and "EMPTY PANEL" in out and "FAIL in synthetic.ipynb" in out


@pytest.mark.parametrize("problem", ["error", "stderr", "empty", "unexecuted"])
def test_check_fails_on_each_problem_alone(check, tmp_path, problem):
    assert check.main([str(_synthetic(tmp_path, **{problem: True}))]) == 1


def test_committed_notebooks_pass_check(check, capsys):
    """The committed notebooks were saved clean: no error, no stderr, no empty panel, no unexecuted cell."""
    assert check.main([]) == 0, capsys.readouterr().out
