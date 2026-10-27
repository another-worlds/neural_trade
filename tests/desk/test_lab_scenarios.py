"""DESK-5.2: Lab scenarios (discover parity with the panel, invalid edits write nothing, copies are valid specs)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from neural_trade.desk.config import PROJECT_ROOT
from neural_trade.desk.lab import scenarios as sc

REF = PROJECT_ROOT / "configs" / "scenarios" / "reference.yaml"


def _snapshot(d: Path):
    return sorted((p.name, p.read_bytes()) for p in d.rglob("*")) if d.exists() else None


@pytest.fixture()
def tdir(tmp_path):
    d = tmp_path / "desk"
    d.mkdir()
    return d


def test_discover_parity_with_panel():
    from neural_trade.notebook.panel_data import discover_scenarios

    for rel in ("configs/scenarios", "configs"):
        want = discover_scenarios(PROJECT_ROOT / rel)
        got = sc.discover(rel)
        assert [r["name"] for r in got] == [c.name for c in want]
        assert [r["path"] for r in got] == [str(c.path) for c in want]
        assert [r["problem"] for r in got] == [c.problem for c in want]
        assert [r["trains"] for r in got] == [c.trains for c in want]
    assert "reference_default" in {r["name"] for r in sc.discover()}


def test_discover_lists_a_broken_file_with_its_problem(tmp_path):
    (tmp_path / "bad.yaml").write_text("schema_version: 1\nname: bad name!\n", encoding="utf-8")
    (tmp_path / "list.yaml").write_text("- 1\n", encoding="utf-8")
    rows = {r["name"]: r for r in sc.discover(tmp_path)}
    assert rows["bad"]["problem"] and rows["list"]["problem"]
    assert sc.discover(tmp_path / "missing") == []


def test_discover_all_marks_source(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "desk").mkdir()
    shutil.copy(REF, tmp_path / "a" / "x.yaml")
    shutil.copy(REF, tmp_path / "a" / "desk" / "y.yaml")
    rows = sc.discover_all(("a", "a/desk"), root=tmp_path)
    assert [(r["name"], r["source"]) for r in rows] == [("reference_default", "repo"), ("reference_default", "desk")]


def test_read_returns_text_and_spec():
    r = sc.read(REF)
    assert r["problem"] is None and r["name"] == "reference_default"
    assert r["text"] == REF.read_text(encoding="utf-8") and r["spec"]["folds"] == [-3, -2, -1]


def test_read_broken_spec(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("schema_version: 2\n", encoding="utf-8")
    r = sc.read(p)
    assert r["spec"] is None and "schema_version" in r["problem"]


def test_validate_returns_text_and_never_raises(tmp_path):
    assert sc.validate(REF) is None
    assert sc.validate(tmp_path / "nope.yaml").startswith("ScenarioError")
    spec = yaml.safe_load(REF.read_text(encoding="utf-8"))
    spec["seeds"] = "x"
    assert "seeds" in sc.validate(spec, base_dir=REF.parent)
    spec["seeds"] = [0]
    spec["overrides"] = {"NOT_A_FIELD": 1}
    assert sc.validate(spec, base_dir=REF.parent)


def test_plan_table_counts(tmp_path):
    out = sc.plan_table(REF, tmp_path / "store")
    assert out["name"] == "reference_default" and out["n_cells"] == 9
    assert out["n_pending"] == 9 and out["n_done"] == 0 and len(out["cells"]) == 9
    assert {"cell", "state", "seed", "fold"} <= set(out["cells"][0])


def test_plan_table_reports_a_problem(tmp_path):
    assert "problem" in sc.plan_table(tmp_path / "nope.yaml", tmp_path / "store")


def test_copy_is_a_valid_spec_with_a_new_name(tdir):
    r = sc.copy_with_edits(REF, "my_copy", {"seeds": [0], "strategy.name": "calibrated_quantile"}, tdir)
    assert r["ok"] and Path(r["path"]) == tdir / "my_copy.yaml"
    from neural_trade.experiments.scenario import Scenario

    copy = Scenario.from_yaml(tdir / "my_copy.yaml")
    assert copy.name == "my_copy" and copy.seeds == [0] and copy.folds == [-3, -2, -1]
    assert sc.validate(tdir / "my_copy.yaml") is None
    assert copy.base().LR == Scenario.from_yaml(REF).base().LR       # base_config was rebased, still the same file
    assert [p.name for p in tdir.iterdir()] == ["my_copy.yaml"]


@pytest.mark.parametrize("edits", [{"seeds": "x"}, {"overrides": {"NOT_A_FIELD": 1}}, {"folds": []},
                                   {"name": "other"}, {"bogus_key": 1}, {"run.train": "yes"}])
def test_invalid_edit_returns_text_and_writes_nothing(tdir, edits):
    before = _snapshot(tdir)
    r = sc.copy_with_edits(REF, "bad_copy", edits, tdir)
    assert r["ok"] is False and r["problem"]
    assert _snapshot(tdir) == before == []


def test_invalid_edit_does_not_create_the_target_dir(tmp_path):
    target = tmp_path / "new" / "desk"
    assert sc.copy_with_edits(REF, "c", {"seeds": "x"}, target)["ok"] is False
    assert not target.exists()


def test_refuses_overwrite(tdir):
    assert sc.copy_with_edits(REF, "once", {}, tdir)["ok"]
    before = (tdir / "once.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        sc.copy_with_edits(REF, "once", {"seeds": [5]}, tdir)
    assert (tdir / "once.yaml").read_bytes() == before


@pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "/abs", "C:\\x", "C:x", "..", "", "a" * 49, "bad name", "x.", "x ",
                                  "CON", "con", "NUL.txt", "aux.", "com1", "LPT9", "Com3.x", "COM1 ", "é", ".hidden", "a:b"])
def test_refuses_unsafe_names(tdir, tmp_path, name):
    with pytest.raises(sc.ScenarioRefused):
        sc.copy_with_edits(REF, name, {}, tdir)
    assert list(tdir.iterdir()) == [] and list(tmp_path.glob("*.yaml")) == []


def test_missing_source(tdir):
    with pytest.raises(FileNotFoundError):
        sc.copy_with_edits(tdir / "nope.yaml", "x", {}, tdir)
