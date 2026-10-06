"""NT-122: ``scripts/golden_run.py verify`` must fail when a number turns NaN or an infinity changes.

``run()`` is monkeypatched, so nothing trains and TensorFlow is never imported (asserted). The
verdict is checked through the CLI exit code, the printed per-key and by-prefix lines and the
``golden_equal`` JSON line, which must all agree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "golden_run.py"
NAN, INF = np.nan, np.inf


@pytest.fixture(scope="module")
def golden():
    spec = importlib.util.spec_from_file_location("golden_run_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify(golden, monkeypatch, tmp_path, capsys, recorded, new, *extra):
    rec = tmp_path / "rec.npz"
    np.savez(rec, **{k: np.asarray(v, dtype=np.float64) for k, v in recorded.items()})
    monkeypatch.setattr(golden, "run", lambda repo: {k: np.asarray(v, dtype=np.float64) for k, v in new.items()})
    tf_before = "tensorflow" in sys.modules  # another test in this worker may have imported it already
    code = golden.main(["verify", str(rec), *extra])
    out = capsys.readouterr().out
    verdict = json.loads(out.strip().splitlines()[-1])
    assert tf_before or "tensorflow" not in sys.modules  # verify with run() stubbed imports no TF
    return code, out, verdict


FAILS = {
    "nan_one_side_new": ([1.0, 2.0, 3.0], [1.0, NAN, 3.0]),
    "nan_one_side_recorded": ([1.0, NAN, 3.0], [1.0, 2.0, 3.0]),
    "whole_array_nan": ([1.0, 2.0, 3.0], [NAN, NAN, NAN]),
    "whole_array_nan_both_recorded": ([NAN, NAN], [1.0, 2.0]),
    "inf_to_finite": ([1.0, INF], [1.0, 5.0]),
    "finite_to_inf": ([1.0, 5.0], [1.0, INF]),
    "inf_sign_flip": ([1.0, INF], [1.0, -INF]),
    "inf_to_nan": ([INF], [NAN]),
    "shape_change": ([1.0, 2.0], [1.0, 2.0, 3.0]),
    "finite_change_beyond_tol": ([1.0, 2.0], [1.0, 2.1]),
}


@pytest.mark.parametrize("name", sorted(FAILS))
def test_verify_fails(golden, monkeypatch, tmp_path, capsys, name):
    a, b = FAILS[name]
    code, out, verdict = _verify(golden, monkeypatch, tmp_path, capsys, {"hist/x": a, "pred/y": [1.0]},
                                 {"hist/x": b, "pred/y": [1.0]})
    assert code == 1 and verdict["golden_equal"] is False and verdict["n_fail"] == 1
    assert "FAIL hist/x" in out and "ok   pred/y" in out
    assert "hist=0/1" in out and "pred=1/0" in out


def test_verify_fails_missing_key(golden, monkeypatch, tmp_path, capsys):
    code, out, verdict = _verify(golden, monkeypatch, tmp_path, capsys, {"a/x": [1.0], "b/y": [2.0]}, {"a/x": [1.0]})
    assert code == 1 and verdict == {"golden_equal": False, "n_fail": 0, "n_missing": 1}


PASSES = {
    "identical": ([1.0, 2.0], [1.0, 2.0]),
    "nan_same_positions": ([1.0, NAN, 3.0], [1.0, NAN, 3.0]),
    "all_nan_both": ([NAN, NAN], [NAN, NAN]),
    "inf_equal": ([INF, -INF, 1.0], [INF, -INF, 1.0]),
    "within_atol": ([1.0, 2.0], [1.0 + 5e-7, 2.0]),
    "within_rtol": ([1000.0], [1000.0 + 5e-3]),
    "empty": ([], []),
}


@pytest.mark.parametrize("name", sorted(PASSES))
def test_verify_passes(golden, monkeypatch, tmp_path, capsys, name):
    a, b = PASSES[name]
    code, out, verdict = _verify(golden, monkeypatch, tmp_path, capsys, {"hist/x": a}, {"hist/x": b})
    assert code == 0 and verdict == {"golden_equal": True, "n_fail": 0, "n_missing": 0}
    assert "ok   hist/x" in out and "hist=1/0" in out


def test_verify_passes_extra_key(golden, monkeypatch, tmp_path, capsys):
    code, out, verdict = _verify(golden, monkeypatch, tmp_path, capsys, {"a/x": [1.0]}, {"a/x": [1.0], "b/new": [9.0]})
    assert code == 0 and verdict["golden_equal"] is True and "extra=['b/new']" in out


def test_skip_prefix_ignores_nonfinite(golden, monkeypatch, tmp_path, capsys):
    code, _, verdict = _verify(golden, monkeypatch, tmp_path, capsys, {"hist/x": [1.0], "p/y": [1.0]},
                               {"hist/x": [NAN], "p/y": [1.0]}, "--skip", "hist/")
    assert code == 0 and verdict["golden_equal"] is True


def test_by_prefix_agrees_with_verdict(golden, monkeypatch, tmp_path, capsys):
    rec = {f"k{i}/a": [float(i), 1.0] for i in range(4)}
    new = {k: list(v) for k, v in rec.items()}
    new["k2/a"][1] = NAN
    code, out, verdict = _verify(golden, monkeypatch, tmp_path, capsys, rec, new)
    assert code == 1 and verdict["n_fail"] == 1
    assert "k2=0/1" in out and out.count("FAIL") == 1
