"""NT-059: the window-free benchmark kit (scripts/bench/window_free.py) runs on tiny sizes in a few
seconds and its checks are wired correctly - G-A1's census half actually flags a banned op, and the
kit's own ``--smoke`` run (real, tiny data) passes G-A1 end to end. Nothing under src/ is exercised
except today's LearnableIndicators layer, imported read-only for comparison (window-free plan README
"Stages and dependencies", stage 1)."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import tensorflow as tf  # noqa: F401  (so conftest skips/marks this file correctly without TF)

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "scripts" / "bench"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_kit_runs_and_gates_pass(tmp_path):
    window_free = _load("window_free")
    out_path = tmp_path / "smoke.json"
    t0 = time.perf_counter()
    rc = window_free.main(["--smoke", "--out", str(out_path)])
    seconds = time.perf_counter() - t0

    assert rc == 0
    assert seconds < 60, f"the smoke run took {seconds:.1f}s, expected a few seconds"
    assert out_path.exists()

    import json
    out = json.loads(out_path.read_text())
    assert out["env"]["device"] == "cpu"
    assert "tf_version" in out["env"]
    assert "op_determinism_enabled" in out["env"]
    assert "tf32_execution_enabled" in out["env"]

    for group in ("kernel_v1", "assembly_d6b", "a2_layer", "today_layer"):
        assert out[group], f"{group} produced no entries"
        for name, entry in out[group].items():
            assert entry["cpu_fwdbwd"]["reps"] >= 2, name
            assert "median_s" in entry["cpu_fwdbwd"]

    assert out["g_a1"]["census"]["PASS"] is True, out["g_a1"]["census"]["offending_ops"]
    assert out["g_a1"]["precision"]["PASS"] is True
    assert out["g_a1"]["PASS"] is True


def test_check_census_flags_a_banned_op():
    """The census gate must actually fail when a benchmarked variant used a banned op - it is not a
    check that always returns PASS."""
    window_free = _load("window_free")
    bad = {"x": {"raise_on_gpu": ["UnsortedSegmentSum"], "host_round_trip_on_gpu": [], "matmul_like_ops": []}}
    result = window_free.check_census(bad)
    assert result["PASS"] is False
    assert "UnsortedSegmentSum" in result["offending_ops"]

    bad_matmul = {"x": {"raise_on_gpu": [], "host_round_trip_on_gpu": [], "matmul_like_ops": ["Einsum"]}}
    assert window_free.check_census(bad_matmul)["PASS"] is False

    clean = {"x": {"raise_on_gpu": [], "host_round_trip_on_gpu": [], "matmul_like_ops": []},
             "y": {"raise_on_gpu": [], "host_round_trip_on_gpu": [], "matmul_like_ops": []}}
    assert window_free.check_census(clean) == {"PASS": True, "offending_ops": []}


def test_main_exits_nonzero_and_reports_the_failed_check(tmp_path, monkeypatch):
    """If G-A1 fails, main() must exit non-zero and say which check failed (acceptance criterion 2)."""
    window_free = _load("window_free")
    monkeypatch.setattr(window_free, "check_census",
                        lambda a2_results: {"PASS": False, "offending_ops": ["Einsum"]})
    out_path = tmp_path / "failing.json"
    rc = window_free.main(["--smoke", "--out", str(out_path)])
    assert rc == 1


def test_a2_layer_census_has_no_matmul_like_or_determinism_raising_ops():
    """A direct, minimal census of the whole A2 layer (forward and backward): G-A1's census half,
    exercised without going through the CLI."""
    window_free = _load("window_free")
    close = window_free.common.load_close(window_free.CSV)
    scale = window_free.common.target_scale(window_free.CSV)
    fwd, fb = window_free.a2_layer_pass(600, close, scale, B=8, min_history=0)
    census = window_free.common.graph_census(fb)
    assert census["matmul_like_ops"] == []
    assert census["raise_on_gpu"] == []
    assert census["host_round_trip_on_gpu"] == []
