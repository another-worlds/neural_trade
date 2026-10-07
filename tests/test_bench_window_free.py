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

import pytest
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
            assert len(entry["fwd_sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["fwd_sha256"])

    assert out["g_a1"]["census"]["PASS"] is True, out["g_a1"]["census"]["offending_ops"]
    assert out["g_a1"]["precision"]["PASS"] is True
    assert out["g_a1"]["PASS"] is True

    # the two-stage MACD cascade (the lead's review: the single-stage channels above don't exercise it)
    for mode in ("constant", "per_bar"):
        channels = out["g_a1"]["precision"]["channels"][mode]
        for cname in ("macd_line", "macd_signal"):
            assert channels[cname]["PASS"] is True, (mode, cname, channels[cname])
            assert set(channels[cname]["per_setting_rel_max"]) == set(out["g_a1"]["precision"]["macd_settings"])


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


def test_only_the_gpu_device_enables_op_determinism(monkeypatch):
    """CPU, including the smoke kit, must not call enable_op_determinism. GPU calls it once, before any bench."""
    window_free = _load("window_free")
    calls = []

    def stop(*_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(window_free.tf.config.experimental, "enable_op_determinism", lambda: calls.append(1))
    monkeypatch.setattr(window_free.common, "target_scale", stop)

    with pytest.raises(RuntimeError, match="stop"):
        window_free.main(["--device", "cpu", "--smoke", "--out", "unused.json"])
    assert calls == []
    with pytest.raises(RuntimeError, match="stop"):
        window_free.main(["--device", "gpu", "--smoke", "--out", "unused.json"])
    assert calls == [1]


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


def test_today_layer_bitwise_check_passes_because_its_dense_layer_is_seeded():
    """NT-120: two builds of today's layer (LearnableIndicators + the meta Dense) hold the same weights,
    so the bitwise check passes. Before the fix the Dense kernel was a fresh Glorot draw per build."""
    window_free = _load("window_free")
    close = window_free.common.load_close(window_free.CSV)
    scale = window_free.common.target_scale(window_free.CSV)
    result = window_free.check_bitwise(lambda: window_free.today_layer_pass(close, scale, 6))
    assert result["PASS"] is True, result
    assert result["sha256_first"] == result["sha256_second"]


def _stub_sampler(median_s, rng, outlier_frac=0.2, jitter=0.03):
    """Timings around ``median_s`` with ``jitter`` relative noise; ``outlier_frac`` of the repeats are
    outliers spread over 3.4-30.8 ms (the recorded denominator spread)."""
    def sample():
        if rng.random() < outlier_frac:
            return float(rng.uniform(3.4e-3, 30.8e-3))
        return float(median_s * (1 + jitter * rng.standard_normal()))
    return sample


@pytest.mark.parametrize("true_ratio, expect_pass", [(1.05, True), (1.20, False)])
def test_g_a2_ratio_gate_is_stable_under_a_noisy_denominator(true_ratio, expect_pass):
    """NT-120: the same underlying ratio gives the same call in 20 of 20 seeds, although the denominator
    carries the recorded 3.4-30.8 ms outliers."""
    import numpy as np
    common = _load("common")
    calls = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        res = common.stable_ratio_gate(_stub_sampler(15.8e-3 * true_ratio, rng, outlier_frac=0.05),
                                       _stub_sampler(15.8e-3, rng), reps=20, warm=2)
        calls.append(res["PASS"])
        assert res["numerator"]["reps"] == 20 and res["denominator"]["reps"] == 20
        assert res["numerator"]["iqr_s"] >= 0 and res["denominator"]["iqr_s"] >= 0
        assert "median" in res["statistic"] and "IQR" in res["statistic"]
    assert calls == [expect_pass] * 20


def test_g_a2_gate_drops_warm_up_interleaves_and_needs_20_repeats():
    common = _load("common")
    order = []

    def a():
        order.append("A")
        return 1.0

    def b():
        order.append("B")
        return 1.0
    common.stable_ratio_gate(a, b, reps=20, warm=2)
    assert order == ["A", "B"] * 22                      # 2 warm-up pairs, then 20 interleaved pairs
    with pytest.raises(ValueError):
        common.stable_ratio_gate(a, b, reps=19)


def test_smoke_output_carries_the_g_a2_block(tmp_path):
    import json
    window_free = _load("window_free")
    out_path = tmp_path / "smoke.json"
    assert window_free.main(["--smoke", "--out", str(out_path)]) == 0
    g = json.loads(out_path.read_text())["g_a2"]
    assert g["bitwise"]["today_layer"]["PASS"] and g["bitwise"]["a2_layer"]["PASS"]
    for entry in g["ratio"].values():
        assert entry["numerator"]["reps"] >= 20 and entry["denominator"]["reps"] >= 20
        assert "iqr_s" in entry["numerator"] and "iqr_s" in entry["denominator"]
        assert entry["statistic"]
