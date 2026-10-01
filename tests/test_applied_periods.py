"""neural_trade.evaluation.applied_periods (NT-097 point 5): p5/p50/p95 of the per-window applied
period on real evaluation windows, recovered from a built model's ``meta_adjust`` tensor."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.evaluation.applied_periods import (applied_period_samples, applied_period_stats,
                                                     write_applied_period_report)
from neural_trade.models.registry import Models

OLD = dict(INPUT_SERIES=["close"], INDICATOR_FAMILIES={})  # the pre-NT-047 four families, 18 logits
N, L = 64, 60


def _model(cfg):
    return Models.build(getattr(cfg, "MODEL_NAME", None), cfg)


def _windows(seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, (N, L)), axis=1).astype(np.float32)


def test_meta_adjust_layer_is_named_and_reachable():
    cfg = Config(**OLD)
    model = _model(cfg)
    layer = model.get_layer("meta_adjust")
    assert isinstance(layer, tf.keras.layers.Dense)


def test_meta_adjust_bias_default_true_matches_todays_dense_default():
    cfg = Config(**OLD)
    assert cfg.META_ADJUST_BIAS is True
    model = _model(cfg)
    assert model.get_layer("meta_adjust").use_bias is True


def test_meta_adjust_bias_false_removes_the_bias_weight():
    cfg = Config(**OLD, META_ADJUST_BIAS=False)
    model = _model(cfg)
    layer = model.get_layer("meta_adjust")
    assert layer.use_bias is False
    assert len(layer.get_weights()) == 1  # kernel only, no bias vector


def test_applied_period_samples_returns_18_logits_with_finite_periods():
    cfg = Config(**OLD)
    model = _model(cfg)
    x = _windows()
    samples = applied_period_samples(model, x)
    assert len(samples) == 18
    for name, v in samples.items():
        assert v.shape == (N,), name
        assert np.all(np.isfinite(v)), name


def test_applied_period_stats_p5_p50_p95_ordered():
    cfg = Config(**OLD)
    model = _model(cfg)
    x = _windows(1)
    stats = applied_period_stats(model, x)
    assert len(stats) == 18
    for name, s in stats.items():
        assert set(s) == {"p5", "p50", "p95"}
        assert s["p5"] <= s["p50"] <= s["p95"], (name, s)


def test_applied_period_stats_reflects_bound_applied_switch():
    """With INDICATOR_BOUND_APPLIED on, every window's applied period stays inside the
    [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX] bound (NT-097 point 1), so the p5/p95 report does too."""
    cfg = Config(**OLD, INDICATOR_BOUND_APPLIED=True, MOMENTUM_CLIP_MIN=2.0, MOMENTUM_CLIP_MAX=60.0)
    model = _model(cfg)
    x = _windows(2)
    stats = applied_period_stats(model, x)
    for name, s in stats.items():
        assert s["p5"] >= cfg.MOMENTUM_CLIP_MIN - 1e-4, name
        assert s["p95"] <= cfg.MOMENTUM_CLIP_MAX + 1e-4, name


def test_write_applied_period_report_writes_valid_json(tmp_path):
    cfg = Config(**OLD)
    model = _model(cfg)
    x = _windows(3)
    out = tmp_path / "applied_periods.json"
    stats = write_applied_period_report(model, x, out)
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert set(on_disk) == set(stats)
    for name, s in stats.items():
        for k, v in s.items():
            assert on_disk[name][k] == pytest.approx(v)


# ---------------------------------------------------------------- NT-097 point 5: wired into score_result


@pytest.mark.slow
def test_a_scored_default_config_run_carries_the_applied_period_report_in_its_health_section(tmp_path, synthetic_bars):
    """A real, 1-epoch, default-config run scored through the engine (experiments.scorer.score_result,
    called by the Runner exactly as production does) must carry p5/p50/p95 of every learnable
    parameter's applied period in its health section and its eval_report_*.json on disk."""
    from neural_trade.experiments.runner import Runner
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import RunStore

    bars_csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(bars_csv, index=False)
    spec = {"schema_version": 1, "name": "applied_periods_smoke", "description": "NT-097 point 5",
           "overrides": {"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32},
           "variants": {"default": {}}, "folds": [-1], "seeds": [0],
           "strategy": {"name": "calibrated_quantile", "params": {}}, "backtest": {"random_seeds": 5},
           "run": {"calibrate": False, "save_artifacts": False}}
    store = RunStore(tmp_path / "runs")
    report = Runner(Scenario.from_dict(spec), store).run()
    assert len(report.ran) == 1 and not report.failed

    from pathlib import Path

    run_dir = Path(report.ran[0]["run_dir"])
    doc = json.loads((run_dir / "eval_report_test.json").read_text(encoding="utf-8"))
    applied = doc["health"]["applied_periods"]
    assert applied, "a default-config run must report at least one learnable parameter"
    for name, s in applied.items():
        assert set(s) == {"p5", "p50", "p95"}, name
        assert s["p5"] <= s["p50"] <= s["p95"], (name, s)
        assert all(np.isfinite(v) for v in s.values()), (name, s)
