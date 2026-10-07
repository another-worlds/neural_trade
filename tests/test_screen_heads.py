"""Screen mode scores all nine heads (delta, P(up), variance per horizon) in every trial and can keep the
validation predictions (``run.save_predictions``): the keys, the maths on hand-made data, the npz."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
from neural_trade.experiments.screen import (ScreenError, ScreenSpec, _head_metrics_one, build_trials,
                                             run_screen, run_trial)

SAFE_DATA_END = "2025-10-13T04:19:00+00:00"
SAFE_PROTECTED_DAYS = 0.001

DIRECTION_KEYS = {"auc", "brier", "log_loss", "hit_rate", "mean_abs_p_dev", "n", "n_eff"}
DELTA_KEYS = {"corr", "skill_vs_zero", "n", "n_eff"}
VARIANCE_KEYS = {"crps", "crpss", "nll", "coverage90", "width90", "corr_var_err2_spearman", "n", "n_eff"}


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("screen_heads_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def _spec_dict(csv, **run):
    return {"schema_version": 1, "name": "tiny_heads", "description": "test",
            "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2,
                          "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
                          "DATA_END_PROTECTED_DAYS": SAFE_PROTECTED_DAYS},
            "grid": {"axes": {}}, "slices": [SAFE_DATA_END], "seeds": [0],
            "run": {"calibrate": False, "epochs": 1, **run}, "rules": {"finite": True}}


def _check_head_metrics(hm):
    assert set(hm) == {"h0", "h1", "h2"}
    for h in hm.values():
        assert set(h["direction"]) == DIRECTION_KEYS
        assert set(h["delta"]) == DELTA_KEYS
        assert set(h["variance"]) == VARIANCE_KEYS
        for group in h.values():
            for k, v in group.items():
                assert v is None or isinstance(v, (int, float)) and math.isfinite(v), (k, v)
        assert h["variance"]["n"] > 0 and h["variance"]["n_eff"] <= h["variance"]["n"]
        assert h["direction"]["auc"] is not None and h["variance"]["crps"] is not None
        assert h["variance"]["crpss"] is not None


# ------------------------------------------------------------------ the maths on hand-made data
def test_true_variance_beats_a_constant_variance_and_covers_90_percent():
    rng = np.random.default_rng(0)
    n = 20000
    sigma = np.where(rng.random(n) < 0.5, 1.0, 5.0)          # two volatility regimes
    y = rng.normal(0.0, sigma)
    y_train = rng.normal(0.0, sigma)                          # same mixture: the constant reference
    last_close = np.full(n, 100.0)
    prob = np.full(n, 0.5)
    good = _head_metrics_one(y, last_close, np.zeros(n), prob, sigma, y_train, 0.0, 10)
    flat = _head_metrics_one(y, last_close, np.zeros(n), prob, np.full(n, np.std(y_train)), y_train, 0.0, 10)
    assert good["variance"]["crpss"] > 0.05            # knowing the regime beats a constant variance
    assert abs(flat["variance"]["crpss"]) < 0.01       # the constant itself has no skill over itself
    assert good["variance"]["coverage90"] == pytest.approx(0.90, abs=0.01)
    assert good["variance"]["corr_var_err2_spearman"] > 0.1
    assert good["variance"]["nll"] < flat["variance"]["nll"]
    assert good["variance"]["width90"] < flat["variance"]["width90"] * 1.5
    assert good["variance"]["n"] == n and good["variance"]["n_eff"] == n // 10


def test_direction_and_delta_heads_on_a_perfect_and_a_useless_predictor():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.normal(0.0, 1.0, n)
    last_close = np.full(n, 100.0)
    sigma = np.ones(n)
    perfect = _head_metrics_one(y, last_close, y, np.where(y > 0, 0.9, 0.1), sigma, y, 0.0, 15)
    assert perfect["delta"]["corr"] == pytest.approx(1.0)
    assert perfect["delta"]["skill_vs_zero"] == pytest.approx(1.0)
    assert perfect["direction"]["auc"] == pytest.approx(1.0)
    assert perfect["direction"]["hit_rate"] == pytest.approx(1.0)
    assert perfect["direction"]["mean_abs_p_dev"] == pytest.approx(0.4)
    assert perfect["direction"]["brier"] == pytest.approx(0.01)
    assert perfect["direction"]["log_loss"] == pytest.approx(-math.log(0.9))
    useless = _head_metrics_one(y, last_close, np.zeros(n), np.full(n, 0.5), sigma, y, 0.0, 15)
    assert useless["delta"]["skill_vs_zero"] == pytest.approx(0.0)
    assert useless["direction"]["brier"] == pytest.approx(0.25)
    assert useless["direction"]["auc"] == pytest.approx(0.5)


def test_deadband_mask_and_non_finite_inputs_give_none_not_nan():
    n = 50
    y = np.zeros(n)
    last_close = np.full(n, 100.0)
    m = _head_metrics_one(y, last_close, np.full(n, np.nan), np.full(n, 0.5), np.ones(n), None, 0.0, 10)
    assert m["delta"]["n"] == 0 and m["delta"]["corr"] is None
    assert m["direction"]["auc"] is None and m["variance"]["crps"] is None
    json.dumps(m, allow_nan=False)
    db = _head_metrics_one(np.full(n, 0.001), last_close, np.zeros(n), np.full(n, 0.5), np.ones(n), None, 1e4, 10)
    assert db["direction"]["n"] == 0                    # everything inside the deadband
    assert db["variance"]["crpss"] is None             # no training block given


# ------------------------------------------------------------------ the spec option
def test_save_predictions_is_validated_and_keeps_old_spec_hashes(bars_csv):
    base = ScreenSpec.from_dict(_spec_dict(bars_csv))
    assert base.run.save_predictions is False
    assert "save_predictions" not in base.to_dict()["run"]          # existing spec hashes unchanged
    on = ScreenSpec.from_dict(_spec_dict(bars_csv, save_predictions=True))
    assert on.run.save_predictions is True and on.spec_hash != base.spec_hash
    with pytest.raises(ScreenError, match="save_predictions must be true or false"):
        ScreenSpec.from_dict(_spec_dict(bars_csv, save_predictions="yes"))


# ------------------------------------------------------------------ real tiny trials
@pytest.mark.slow
@pytest.mark.parametrize("reuse_graph", [False, True])
def test_a_tiny_trial_records_head_metrics_for_all_horizons_and_saves_predictions(bars_csv, tmp_path, reuse_graph):
    spec = ScreenSpec.from_dict(_spec_dict(bars_csv, save_predictions=True, reuse_graph=reuse_graph))
    report = run_screen(spec, store=tmp_path)
    assert report.ran == 1
    rows = [json.loads(line) for line in
            (tmp_path / "screens" / spec.name / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    _check_head_metrics(row["head_metrics"])
    for h in ("h0", "h1", "h2"):                          # the old key is untouched
        assert set(row["direction_auc"][h]) == {"auc", "n", "n_eff"}
        assert row["head_metrics"][h]["direction"]["auc"] == row["direction_auc"][h]["auc"]
    npz = tmp_path / "screens" / spec.name / "preds" / f"{row['trial_key']}.npz"
    with np.load(npz) as z:
        n = len(z["y"])
        assert z["y"].shape == (n, 3) and z["last_close"].shape == (n,)
        for h in ("h0", "h1", "h2"):
            for prefix in ("delta", "p_up", "var"):
                assert z[f"{prefix}_{h}"].shape == (n,) and z[f"{prefix}_{h}"].dtype == np.float32
        assert z["y"].dtype == np.float32 and list(z["horizon_steps"]) == list(Config().HORIZON_STEPS)
        assert {"pred_scale", "pred_mean", "deadband_bps"} <= set(z.files)
    assert not list((tmp_path / "screens" / spec.name / "preds").glob("*.tmp*"))


@pytest.mark.slow
def test_nothing_is_written_when_save_predictions_is_off(bars_csv, tmp_path):
    spec = ScreenSpec.from_dict(_spec_dict(bars_csv))
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {}, store=tmp_path)
    _check_head_metrics(row["head_metrics"])
    assert not (tmp_path / "screens").exists() or not list((tmp_path / "screens").rglob("*.npz"))
