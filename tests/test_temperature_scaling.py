"""Temperature scaling reaches the NLL minimum (NT-124): bounded scalar search, flagged bounds. Pure numpy."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from neural_trade.calibration import CalibrationPipeline
from neural_trade.calibration.temperature_scaling import (
    T_MAX, T_MIN, TemperatureScaler, _fit_temperature_status, _logit, _nll, _sigmoid)

from tests.test_calibration import SHARPEN, _synthetic


class _capture(logging.Handler):
    """Collect the records of one logger directly (the package's loggers do not propagate to caplog)."""

    def __init__(self, name):
        super().__init__(logging.DEBUG)
        self.logger, self.items = logging.getLogger(name), []

    def emit(self, record):
        self.items.append(record)

    def __enter__(self):
        self._level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self._level)


def _block(sd, t_star, n, seed):
    """Logits z ~ N(0, sd); the head reports sigmoid(z); labels are drawn from sigmoid(z / t_star)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, sd, n)
    y = (rng.uniform(size=n) < _sigmoid(z / t_star)).astype(float)
    return _sigmoid(z), y


@pytest.mark.parametrize("sd", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("t_star", [0.33, 0.5, 2.0, 5.0])
def test_fit_reaches_the_nll_minimum_of_a_dense_log_grid(sd, t_star):
    p, y = _block(sd, t_star, 20_000, seed=int(sd * 10 + t_star * 100))
    t, _ = _fit_temperature_status(p, y)
    z = _logit(p)
    grid = np.exp(np.linspace(np.log(1e-2), np.log(1e3), 4001))
    best_grid = min(_nll(z / g, y) for g in grid)
    assert _nll(z / t, y) <= best_grid + 1e-9, (sd, t_star, t)


def test_weak_underconfident_head_is_sharpened_below_half():
    # the old 500-step lr-0.05 descent stopped near T = 1 here (the gradient scales with sd^2 = 0.01)
    p, y = _block(0.1, 0.33, 20_000, seed=3)
    t, status = _fit_temperature_status(p, y)
    assert t < 0.45, t
    assert status == "ok"


def test_fit_is_deterministic_and_bounded():
    p, y = _block(1.0, 2.0, 5_000, seed=1)
    assert _fit_temperature_status(p, y) == _fit_temperature_status(p, y)
    t, _ = _fit_temperature_status(p, y)
    assert T_MIN <= t <= T_MAX


def test_a_fit_at_a_bound_is_flagged_logged_recorded_and_never_called_well_calibrated(tmp_path):
    rng = np.random.default_rng(0)
    n = 4_000
    z = rng.normal(0.0, 1.0, n)
    y = (rng.uniform(size=n) < 0.5).astype(float)   # labels independent of the head: NLL falls as T grows
    p = _sigmoid(z)
    t, status = _fit_temperature_status(p, y)
    assert status == "upper_bound" and t >= T_MAX * 0.99

    sc = TemperatureScaler()
    records = _capture("neural_trade.calibration.temperature_scaling")
    with records:
        sc.fit(p, y, p, y, p, y)
    assert sc.at_bound() == {"h0": "upper_bound", "h1": "upper_bound", "h2": "upper_bound"}
    assert any(r.levelno == logging.WARNING and "upper bound" in r.getMessage() for r in records.items)

    # perfectly separable labels: T falls to the lower bound
    y2 = (z > 0).astype(float)
    t2, status2 = _fit_temperature_status(_sigmoid(z), y2)
    assert status2 == "lower_bound" and t2 <= T_MIN * 1.01

    # the pipeline summary never says "well-calibrated" for a bound fit, and pipeline_meta records it
    rng = np.random.default_rng(5)
    preds, yy, lc, _ = _synthetic(rng, 2_000)
    pipe = CalibrationPipeline().fit_from_arrays(preds, yy, lc, deadband_bps=0.0, conformal_alpha=0.1)
    pipe.temperature_scaler.temperatures["h1"] = 1.0           # would read "well-calibrated (T ~ 1)"
    pipe.temperature_scaler.fit_status["h1"] = "upper_bound"
    records = _capture("neural_trade.calibration.pipeline")
    with records:
        pipe.summary()
    h1_line = [r.getMessage() for r in records.items if r.getMessage().strip().startswith("h1: T =")][0]
    assert "well-calibrated" not in h1_line and "upper bound" in h1_line
    pipe.save(str(tmp_path / "c"))
    meta = json.loads((tmp_path / "c" / "pipeline_meta.json").read_text())
    assert meta["temperature_at_bound"] == {"h1": "upper_bound"}
    loaded = CalibrationPipeline.load(str(tmp_path / "c"))
    assert loaded.temperature_scaler.at_bound() == {"h1": "upper_bound"}


def test_sharpen_fixture_recovers_the_temperature_within_5_percent():
    rng = np.random.default_rng(1)
    preds, y, lc, _ = _synthetic(rng, 40_000)
    pipe = CalibrationPipeline().fit_from_arrays(preds, y, lc, deadband_bps=0.0, conformal_alpha=0.1)
    for h in ("h0", "h1", "h2"):
        t = pipe.temperature_scaler.temperatures[h]
        assert abs(t / SHARPEN - 1.0) < 0.05, (h, t)
        assert pipe.temperature_scaler.fit_status[h] == "ok"


def test_the_deadband_mask_is_respected():
    # labels inside the deadband are noise; the pipeline passes only masked samples, the scaler fits what it gets
    p, y = _block(1.0, 2.0, 6_000, seed=9)
    keep = np.abs(_logit(p)) > 0.5
    t_all, _ = _fit_temperature_status(p, y)
    t_masked, _ = _fit_temperature_status(p[keep], y[keep])
    assert t_all != t_masked


def test_old_temperature_json_without_fit_status_still_loads(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"temperatures": {"h0": 2.0, "h1": 1.0, "h2": 0.5}}))
    sc = TemperatureScaler.load(str(path))
    assert sc.temperatures["h0"] == 2.0 and sc.at_bound() == {}
