"""NT-118: calibration keeps a configured zero / damping-0 loss weight at its configured value.

Before: ``rescale_weight`` clipped every measured weight to [CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX],
also a damping-0 one (LAMBDA_HD 0.03 became 0.1, LAMBDA_CASIMIR 25 became 20), and LAMBDA_VOL had
no active gate, so a configured 0 was lifted to 0.1 (the arm NT-099 tested, D-058). The default
path must not change: ``CALIB_VOL_ZERO_TO_FLOOR`` (default True) states that lift explicitly.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from neural_trade.training.lambda_calibration import rescale_weight


class _Cap(logging.Handler):
    """Captures the module logger's records directly (the package logger may not propagate)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def caplog():
    log = logging.getLogger("neural_trade.training.lambda_calibration")
    cap = _Cap()
    old = log.level
    log.addHandler(cap)
    log.setLevel(logging.DEBUG)
    yield cap
    log.removeHandler(cap)
    log.setLevel(old)


def test_rescale_weight_damping_zero_keeps_the_configured_weight_unclamped():
    assert rescale_weight(0.03, 1.0, 0.0, 5.0, 0.1, 20.0) == 0.03
    assert rescale_weight(25.0, 1.0, 0.0, 5.0, 0.1, 20.0) == 25.0
    assert rescale_weight(0.0, 1.0, 0.0, 5.0, 0.1, 20.0) == 0.0


def test_rescale_weight_with_damping_respects_the_clamp_and_warns(caplog):
    if True:
        assert rescale_weight(1.0, 1e-4, 1.0, 5.0, 0.1, 20.0, name="lambda_x") == 20.0
        assert rescale_weight(1.0, 1e4, 1.0, 5.0, 0.1, 20.0, name="lambda_y") == 0.1
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("lambda_x" in m and "20" in m for m in msgs)
    assert any("lambda_y" in m for m in msgs)


def test_rescale_weight_unbinding_clip_is_silent(caplog):
    if True:
        out = rescale_weight(1.0, 5.0, 1.0, 5.0, 0.1, 20.0, name="lambda_z")
    assert np.isclose(out, 1.0)
    assert not caplog.records


def test_rescale_weight_quiet_lowers_the_warning_to_info(caplog):
    if True:
        assert rescale_weight(0.0, 1.0, 1.0, 5.0, 0.1, 20.0, name="lambda_vol", quiet=True) == 0.1
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_vol_floor_switch_default_is_on():
    from neural_trade.core.config import Config

    cfg = Config()
    assert cfg.CALIB_VOL_ZERO_TO_FLOOR is True
    assert cfg.LAMBDA_VOL == 0.0


def _calibrate(cfg, tmp_path, synthetic_bars, monkeypatch, mode):
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = mode
    cfg.validate()
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)
    assert result.calibration_lambdas is not None
    assert not result.calibration_lambdas.get("calib_failed")
    return result.calibration_lambdas, result.model


@pytest.mark.parametrize("mode", ["value", "gradient"])
def test_default_path_still_lifts_vol_zero_to_the_floor(tiny_config, tmp_path, synthetic_bars, monkeypatch, mode):
    """The shipped defaults (LAMBDA_VOL 0, switch on) train with vol at CALIB_LAMBDA_MIN (D-058)."""
    cal, model = _calibrate(tiny_config, tmp_path, synthetic_bars, monkeypatch, mode)
    assert cal["lambda_vol"] == pytest.approx(0.1)
    assert float(model.lambda_vol) == pytest.approx(0.1)
    assert cal["lambda_soft_ece"] == 0.0


@pytest.mark.parametrize("mode", ["value", "gradient"])
def test_vol_zero_stays_zero_when_the_floor_switch_is_off(tiny_config, tmp_path, synthetic_bars, monkeypatch, mode):
    tiny_config.CALIB_VOL_ZERO_TO_FLOOR = False
    cal, model = _calibrate(tiny_config, tmp_path, synthetic_bars, monkeypatch, mode)
    assert cal["lambda_vol"] == 0.0
    assert float(model.lambda_vol) == 0.0
    if mode == "gradient":
        assert "lambda_vol" not in cal["grad_norms_pre"]


@pytest.mark.parametrize("mode", ["value", "gradient"])
def test_damping_zero_weights_come_out_unchanged_and_ext_zero_stays_zero(tiny_config, tmp_path, synthetic_bars,
                                                                        monkeypatch, mode):
    tiny_config.LAMBDA_HD = 0.03
    tiny_config.LAMBDA_T_PERP = 0.02
    tiny_config.LAMBDA_CASIMIR = 25.0
    tiny_config.LAMBDA_IFE = 0.1
    tiny_config.LAMBDA_EXTENDED_TREND = 0.0
    cal, model = _calibrate(tiny_config, tmp_path, synthetic_bars, monkeypatch, mode)
    assert cal["lambda_hd"] == pytest.approx(0.03, rel=1e-6)
    assert cal["lambda_t_perp"] == pytest.approx(0.02, rel=1e-6)
    assert cal["lambda_casimir"] == pytest.approx(25.0, rel=1e-6)  # above CALIB_LAMBDA_MAX: not clipped
    assert cal["lambda_ife"] == pytest.approx(0.1, rel=1e-6)
    assert cal["lambda_extended_trend"] == 0.0
    assert float(model.lambda_extended_trend) == 0.0
    # rescaled (damping > 0) weights still respect the clamp
    for name in ("lambda_short", "lambda_point", "lambda_long", "lambda_dir", "lambda_var", "lambda_crps",
                 "lambda_vol"):
        assert 0.1 - 1e-9 <= cal[name] <= 20.0 + 1e-9, (name, cal[name])
