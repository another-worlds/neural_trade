"""The pre-training lambda-calibration pass must restore the configured lambdas when it fails.

Pre-fix: the pass reset 13 lambdas to 1.0 for sampling and, on any exception,
printed "proceeding with default lambdas" while leaving all 13 at 1.0.
"""
from __future__ import annotations

import numpy as np


def test_calibration_pass_restores_lambdas_on_failure(tf, tiny_config, tmp_path, synthetic_bars, monkeypatch):
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    # Non-default values, so a "reset to 1.0" is distinguishable from a restore.
    expected = {
        "lambda_short": 0.7, "lambda_dir": 0.9, "lambda_extended_trend": 0.4,
        "lambda_crps": 0.6, "lambda_t_perp": 0.3, "lambda_hd": 0.2, "lambda_casimir": 0.15,
    }
    cfg.LAMBDA_SHORT = expected["lambda_short"]
    cfg.LAMBDA_DIR = expected["lambda_dir"]
    cfg.LAMBDA_EXTENDED_TREND = expected["lambda_extended_trend"]
    cfg.LAMBDA_CRPS = expected["lambda_crps"]
    cfg.LAMBDA_T_PERP = expected["lambda_t_perp"]
    cfg.LAMBDA_HD = expected["lambda_hd"]
    cfg.LAMBDA_CASIMIR = expected["lambda_casimir"]

    original = CustomTrainModel.custom_loss
    calls = {"n": 0}

    def explode_once(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the calibration sampler's first call, i.e. after the reset to 1.0
            raise RuntimeError("boom inside the calibration sampler")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CustomTrainModel, "custom_loss", explode_once)
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)

    assert calls["n"] >= 1, "the calibration pass never sampled the loss"
    for name, value in expected.items():
        got = float(getattr(result.model, name))
        assert np.isclose(got, value), f"{name}: expected restored {value}, got {got}"
    assert result.calibration_lambdas is None
