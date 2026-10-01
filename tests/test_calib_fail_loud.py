"""NT-111: a failed loss-weight calibration must not silently fall back to the configured
lambdas as if nothing happened.

QA of NT-101 found an error inside ``calibrate_loss_weights`` (a CPU OOM in the backward pass, on
the default model, under load) caught and swallowed with only a ``logger.warning`` — the run then
trained (and could be scored and ranked) with whatever lambdas happened to be in effect, with no
record that calibration never ran. ``CALIB_MODE='gradient'`` now always records the failure
(``calib_failed: true``, ``calib_mode``, ``calib_error``) instead of returning ``None``;
``CALIB_MODE='value'`` does the same only when ``Config.CALIB_FAIL_LOUD`` is set (default False
keeps the pre-NT-111 "restore and return None" byte-for-byte — see test_calibration_pass.py).
Either way ``experiments.scorer.score_result`` refuses to score such a run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


def test_calib_fail_loud_defaults_to_false():
    from neural_trade.core.config import Config

    cfg = Config()
    assert cfg.CALIB_FAIL_LOUD is False
    cfg.CALIB_FAIL_LOUD = True
    cfg.validate()  # no error: a plain bool switch


def _explode_once(original):
    calls = {"n": 0}

    def _fn(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom inside the calibration sampler")
        return original(self, *args, **kwargs)
    return _fn, calls


def test_value_mode_fail_loud_records_the_failure_instead_of_returning_none(
        tiny_config, tmp_path, synthetic_bars, monkeypatch):
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_FAIL_LOUD = True
    cfg.LAMBDA_DIR = 0.77  # distinguishable from a reset-to-1.0

    explode_once, calls = _explode_once(CustomTrainModel.custom_loss)
    monkeypatch.setattr(CustomTrainModel, "custom_loss", explode_once)
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)

    assert calls["n"] >= 1
    assert np.isclose(float(result.model.lambda_dir), 0.77), "lambdas are still restored, not left at 1.0"
    cal = result.calibration_lambdas
    assert cal is not None and cal["calib_failed"] is True
    assert cal["calib_mode"] == "value"
    assert cal["calib_error"]["type"] == "RuntimeError"
    assert "boom inside the calibration sampler" in cal["calib_error"]["message"]
    assert np.isclose(cal["lambda_dir"], 0.77)


def test_value_mode_default_switch_off_stays_silent_none(tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """CALIB_FAIL_LOUD defaults to False: unchanged from before this item (golden-run bit-for-bit)."""
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    assert cfg.CALIB_FAIL_LOUD is False

    explode_once, calls = _explode_once(CustomTrainModel.custom_loss)
    monkeypatch.setattr(CustomTrainModel, "custom_loss", explode_once)
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)

    assert calls["n"] >= 1
    assert result.calibration_lambdas is None


def test_scorer_refuses_to_score_a_run_with_a_failed_calibration():
    from types import SimpleNamespace

    from neural_trade.experiments.scorer import ScoringError, score_result

    fake_result = SimpleNamespace(calibration_lambdas={
        "calib_failed": True, "calib_mode": "gradient",
        "calib_error": {"type": "RuntimeError", "message": "boom"},
    })
    with pytest.raises(ScoringError, match="boom"):
        score_result(fake_result, role="dev")


def test_scorer_runs_normally_when_calibration_lambdas_has_no_failure_flag(monkeypatch):
    """A guard-rail against the new check firing on an ordinary, successful run: patch
    split_arrays to blow up right after the calib_failed check so we can tell the check itself let
    a normal ``calibration_lambdas`` dict (no ``calib_failed`` key, or ``None``) straight through."""
    from types import SimpleNamespace

    import neural_trade.data.processor as processor_mod
    from neural_trade.experiments.scorer import score_result

    sentinel_msg = "reached split_arrays: the calib_failed guard did not fire"

    def _boom(cfg):
        raise RuntimeError(sentinel_msg)

    monkeypatch.setattr(processor_mod, "split_arrays", _boom)
    for calib in (None, {}, {"lambda_dir": 1.0}):
        fake_result = SimpleNamespace(calibration_lambdas=calib, config=object())
        with pytest.raises(RuntimeError, match="reached split_arrays"):
            score_result(fake_result, role="dev")


def test_calib_failure_is_recorded_in_meta_json_and_status_json(tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """End to end, through a real RunContext (as the experiment engine uses): the trained run's
    artifacts/meta.json and status.json both show the calibration failure (acceptance (1))."""
    import neural_trade.training.trainer as trainer_mod
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.training.artifacts import ArtifactBundle
    from neural_trade.training.custom_model import CustomTrainModel

    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(csv)
    cfg.CALIB_MODE = "gradient"  # always loud, no switch needed
    ctx = RunContext.create(cfg, root=tmp_path / "runs")

    explode_once, calls = _explode_once(CustomTrainModel.custom_loss)
    monkeypatch.setattr(CustomTrainModel, "custom_loss", explode_once)
    result = trainer_mod.train_and_evaluate(config=ctx.config, run_context=ctx, epochs=1, force=True,
                                            calibrate=True, fit_calibration=False)
    assert calls["n"] >= 1
    assert result.calibration_lambdas["calib_failed"] is True

    bundle = ArtifactBundle.from_result(result)
    bundle.save(ctx.run_dir / "artifacts")
    meta = json.loads((ctx.run_dir / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    assert meta["calibration_lambdas"]["calib_failed"] is True
    assert meta["calibration_lambdas"]["calib_mode"] == "gradient"

    status = json.loads((ctx.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["calib_failed"] is True
    assert status["calib_mode"] == "gradient"
    assert status["calib_error"]["type"] == "RuntimeError"
