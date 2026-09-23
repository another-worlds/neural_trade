"""Reproducibility (plan C6): same seed -> identical run; SEED given as an override takes effect."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _run(tmp_path, synthetic_bars, seed, tag):
    import tensorflow as tf

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

    d = tmp_path / tag
    d.mkdir()
    synthetic_bars.to_csv(d / "bars.csv", index=False)
    cfg = Config().override(EPOCHS=1, BATCH_SIZE=32, MAX_SEQUENCE_COUNT=600, CSV_PATH=str(d / "bars.csv"),
                            SCALER_PATH=str(d / "scaler.joblib"), MODEL_PATH=str(d / "weights.h5"),
                            CALLBACKS=["early_stopping"])
    # Unseeded Keras/TF ops take their op seed from per-process counters: repeating a run in the
    # same process needs a cleared session (the ablation runner uses one process per run).
    tf.keras.backend.clear_session()
    res = train_and_evaluate(config=cfg, config_overrides={"SEED": seed}, epochs=1, force=True, calibrate=False,
                             fit_calibration=False, save_artifacts=False)
    return np.concatenate([res.predictions[k][h] for k in ("delta", "direction_prob", "variance")
                           for h in ("h0", "h1", "h2")])


def test_same_seed_identical_and_seed_override_is_honoured(tf, tmp_path, synthetic_bars, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _run(tmp_path, synthetic_bars, 11, "a")
    b = _run(tmp_path, synthetic_bars, 11, "b")
    c = _run(tmp_path, synthetic_bars, 12, "c")
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c), "SEED override had no effect (seeded before overrides were applied)"


def test_seed_everything_seeds_python_numpy_and_tf(tf):
    import random

    from neural_trade.utils.seeding import seed_everything

    seed_everything(5)
    r1, n1, t1 = random.random(), np.random.rand(), float(tf.random.uniform(()))
    seed_everything(5)
    assert (r1, n1, t1) == (random.random(), np.random.rand(), float(tf.random.uniform(())))
