"""NT-119 / D-051: the Predictor path carries the raw heads to SignalFrame.

``PredictionBatch.to_prediction_frame`` (used by ``cli backtest`` and serving) sets
``meta["delta_raw"]`` like ``PredictionFrame.from_result``; a frame without it is logged, not silent.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from neural_trade.serving.predictor import PredictionBatch, _tail_batch
from neural_trade.strategy.signals import SignalFrame

H = ("h0", "h1", "h2")


def _batch(raw, served, n):
    return PredictionBatch(
        delta={h: served[:, i] for i, h in enumerate(H)},
        direction_prob={h: np.full(n, 0.8) for h in H},
        direction_prob_calibrated={h: np.full(n, 0.8) for h in H},
        sigma={h: np.full(n, 10.0) for h in H},
        variance_scaled={h: np.full(n, 0.01) for h in H},
        gauss_up_prob={h: np.full(n, 0.6) for h in H},
        interval={h: (np.full(n, -1.0), np.full(n, 1.0)) for h in H},
        last_close=np.full(n, 100_000.0), horizon_steps=(10, 15, 20),
        delta_raw=None if raw is None else {h: raw[:, i] for i, h in enumerate(H)})


@pytest.fixture
def records():
    """The strategy.signals logger's records (the package logger may not propagate to caplog)."""
    out = []

    class _H(logging.Handler):
        def emit(self, record):
            out.append(record)

    lg = logging.getLogger("neural_trade.strategy.signals")
    h = _H(level=logging.WARNING)
    lg.addHandler(h)
    yield out
    lg.removeHandler(h)


RAW = np.array([[1.0, 3.0, 9.0], [4.0, 2.0, 8.0], [-2.0, -5.0, -1.0], [1.0, -4.0, 7.0]])


def test_prediction_frame_carries_raw_heads_and_coherence_uses_them():
    batch = _batch(RAW, 0.25 * RAW, 4)
    frame = batch.to_prediction_frame(100.0, 0.0)
    for i, h in enumerate(H):
        np.testing.assert_array_equal(frame.meta["delta_raw"][h], RAW[:, i])
        np.testing.assert_array_equal(frame.delta[h], 0.25 * RAW[:, i])  # served delta untouched
    s = SignalFrame.build(frame, 1.0)
    assert s.coherence_on_raw
    expect = (np.abs(RAW[:, 0]) <= np.abs(RAW[:, 1])) & (np.abs(RAW[:, 1]) <= np.abs(RAW[:, 2]))
    np.testing.assert_array_equal(s.magnitude_coherent, expect)
    np.testing.assert_allclose(s.delta, 0.25 * RAW)  # size / entry stay on the served delta


def test_tail_batch_keeps_raw_heads():
    tail = _tail_batch(_batch(RAW, 0.25 * RAW, 4))
    assert all(len(tail.delta_raw[h]) == 1 for h in H)
    np.testing.assert_array_equal(tail.delta_raw["h1"], RAW[-1:, 1])


def test_missing_raw_heads_warn_and_fall_back(records):
    frame = _batch(None, 0.25 * RAW, 4).to_prediction_frame(100.0, 0.0)
    assert "delta_raw" not in frame.meta
    s = SignalFrame.build(frame, 1.0)
    assert not s.coherence_on_raw
    assert any("raw price heads" in r.getMessage() for r in records)


def test_raw_heads_present_logs_nothing(records):
    frame = _batch(RAW, 0.25 * RAW, 4).to_prediction_frame(100.0, 0.0)
    SignalFrame.build(frame, 1.0)
    assert not records
