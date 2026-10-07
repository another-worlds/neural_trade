"""apply_gpu_memory_limit: opt-in cap, never a crash."""
import logging

import pytest

import neural_trade  # noqa: F401  (before tensorflow)
import tensorflow as tf
from neural_trade.utils import gpu


@pytest.fixture
def calls(monkeypatch):
    rec = []
    monkeypatch.setattr(tf.config, "set_logical_device_configuration", lambda *a, **k: rec.append(a))
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind=None: ["gpu0"])
    return rec


@pytest.mark.parametrize("val", [None, "", "0", "-5", "abc", "12.5"])
def test_unset_zero_invalid_do_nothing(monkeypatch, calls, val):
    if val is None:
        monkeypatch.delenv(gpu.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(gpu.ENV_VAR, val)
    assert gpu.apply_gpu_memory_limit() is False
    assert calls == []


def test_positive_value_sets_limit_once(monkeypatch, calls):
    monkeypatch.setenv(gpu.ENV_VAR, "4096")
    assert gpu.apply_gpu_memory_limit() is True
    assert len(calls) == 1
    dev, cfgs = calls[0]
    assert dev == "gpu0" and cfgs[0].memory_limit == 4096


def test_no_gpu_is_not_applied(monkeypatch, calls):
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind=None: [])
    monkeypatch.setenv(gpu.ENV_VAR, "4096")
    assert gpu.apply_gpu_memory_limit() is False
    assert calls == []


def test_runtime_error_warns_not_raises(monkeypatch, calls, caplog):
    def boom(*a, **k):
        raise RuntimeError("already initialized")
    monkeypatch.setattr(tf.config, "set_logical_device_configuration", boom)
    monkeypatch.setenv(gpu.ENV_VAR, "4096")
    gpu.LOG.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=gpu.LOG.name):
            assert gpu.apply_gpu_memory_limit() is False
    finally:
        gpu.LOG.removeHandler(caplog.handler)
    assert any(r.levelno == logging.WARNING and "not applied" in r.message for r in caplog.records)
