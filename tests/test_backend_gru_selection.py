import sys
from pathlib import Path

import pytest

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

import model


def test_runtime_gru_overrides_cuda(monkeypatch):
    monkeypatch.setattr(model, "_is_cuda_build", lambda: True)

    overrides = model._runtime_gru_overrides()

    assert overrides.get("use_cudnn") == "auto"
    assert "recurrent_dropout" not in overrides


def test_runtime_gru_overrides_non_cuda(monkeypatch):
    monkeypatch.setattr(model, "_is_cuda_build", lambda: False)

    overrides = model._runtime_gru_overrides()

    assert overrides.get("use_cudnn") is False
    assert overrides.get("recurrent_dropout") == pytest.approx(0.1)


def test_build_runtime_aware_gru_falls_back_when_use_cudnn_unsupported(monkeypatch):
    calls = []

    class DummyLayer:
        pass

    def fake_gru(**kwargs):
        calls.append(dict(kwargs))
        if "use_cudnn" in kwargs:
            raise TypeError("unexpected keyword argument 'use_cudnn'")
        return DummyLayer()

    monkeypatch.setattr(model, "_runtime_gru_overrides", lambda: {"use_cudnn": False, "recurrent_dropout": 0.1})
    monkeypatch.setattr(model.layers, "GRU", fake_gru)

    layer = model._build_runtime_aware_gru(units=64, return_sequences=True)

    assert isinstance(layer, DummyLayer)
    assert len(calls) == 2
    assert "use_cudnn" in calls[0]
    assert "use_cudnn" not in calls[1]
    assert calls[1]["recurrent_dropout"] == pytest.approx(0.1)
