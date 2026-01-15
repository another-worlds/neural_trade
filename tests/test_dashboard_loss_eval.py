import pytest
from neural_trade import model


def test_classify_epoch_loss_improving():
    # previous 0.5 -> current 0.45 (10% improvement) => good
    status, info = model.classify_epoch_loss(0.45, val_history=[0.5, 0.45])
    assert status == 'good'
    assert pytest.approx(info['delta_pct'], rel=1e-6) == pytest.approx((0.5 - 0.45) / 0.5)


def test_classify_epoch_loss_stable():
    # tiny change => stable
    status, info = model.classify_epoch_loss(0.5005, val_history=[0.5, 0.5005])
    assert status == 'stable'


def test_classify_epoch_loss_worse():
    # increase => bad
    status, info = model.classify_epoch_loss(0.55, val_history=[0.5, 0.55])
    assert status == 'bad'


def test_classify_epoch_loss_no_history():
    status, info = model.classify_epoch_loss(0.42, val_history=[])
    assert status == 'stable'
