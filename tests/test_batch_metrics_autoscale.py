from neural_trade import model


def test_compute_loss_y_range_single():
    r = model.compute_loss_y_range([1.0])
    assert r is not None
    y_min, y_max = r
    assert y_min < 1.0 and y_max > 1.0


def test_compute_loss_y_range_span():
    r = model.compute_loss_y_range([0.5, 1.0, 2.0])
    assert r is not None
    y_min, y_max = r
    assert y_min >= 0.0
    assert y_max > 2.0


def test_compute_metrics_y_range_with_keys():
    hist = {'dir_acc': [0.49, 0.51, 0.5], 'f1': [0.48, 0.52]}
    y_min, y_max = model.compute_metrics_y_range(hist, keys=['dir_acc', 'f1'])
    assert 0.45 <= y_min < 0.5
    assert 0.5 < y_max <= 0.55
