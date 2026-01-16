from neural_trade import model


def test_y_range_defaults_empty():
    # empty history -> default
    y_min, y_max = model.compute_metrics_y_range({})
    assert y_min == -0.05 and y_max == 1.05


def test_y_range_centered_small_oscillation():
    hist = {'val_dir_acc_avg': [0.495, 0.502, 0.498], 'dir_acc_avg': [0.497, 0.499]}
    y_min, y_max = model.compute_metrics_y_range(hist)
    # Should be centered near 0.5 and tighter than full [0,1]
    assert 0.45 <= y_min < 0.5
    assert 0.5 < y_max <= 0.55


def test_y_range_larger_spread():
    hist = {'val_dir_acc_avg': [0.2, 0.6, 0.55], 'bal_acc_avg': [0.3, 0.4]}
    y_min, y_max = model.compute_metrics_y_range(hist)
    assert y_min >= 0.0 and y_max <= 1.0
    assert y_max - y_min > 0.3
