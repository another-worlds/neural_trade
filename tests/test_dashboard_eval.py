import pytest
from neural_trade import model


def test_classify_dir_acc():
    assert model.classify_dir_acc(0.65) == 'good'
    assert model.classify_dir_acc(0.575) == 'stable'
    assert model.classify_dir_acc(0.5) == 'bad'


def test_classify_ece_and_brier():
    assert model.classify_ece(0.03) == 'good'
    assert model.classify_ece(0.08) == 'stable'
    assert model.classify_ece(0.2) == 'bad'
    assert model.classify_brier(0.02) == 'good'


def test_classify_stability_conv_gap_coh():
    assert model.classify_stability_metric(0.9) == 'good'
    assert model.classify_stability_metric(0.6) == 'stable'
    assert model.classify_stability_metric(0.3) == 'bad'

    assert model.classify_convergence_metric(0.002) == 'good'
    assert model.classify_convergence_metric(0.0) == 'stable'
    assert model.classify_convergence_metric(-0.01) == 'bad'

    assert model.classify_gen_gap_metric(0.05) == 'good'
    assert model.classify_gen_gap_metric(0.2) == 'stable'
    assert model.classify_gen_gap_metric(0.4) == 'bad'

    assert model.classify_coherence_metric(0.9) == 'good'
    assert model.classify_coherence_metric(0.6) == 'stable'
    assert model.classify_coherence_metric(0.4) == 'bad'
