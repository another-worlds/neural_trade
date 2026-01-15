from neural_trade import model


def test_metric_limits_text_dir():
    txt = model.metric_limits_text('Val Dir Acc')
    assert '0.60' in txt and '0.55' in txt


def test_metric_limits_text_ece():
    txt = model.metric_limits_text('ECE')
    assert '0.05' in txt and '0.10' in txt


def test_metric_limits_text_epoch_loss():
    txt = model.metric_limits_text('Epoch Loss')
    assert '1.00%' in txt and '0.20%' in txt
