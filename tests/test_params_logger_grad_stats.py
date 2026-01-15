import pandas as pd
from neural_trade.model import ParamsLogger


class DummyLayer:
    def get_learned_parameters(self):
        return {'ma_period_0': 10.0}


class DummyModel:
    def __init__(self):
        # simulate three batches with small gradients
        self._grad_batch_stats = [
            {'avg_pre': 0.0, 'avg_post': 0.0, 'frac_zero_pre': 1.0, 'frac_zero_post': 1.0},
            {'avg_pre': 1e-8, 'avg_post': 1e-7, 'frac_zero_pre': 0.9, 'frac_zero_post': 0.8},
        ]


def test_params_logger_includes_grad_stats(tmp_path):
    out = tmp_path / 'params_test.csv'
    pl = ParamsLogger(layer=DummyLayer(), out_csv=str(out))
    # Attach a dummy model to the callback
    pl.model = DummyModel()
    # Simulate epoch end
    pl.on_epoch_end(epoch=1, logs={'val_loss': 0.5})
    df = pd.read_csv(str(out))
    assert 'grad_indicator_avg_pre' in df.columns
    assert 'grad_indicator_avg_post' in df.columns
    assert 'grad_indicator_frac_zero_pre' in df.columns
    assert 'grad_indicator_frac_zero_post' in df.columns
