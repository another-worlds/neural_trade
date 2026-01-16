import pandas as pd
import numpy as np
from neural_trade.model import load_indicator_params_df, plot_indicator_movements


def _make_csv(path):
    rows = [
        {'epoch': 0, 'change_ma_period_0': 0.0, 'change_ma_period_1': 0.0, 'change_rsi_period_0': 0.0},
        {'epoch': 1, 'change_ma_period_0': 5.0, 'change_ma_period_1': 2.0, 'change_rsi_period_0': 1.0},
        {'epoch': 2, 'change_ma_period_0': 4.0, 'change_ma_period_1': 3.0, 'change_rsi_period_0': 2.5},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_and_plot(tmp_path):
    p = tmp_path / 'params_plot.csv'
    _make_csv(str(p))
    df = load_indicator_params_df(csv_path=str(p))
    assert not df.empty
    assert 'ma_period_0' in df.columns
    fig = plot_indicator_movements(df, max_params_per_group=2)
    # fig may be None in headless env, test that function returns a Figure-like object or None
    if fig is not None:
        assert hasattr(fig, 'data')
        assert len(fig.data) >= 1
