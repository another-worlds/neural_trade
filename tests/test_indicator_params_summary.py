import os
import pandas as pd
from neural_trade import model


def _make_csv(path):
    rows = [
        {'epoch': 0, 'timestamp': 't0', 'ma_period_0': 10.0, 'change_ma_period_0': 0.0, 'mean_param_change_pct': 0.0, 'convergence_score': 0.0},
        {'epoch': 1, 'timestamp': 't1', 'ma_period_0': 10.5, 'change_ma_period_0': 5.0, 'mean_param_change_pct': 5.0, 'convergence_score': 0.4},
        {'epoch': 2, 'timestamp': 't2', 'ma_period_0': 11.0, 'change_ma_period_0': 4.76, 'mean_param_change_pct': 4.76, 'convergence_score': 0.6},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_summarize_indicator_params(tmp_path):
    p = tmp_path / 'params.csv'
    _make_csv(str(p))
    summary = model.summarize_indicator_params(csv_path=str(p))
    assert 'convergence_score' in summary
    assert summary['convergence_score'] == 0.6
    assert isinstance(summary['top_changes'], list)
    assert len(summary['top_changes']) >= 1
    # top change entry structure: (param_name, change_pct, current_value)
    name, change_pct, curr = summary['top_changes'][0]
    assert name == 'ma_period_0'
    assert abs(change_pct - 4.76) < 1e-2
    assert curr == 11.0
