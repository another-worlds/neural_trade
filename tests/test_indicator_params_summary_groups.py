import pandas as pd
from neural_trade import model


def _make_csv(path):
    rows = [
        {'epoch': 0, 'timestamp': 't0', 'ma_period_0': 10.0, 'change_ma_period_0': 0.0, 'ma_period_1': 20.0, 'change_ma_period_1': 1.0, 'rsi_period_0': 14.0, 'change_rsi_period_0': 3.0, 'mean_param_change_pct': 1.3, 'convergence_score': 0.2},
        {'epoch': 1, 'timestamp': 't1', 'ma_period_0': 10.5, 'change_ma_period_0': 5.0, 'ma_period_1': 20.4, 'change_ma_period_1': 2.0, 'rsi_period_0': 14.1, 'change_rsi_period_0': 0.71, 'mean_param_change_pct': 2.57, 'convergence_score': 0.45},
        {'epoch': 2, 'timestamp': 't2', 'ma_period_0': 11.0, 'change_ma_period_0': 4.76, 'ma_period_1': 20.6, 'change_ma_period_1': 0.98, 'rsi_period_0': 14.5, 'change_rsi_period_0': 2.84, 'mean_param_change_pct': 2.86, 'convergence_score': 0.65},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_group_stats(tmp_path):
    p = tmp_path / 'params_grp.csv'
    _make_csv(str(p))
    summary = model.summarize_indicator_params(csv_path=str(p))
    groups = summary.get('group_stats', {})
    assert 'ma' in groups
    assert 'rsi' in groups
    ma_stats = groups['ma']
    # For ma group, changes are [4.76,0.98] in last row filtered? We take last-row changes per param, so should reflect the last row values
    assert isinstance(ma_stats['avg'], float)
    assert ma_stats['min'] <= ma_stats['avg'] <= ma_stats['max']
    # rsi group should have a numeric avg
    assert isinstance(groups['rsi']['avg'], float)
