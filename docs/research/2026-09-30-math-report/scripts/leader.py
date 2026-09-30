import glob, json, os
import pandas as pd

base = 'D:/neural_trade/runs/scenarios/long_360d_stab/'
for d in sorted(glob.glob(base + '2026*')):
    t = pd.read_csv(d + '/training_log.csv')
    st = json.load(open(d + '/status.json'))
    print(os.path.basename(d), 'weights_epoch', st.get('weights_epoch'), 'weights_val_loss', st.get('weights_val_loss'),
          'sec_per_step', round(st.get('sec_per_step', 0), 4))
    print('  epoch', list(t['epoch']))
    print('  grad_global_norm', [round(x, 2) for x in t['grad_global_norm']])
    print('  nonfinite', list(t['nonfinite_grad_steps']))
    print('  loss', [round(x, 3) for x in t['loss']])
    print('  val_loss', [round(x, 3) for x in t['val_loss']])
    print('  lr', list(t['lr_used']))

lead = base + '20260930T094257Z-dce15ed-e3669618-default__f-2__s0'
t = pd.read_csv(lead + '/training_log.csv')
meta = json.load(open(lead + '/artifacts/meta.json'))
L = meta['lambda_values_final']
print('\nlambdas', L)


def weighted(row, pre=''):
    g = lambda k: float(row[pre + k])
    w = {
        'point (log-cosh, x lambda_short/point/long)': g('point_loss'),
        'extended trend (x lambda_ext, x outer)': L['lambda_trend_outer'] * g('trend_loss'),
        'direction BCE': L['lambda_dir_outer'] * L['lambda_dir'] * g('dir_loss'),
        'Gaussian NLL': L['lambda_nll_outer'] * L['lambda_var'] * g('nll_loss'),
        'CRPS': L['lambda_crps'] * g('crps_loss'),
        'soft ECE': L['lambda_soft_ece'] * g('soft_ece_loss'),
        'vol (0.1 x lambda_vol x |std diff|)': 0.1 * g('vol_loss'),
        'inter_reg (0.1 x LAMBDA_INTER x layer L2)': 0.1 * g('inter_reg'),
        'T-perp': g('t_perp_loss'),
        'Casimir': g('casimir_loss'),
        'vacuum bandwidth': g('vac_loss'),
        'HD': g('hd_loss'),
        'IFE': g('ife_loss'),
        'vacuum overflow': g('vac_overflow_loss'),
        'pnl': g('pnl_val'),
    }
    tot = g('loss')
    w['coherence (residual = total - sum)'] = tot - sum(w.values())
    return w, tot


for pre, label in (('val_', 'VALIDATION (exact)'), ('', 'TRAINING (loss exact; terms sampled every 10 steps)')):
    for ep in (0, 3):
        row = t[t['epoch'] == ep].iloc[0]
        w, tot = weighted(row, pre)
        print(f'\n{label} epoch index {ep}: total {tot:.4f}')
        for k, v in w.items():
            print(f'  {k:48s} {v:9.4f}  {100*v/tot:6.2f}%')
    # unweighted
for ep in (0, 3):
    row = t[t['epoch'] == ep].iloc[0]
    print('\nraw val per-horizon, epoch', ep, {k: round(float(row['val_' + k]), 4) for k in
          ['dir_loss_h0', 'dir_loss_h1', 'dir_loss_h2', 'nll_h0', 'nll_h1', 'nll_h2', 'crps_h0', 'crps_h1', 'crps_h2',
           'soft_ece_h0', 'soft_ece_h1', 'soft_ece_h2', 'point_h0', 'point_h1', 'point_h2', 'vol_loss', 'pit_ks_h0', 'pit_ks_h1', 'pit_ks_h2']})
