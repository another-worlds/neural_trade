import json, glob, math, collections
import numpy as np

rows = []
for f in sorted(glob.glob('D:/neural_trade/runs/screens/l1_*/results.shard-*.jsonl')):
    for line in open(f, encoding='utf-8'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print('trials', len(rows))

def q(a, ps=(0, 25, 50, 75, 90, 100)):
    a = np.array([x for x in a if x is not None and math.isfinite(x)])
    if len(a) == 0:
        return 'n/a'
    return ' '.join(f'p{p}={np.percentile(a, p):.3g}' for p in ps) + f' (n={len(a)})'

by = collections.defaultdict(list)
for r in rows:
    by[r['screen']].append(r)
by['ALL'] = rows
for s, rs in by.items():
    h = [r['health'] for r in rs]
    npass = sum(r['passed'] for r in rs)
    print(f'\n=== {s}: n={len(rs)} passed={npass} ({npass/len(rs):.1%})')
    print(' finite all:', sum(x['finite'] for x in h), '/', len(h))
    print(' nonfinite_grad_steps>0:', sum((x['nonfinite_grad_steps'] or 0) > 0 for x in h), 'sum', sum(x['nonfinite_grad_steps'] or 0 for x in h))
    print(' clipped_share', q([x['clipped_share'] for x in h]))
    print(' gnorm_max', q([x['grad_global_norm_max'] for x in h]))
    print(' gnorm_mean', q([x['grad_global_norm_mean'] for x in h]))
    print(' train_loss_drop', q([x['train_loss_drop'] for x in h]))
    print(' max_term_share', q([x['max_term_share'] for x in h]))
    reasons = collections.Counter()
    for r in rs:
        for re_ in r['reasons']:
            reasons[re_.split(' ')[0]] += 1
    print(' reasons', dict(reasons))
    # dominant term
    dom = collections.Counter()
    for x in h:
        ts = x.get('loss_term_shares') or {}
        if ts:
            dom[max(ts, key=ts.get)] += 1
    print(' dominant term', dict(dom))
    # mean share per term
    keys = sorted({k for x in h for k in (x.get('loss_term_shares') or {})})
    ms = {k: np.median([x['loss_term_shares'].get(k, 0) for x in h if x.get('loss_term_shares')]) for k in keys}
    print(' median shares', {k: round(v, 4) for k, v in ms.items() if v > 1e-4})

# drivers: block A by grad clip norm, and by LR
print('\n--- Block A by GRAD_CLIP_NORM')
A = by['l1_A_hyper']
for c in (5.0, 20.0, 100.0):
    rs = [r for r in A if r['config_diff'].get('GRAD_CLIP_NORM') == c]
    print(c, 'n', len(rs), 'pass', sum(r['passed'] for r in rs), 'clipped_share', q([r['health']['clipped_share'] for r in rs], (50,)),
          'gmax', q([r['health']['grad_global_norm_max'] for r in rs], (50, 100)))
print('--- Block A by LR bin')
for lo, hi in ((1e-4, 3e-4), (3e-4, 1e-3), (1e-3, 3e-3), (3e-3, 1.01e-2)):
    rs = [r for r in A if lo <= r['config_diff'].get('LR', 1e-3) < hi]
    print(f'LR [{lo},{hi})', 'n', len(rs), 'pass', sum(r['passed'] for r in rs),
          'nonfinite trials', sum(r['health']['nonfinite_grad_steps'] > 0 for r in rs),
          'gmax med', q([r['health']['grad_global_norm_max'] for r in rs], (50,)),
          'drop', q([r['health']['train_loss_drop'] for r in rs], (50,)))
print('--- Block A by INDICATOR_LR_MULT bin')
for lo, hi in ((1, 3), (3, 8), (8, 21)):
    rs = [r for r in A if lo <= r['config_diff'].get('INDICATOR_LR_MULT', 5) < hi]
    print(f'ILM [{lo},{hi})', 'n', len(rs), 'pass', sum(r['passed'] for r in rs))

# correlation of log(knob) with log(gnorm_mean) and clipped_share for B and C
for s in ('l1_B_loss_weights', 'l1_C_physics'):
    rs = by[s]
    knobs = sorted({k for r in rs for k in r['config_diff'] if isinstance(r['config_diff'][k], (int, float))})
    print(f'\n--- {s}: Spearman(knob, gnorm_mean / clipped_share / pass)')
    from scipy.stats import spearmanr
    for k in knobs:
        xs = [r['config_diff'].get(k) for r in rs]
        ok = [i for i, x in enumerate(xs) if x is not None]
        if len(ok) < 10:
            continue
        x = [xs[i] for i in ok]
        g = [rs[i]['health']['grad_global_norm_mean'] or np.nan for i in ok]
        c = [rs[i]['health']['clipped_share'] if rs[i]['health']['clipped_share'] is not None else np.nan for i in ok]
        p = [float(rs[i]['passed']) for i in ok]
        print(f' {k:24s} n={len(ok)} rho_gmean={spearmanr(x, g, nan_policy="omit")[0]:+.2f} '
              f'rho_clip={spearmanr(x, c, nan_policy="omit")[0]:+.2f} rho_pass={spearmanr(x, p)[0]:+.2f}')
    fails = [r for r in rs if not r['passed']]
    print(' failing trials:', len(fails))

# C ablations
print('\n--- Block C ablations (grid)')
C = by['l1_C_physics']
for ab in ([], ['LAMBDA_T_PERP'], ['LAMBDA_CASIMIR'], ['LAMBDA_HD'], ['LAMBDA_IFE'], ['LAMBDA_VAC_OVERFLOW']):
    rs = [r for r in C if r['config_diff'].get('ABLATE_LAMBDAS') == ab and r['source'] == 'grid']
    print(ab, 'n', len(rs), 'pass', sum(r['passed'] for r in rs), 'gmean', q([r['health']['grad_global_norm_mean'] for r in rs], (50,)))

print('\n--- Block D')
D = by['l1_D_loss_choice']
for dl in ('bce', 'focal_dice'):
    for lp in (0.0, 0.25, 0.5, 1.0):
        rs = [r for r in D if r['config_diff'].get('DIRECTION_LOSS') == dl and r['config_diff'].get('LAMBDA_PNL') == lp]
        print(dl, lp, 'n', len(rs), 'pass', sum(r['passed'] for r in rs),
              'clip', q([r['health']['clipped_share'] for r in rs], (50,)),
              'gmean', q([r['health']['grad_global_norm_mean'] for r in rs], (50,)),
              'reasons', collections.Counter(x.split(' ')[0] for r in rs for x in r['reasons']))
print('\n--- Block E')
for r in by['l1_E_maths']:
    print(r['config_diff'], r['passed'], r['reasons'], round(r['health']['grad_global_norm_mean'], 2), r['health']['clipped_share'])

# which sample knobs of block B shift pass: list failing configs summary
print('\nfailing reasons examples B:')
for r in by['l1_B_loss_weights']:
    if not r['passed']:
        print(r['reasons'][:2], {k: round(v, 3) for k, v in r['config_diff'].items() if isinstance(v, float)})
        break
