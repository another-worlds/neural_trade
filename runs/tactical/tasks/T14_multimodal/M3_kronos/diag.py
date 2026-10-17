"""Diagnostics (not part of the verdict): standalone quality of the Kronos columns."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import anchors as AN, view_cache as VC
from sklearn.metrics import roc_auc_score
for src in ['cut', 'fwd']:
    b = AN.load_base(src)
    for nm in ['M3_kronos', 'M3b_kronos_blend']:
        v = VC.read_view(nm, src); s = pd.Series(b.ok, index=b.ta); ix = v.index[s.reindex(v.index).values]
        pos = pd.Series(np.arange(len(b.ta)), index=b.ta)[ix].values
        lv = b.arr['A_LV0'][pos]; tr = lv + b.arr['A_dA'][pos]; y = b.arr['B_y50'][pos]
        pt = v.loc[ix, 'point_log_rv'].values; col = 'kr_phit50' if nm == 'M3_kronos' else 'kr_phit_logit'
        print(src, nm, 'n', len(ix), 'corr(point,true lnRV) %.3f' % np.corrcoef(pt, tr)[0, 1], 'corr(LV0,true) %.3f' % np.corrcoef(lv, tr)[0, 1],
              'AUC %s vs y50 %.3f' % (col, roc_auc_score(y, v.loc[ix, col])), 'mean(point-true) %.2f sd %.2f' % ((pt - tr).mean(), (pt - tr).std()))
