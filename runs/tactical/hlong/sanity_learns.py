"""Sanity: a net can leave its linear start when the label depends on a pattern that only the sequence holds (planted: the sum of
channel 0 over bars 60-100 of a 168-bar window). Prints AUC of start vs trained and the best epoch. usage: python sanity_learns.py"""
import numpy as np
import hwf
hwf.setup_device(cpu=True)
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0); n, L = 9000, 168
S = rng.normal(size=(n, L, 7)).astype(np.float32); V = rng.normal(size=(n, 20)).astype(np.float32)
z = S[:, 60:100, 0].sum(1) / np.sqrt(40); y = (z + rng.normal(size=n) > 0).astype(int)
tr, te = np.arange(7000), np.arange(7200, n)
for arch in ("patch", "tcn"):
    zt, z0, info = hwf.fit_net(arch, L, V[tr], S[tr], y[tr], V[te], S[te], 0)
    print(arch, "start AUC %.3f  trained AUC %.3f" % (roc_auc_score(y[te], z0), roc_auc_score(y[te], zt)), info, flush=True)
