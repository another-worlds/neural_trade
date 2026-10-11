"""Neural tower variant of tdvar (SPEC.md "Neural tower"): numpy MLP, QLIKE loss in levels, full-batch Adam, CPU.
log f = b + w.z + MLP(z); nn_direct trained on rate_H; nn_td: g_s on rate_s, then the H head on the bootstrap target
Z = (s rate_s + s g_s(x_(a+s))) / H with g_s frozen (the target network)."""
import numpy as np
import pandas as pd

import tdvar_lib as L

HID, STEPS, LR, L2 = 16, 200, 0.01, 1e-3
EPS = 1e-3


def zfeat(X, mu, sd):
    return (np.log(X + EPS) - mu) / sd


class Net:
    def __init__(self, z, y, y_floor, seed=0):
        rng = np.random.default_rng(seed)
        d = z.shape[1]
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, HID)); self.b1 = np.zeros(HID)
        self.w2 = np.zeros(HID); self.wl = np.zeros(d); self.b = float(np.log(np.mean(y)))
        self.fit(z, np.maximum(y, y_floor))

    def logf(self, z):
        h = np.tanh(z @ self.W1 + self.b1)
        return self.b + z @ self.wl + h @ self.w2

    def fit(self, z, y):
        P = ["W1", "b1", "w2", "wl", "b"]
        m = {k: 0.0 * np.asarray(getattr(self, k)) for k in P}; v = {k: 0.0 * np.asarray(getattr(self, k)) for k in P}
        n = len(y)
        for t in range(1, STEPS + 1):
            h = np.tanh(z @ self.W1 + self.b1)
            lf = np.clip(self.b + z @ self.wl + h @ self.w2, -50, 50)
            g = (1.0 - y * np.exp(-lf)) / n                      # d loss / d logf for loss = mean(x - log x - 1), x = y e^-logf
            gh = np.outer(g, self.w2) * (1 - h * h)
            grads = dict(W1=z.T @ gh + 2 * L2 * self.W1, b1=gh.sum(0), w2=h.T @ g + 2 * L2 * self.w2, wl=z.T @ g, b=np.asarray(g.sum()))
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]; v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                upd = LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
                setattr(self, k, getattr(self, k) - (float(upd) if k == "b" else upd))

    def predict(self, z):
        return np.exp(np.clip(self.logf(z), -50, 50))


def cell_nn(D, t0, t1, W, H, end_limit_pos):
    s = H // 2
    pos0 = int(D.index.searchsorted(t0)); lo = int(D.index.searchsorted(t0 - pd.Timedelta(days=W)))
    A = D.anch
    tr = (A >= lo) & (A + H <= pos0) & D.contiguous(A, H)
    pos1 = int(D.index.searchsorted(t1))
    te = (A >= pos0) & (A < pos1) & (A + H <= end_limit_pos) & D.contiguous(A, H)
    a_tr, a_te = A[tr], A[te]
    if len(a_tr) < 20 or len(a_te) < 3:
        return None
    X, Xte = D.feat(a_tr), D.feat(a_te)
    lx = np.log(X + EPS); mu, sd = lx.mean(0), lx.std(0) + 1e-9
    z, zte = zfeat(X, mu, sd), zfeat(Xte, mu, sd)
    zn = zfeat(D.feat(a_tr + s), mu, sd)
    y_tr, y_te, ys = D.rate(a_tr, H), D.rate(a_te, H), D.rate(a_tr, s)
    med = float(np.median(y_tr)); f_floor, y_floor = L.FLOOR_REL * med, L.Y_FLOOR_REL * med
    net_d = Net(z, y_tr, y_floor)
    net_s = Net(z, ys, y_floor * s / H)
    Z = (s * ys + (H - s) * net_s.predict(zn)) / H
    net_t = Net(z, Z, y_floor)
    rows = []
    for name, net in (("nn_direct", net_d), ("nn_td", net_t)):
        f = np.maximum(net.predict(zte), f_floor)
        rows.append(dict(model=name, qlike=L.qlike(y_te, f, y_floor, f_floor), spearman=L.spearman(f, y_te),
                         mean_f=float(f.mean()), mean_y=float(y_te.mean())))
    info = dict(n_train=int(len(a_tr)), n_test=int(len(a_te)), n_eff_train=float(len(a_tr) * D.stride / H),
                n_eff_test=float(len(a_te) * D.stride / H), win_days_avail=float((pos0 - lo) / 1440.0))
    return rows, info
