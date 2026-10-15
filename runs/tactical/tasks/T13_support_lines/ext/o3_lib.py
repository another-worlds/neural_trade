# T13 COPY of runs/tactical/outlook/O3_synth_volind/o3_lib.py from origin/nt-tactical-outlook at ed870691ab56c2873660b8070d1e4838d4aaea5b; ONLY the sys.path/TAC lines changed (marked 'T13 copy').
"""O3 (SPEC.md): synthetic volatility indicators as learned memory kernels over a bank of EWMAs. Library: the causal EWMA bank, rows and
walk-forward fits (ridge R0 / BANK, the kernel model LK, its shuffled control LKsh). CPU only, numpy (manual gradients, Adam).
Anchor a sees bars < a only (bank value at bar a-1)."""
import os, sys, math
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # T13 copy: ext/ is three levels below runs/tactical
sys.path.insert(0, os.path.join(TAC, "tasks", "T01_vol_forward")); sys.path.insert(0, os.path.join(TAC, "volnet"))
import t01_lib as T  # noqa: E402

HORIZONS = (15, 60, 240)
TAUS = tuple(2 ** i for i in range(12))                  # 1, 2, 4, ..., 2048 minutes
NT = len(TAUS)
CHANNELS = ("r2", "range", "volume")
EPS_CH = np.array([1e-10, 1e-6, 1e-3])                     # ~1e-3 of each channel's typical level (r^2 1e-7, range 1e-3, volume ratio 1)
K_PER = 2
CHMAP = np.repeat(np.arange(3), K_PER)                      # indicator k reads channel CHMAP[k]
K = len(CHMAP)
LOAD_FROM = pd.Timestamp("2019-12-01 00:00")
FIRST = 3 * 2048                                            # warm-up bars per file; never anchors
M_SEL = [str(m) for m in pd.period_range("2021-01", "2024-06", freq="M")]
M_LOOK = [str(m) for m in pd.period_range("2024-07", "2025-07", freq="M")]
M_FWD = list(T.FWD_MONTHS)
TRAIN_END = T.CUTOFF + pd.Timedelta(minutes=1)
S3_NAMES = ["rv15", "rv60", "rv240", "rv1440", "tod_sin", "tod_cos", "vol15r", "vol60r", "vol240r", "rng15", "rng60", "rv5",
            "lrv_h2", "lrv_h3", "lrv_h4", "lrv_disp"]


# ------------------------------------------------------------------ the bank
def channels(A):
    """[n, 3] positive scale-free channels at every bar t (bar t and earlier only): squared 1-minute log return; (high-low)/close;
    exp(ln(volume+1) - its trailing 1,440-bar mean), a positive volume ratio near 1."""
    c, h, l, v = A[:, 3], A[:, 1], A[:, 2], A[:, 4]
    lr = np.concatenate([[0.0], np.diff(np.log(c))])
    lv = np.log(v + 1.0); cs = np.concatenate([[0.0], np.cumsum(lv)]); n = len(lv); t = np.arange(n)
    lo = np.maximum(t - 1439, 0); m = (cs[t + 1] - cs[lo]) / (t + 1 - lo)
    return np.stack([lr ** 2, (h - l) / c, np.exp(lv - m)], 1)


def ewma(x, tau):
    """y[t] = a x[t] + (1-a) y[t-1], a = 1/tau (tau = 1 is x itself); recursive filter in float64, started at x[0]."""
    al = 1.0 / tau
    if tau == 1:
        return x.astype(np.float64).copy()
    return lfilter([al], [1.0, -(1.0 - al)], x.astype(np.float64), zi=[(1.0 - al) * float(x[0])])[0]


def bank_at(A, idx):
    """[len(idx), 3, 12] EWMA values at bars idx (inclusive) of the whole-series causal recursion."""
    X = channels(A); idx = np.asarray(idx); out = np.empty((len(idx), 3, NT))
    for ci in range(3):
        for j, tau in enumerate(TAUS):
            out[:, ci, j] = ewma(X[:, ci], tau)[idx]
    return out


# ------------------------------------------------------------------ rows
def build_features(s, first=FIRST):
    """dict for one bar frame: anchor time, S3, the bank at a-1, per-H realised RV and ok masks (zero-RV windows dropped)."""
    B = T.build(s, first); D = B["D"]; a = D.anch; cs = D.cs2
    S3 = B["F"]["S3"].astype(np.float64)
    prev_ok = np.all([(cs[a - 60 * k] - cs[a - 60 * (k + 1)]) > 0 for k in range(1, 5)] + [(cs[a] - cs[a - 60]) > 0], 0)
    okf = prev_ok & np.all(D.har[:, :4] > -20, 1) & np.isfinite(S3).all(1)
    out = {"ta": D.ta.values.astype("datetime64[ns]").astype(np.int64), "S3": S3, "okf": okf,
           "bank": bank_at(D.A, a - 1), "anch": a, "A": D.A, "index": s.index}
    for H in HORIZONS:
        rv = cs[a + H] - cs[a]
        out[f"rv{H}"] = rv; out[f"ok{H}"] = okf & (rv > 0)
    out["ta"] = out["ta"].view("datetime64[ns]")
    return out


def train_rows(ta, ok, H, m0, days=365):
    lo = np.datetime64(m0 - pd.Timedelta(days=days), "ns"); hi = np.datetime64(m0 - pd.Timedelta(minutes=H), "ns")
    return np.where((ta >= lo) & (ta <= hi) & ok)[0]


def test_rows(ta, ok, m0, m1):
    return np.where((ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")) & ok)[0]


def split_fit_hold(tr, H, stride=7, frac=0.15):
    k = int(len(tr) * (1 - frac)); gap = int(math.ceil(H / stride))
    return tr[:k], tr[k + gap:]


# ------------------------------------------------------------------ ridge and QLIKE
def ridge_fit(X, y, alpha=1.0):
    mu = X.mean(0); sd = X.std(0) + 1e-12; Xs = (X - mu) / sd; ym = float(y.mean())
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "ym": ym}


def ridge_pred(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ym"]


def qlike_c(v_hold, mu_hold):
    """scale c with s = c exp(mu): c = mean(v / exp(mu)) on the hold rows (SPEC: v / (c e^mu) - ln(.) - 1)."""
    return float(np.mean(v_hold / np.exp(mu_hold)))


def qlike(v, mu, c):
    q = v / (c * np.exp(mu)); return float(np.mean(q - np.log(q) - 1))


def log_bank(bank):
    """[n, 36] ln(EWMA + eps_channel), the BANK features."""
    return np.log(bank + EPS_CH[None, :, None]).reshape(len(bank), -1)


# ------------------------------------------------------------------ the kernel model
def softmax(a):
    e = np.exp(a - a.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def indicators(Bk, a, eps=None):
    """I_k = ln(sum_j softmax(a_k)_j * bank_k,j + eps_k). Bk [n, K, 12] (bank gathered per indicator), a [K, 12] -> [n, K]."""
    eps = EPS_CH[CHMAP] if eps is None else eps
    return np.log(np.einsum("nkj,kj->nk", Bk, softmax(a)) + eps)


def gather(bank):
    """[n, 3, 12] -> [n, K, 12] (indicator k reads channel CHMAP[k])."""
    return bank[:, CHMAP, :]


def _forward(p, S3s, Bk, zm, zs):
    P = softmax(p["a"]); M = np.einsum("nkj,kj->nk", Bk, P); I = np.log(M + EPS_CH[CHMAP]); Z = (I - zm) / zs
    pred = S3s @ p["w"] + Z @ p["b"] + p["c"]
    return pred, (P, M, Z)


def _grads(p, S3s, Bk, y, zs, pred, cache):
    P, M, Z = cache; n = len(y); g = 2.0 * (pred - y) / n
    dP = np.einsum("n,k,nkj->kj", g * 1.0, p["b"] / zs, Bk / (M + EPS_CH[CHMAP])[:, :, None])
    da = P * (dP - (P * dP).sum(1, keepdims=True))
    return {"a": da, "b": Z.T @ g, "w": S3s.T @ g, "c": float(g.sum())}


def train_lk(S3s_fit, Bk_fit, y_fit, S3s_ho, Bk_ho, y_ho, w0, c0, seed, max_epochs=30, patience=3, lr=1e-2, bs=4096, a_noise=0.1):
    """Adam on MSE; hold early stopping, best epoch kept. a starts at a small seeded perturbation of 0 (uniform kernel; exactly 0 would keep the two
    indicators of a channel identical forever), b at 0, the S3 readout at the ridge solution. Returns (params, zm, zs, hold_mse, epochs)."""
    rng = np.random.default_rng(seed)
    I0 = np.log(Bk_fit.mean(2) + EPS_CH[CHMAP]); zm = I0.mean(0); zs = I0.std(0) + 1e-12      # fixed scaling from the uniform kernel
    p = {"a": rng.normal(0, a_noise, (K, NT)), "b": np.zeros(K), "w": w0.copy(), "c": float(c0)}
    mom = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}; var = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}
    best = (np.inf, None, 0); bad = 0; t = 0; n = len(y_fit)
    for ep in range(1, max_epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b_ = perm[i:i + bs]; pred, cache = _forward(p, S3s_fit[b_], Bk_fit[b_], zm, zs)
            gr = _grads(p, S3s_fit[b_], Bk_fit[b_], y_fit[b_], zs, pred, cache); t += 1
            for k in p:
                mom[k] = 0.9 * mom[k] + 0.1 * gr[k]; var[k] = 0.999 * var[k] + 0.001 * np.asarray(gr[k]) ** 2
                p[k] = p[k] - lr * (mom[k] / (1 - 0.9 ** t)) / (np.sqrt(var[k] / (1 - 0.999 ** t)) + 1e-8)
        ph, _ = _forward(p, S3s_ho, Bk_ho, zm, zs); hl = float(np.mean((ph - y_ho) ** 2))
        if hl < best[0] - 1e-12:
            best = (hl, {k: np.array(v, float).copy() if np.ndim(v) else float(v) for k, v in p.items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1], zm, zs, best[0], best[2]


def lk_predict(p, zm, zs, S3s, Bk):
    return _forward(p, S3s, Bk, zm, zs)[0]


def fit_month(F, H, m0, seeds=(0, 1), shuffle_seed=0, lk_kwargs=None):
    """All fitted models for a test starting at m0 (or the frozen fit when m0 = TRAIN_END). Returns a state for score_state."""
    tr = train_rows(F["ta"], F[f"ok{H}"], H, m0); fit, hold = split_fit_hold(tr, H)
    y = np.log(F[f"rv{H}"]); v = F[f"rv{H}"]
    X3 = F["S3"]
    Lb = {"tr": log_bank(F["bank"][tr])}                            # BANK features of the training rows, in tr order
    pos = {r: i for i, r in enumerate(tr)}; fi = np.array([pos[r] for r in fit]); hi = np.array([pos[r] for r in hold])
    S3t = X3[tr]; XBt = np.concatenate([S3t, Lb["tr"]], 1); yt = y[tr]; vt = v[tr]
    st = {"H": H, "m": {}, "c": {}, "n_tr": int(len(tr))}
    for nm, Xt in (("R0", S3t), ("BANK", XBt)):
        mf = ridge_fit(Xt[fi], yt[fi]); mfull = ridge_fit(Xt, yt)
        st["m"][nm + "f"] = mf; st["m"][nm] = mfull
        c = qlike_c(vt[hi], ridge_pred(mf, Xt[hi])); st["c"][nm] = c; st["c"][nm + "f"] = c
    # LK (and LKsh: the bank rows permuted across the training rows)
    mfs = ridge_fit(S3t[fi], yt[fi]); mu3, sd3 = mfs["mu"], mfs["sd"]
    S3s = (S3t - mu3) / sd3; Bk_all = gather(F["bank"][tr])
    rng = np.random.default_rng(shuffle_seed + 17 * H); perm = rng.permutation(len(tr))
    for nm, Bk in (("LK", Bk_all), ("LKsh", Bk_all[perm])):
        best = None
        for sd in seeds:
            out = train_lk(S3s[fi], Bk[fi], yt[fi], S3s[hi], Bk[hi], yt[hi], mfs["w"], mfs["ym"], seed=sd, **(lk_kwargs or {}))
            if best is None or out[3] < best[3]:
                best = out + (sd,)
        p, zm, zs, hl, ep, sd = best
        st["m"][nm] = {"p": p, "zm": zm, "zs": zs, "mu3": mu3, "sd3": sd3, "hold_mse": hl, "epochs": ep, "seed": sd}
        st["c"][nm] = qlike_c(vt[hi], lk_predict(p, zm, zs, S3s[hi], Bk[hi]))
    return st


def score_state(st, F, te, span, month, seed=0):
    H = st["H"]; v = F[f"rv{H}"][te]; S3 = F["S3"][te]; bank = F["bank"][te]
    rng = np.random.default_rng(seed + 1000 + H); permt = rng.permutation(len(te))
    preds = {}
    XB = np.concatenate([S3, log_bank(bank)], 1)
    for nm, X in (("R0", S3), ("BANK", XB)):
        preds[nm] = ridge_pred(st["m"][nm], X); preds[nm + "f"] = ridge_pred(st["m"][nm + "f"], X)
    for nm, Bk in (("LK", gather(bank)), ("LKsh", gather(bank)[permt])):
        m = st["m"][nm]; preds[nm] = lk_predict(m["p"], m["zm"], m["zs"], (S3 - m["mu3"]) / m["sd3"], Bk)
    rows = []
    for nm, mu in preds.items():
        rows.append({"span": span, "month": month, "H": H, "model": nm, "n_te": int(len(te)),
                     "spearman": float(spearmanr(mu, v)[0]) if np.std(mu) > 0 else None, "qlike": qlike(v, mu, st["c"][nm])})
    return rows


def kernel_record(st):
    """the learned kernels of one LK fit: softmax weights over the 12 taus, readout b_k, scaling, epochs."""
    m = st["m"]["LK"]
    return {"P": softmax(m["p"]["a"]).tolist(), "b": m["p"]["b"].tolist(), "zm": m["zm"].tolist(), "zs": m["zs"].tolist(),
            "epochs": int(m["epochs"]), "seed": int(m["seed"]), "hold_mse": float(m["hold_mse"])}
