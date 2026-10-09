"""Triple barrier + meta-labelling (owner approved 2026-10-09, option 1): judge the regression by the TRADE, not by the sign
of a fixed-horizon return. H27: the regression hits 61% of its top-10% bets but earns no more than a random side, because it
wins small moves and loses big ones.
Per anchor (every STRIDE bars; entry at the close of the window's last bar): sigma = std of the last 60 one-bar log returns;
take-profit / stop at +-M * sigma * sqrt(T) (symmetric, in the trade's direction); time limit T bars. Outcome of a trade on
side s: +barrier if the TP is touched first (high/low of the following bars), -barrier if the stop is (both in one bar: the
stop, conservative), else s * return at T. Costs 0 (D-044).
  primary  a logistic regression on tb7 (or rich) predicts P(up at T); side = sign(P - 0.5); the top-q confidence trades
  meta     a second model (logistic, then HistGradientBoosting) predicts P(trade wins) from the features + |P-0.5| + sigma;
           trades whose meta probability exceeds a threshold fitted on the last 25% of the training period (never the test)
Walk-forward: 8 folds over the chosen span; each trained on everything before it minus a 2T gap. Reported per fold and as the
mean with a 95% t-interval over folds: trades per day, hit rate, mean P&L in bps and in units of sigma, and the random-side
null (same trades, random sides; its 95th percentile).
usage: python barrier.py [--bars 1min|1h|1d] [--T 15] [--M 1.0] [--set tb7|rich] [--start 2020-01-01] [--q 0.1]"""
import argparse, math, os, sys, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import SETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bars", default="1min"); ap.add_argument("--T", type=int, default=15); ap.add_argument("--M", type=float, default=1.0)
ap.add_argument("--set", default="tb7"); ap.add_argument("--start", default="2020-01-01"); ap.add_argument("--q", type=float, default=0.1)
ap.add_argument("--stride", type=int, default=0)
a = ap.parse_args()
t0 = time.time(); L = 60; T = a.T; TQ = 2.36  # t(0.975, 7)
src = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
src.index = pd.to_datetime(src["timestamp"]); src = src.drop(columns="timestamp").sort_index()
src = src[src.index >= a.start]
if a.bars != "1min":
    src = src.resample(a.bars).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
A = src.values.astype(np.float64); n_bars = len(A)
stride = a.stride or (5 if a.bars == "1min" else 1)
anch = np.arange(L, n_bars - T, stride)                     # window = bars [i-L, i), entry at close[i-1]
c0 = A[anch - 1, 3]
lr = np.diff(np.log(A[:, 3])); sig = np.array([lr[i - L:i - 1].std() for i in anch]) if len(anch) < 50000 else \
    swv(lr, L - 1)[anch - L].std(1)
sig = np.maximum(sig, 1e-6)
barrier = a.M * sig * math.sqrt(T)                           # as a fraction of the entry price
Hf = swv(A[:, 1], T)[anch]; Lf = swv(A[:, 2], T)[anch]; cT = A[anch + T - 1, 3]   # bars i .. i+T-1
up_hit = Hf >= (c0 * (1 + barrier))[:, None]; dn_hit = Lf <= (c0 * (1 - barrier))[:, None]
first = lambda m: np.where(m.any(1), m.argmax(1), T + 1)
t_up, t_dn = first(up_hit), first(dn_hit); r_T = cT / c0 - 1


def pnl(side, idx):
    """P&L fraction of trades on `side` (+1 long, -1 short) at anchors `idx` (a slice or index array)."""
    u, d, b, r = t_up[idx], t_dn[idx], barrier[idx], r_T[idx]
    tp = np.where(side > 0, u, d); sl = np.where(side > 0, d, u)
    return np.where(sl <= tp, np.where(sl <= T, -b, side * r), np.where(tp <= T, b, side * r))


# features in chunks (windows built on the fly)
fs = SETS[a.set]; F = []
for k in range(0, len(anch), 50000):
    idx = anch[k:k + 50000]
    W = np.stack([A[i - L:i] for i in idx]).astype(np.float32)
    F.append(np.nan_to_num(fs(W)))
F = np.concatenate(F); y = (r_T > 0).astype(int)
days = (src.index[anch[-1]] - src.index[anch[0]]).days or 1
print(f"{a.bars} bars {n_bars}, anchors {len(anch)}, T {T}, M {a.M}, set {a.set}, mean barrier {np.mean(barrier) * 1e4:.1f} bps, "
      f"TP-first {np.mean(t_up < t_dn):.3f} SL-first {np.mean(t_dn < t_up):.3f} timeout {np.mean((t_up > T) & (t_dn > T)):.3f}; "
      f"prep {time.time() - t0:.0f}s", flush=True)

edges = np.linspace(len(anch) * 0.3, len(anch), 9).astype(int); gap = max(1, 2 * T // stride)
rng = np.random.default_rng(0); R = {k: [] for k in ("p_hit", "p_bps", "p_sig", "p_null95", "p_tpd", "m_hit", "m_bps", "m_sig", "m_null95",
                                                     "m_tpd", "g_hit", "g_bps", "g_null95", "g_tpd", "all_hit", "all_bps")}
for f in range(8):
    tr, te = slice(0, edges[f] - gap), slice(edges[f], edges[f + 1])
    sc = StandardScaler().fit(F[tr]); Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
    prim = LogisticRegression(C=0.1, max_iter=1000).fit(Xtr, y[tr])
    ptr, pte = prim.predict_proba(Xtr)[:, 1], prim.predict_proba(Xte)[:, 1]
    side_tr, side_te = np.where(ptr > 0.5, 1, -1), np.where(pte > 0.5, 1, -1)
    pl_tr, pl_te = pnl(side_tr, tr), pnl(side_te, te)
    te_idx = np.arange(len(anch))[te]
    bar_te = barrier[te]; ndays = max(1, (src.index[anch[te][-1]] - src.index[anch[te][0]]).days)

    def record(tag, k):
        side, p = side_te[k], pl_te[k]
        null = [np.mean(pnl(np.where(rng.random(k.sum()) < 0.5, 1, -1), te_idx[k])) for _ in range(200)]  # random sides, same trades
        R[f"{tag}_hit"].append(float(np.mean(p > 0))); R[f"{tag}_bps"].append(float(np.mean(p) * 1e4))
        R[f"{tag}_null95"].append(float(np.quantile(null, 0.95) * 1e4)); R[f"{tag}_tpd"].append(float(k.sum() / ndays * stride))
        if f"{tag}_sig" in R:
            R[f"{tag}_sig"].append(float(np.mean(p / bar_te[k])))

    conf_tr, conf_te = np.abs(ptr - 0.5), np.abs(pte - 0.5)
    k_p = conf_te >= np.quantile(conf_tr, 1 - a.q)                       # threshold from TRAIN confidences
    record("p", k_p)
    # meta: P(trade wins) from features + confidence + sigma; threshold on the last 25% of train (inner)
    Mtr = np.column_stack([Xtr, conf_tr, np.log(sig[tr])]); Mte = np.column_stack([Xte, conf_te, np.log(sig[te])])
    win_tr = (pl_tr > 0).astype(int); cut = int(len(Mtr) * 0.75)
    for tag, mk in (("m", lambda: LogisticRegression(C=0.1, max_iter=1000)),
                    ("g", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                                 min_samples_leaf=500, random_state=0))):
        inner = mk().fit(Mtr[:cut], win_tr[:cut]); thr = np.quantile(inner.predict_proba(Mtr[cut:])[:, 1], 1 - a.q)
        meta = mk().fit(Mtr, win_tr); record(tag, meta.predict_proba(Mte)[:, 1] >= thr)
    R["all_hit"].append(float(np.mean(pl_te > 0))); R["all_bps"].append(float(np.mean(pl_te) * 1e4))
    print(f"  fold {f}: primary top-{a.q:.0%} hit {R['p_hit'][-1]:.3f} {R['p_bps'][-1]:+.2f} bps (null95 {R['p_null95'][-1]:+.2f}) | "
          f"meta-logit hit {R['m_hit'][-1]:.3f} {R['m_bps'][-1]:+.2f} (null95 {R['m_null95'][-1]:+.2f}) | "
          f"meta-gbdt hit {R['g_hit'][-1]:.3f} {R['g_bps'][-1]:+.2f} (null95 {R['g_null95'][-1]:+.2f}) | all trades {R['all_bps'][-1]:+.2f}", flush=True)
ci = lambda v: f"{np.mean(v):+.3f} [{np.mean(v) - TQ * np.std(v, ddof=1) / math.sqrt(len(v)):+.3f},{np.mean(v) + TQ * np.std(v, ddof=1) / math.sqrt(len(v)):+.3f}]"
print(f"SUMMARY {a.bars} T={T} M={a.M} {a.set} q={a.q} ({time.time() - t0:.0f}s)")
for tag, name in (("p", "primary top-q"), ("m", "meta logistic"), ("g", "meta boosting")):
    print(f"  {name:14s} hit {ci(R[tag + '_hit'])} | bps {ci(R[tag + '_bps'])} | null95 {np.mean(R[tag + '_null95']):+.2f} | "
          f"trades/day {np.mean(R[tag + '_tpd']):.1f}" + (f" | P&L/sigma-barrier {ci(R[tag + '_sig'])}" if tag + "_sig" in R else ""))
print(f"  all primary trades: hit {ci(R['all_hit'])} bps {ci(R['all_bps'])}")
