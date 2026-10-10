"""JEPA embedding and HAR-RV as meta-model features on the minute triple barrier (H33 setup), tactical, CPU only.
Setup (H33, barrier.py): window 60 bars, entry at the close of the window's last bar, T 15 bars, barrier +-M*sigma*sqrt(T) with M 1.0
and sigma the std of the 59 returns inside the window, stop first when both are touched in one bar, costs 0, anchors every 5 bars.
Primary: logistic regression (C 0.1) on tb7, side = sign(P-0.5). Meta model: predicts "this trade wins" from features; the trades whose
meta probability is in the top q = 10% (threshold fitted on the last train block) are taken.
Differences to H33 (as hourly.py, H38): the meta model is fitted on OUT-OF-FOLD primary predictions (train split in 4 blocks; block k's
primary comes from a model fitted on the earlier blocks minus a gap), the threshold on block 4.
Evaluation span: entries 2020-01-01 .. labels ending before 2024-07-01 (2024-07-01 onward is the spent held-out period and is never
read). 8 walk-forward folds (expanding train, 2T gap), the first starting at 30% of the span. Choices are made on these folds only.
Meta feature sets: a = base (tb7 standardised, |P-0.5|, log sigma), b = a + HAR-RV log vol, c = a + JEPA embedding (32),
d = a + both. HAR-RV: OLS (fitted on the fold's train anchors only) of log mean squared 1-bar return over the next T bars on the log
mean squared return over the last 15, 60, 240, 1440 bars; the feature is its prediction. Both meta models are reported:
m = logistic (C 0.1), g = HistGradientBoosting (200 iters, as hourly.py).
trades/day (tpd) counts the sampled anchors (one anchor every 5 bars); H33 multiplied by the stride (x5, overlapping trades).
usage: python tradefilter.py --emb NAME=ckpt_dir[,NAME2=dir2] [--out results.jsonl]   (CPU)"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2"); JEPA = os.path.join(HERE, "..")
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
START, HOLD = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-07-01")
L, WARM = 60, 2000
HAR_LAGS = (15, 60, 240, 1440)
TQ8 = 2.36                      # t(0.975, 7)
SCRATCH = os.environ.get("JEPA3_SCRATCH", "D:/nt/nt_scratch/jepa3")


# ------------------------------------------------------------------ data and causal features
def load_bars():
    """bars with WARM bars before START (for the HAR lags) up to the last bar before HOLD (never later)."""
    d = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    d = d.sort_values("timestamp"); i0 = int(np.searchsorted(d.timestamp.values, np.datetime64(START)))
    i1 = int(np.searchsorted(d.timestamp.values, np.datetime64(HOLD)))
    d = d.iloc[i0 - WARM:i1]
    return d.timestamp.values, d[["open", "high", "low", "close", "volume"]].values.astype(np.float64)


def make_anchors(n, T, stride):
    """anchor i: window bars [i-L, i), entry at close[i-1], future bars i .. i+T-1 (all inside the array)."""
    return np.arange(WARM + L, n - T, stride)


def causal_features(A, anch):
    """features known at the entry bar e = anch-1 (they read bars <= e only): sigma and the HAR inputs (log trailing mean squared
    returns over HAR_LAGS bars). Returns (sig [N], logv [N, 4])."""
    lr = np.diff(np.log(A[:, 3]))                                   # lr[j] = return from bar j to bar j+1
    sig = np.maximum(swv(lr, L - 1)[anch - L].std(1), 1e-6)         # returns lr[i-L .. i-2]: bars i-L .. i-1
    cs = np.concatenate([[0.0], np.cumsum(lr * lr)])
    e = anch - 1
    logv = np.stack([np.log((cs[e] - cs[e - k]) / k + 1e-12) for k in HAR_LAGS], 1)   # sums lr[e-k .. e-1]
    return sig, logv


def har_target(A, anch, T):
    """the LABEL of the HAR regression: log mean squared 1-bar return over the next T bars (reads the future; train rows only)."""
    lr = np.diff(np.log(A[:, 3])); cs = np.concatenate([[0.0], np.cumsum(lr * lr)]); e = anch - 1
    return np.log((cs[e + T] - cs[e]) / T + 1e-12)


def window_features(A, anch, fs, chunk=50000):
    out = []
    for k in range(0, len(anch), chunk):
        W = np.stack([A[i - L:i] for i in anch[k:k + chunk]]).astype(np.float32)
        out.append(np.nan_to_num(fs(W)))
    return np.concatenate(out)


def barrier_outcomes(A, anch, sig, T, M):
    """triple-barrier P&L tables as barrier.py: returns (barrier [N], t_up, t_dn, r_T)."""
    c0 = A[anch - 1, 3]; barrier = M * sig * math.sqrt(T)
    Hf, Lf = swv(A[:, 1], T)[anch], swv(A[:, 2], T)[anch]; cT = A[anch + T - 1, 3]
    first = lambda m: np.where(m.any(1), m.argmax(1), T + 1)
    return barrier, first(Hf >= (c0 * (1 + barrier))[:, None]), first(Lf <= (c0 * (1 - barrier))[:, None]), cT / c0 - 1


def make_pnl(T, barrier, t_up, t_dn, r_T):
    def pnl(side, idx):
        u, d, b, r = t_up[idx], t_dn[idx], barrier[idx], r_T[idx]
        tp = np.where(side > 0, u, d); sl = np.where(side > 0, d, u)
        return np.where(sl <= tp, np.where(sl <= T, -b, side * r), np.where(tp <= T, b, side * r))
    return pnl


# ------------------------------------------------------------------ embeddings
def load_encoder(ckpt):
    sys.path[:0] = [V2, JEPA]
    from channels import N_CH, Standardiser, CTX
    import model2 as M2
    meta = json.load(open(ckpt + "/meta.json"))
    std = Standardiser(np.array(meta["std"]["mean"]), np.array(meta["std"]["std"]))
    enc = M2.Encoder2(N_CH); enc(np.zeros((2, CTX, N_CH), np.float32)); enc.load_weights(ckpt + "/enc.h5")
    return enc, std


def embed_anchors(enc, std, A, anch, bs=4096):
    """JEPA embedding of the window ending at the entry bar (bars [i-L, i)); a function of those 60 bars only."""
    sys.path[:0] = [JEPA]
    from channels import context_channels
    out = []
    for s in range(0, len(anch), bs):
        W = np.stack([A[i - L:i] for i in anch[s:s + bs]])
        ch, _, _ = context_channels(W); out.append(enc(std(ch), training=False).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ walk-forward
def fold_splits(n, stride, T, nf=8):
    """expanding train / test position ranges; the train labels end before the test entries (gap 2T bars)."""
    edges = np.linspace(n * 0.3, n, nf + 1).astype(int); gap = math.ceil(2 * T / stride)
    return [(np.arange(0, edges[f] - gap), np.arange(edges[f], edges[f + 1])) for f in range(nf)]


def fit_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=1000)


def fit_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                          l2_regularization=1.0, random_state=0)


def run_fold(D, tr, te, sets, q, rng, gap):
    """D: dict with F (tb7), sig, logv, har_y, y, pnl, days (test span in days); sets: {name: embedding array or None / 'har' flags}."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(D["F"][tr]); X = sc.transform(D["F"]); y = D["y"]
    blocks = np.array_split(tr, 4); oof_i, oof_p = [], []
    for k in (1, 2, 3):
        fi = np.concatenate(blocks[:k]); fi = fi[fi < blocks[k][0] - gap]
        oof_i.append(blocks[k]); oof_p.append(fit_lr().fit(X[fi], y[fi]).predict_proba(X[blocks[k]])[:, 1])
    p_te = fit_lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    # HAR-RV: OLS on the train anchors only, then the prediction at every anchor
    Z = np.column_stack([np.ones(len(D["logv"])), D["logv"]]); beta = np.linalg.lstsq(Z[tr], D["har_y"][tr], rcond=None)[0]
    har = Z @ beta; har_z = ((har - har[tr].mean()) / (har[tr].std() + 1e-12))[:, None]
    emb_z = {}
    for name, E in D["emb"].items():
        s = StandardScaler().fit(E[tr]); emb_z[name] = s.transform(E)
    pnl, side = D["pnl"], np.where(p_te > 0.5, 1, -1); pl = pnl(side, te)
    i23, p23 = np.concatenate(oof_i[:2]), np.concatenate(oof_p[:2]); i4, p4 = oof_i[2], oof_p[2]
    win23 = (pnl(np.where(p23 > 0.5, 1, -1), i23) > 0).astype(int)
    ndays = max(1, D["days"](te))

    def metrics(k):
        if k.sum() < 10:
            return None
        null = [np.mean(pnl(np.where(rng.random(k.sum()) < 0.5, 1, -1), te[k])) for _ in range(200)]
        return {"n": int(k.sum()), "tpd": float(k.sum() / ndays), "hit": float(np.mean(pl[k] > 0)),
                "bps": float(np.mean(pl[k]) * 1e4), "null95": float(np.quantile(null, 0.95) * 1e4)}

    def meta_x(idx, p, extra):
        return np.hstack([X[idx], np.abs(p - 0.5)[:, None], np.log(D["sig"][idx])[:, None]] + [e[idx] for e in extra])

    out = {"primary": metrics(np.abs(p_te - 0.5) >= np.quantile(np.abs(p4 - 0.5), 1 - q)), "all": metrics(np.ones(len(te), bool))}
    for sname, (use_har, emb_name) in sets.items():
        extra = ([har_z] if use_har else []) + ([emb_z[emb_name]] if emb_name else [])
        for mname, mk in (("m", fit_lr), ("g", fit_hgb)):
            meta = mk().fit(meta_x(i23, p23, extra), win23)
            thr = np.quantile(meta.predict_proba(meta_x(i4, p4, extra))[:, 1], 1 - q)
            out[f"{mname}:{sname}"] = metrics(meta.predict_proba(meta_x(te, p_te, extra))[:, 1] >= thr)
    return out


def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = TQ8 * v.std(ddof=1) / math.sqrt(len(v))
    return [round(float(m), 4), round(float(m - h), 4), round(float(m + h), 4)]


def summarise(folds):
    keys = [k for k in folds[0] if all(f.get(k) for f in folds)]
    S = {}
    for k in keys:
        S[k] = {m: ci([f[k][m] for f in folds]) for m in ("tpd", "hit", "bps", "null95")}
        S[k]["excess_bps"] = ci([f[k]["bps"] - f[k]["null95"] for f in folds])
        if ":" in k and not k.endswith(":a"):
            base = k.split(":")[0] + ":a"
            S[k]["paired_vs_a"] = {m: ci([f[k][m] - f[base][m] for f in folds]) for m in ("bps", "hit", "tpd")}
            S[k]["paired_vs_a"]["excess_bps"] = ci([(f[k]["bps"] - f[k]["null95"]) - (f[base]["bps"] - f[base]["null95"]) for f in folds])
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emb", required=True, help="NAME=ckpt_dir[,NAME=dir]")
    ap.add_argument("--T", type=int, default=15); ap.add_argument("--M", type=float, default=1.0); ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--out", default=HERE + "/results.jsonl"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); t0 = time.time(); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    sys.path.insert(0, os.path.join(HERE, "..", "..", "lab"))
    import lab
    ts, A = load_bars(); anch = make_anchors(len(A), a.T, a.stride); T = a.T
    sig, logv = causal_features(A, anch); har_y = har_target(A, anch, T)
    barrier, t_up, t_dn, r_T = barrier_outcomes(A, anch, sig, T, a.M)
    F = window_features(A, anch, lab.fs_tb7)
    os.makedirs(SCRATCH, exist_ok=True); emb = {}; sets = {"a": (False, None), "b": (True, None)}
    for spec in a.emb.split(","):
        name, ck = spec.split("="); cache = f"{SCRATCH}/emb_{name}.npy"
        if os.path.exists(cache) and np.load(cache, mmap_mode="r").shape[0] == len(anch):
            emb[name] = np.load(cache)
        else:
            enc, std = load_encoder(ck); emb[name] = embed_anchors(enc, std, A, anch); np.save(cache, emb[name])
        sets[f"c:{name}"] = (False, name); sets[f"d:{name}"] = (True, name)
    print(f"anchors {len(anch)} ({str(ts[anch[0]])[:10]}..{str(ts[anch[-1] + T])[:10]}), embeddings {list(emb)}, prep {time.time() - t0:.0f}s", flush=True)
    D = dict(F=F, sig=sig, logv=logv, har_y=har_y, y=(r_T > 0).astype(int), emb=emb, pnl=make_pnl(T, barrier, t_up, t_dn, r_T), stride=a.stride,
             days=lambda te: (ts[anch[te[-1]]] - ts[anch[te[0]]]) / np.timedelta64(1, "D"))
    rng = np.random.default_rng(0); gap = math.ceil(2 * T / a.stride); folds = []
    # meta feature set keys: "m:a", "g:a", "m:b", ..., "m:c:NAME", "g:d:NAME"
    for f, (tr, te) in enumerate(fold_splits(len(anch), a.stride, T)):
        assert anch[tr[-1]] + T - 1 < anch[te[0]] - 1, "train labels must end before the first test entry"
        r = run_fold(D, tr, te, sets, a.q, rng, gap); folds.append(r)
        print(f"fold {f} test {str(ts[anch[te[0]]])[:10]}..{str(ts[anch[te[-1]]])[:10]}: " +
              " | ".join(f"{k} {v['bps']:+.2f}bps/{v['hit']:.3f}" for k, v in r.items() if v and k.startswith(("primary", "g:", "m:a"))), flush=True)
    S = summarise(folds)
    rec = {"tag": a.tag or "tradefilter", "config": dict(T=T, M=a.M, q=a.q, stride=a.stride, start=str(START)[:10], end_exclusive=str(HOLD)[:10],
                                                         emb={k: v for k, v in (s.split("=") for s in a.emb.split(","))}),
           "n_anchors": int(len(anch)), "folds": folds, "summary": S, "seconds": round(time.time() - t0)}
    open(a.out, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f},{x[2]:+.2f}]"
    print("SUMMARY (mean over 8 folds [95% t-CI]): key | trades/day | hit | bps | null95 | excess | paired vs a (bps / hit / excess)")
    for k, s in S.items():
        pv = s.get("paired_vs_a")
        print(f"{k:12s} | {s['tpd'][0]:6.1f} | {s['hit'][0]:.4f} | {fmt(s['bps'])} | {s['null95'][0]:+.2f} | {fmt(s['excess_bps'])} | " +
              (f"{fmt(pv['bps'])} / {pv['hit'][0]:+.4f} / {fmt(pv['excess_bps'])}" if pv else "-"))


if __name__ == "__main__":
    main()
