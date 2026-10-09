"""Hourly triple-barrier study, one configuration per call (SPEC.md in this folder, fixed before any run).
usage: python hourly.py --T 24 --tp 1 --sl 1 [--feat tb7|rich|ctx] [--primary logreg|hgb] [--target sign|barrier]
                        [--meta base|mag] [--q 0.2] [--final]
Without --final: 8 walk-forward folds inside the dev period (entries before 2024-07-01 whose labels end before it).
With --final: train on the whole dev period (minus the 2T gap), score the held-out period 2024-07-01..2025-09-29 once.
Appends one JSON line to runs/tactical/hourly/results.jsonl (dev) or final.jsonl (held-out)."""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import fs_rich, fs_tb7  # noqa: E402

OUT = "runs/tactical/hourly"; L = 60; HOLD = pd.Timestamp("2024-07-01"); COSTS = (5, 10)
TQ = {7: 2.36, 14: 2.14, 15: 2.13, 16: 2.12}


def tq(n): return TQ.get(n - 1, 2.0)


def bars():
    p = f"{OUT}/bars_1h.npz"
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True); return pd.DataFrame(d["A"], index=pd.to_datetime(d["t"]), columns=list("ohlcv"))
    s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    b = s.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    b.columns = list("ohlcv"); np.savez(p, A=b.values, t=b.index.values.astype("datetime64[ns]")); return b


def ctx_features(b):
    """hourly context at each bar (known at its close): multi-scale returns over vol, vol ratios, range position, calendar, volume."""
    c, h, l, v = b["c"], b["h"], b["l"], b["v"]; r1 = np.log(c).diff()
    vol24 = r1.rolling(24).std(); vol168 = r1.rolling(168).std(); vol6 = r1.rolling(6).std()
    f = {f"ret{k}": np.log(c / c.shift(k)) / (vol24 * math.sqrt(k)) for k in (1, 4, 12, 24, 72, 168)}
    f["v24_168"] = np.log(vol24 / vol168); f["v6_24"] = np.log(vol6 / vol24); f["lvol24"] = np.log(vol24)
    for k in (24, 168):
        lo, hi = l.rolling(k).min(), h.rolling(k).max(); f[f"pos{k}"] = (c - lo) / (hi - lo) - 0.5
    hr, dw = b.index.hour.values, b.index.dayofweek.values
    f["hs"], f["hc"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["ds"], f["dc"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    f["vr24"] = np.log1p(v / v.rolling(24).mean()); f["vr24_168"] = np.log1p(v.rolling(24).mean() / v.rolling(168).mean())
    return pd.DataFrame(f, index=b.index).replace([np.inf, -np.inf], np.nan).values


def first_touch(M):
    return np.where(M.any(1), M.argmax(1), 10 ** 6)


def build(a):
    b = bars(); A = b.values.astype(np.float64); n = len(A); T = a.T
    anch = np.arange(L + 168, n - T)                                   # window [i-L, i), entry at close[i-1]
    t_entry = b.index[anch - 1]; t_exit = b.index[anch + T - 1]
    c0 = A[anch - 1, 3]; lr = np.diff(np.log(A[:, 3]))
    sig = np.maximum(np.lib.stride_tricks.sliding_window_view(lr, L - 1)[anch - L].std(1), 1e-6)
    unit = sig * math.sqrt(T)
    Hf = np.lib.stride_tricks.sliding_window_view(A[:, 1], T)[anch]; Lf = np.lib.stride_tricks.sliding_window_view(A[:, 2], T)[anch]
    rT = A[anch + T - 1, 3] / c0 - 1
    tp, sl = a.tp * unit, a.sl * unit
    lt, ls = first_touch(Hf >= (c0 * (1 + tp))[:, None]), first_touch(Lf <= (c0 * (1 - sl))[:, None])
    st_, ss = first_touch(Lf <= (c0 * (1 - tp))[:, None]), first_touch(Hf >= (c0 * (1 + sl))[:, None])
    pl_long = np.where(ls <= lt, np.where(ls < T, -sl, rT), np.where(lt < T, tp, rT))
    pl_short = np.where(ss <= st_, np.where(ss < T, -sl, -rT), np.where(st_ < T, tp, -rT))
    W = np.stack([A[i - L:i] for i in anch]).astype(np.float32)
    feats = {"tb7": lambda: fs_tb7(W), "rich": lambda: fs_rich(W),
             "ctx": lambda: np.column_stack([fs_rich(W), ctx_features(b)[anch - 1]])}
    F = np.nan_to_num(feats[a.feat]()).astype(np.float64)
    return dict(F=F, rT=rT, pl_long=pl_long, pl_short=pl_short, sig=sig, unit=unit, t_entry=t_entry, t_exit=t_exit)


def fit_clf(kind, X, y):
    if kind == "logreg":
        return LogisticRegression(C=0.1, max_iter=1000).fit(X, y)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


def run_split(D, a, tr, te, rng):
    """train on index array tr, score on te. Out-of-fold primary predictions inside train feed the meta model (no in-sample bias)."""
    F, y_sign = D["F"], (D["rT"] > 0).astype(int)
    y = y_sign if a.target == "sign" else (D["pl_long"] > D["pl_short"]).astype(int)
    pnl = lambda side, idx: np.where(side > 0, D["pl_long"][idx], D["pl_short"][idx])
    gap = 2 * a.T; blocks = np.array_split(tr, 4)
    sc = StandardScaler().fit(F[tr]); X = sc.transform(F)
    # out-of-fold primary on train blocks 2..4 (each from a model trained on earlier blocks)
    oof_idx, oof_p = [], []
    for k in (1, 2, 3):
        fit_idx = np.concatenate(blocks[:k]); fit_idx = fit_idx[fit_idx < blocks[k][0] - gap]
        m = fit_clf(a.primary, X[fit_idx], y[fit_idx]); oof_idx.append(blocks[k]); oof_p.append(m.predict_proba(X[blocks[k]])[:, 1])
    prim = fit_clf(a.primary, X[tr], y[tr]); p_te = prim.predict_proba(X[te])[:, 1]

    def meta_X(idx, p):
        cols = [X[idx], np.abs(p - 0.5)[:, None], np.log(D["sig"][idx])[:, None]]
        if a.meta == "mag":
            cols.append(mag.predict(X[idx])[:, None])
        return np.hstack(cols)
    mag = None
    if a.meta == "mag":
        mag = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                            random_state=0).fit(X[tr], np.log(np.abs(D["rT"][tr]) + 1e-5))
    i23, p23 = np.concatenate(oof_idx[:2]), np.concatenate(oof_p[:2]); i4, p4 = oof_idx[2], oof_p[2]
    win23 = (pnl(np.where(p23 > 0.5, 1, -1), i23) > 0).astype(int)
    meta = fit_clf("hgb", meta_X(i23, p23), win23)
    thr_meta = np.quantile(meta.predict_proba(meta_X(i4, p4))[:, 1], 1 - a.q)
    thr_conf = np.quantile(np.abs(p4 - 0.5), 1 - a.q)
    side = np.where(p_te > 0.5, 1, -1); pl = pnl(side, te)
    sel = {"primary": np.abs(p_te - 0.5) >= thr_conf, "meta": meta.predict_proba(meta_X(te, p_te))[:, 1] >= thr_meta,
           "all": np.ones(len(te), bool)}
    out = {}
    for name, k in sel.items():
        if k.sum() < 10:
            out[name] = None; continue
        null = [np.mean(np.where(rng.random(k.sum()) < 0.5, D["pl_long"][te][k], D["pl_short"][te][k])) for _ in range(200)]
        out[name] = {"n": int(k.sum()), "hit": float(np.mean(pl[k] > 0)), "bps": float(np.mean(pl[k]) * 1e4),
                     "null95": float(np.quantile(null, 0.95) * 1e4), "per_unit": float(np.mean(pl[k] / D["unit"][te][k])),
                     "trades_per_day": float(k.sum() / max(1, (D["t_entry"][te[-1]] - D["t_entry"][te[0]]).days))}
        out[name]["_pl"] = pl[k]; out[name]["_t"] = D["t_entry"][te][k]
    return out


def main():
    ap = argparse.ArgumentParser()
    for k, v in (("--T", 24), ("--tp", 1.0), ("--sl", 1.0), ("--q", 0.2)):
        ap.add_argument(k, type=type(v), default=v)
    ap.add_argument("--feat", default="tb7"); ap.add_argument("--primary", default="logreg"); ap.add_argument("--target", default="sign")
    ap.add_argument("--meta", default="base"); ap.add_argument("--final", action="store_true")
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(0)
    cfg = {k: getattr(a, k) for k in ("T", "tp", "sl", "q", "feat", "primary", "target", "meta")}
    D = build(a); idx = np.arange(len(D["F"]))
    dev = idx[D["t_exit"] < HOLD]; held = idx[D["t_entry"] >= HOLD]
    rec = {"config": cfg, "tag": "T{T}_tp{tp}_sl{sl}_{feat}_{primary}_{target}_{meta}".format(**cfg)}
    if not a.final:
        edges = np.linspace(len(dev) * 0.3, len(dev), 9).astype(int); folds = []
        for f in range(8):
            tr = dev[:max(1, edges[f] - 2 * a.T)]; te = dev[edges[f]:edges[f + 1]]
            r = run_split(D, a, tr, te, rng); folds.append({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if v else None)
                                                             for k, v in r.items()})
        summ = {}
        for name in ("primary", "meta", "all"):
            vals = [f[name] for f in folds if f[name]]
            if len(vals) < 3:
                continue
            for m in ("bps", "hit", "null95", "per_unit", "trades_per_day"):
                v = np.array([x[m] for x in vals]); se = v.std(ddof=1) / math.sqrt(len(v))
                summ[f"{name}_{m}"] = [round(float(v.mean()), 4), round(float(v.mean() - tq(len(v)) * se), 4), round(float(v.mean() + tq(len(v)) * se), 4)]
            ex = np.array([x["bps"] - x["null95"] for x in vals]); se = ex.std(ddof=1) / math.sqrt(len(ex))
            summ[f"{name}_excess"] = [round(float(ex.mean()), 4), round(float(ex.mean() - tq(len(ex)) * se), 4), round(float(ex.mean() + tq(len(ex)) * se), 4)]
        rec.update(folds=folds, summary=summ); path = f"{OUT}/results.jsonl"
    else:
        tr = dev[D["t_exit"][dev] < HOLD - pd.Timedelta(hours=2 * a.T)]
        r = run_split(D, a, tr, held, rng); summ = {}
        for name, v in r.items():
            if not v:
                continue
            months = pd.Series(v["_pl"], index=pd.DatetimeIndex(v["_t"])).groupby(pd.Grouper(freq="M")).mean().dropna() * 1e4
            se = months.std(ddof=1) / math.sqrt(len(months))
            summ[name] = {**{k: vv for k, vv in v.items() if not k.startswith("_")},
                          "monthly_bps_ci": [round(float(months.mean() - tq(len(months)) * se), 3), round(float(months.mean() + tq(len(months)) * se), 3)],
                          "n_months": int(len(months)), **{f"net{c}_bps": v["bps"] - c for c in COSTS}}
        rec.update(final=summ); path = f"{OUT}/final.jsonl"
    rec["seconds"] = round(time.time() - t0, 1)
    open(path, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    s = rec.get("summary") or rec.get("final")
    print(rec["tag"], "final" if a.final else "dev", json.dumps(s.get("meta_excess") if not a.final else s.get("meta")), f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
