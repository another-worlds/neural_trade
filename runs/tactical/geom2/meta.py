"""Night experiment 4: geometry as META-MODEL features in the triple barrier (hourly.py's method, out-of-fold primary predictions).
Setups (dev only, entries and exits before 2024-07-01; the held-out period is never touched):
  minute  1-minute bars from 2020-01-01, anchors every 5 bars, T 15, tp = sl = 1.0 sigma sqrt(T), q 0.10 (barrier.py's setup)
  hourly  1-hour bars, T 12, tp = sl = 1.5 (the hourly dev winner's geometry), q 0.20
8 walk-forward folds (edges as hourly.py: linspace(0.3 n, n, 9), gap 2T). Primary: logistic regression (C 0.1) on the sign of the T-bar
return; the meta model (HistGradientBoosting) predicts whether the primary side wins from features + |P - 0.5| + log sigma, trained on
the primary's OUT-OF-FOLD predictions inside the training span (3 blocks, each predicted by a model trained on earlier blocks).
Variants:
  base    primary on F (tb7 or rich); meta features: F std + conf + sigma
  mG5     meta also gets the old 5-channel geometry g5
  mGR     meta also gets the 22-channel geometry gR (crossing windows 3/5/10/20)
  pmGR    primary AND meta use F + gR
Per fold: n trades, hit, mean bps, random-side null95 (same trades), trades/day; for the primary top-q ("primary") and the meta top-q ("meta").
usage: python meta.py build --bars 1min|1h   |   python meta.py run --bars 1min|1h --feat tb7|rich --variant base|mG5|mGR|pmGR   |   python meta.py report"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "lab"))
import lab  # noqa: E402  (chdirs to the worktree)
import geomlib as G  # noqa: E402

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"; L = 60; HOLD = pd.Timestamp("2024-07-01")
FC = os.path.join(HERE, "featcache"); os.makedirs(FC, exist_ok=True)
SETUP = {"1min": dict(T=15, tp=1.0, sl=1.0, q=0.10, stride=5, start="2020-01-01", min_leaf=500),
         "1h": dict(T=12, tp=1.5, sl=1.5, q=0.20, stride=1, start="2017-01-01", min_leaf=300),
         "1min_a": dict(T=15, tp=1.0, sl=1.0, q=0.10, stride=5, start="2017-01-01", end="2020-01-01", min_leaf=500)}   # replication on 2017-2019 (dev, before 2024-07)
TQ = {8: 2.36}


def bars(kind):
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    s = s[(s.index >= SETUP[kind]["start"]) & (s.index < pd.Timestamp(SETUP[kind].get("end", HOLD)))]
    if kind == "1h":
        s = s.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    s.columns = list("ohlcv"); return s


def first_touch(M):
    return np.where(M.any(1), M.argmax(1), 10 ** 6)


def labels(kind, T, tp_m, sl_m):
    """Triple-barrier labels for the anchors of `kind` at time limit T (bars) and barriers tp_m / sl_m sigma sqrt(T)."""
    S = SETUP[kind]; b = bars(kind); A = b.values.astype(np.float64); n = len(A)
    first = L + 168 if kind == "1h" else L
    anch = np.arange(first, n - T, S["stride"]); t_entry = b.index[anch - 1]; t_exit = b.index[anch + T - 1]
    c0 = A[anch - 1, 3]; lr = np.diff(np.log(A[:, 3]))
    sig = np.maximum(swv(lr, L - 1)[anch - L].std(1), 1e-6); unit = sig * math.sqrt(T)
    Hf, Lf = swv(A[:, 1], T)[anch], swv(A[:, 2], T)[anch]; rT = A[anch + T - 1, 3] / c0 - 1
    tp, sl = tp_m * unit, sl_m * unit
    lt, ls = first_touch(Hf >= (c0 * (1 + tp))[:, None]), first_touch(Lf <= (c0 * (1 - sl))[:, None])
    st_, ss = first_touch(Lf <= (c0 * (1 - tp))[:, None]), first_touch(Hf >= (c0 * (1 + sl))[:, None])
    pl_long = np.where(ls <= lt, np.where(ls < T, -sl, rT), np.where(lt < T, tp, rT))
    pl_short = np.where(ss <= st_, np.where(ss < T, -sl, -rT), np.where(st_ < T, tp, -rT))
    return dict(rT=rT, pl_long=pl_long, pl_short=pl_short, sig=sig, unit=unit,
                t_entry=t_entry.values.astype("datetime64[ns]").astype(np.int64), t_exit=t_exit.values.astype("datetime64[ns]").astype(np.int64)), A, anch


def build(kind):
    S = SETUP[kind]; lab_, A, anch = labels(kind, S["T"], S["tp"], S["sl"])
    F = {k: [] for k in ("tb7", "rich", "g5", "gR")}
    for k in range(0, len(anch), 20000):
        W = np.stack([A[i - L:i] for i in anch[k:k + 20000]]).astype(np.float32)
        F["tb7"].append(np.nan_to_num(lab.fs_tb7(W))); F["rich"].append(np.nan_to_num(lab.fs_rich(W)))
        F["g5"].append(G.geom_base5(W)); F["gR"].append(G.geom_rich(W))
    F = {k: np.concatenate(v).astype(np.float32) for k, v in F.items()}
    np.savez(f"{FC}/meta_{kind}.npz", **lab_, **{f"F_{k}": v for k, v in F.items()})
    print(kind, len(anch), "anchors", {k: v.shape[1] for k, v in F.items()}, flush=True)


def hgb(min_leaf):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=min_leaf, l2_regularization=1.0, random_state=0)


def split(D, S, Fp, Fm, tr, te, rng):
    """Fp: primary features, Fm: meta features (columns beyond Fp may be added); out-of-fold primary inside tr."""
    y = (D["rT"] > 0).astype(int); gap = max(1, 2 * S["T"] // S["stride"]); blocks = np.array_split(tr, 4)
    sp = StandardScaler().fit(Fp[tr]); Xp = sp.transform(Fp); sm = StandardScaler().fit(Fm[tr]); Xm = sm.transform(Fm)
    clf = (lambda: hgb(S['min_leaf'])) if S.get('primary') == 'hgb' else (lambda: LogisticRegression(C=0.1, max_iter=1000))
    oof_idx, oof_p = [], []
    for k in (1, 2, 3):
        fi = np.concatenate(blocks[:k]); fi = fi[fi < blocks[k][0] - gap]
        oof_idx.append(blocks[k]); oof_p.append(clf().fit(Xp[fi], y[fi]).predict_proba(Xp[blocks[k]])[:, 1])
    p_te = clf().fit(Xp[tr], y[tr]).predict_proba(Xp[te])[:, 1]
    mx = lambda idx, p: np.hstack([Xm[idx], np.abs(p - 0.5)[:, None], np.log(D["sig"][idx])[:, None]])
    pnl = lambda side, idx: np.where(side > 0, D["pl_long"][idx], D["pl_short"][idx])
    i23, p23 = np.concatenate(oof_idx[:2]), np.concatenate(oof_p[:2]); i4, p4 = oof_idx[2], oof_p[2]
    win23 = (pnl(np.where(p23 > 0.5, 1, -1), i23) > 0).astype(int)
    meta = hgb(S["min_leaf"]).fit(mx(i23, p23), win23)
    thr_meta = np.quantile(meta.predict_proba(mx(i4, p4))[:, 1], 1 - S["q"]); thr_conf = np.quantile(np.abs(p4 - 0.5), 1 - S["q"])
    side = np.where(p_te > 0.5, 1, -1); pl = pnl(side, te)
    sel = {"primary": np.abs(p_te - 0.5) >= thr_conf, "meta": meta.predict_proba(mx(te, p_te))[:, 1] >= thr_meta}
    ndays = max(1, (D["t_entry"][te[-1]] - D["t_entry"][te[0]]) // (86400 * 10 ** 9)); out = {}
    for name, k in sel.items():
        if k.sum() < 10:
            out[name] = None; continue
        null = [np.mean(np.where(rng.random(k.sum()) < 0.5, D["pl_long"][te][k], D["pl_short"][te][k])) for _ in range(200)]
        out[name] = dict(n=int(k.sum()), hit=float(np.mean(pl[k] > 0)), bps=float(np.mean(pl[k]) * 1e4), null95=float(np.quantile(null, 0.95) * 1e4),
                         per_day=float(k.sum() / ndays * S["stride"]))
    return out


def run(kind, feat, variant, q=None, T=None, M=None, primary=None):
    S = dict(SETUP[kind]); S["q"] = q or S["q"]; S["primary"] = primary or "logreg"; D = dict(np.load(f"{FC}/meta_{kind}.npz"))
    if T or M:      # other barrier geometry: new labels on the first n_T anchors of the cached features (T >= the cached T keeps them a prefix)
        assert (T or S["T"]) >= S["T"]; S["T"] = T or S["T"]; S["tp"] = S["sl"] = M or S["tp"]
        lab_, _, anch = labels(kind, S["T"], S["tp"], S["sl"]); D.update(lab_)
        D = {k: (v[:len(anch)] if k.startswith("F_") else v) for k, v in D.items()}
    t0 = time.time(); rng = np.random.default_rng(0)
    F, g5, gR = D[f"F_{feat}"], D["F_g5"], D["F_gR"]
    perm = np.random.default_rng(1).permutation(len(F)); gP = gR[perm]       # controls: geometry rows shuffled across time (same marginals, no information)
    Fp, Fm = {"base": (F, F), "mG5": (F, np.hstack([F, g5])), "mGR": (F, np.hstack([F, gR])), "pmGR": (np.hstack([F, gR]), np.hstack([F, gR])),
              "mRich": (F, np.hstack([F, D["F_rich"]])), "mPerm": (F, np.hstack([F, gP])), "pmPerm": (np.hstack([F, gP]), np.hstack([F, gP])),
              "pmRich": (np.hstack([F, D["F_rich"]]), np.hstack([F, D["F_rich"]])), "pmG5": (np.hstack([F, g5]), np.hstack([F, g5]))}.get(variant, (F, F))
    if variant.startswith("pmG_"):      # ablation by channel group / feature type: primary and meta get F + that part of the 22-channel geometry
        names = G.geom_names(None, (3, 5, 10, 20)); grp = variant[4:]
        CH = {"trend": ("ema", "sma", "vwap"), "band": ("bb_", "kel", "don"), "osc": ("rsi", "macd", "stoch", "cci", "mfi", "dmi"), "vol": ("adx", "atr", "obv")}
        if grp in CH:
            cols = [i for i, nm in enumerate(names) if nm.startswith(CH[grp])]
        else:
            cols = [i for i, nm in enumerate(names) if nm.split(".")[1].startswith({"slope": "slope", "pos": "pos", "cross": "cross", "squeeze": "squeeze"}[grp])]
        Fp = Fm = np.hstack([F, gR[:, cols]])
    n = len(F); edges = np.linspace(n * 0.3, n, 9).astype(int); gap = max(1, 2 * S["T"] // S["stride"])
    path = os.path.join(HERE, "meta_rows.jsonl")
    for f in range(8):
        tr = np.arange(0, max(1, edges[f] - gap)); te = np.arange(edges[f], edges[f + 1]); r = split(D, S, Fp, Fm, tr, te, rng)
        open(path, "a", encoding="utf-8").write(json.dumps(dict(bars=kind, feat=feat, variant=variant, q=S["q"], T=S["T"], M=S["tp"], pkind=S["primary"], fold=f, **r)) + "\n")
        print(f"{kind} {feat} {variant} fold {f}: " + " ".join(f"{k} {v['bps']:+.2f} (null {v['null95']:+.2f}, n {v['n']})" for k, v in r.items() if v),
              f"{time.time() - t0:.0f}s", flush=True)


def report():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "meta_rows.jsonl"))]; ci = lambda v: (np.mean(v), np.mean(v) - TQ[8] * np.std(v, ddof=1) / math.sqrt(8),
                                                                                               np.mean(v) + TQ[8] * np.std(v, ddof=1) / math.sqrt(8))
    fmt = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"; lines = []
    cells = {}
    for r in rows:
        cells.setdefault((r["bars"], r["feat"], r["variant"], r.get("q", SETUP[r["bars"]]["q"]), r.get("T", SETUP[r["bars"]]["T"]), r.get("M", SETUP[r["bars"]]["tp"]), r.get("pkind", "logreg")), {})[r["fold"]] = r
    for (bars_, feat, var, q_, T_, M_, pr_), fd in sorted(cells.items()):
        if len(fd) < 8 or any(fd[f].get("meta") is None or fd[f].get("primary") is None for f in range(8)):
            continue
        for sel in ("primary", "meta"):
            v = {k: np.array([fd[f][sel][k] for f in range(8)]) for k in ("bps", "hit", "null95", "per_day", "n")}
            lines.append(f"{bars_:4s} {feat:4s} {pr_[:3]} T{T_} M{M_} q{q_:.2f} {var:6s} {sel:8s} bps {fmt(ci(v['bps']))} hit {v['hit'].mean():.3f} null95 {v['null95'].mean():+.2f} "
                         f"excess {fmt(ci(v['bps'] - v['null95']))} trades/day {v['per_day'].mean():.1f}")
        base = cells.get((bars_, feat, "base", q_, T_, M_, pr_))
        if var != "base" and base and len(base) == 8:
            for sel in ("primary", "meta"):
                d = np.array([fd[f][sel]["bps"] - base[f][sel]["bps"] for f in range(8)])
                e = np.array([(fd[f][sel]["bps"] - fd[f][sel]["null95"]) - (base[f][sel]["bps"] - base[f][sel]["null95"]) for f in range(8)])
                lines.append(f"       paired vs base ({sel}): d bps {fmt(ci(d))}, d excess over null95 {fmt(ci(e))}")
    PAIRS = [("pmGR", "pmRich"), ("pmGR", "pmPerm"), ("mGR", "mRich"), ("mGR", "mPerm"), ("pmG5", "pmRich"), ("pmGR", "pmG5")]
    lines.append("\nDirect pairs (a minus b, 8 folds, d bps and d excess over null95):")
    for (bars_, feat, var, q_, T_, M_, pr_), fd in sorted(cells.items()):
        for a_, b_ in PAIRS:
            if var != a_ or len(fd) < 8:
                continue
            ob = cells.get((bars_, feat, b_, q_, T_, M_, pr_))
            if not ob or len(ob) < 8:
                continue
            for sel in ("primary", "meta"):
                d = np.array([fd[f][sel]["bps"] - ob[f][sel]["bps"] for f in range(8)])
                e = np.array([(fd[f][sel]["bps"] - fd[f][sel]["null95"]) - (ob[f][sel]["bps"] - ob[f][sel]["null95"]) for f in range(8)])
                lines.append(f"{bars_:4s} {feat:4s} {pr_[:3]} T{T_} M{M_} q{q_:.2f} {a_} - {b_} {sel:8s}: d bps {fmt(ci(d))}, d excess {fmt(ci(e))}")
    txt = "\n".join(lines); print(txt); open(os.path.join(HERE, "meta_report.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--bars", default="1h"); ap.add_argument("--feat", default="tb7"); ap.add_argument("--variant", default="base"); ap.add_argument("--q", type=float, default=0.0); ap.add_argument("--T", type=int, default=0); ap.add_argument("--M", type=float, default=0.0); ap.add_argument("--primary", default="")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.bars)
    elif a.cmd == "run":
        run(a.bars, a.feat, a.variant, a.q or None, a.T or None, a.M or None, a.primary or None)
    else:
        report()
