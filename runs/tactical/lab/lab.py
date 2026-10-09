"""Logistic-regression lab (owner, 2026-10-09): seconds per experiment on the CPU, on exactly the network's data blocks.
Each run: features from each window (a named feature set) -> one model per horizon fitted on the train block -> scored on
the val block of each of the 6 climb slices. Reported, all with a 95% t-interval over slices (the unit of inference):
  auc        direction AUC per horizon and the mean of h0-h2 (deadband-masked, as the network's direction_auc)
  ll, brier  log loss and Brier of P(up) on the val block (h1)
  vs_net     the paired per-slice difference in mean-of-3 AUC against the network run hc5_noprice3 (seeds averaged)
  honest     the mean of the 3 horizons' P(up); thresholds for the top 10% / 5% |P-0.5| fitted on the FIRST half of the val
             block, applied to the second half after a 20-bar gap; hit rate and gross bps on h1 (honest_split.py's method)
Results are appended to runs/tactical/lab/results.jsonl.
usage: python lab.py --set tb7|rich|... [--model logreg|hgb] [--C 0.1] [--tag name] [--quiet]"""
import argparse, glob, json, math, os, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# The heavy cache is read in place from the main tactical checkout (untracked); results go to this checkout's lab folder.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache") + "/*.npz"
LAB = "runs/tactical/lab"; GAP = 20; TQ = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45}


# ---------------------------------------------------------------- indicator primitives (vectorised over windows)
def ema(x, p):
    a = 2.0 / (p + 1.0); out = np.empty_like(x); out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * x[:, t] + (1 - a) * out[:, t - 1]
    return out


def parts(W):
    W = W.astype(np.float64)
    o, h, l, c, v = (W[..., i] for i in range(5))
    d = np.diff(c, axis=1); sd = d.std(1) + 1e-12
    return o, h, l, c, v, d, sd


def f_ret(c, sd, k): return (c[:, -1] - c[:, -1 - k]) / sd
def f_ema(c, sd, p): return (c[:, -1] - ema(c, p)[:, -1]) / sd


def f_rsi(d, p):
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    rs = ema(up, p)[:, -1] / (ema(dn, p)[:, -1] + 1e-12); return (100 - 100 / (1 + rs) - 50) / 50


def f_macd(c, sd, f, s, g):
    m = ema(c, f) - ema(c, s); return (m - ema(m, g))[:, -1] / sd


def f_bb(c, p):
    w = c[:, -p:]; return (c[:, -1] - w.mean(1)) / (2 * w.std(1) + 1e-12)


def f_stoch(h, l, c, p):
    lo, hi = l[:, -p:].min(1), h[:, -p:].max(1); return (c[:, -1] - lo) / (hi - lo + 1e-12) - 0.5


def f_cci(h, l, c, p):
    tp = (h + l + c)[:, -p:] / 3; md = np.abs(tp - tp.mean(1, keepdims=True)).mean(1)
    return np.clip((tp[:, -1] - tp.mean(1)) / (0.015 * md + 1e-12) / 100, -5, 5)


def f_atr(h, l, c, sd, p):
    tr = np.maximum(h[:, 1:] - l[:, 1:], np.maximum(np.abs(h[:, 1:] - c[:, :-1]), np.abs(l[:, 1:] - c[:, :-1])))
    return ema(tr, p)[:, -1] / sd


def f_mfi(h, l, c, v, p):
    tp = (h + l + c) / 3; flow = tp * v; dtp = np.diff(tp, axis=1)
    pos = (flow[:, 1:] * (dtp > 0))[:, -p:].sum(1); neg = (flow[:, 1:] * (dtp < 0))[:, -p:].sum(1)
    return pos / (pos + neg + 1e-12) - 0.5


# ---------------------------------------------------------------- feature sets
def fs_tb7(W):
    """direct_search.py's textbook set (H23): 7 indicators at textbook periods + 6 trailing returns."""
    o, h, l, c, v, d, sd = parts(W)
    f = [f_ema(c, sd, 20), f_rsi(d, 14), f_macd(c, sd, 12, 26, 9), f_bb(c, 20), f_atr(h, l, c, sd, 14),
         f_stoch(h, l, c, 14), f_cci(h, l, c, 20)] + [f_ret(c, sd, k) for k in (1, 5, 10, 15, 20, 30)]
    return np.stack(f, 1)


def fs_rich(W):
    """all 14 families' ideas, several periods each, plus returns, volatility regime, volume and candle shape."""
    o, h, l, c, v, d, sd = parts(W)
    f = [f_ret(c, sd, k) for k in (1, 2, 3, 5, 10, 15, 20, 30, 45, 59)]
    f += [f_ema(c, sd, p) for p in (5, 10, 20, 40)]
    f += [f_rsi(d, p) for p in (7, 14, 28)]
    f += [f_macd(c, sd, 12, 26, 9), f_macd(c, sd, 5, 13, 5)]
    f += [f_bb(c, p) for p in (10, 20, 40)]
    f += [f_stoch(h, l, c, p) for p in (5, 14, 30, 60)]
    f += [f_cci(h, l, c, p) for p in (10, 20, 40)]
    f += [f_atr(h, l, c, sd, p) for p in (7, 14)]
    f += [f_mfi(h, l, c, v, p) for p in (14, 30)]
    rng = h - l + 1e-12
    f += [(c[:, -1] - o[:, -1]) / rng[:, -1], (h[:, -1] - np.maximum(o[:, -1], c[:, -1])) / rng[:, -1],
          (np.minimum(o[:, -1], c[:, -1]) - l[:, -1]) / rng[:, -1], rng[:, -1] / sd]
    f += [d[:, -10:].std(1) / sd, d[:, -5:].std(1) / sd]                        # volatility regime inside the window
    vm = v.mean(1) + 1e-12
    f += [np.log1p(v[:, -1] / vm), np.log1p(v[:, -5:].mean(1) / vm)]
    obv = np.cumsum(np.sign(d) * v[:, 1:], axis=1)
    f += [(obv[:, -1] - obv[:, -11]) / (v[:, -10:].sum(1) + 1e-12)]
    vw = (c[:, -20:] * v[:, -20:]).sum(1) / (v[:, -20:].sum(1) + 1e-12)
    f += [(c[:, -1] - vw) / sd]
    return np.stack(f, 1)


def fs_returns(W):
    """only trailing returns (the DIRECTION_SKIP idea): how much of the signal is plain momentum / reversal."""
    o, h, l, c, v, d, sd = parts(W)
    return np.stack([f_ret(c, sd, k) for k in (1, 2, 3, 5, 10, 15, 20, 30, 45, 59)], 1)


SETS = {"tb7": fs_tb7, "rich": fs_rich, "returns": fs_returns}


# ---------------------------------------------------------------- evaluation
def model_of(kind, C):
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=1000)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def ci(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v)); t = TQ.get(len(v), 2.0)
    return [round(float(m), 4), round(float(m - t * se), 4), round(float(m + t * se), 4)]


def network_auc():
    net = {}
    for f in glob.glob("runs/tactical/screens/hc5_noprice3/results*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            net.setdefault(r["data_end"][:13], []).append(np.mean([r["direction_auc"][h]["auc"] for h in ("h0", "h1", "h2")]))
    return {k: float(np.mean(v)) for k, v in net.items()}


def evaluate_slice(D, P, rng):
    """Score P_val [N,3] (P(up) per horizon) on one cached slice D (an npz with rtr, rva, deadband). Returns the per-slice
    values: auc_h0..2, auc3 (mean), ll_h1 / ll_const_h1 (log loss of P and of the train base rate), brier_h1, and the honest
    tail (thresholds on the first half of val, tested on the second half after GAP bars): hon{10,5}_hit / _bps, hon10_null95
    (random-sign null, drawn from rng)."""
    db = float(D["deadband"]); rtr, rva = D["rtr"], D["rva"]; out = {}
    for i in range(3):
        mtr, mva = np.abs(rtr[:, i]) > db, np.abs(rva[:, i]) > db
        y = (rva[mva, i] > 0).astype(int); out[f"auc_h{i}"] = roc_auc_score(y, P[mva, i])
        if i == 1:
            q = np.clip(P[mva, 1], 1e-7, 1 - 1e-7); base = (rtr[mtr, 1] > 0).mean()
            out["ll_h1"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
            out["ll_const_h1"] = float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base)))
            out["brier_h1"] = float(np.mean((q - y) ** 2))
    out["auc3"] = np.mean([out[f"auc_h{i}"] for i in range(3)])
    n = len(P); A = np.zeros(n, bool); A[:n // 2] = True; B = np.zeros(n, bool); B[n // 2 + GAP:] = True
    s = P.mean(1); conf = np.abs(s - 0.5); ret = rva[:, 1]; m = np.abs(ret) > db
    for cov in (10, 5):
        thr = np.quantile(conf[A], 1 - cov / 100); k = B & (conf >= thr) & m
        mv = np.sign(s[k] - 0.5) * ret[k] * 1e4
        out[f"hon{cov}_hit"] = float(np.mean(mv > 0)); out[f"hon{cov}_bps"] = float(np.mean(mv))
        if cov == 10:
            null = [np.mean(np.sign(s[k] - 0.5) * rng.choice([-1, 1], k.sum()) * ret[k] * 1e4) for _ in range(300)]
            out["hon10_null95"] = float(np.quantile(null, 0.95))
    return out


def run(set_name, kind="logreg", C=0.1, tag=None, quiet=False):
    t0 = time.time(); fs = SETS[set_name]; net = network_auc(); rng = np.random.default_rng(0)
    per = {"auc_h0": [], "auc_h1": [], "auc_h2": [], "auc3": [], "ll_h1": [], "ll_const_h1": [], "brier_h1": [],
           "vs_net": [], "hon10_hit": [], "hon10_bps": [], "hon5_hit": [], "hon5_bps": [], "hon10_null95": []}
    for p in sorted(glob.glob(CACHE)):
        D = np.load(p); db = float(D["deadband"]); key = str(D["data_end"])[:13]
        Ftr, Fva = np.nan_to_num(fs(D["Wtr"])), np.nan_to_num(fs(D["Wva"]))
        sc = StandardScaler().fit(Ftr); Ftr, Fva = sc.transform(Ftr), sc.transform(Fva)
        P = np.zeros((len(Fva), 3))
        for i in range(3):
            rtr = D["rtr"][:, i]; mtr = np.abs(rtr) > db
            m = model_of(kind, C).fit(Ftr[mtr], (rtr[mtr] > 0).astype(int))
            P[:, i] = m.predict_proba(Fva)[:, 1]
        r = evaluate_slice(D, P, rng)
        for k, v in r.items():
            per[k].append(v)
        if key in net:
            per["vs_net"].append(per["auc3"][-1] - net[key])
    res = {"tag": tag or f"{set_name}_{kind}_C{C}", "set": set_name, "model": kind, "C": C, "n_features": int(Ftr.shape[1]),
           "seconds": round(time.time() - t0, 1), "per_slice": {k: [round(float(x), 4) for x in v] for k, v in per.items()},
           **{k: ci(v) for k, v in per.items() if len(v) > 1}}
    open(f"{LAB}/results.jsonl", "a", encoding="utf-8").write(json.dumps(res) + "\n")
    if not quiet:
        f = lambda k: f"{res[k][0]:.4f} [{res[k][1]:.4f},{res[k][2]:.4f}]"
        print(f"{res['tag']}: {res['n_features']} features, {res['seconds']} s")
        print(f"  AUC h0 {res['auc_h0'][0]:.4f} h1 {res['auc_h1'][0]:.4f} h2 {res['auc_h2'][0]:.4f} | mean3 {f('auc3')}")
        print(f"  vs network (mean3, paired)  {f('vs_net')}")
        print(f"  h1 log loss {res['ll_h1'][0]:.4f} (constant {res['ll_const_h1'][0]:.4f}), Brier {res['brier_h1'][0]:.4f}")
        print(f"  honest top10% hit {f('hon10_hit')} {res['hon10_bps'][0]:+.2f} bps (null95 {res['hon10_null95'][0]:+.2f}) | "
              f"top5% hit {f('hon5_hit')} {res['hon5_bps'][0]:+.2f} bps")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="tb7"); ap.add_argument("--model", default="logreg")
    ap.add_argument("--C", type=float, default=0.1); ap.add_argument("--tag"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); run(a.set, a.model, a.C, a.tag, a.quiet)
