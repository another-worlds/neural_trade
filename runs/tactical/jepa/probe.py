"""Probes on the frozen JEPA context embedding, on the cached lab slices whose train block starts after 2020-01-02 (the
pretraining span ends 2019-12-31). Per slice: fit on the train block, score on the val block with lab.evaluate_slice.
  (a) tb7 logreg C 0.1   (b) embedding logreg   (c) tb7 + embedding logreg ; volatility: ridge on log|r_h1| for tb7 / emb / both.
usage: python probe.py [--ckpt ckpt] [--tag jepa_v1]"""
import argparse, glob, json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "lab"))
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
CACHE = "D:/nt/nt_tactical/runs/tactical/lab/cache/*.npz"
CUTOFF = np.datetime64("2020-01-02T00:00")
PROBES = ("a_tb7", "b_emb", "c_both")


def file_bars():
    d = pd.read_csv(CSV, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    return d.timestamp.values, d.close.values


def locate(close, W):
    """index in the file of the first bar of the first cached window; asserts the whole block is consecutive file bars."""
    c0 = W[0, :, 3].astype(np.float64)
    cand = np.where(np.abs(close - c0[0]) < 1e-6 * abs(c0[0]) + 0.02)[0]
    ok = [i for i in cand if i + 60 <= len(close) and np.allclose(close[i:i + 60], c0, rtol=1e-6, atol=0.02)]
    if len(ok) != 1: return None
    i = ok[0]; last = close[i + 59: i + 59 + len(W)]
    assert len(last) == len(W) and np.allclose(last, W[:, -1, 3], rtol=1e-6, atol=0.02), "cached windows are not consecutive file bars"
    return i


def embed(enc, std, W, bs=2048):
    from channels import context_channels
    out = []
    for s in range(0, len(W), bs):
        ch, _, _ = context_channels(W[s:s + bs]); out.append(enc(std(ch), training=False).numpy())
    return np.concatenate(out)


def standardise(Ft, Fv):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ft); return sc.transform(Ft), sc.transform(Fv)


def logit_probs(Ft, Fv, rtr, db):
    """one logistic regression (C 0.1) per horizon on the deadband-masked train rows; P(up) on every val row."""
    from sklearn.linear_model import LogisticRegression
    P = np.zeros((len(Fv), 3))
    for h in range(3):
        m = np.abs(rtr[:, h]) > db
        P[:, h] = LogisticRegression(C=0.1, max_iter=1000).fit(Ft[m], (rtr[m, h] > 0).astype(int)).predict_proba(Fv)[:, 1]
    return P


def ci(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v)); t = stats.t.ppf(0.975, len(v) - 1)
    return [round(float(m), 4), round(float(m - t * se), 4), round(float(m + t * se), 4)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=HERE + "/ckpt"); ap.add_argument("--tag", default="jepa_v1")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import neural_trade  # noqa: F401
    import lab
    from sklearn.linear_model import Ridge
    import model as M
    from channels import N_CH, Standardiser, CTX
    meta = json.load(open(a.ckpt + "/meta.json"))
    std = Standardiser(np.array(meta["std"]["mean"]), np.array(meta["std"]["std"]))
    enc = M.Encoder(N_CH); enc(np.zeros((2, CTX, N_CH), np.float32)); enc.load_weights(a.ckpt + "/enc.h5")
    ts, close = file_bars(); rng = np.random.default_rng(0)
    per = {p: {} for p in PROBES}; vol = {p: [] for p in PROBES}; used = []
    for path in sorted(glob.glob(CACHE)):
        D = np.load(path); i = locate(close, D["Wtr"])
        if i is None or ts[i] < CUTOFF: continue
        key = str(D["data_end"])[:13]; used.append({"data_end": key, "train_start": str(ts[i])[:16]})
        db = float(D["deadband"])
        T_tb, V_tb = np.nan_to_num(lab.fs_tb7(D["Wtr"])), np.nan_to_num(lab.fs_tb7(D["Wva"]))
        T_em, V_em = embed(enc, std, D["Wtr"]), embed(enc, std, D["Wva"])
        feats = {"a_tb7": (T_tb, V_tb), "b_emb": (T_em, V_em), "c_both": (np.hstack([T_tb, T_em]), np.hstack([V_tb, V_em]))}
        for p, (Ft, Fv) in feats.items():
            Ft, Fv = standardise(Ft, Fv)
            P = logit_probs(Ft, Fv, D["rtr"], db)
            for k, v in lab.evaluate_slice(D, P, rng).items(): per[p].setdefault(k, []).append(v)
            yt = np.log(np.abs(D["rtr"][:, 1]) + 1e-5)
            pv = Ridge(alpha=1.0).fit(Ft, yt).predict(Fv)
            vol[p].append(stats.spearmanr(pv, np.abs(D["rva"][:, 1])).correlation)
        print(key, {p: round(per[p]["auc3"][-1], 4) for p in PROBES}, {p: round(vol[p][-1], 3) for p in PROBES}, flush=True)
    res = {"tag": a.tag, "n_slices": len(used), "slices": used, "pretrain": meta, "probes": {}}
    for p in PROBES:
        res["probes"][p] = {**{k: ci(v) for k, v in per[p].items()}, "vol_spearman": ci(vol[p]),
                            "per_slice": {k: [round(float(x), 4) for x in v] for k, v in per[p].items()}, "per_slice_vol": [round(float(x), 4) for x in vol[p]]}
    res["paired"] = {}
    for p in ("b_emb", "c_both"):
        res["paired"][p + "-a_tb7"] = {"auc3": ci(np.array(per[p]["auc3"]) - np.array(per["a_tb7"]["auc3"])),
                                       "vol_spearman": ci(np.array(vol[p]) - np.array(vol["a_tb7"])),
                                       "hon10_bps": ci(np.array(per[p]["hon10_bps"]) - np.array(per["a_tb7"]["hon10_bps"]))}
    open(HERE + "/results.jsonl", "a", encoding="utf-8").write(json.dumps(res) + "\n")
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
    print(f"\n{len(used)} slices")
    for p in PROBES:
        r = res["probes"][p]
        print(f"{p}: mean3 AUC {f(r['auc3'])} | h0/h1/h2 {r['auc_h0'][0]:.4f}/{r['auc_h1'][0]:.4f}/{r['auc_h2'][0]:.4f} | top10 hit {f(r['hon10_hit'])} "
              f"bps {r['hon10_bps'][0]:+.2f} null95 {r['hon10_null95'][0]:+.2f} | vol Spearman {f(r['vol_spearman'])}")
    for k, v in res["paired"].items(): print(f"paired {k}: AUC {f(v['auc3'])} vol {f(v['vol_spearman'])} top10 bps {f(v['hon10_bps'])}")


if __name__ == "__main__":
    main()
