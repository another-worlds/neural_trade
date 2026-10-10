"""v4 probe: the v2 protocol (probe2.py) on cached lab slices whose train block starts after the pretraining span ends + 1 day.
Per slice, fit on the train block, score on the val block. Features: tb7, emb, tb7+emb (per variant). Direction: logreg C 0.1, mean3 AUC
(lab.evaluate_slice). Volatility: Spearman of ridge and boosting predictions of log|r_h1| against realised |r_h1|.
Paired differences (95% t-CI over slices) against tb7 and against the reference encoder (--ref, default the v2 variant c, 60-bar ctx).
Context of a window with ctx > 60 is read from the file bars (the ctx bars before the window's next bar), never from the future.
usage: python probe4.py --variants e1,m50 --tag exp1 [--ref v2c] [--cutoff 2022-01-02]   (CPU)"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "v2"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lab"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pandas as pd  # noqa: E402
import probe as P1   # noqa: E402
from probe2 import vol_scores, eff_rank_np  # noqa: E402

REF_V2C = "D:/nt/nt_wt_jepa2/runs/tactical/jepa/v2/ckpt/c"   # weights are not in git; read in place


def file_ohlcv():
    d = pd.read_csv(P1.CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    return d.timestamp.values, d[["open", "high", "low", "close", "volume"]].values.astype(np.float64)


def load_enc(name):
    """name 'v2c' = the v2 checkpoint (Encoder2, 60 bars); otherwise ckpt/<name> (Encoder4)."""
    from channels import N_CH, Standardiser
    d = REF_V2C if name == "v2c" else f"{HERE}/ckpt/{name}"
    meta = json.load(open(d + "/meta.json"))
    std = Standardiser(np.array(meta["std"]["mean"]), np.array(meta["std"]["std"]))
    if name == "v2c":
        import model2 as M2; enc = M2.Encoder2(N_CH); ctx = 60
    else:
        import model4 as M4; ctx = meta["ctx"]; enc = M4.Encoder4(N_CH, **meta["arch"])
    enc(np.zeros((2, ctx, N_CH), np.float32)); enc.load_weights(d + "/enc.h5")
    return enc, std, meta, ctx


def embed_block(enc, std, bars, i0, n, ctx, bs=2048):
    """embedding of the n consecutive windows of a block whose first window starts at file bar i0 (window j = bars i0+j .. i0+j+59);
    the context is the ctx bars ending at the window's last bar (ctx 60 = the window itself)."""
    from channels import context_channels
    out = []
    for s in range(0, n, bs):
        j = np.arange(s, min(s + bs, n)); end = i0 + j + 60            # exclusive end = bar after the window's last bar
        W = bars[end[:, None] + np.arange(-ctx, 0)]
        ch, _, _ = context_channels(W); out.append(enc(std(ch), training=False).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default="v2c"); ap.add_argument("--cutoff", default=None)
    ap.add_argument("--results", default=HERE + "/results.jsonl")
    a = ap.parse_args(); variants = a.variants.split(",")
    names = variants + ([a.ref] if a.ref and a.ref not in variants else [])
    encs = {v: load_enc(v) for v in names}
    if a.cutoff is None:
        ends = [pd.Timestamp(encs[v][2]["span"][1]) for v in variants]
        cutoff = np.datetime64((max(ends) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00"))
    else:
        cutoff = np.datetime64(a.cutoff)
    import neural_trade  # noqa: F401
    import lab
    ts, bars = file_ohlcv(); close = bars[:, 3]; rng = np.random.default_rng(0)
    per, volr, volh, used = {}, {}, {}, []
    diag = {v: {"std_mean": [], "std_min": [], "eff_rank": []} for v in names}
    for path in sorted(glob.glob(P1.CACHE)):
        D = np.load(path); i = P1.locate(close, D["Wtr"]); iv = P1.locate(close, D["Wva"])
        if i is None or iv is None or ts[i] < cutoff or i < 130 or iv < 130: continue
        key = str(D["data_end"])[:13]; used.append({"data_end": key, "train_start": str(ts[i])[:16]}); db = float(D["deadband"])
        T_tb, V_tb = np.nan_to_num(lab.fs_tb7(D["Wtr"])), np.nan_to_num(lab.fs_tb7(D["Wva"]))
        sets = {"tb7": (T_tb, V_tb)}
        for v, (enc, std, _, ctx) in encs.items():
            T_em, V_em = embed_block(enc, std, bars, i, len(D["Wtr"]), ctx), embed_block(enc, std, bars, iv, len(D["Wva"]), ctx)
            sets[(v, "emb")] = (T_em, V_em); sets[(v, "both")] = (np.hstack([T_tb, T_em]), np.hstack([V_tb, V_em]))
            sv = V_em.std(0); diag[v]["std_mean"].append(float(sv.mean())); diag[v]["std_min"].append(float(sv.min()))
            diag[v]["eff_rank"].append(eff_rank_np(V_em[:4000]))
        line = {}
        for k, (Ft, Fv) in sets.items():
            Ft, Fv = P1.standardise(Ft, Fv)
            P = P1.logit_probs(Ft, Fv, D["rtr"], db)
            for m, x in lab.evaluate_slice(D, P, rng).items(): per.setdefault(k, {}).setdefault(m, []).append(x)
            r, h = vol_scores(Ft, Fv, D["rtr"], D["rva"]); volr.setdefault(k, []).append(r); volh.setdefault(k, []).append(h)
            line[str(k)] = (round(per[k]["auc3"][-1], 3), round(r, 3), round(h, 3))
        print(key, line, flush=True)
    n = len(used); ci = P1.ci
    arr = lambda d, k, m=None: np.array(d[k] if m is None else d[k][m])
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
    rows = []
    for v in names:
        res = {"tag": f"{a.tag}_{v}", "variant": v, "experiment": a.tag, "n_slices": n, "slices": used, "pretrain": encs[v][2], "cutoff": str(cutoff),
               "probes": {}, "paired_vs_tb7": {}, "paired_vs_ref": {}, "collapse": {k: ci(x) for k, x in diag[v].items()} if n > 1 else {}}
        for name, k in (("a_tb7", "tb7"), ("b_emb", (v, "emb")), ("c_both", (v, "both"))):
            res["probes"][name] = {**{m: ci(x) for m, x in per[k].items()}, "vol_ridge": ci(volr[k]), "vol_hgb": ci(volh[k]),
                                   "per_slice_auc3": [round(float(x), 4) for x in per[k]["auc3"]],
                                   "per_slice_vol_ridge": [round(float(x), 4) for x in volr[k]], "per_slice_vol_hgb": [round(float(x), 4) for x in volh[k]]}
        for name, k in (("b_emb", (v, "emb")), ("c_both", (v, "both"))):
            res["paired_vs_tb7"][name] = {"auc3": ci(arr(per, k, "auc3") - arr(per, "tb7", "auc3")), "vol_ridge": ci(arr(volr, k) - arr(volr, "tb7")),
                                          "vol_hgb": ci(arr(volh, k) - arr(volh, "tb7"))}
            if a.ref in encs and v != a.ref:
                k1 = (a.ref, k[1])
                res["paired_vs_ref"][name] = {"auc3": ci(arr(per, k, "auc3") - arr(per, k1, "auc3")), "vol_ridge": ci(arr(volr, k) - arr(volr, k1)),
                                              "vol_hgb": ci(arr(volh, k) - arr(volh, k1))}
        open(a.results, "a", encoding="utf-8").write(json.dumps(res) + "\n"); rows.append(res)
    print(f"\n{a.tag}: {n} slices from {[s['train_start'] for s in used]}")
    t = rows[0]["probes"]["a_tb7"]; print(f"tb7: auc {f(t['auc3'])} | ridge {f(t['vol_ridge'])} | hgb {f(t['vol_hgb'])}")
    for r in rows:
        for p in ("b_emb", "c_both"):
            q = r["probes"][p]; d = r["paired_vs_tb7"][p]
            print(f"{r['variant']:6s} {p}: auc {f(q['auc3'])} (d vs tb7 {f(d['auc3'])}) | ridge {f(q['vol_ridge'])} (d {f(d['vol_ridge'])}) | hgb {f(q['vol_hgb'])} (d {f(d['vol_hgb'])})")
            if p in r["paired_vs_ref"]:
                d = r["paired_vs_ref"][p]; print(f"       vs {a.ref}: dAUC {f(d['auc3'])} | d ridge {f(d['vol_ridge'])} | d hgb {f(d['vol_hgb'])}")
        if r["collapse"]:
            c = r["collapse"]; print(f"       collapse: std {c['std_mean'][0]:.3f} min {c['std_min'][0]:.3f} eff_rank {c['eff_rank'][0]:.1f}")


if __name__ == "__main__":
    main()
