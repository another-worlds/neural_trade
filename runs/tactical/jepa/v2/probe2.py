"""v2 probes on the frozen JEPA context embeddings, same protocol as v1 (probe.py) on the same 14 slices (cached lab slices whose
train block starts after 2020-01-02): per slice fit on the train block, score on the val block.
  direction: logreg C 0.1 on (a) tb7 (b) embedding (c) tb7 + embedding, lab.evaluate_slice  (identical to v1)
  volatility: Spearman between the prediction of log|r_h1| and the realised |r_h1| on val, readouts
     ridge (v1's)   : tb7 / emb / both
     boosting (new) : HistGradientBoostingRegressor on tb7 / emb / both
usage: python probe2.py --variants v1,ctl,a,b,c  (CPU)   One results row per variant is appended to results.jsonl."""
import argparse, glob, json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "lab"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import probe as P1   # noqa: E402  v1's locate / embed / standardise / logit_probs / ci / file_bars (unchanged)

FEATS = ("tb7", "emb", "both")


def eff_rank_np(z):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False); p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def load_enc(variant):
    d = f"{HERE}/ckpt/{variant}"; meta = json.load(open(d + "/meta.json"))
    from channels import N_CH, Standardiser, CTX
    std = Standardiser(np.array(meta["std"]["mean"]), np.array(meta["std"]["std"]))
    if variant == "v1":
        import model as M; enc = M.Encoder(N_CH)
    else:
        import model2 as M2; enc = M2.Encoder2(N_CH)
    enc(np.zeros((2, CTX, N_CH), np.float32)); enc.load_weights(d + "/enc.h5")
    return enc, std, meta


def vol_scores(Ft, Fv, rtr, rva):
    """ridge and boosting readouts for the volatility probe (standardised features, same target as v1)."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    yt = np.log(np.abs(rtr[:, 1]) + 1e-5); yv = np.abs(rva[:, 1])
    rid = stats.spearmanr(Ridge(alpha=1.0).fit(Ft, yt).predict(Fv), yv).correlation
    hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                        l2_regularization=1.0, early_stopping=False, random_state=0).fit(Ft, yt)
    return float(rid), float(stats.spearmanr(hgb.predict(Fv), yv).correlation)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", default="v1,ctl,a,b,c"); ap.add_argument("--results", default=HERE + "/results.jsonl")
    ap.add_argument("--max_slices", type=int, default=99)
    a = ap.parse_args(); variants = a.variants.split(",")
    import neural_trade  # noqa: F401
    import lab
    from sklearn.preprocessing import StandardScaler
    encs = {v: load_enc(v) for v in variants}
    ts, close = P1.file_bars(); rng = np.random.default_rng(0)
    # per[key] = {metric: [per slice]} ; key = "tb7" or (variant, "emb"/"both")
    per, volr, volh, used = {}, {}, {}, []
    diag = {v: {"std_mean": [], "std_min": [], "eff_rank": []} for v in variants}
    for path in sorted(glob.glob(P1.CACHE)):
        D = np.load(path); i = P1.locate(close, D["Wtr"])
        if i is None or ts[i] < P1.CUTOFF: continue
        if len(used) >= a.max_slices: break
        key = str(D["data_end"])[:13]; used.append({"data_end": key, "train_start": str(ts[i])[:16]}); db = float(D["deadband"])
        T_tb, V_tb = np.nan_to_num(lab.fs_tb7(D["Wtr"])), np.nan_to_num(lab.fs_tb7(D["Wva"]))
        sets = {"tb7": (T_tb, V_tb)}
        for v, (enc, std, _) in encs.items():
            T_em, V_em = P1.embed(enc, std, D["Wtr"]), P1.embed(enc, std, D["Wva"])
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
    for v in variants:
        res = {"tag": f"jepa2_{v}", "variant": v, "n_slices": n, "slices": used, "pretrain": encs[v][2], "probes": {}, "paired_vs_tb7": {},
               "paired_vs_v1": {}, "collapse": {k: ci(x) for k, x in diag[v].items()}}
        for name, k in (("a_tb7", "tb7"), ("b_emb", (v, "emb")), ("c_both", (v, "both"))):
            res["probes"][name] = {**{m: ci(x) for m, x in per[k].items()}, "vol_ridge": ci(volr[k]), "vol_hgb": ci(volh[k]),
                                   "per_slice_auc3": [round(float(x), 4) for x in per[k]["auc3"]],
                                   "per_slice_vol_ridge": [round(float(x), 4) for x in volr[k]], "per_slice_vol_hgb": [round(float(x), 4) for x in volh[k]]}
        for name, k in (("b_emb", (v, "emb")), ("c_both", (v, "both"))):
            res["paired_vs_tb7"][name] = {"auc3": ci(arr(per, k, "auc3") - arr(per, "tb7", "auc3")), "hon10_bps": ci(arr(per, k, "hon10_bps") - arr(per, "tb7", "hon10_bps")),
                                          "vol_ridge": ci(arr(volr, k) - arr(volr, "tb7")), "vol_hgb": ci(arr(volh, k) - arr(volh, "tb7"))}
            if "v1" in variants and v != "v1":
                k1 = ("v1", k[1])
                res["paired_vs_v1"][name] = {"auc3": ci(arr(per, k, "auc3") - arr(per, k1, "auc3")),
                                             "vol_ridge": ci(arr(volr, k) - arr(volr, k1)), "vol_hgb": ci(arr(volh, k) - arr(volh, k1))}
        open(a.results, "a", encoding="utf-8").write(json.dumps(res) + "\n")
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
    print(f"\n{n} slices; mean3 AUC / vol Spearman ridge / vol Spearman boosting")
    r0 = json.loads(open(a.results).readlines()[-len(variants)])
    t = r0["probes"]["a_tb7"]; print(f"tb7: auc {f(t['auc3'])} | ridge {f(t['vol_ridge'])} | hgb {f(t['vol_hgb'])}")
    for v in variants:
        rr = [json.loads(l) for l in open(a.results)][-len(variants):]; r = [x for x in rr if x["variant"] == v][0]
        for p in ("b_emb", "c_both"):
            q = r["probes"][p]; d = r["paired_vs_tb7"][p]
            print(f"{v:4s} {p}: auc {f(q['auc3'])} (d {f(d['auc3'])}) | ridge {f(q['vol_ridge'])} (d {f(d['vol_ridge'])}) | hgb {f(q['vol_hgb'])} (d {f(d['vol_hgb'])})")
        print(f"     collapse: std {r['collapse']['std_mean'][0]:.3f} min {r['collapse']['std_min'][0]:.3f} eff_rank {r['collapse']['eff_rank'][0]:.1f}")


if __name__ == "__main__":
    main()
