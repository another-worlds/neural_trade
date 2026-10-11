"""Novel geometry hypotheses N1-N7: the 24-slice evaluation (SPEC.md in this folder; fixed before any run).
usage: python run.py slices [--workers 4]    one part per slice -> parts/<slice>.json (resumable)
       python run.py report                  tables + results.jsonl + report.txt (Holm, vol read, honest tail)
Stage 2 (1-year spans) and the FINAL check are in long.py / final.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, glob, json, math, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lab"))
import features as F  # noqa: E402
import lab  # noqa: E402  (chdirs to this worktree's root)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PARTS = f"{HERE}/parts"; GROUPS = F.FAMILIES + ["ALL"]; CLIP = 8.0
VOL = os.path.join(os.path.dirname(HERE), "vol", "results.jsonl")


def lowprio():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def level_feats(W):
    """longtrain.level_feats: the volatility level the scale-free sets lack (log sd/close, log mean range/close, log1p mean volume)."""
    o, h, l, c, v, d, sd = lab.parts(W)
    return np.stack([np.log(sd / c[:, -1] + 1e-12), np.log(np.mean(h - l, 1) / c[:, -1] + 1e-12), np.log1p(v.mean(1))], 1)


def group_cols(fam, group):
    return np.hstack([fam[g] for g in (F.FAMILIES if group == "ALL" else [group])])


def fit_matrix(blocks_tr, blocks_va, clip_from):
    """Standardise on the train block; columns from `clip_from` on (the new family columns) are clipped to +-8 sd."""
    Xtr, Xva = np.hstack(blocks_tr).astype(np.float64), np.hstack(blocks_va).astype(np.float64)
    sc = StandardScaler().fit(Xtr); Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    Xtr[:, clip_from:] = np.clip(Xtr[:, clip_from:], -CLIP, CLIP); Xva[:, clip_from:] = np.clip(Xva[:, clip_from:], -CLIP, CLIP)
    return Xtr, Xva


def predict(Xtr, Xva, D, kind):
    db = float(D["deadband"]); P = np.zeros((len(Xva), 3))
    for i in range(3):
        rtr = D["rtr"][:, i]; m = np.abs(rtr) > db
        P[:, i] = lab.model_of(kind, 0.1).fit(Xtr[m], (rtr[m] > 0).astype(int)).predict_proba(Xva)[:, 1]
    return P


def ridge_vol(Xtr, Xva, rtr):
    """Ridge on log(|r| + 1e-5) per horizon, alpha on the last 15% of train (80 bars apart): volbench.ridge_forecast's rule."""
    Y = np.log(np.abs(rtr) + 1e-5); n = len(Xtr); cut = int(round(0.85 * n)); P = np.zeros((len(Xva), 3))
    for h in range(3):
        err = min((float(np.mean((Ridge(alpha=a).fit(Xtr[:cut - 80], Y[:cut - 80, h]).predict(Xtr[cut:]) - Y[cut:, h]) ** 2)), a)
                  for a in (1.0, 100.0, 1e4))
        P[:, h] = Ridge(alpha=err[1]).fit(Xtr, Y[:, h]).predict(Xva)
    return P


def one_slice(path):
    lowprio(); t0 = time.time(); D = np.load(path); name = os.path.basename(path)[:-4]
    Wtr, Wva = D["Wtr"], D["Wva"]
    tb_tr, tb_va = np.nan_to_num(lab.fs_tb7(Wtr)), np.nan_to_num(lab.fs_tb7(Wva))
    ftr, fva = F.compute(Wtr), F.compute(Wva)
    lv_tr, lv_va = np.nan_to_num(level_feats(Wtr)), np.nan_to_num(level_feats(Wva))
    t_feat = time.time() - t0
    Dd = dict(deadband=D["deadband"], rtr=D["rtr"], rva=D["rva"]); rva = D["rva"].astype(np.float64)
    out = dict(name=name, auc={}, vol={}, n_train=int(len(Wtr)), n_val=int(len(Wva)), t_feat=round(t_feat, 1))
    cfgs = [("tb7", None, False)] + [(g, g, a) for g in GROUPS for a in (True, False)]    # (label, group, alone)
    for label, g, alone in cfgs:
        if g is None:
            Xtr, Xva = fit_matrix([tb_tr], [tb_va], 10 ** 6); tag = "tb7"
        elif alone:
            Xtr, Xva = fit_matrix([group_cols(ftr, g)], [group_cols(fva, g)], 0); tag = f"{g}_alone"
        else:
            Xtr, Xva = fit_matrix([tb_tr, group_cols(ftr, g)], [tb_va, group_cols(fva, g)], tb_tr.shape[1]); tag = f"tb7+{g}"
        for kind, short in (("logreg", "lr"), ("hgb", "hgb")):
            t1 = time.time(); P = predict(Xtr, Xva, Dd, kind)
            r = lab.evaluate_slice(Dd, P, np.random.default_rng(0))
            out["auc"][f"{tag}|{short}"] = {k: float(v) for k, v in r.items()} | {"sec": round(time.time() - t1, 1)}
    # the volatility read: ridge on tb7 + family (SPEC), and with the three level features (extra, labelled)
    vcfg = [("tb7", [tb_tr], [tb_va], 10 ** 6), ("tb7+lev", [tb_tr, lv_tr], [tb_va, lv_va], tb_tr.shape[1])]
    for g in GROUPS:
        vcfg.append((f"tb7+{g}", [tb_tr, group_cols(ftr, g)], [tb_va, group_cols(fva, g)], tb_tr.shape[1]))
        vcfg.append((f"tb7+lev+{g}", [tb_tr, lv_tr, group_cols(ftr, g)], [tb_va, lv_va, group_cols(fva, g)], tb_tr.shape[1]))
    for tag, btr, bva, cf in vcfg:
        Xtr, Xva = fit_matrix(btr, bva, cf); P = ridge_vol(Xtr, Xva, D["rtr"].astype(np.float64))
        out["vol"][tag] = float(np.mean([stats.spearmanr(P[:, h], np.abs(rva[:, h]))[0] for h in range(3)]))
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(PARTS, exist_ok=True); json.dump(out, open(f"{PARTS}/{name}.json", "w"))
    print(f"{name}: {out['seconds']} s (features {t_feat:.0f} s)", flush=True)
    return name


def cmd_slices(workers):
    from concurrent.futures import ProcessPoolExecutor
    paths = [p for p in sorted(glob.glob(lab.CACHE)) if not os.path.exists(f"{PARTS}/{os.path.basename(p)[:-4]}.json")]
    print(len(paths), "slices to do", flush=True); t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(one_slice, paths))
    print(f"all done in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ report
T975 = lambda n: float(stats.t.ppf(0.975, n - 1))  # noqa: E731


def tci(v):
    v = np.asarray(v, float); m = float(v.mean()); hw = T975(len(v)) * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - hw, m + hw


def paired(a, b):
    """mean, CI low, CI high, two-sided t p of the paired differences a - b over slices."""
    d = np.asarray(a) - np.asarray(b); m, lo, hi = tci(d); t = m / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-300)
    return m, lo, hi, float(2 * stats.t.sf(abs(t), len(d) - 1))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def load_parts():
    P = [json.load(open(p)) for p in sorted(glob.glob(f"{PARTS}/*.json"))]
    return P


def har_tod():
    out = {}
    for line in open(VOL, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "unit" and r.get("setup") == "minute" and r.get("forecast") == "har_tod":
            out[r["unit"]] = r["spearman"]
    return out


def cmd_report():
    P = load_parts(); names = [p["name"] for p in P]; n = len(P); lines = []
    pr = lambda s="": lines.append(s)  # noqa: E731
    cfg = lambda k, m="auc3": np.array([p["auc"][k][m] for p in P])  # noqa: E731
    base_lr, base_hgb = cfg("tb7|lr"), cfg("tb7|hgb")
    pr(f"Novel geometry N1-N7: {n} slices (mean of 3 horizons' direction AUC, paired per slice)")
    pr(f"tb7 logistic {tci(base_lr)[0]:.4f} [{tci(base_lr)[1]:.4f},{tci(base_lr)[2]:.4f}]; tb7 boosting {tci(base_hgb)[0]:.4f} "
       f"[{tci(base_hgb)[1]:.4f},{tci(base_hgb)[2]:.4f}]; boosting minus logistic {paired(base_hgb, base_lr)[0]:+.4f}")
    rows = []
    for g in GROUPS:
        for short in ("lr", "hgb"):
            for kind, key in (("tb7+family", f"tb7+{g}|{short}"), ("alone", f"{g}_alone|{short}")):
                a = cfg(key); m, lo, hi, p = paired(a, base_lr); mr, lor, hir, pr_ = paired(a, base_hgb if short == "hgb" else base_lr)
                rows.append(dict(group=g, readout=short, kind=kind, auc=float(a.mean()), d=m, lo=lo, hi=hi, p=p, d_same=mr, lo_same=lor,
                                 hi_same=hir, key=key, per_slice=[round(float(x), 5) for x in a]))
    for kind in ("tb7+family", "alone"):
        sel = [r for r in rows if r["kind"] == kind]; adj = holm(np.array([r["p"] for r in sel]))
        for r, a in zip(sel, adj):
            r["p_holm"] = float(a)
    pr(); pr("PRIMARY: tb7+family, paired mean3 AUC difference vs tb7 LOGISTIC (16 tests: 8 groups x 2 readouts; Holm over the 16)")
    pr(f"{'group':5s} {'readout':7s} {'AUC':>7s} {'diff vs tb7-lr [95% CI]':>30s} {'p':>8s} {'p_holm':>8s} {'adds':>5s} | {'diff vs same-readout tb7 [CI]':>32s}")
    for r in rows:
        if r["kind"] != "tb7+family":
            continue
        adds = r["p_holm"] < 0.05 and r["d"] > 0
        pr(f"{r['group']:5s} {r['readout']:7s} {r['auc']:7.4f} {r['d']:+8.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:8.4f} {r['p_holm']:8.4f} "
           f"{'YES' if adds else 'no':>5s} | {r['d_same']:+8.4f} [{r['lo_same']:+.4f},{r['hi_same']:+.4f}]")
        r["adds"] = bool(adds)
    pr(); pr("DESCRIPTIVE: family alone (no tb7), vs tb7 logistic; Holm over its own 16")
    for r in rows:
        if r["kind"] == "alone":
            pr(f"{r['group']:5s} {r['readout']:7s} {r['auc']:7.4f} {r['d']:+8.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p {r['p']:.4f} p_holm {r['p_holm']:.4f}")
    # stage 2 trigger: boosting readout of tb7+family gains more than +0.003 over tb7 boosting
    trig = [r["group"] for r in rows if r["kind"] == "tb7+family" and r["readout"] == "hgb" and r["d_same"] > 0.003]
    pr(); pr(f"Step (c) rule (tb7+family boosting minus tb7 boosting > +0.003): {trig if trig else 'no family'}")
    # honest tail
    pr(); pr("HONEST TAIL (top 10% |P-0.5|, thresholds on the first half of val, scored on the second after a 20-bar gap), mean over slices:")
    pr(f"{'config':18s} {'hit10':>7s} {'bps10':>8s} {'null95':>8s} {'hit5':>7s} {'bps5':>8s} {'Brier h1':>9s} {'ll h1':>7s} (const {np.mean([p['auc']['tb7|lr']['ll_const_h1'] for p in P]):.4f})")
    for key in ["tb7|lr", "tb7|hgb"] + [f"tb7+{g}|{s}" for g in GROUPS for s in ("lr", "hgb")]:
        f = lambda m: np.mean([p["auc"][key][m] for p in P])  # noqa: E731
        pr(f"{key:18s} {f('hon10_hit'):7.4f} {f('hon10_bps'):+8.2f} {f('hon10_null95'):+8.2f} {f('hon5_hit'):7.4f} {f('hon5_bps'):+8.2f} {f('brier_h1'):9.4f} {f('ll_h1'):7.4f}")
    # volatility read
    ht = har_tod(); have = [p for p in P if p["name"] in ht]
    pr(); pr(f"VOLATILITY READ: ridge on log|r| (11.5-day train), Spearman with |r| (mean of 3 horizons), {len(have)} slices; HAR+tod is the 1-year-trained baseline")
    h = np.array([ht[p["name"]] for p in have]); tbv = np.array([p["vol"]["tb7"] for p in have])
    pr(f"HAR+tod {h.mean():.4f}; ridge tb7 {tbv.mean():.4f}; ridge tb7+level {np.mean([p['vol']['tb7+lev'] for p in have]):.4f}")
    pr(f"{'variant':14s} {'Spearman':>8s} {'d vs HAR+tod [95% CI]':>28s} {'d vs ridge tb7 (same level feats) [CI]':>42s}")
    vrows = []
    for g in GROUPS:
        for pre, ref in (("tb7+", "tb7"), ("tb7+lev+", "tb7+lev")):
            k = f"{pre}{g}"; a = np.array([p["vol"][k] for p in have]); m1 = paired(a, h); m2 = paired(a, np.array([p["vol"][ref] for p in have]))
            pr(f"{k:14s} {a.mean():8.4f} {m1[0]:+8.4f} [{m1[1]:+.4f},{m1[2]:+.4f}] {m2[0]:+12.4f} [{m2[1]:+.4f},{m2[2]:+.4f}]")
            vrows.append(dict(variant=k, spearman=float(a.mean()), d_har=m1[:3], d_ref=m2[:3], per_slice=[round(float(x), 5) for x in a]))
    txt = "\n".join(lines); print(txt)
    open(f"{HERE}/report_stage1.txt", "w", encoding="utf-8").write(txt + "\n")
    with open(f"{HERE}/results.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dict(stage="slices24", **r, slices=names)) + "\n")
        for r in vrows:
            fh.write(json.dumps(dict(stage="vol24", **r, slices=[p["name"] for p in have])) + "\n")
        fh.write(json.dumps(dict(stage="slices24_summary", step_c_families=trig, tb7_lr=float(base_lr.mean()), tb7_hgb=float(base_hgb.mean()),
                                 adds=[r["group"] + "|" + r["readout"] for r in rows if r.get("adds")])) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["slices", "report"]); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "slices":
        cmd_slices(a.workers)
    else:
        cmd_report()
