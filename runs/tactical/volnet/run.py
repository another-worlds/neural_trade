"""volnet walk-forward (SPEC.md). usage: python run.py --stage sel|look --part 0 --nparts 2 [--models har,hgb,mlp,seq,ens] [--cpu]
Writes results_<stage>_<part>.jsonl (one row per month x horizon x model) and meta_<stage>_<part>.jsonl (times, params, epochs)."""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import volnet_lib as V  # noqa: E402

STAGES = {"sel": ("2021-01", "2024-06"), "look": ("2024-07", "2025-07")}


def month_models(D, m, want, dev_log):
    m0, m1 = m.start_time, (m + 1).start_time; Hs = D.horizons; nH = len(Hs); Hmax = max(Hs)
    # the training rows must satisfy the longest horizon's rule so one row set serves all horizons (the strictest)
    tr, te = V.month_masks(D.ta, m0, m1, Hmax)
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    if len(tr_idx) < 5000 or len(te_idx) < 500:
        return None
    fit, ho = V.fit_hold(tr_idx, Hmax)
    Y, absr, v2 = D.y, D.absr, D.v2
    Xf_all = D.vec_full(); Xh = D.har
    out, meta = {}, {"month": str(m), "n_fit": len(fit), "n_hold": len(ho), "n_test": len(te_idx)}
    t0 = time.time()
    # 1. HAR ridge
    mean, std, coef, ic = V.ridge_fit(Xh[fit], Y[fit])
    pred = lambda X, idx: ((X[idx] - mean) / std) @ coef.T + ic
    out["har"] = (pred(Xh, te_idx), pred(Xh, ho)); meta["t_har"] = time.time() - t0
    # diagnostic ridge on the full vector
    fm, fs, fc, fi = V.ridge_fit(Xf_all[fit], Y[fit])
    predf = lambda idx: ((Xf_all[idx] - fm) / fs) @ fc.T + fi
    out["ridge_full"] = (predf(te_idx), predf(ho))
    seed = (m.year - 2000) * 12 + m.month
    if "hgb" in want or "ens" in want:
        from sklearn.ensemble import HistGradientBoostingRegressor
        t0 = time.time(); pt, ph, its = np.zeros((len(te_idx), nH)), np.zeros((len(ho), nH)), []
        for j in range(nH):
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=200, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed).fit(Xf_all[fit], Y[fit, j])
            errs = [np.mean((Y[ho, j] - p) ** 2) for p in g.staged_predict(Xf_all[ho])]
            k = int(np.argmin(errs)) + 1; its.append(k)
            for i, p in enumerate(g.staged_predict(Xf_all[te_idx])):
                if i + 1 == k:
                    pt[:, j] = p; break
            for i, p in enumerate(g.staged_predict(Xf_all[ho])):
                if i + 1 == k:
                    ph[:, j] = p; break
        out["hgb"] = (pt, ph); meta["t_hgb"] = time.time() - t0; meta["hgb_iters"] = its
    if "mlp" in want or "ens" in want:
        t0 = time.time(); tf = V.get_tf(); tf.keras.backend.clear_session()
        net = V.build_net("mlp", Xf_all.shape[1], Xh.shape[1], nH)
        # skip works on standardised inputs: feed standardised vectors
        Z = (Xf_all - fm) / fs; V.set_skip(net, fc, fi)
        r = V.train_net(net, [Z[fit]], Y[fit], [Z[ho]], Y[ho], seed)
        out["mlp"] = (r["predict"]([Z[te_idx]]), r["predict"]([Z[ho]]))
        meta.update(t_mlp=time.time() - t0, mlp_params=V.n_params(net), mlp_best_epoch=r["best_epoch"], mlp_epochs=len(r["hold_mse_by_epoch"]) - 1)
    if "seq" in want or "ens" in want:
        t0 = time.time(); tf = V.get_tf(); tf.keras.backend.clear_session()
        net = V.build_net("seq", Xh.shape[1], Xh.shape[1], nH); Zh = (Xh - mean) / std; V.set_skip(net, coef, ic)
        S = lambda idx: D.seq(idx)
        Sf, Sh_, St = S(fit), S(ho), S(te_idx)
        r = V.train_net(net, [Zh[fit], Sf], Y[fit], [Zh[ho], Sh_], Y[ho], seed)
        out["seq"] = (r["predict"]([Zh[te_idx], St]), r["predict"]([Zh[ho], Sh_]))
        meta.update(t_seq=time.time() - t0, seq_params=V.n_params(net), seq_best_epoch=r["best_epoch"], seq_epochs=len(r["hold_mse_by_epoch"]) - 1)
        del Sf, Sh_, St
    if "ens" in want:
        parts = [k for k in ("hgb", "mlp", "seq") if k in out]
        out["ens"] = (np.mean([out[k][0] for k in parts], 0), np.mean([out[k][1] for k in parts], 0))
    rows = []
    for name, (mt, mh) in out.items():
        if name not in want and name not in ("har", "ridge_full"):
            continue
        for j, H in enumerate(Hs):
            r_ = V.metrics(mt[:, j], mh[:, j], absr[te_idx, j], v2[te_idx, j], v2[ho, j])
            rows.append({"month": str(m), "H": H, "model": name, **r_, "n_te": len(te_idx)})
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sel"); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--models", default="har,hgb,mlp,seq,ens"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--months", default=None, help="override, e.g. 2023-01:2023-03"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); want = set(a.models.split(","))
    dev = V.setup_device(4000, a.cpu); print("device", dev, flush=True)
    lo, hi = STAGES[a.stage] if not a.months else a.months.split(":")
    months = list(pd.period_range(lo, hi, freq="M"))[a.part::a.nparts]
    s = V.load_bars(); D = V.Data(s.values.astype(np.float64), s.index)
    print(f"anchors {len(D.anch)}, vec {D.vec_full().shape[1]}, months {len(months)}", flush=True)
    rp = f"{HERE}/results_{a.stage}{a.tag}_{a.part}.jsonl"; mp = f"{HERE}/meta_{a.stage}{a.tag}_{a.part}.jsonl"
    done = set(json.loads(l)["month"] for l in open(mp)) if os.path.exists(mp) else set()
    for m in months:
        if str(m) in done:
            continue
        t0 = time.time(); res = month_models(D, m, want, dev)
        if res is None:
            print(m, "skipped", flush=True); continue
        rows, meta = res; meta["t_total"] = time.time() - t0
        with open(rp, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(mp, "a") as f:
            f.write(json.dumps(meta) + "\n")
        q = {(r["model"], r["H"]): r["spearman"] for r in rows}
        print(m, f"{meta['t_total']:.0f}s", {k: round(v, 3) for k, v in q.items() if k[1] == 15}, flush=True)


if __name__ == "__main__":
    main()
