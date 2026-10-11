"""Hypothesis #4 (SPEC section 6): rotation or shrinkage of the tb7 edge. Half-year logistic fits 2017H1..2024H1 on the standardised tb7
features (next-open labels), norm and angle, the scaling-model likelihood-ratio test against a parametric bootstrap of pure
shrinkage, a Kalman/RLS logistic and the Nagel recency-weighted momentum baseline. Exploratory; last bar 2025-07-20 (cutoff).
usage: python rotation.py [--workers 6] [--boot 200]"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import shapelib as S

RES = os.path.join(HERE, "results"); GAP = 40; B_FILES = 13
HALVES = [(f"{y}H{h}", pd.Timestamp(f"{y}-{'01' if h == 1 else '07'}-01"), pd.Timestamp(f"{y}-{'07' if h == 1 else '12'}-01") if h == 1 else pd.Timestamp(f"{y + 1}-01-01"))
          for y in range(2017, 2025) for h in (1, 2) if not (y == 2024 and h == 2)]
_G = {}


def prep():
    ds = S.load_dataset("real"); t = pd.DatetimeIndex(ds["t"].astype("datetime64[ns]")); r = ds["rdir"]
    m = np.isfinite(r) & (np.abs(r) > S.DB); y = (r > 0).astype(int)
    tr0 = np.asarray(t < pd.Timestamp("2019-01-01")) & m
    mu = ds["tb7"][tr0].mean(0); sd = ds["tb7"][tr0].std(0) + 1e-9
    X = np.clip((ds["tb7"] - mu) / sd, -8, 8).astype(np.float64)
    return ds, t, m, y, X


def logreg(X, y, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=500).fit(X, y)


def ll(mdl_or_z, y, z=None):
    d = mdl_or_z.decision_function(z) if z is not None else mdl_or_z
    return float(np.sum(y * -np.logaddexp(0, -d) + (1 - y) * -np.logaddexp(0, d)))


def cosang(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))); return c, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def half_data(t, m, X, y):
    out = []
    for name, a, b in HALVES:
        idx = np.flatnonzero(np.asarray((t >= a) & (t < b)) & m); out.append((name, idx))
    return out


def procedure(X, hd, refs, ys, split=True):
    """given labels ys (dict half index -> y array aligned with idx), returns Lambda, per-half held-out LL pairs, full-half fits."""
    lam = 0.0; per = []; betas = []
    for k, (name, idx) in enumerate(hd):
        yk = ys[k]; Xk = X[idx]; n = len(idx); c1 = int(n * 2 / 3)
        free = logreg(Xk[:c1 - GAP], yk[:c1 - GAP], 1.0)
        z = Xk @ refs[k]; sc = logreg(z[:c1 - GAP, None], yk[:c1 - GAP], 1e6)
        llf = ll(free, yk[c1:], Xk[c1:]); lls = ll(sc, yk[c1:], z[c1:, None])
        lam += llf - lls; per.append((llf, lls)); betas.append(logreg(Xk, yk, 1.0))
    return lam, per, betas


def _init(args):
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    _G.update(args)


def _boot(seed):
    rng = np.random.default_rng(seed); X, hd, refs, ps = _G["X"], _G["hd"], _G["refs"], _G["ps"]
    ys = {k: (rng.random(len(idx)) < ps[k]).astype(int) for k, (n, idx) in enumerate(hd)}
    lam, per, betas = procedure(X, hd, refs, ys)
    cons = [cosang(betas[k].coef_[0], betas[k + 1].coef_[0])[0] for k in range(len(hd) - 1)]
    # pooled calendar-2019 fit vs 2024H1
    i19 = [k for k, (n, _) in enumerate(hd) if n.startswith("2019")]
    Xp = np.concatenate([X[hd[k][1]] for k in i19]); yp = np.concatenate([ys[k] for k in i19])
    b19 = logreg(Xp, yp, 1.0).coef_[0]; c19 = cosang(b19, betas[-1].coef_[0])[0]
    return lam, cons, c19, float(np.linalg.norm(b19)), float(np.linalg.norm(betas[-1].coef_[0]))


# ------------------------------------------------------------------ Kalman / RLS logistic and the Nagel baseline
def kalman_run(t, X1, y, m, q, p0=0.01, delay_min=S.HDIR + 1, t_start=pd.Timestamp("2017-01-01")):
    """daily Laplace-EKF update of random-walk logistic coefficients (last column of X1 = constant). Returns predictions (logit) for every row.
    The day's update uses only rows whose label closed (anchor time + 16 min) by the day start."""
    d = X1.shape[1]; beta = np.zeros(d); P = np.eye(d) * p0; pred = np.full(len(t), np.nan)
    tn = t.values.astype("datetime64[ns]").astype(np.int64); day = (tn // (86400 * 10 ** 9)); udays = np.unique(day)
    close_t = tn + delay_min * 60 * 10 ** 9; used = 0; mi = np.flatnonzero(m)
    order = np.argsort(close_t[mi], kind="stable"); ready_idx = mi[order]; ready_t = close_t[ready_idx]
    for dd in udays:
        s = dd * 86400 * 10 ** 9
        hi = np.searchsorted(ready_t, s, side="right")
        if hi > used:
            ix = ready_idx[used:hi]; used = hi
            Xd, yd = X1[ix], y[ix]
            Pp = P + q * np.eye(d); Pi = np.linalg.inv(Pp); b = beta.copy()
            for _ in range(3):
                p = expit(Xd @ b); g = Xd.T @ (yd - p) - Pi @ (b - beta)
                H = (Xd * (p * (1 - p))[:, None]).T @ Xd + Pi
                b = b + np.linalg.solve(H, g)
            p = expit(Xd @ b); H = (Xd * (p * (1 - p))[:, None]).T @ Xd + Pi
            beta, P = b, np.linalg.inv(H)
        rows = np.flatnonzero(day == dd)
        pred[rows] = X1[rows] @ beta
    return pred


def monthly_auc(t, pred, y, m, a, b):
    from sklearn.metrics import roc_auc_score
    out = {}
    for mo in pd.period_range(a, b, freq="M"):
        sel = np.asarray((t >= mo.start_time) & (t < (mo + 1).start_time)) & m & np.isfinite(pred)
        if sel.sum() > 500 and len(np.unique(y[sel])) > 1:
            out[str(mo)] = float(roc_auc_score(y[sel], pred[sel]))
    return out


def paired(a, b):
    ks = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in ks]); n = len(d); se = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
    return {"n_months": n, "mean": float(d.mean()), "se": float(se), "lo95": float(d.mean() - tq * se), "hi95": float(d.mean() + tq * se)}


def nagel_scores(ds, hl):
    df = S.load_bars(); lc = np.log(df["close"].to_numpy()); r = np.concatenate([[0.0], np.diff(lc)])
    s = pd.Series(r).ewm(halflife=hl, adjust=False).mean().to_numpy()
    return s[ds["anch"] - 1]


def main():
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 200
    t0 = time.time(); os.makedirs(RES, exist_ok=True)
    ds, t, m, y, X = prep(); hd = half_data(t, m, X, y); res = {"cutoff": str(S.CUTOFF), "last_anchor": str(t.max())}
    assert t.max() <= S.CUTOFF
    # --- half-year fits
    fits = []
    for name, idx in hd:
        mdl = logreg(X[idx], y[idx], 1.0); b = mdl.coef_[0]
        p = expit(X[idx] @ b + mdl.intercept_[0]); Wt = p * (1 - p)
        H = (X[idx] * Wt[:, None]).T @ X[idx] + np.eye(X.shape[1]); se = np.sqrt(np.diag(np.linalg.inv(H))) * np.sqrt(1.5)     # x1.5: overlap of 15-minute labels at stride 10
        fits.append({"half": name, "n": int(len(idx)), "norm": float(np.linalg.norm(b)), "norm_se_approx": float(np.sqrt((se ** 2).sum())), "beta": b.tolist()})
    B_all = np.array([f["beta"] for f in fits])
    i19 = np.flatnonzero(np.asarray((t >= "2019-01-01") & (t < "2020-01-01")) & m)
    b19 = logreg(X[i19], y[i19], 1.0).coef_[0]
    cos = np.array([[cosang(a, b)[0] for b in B_all] for a in B_all])
    cons = [cosang(B_all[k], B_all[k + 1]) for k in range(len(fits) - 1)]
    last = fits[-1]; c_ref_last = cosang(b19, B_all[-1])
    res["halves"] = [{k: v for k, v in f.items() if k != "beta"} for f in fits]
    res["cos_matrix"] = cos.round(3).tolist(); res["consecutive_cos_deg"] = [{"pair": f"{fits[k]['half']}-{fits[k + 1]['half']}", "cos": c, "deg": d} for k, (c, d) in enumerate(cons)]
    res["norm_2019"] = float(np.linalg.norm(b19)); res["norm_ratio_2024H1_over_2019"] = float(last["norm"] / np.linalg.norm(b19))
    res["cos_2019_2024H1"] = c_ref_last[0]; res["deg_2019_2024H1"] = c_ref_last[1]
    # --- scaling-model LR test
    refs = []
    for k, (name, idx) in enumerate(hd):
        other = np.concatenate([hd[j][1] for j in range(len(hd)) if abs(j - k) > 1]); refs.append(logreg(X[other], y[other], 1.0).coef_[0])
    ys = {k: y[idx] for k, (n, idx) in enumerate(hd)}
    lam, per, betas = procedure(X, hd, refs, ys)
    # null parameters: scaled fit on the whole half (c_h, a_h)
    ps = {}
    for k, (name, idx) in enumerate(hd):
        z = X[idx] @ refs[k]; sc = logreg(z[:, None], y[idx], 1e6); ps[k] = expit(sc.coef_[0][0] * z + sc.intercept_[0])
    res["lambda"] = lam; res["lambda_per_half"] = [{"half": hd[k][0], "ll_free": a, "ll_scaled": b, "diff": a - b} for k, (a, b) in enumerate(per)]
    import multiprocessing as mp
    with mp.Pool(w, initializer=_init, initargs=({"X": X, "hd": hd, "refs": refs, "ps": ps},)) as pool:
        out = pool.map(_boot, range(1000, 1000 + B))
    lams = np.array([o[0] for o in out]); cons_null = np.array([o[1] for o in out]); c19n = np.array([o[2] for o in out])
    res["lambda_null_median"] = float(np.median(lams)); res["lambda_null_p95"] = float(np.percentile(lams, 95)); res["lambda_p"] = float((1 + (lams >= lam).sum()) / (B + 1)); res["B"] = B
    obs_c = [c for c, d in cons]
    res["consecutive_cos_null"] = [{"pair": f"{fits[k]['half']}-{fits[k + 1]['half']}", "obs": obs_c[k], "null_median": float(np.median(cons_null[:, k])),
                                    "null_p05": float(np.percentile(cons_null[:, k], 5))} for k in range(len(fits) - 1)]
    res["cos_2019_2024H1_null"] = {"obs": c_ref_last[0], "median": float(np.median(c19n)), "p05": float(np.percentile(c19n, 5)), "p95": float(np.percentile(c19n, 95))}
    # --- adaptation (Kalman) and Nagel; static baseline = the monthly 365-day refit AUCs of the shape selection run
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    qs = [1e-8, 1e-7, 1e-6, 1e-5]; tune = {}
    for q in qs:
        pr = kalman_run(t, X1, y, m, q); tune[q] = float(np.mean(list(monthly_auc(t, pr, y, m, "2019-01", "2020-12").values())))
    qb = max(tune, key=tune.get); pr = kalman_run(t, X1, y, m, qb)
    # Nagel: half-life and sign on 2019-2020
    nt = {}
    for hl in (5, 15, 30, 60):
        sc = nagel_scores(ds, hl); a = np.mean(list(monthly_auc(t, sc, y, m, "2019-01", "2020-12").values())); nt[hl] = float(a)
    hb = max(nt, key=lambda h: abs(nt[h] - 0.5)); sg = 1.0 if nt[hb] >= 0.5 else -1.0
    nag = monthly_auc(t, sg * nagel_scores(ds, hb), y, m, "2021-01", "2024-06")
    kal = monthly_auc(t, pr, y, m, "2021-01", "2024-06")
    stat = {r["month"]: r["auc_tb7"] for r in map(json.loads, open(f"{RES}/sel.jsonl", encoding="utf-8")) if "2021-01" <= r["month"] <= "2024-06"}
    res["kalman_q_tuning_auc_2019_2020"] = {str(k): v for k, v in tune.items()}; res["kalman_q"] = qb
    res["nagel_tuning_auc_2019_2020"] = {str(k): v for k, v in nt.items()}; res["nagel_halflife"] = hb; res["nagel_sign"] = sg
    res["auc_mean_2021_2024H1"] = {"static": float(np.mean(list(stat.values()))), "kalman": float(np.mean(list(kal.values()))), "nagel": float(np.mean(list(nag.values())))}
    res["paired_kalman_minus_static"] = paired(kal, stat); res["paired_nagel_minus_static"] = paired(nag, stat); res["paired_kalman_minus_nagel"] = paired(kal, nag)
    # --- verdict
    shr_geom = res["norm_ratio_2024H1_over_2019"] <= 0.5 and res["cos_2019_2024H1"] >= 0.8
    g = res["paired_kalman_minus_static"]
    shrink = shr_geom and g["mean"] <= 0.002 and res["lambda_p"] >= 0.05
    rotate = (g["mean"] >= 0.005 and g["lo95"] > 0 and res["paired_kalman_minus_nagel"]["lo95"] > 0) or (res["lambda_p"] < 0.05 and res["cos_2019_2024H1"] < 0.8)
    res["verdict"] = {"geometry_prediction_holds": bool(shr_geom), "shrinkage": bool(shrink), "rotation": bool(rotate),
                      "label": "shrinkage" if shrink else ("rotation" if rotate else "inconclusive")}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{RES}/rotation.json", "w"), indent=1)
    L = [f"== #4 rotation vs shrinkage ({res['seconds']}s, B={B}); last anchor {res['last_anchor']}",
         f"{'half':7s} {'n':>7s} {'norm':>7s} {'~SE(norm)':>9s}"] + [f"{f['half']:7s} {f['n']:7d} {f['norm']:7.4f} {f['norm_se_approx']:9.4f}" for f in res["halves"]]
    L += [f"norm(2019)={res['norm_2019']:.4f}; norm ratio 2024H1/2019 = {res['norm_ratio_2024H1_over_2019']:.3f}; cos(beta_2019, beta_2024H1) = {res['cos_2019_2024H1']:.3f} "
          f"({res['deg_2019_2024H1']:.1f} deg); pure-shrinkage null for that cosine: median {res['cos_2019_2024H1_null']['median']:.3f}, 90% [{res['cos_2019_2024H1_null']['p05']:.3f}, {res['cos_2019_2024H1_null']['p95']:.3f}]",
          "consecutive cosines (observed | null median, null 5%):"]
    L += [f"  {c['pair']:15s} {c['obs']:+.3f} | {c['null_median']:+.3f}, {c['null_p05']:+.3f}" for c in res["consecutive_cos_null"]]
    L += [f"LR test Lambda = {lam:+.2f} (null median {res['lambda_null_median']:+.2f}, p95 {res['lambda_null_p95']:+.2f}), p = {res['lambda_p']:.3f}",
          f"Kalman q = {qb} (tuning {tune}); Nagel half-life {hb} min sign {sg:+.0f} (tuning {nt})",
          f"mean monthly AUC 2021-01..2024-06: {res['auc_mean_2021_2024H1']}",
          f"Kalman - static: {res['paired_kalman_minus_static']}", f"Nagel - static: {res['paired_nagel_minus_static']}", f"Kalman - Nagel: {res['paired_kalman_minus_nagel']}",
          f"VERDICT: {res['verdict']}"]
    open(f"{RES}/rotation_table.txt", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
