"""Analysis of the arrow study against SPEC.md (CPU). usage: python analyze.py -> results.json, results.md, arrow.png
Pure helpers (edges_for, spearman_detrended, ...) are tested in test_arrow.py."""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arrowlib as AL

RES = f"{HERE}/results"
AUC_FIRST, AUC_LAST = "2021-01", "2024-06"      # exploratory selection months of H56/H63/H64 (tb7 AUC, gates)
EP_FIRST, EP_LAST = "2020-07", "2025-06"
EDGE_MONTHS = 6                                   # gate quintile edges: the 6 preceding months' own out-of-sample scores
NBIN = 5
MIN_BIN = 150
TRAIN_DAYS = 365
SEEDS = (0, 1)
BOOT = 2000


def tci(x, level=0.95):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    m = x.mean(); h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h), n


# ---------------------------------------------------------------- causal gate helpers (tested)
def edges_for(month, month_of_row, values, nbin=NBIN, k=EDGE_MONTHS):
    """quantile edges of `values` over the rows of the k months strictly before `month` (nothing from `month` or later)."""
    m = pd.Period(month, "M"); lo, hi = m - k, m
    mo = np.asarray(month_of_row)
    sel = (mo >= lo.ordinal) & (mo < hi.ordinal)
    v = np.asarray(values)[sel]; v = v[np.isfinite(v)]
    if len(v) < 1000:
        return None
    return np.quantile(v, np.linspace(0, 1, nbin + 1)[1:-1])


def assign_bins(values, edges):
    return np.searchsorted(edges, values, side="right")


def spearman_detrended(a, b, years, kind):
    """kind 'diff': first differences; 'dev': deviation from the calendar-year mean. a, b monthly series, years the month's year."""
    a, b, years = (np.asarray(x, float) for x in (a, b, years))
    if kind == "diff":
        a, b = np.diff(a), np.diff(b)
    else:
        a, b = a.copy(), b.copy()
        for y in np.unique(years):
            m = years == y
            a[m] -= a[m].mean(); b[m] -= b[m].mean()
    return float(stats.spearmanr(a, b).correlation), a, b


def block_boot_spearman(a, b, block=3, reps=BOOT, seed=0):
    rs = np.random.RandomState(seed); n = len(a); out = []
    nb = int(np.ceil(n / block))
    for _ in range(reps):
        st = rs.randint(0, n - block + 1, nb); ix = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        out.append(stats.spearmanr(a[ix], b[ix]).correlation)
    out = np.array(out); out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def partial_spearman(a, b, c):
    ra, rb, rc = (stats.rankdata(x) for x in (a, b, c))
    res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
    return float(stats.pearsonr(res(ra, rc), res(rb, rc))[0])


# ---------------------------------------------------------------- loading
def load_cache():
    d = {k: np.load(f"{AL.CACHE}/{k}.npy", mmap_mode="r" if k == "X" else None) for k in ("X", "sig", "t", "pos", "idx", "r_no")}
    d["month"] = pd.DatetimeIndex(d["t"].astype("datetime64[ns]")).to_period("M").asi8
    d["day"] = pd.DatetimeIndex(d["t"].astype("datetime64[ns]")).normalize().asi8
    return d


def load_ep(n_rows):
    """loss[model][seed] -> float32 [n_rows] per-window loss (NaN where the row was not scored); score[model]: antisymmetrised log-odds
    (zf - zr)/2 averaged over seeds."""
    loss, score = {}, {}
    for mn in AL.MODELS:
        loss[mn] = {}
        sc_sum = np.zeros(n_rows); sc_n = np.zeros(n_rows)
        for s in SEEDS:
            L = np.full(n_rows, np.nan, np.float32)
            for f in sorted(glob.glob(f"{RES}/{mn}_s{s}_*.npz")):
                z = np.load(f)
                if "skipped" in z.files:
                    continue
                rows = z["rows"]; zf, zr = z["zf"].astype(np.float64), z["zr"].astype(np.float64)
                L[rows] = (0.5 * (AL.bce_logits(zf, 1.0) + AL.bce_logits(zr, 0.0))).astype(np.float32)
                sc_sum[rows] += 0.5 * (zf - zr); sc_n[rows] += 1
            loss[mn][s] = L
        score[mn] = np.where(sc_n > 0, sc_sum / np.maximum(sc_n, 1), np.nan)
    return loss, score


# ---------------------------------------------------------------- tb7 walk-forward with per-row predictions
def walk_forward_p(X, t_ns, r, first, last, extra_mask=None, kind="logreg", C=0.1):
    """P(up) for the rows of each month in [first, last], model trained on the T days before it (labels closed by the month start).
    Returns p [N] (NaN outside), masked (|r| > deadband)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t = pd.DatetimeIndex(t_ns.astype("datetime64[ns]")); mask = np.abs(r) > AL.DEADBAND; y = (r > 0).astype(int)
    p = np.full(len(r), np.nan)
    for m in pd.period_range(first, last, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        lab_end = t + pd.Timedelta(minutes=AL.H + 1)
        tr = np.asarray((t >= m0 - pd.Timedelta(days=TRAIN_DAYS)) & (lab_end <= m0)) & mask
        te = np.asarray((t >= m0) & (t < m1))
        if tr.sum() < 500 or (te & mask).sum() < 300 or len(np.unique(y[tr])) < 2:
            continue
        Xtr = np.asarray(X[tr], np.float32); sc = StandardScaler().fit(Xtr)
        f = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -8, 8)
        mdl = LogisticRegression(C=C, max_iter=1000).fit(f(Xtr), y[tr])
        p[te] = mdl.predict_proba(f(X[te]))[:, 1]
    return p, mask, y


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def month_list(a, b):
    return [str(m) for m in pd.period_range(a, b, freq="M")]


# ---------------------------------------------------------------- main
def main():
    warnings.filterwarnings("ignore")
    C = load_cache(); n = len(C["t"]); t = C["t"]; mo = C["month"]; sig = C["sig"]
    loss, score = load_ep(n)
    tb7 = np.load(f"{AL.CACHE_IN}/1m_tb7.npy", mmap_mode="r")[C["idx"]]
    out = {"cutoff": str(AL.LAST_BAR), "last_anchor": str(pd.Timestamp(t.max())), "n_anchors": int(n)}

    # ---- monthly EP (seed-averaged window losses), by model
    ep_months = month_list(EP_FIRST, EP_LAST)
    gain = {mn: np.log(2) - np.nanmean([loss[mn][s] for s in SEEDS], axis=0) for mn in AL.MODELS}   # per-window gain, mean over seeds
    mon = {}
    for mn in AL.MODELS:
        mon[mn] = []
        for m in ep_months:
            sel = mo == pd.Period(m, "M").ordinal
            g = gain[mn][sel]; g = g[np.isfinite(g)]
            per_seed = [float(np.log(2) - np.nanmean(loss[mn][s][sel])) for s in SEEDS if np.isfinite(loss[mn][s][sel]).any()]
            mon[mn].append({"month": m, "ep": float(g.mean()) if len(g) else float("nan"), "n": int(len(g)),
                            "seed_values": per_seed})
    out["monthly_ep"] = mon
    ep = {mn: np.array([r["ep"] for r in mon[mn]]) for mn in AL.MODELS}
    out["ep_summary"] = {mn: {"mean": float(np.nanmean(ep[mn])), "min": float(np.nanmin(ep[mn])), "max": float(np.nanmax(ep[mn])),
                              "seed_abs_diff_mean": float(np.nanmean([abs(r["seed_values"][0] - r["seed_values"][1]) for r in mon[mn]
                                                                        if len(r["seed_values"]) == 2]))} for mn in AL.MODELS}

    # ---- tb7 walk-forward (next-open labels) and monthly AUC
    p, mask, y = walk_forward_p(tb7, t, C["r_no"], AUC_FIRST, AUC_LAST)
    auc_months = month_list(AUC_FIRST, AUC_LAST)
    auc_m = []
    for m in auc_months:
        sel = (mo == pd.Period(m, "M").ordinal) & mask & np.isfinite(p)
        auc_m.append(auc(y[sel], p[sel]) if sel.sum() >= 300 else float("nan"))
    auc_m = np.array(auc_m); out["tb7_auc_monthly"] = dict(zip(auc_months, auc_m.tolist()))
    out["tb7_auc_mean"] = tci(auc_m)

    # ---- (i) EP tracks AUC? detrended Spearman
    sel_m = [ep_months.index(m) for m in auc_months]
    ep_a = ep["full"][sel_m]; years = np.array([int(m[:4]) for m in auc_months])
    smean = np.array([np.log(sig[mo == pd.Period(m, "M").ordinal]).mean() for m in auc_months])
    Q = {}
    for name, series in (("EP_full", ep_a), ("EP_vol(abs)", ep["abs"][sel_m]), ("EP_dir(full-abs)", ep["full"][sel_m] - ep["abs"][sel_m]),
                         ("log_sigma", smean)):
        Q[name] = {"raw_spearman": float(stats.spearmanr(series, auc_m).correlation)}
        for kind in ("diff", "dev"):
            r_, a1, b1 = spearman_detrended(series, auc_m, years, kind)
            lo, hi = block_boot_spearman(a1, b1)
            Q[name][kind] = {"rho": r_, "ci95_block_boot": [lo, hi], "n": int(len(a1))}
    Q["EP_full_given_sigma_partial_rho_diff"] = partial_spearman(np.diff(ep_a), np.diff(auc_m), np.diff(smean))
    Q["EP_full_vs_sigma_rho_diff"] = float(stats.spearmanr(np.diff(ep_a), np.diff(smean)).correlation)
    out["i_tracking"] = Q

    # ---- (ii) gates: AUC by quintile of the EP score vs the sigma quintile
    gates = {"EP_full": np.abs(score["full"]), "EP_abs": np.abs(score["abs"]), "EP_sign": np.abs(score["sign"]),
             "EP_order": np.abs(score["order"]), "sigma": sig.astype(float)}
    spreads = {g: [] for g in list(gates) + ["EP_full|sigma"]}
    binauc = {g: [] for g in list(gates) + ["EP_full|sigma"]}
    used = []
    for m in auc_months:
        sel = (mo == pd.Period(m, "M").ordinal) & mask & np.isfinite(p)
        row = {}
        ok = True
        for g, v in gates.items():
            e = edges_for(m, mo, v)
            if e is None:
                ok = False; break
            b = assign_bins(v, e); aucs = []
            for q in range(NBIN):
                s2 = sel & (b == q) & np.isfinite(v)
                aucs.append(auc(y[s2], p[s2]) if s2.sum() >= MIN_BIN else float("nan"))
            row[g] = aucs
        if ok:
            # EP given sigma: EP quintile edges within each sigma quintile (edges from the preceding months)
            es = edges_for(m, mo, gates["sigma"]); bs = assign_bins(gates["sigma"], es)
            prev = (mo >= (pd.Period(m, "M") - EDGE_MONTHS).ordinal) & (mo < pd.Period(m, "M").ordinal)
            bq = np.full(n, -1)
            for qs in range(NBIN):
                pv = gates["EP_full"][prev & (bs == qs) & np.isfinite(gates["EP_full"])]
                ee = np.quantile(pv, np.linspace(0, 1, NBIN + 1)[1:-1])
                mm = (bs == qs) & (mo == pd.Period(m, "M").ordinal)
                bq[mm] = assign_bins(gates["EP_full"][mm], ee)
            aucs = []
            for q in range(NBIN):
                s2 = sel & (bq == q)
                aucs.append(auc(y[s2], p[s2]) if s2.sum() >= MIN_BIN else float("nan"))
            row["EP_full|sigma"] = aucs
            for g in row:
                binauc[g].append(row[g]); spreads[g].append(row[g][-1] - row[g][0])
            used.append(m)
    sp = {g: np.array(v) for g, v in spreads.items()}
    G = {"months_used": len(used), "bin_auc_mean": {g: np.nanmean(np.array(binauc[g]), axis=0).tolist() for g in binauc},
         "spread_top_minus_bottom": {g: tci(sp[g]) for g in sp}}
    sgn = np.sign(np.nanmean(sp["sigma"]))
    G["sigma_orientation_sign"] = float(sgn)
    G["EP_minus_best_oriented_sigma"] = tci(sp["EP_full"] - sgn * sp["sigma"])
    ep_mean = float(np.nanmean(sp["EP_full"])); diff_mean = float(np.nanmean(sp["EP_full"] - sgn * sp["sigma"]))
    G["criterion"] = {"spread_ge_0.015": bool(ep_mean >= 0.015), "beats_sigma_by_0.01": bool(diff_mean >= 0.01),
                      "spread_mean": ep_mean, "minus_sigma_mean": diff_mean}
    # how much EP is volatility: rank correlation of the window scores
    ok = np.isfinite(gates["EP_full"]) & (mo >= pd.Period(AUC_FIRST, "M").ordinal) & (mo <= pd.Period(AUC_LAST, "M").ordinal)
    rr = np.random.RandomState(1).choice(np.nonzero(ok)[0], 50000, replace=False)
    G["window_spearman_EPfull_sigma"] = float(stats.spearmanr(gates["EP_full"][rr], sig[rr]).correlation)
    G["window_spearman_EPfull_EPabs"] = float(stats.spearmanr(gates["EP_full"][rr], gates["EP_abs"][rr]).correlation)
    out["ii_gates"] = G

    # ---- (iii) decomposition, pooled and by year, day-block bootstrap
    ok = np.all([np.isfinite(gain[m_]) for m_ in AL.MODELS], axis=0)
    days = C["day"][ok]; ug, inv = np.unique(days, return_inverse=True)
    def boot(v, reps=BOOT, seed=2):
        s = np.bincount(inv, v); c = np.bincount(inv).astype(float); rs = np.random.RandomState(seed); k = len(ug); out_ = []
        for _ in range(reps):
            ix = rs.randint(0, k, k); out_.append(s[ix].sum() / c[ix].sum())
        return float(np.mean(v)), float(np.percentile(out_, 2.5)), float(np.percentile(out_, 97.5))
    dirv = (gain["full"] - gain["abs"])[ok]
    D = {"EP_total(full)": boot(gain["full"][ok]), "EP_vol(abs)": boot(gain["abs"][ok]), "EP_dir(full-abs)": boot(dirv),
         "EP_sign_control": boot(gain["sign"][ok]), "EP_order_control": boot(gain["order"][ok])}
    D["share_vol_of_total"] = D["EP_vol(abs)"][0] / D["EP_total(full)"][0] if D["EP_total(full)"][0] else float("nan")
    yr = pd.DatetimeIndex(t[ok].astype("datetime64[ns]")).year
    D["by_year"] = {int(y_): {"EP_total": float(gain["full"][ok][yr == y_].mean()), "EP_vol": float(gain["abs"][ok][yr == y_].mean()),
                              "EP_dir": float(dirv[yr == y_].mean()), "n": int((yr == y_).sum())} for y_ in np.unique(yr)}
    out["iii_decomposition"] = D

    # ---- control: sign -> future sign vs sign -> future |r| (leverage), monthly walk-forward
    body = np.asarray(C["X"][:, -10:, 0]); S = np.sign(body)
    netsign = np.sign(np.asarray(C["X"][:, :, 0]).sum(1)); s5 = np.sign(body[:, -5:].sum(1))
    SF = np.column_stack([S, netsign, s5]).astype(np.float32)
    ps, mk, ys = walk_forward_p(SF, t, C["r_no"], AUC_FIRST, AUC_LAST)
    aucss = np.array([auc(ys[(mo == pd.Period(m, "M").ordinal) & mk & np.isfinite(ps)], ps[(mo == pd.Period(m, "M").ordinal) & mk & np.isfinite(ps)])
                      for m in auc_months])
    # sign -> future |r| / (sigma sqrt(H)) above its training median
    zr_ = np.abs(C["r_no"]) / (sig * np.sqrt(AL.H)); zthr = np.full(n, np.nan)
    from sklearn.linear_model import LogisticRegression
    aucsa = []
    tt = pd.DatetimeIndex(t.astype("datetime64[ns]"))
    for m in auc_months:
        mm = pd.Period(m, "M"); m0, m1 = mm.start_time, (mm + 1).start_time
        tr = np.asarray((tt >= m0 - pd.Timedelta(days=TRAIN_DAYS)) & (tt + pd.Timedelta(minutes=AL.H + 1) <= m0)); te = np.asarray((tt >= m0) & (tt < m1))
        yy = (zr_ > np.median(zr_[tr])).astype(int)
        mdl = LogisticRegression(C=1.0, max_iter=500).fit(SF[tr], yy[tr])
        aucsa.append(auc(yy[te], mdl.predict_proba(SF[te])[:, 1]))
    aucsa = np.array(aucsa)
    epdir = (ep["full"] - ep["abs"])[sel_m]
    Cn = {"auc_sign_to_future_sign_mean": tci(aucss), "auc_sign_to_future_absr_mean": tci(aucsa), "tb7_auc_mean": tci(auc_m)}
    for name, v in (("sign_to_sign_auc", aucss), ("sign_to_absr_auc", aucsa)):
        Cn[f"EP_dir_vs_{name}"] = {k: spearman_detrended(epdir, v, years, k)[0] for k in ("diff", "dev")}
    Cn["EP_dir_mean_months"] = tci(epdir)
    out["control_sign_to_sign"] = Cn

    json.dump(out, open(f"{HERE}/results.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("ep_summary", "i_tracking", "ii_gates", "iii_decomposition", "control_sign_to_sign")}, indent=1, default=float))
    figure(out, auc_months, ep_months, ep, auc_m)


def figure(out, auc_months, ep_months, ep, auc_m):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    x = pd.PeriodIndex(ep_months, freq="M").to_timestamp()
    for mn, c in (("full", "#3987e5"), ("abs", "#d95926"), ("order", "#199e70"), ("sign", "#777777")):
        ax[0].plot(x, ep[mn], label=f"EP {mn}", color=c)
    ax[0].set_ylabel("EP (nats / window)"); ax[0].legend(ncol=4); ax[0].set_title("Monthly held-out log-loss gain over ln 2 (mean of 2 seeds)")
    xa = pd.PeriodIndex(auc_months, freq="M").to_timestamp()
    ax[1].plot(xa, auc_m, color="k"); ax[1].axhline(0.5, color="#aaa"); ax[1].set_ylabel("tb7 AUC"); ax[1].set_title("tb7 next-open AUC by month (h = 15, 365-day training)")
    g = out["ii_gates"]["bin_auc_mean"]
    for k, c in (("EP_full", "#3987e5"), ("sigma", "#d95926"), ("EP_full|sigma", "#199e70")):
        ax[2].plot(range(1, NBIN + 1), g[k], marker="o", label=k, color=c)
    ax[2].set_xlabel("quintile of the gate (1 = lowest)"); ax[2].set_ylabel("tb7 AUC"); ax[2].legend(); ax[2].set_title("AUC by gate quintile, mean over 2021-01..2024-06")
    fig.tight_layout(); fig.savefig(f"{HERE}/arrow.png", dpi=110)


if __name__ == "__main__":
    main()
