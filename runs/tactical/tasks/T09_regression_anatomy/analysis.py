"""T09 analyses Q1-Q7 on an anchor dataset + its monthly walk-forwards (lib.py). Pure functions; the run scripts call them.
Aggregation rule everywhere: a statistic is computed per calendar month (the unit of inference, D-012), then averaged per year or period
with a 95% t-interval over months."""
import math

import numpy as np
import pandas as pd

import lib
from lib import ENTRIES, FEAT_NAMES, GROUPS, HS, H0, KS, auc, tint

PERIODS = {"2017-19": ("2017", "2019"), "2020-23": ("2020", "2023"), "2024-25": ("2024", "2025")}
N_EFF = 10 / 15          # anchors every 10 minutes, 15-minute labels: about two thirds of a sample is new (D-012)


def year_of(m):
    return m[:4]


def period_of(m):
    y = m[:4]
    for k, (a, b) in PERIODS.items():
        if a <= y <= b:
            return k
    return None


def agg(rows, key, by="year", f=None):
    """rows: list of dicts with 'month'; returns {group: tint}. `key` may be a scalar field or (with f) a function of the row."""
    g = {}
    for r in rows:
        v = f(r) if f else r.get(key)
        if v is None or not np.isfinite(v):
            continue
        k = year_of(r["month"]) if by == "year" else ("all" if by == "all" else period_of(r["month"]))
        if k is not None:
            g.setdefault(k, []).append(v)
    return {k: tint(v) for k, v in sorted(g.items())}


# ------------------------------------------------------------------ Q1
def q1_tables(W):
    """Per feature: year-mean (with CI) of the standardised coefficient, the univariate AUC, the drop-one AUC loss, the permutation loss;
    per group: grouped permutation loss; per year: the ranking by drop-one loss and the top-3."""
    rows = W["months"]; out = {"features": FEAT_NAMES, "years": sorted({year_of(r["month"]) for r in rows})}
    for key in ("coef", "uni", "drop", "perm"):
        out[key] = {nm: agg(rows, None, f=lambda r, j=j, key=key: r[key][j]) for j, nm in enumerate(FEAT_NAMES)} if rows and key in rows[0] else {}
    for key in ("gperm", "gdrop", "gonly"):
        out[key] = {g: agg(rows, None, f=lambda r, i=i, key=key: r[key][i]) for i, g in enumerate(GROUPS)} if rows and rows[0].get(key) else {}
    out["auc"] = agg(rows, "auc")
    top3 = {}; rank = {}
    for y in out["years"]:
        sc = {nm: out["drop"][nm][y]["mean"] for nm in FEAT_NAMES} if out["drop"] else {}
        if sc:
            order = sorted(sc, key=lambda k: -sc[k]); top3[y] = order[:3]; rank[y] = order
    out["top3_by_year"] = top3; out["rank_by_year"] = rank
    # rank agreement between years (Spearman of the drop-one means)
    if rank:
        from scipy.stats import spearmanr
        ys = list(rank); mat = np.array([[out["drop"][nm][y]["mean"] for nm in FEAT_NAMES] for y in ys])
        out["rank_spearman"] = {"years": ys, "matrix": np.round(np.corrcoef(pd.DataFrame(mat).rank(axis=1).values), 3).tolist()}
        uni = np.array([[abs(out["uni"][nm][y]["mean"] - 0.5) for nm in FEAT_NAMES] for y in ys])
        out["uni_abs_dev"] = {"years": ys, "values": np.round(uni, 4).tolist()}
    return out


def subset_models(D, entry, feats_by_rank, first=None, last=None):
    """Monthly walk-forward restricted to the top-1, top-2, top-3 features (and all): yearly AUC and the share of the full model's edge."""
    out = {}
    for n in (1, 2, 3):
        cols = [FEAT_NAMES.index(f) for f in feats_by_rank[:n]]
        Wn = lib.walk(D, entry, feature_cols=cols, first=first, last=last)
        out[n] = {"features": feats_by_rank[:n], "auc_by_year": agg(Wn["months"], "auc"), "months": [(r["month"], r["auc"]) for r in Wn["months"]]}
    return out


# ------------------------------------------------------------------ Q2
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def q2_partial_dependence(D, W, entry, features, nb=20):
    """Per era and feature: 20 quantile bins of the feature (era pool of masked test anchors): empirical P(up) with a Wilson interval on
    n_eff = n*N_EFF, the model-implied partial dependence (average monthly-model P(up) with the feature set to the bin median),
    the mean feature value; plus linearity / symmetry / tail summaries."""
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); F = D["F"]
    r = D["r_" + entry][:, HS.index(H0)]; y = (r > 0).astype(int); mask = np.isfinite(r) & (np.abs(np.nan_to_num(r)) > lib.DB0)
    mon = np.array([str(x) for x in t.to_period("M")])
    months = {rec["month"] for rec in W["months"]}
    out = {}
    for era, (a, b) in PERIODS.items():
        sel = np.array([(m in months) and (a <= m[:4] <= b) for m in mon]) & mask & np.isfinite(W["p"]) & ~D["seam"]       # DST / splice seam anchors excluded
        idx = np.flatnonzero(sel)
        for fn in features:
            j = FEAT_NAMES.index(fn); x = F[idx, j]
            edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
            b_ = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            rows = []
            for k in range(len(edges) - 1):
                ii = idx[b_ == k]
                if len(ii) == 0:
                    continue
                n = len(ii); up = int(y[ii].sum()); ne = max(n * N_EFF, 1); lo, hi = wilson(up / n * ne, ne)
                rows.append({"bin": k, "x_mean": float(F[ii, j].mean()), "x_median": float(np.median(F[ii, j])), "n": n, "p_up": up / n, "lo": lo, "hi": hi})
            # model-implied PD: set feature j to each bin's median in the month's own model
            pd_ = np.zeros(len(rows))
            for mname, (sc, mdl) in W["models"].items():
                if not (a <= mname[:4] <= b):
                    continue
                mi = idx[mon[idx] == mname]
                if len(mi) == 0:
                    continue
                X = sc.transform(F[mi]).copy(); w = len(mi)
                for k, rr in enumerate(rows):
                    Xk = X.copy(); Xk[:, j] = (rr["x_median"] - sc.mean_[j]) / sc.scale_[j]
                    pd_[k] += mdl.predict_proba(Xk)[:, 1].mean() * w
            tot = sum((mon[idx] == mm).sum() for mm in W["models"] if a <= mm[:4] <= b)
            for k, rr in enumerate(rows):
                rr["pd_model"] = float(pd_[k] / max(tot, 1))
            xs = np.array([r_["x_median"] for r_ in rows]); ps = np.array([r_["p_up"] for r_ in rows]); ws = np.array([r_["n"] for r_ in rows], float)
            from scipy.stats import spearmanr
            A = np.vstack([xs, np.ones_like(xs)]).T; coef = np.linalg.lstsq(A * np.sqrt(ws)[:, None], ps * np.sqrt(ws), rcond=None)[0]
            res = ps - A @ coef; ss_res = float((ws * res ** 2).sum()); mu = float((ws * ps).sum() / ws.sum()); ss_tot = float((ws * (ps - mu) ** 2).sum())
            h = len(rows) // 2
            def slope(i0, i1):
                if i1 - i0 < 3:
                    return float("nan")
                return float(np.polyfit(xs[i0:i1], ps[i0:i1], 1)[0])
            # antisymmetry: (p_up(x) - 0.5) + (p_up(-x) - 0.5) over mirrored bins, compared to the mean absolute edge
            mir = [(ps[i] - 0.5) + (ps[len(ps) - 1 - i] - 0.5) for i in range(h)]
            edge = np.abs(ps - 0.5)
            # tails: the "fade" hit rate by |x| decile (sign(-x) predicts the move) on the era pool
            ax = np.abs(x); q = np.quantile(ax, np.linspace(0, 1, 11)); tails = []
            for k in range(10):
                ii = idx[(ax >= q[k]) & (ax <= q[k + 1] if k == 9 else ax < q[k + 1])]
                if len(ii) == 0:
                    continue
                hit = ((F[ii, j] < 0) == (y[ii] == 1)); tails.append({"decile": k, "n": len(ii), "fade_hit": float(hit.mean()), "ax_mean": float(ax[(ax >= q[k]) & (ax <= q[k + 1] if k == 9 else ax < q[k + 1])].mean())})
            mu = float((ws * ps).sum() / ws.sum()); k3 = min(3, len(ps) // 2)
            low_tail = float(np.average(ps[:k3], weights=ws[:k3]) - mu); high_tail = float(mu - np.average(ps[-k3:], weights=ws[-k3:]))
            out.setdefault(era, {})[fn] = {"mean_p_up": mu, "edge_low_tail": low_tail, "edge_high_tail": high_tail,
                "bins": rows, "n": int(len(idx)), "spearman": float(spearmanr(xs, ps)[0]), "r2_linear": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                "slope_lower_half": slope(0, h), "slope_upper_half": slope(len(rows) - h, len(rows)), "mirror_asym_mean": float(np.mean(mir)),
                "mean_abs_edge": float(edge.mean()), "tails": tails,
                "edge_ends_vs_middle": {"ends": float(np.mean(np.r_[edge[:3], edge[-3:]])), "middle": float(edge[len(edge) // 2 - 3:len(edge) // 2 + 3].mean())}}
    return out


# ------------------------------------------------------------------ Q3
def impulse(D, entries=ENTRIES, ks=KS, hs=HS, first_month="2017-04", last_month=None, exclude_seam=True):
    """Per month, entry, lag k, horizon h: the AUC of fading the last k-minute return (score = -return) against the sign of the h-minute return
    (|r| above the deadband of that horizon) and the mean signed return of the fade rule (bps, anchors with a non-zero last return).
    Returns {entry: {"months": [...], "auc": [M,K,H], "bps": [M,K,H], "n": [M]}}."""
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); mon = np.array([str(x) for x in t.to_period("M")])
    LR = D["lastret"]; out = {}
    ms = [m for m in sorted(set(mon)) if m >= first_month and (last_month is None or m <= last_month)]
    for e in entries:
        R = D["r_" + e]; A = np.full((len(ms), len(ks), len(hs)), np.nan); B = np.full_like(A, np.nan); N = np.zeros(len(ms), int)
        for i, m in enumerate(ms):
            ii = np.flatnonzero((mon == m) & ~D["seam"]) if exclude_seam else np.flatnonzero(mon == m); N[i] = len(ii)
            for a, k in enumerate(ks):
                lr = LR[ii, KS.index(k)]
                for b, h in enumerate(hs):
                    r = R[ii, HS.index(h)]; ok = np.isfinite(lr) & np.isfinite(r)
                    m_ = ok & (np.abs(r) > lib.deadband(h))
                    if m_.sum() >= 200:
                        A[i, a, b] = auc((r[m_] > 0).astype(int), -lr[m_])
                    nz = ok & (lr != 0)
                    if nz.sum() >= 200:
                        B[i, a, b] = float(np.mean(-np.sign(lr[nz]) * r[nz])) * 1e4
        out[e] = {"months": ms, "auc": A, "bps": B, "n": N}
    return out


def impulse_by_year(IR, entry, stat="auc"):
    ms = IR[entry]["months"]; X = IR[entry][stat]; yrs = sorted({m[:4] for m in ms}); res = {}
    for y in yrs:
        ii = [i for i, m in enumerate(ms) if m[:4] == y]
        res[y] = np.nanmean(X[ii], axis=0)
    return res


def simple_rule_share(IR, W, entry):
    """Per year: the regression's mean monthly AUC, the AUC of the fade-k rule at h = 15 for the best k in the year (optimistic), for the k that was
    best in the previous year (causal), and for k = 1, with the share of the regression's edge ((AUC-0.5)/(regression AUC-0.5))."""
    ms = IR[entry]["months"]; A = IR[entry]["auc"][:, :, HS.index(H0)]
    reg = {r["month"]: r["auc"] for r in W["months"]}; res = {}
    prev_k = None
    for y in sorted({m[:4] for m in ms}):
        ii = [i for i, m in enumerate(ms) if m[:4] == y and m in reg]
        if not ii:
            continue
        regs = np.array([reg[ms[i]] for i in ii]); mean_by_k = np.nanmean(A[ii], axis=0); kbest = int(np.nanargmax(mean_by_k))
        row = {"regression_auc": float(regs.mean()), "auc_by_k": {str(k): float(v) for k, v in zip(KS, mean_by_k)}, "best_k": KS[kbest],
               "best_auc": float(mean_by_k[kbest]), "k1_auc": float(mean_by_k[0])}
        row["share_best"] = (row["best_auc"] - 0.5) / (row["regression_auc"] - 0.5)
        row["share_k1"] = (row["k1_auc"] - 0.5) / (row["regression_auc"] - 0.5)
        if prev_k is not None:
            row["prev_year_k"] = KS[prev_k]; row["causal_auc"] = float(mean_by_k[prev_k])
            row["share_causal"] = (row["causal_auc"] - 0.5) / (row["regression_auc"] - 0.5)
        prev_k = kbest; res[y] = row
    return res


# ------------------------------------------------------------------ Q4 / Q5 conditions
def cond_values(D):
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]"))
    return {"vol": D["sig"], "volume": D["vmean"], "hl": D["hl"], "cs": D["cs"], "move1bps": D["move1bps"], "move1sig": D["move1sig"],
            "weekend": (t.dayofweek >= 5).astype(float), "hour": D["hour_utc"].astype(float)}


QEDGES = {"move1bps": (0, 1 / 3, 2 / 3, 0.9, 0.97, 1), "move1sig": (0, 1 / 3, 2 / 3, 0.9, 0.97, 1)}


def cond_bins(name, v_tr, v_te):
    """Bin index of the test values: categorical conditions by value; the others by the training span's quantile edges (terciles, or the
    finer QEDGES for the size of the last move). Returns (bin index, labels)."""
    if name == "weekend":
        return v_te.astype(int), ["weekday", "weekend"]
    if name == "hour":
        return v_te.astype(int), [f"{h:02d}" for h in range(24)]
    qs = QEDGES.get(name, (0, 1 / 3, 2 / 3, 1)); e = np.quantile(v_tr, qs[1:-1])
    if name == "cs" and e[0] <= 0:          # the Corwin-Schultz estimate is clipped at 0 and is 0 on most bars from 2022: zero | low positive | high positive
        pos = v_tr[v_tr > 0]; e = np.array([1e-12, np.median(pos) if len(pos) else 1.0])
    # ties at the cut points (e.g. a constant last move): make the edges non-decreasing and use right-open bins
    b = np.searchsorted(e, v_te, side="right")
    labs = ["low", "mid", "high"] if len(qs) == 4 else ["q0-33", "q33-67", "q67-90", "q90-97", "q97-100"]
    return b, labs


def conditional_table(D, W, entry):
    """Per month and (condition, bin): n (masked), AUC, mean signed return of trading every anchor on the model's side (bps, unmasked), and
    the top-10% trades (threshold from the training span) with their count and mean bps."""
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); CV = cond_values(D)
    r = D["r_" + entry][:, HS.index(H0)]; y = (r > 0).astype(int); ok = np.isfinite(r); mask = ok & (np.abs(np.nan_to_num(r)) > lib.DB0)
    p = W["p"]; rows = []
    for rec in W["months"]:
        m = pd.Period(rec["month"], "M"); m0, m1 = m.start_time, (m + 1).start_time
        tr, te = lib.split(D["t"], m0, m1, 60)
        tem = te & mask; ptest = p[te]
        for cname, v in CV.items():
            b_te, labs = cond_bins(cname, v[tr & mask], v[te])
            side = np.where(ptest > 0.5, 1, -1); rt = r[te]; okte = ok[te]
            top = np.abs(ptest - 0.5) >= rec["thr"]
            mt = mask[te]; yt = y[te]
            if cname == "hour":                      # the hour-of-day split leaves out the seam anchors (the hour label is ambiguous there)
                mt = mt & ~D["seam"][te]
            for k, lab in enumerate(labs):
                sel = b_te == k
                s_m = sel & mt
                a = auc(yt[s_m], ptest[s_m]) if s_m.sum() >= (40 if cname == "hour" else 100) else float("nan")
                s_a = sel & okte
                rows.append({"month": rec["month"], "cond": cname, "bin": lab, "n": int(s_m.sum()), "auc": a,
                             "bps_all": float(np.mean(side[s_a] * rt[s_a]) * 1e4) if s_a.sum() >= 30 else float("nan"),
                             "n_top": int((sel & top & okte).sum()),
                             "bps_top": float(np.mean(side[sel & top & okte] * rt[sel & top & okte]) * 1e4) if (sel & top & okte).sum() >= 10 else float("nan")})
    return pd.DataFrame(rows)


def cond_summary(df, by="period"):
    """Aggregate the conditional table: per (cond, bin, year or period) the mean over months with a 95% t-interval."""
    df = df.copy(); df["grp"] = df["month"].map(period_of if by == "period" else (year_of if by == "year" else (lambda m: "all"))); df = df[df["grp"].notna()]
    rows = []
    for (c, b, g), s in df.groupby(["cond", "bin", "grp"], sort=False):
        a = tint(s["auc"].values); bp = tint(s["bps_all"].values); bt = s["bps_top"].values; nt = s["n_top"].values
        wt = np.nansum(bt * nt) / max(nt[np.isfinite(bt)].sum(), 1) if np.isfinite(bt).any() else float("nan")
        rows.append({"cond": c, "bin": b, "grp": g, "auc": a["mean"], "auc_lo": a["lo"], "auc_hi": a["hi"], "n_months": a["n"],
                     "bps_all": bp["mean"], "bps_all_lo": bp["lo"], "bps_all_hi": bp["hi"], "bps_top": float(wt), "n_top": int(np.nansum(nt))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Q6
def fixed_model(D, entry, year="2019"):
    """The stable coefficients: one logistic regression fitted on all masked anchors of `year`; its P(up) on every anchor."""
    from sklearn.preprocessing import StandardScaler
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); r = D["r_" + entry][:, HS.index(H0)]
    mask = np.isfinite(r) & (np.abs(np.nan_to_num(r)) > lib.DB0); y = (r > 0).astype(int)
    tr = (t.year == int(year)) & mask & (t <= pd.Timestamp(f"{year}-12-31 23:59") - pd.Timedelta(minutes=H0 + 2))
    sc = StandardScaler().fit(D["F"][tr]); mdl = lib.fit_logreg(sc.transform(D["F"][tr]), y[tr])
    return {"p": mdl.predict_proba(sc.transform(D["F"]))[:, 1].astype(np.float32), "coef": mdl.coef_[0].tolist(), "n_train": int(tr.sum())}


def decomposition(D, W, entry, fixed, base="2019", cell_vars=("sig", "vmean")):
    """Per month: A_full (the monthly refit), A_fixed (the stable base-year coefficients), A_fixed_rew (A_fixed with the month's anchors reweighted
    so that the (volatility tercile x CS-spread tercile) cell mix equals the base year's; cuts from the base year's anchors) and the market-quality
    columns; then the waterfall per year against the base year."""
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); r = D["r_" + entry][:, HS.index(H0)]; y = (r > 0).astype(int)
    mask = np.isfinite(r) & (np.abs(np.nan_to_num(r)) > lib.DB0); mon = np.array([str(x) for x in t.to_period("M")])
    inb = (t.year == int(base)) & mask
    va, vb = cell_vars
    ev = np.quantile(D[va][inb], [1 / 3, 2 / 3]); ec = np.quantile(D[vb][inb], [1 / 3, 2 / 3])
    cell = np.searchsorted(ev, D[va], side="right") * 3 + np.searchsorted(ec, D[vb], side="right")
    pb = np.bincount(cell[inb], minlength=9) / inb.sum()
    from sklearn.metrics import roc_auc_score
    rows = []
    for rec in W["months"]:
        ii = np.flatnonzero((mon == rec["month"]) & mask)
        pm = np.bincount(cell[ii], minlength=9) / len(ii)
        w = np.where(pm[cell[ii]] > 0, pb[cell[ii]] / np.maximum(pm[cell[ii]], 1e-12), 0.0)
        a_fix = auc(y[ii], fixed["p"][ii])
        try:
            a_rew = float(roc_auc_score(y[ii], fixed["p"][ii], sample_weight=w))
        except ValueError:
            a_rew = float("nan")
        rows.append({"month": rec["month"], "full": rec["auc"], "fixed": a_fix, "fixed_rew": a_rew, "cell_share": pm.tolist(),
                     "cosine_to_base": None})
    base_coef = np.array(fixed["coef"])
    for rw, rec in zip(rows, W["months"]):
        c = np.array(rec["coef"]); rw["cosine_to_base"] = float(c @ base_coef / (np.linalg.norm(c) * np.linalg.norm(base_coef) + 1e-12))
    return {"rows": rows, "cell_edges": {va: ev.tolist(), vb: ec.tolist()}, "cell_vars": [va, vb], "base_cell_share": pb.tolist()}


def waterfall(dec, base="2019"):
    rows = dec["rows"]; yr = {}
    for r in rows:
        yr.setdefault(r["month"][:4], []).append(r)
    mean = lambda y, k: float(np.nanmean([r[k] for r in yr[y]]))
    out = {}
    fb, xb = mean(base, "full"), mean(base, "fixed")
    for y in sorted(yr):
        full, fixed, rew = mean(y, "full"), mean(y, "fixed"), mean(y, "fixed_rew")
        out[y] = {"full": full, "fixed": fixed, "fixed_rew": rew, "total": full - fb, "coef": (full - fixed) - (fb - xb), "cond": fixed - rew,
                  "strength": rew - xb, "cosine_to_base": mean(y, "cosine_to_base")}
    return out


def mq_table(D):
    ms = list(D["mq_month"]); return {m: dict(zip(lib.MQ_COLS, v)) for m, v in zip(ms, D["mq"].tolist())}


def mq_correlations(W, MQ, uni_by_month=None):
    """Spearman correlation across months between the monthly AUC and each market-quality proxy, raw and after removing a linear time trend
    (the proxies drift with time as the AUC does, so the raw correlation alone says little)."""
    from scipy.stats import spearmanr, pearsonr
    ms = [r["month"] for r in W["months"] if r["month"] in MQ]; a = np.array([r["auc"] for r in W["months"] if r["month"] in MQ])
    ti = np.arange(len(ms)); res = {}
    def detr(x):
        c = np.polyfit(ti, x, 1); return x - np.polyval(c, ti)
    for col in lib.MQ_COLS:
        x = np.array([MQ[m][col] for m in ms], float); xv = np.log(x) if col in ("volume_mean",) else x
        ok = np.isfinite(xv) & np.isfinite(a)
        rho = spearmanr(xv[ok], a[ok])[0]; rd = pearsonr(detr(xv[ok]), detr(a[ok]))
        res[col] = {"spearman": float(rho), "detrended_pearson": float(rd[0]), "detrended_p": float(rd[1]), "n_months": int(ok.sum())}
    return res


# ------------------------------------------------------------------ Q7
def top10_trades(D, W_by_entry):
    """Top-10% most confident anchors per month (threshold from the training predictions, H51's rule), traded on every anchor of the month that
    clears it: mean gross bps per trade by entry convention. Rows per month."""
    rows = []
    for rec in W_by_entry["close"]["months"]:
        m = rec["month"]; per = {}
        mm = pd.Period(m, "M"); t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); te = np.asarray((t >= mm.start_time) & (t < (mm + 1).start_time))
        for name, (model_e, entry_e) in {"close": ("close", "close"), "nopen": ("nopen", "nopen"), "mid": ("nopen", "mid"),
                                           "close_model_at_nopen": ("close", "nopen")}.items():
            recm = next((x for x in W_by_entry[model_e]["months"] if x["month"] == m), None)
            if recm is None:
                continue
            p = W_by_entry[model_e]["p"][te]; r = D["r_" + entry_e][te][:, HS.index(H0)]; top = (np.abs(p - 0.5) >= recm["thr"]) & np.isfinite(r)
            side = np.where(p > 0.5, 1, -1)
            per[name] = (float(np.mean(side[top] * r[top]) * 1e4) if top.sum() else float("nan"), int(top.sum()),
                         float(np.mean((side[top] * r[top]) > 0)) if top.sum() else float("nan"))
        recm = next((x for x in W_by_entry["close"]["months"] if x["month"] == m), None)
        p = W_by_entry["close"]["p"][te]; top = np.abs(p - 0.5) >= recm["thr"]
        row = {"month": m, "cs_trades": float(np.median(D["cs"][te][top])) if top.any() else float("nan"), "hl_trades": float(np.median(D["hl"][te][top])) if top.any() else float("nan"),
               "cs_all": float(np.median(D["cs"][te])), "hl_all": float(np.median(D["hl"][te]))}
        for k, (b, n, h) in per.items():
            row[f"bps_{k}"] = b; row[f"n_{k}"] = n; row[f"hit_{k}"] = h
        rows.append(row)
    return rows
