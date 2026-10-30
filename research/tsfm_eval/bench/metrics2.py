"""Score every forecast file in RESULTS against causal baselines on identical origins.

All quantities are log returns in bps relative to the last observed close t; target y_h = lc[t+h] - lc[t].
Reference baseline for paired tests: EWMA-Student-t (lambda and nu fitted on pre-holdout origins only).
Writes RESULTS/metrics2.json and RESULTS/metrics2.md. Set BENCH_DATA=post for the post-release set.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import chi2, norm, spearmanr, t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, HORIZONS, QLEVELS, RESULTS, load, origins  # noqa: E402

LV = np.array(QLEVELS)
DEC = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])          # indices of q0.1..q0.9 in the 21-level grid
EDGES = np.concatenate([[0.0], (LV[1:] + LV[:-1]) / 2, [1.0]])
W_MEAN = np.diff(EDGES)                                   # midpoint weights for E[X] over the quantile function
COSTS = [4.0, 10.0, 20.0]   # round trip bps: futures maker 2+2, futures taker 5+5, spot regular 10+10
NBOOT = 1000
rs = np.random.default_rng(0)

df, info = load()
lc = np.log(df["close"].to_numpy(np.float64))
r1 = np.concatenate([[0.0], np.diff(lc)]) * 1e4
N = len(lc)
ORG = origins(N)
PRE = np.arange(1440 + 60, ORG[0] - max(HORIZONS), 3)       # pre-holdout origins for fitting baselines


# ---------------------------------------------------------------- helpers
_BOOT = {}


def boot(n, block):
    if (n, block) not in _BOOT:
        st = rs.integers(0, n - block + 1, size=(NBOOT, int(np.ceil(n / block))))
        _BOOT[(n, block)] = (st[:, :, None] + np.arange(block)).reshape(NBOOT, -1)[:, :n]
    return _BOOT[(n, block)]


def ci_mean(d, block):
    m = d[boot(len(d), block)].mean(axis=1)
    return [round(float(v), 5) for v in np.percentile(m, [2.5, 97.5])]


def ci_stat(fn, arrays, block):
    vals = []
    for b in boot(len(arrays[0]), block):
        try:
            vals.append(fn(*[a[b] for a in arrays]))
        except ValueError:
            pass
    return None if not vals else [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


def pinball_rows(q, y, idx=None):
    lv = LV if idx is None else LV[idx]
    qq = q if idx is None else q[:, idx]
    d = y[:, None] - qq
    return np.mean(np.maximum(lv * d, (lv - 1) * d), axis=1)


def p_up(q):
    out = np.empty(len(q))
    for i, row in enumerate(q):
        out[i] = 1.0 - np.interp(0.0, np.maximum.accumulate(row), LV, left=0.0, right=1.0)
    return np.clip(out, 0.01, 0.99)


def var_from_q(q):
    m = q @ W_MEAN
    return np.maximum((q ** 2) @ W_MEAN - m ** 2, 1e-6)


def kupiec(hits, p):
    n, x = len(hits), int(hits.sum())
    if x in (0, n):
        return None
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ph = x / n
    ll1 = (n - x) * np.log(1 - ph) + x * np.log(ph)
    return round(float(chi2.sf(-2 * (ll0 - ll1), 1)), 4)


def nonoverlap_trades(pred_bps, y, org, h, cost):
    """One position at a time: enter when |pred| > cost, hold h minutes, then look again."""
    pnl, next_free = [], -1
    for i, t in enumerate(org):
        if t < next_free or abs(pred_bps[i]) <= cost:
            continue
        pnl.append(np.sign(pred_bps[i]) * y[i] - cost)
        next_free = t + h
    pnl = np.array(pnl)
    if len(pnl) < 2:
        return {"n": int(len(pnl))}
    return {"n": int(len(pnl)), "net_bps": round(float(pnl.mean()), 3),
            "t_stat": round(float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))), 2),
            "hit": round(float((pnl + cost > 0).mean()), 4)}


# ---------------------------------------------------------------- reference baseline: EWMA-t fitted pre-holdout
def ewma_var(lam):
    v = np.zeros(N)
    v[0] = r1[1] ** 2
    for i in range(1, N):
        v[i] = lam * v[i - 1] + (1 - lam) * r1[i] ** 2
    return v


LAMS = [0.90, 0.94, 0.97, 0.985, 0.995]
NUS = [2.5, 3.0, 4.0, 6.0, 10.0, 1e6]
EW = {lam: ewma_var(lam) for lam in LAMS}


def t_quantiles(sig, nu):
    z = norm.ppf(LV) if nu > 1e5 else student_t.ppf(LV, nu) * np.sqrt((nu - 2) / nu)
    return sig[:, None] * z[None, :]


FIT = {}
for h in HORIZONS:
    yp = (lc[PRE + h] - lc[PRE]) * 1e4
    best = None
    for lam in LAMS:
        sig = np.sqrt(np.maximum(EW[lam][PRE] * h, 1e-6))
        for nu in NUS:
            loss = pinball_rows(t_quantiles(sig, nu), yp).mean()
            if best is None or loss < best[0]:
                best = (loss, lam, nu)
    FIT[h] = {"lambda": best[1], "nu": best[2], "pre_pinball": round(float(best[0]), 4), "n_pre": int(len(PRE))}

# logistic direction baseline on past returns, fitted pre-holdout with a 15-min purge
LR = {}


def past_feats(org):
    return np.column_stack([(lc[org] - lc[org - k]) * 1e4 for k in (1, 5, 15, 60)]
                           + [np.array([r1[t - 59:t + 1].std() for t in org])])


Xpre = past_feats(PRE[PRE < ORG[0] - 15 - max(HORIZONS)])
for h in HORIZONS:
    pre_o = PRE[PRE < ORG[0] - 15 - max(HORIZONS)]
    yp = (lc[pre_o + h] - lc[pre_o]) > 0
    mu, sd = Xpre.mean(0), Xpre.std(0) + 1e-9
    LR[h] = (LogisticRegression(max_iter=1000).fit((Xpre - mu) / sd, yp), mu, sd)


_CSUM = np.concatenate([[0.0], np.cumsum(lc)])


def ctx_mean(org, c):
    """Mean log price over the c bars ending at t (what a TSFM's context normalisation centres on)."""
    return (_CSUM[org + 1] - _CSUM[org + 1 - c]) / c


# ---------------------------------------------------------------- scoring
def score(name, h, org, y, q=None, point=None, dir_prob=None, ref=None, block=20):
    """q: (n,21) quantiles of the h-step return in bps (or None); point: predicted return bps."""
    r = {"model": name, "h": h, "n": int(len(y))}
    if point is None and q is not None:
        point = q[:, 10]
    rw = np.abs(y)
    if point is not None:
        err = np.abs(point - y)
        r["mae_skill_vs_rw"] = round(float(1 - err.mean() / rw.mean()), 5)
        r["mae_skill_ci95"] = ci_stat(lambda e, a: 1 - e.mean() / a.mean(), [err, rw], block)
        s = np.sign(point)
        m = (y != 0) & (s != 0)
        if m.sum() > 10 and len(set(s[m])) > 1:
            yy, ss = (y[m] > 0).astype(int), (s[m] > 0).astype(int)
            r["dir_acc"] = round(float((yy == ss).mean()), 4)
            r["dir_mcc"] = round(float(matthews_corrcoef(yy, ss)), 4)
            r["dir_mcc_ci95"] = ci_stat(lambda a, b, mm: matthews_corrcoef((b[mm] > 0).astype(int), (a[mm] > 0).astype(int)),
                                        [s, y, m], block)
            r["pred_up_frac"] = round(float(ss.mean()), 4)
        sma = {c: ctx_mean(org, c) for c in (512, 2048) if org[0] >= c}
        for c, v in sma.items():
            rev = np.sign(v - lc[org])
            mm = (rev != 0) & (s != 0)
            r[f"agree_ctxmean_rev{c}"] = round(float((rev[mm] == s[mm]).mean()), 4)
        r["trades_nonoverlap"] = {f"cost{c:g}": nonoverlap_trades(point, y, org, h, c) for c in COSTS}
    mz = y != 0
    yb = (y[mz] > 0).astype(int)
    if q is not None:
        pu = p_up(q)
        r["auc_up"] = round(float(roc_auc_score(yb, pu[mz])), 4)
        r["auc_up_ci95"] = ci_stat(lambda p, t, mm: roc_auc_score((t[mm] > 0).astype(int), p[mm]), [pu, y, mz], block)
        r["brier_skill"] = round(float(1 - np.mean((pu[mz] - yb) ** 2) / np.mean((yb.mean() - yb) ** 2)), 5)
        pb21, pb9 = pinball_rows(q, y), pinball_rows(q, y, DEC)
        r["pinball21_bps"], r["pinball9_bps"] = round(float(pb21.mean()), 4), round(float(pb9.mean()), 4)
        for lo, hi, lab in [(5, 15, "cov50"), (2, 18, "cov80"), (0, 20, "cov98")]:
            r[lab] = round(float(((y >= q[:, lo]) & (y <= q[:, hi])).mean()), 4)
        for j, lab in [(0, "hit_q01"), (1, "hit_q05"), (19, "hit_q95"), (20, "hit_q99")]:
            r[lab] = round(float((y < q[:, j]).mean()), 4)
        r["kupiec_p_q01"] = kupiec(y < q[:, 0], 0.01)
        r["kupiec_p_q99"] = kupiec(y > q[:, 20], 0.01)
        var = var_from_q(q)
        ql = np.log(var) + y ** 2 / var
        r["qlike"] = round(float(ql.mean()), 4)
        spread = q[:, 18] - q[:, 2]
        r["vol_spearman"] = round(float(spearmanr(spread, rw).statistic), 4)
        if ref is not None:
            for key, mine, theirs in [("pinball9", pb9, ref["pb9"]), ("pinball21", pb21, ref["pb21"]), ("qlike", ql, ref["ql"])]:
                d = mine - theirs
                r[f"{key}_vs_ref_pct"] = round(float(100 * d.mean() / abs(theirs.mean())), 3)
                lo, hi = ci_mean(d, block)
                den = abs(float(theirs.mean()))
                r[f"{key}_vs_ref_ci95_pct"] = [round(100 * lo / den, 3), round(100 * hi / den, 3)]
        r["max_q99_minus_q50_bps"] = round(float(np.max(q[:, 20] - q[:, 10])), 2)
    if dir_prob is not None:
        r["auc_dirhead"] = round(float(roc_auc_score(yb, dir_prob[mz])), 4)
        r["auc_dirhead_ci95"] = ci_stat(lambda p, t, mm: roc_auc_score((t[mm] > 0).astype(int), p[mm]), [dir_prob, y, mz], block)
        r["dirhead_up_frac"] = round(float((dir_prob > 0.5).mean()), 4)
    return r


rows, refs, set_info = [], {}, {}


def baselines(org, label, block):
    out = []
    W = 1440
    for h in HORIZONS:
        y = (lc[org + h] - lc[org]) * 1e4
        lam, nu = FIT[h]["lambda"], FIT[h]["nu"]
        sig = np.sqrt(np.maximum(EW[lam][org] * h, 1e-6))
        qref = t_quantiles(sig, nu)
        ref = {"pb9": pinball_rows(qref, y, DEC), "pb21": pinball_rows(qref, y),
               "ql": np.log(var_from_q(qref)) + y ** 2 / var_from_q(qref)}
        refs[(label, h)] = ref
        out.append(score(f"REF_ewma_t(lam={lam},nu={nu:g})", h, org, y, qref, point=np.zeros_like(y), block=block))
        sig94 = np.sqrt(np.maximum(EW[0.94][org] * h, 1e-6))
        out.append(score("ewma_gauss_0.94", h, org, y, t_quantiles(sig94, 1e6), point=np.zeros_like(y), ref=ref, block=block))
        hq = np.stack([np.quantile((lc[t - W + h:t + 1] - lc[t - W:t + 1 - h]) * 1e4, LV) for t in org])
        out.append(score("hist_quantiles_1d", h, org, y, hq, ref=ref, block=block))
        mom = (lc[org] - lc[org - h]) * 1e4
        out.append(score("momentum_sign", h, org, y, point=np.sign(mom) * 1e-6, block=block))
        out.append(score("reversal_sign", h, org, y, point=-np.sign(mom) * 1e-6, block=block))
        for c in (512, 2048):
            if org[0] >= c:
                v = ctx_mean(org, c)
                out.append(score(f"ctxmean_reversion_{c}", h, org, y, point=np.sign(v - lc[org]) * 1e-6, block=block))
        m, mu, sd = LR[h]
        pl = m.predict_proba((past_feats(org) - mu) / sd)[:, 1]
        out.append(score("logreg_pastret_prefit", h, org, y, point=(pl - 0.5) * 1e-3, dir_prob=pl, block=block))
    for r in out:
        r["set"] = label
    rows.extend(out)
    yy = {h: (lc[org + h] - lc[org]) * 1e4 for h in HORIZONS}
    set_info[label] = {"n_origins": int(len(org)), "stride_min": int(np.median(np.diff(org))),
                       "first": str(df["datetime"].iloc[org[0]]), "last": str(df["datetime"].iloc[org[-1]]),
                       "mean_abs_ret_bps": {h: round(float(np.abs(yy[h]).mean()), 3) for h in HORIZONS},
                       "up_frac": {h: round(float((yy[h] > 0).mean()), 4) for h in HORIZONS},
                       "window_return_pct": round(float(100 * (np.exp(lc[org[-1]] - lc[org[0]]) - 1)), 2),
                       "r1_std_bps": round(float(r1[org[0]:org[-1]].std()), 3)}


done = set()
files = sorted(glob.glob(f"{RESULTS}/*.npz"))
for path in files:
    tag = os.path.basename(path)[:-4]
    try:
        z = np.load(path, allow_pickle=True)
        org = z["origins"]
    except Exception:
        print(f"skip {tag}: unreadable")
        continue
    if not np.isin(org, ORG).all():
        print(f"skip {tag}: origins not in grid")
        continue
    step = int(np.median(np.diff(org)))
    label = "full" if np.array_equal(org, ORG) else f"n{len(org)}_step{step}"
    block = max(5, int(round(60 / step)))
    if label not in done:
        baselines(org, label, block)
        done.add(label)
    framing = str(z["framing"])
    q = z["q"].astype(np.float64)
    for hi, h in enumerate(HORIZONS):
        y = (lc[org + h] - lc[org]) * 1e4
        dp = z["dir_prob"][:, hi].astype(np.float64) if "dir_prob" in z else None
        if framing in ("price", "price_cov"):
            qb = (np.log(np.maximum(q[:, hi, :], 1e-9)) - lc[org][:, None]) * 1e4
            r = score(tag, h, org, y, qb, dir_prob=dp, ref=refs[(label, h)], block=block)
        elif framing == "logret":
            point = z["med_path"][:, :h].sum(axis=1).astype(np.float64)
            r = score(tag, h, org, y, q[:, hi, :] if h == 1 else None, point=point,
                      ref=refs[(label, h)] if h == 1 else None, block=block)
        else:
            continue
        r["set"] = label
        r["ms_per_series_batched"] = round(1000 * float(z["elapsed"]) / len(org), 2)
        rows.append(r)

json.dump({"data": info, "dataset": DATA, "baseline_fit": FIT, "sets": set_info, "rows": rows},
          open(f"{RESULTS}/metrics2.json", "w"), indent=1)

cols = ["set", "model", "h", "mae_skill_vs_rw", "dir_mcc", "dir_mcc_ci95", "auc_up", "auc_up_ci95", "pinball9_vs_ref_pct",
        "pinball9_vs_ref_ci95_pct", "qlike_vs_ref_pct", "qlike_vs_ref_ci95_pct", "cov80", "cov98", "hit_q01", "hit_q99",
        "vol_spearman"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in sorted(rows, key=lambda r: (r["set"], r["h"], r["model"])):
    lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
open(f"{RESULTS}/metrics2.md", "w").write("\n".join(lines) + "\n")
print(json.dumps({"baseline_fit": FIT, "sets": set_info}, indent=1))
print("\n".join(lines))
