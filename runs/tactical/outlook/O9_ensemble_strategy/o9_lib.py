"""O9 (SPEC.md): an ensemble breakout strategy. Gates from three forecasts at each decision (every 15 minutes on the clock):
  sigma_60 (O8: L1 ridge), dV (T01's S3 ridge on the volatility CHANGE target dA), p_up (logistic regression on fs_tb7 of the 60 bars before t),
plus a target exit, a vol-targeted size and a meta-labelling variant (META). CPU only. Reuses o8_lib unchanged (decision frame, sigma, `_exit`,
metrics). Everything at a decision uses bars < t; every model is a monthly walk-forward fit on the 365 days before the month with labels ending
at or before the month start.

Reading of the SPEC grid: delta = None means the regression gate is OFF (the SPEC's NO_REG arm: "delta = 0, the regression gate off"); delta = 0.02
is the gate p_up >= 0.52 (long) / <= 0.48 (short). g = None is "no volatility gate", g = 0.0 is dV >= 0.
"""
import json, os, sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE); TAC = os.path.dirname(OUTLOOK)
sys.path.insert(0, os.path.join(OUTLOOK, "O8_volband_strategy"))
import o8_lib as L8  # noqa: E402
sys.path.remove(os.path.join(OUTLOOK, "O8_volband_strategy"))
sys.path.insert(0, os.path.join(TAC, "lab"))
import lab as LAB  # noqa: E402  (fs_tb7; read-only import)
sys.path.remove(os.path.join(TAC, "lab"))
P5 = L8.P5; O1 = L8.O1; O3 = L8.O3; T = L8.T
SCRATCH = "D:/nt/nt_scratch/o9"
N_S3 = P5.N_S3                                    # 16
H = 60                                            # strategy horizon: sigma_60, dV target, p_up label, time stop
DEAD = 5e-4                                       # 5 bps dead band for p_up's training labels
C_LR = 0.1
SIZE_CAP = 2.0
STEP = L8.STEP
META_MIN_TRADES = 500
META_Q = 0.60
POOL_FIRST = "2019-01"                            # first month of the META trade pool (sigma, dV, p_up walk-forward start here)
SEL_MONTHS = list(L8.S_SEL)                       # 2021-01 .. 2024-06
POOL_MONTHS = [str(m) for m in pd.period_range(POOL_FIRST, "2024-06", freq="M")]
SIG_MONTHS = [str(m) for m in pd.period_range("2018-12", "2024-06", freq="M")]   # 2018-12 only supplies the first month's sigma_ref
K_META = 1.5


# ------------------------------------------------------------------ tb7 at arbitrary anchors
def tb7_at(A, anchors, chunk=20000):
    """fs_tb7 (13 features) of the 60 bars before each anchor (A: [n, 5] open, high, low, close, volume)."""
    a = np.asarray(anchors); A = np.asarray(A, np.float64); W = swv(A, (H, 5))[:, 0]          # [n-59, 60, 5]; window j = bars j..j+59
    out = []
    for i in range(0, len(a), chunk):
        w = W[a[i:i + chunk] - H].astype(np.float32); out.append(np.nan_to_num(LAB.fs_tb7(w)))
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 13), np.float32)


# ------------------------------------------------------------------ training-grid extras (stride-7 frame of O6)
def train_extras(s, ta):
    """For the stride-7 frame times `ta` (= close time of bar a-1): the bar index a, the vol-change target dA = ln RV(a, a+60) - ln RV(a-60, a)
    (T01's, with sum of squared 1-minute close log returns), ret60 = open[a+60] / open[a] - 1 (nan across a data gap) and tb7."""
    A = s.values.astype(np.float64); idx = s.index; n = len(A)
    pos = idx.searchsorted(pd.DatetimeIndex(ta) - pd.Timedelta(minutes=1)); assert (idx[pos] == pd.DatetimeIndex(ta) - pd.Timedelta(minutes=1)).all()
    a = pos + 1
    lr = np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]); cs = np.concatenate([[0.0], np.cumsum(lr ** 2)])
    ok = a + H < n; a_ = np.where(ok, a, 0)
    nxt = np.where(ok, cs[np.minimum(a_ + H, n)] - cs[a_], np.nan); prv = cs[a_] - cs[a_ - H]
    dA = np.log(nxt + 1e-12) - np.log(prv + 1e-12)
    tm = idx.values.astype("datetime64[m]").astype(np.int64)
    cont = ok & ((tm[np.minimum(a_ + H, n - 1)] - tm[a_]) == H)
    ret60 = np.where(cont, A[np.minimum(a_ + H, n - 1), 0] / A[a_, 0] - 1, np.nan)
    return {"a": a, "dA": dA, "ret60": ret60, "tb7": tb7_at(A, np.where(ok, a, H))}


def load_train_extras(TF, s):
    os.makedirs(SCRATCH, exist_ok=True); p = os.path.join(SCRATCH, "train_extra.npz")
    if os.path.exists(p):
        z = np.load(p); X = {k: z[k] for k in z.files}
        assert len(X["a"]) == len(TF["ta"])
    else:
        X = train_extras(s, TF["ta"]); np.savez(p, **X)
    X["ta"] = TF["ta"]; X["S3"] = np.ascontiguousarray(TF["XB"][:, :N_S3]); X["ok60"] = TF["ok60"]
    X["dok"] = X["ok60"] & np.isfinite(X["dA"])
    X["pok"] = np.isfinite(X["ret60"]) & (np.abs(X["ret60"]) > DEAD) & np.isfinite(X["tb7"]).all(1) & TF["ok60"]
    return X


def load_dec_extras(DF, s):
    os.makedirs(SCRATCH, exist_ok=True); p = os.path.join(SCRATCH, "dec_extra.npz")
    if os.path.exists(p):
        z = np.load(p); tb = z["tb7"]; assert len(tb) == len(DF["a"])
    else:
        tb = tb7_at(s.values.astype(np.float64), DF["a"]); np.savez(p, tb7=tb)
    return {"tb7": tb}


# ------------------------------------------------------------------ the two walk-forward models
def dv_fit(X, m0):
    """T01's S3 ridge (alpha 1 on standardised S3) for the next-hour vol-change target dA. Rows: the 365 days before m0, labels ending at or before m0."""
    tr = O3.train_rows(X["ta"], X["dok"], H, m0); assert len(tr) > 0
    return {"m": O3.ridge_fit(X["S3"][tr], X["dA"][tr]), "n_tr": int(len(tr)), "last_label_end": pd.Timestamp(X["ta"][tr].max()) + pd.Timedelta(minutes=H)}


def pup_fit(X, m0):
    """Logistic regression (L2, C 0.1, standardised) on fs_tb7; label = sign of the next-open 60-minute return, rows with |ret| <= 5 bps dropped.
    Rows: the 365 days before m0, labels (the open 60 minutes later) ending at or before m0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = O3.train_rows(X["ta"], X["pok"], H, m0); assert len(tr) > 0
    y = (X["ret60"][tr] > 0).astype(int); sc = StandardScaler().fit(X["tb7"][tr])
    lr = LogisticRegression(C=C_LR, max_iter=500).fit(sc.transform(X["tb7"][tr]), y)
    return {"sc": sc, "lr": lr, "n_tr": int(len(tr)), "last_label_end": pd.Timestamp(X["ta"][tr].max()) + pd.Timedelta(minutes=H), "base": float(y.mean())}


def pup_predict(st, tb7):
    return st["lr"].predict_proba(st["sc"].transform(tb7))[:, 1]


def month_rows(ta, mo):
    m0, m1 = P5.month_bounds(mo); return L8.month_rows(ta, m0, m1)


def signals(TF, X, DF, DX, months, log=None):
    """sigma_60 (L1), dV and p_up at the decision rows of `months` (nan elsewhere and where the features are invalid)."""
    n = len(DF["ta"]); sig = np.full(n, np.nan); dv = np.full(n, np.nan); pu = np.full(n, np.nan)
    arms, _ = L8.sigma_arms(TF, DF, months, H); sig[:] = arms["L1"]
    for mo in months:
        m0, _ = P5.month_bounds(mo); rows = month_rows(DF["ta"], mo); rows = rows[DF["ok"][rows]]
        if len(rows) == 0:
            continue
        sd = dv_fit(X, m0); assert sd["last_label_end"] <= m0
        dv[rows] = O3.ridge_pred(sd["m"], DF["XB"][rows][:, :N_S3])
        sp = pup_fit(X, m0); assert sp["last_label_end"] <= m0
        pu[rows] = pup_predict(sp, DX["tb7"][rows])
        if log:
            log("signals", mo, "rows", len(rows), "dV n_tr", sd["n_tr"], "p_up n_tr", sp["n_tr"], "base", round(sp["base"], 3))
    bad = ~(np.isfinite(sig) & np.isfinite(dv) & np.isfinite(pu)); sig[bad] = np.nan; dv[bad] = np.nan; pu[bad] = np.nan
    return {"sigma": sig, "dv": dv, "p": pu}


# ------------------------------------------------------------------ gates and size
def gate_masks(p, dv, delta, g):
    """Per-decision order permissions. delta None = the regression gate is off; else long needs p >= 0.5 + delta, short p <= 0.5 - delta.
    g None = no volatility gate; else both sides need dV >= g. A nan input blocks the side that needs it."""
    p = np.asarray(p, float); dv = np.asarray(dv, float); n = len(p)
    al = np.ones(n, bool); as_ = np.ones(n, bool)
    if delta is not None:
        al &= p >= 0.5 + delta; as_ &= p <= 0.5 - delta
    if g is not None:
        gv = dv >= g; al &= gv; as_ &= gv
    return al, as_


def sigma_ref_by_month(sig, ta, months):
    """{month: median sigma_60 of the previous calendar month's valid decisions} for `months` (nan when the previous month has none)."""
    mo_of = pd.DatetimeIndex(ta).to_period("M").astype(str).values; out = {}
    for mo in months:
        prev = str(pd.Period(mo, freq="M") - 1); r = np.isfinite(sig) & (mo_of == prev)
        out[mo] = float(np.median(sig[r])) if r.any() else np.nan
    return out


def size_weights(sig, ref, cap=SIZE_CAP):
    """w = min(cap, sigma_ref / sigma_60) (nan when ref or sigma is not usable)."""
    sig = np.asarray(sig, float); ref = np.asarray(ref, float)
    return np.minimum(cap, ref / sig)


# ------------------------------------------------------------------ engine (BRK with per-side permissions, a target, both fill variants)
def simulate9(O, Hi, Lo, Cl, dec, m, sig, k, Hh, stop_s, allow_long=None, allow_short=None, tau=None, valid=STEP, tmin=None, pess=False):
    """O8's BRK engine with (a) a permission per side at each decision (an order that is not permitted is not placed), (b) an optional target at
    tau * sigma beyond the entry price (a limit sell above a long, with O8's 1 bp trade-through rule; a stop and a target in one bar resolve to the
    stop). Output as L8.simulate, plus `j`: the position of each trade's decision in the input arrays."""
    n = len(O); dec = np.asarray(dec); m = np.asarray(m, float); sig = np.asarray(sig, float)
    good = np.isfinite(sig) & (sig > 0) & (dec + valid < n) & (dec + valid + Hh < n)
    if tmin is not None:
        ok_g = np.zeros(len(dec), bool); gi = np.where(good)[0]
        ok_g[gi] = (tmin[dec[gi] + valid + Hh] - tmin[dec[gi]]) == valid + Hh; good &= ok_g
    idx = np.where(good)[0]
    d = dec[idx]; mm = m[idx]; sg = sig[idx]
    aL = np.ones(len(d), bool) if allow_long is None else np.asarray(allow_long, bool)[idx]
    aS = np.ones(len(d), bool) if allow_short is None else np.asarray(allow_short, bool)[idx]
    W = d[:, None] + np.arange(valid)[None, :]
    Ow, Hw, Lw = O[W], Hi[W], Lo[W]
    Lv = mm * (1 - k * sg); Uv = mm * (1 + k * sg)
    hb = (Hw >= Uv[:, None]) & aL[:, None]; hs = (Lw <= Lv[:, None]) & aS[:, None]
    big = valid + 1
    t_long = np.where(hb.any(1), hb.argmax(1), big); t_short = np.where(hs.any(1), hs.argmax(1), big)
    has = (t_long < big) | (t_short < big); both = (t_long == t_short) & (t_long < big)
    ar = np.arange(len(d)); tl = np.minimum(t_long, valid - 1); ts_ = np.minimum(t_short, valid - 1)
    o_l = Ow[ar, tl]; o_s = Ow[ar, ts_]
    take_long = np.where(both, np.abs(o_l - Uv) <= np.abs(o_s - Lv), t_long < t_short)
    ebar_rel = np.where(take_long, t_long, t_short); o_e = Ow[ar, np.minimum(ebar_rel, valid - 1)]
    epx = np.where(take_long, np.maximum(Uv, o_e), np.minimum(Lv, o_e))
    res = {"j": [], "dec": [], "ebar": [], "xbar": [], "side": [], "epx": [], "xpx": [], "reason": [], "both": [], "sig": []}
    free_from = -1
    for jj in np.where(has)[0]:
        dj = int(d[jj])
        if dj < free_from:
            continue
        e = dj + int(ebar_rel[jj]); side = 1 if take_long[jj] else -1; p0 = float(epx[jj]); s0 = float(sg[jj])
        stop_px = None if stop_s is None else (p0 * (1 - stop_s * s0) if side == 1 else p0 * (1 + stop_s * s0))
        tgt = None if tau is None else p0 * (1 + side * tau * s0)
        if pess and stop_px is not None and ((Lo[e] <= stop_px) if side == 1 else (Hi[e] >= stop_px)):
            xbar, why = e, "stop"; xpx = float(min(stop_px, Cl[e]) if side == 1 else max(stop_px, Cl[e]))
        else:
            xbar, xpx, why = L8._exit(O, Hi, Lo, e, Hh, side, stop_px, tgt)
        res["j"].append(int(idx[jj])); res["dec"].append(dj); res["ebar"].append(e); res["xbar"].append(xbar); res["side"].append(side); res["epx"].append(p0)
        res["xpx"].append(xpx); res["reason"].append(why); res["both"].append(bool(both[jj])); res["sig"].append(s0)
        free_from = xbar if why == "time" else xbar + 1
    out = {k_: np.array(v) for k_, v in res.items()}
    out["ret"] = np.log(out["xpx"] / out["epx"]) * out["side"] if len(out["dec"]) else np.array([])
    return out


# ------------------------------------------------------------------ META (meta-labelling of BRK k = 1.5 triggers)
def geom_cols():
    """The 15 largest-coefficient geometry features of O1 (frozen interp.json, names) as column indices into G = [G1 | G15]."""
    top = json.load(open(os.path.join(OUTLOOK, "O1_geom_vol", "interp.json")))["top"][:15]
    gn = O1.G.geom_names(None, O1.CROSS); cols = []
    for t in top:
        pre, nm = t["name"].split(".", 1); i = gn.index(nm); cols.append(i if pre == "g1" else len(gn) + i)
        assert cols[-1] == t["col"] - N_S3, t
    return np.array(cols)


def meta_X(p, dv, sig, G15, hour, side):
    """[p_up aligned to the side, dV, ln sigma_60, 15 geometry features, hour sin, hour cos]."""
    pa = np.where(np.asarray(side) == 1, p, 1 - np.asarray(p))
    return np.column_stack([pa, dv, np.log(sig), np.asarray(G15, np.float64), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])


def meta_fit(Xf, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xf); return {"sc": sc, "lr": LogisticRegression(C=C_LR, max_iter=500).fit(sc.transform(Xf), y), "n": int(len(y))}


def meta_score(st, Xf):
    return st["lr"].predict_proba(st["sc"].transform(Xf))[:, 1]


def meta_train_rows(t_exit, m0):
    """Candidate trades usable for a model starting at m0: those whose exit is known by m0 (exit bar closed at or before m0)."""
    return np.where(np.asarray(t_exit, "datetime64[ns]") <= np.datetime64(m0, "ns"))[0]


def meta_gate(cand, DF, sig, dv, p, gcols, months, log=None):
    """Walk-forward META. `cand`: the candidate trades (BRK k=1.5, ungated, PESS, 1x) with arrays row (decision row), side, t_exit, y (net at 2 bps > 0).
    For each month in order: fit on candidates whose exit precedes the month, score the month's two potential orders at every decision, the threshold
    is the META_Q quantile of the PREVIOUS month's candidate-trade scores (scored by that month's model). Returns (allow_long, allow_short, info)."""
    n = len(DF["ta"]); aL = np.zeros(n, bool); aS = np.zeros(n, bool)
    hour = pd.DatetimeIndex(DF["ta"]).hour.values; mo_of = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values
    G15 = DF["G"][:, gcols]
    cmo = mo_of[cand["row"]]; prev_scores = None; info = {}
    for mo in months:
        m0, _ = P5.month_bounds(mo); tr = meta_train_rows(cand["t_exit"], m0); st = None
        rows = np.where((mo_of == mo) & np.isfinite(sig))[0]
        if len(tr) >= META_MIN_TRADES and 0 < cand["y"][tr].mean() < 1:
            r = cand["row"][tr]
            st = meta_fit(meta_X(p[r], dv[r], sig[r], G15[r], hour[r], cand["side"][tr]), cand["y"][tr])
        thr = float(np.quantile(prev_scores, META_Q)) if (st is not None and prev_scores is not None and len(prev_scores) > 20) else np.nan
        if st is not None and len(rows):
            sL = meta_score(st, meta_X(p[rows], dv[rows], sig[rows], G15[rows], hour[rows], np.ones(len(rows))))
            sS = meta_score(st, meta_X(p[rows], dv[rows], sig[rows], G15[rows], hour[rows], -np.ones(len(rows))))
            if np.isfinite(thr):
                aL[rows] = sL >= thr; aS[rows] = sS >= thr
        ci = np.where(cmo == mo)[0]
        if st is not None and len(ci):
            r = cand["row"][ci]; prev_scores = meta_score(st, meta_X(p[r], dv[r], sig[r], G15[r], hour[r], cand["side"][ci]))
        else:
            prev_scores = None
        info[mo] = {"n_train": int(len(tr)), "thr": thr, "n_cand": int(len(ci))}
        if log:
            log("META", mo, info[mo])
    return aL, aS, info


# ------------------------------------------------------------------ metrics (weighted)
def monthly9(ret, w, em, months, cost_bps):
    """Sum of w * (log return - cost) by entry month; 0 for a month with no trade."""
    if len(ret) == 0:
        return np.zeros(len(months))
    g = pd.Series(np.asarray(w) * (np.asarray(ret) - cost_bps * 1e-4), index=pd.Index(em)).groupby(level=0).sum()
    return np.array([float(g.get(mo, 0.0)) for mo in months])


def summarise9(tr, w, em, months):
    """Selection numbers of one trade list: trades (per month), monthly mean net at 0/2/5 bps, t at 2 bps, months positive, hit, max drawdown, reasons."""
    ret = tr["ret"]; n = len(ret); out = {"n_trades": int(n), "trades_per_month": n / len(months)}
    for c in L8.COSTS_BPS:
        mr = monthly9(ret, w, em, months, c); out[f"mean_m_{int(c)}bps"] = float(mr.mean())
        if c == 2.0:
            out["t_2bps"] = L8.t_stat(mr); out["months_pos_2bps"] = int((mr > 0).sum()); out["n_months"] = len(months)
    net = np.asarray(w) * (ret - 2e-4) if n else np.array([])
    out["hit_2bps"] = float((net > 0).mean()) if n else None
    out["max_dd_2bps"] = L8.max_drawdown(net)
    out["reasons"] = {r: int((tr["reason"] == r).sum()) for r in ("target", "stop", "time")} if n else {}
    out["long_share"] = float((tr["side"] == 1).mean()) if n else None
    out["mean_w"] = float(np.mean(w)) if n else None
    return out


def grid():
    """The SPEC grid read literally: k x delta x g x tau x size = 32 (delta None / g None = gate off)."""
    return [{"k": k, "delta": d, "g": g, "tau": t, "size": sz} for k in (1.0, 1.5) for d in (None, 0.02) for g in (None, 0.0) for t in (None, 2.0) for sz in ("1x", "vt")]


def cfg_name(c):
    if c.get("meta"):
        return f"META_k{c['k']}"
    f = lambda v: "off" if v is None else v
    return f"ENS_k{c['k']}_d{f(c['delta'])}_g{f(c['g'])}_tau{f(c['tau'])}_sz{c['size']}"
