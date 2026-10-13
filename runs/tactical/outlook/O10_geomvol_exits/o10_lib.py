"""O10 (SPEC.md): geometry and volatility as the EXIT of a trade. The entry is fixed = O9's ROBUST (breakout stop at m(1 +- 1.0 sigma_60), gated by the
regression p_up >= 0.52 / <= 0.48, 1x, one position at a time, valid 15 bars); only the exit varies. CPU only. Reuses o9_lib / o9_p2 (and through them o8_lib,
o5_lib, o1_lib) unchanged: decision frame, sigma_60 (L1), dV, p_up, the pessimistic fills, the freeze code, the bootstrap.

Exit rules (every rule keeps the protective stop at 1 sigma_60 from the entry and a time stop T):
  E0   target at 2 sigma_60 (ROBUST's own exit, O8's `_exit` unchanged)
  E1   trailing stop at c * sigma_15 from the best price of the CLOSED bars since entry (sigma_15 refreshed at every 15-minute decision; a ratchet)
  E2a  1-minute EMA20 cross against the side, on closed bars; market exit at the next open
  E2b  at a decision time: slope3 of the 15-minute EMA20 (g15.ema20.slope3 of O1's frame) points against the side; exit at that decision bar's open
  E3   at a decision time: dV (H72/T01 forecast of the next-hour change of ln RV) < 0
  E4   at a decision time: a logistic model (C 0.1, standardised) predicts that holding to the next checkpoint earns > 0; close if P < 0.5
All signals use bars < the bar they act on (decision features are those of the frame, the EMA is causal, E4's features are built from closed bars).
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE); O9DIR = os.path.join(OUTLOOK, "O9_ensemble_strategy")
sys.path.insert(0, O9DIR)
import o9_lib as L9  # noqa: E402
import o9_p2 as P2  # noqa: E402
while O9DIR in sys.path:
    sys.path.remove(O9DIR)
L8 = L9.L8; P5 = L9.P5; O1 = L9.O1; O3 = L9.O3; T = L9.T
SCRATCH = "D:/nt/nt_scratch/o10"
H = L9.H                                              # 60
VALID = L8.STEP                                       # orders valid 15 bars
TAU = 2.0                                             # E0 target
ENTRY = {"k": 1.0, "delta": 0.02, "g": None}          # ROBUST (choice_robust.json, ca1bd23)
EXITS = ("E0", "E1c1", "E1c2", "E2a", "E2b", "E3", "E4")
TS = (60, 240)
SEED = 0
NG = 15                                               # geometry features of E4 (O1's top 15)
E4_MIN = 300                                          # fewest checkpoints for a fitted E4 model
REASON = {"E1": "trail", "E2a": "ema", "E2b": "geo", "E3": "vol", "E4": "learn"}
_GN = O1.G.geom_names(None, O1.CROSS)
SLOPE_COL = len(_GN) + _GN.index("ema20.slope3")      # column of g15.ema20.slope3 in G = [G1 | G15]
POOL_MONTHS = L9.POOL_MONTHS; SEL_MONTHS = L9.SEL_MONTHS; SIG_MONTHS = L9.SIG_MONTHS


# ------------------------------------------------------------------ configurations
def grid():
    return [{"exit": e, "T": t} for t in TS for e in EXITS]


def cfg_name(c):
    return f"{c['exit']}_T{c['T']}"


def spec_of(c, models=None, plog=None, shuf=None, seed=SEED):
    """The exit specification of a configuration (dict). models: E4 models by month (or {'*': model}); plog: a list that collects (month, P) of E4's
    checkpoints; shuf: {month: array of P} (E4's SHUF arm draws its probabilities from the FULL arm's multiset of the same month)."""
    e = c["exit"]; kind = "E1" if e.startswith("E1") else e
    return {"kind": kind, "c": int(e[3]) if kind == "E1" else None, "T": int(c["T"]), "models": models or {}, "plog": plog, "shuf": shuf,
            "rng": np.random.default_rng([seed, 4242]) if shuf is not None else None}


# ------------------------------------------------------------------ context: bars, decision rows and the exit signals
def make_ctx(s, DF, S, gcols):
    """Everything an exit needs. s: bars; DF: decision frame (dict with a, m, ta, G); S: arrays over DF rows: sig (sigma_60), dv, p, s15 (sigma_15)."""
    O, Hi, Lo, Cl = (s[k].values for k in ("open", "high", "low", "close")); n = len(O); a = np.asarray(DF["a"])
    barrow = np.full(n, -1, np.int32); barrow[a] = np.arange(len(a), dtype=np.int32)
    ema = s["close"].ewm(span=20, adjust=False).mean().values                      # causal: bar j uses closes <= j
    pos = np.sign(Cl - ema); prev = np.concatenate([[0.0], pos[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        FB = np.column_stack([np.asarray(DF["G"])[:, gcols].astype(np.float64), np.log(S["s15"]), np.log(S["sig"]), S["dv"]])
    return {"O": O, "Hi": Hi, "Lo": Lo, "Cl": Cl, "n": n, "tmin": s.index.values.astype("datetime64[m]").astype(np.int64), "idx": s.index.values.astype("datetime64[ns]"),
            "a": a, "m": np.asarray(DF["m"]), "ta": DF["ta"], "barrow": barrow, "ema": ema, "evL": (pos < 0) & (prev >= 0), "evS": (pos > 0) & (prev <= 0),
            "slope": np.asarray(DF["G"])[:, SLOPE_COL].astype(np.float64), "dvx": np.asarray(S["dv"], float), "s15": np.asarray(S["s15"], float),
            "FB": FB, "mo_row": pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values}


def shuffle_bars(flags, idx, months, seed):
    """Permute a per-bar boolean signal within each calendar month of `months` (the event count per month is kept)."""
    out = flags.copy()
    for mo in months:
        m0, m1 = P5.month_bounds(mo); lo = int(np.searchsorted(idx, np.datetime64(m0, "ns"))); hi = int(np.searchsorted(idx, np.datetime64(m1, "ns")))
        if hi > lo:
            out[lo:hi] = np.random.default_rng([seed, int(pd.Period(mo, freq="M").ordinal)]).permutation(flags[lo:hi])
    return out


def make_shuf(ctx, sig, kind, months, seed=SEED):
    """The SHUF arm of an exit kind: the exit signal is permuted within each calendar month, the entries are unchanged. Returns (ctx2, sig_x).
    E0 has no signal: its stop and target distance (sigma_60 for the exit levels) is permuted instead. E4 is handled by `spec_of(shuf=...)`."""
    c2 = dict(ctx); sx = sig
    if kind == "E0":
        sx = P2.shuffle_month(sig, ctx["ta"], months, seed)
    elif kind == "E1":
        c2["s15"] = P2.shuffle_month(ctx["s15"], ctx["ta"], months, seed + 1)
    elif kind == "E2a":
        c2["evL"] = shuffle_bars(ctx["evL"], ctx["idx"], months, seed + 2); c2["evS"] = shuffle_bars(ctx["evS"], ctx["idx"], months, seed + 3)
    elif kind == "E2b":
        c2["slope"] = P2.shuffle_month(ctx["slope"], ctx["ta"], months, seed + 4)
    elif kind == "E3":
        c2["dvx"] = P2.shuffle_month(ctx["dvx"], ctx["ta"], months, seed + 5)
    return c2, sx


# ------------------------------------------------------------------ E1: the trailing stop
def trail_levels(side, p0, stop_px, ext, sg, c):
    """Effective stop level of every bar j = e+1 .. e+m (m = len(sg)). ext: the favourable extreme (high for a long, low for a short) of bars e+1 .. e+m-1
    (len m-1); the best price at bar j is the extreme of the CLOSED bars before j (and the entry price). sg: sigma_15 in force at each bar (nan = none).
    The level only moves in the position's favour (a ratchet) and never lies beyond the fixed 1-sigma_60 stop."""
    m = len(sg)
    if side == 1:
        best = np.maximum.accumulate(np.concatenate([[p0], ext]))[:m]
        raw = best * (1 - c * sg); lv = np.maximum.accumulate(np.where(np.isfinite(raw), raw, -np.inf)); return np.maximum(stop_px, lv)
    best = np.minimum.accumulate(np.concatenate([[p0], ext]))[:m]
    raw = best * (1 + c * sg); lv = np.minimum.accumulate(np.where(np.isfinite(raw), raw, np.inf)); return np.minimum(stop_px, lv)


def sigma15_path(ctx, e, m):
    """sigma_15 in force at bars e+1 .. e+m: the value of the latest decision row at or before the bar (the one before the entry for the first bars)."""
    r0 = int(np.searchsorted(ctx["a"], e, side="right")) - 1; init = ctx["s15"][r0] if r0 >= 0 else np.nan
    rr = ctx["barrow"][e + 1:e + 1 + m]; vals = np.full(m, np.nan); mk = rr >= 0; vals[mk] = ctx["s15"][rr[mk]]
    k = np.maximum.accumulate(np.where(np.isfinite(vals), np.arange(m), -1))
    return np.where(k >= 0, vals[np.maximum(k, 0)], init)


def _exit_trail(ctx, sp, e, side, p0, stop_px):
    Tt = sp["T"]; m = Tt - 1; O, Hi, Lo = ctx["O"], ctx["Hi"], ctx["Lo"]
    ext = (Hi if side == 1 else Lo)[e + 1:e + Tt - 1]
    eff = trail_levels(side, p0, stop_px, ext, sigma15_path(ctx, e, m), sp["c"])
    hit = (Lo[e + 1:e + Tt] <= eff) if side == 1 else (Hi[e + 1:e + Tt] >= eff)
    if hit.any():
        i = int(np.argmax(hit)); j = e + 1 + i; lv = float(eff[i])
        px = min(lv, float(O[j])) if side == 1 else max(lv, float(O[j]))
        return j, px, ("trail" if (lv > stop_px if side == 1 else lv < stop_px) else "stop"), False
    return e + Tt, float(O[e + Tt]), "time", True


# ------------------------------------------------------------------ E4: the learned exit
def cp_x(ctx, r, a, e, side, p0, s0):
    """E4's feature vector at the checkpoint of decision row r (bar a) of a position entered at bar e: O1's top-15 geometry features signed by the side,
    ln sigma_15, ln sigma_60, dV (all of the frame, bars < a), the unrealised return in entry-sigma_60 units at the close of bar a-1 (signed) and minutes in trade."""
    b = ctx["FB"][r]; un = side * np.log(ctx["Cl"][a - 1] / p0) / s0
    return np.concatenate([side * b[:NG], b[NG:], [un, float(a - e)]])


def e4_prob(ctx, sp, r, a, e, side, p0, s0):
    """P(holding to the next checkpoint earns > 0) or None (no model yet, or a feature is not finite). Collects/draws per the spec's plog / shuf."""
    mo = ctx["mo_row"][r]; mdl = sp["models"].get(mo) or sp["models"].get("*")
    if mdl is None:
        return None
    x = cp_x(ctx, r, a, e, side, p0, s0)
    if not np.isfinite(x).all():
        return None
    P = 1.0 / (1.0 + np.exp(-(((x - mdl["mean"]) / mdl["scale"]) @ mdl["coef"] + mdl["icpt"])))
    if sp.get("plog") is not None:
        sp["plog"].append((mo, float(P)))
    if sp.get("shuf") is not None:
        arr = sp["shuf"].get(mo)
        if arr is None or len(arr) == 0:
            return None
        P = float(sp["rng"].choice(arr))
    return P


def e4_records(ctx, tr):
    """Checkpoints of positions `tr` (a simulate10 output, E0 exits): for each decision bar a after the entry (up to the exit bar; before it for a time exit)
    the features, the label 'holding to the next checkpoint (or to the position's exit) earns > 0' and the time that label became known."""
    X, y, end = [], [], []; O = ctx["O"]; idx = ctx["idx"]
    for i in range(len(tr["ebar"])):
        e = int(tr["ebar"][i]); x = int(tr["xbar"][i]); side = int(tr["side"][i]); p0 = float(tr["epx"][i]); s0 = float(tr["sig"][i]); why = tr["reason"][i]
        lim = x - 1 if why == "time" else x
        if lim < e + 1:
            continue
        rr = ctx["barrow"][e + 1:lim + 1]; mk = rr >= 0; rows = rr[mk]; bars = np.arange(e + 1, lim + 1)[mk]
        for q in range(len(bars)):
            a = int(bars[q]); r = int(rows[q])
            if q + 1 < len(bars):
                nxt = float(O[bars[q + 1]]); en = idx[bars[q + 1]]
            else:
                nxt = float(tr["xpx"][i]); en = idx[x] + (np.timedelta64(0, "m") if why == "time" else np.timedelta64(1, "m"))
            xv = cp_x(ctx, r, a, e, side, p0, s0)
            if np.isfinite(xv).all():
                X.append(xv); y.append(side * np.log(nxt / O[a]) > 0); end.append(en)
    return {"X": np.array(X).reshape(len(X), NG + 5), "y": np.array(y, bool), "end": np.array(end, "datetime64[ns]")}


def cat_rec(recs):
    recs = [r for r in recs if r is not None and len(r["y"])]
    if not recs:
        return {"X": np.zeros((0, NG + 5)), "y": np.zeros(0, bool), "end": np.zeros(0, "datetime64[ns]")}
    return {k: np.concatenate([r[k] for r in recs]) for k in ("X", "y", "end")}


def e4_fit(rec, m0, min_n=None):
    """Logistic regression (C 0.1, standardised) on the checkpoints whose label became known BEFORE m0. None when fewer than `min_n` or one class only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    min_n = E4_MIN if min_n is None else min_n
    keep = rec["end"] < np.datetime64(m0, "ns"); y = rec["y"][keep]
    if keep.sum() < min_n or y.min() == y.max():
        return None
    sc = StandardScaler().fit(rec["X"][keep]); lr = LogisticRegression(C=L9.C_LR, max_iter=500).fit(sc.transform(rec["X"][keep]), y.astype(int))
    return {"mean": sc.mean_, "scale": sc.scale_, "coef": lr.coef_[0], "icpt": float(lr.intercept_[0]), "n": int(keep.sum()), "last_label_end": str(rec["end"][keep].max())}


def e4_models(rec, months):
    """Monthly walk-forward E4 models {month: model} (months without a model are left out: E4 does not exit there)."""
    out = {}
    for mo in months:
        m = e4_fit(rec, P5.month_bounds(mo)[0])
        if m is not None:
            out[mo] = m
    return out


def e4_to_json(m):
    return {"mean": m["mean"].tolist(), "scale": m["scale"].tolist(), "coef": m["coef"].tolist(), "icpt": m["icpt"], "n": m["n"], "last_label_end": m["last_label_end"]}


def e4_from_json(j):
    return {"mean": np.array(j["mean"]), "scale": np.array(j["scale"]), "coef": np.array(j["coef"]), "icpt": float(j["icpt"]), "n": j["n"], "last_label_end": j["last_label_end"]}


# ------------------------------------------------------------------ one trade's exit
def _first(f):
    return int(np.argmax(f)) if len(f) and f.any() else -1


def _signal_bar(ctx, sp, e, side, p0, s0, lim):
    """The bar whose OPEN is the first signal exit in e+1 .. lim, or None."""
    kind = sp["kind"]
    if kind == "E2a":                                   # signal on the closed bar j = e .. lim-1 -> exit at the open of j+1
        i = _first((ctx["evL"] if side == 1 else ctx["evS"])[e:lim]); return None if i < 0 else e + 1 + i
    if lim < e + 1:
        return None
    rr = ctx["barrow"][e + 1:lim + 1]; mk = rr >= 0; rows = rr[mk]; bars = np.arange(e + 1, lim + 1)[mk]
    if kind == "E2b":
        v = ctx["slope"][rows]; i = _first((v < 0) if side == 1 else (v > 0)); return None if i < 0 else int(bars[i])
    if kind == "E3":
        i = _first(ctx["dvx"][rows] < 0); return None if i < 0 else int(bars[i])
    for r, a in zip(rows, bars):                        # E4
        P = e4_prob(ctx, sp, int(r), int(a), e, side, p0, s0)
        if P is not None and P < 0.5:
            return int(a)
    return None


def exit_trade(ctx, sp, e, side, p0, sx, s0, pess):
    """(exit bar, exit price, reason, market_exit_at_open) of a position entered at bar e at price p0. sx: the sigma_60 that sizes the stop (and the E0 target),
    s0: the entry sigma_60 (E4's unit). The protective stop is O8/O9's, with the pessimistic entry-bar rule; signals act at the open of a later bar."""
    O, Hi, Lo, Cl = ctx["O"], ctx["Hi"], ctx["Lo"], ctx["Cl"]; Tt = sp["T"]; kind = sp["kind"]
    stop_px = p0 * (1 - 1.0 * sx) if side == 1 else p0 * (1 + 1.0 * sx)
    if pess and ((Lo[e] <= stop_px) if side == 1 else (Hi[e] >= stop_px)):
        return e, float(min(stop_px, Cl[e]) if side == 1 else max(stop_px, Cl[e])), "stop", False
    if kind == "E1":
        return _exit_trail(ctx, sp, e, side, p0, stop_px)
    tgt = p0 * (1 + side * TAU * sx) if kind == "E0" else None
    xb, xp, why = L8._exit(O, Hi, Lo, e, Tt, side, stop_px, tgt)
    if kind == "E0":
        return xb, xp, why, why == "time"
    sb = _signal_bar(ctx, sp, e, side, p0, s0, xb - 1 if why == "time" else xb)
    if sb is not None:
        return sb, float(O[sb]), REASON[kind], True
    return xb, xp, why, why == "time"


# ------------------------------------------------------------------ the engine
def simulate10(ctx, rows, sp, sig, sig_x, allow_long, allow_short, k=ENTRY["k"], pess=True, valid=VALID):
    """O9's breakout engine (`simulate9`: permissions per side, one position at a time, orders valid 15 bars) with the exit of `sp` (exit_trade).
    rows: DF rows of the decisions; sig / sig_x / allow_*: arrays over DF rows. Output arrays as simulate9, plus `row` (the DF row) and gross log `ret`."""
    n, Tt = ctx["n"], sp["T"]; O, Hi, Lo = ctx["O"], ctx["Hi"], ctx["Lo"]; tmin = ctx["tmin"]
    rows = np.asarray(rows); dec = ctx["a"][rows]; mm = ctx["m"][rows]; sg = sig[rows]
    good = np.isfinite(sg) & (sg > 0) & (dec + valid < n) & (dec + valid + Tt < n)
    ok_g = np.zeros(len(dec), bool); gi = np.where(good)[0]
    ok_g[gi] = (tmin[dec[gi] + valid + Tt] - tmin[dec[gi]]) == valid + Tt; good &= ok_g
    idx = np.where(good)[0]; d = dec[idx]; m_ = mm[idx]; s_ = sg[idx]; sx_ = sig_x[rows][idx]
    aL = allow_long[rows][idx]; aS = allow_short[rows][idx]
    W = d[:, None] + np.arange(valid)[None, :]; Ow, Hw, Lw = O[W], Hi[W], Lo[W]
    Lv = m_ * (1 - k * s_); Uv = m_ * (1 + k * s_)
    hb = (Hw >= Uv[:, None]) & aL[:, None]; hs = (Lw <= Lv[:, None]) & aS[:, None]
    big = valid + 1
    t_long = np.where(hb.any(1), hb.argmax(1), big); t_short = np.where(hs.any(1), hs.argmax(1), big)
    has = (t_long < big) | (t_short < big); both = (t_long == t_short) & (t_long < big)
    ar = np.arange(len(d)); tl = np.minimum(t_long, valid - 1); ts_ = np.minimum(t_short, valid - 1)
    o_l = Ow[ar, tl]; o_s = Ow[ar, ts_]
    take_long = np.where(both, np.abs(o_l - Uv) <= np.abs(o_s - Lv), t_long < t_short)
    ebar_rel = np.where(take_long, t_long, t_short); o_e = Ow[ar, np.minimum(ebar_rel, valid - 1)]
    epx = np.where(take_long, np.maximum(Uv, o_e), np.minimum(Lv, o_e))
    res = {"row": [], "dec": [], "ebar": [], "xbar": [], "side": [], "epx": [], "xpx": [], "reason": [], "sig": []}
    free_from = -1
    for jj in np.where(has)[0]:
        dj = int(d[jj])
        if dj < free_from:
            continue
        e = dj + int(ebar_rel[jj]); side = 1 if take_long[jj] else -1; p0 = float(epx[jj]); s0 = float(s_[jj])
        xbar, xpx, why, at_open = exit_trade(ctx, sp, e, side, p0, float(sx_[jj]), s0, pess)
        res["row"].append(int(rows[idx[jj]])); res["dec"].append(dj); res["ebar"].append(e); res["xbar"].append(int(xbar)); res["side"].append(side)
        res["epx"].append(p0); res["xpx"].append(xpx); res["reason"].append(why); res["sig"].append(s0)
        free_from = xbar if at_open else xbar + 1
    out = {k_: np.array(v) for k_, v in res.items()}
    out["ret"] = np.log(out["xpx"] / out["epx"]) * out["side"] if len(out["dec"]) else np.array([])
    return out


TCOLS = ["decision_time", "entry_time", "exit_time", "side", "entry_px", "exit_px", "reason", "ret_gross", "w", "sigma", "p_up", "dV", "hold"]


def trades10(ctx, rows, sp, S, sig_x, aL, aS, pess):
    """One configuration on decision rows: the trade frame (O9's columns plus `hold`, the bars from the entry bar to the exit bar)."""
    tr = simulate10(ctx, rows, sp, S["sig"], sig_x, aL, aS, pess=pess)
    return to_frame(ctx, tr, S), tr


def to_frame(ctx, tr, S):
    if len(tr["dec"]) == 0:
        return pd.DataFrame({c: [] for c in TCOLS})
    r = tr["row"]; ix = ctx["idx"]
    return pd.DataFrame({"decision_time": ix[tr["dec"]], "entry_time": ix[tr["ebar"]], "exit_time": ix[tr["xbar"]], "side": tr["side"], "entry_px": tr["epx"], "exit_px": tr["xpx"],
                         "reason": tr["reason"], "ret_gross": tr["ret"], "w": np.ones(len(r)), "sigma": tr["sig"], "p_up": S["p"][r], "dV": S["dv"][r], "hold": tr["xbar"] - tr["ebar"]})


# ------------------------------------------------------------------ statistics
def summarise10(tf, months):
    """Selection numbers of one trade frame (w = 1): trades per month, monthly mean net at 0/2/5 bps, t at 2 bps, months positive, hit, average win and
    loss (bps, net of 2 bps), mean holding minutes, max drawdown, exit reasons."""
    n = len(tf); out = {"n_trades": int(n), "trades_per_month": n / len(months)}
    for c in L8.COSTS_BPS:
        mr = P2.monthly_tf(tf, months, c); out[f"mean_m_{int(c)}bps"] = float(mr.mean())
        if c == 2.0:
            out["t_2bps"] = L8.t_stat(mr); out["months_pos_2bps"] = int((mr > 0).sum()); out["n_months"] = len(months)
    net = tf["ret_gross"].values - 2e-4 if n else np.array([])
    out["hit_2bps"] = float((net > 0).mean()) if n else None
    out["avg_win_bps"] = float(net[net > 0].mean() * 1e4) if n and (net > 0).any() else None
    out["avg_loss_bps"] = float(net[net <= 0].mean() * 1e4) if n and (net <= 0).any() else None
    out["mean_hold_min"] = float(tf["hold"].mean()) if n else None
    out["max_dd_2bps"] = L8.max_drawdown(net)
    out["reasons"] = {k: int(v) for k, v in tf["reason"].value_counts().items()} if n else {}
    out["long_share"] = float((tf["side"] == 1).mean()) if n else None
    return out


def paired_monthly(a, b):
    """a - b per month: mean, 95% t interval (t01_lib.ci), t, months a is better."""
    d = np.asarray(a, float) - np.asarray(b, float); mu, lo, hi, t = T.ci(d)
    return {"mean": float(mu), "lo": float(lo), "hi": float(hi), "t": float(t), "months_better": int((d > 0).sum()), "n": int(len(d))}


# ------------------------------------------------------------------ signals: sigma_15
def sigma15_walk(TF, DF, months, frozen_m0=None):
    """sigma_15 (L1 ridge, the O8 recipe at H = 15) at the decision rows of `months` (monthly walk-forward, or one fit at frozen_m0)."""
    arms, _ = L8.sigma_arms(TF, DF, months, 15, frozen_m0=frozen_m0); return arms["L1"].copy()


def sigma15_frozen(state, DF, months):
    """sigma_15 with a frozen H = 15 state at the decision rows of `months` (nan where the features are invalid)."""
    out = np.full(len(DF["ta"]), np.nan)
    for mo in months:
        rows = L9.month_rows(DF["ta"], mo); rows = rows[DF["ok"][rows]]
        if len(rows):
            out[rows] = np.sqrt(np.exp(P5.predict(state, DF, rows, models=("S3", "L1"))["L1"]))
    return out


def gcols():
    return L9.geom_cols()


# ------------------------------------------------------------------ the frozen artefact
def freeze_core(art9, bars, st15_fit, e4_fit_json, choice):
    """The O10 artefact: O9's frozen models (identical fit: same code, same bars) + the H = 15 sigma state + E4's models per T."""
    art = {k: v for k, v in art9.items() if k != "sha256"}
    art.update({"spec": "O10 SPEC.md freeze", "o9_frozen_sha256": art9["sha256"], "sigma15": P5.state_to_json(st15_fit), "e4": e4_fit_json, "choice": choice,
               "lib_note": "o10_lib / o9_lib / o9_p2 with o8_lib, o5_lib, o3_lib, o1_lib, t01_lib as in the repository at the freeze commit"})
    art["sha256"] = P5.sha_of(art); return art


def load_artefact10(art):
    """Models of the O10 artefact (raises when the sha256 does not match): O9's sigma/dv/pup, the H = 15 sigma state, E4 per T."""
    m = P2.load_artefact(art)                           # checks the sha256
    m["sigma15"] = P5.state_from_json(art["sigma15"]); m["e4"] = {int(t): e4_from_json(j) for t, j in art["e4"].items() if j is not None}
    return m


def signals_frozen10(models, DF, DX, months):
    """sig (sigma_60 L1), dv, p, s15 with the frozen models at the decision rows of `months`."""
    S = P2.signals_frozen(models, DF, DX, months)
    return {"sig": S["L1"], "dv": S["dv"], "p": S["p"], "s15": sigma15_frozen(models["sigma15"], DF, months)}


def freeze_fit10(bars, m0, choice, art9=None, log=None):
    """Fit everything ONCE on `bars` (labels ending at or before m0): O9's models (sigma_60, dV, p_up; `art9` = a stored O9 artefact of the SAME bars and code, else
    refitted here), the H = 15 sigma state and E4's logistic model per T (on the checkpoints of E0 positions over these bars, labels known before m0)."""
    if art9 is None:
        art9 = P2.freeze_fit(bars, m0)
    assert P5.sha_of(art9) == art9["sha256"], "the O9 artefact does not match its sha256"
    F = P5.build_frame(bars, O3.FIRST); st15 = P5.fit_h(F, 15, m0, models=("S3", "L1")); assert pd.Timestamp(st15["last_label_end"]) <= m0
    models = P2.load_artefact(art9); models["sigma15"] = st15
    DF, DX = P2.make_dframe_for(bars, O3.FIRST + 1); months = sorted(set(pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str)))
    S = signals_frozen10(models, DF, DX, months); ctx = make_ctx(bars, DF, S, gcols()); rows = np.where(np.isfinite(S["sig"]))[0]
    aL, aS = L9.gate_masks(S["p"], S["dv"], ENTRY["delta"], ENTRY["g"]); e4 = {}
    for Tt in TS:
        tr = simulate10(ctx, rows, spec_of({"exit": "E0", "T": Tt}), S["sig"], S["sig"], aL, aS, pess=True); rec = e4_records(ctx, tr); m = e4_fit(rec, m0)
        if log:
            log("freeze E4 T", Tt, "E0 positions", len(tr["dec"]), "checkpoints", len(rec["y"]), "model", None if m is None else m["n"])
        e4[str(Tt)] = None if m is None else e4_to_json(m)
    return freeze_core(art9, bars, st15, e4, choice)
