"""O8 (SPEC.md): a volatility-band strategy on the geometry-enhanced size forecast (L1 of O5). CPU only.

Two parts:
  1. sigma at the strategy's decision times (every 15 minutes on the clock). The O5/O6 frame lives on a stride-7 grid, so the features are
     recomputed at the decision anchors with the SAME feature code (`decision_features`, a restatement of t01_lib.build / O3.build_features for an
     arbitrary anchor list; test_o8 asserts equality with the frame code on anchors of both grids). The monthly walk-forward ridge is fitted
     EXACTLY as O5 does (o5_lib.fit_h on the stride-7 training rows); only the prediction rows change.
  2. the backtest engine on 1-minute OHLC (`simulate`), every fill pessimistic, per the SPEC "Mechanics".
Anchor a (the index of the decision bar) sees bars < a only; the decision price m is the open of bar a.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUTLOOK, "O6_geom_hourly"))
import o6_lib as P6  # noqa: E402  (imports o5_lib, o1_lib, o3_lib, t01_lib)
sys.path.remove(os.path.join(OUTLOOK, "O6_geom_hourly"))
P5 = P6.P5; O1 = P6.O1; O3 = P6.O3; T = P6.T; V = T.V
SCRATCH = "D:/nt/nt_scratch/o8"
FRAME = P6.SCRATCH + "/frame.npz"                          # O6's cache: XB, G (G1 + G15 = first 328 columns), rv, ok, on the stride-7 grid
N_L1G = P6.N_G1G15
LOAD_FROM = P6.LOAD_FROM; LOAD_HI = T.CUTOFF               # the whole cut file; no bar after 2025-07-20 23:59
FIRST = P6.FIRST6
HZ = (15, 60)                                              # strategy horizons
STEP = 15                                                  # decision every 15 minutes on the clock; orders valid for 15 bars
TT = 1e-4                                                  # trade-through for limit fills: 1 bp
ARMS = ("L1", "R0", "TRAIL", "SHUF")
S_SEL = list(P5.S_MONTHS)                                  # 2021-01 .. 2024-06
SPANS = {"sel": S_SEL,
         "A": [str(m) for m in pd.period_range("2018-07", "2020-12", freq="M")],
         "B": [str(m) for m in pd.period_range("2024-07", "2025-07", freq="M")]}
COSTS_BPS = (0.0, 2.0, 5.0)


# ------------------------------------------------------------------ bars
def load_bars_cut():
    """OHLCV of the cut file 2017-08-17 .. 2025-07-20 23:59 (cached as npz in the scratch). Guarded by t01_lib.load_bars."""
    os.makedirs(SCRATCH, exist_ok=True); p = os.path.join(SCRATCH, "bars.npz")
    if os.path.exists(p):
        z = np.load(p); return pd.DataFrame(z["v"], index=pd.DatetimeIndex(z["t"].view("datetime64[ns]")), columns=["open", "high", "low", "close", "volume"])
    s = T.load_bars(T.TRAIN_CSV, LOAD_FROM, LOAD_HI); assert s.index.max() <= LOAD_HI
    np.savez(p, v=s.values.astype(np.float64), t=s.index.values.astype("datetime64[ns]").astype(np.int64)); return s


# ------------------------------------------------------------------ decision anchors
def decision_anchors(index, first, n_after=max(HZ), step=STEP):
    """Bar indices a (a >= first, a + n_after < n) whose bar OPENS on the clock (minute % 15 == 0) and whose previous bar is the previous minute
    (so the features' last bar is the one just closed and there is no data gap right before the decision)."""
    n = len(index); a = np.arange(first, n - n_after - 1)
    ok = (index.minute.values[a] % step == 0) & ((index[a] - index[a - 1]) == pd.Timedelta(minutes=1))
    return a[ok]


# ------------------------------------------------------------------ features at arbitrary anchors
def decision_features(s, anch):
    """XB [n, 52] float64 = [S3 | ln bank], G [n, 328] float32 = [G1 | G15], ok [n] (feature-side validity: the previous five hours have RV > 0,
    ln RV15..1440 > -20, S3 finite), cs2 for TRAIL. Same definitions as t01_lib.build + o3_lib.build_features + o1_lib.window_features, at `anch`."""
    A = s.values.astype(np.float64); a = np.asarray(anch)
    lr = np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]); cs = np.concatenate([[0.0], np.cumsum(lr ** 2)])      # = V.Data.cs2
    c, h, l, v = A[:, 3], A[:, 1], A[:, 2], A[:, 4]
    rv = lambda k: np.log(np.maximum(cs[a] - cs[a - k], 0) / k + 1e-12)
    hr = s.index[a - 1].hour.values
    har = np.column_stack([rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
    cv = np.concatenate([[0.0], np.cumsum(v)]); rng = (h - l) / c; cr = np.concatenate([[0.0], np.cumsum(rng)])
    lvm = lambda k: np.log(np.maximum(cv[a] - cv[a - k], 0) / k + 1e-8)
    lrg = lambda k: np.log(np.maximum(cr[a] - cr[a - k], 0) / k + 1e-8)
    extra = np.column_stack([lvm(15) - lvm(1440), lvm(60) - lvm(1440), lvm(240) - lvm(1440), lrg(15), lrg(60), rv(5)])
    rvw = lambda lo, hi: cs[a + hi] - cs[a + lo]
    prev = [rvw(-60 * (k + 1), -60 * k) for k in range(4)]
    ln = lambda x: np.log(x + 1e-12)
    L = np.column_stack([ln(p) for p in prev]); vp = np.log(L.std(1, ddof=1) + 0.01)
    S3 = np.column_stack([har, extra, L[:, 1:4], vp]).astype(np.float64)
    prev_ok = np.all([(cs[a - 60 * k] - cs[a - 60 * (k + 1)]) > 0 for k in range(1, 5)] + [(cs[a] - cs[a - 60]) > 0], 0)
    ok = prev_ok & np.all(har[:, :4] > -20, 1) & np.isfinite(S3).all(1)
    XB = np.concatenate([S3, O3.log_bank(O3.bank_at(A, a - 1))], 1)
    g1, g15 = O1.window_features(A, a)
    G = np.concatenate([g1, g15], 1).astype(np.float32)
    trail = {H: cs[a] - cs[a - H] for H in HZ}
    return {"XB": XB, "G": G, "ok": ok, "trail": trail}


def build_dframe(s, first=FIRST):
    """The decision frame of a bar series: bar index `a`, time `ta` (= open of bar a), open price m, features and TRAIL RV."""
    a = decision_anchors(s.index, first)
    F = decision_features(s, a)
    F["a"] = a; F["ta"] = s.index.values[a].astype("datetime64[ns]"); F["m"] = s["open"].values[a]
    return F


def load_dframe():
    os.makedirs(SCRATCH, exist_ok=True); p = os.path.join(SCRATCH, "dframe.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    s = load_bars_cut(); F = build_dframe(s)
    flat = {"XB": F["XB"], "G": F["G"], "ok": F["ok"], "a": F["a"], "ta": F["ta"], "m": F["m"], **{f"trail{H}": F["trail"][H] for H in HZ}}
    np.savez(p, **flat); return flat


def load_train_frame():
    """O6's stride-7 frame cut to what L1 and R0 need (XB, G1+G15, ta, rv/ok per horizon)."""
    z = np.load(FRAME); F = {k: z[k] for k in ("ta", "XB", *[f"rv{H}" for H in HZ], *[f"ok{H}" for H in HZ])}
    F["G"] = z["G"][:, :N_L1G]; return F


# ------------------------------------------------------------------ sigma
def month_rows(ta, m0, m1):
    return np.where((ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")))[0]


def sigma_arms(TF, DF, months, H, models=("S3", "L1"), frozen_m0=None, shuf_seed=0):
    """sigma_H at the decision rows of `months` for the arms L1, R0, TRAIL, SHUF. TF: stride-7 training frame; DF: decision frame.
    Monthly walk-forward (O5's fit_h at the month start) or, with frozen_m0, one fit at that time for all rows of `months`.
    Returns {arm: array over ALL decision rows (nan outside `months` or when the features are invalid)} and the fit states."""
    n = len(DF["ta"]); out = {k: np.full(n, np.nan) for k in ARMS}; states = {}
    groups = [(None, months)] if frozen_m0 is not None else [(mo, [mo]) for mo in months]
    for key, ms in groups:
        if frozen_m0 is not None:
            m0 = frozen_m0; rows = np.concatenate([month_rows(DF["ta"], *P5.month_bounds(mo)) for mo in ms])
        else:
            m0, m1 = P5.month_bounds(key); rows = month_rows(DF["ta"], m0, m1)
        rows = rows[DF["ok"][rows]]
        if len(rows) == 0:
            continue
        st = P5.fit_h(TF, H, m0, models=models); assert pd.Timestamp(st["last_label_end"]) <= m0
        states[key or "frozen"] = st
        pr = P5.predict(st, DF, rows, models=models)
        out["L1"][rows] = np.sqrt(np.exp(pr["L1"])); out["R0"][rows] = np.sqrt(np.exp(pr["S3"]))
        out["TRAIL"][rows] = np.sqrt(DF[f"trail{H}"][rows])
        if frozen_m0 is None:
            out["SHUF"][rows] = shuffle_within(out["L1"][rows], shuf_seed, H, key)
    if frozen_m0 is not None:                                   # SHUF within each calendar month of the span
        mo_of = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values
        for mo in months:
            r = np.where((mo_of == mo) & np.isfinite(out["L1"]))[0]
            out["SHUF"][r] = shuffle_within(out["L1"][r], shuf_seed, H, mo)
    return out, states


def shuffle_within(x, seed, H, key):
    """A random permutation of x (the multiset is kept); seeded by (seed, H, month)."""
    k = int(pd.Period(key, freq="M").ordinal) if key else 0
    return np.random.default_rng([seed, H, k]).permutation(x)


# ------------------------------------------------------------------ engine
def simulate(O, Hi, Lo, Cl, dec, m, sig, family, k, Hh, stop_s, valid=STEP, tmin=None):
    """Run one configuration over decisions `dec` (bar indices, sorted), decision prices `m`, forecast stds `sig` (nan = no decision).
    Returns a dict of per-trade arrays (gross log return, no cost) sorted by entry. See SPEC "Mechanics":
      REV: buy limit L = m (1 - k sig), sell limit U = m (1 + k sig); limits fill only on a bar that trades THROUGH the level by TT; price = limit,
           or the open if the bar opens beyond it. Target: limit at m (same rule).
      BRK: buy stop at U, sell stop at L; fill at max(U, open) / min(L, open). No target.
      Stop at stop_s * sig beyond the entry price (None = no stop); fills at the stop or at the open if the bar opens beyond it.
      Exits are checked from the bar AFTER the entry bar; a stop and a target in the same bar resolve to the stop; time stop at the open of bar
      entry + Hh; one position at a time; orders valid for `valid` bars from the decision bar. If both entries fill in one bar the one nearer the
      bar's open is taken (flag both_same_bar counts them)."""
    n = len(O); dec = np.asarray(dec); m = np.asarray(m, float); sig = np.asarray(sig, float)
    good = np.isfinite(sig) & (sig > 0) & (dec + valid < n) & (dec + valid + Hh < n)
    if tmin is not None:
        ok_g = np.zeros(len(dec), bool); gi = np.where(good)[0]
        ok_g[gi] = (tmin[dec[gi] + valid + Hh] - tmin[dec[gi]]) == valid + Hh; good &= ok_g
    idx = np.where(good)[0]
    d = dec[idx]; mm = m[idx]; sg = sig[idx]
    W = d[:, None] + np.arange(valid)[None, :]
    Ow, Hw, Lw = O[W], Hi[W], Lo[W]
    Lv = mm * (1 - k * sg); Uv = mm * (1 + k * sg)
    if family == "REV":
        hb = Lw <= (Lv * (1 - TT))[:, None]; hs = Hw >= (Uv * (1 + TT))[:, None]
    else:                                                     # BRK: buy stop above (U), sell stop below (L)
        hb = Hw >= Uv[:, None]; hs = Lw <= Lv[:, None]
    big = valid + 1
    t_long = np.where(hb.any(1), hb.argmax(1), big); t_short = np.where(hs.any(1), hs.argmax(1), big)   # first bar of a long / short entry
    if family == "REV":                                       # long = buy limit at L, short = sell limit at U
        px_long = lambda o: np.minimum(Lv, o); px_short = lambda o: np.maximum(Uv, o); lv_long, lv_short = Lv, Uv
    else:                                                     # long = buy stop at U, short = sell stop at L
        px_long = lambda o: np.maximum(Uv, o); px_short = lambda o: np.minimum(Lv, o); lv_long, lv_short = Uv, Lv
    has = (t_long < big) | (t_short < big)
    both = (t_long == t_short) & (t_long < big)
    ar = np.arange(len(d))
    tl = np.minimum(t_long, valid - 1); ts_ = np.minimum(t_short, valid - 1)
    o_l = Ow[ar, tl]; o_s = Ow[ar, ts_]
    take_long = np.where(both, np.abs(o_l - lv_long) <= np.abs(o_s - lv_short), t_long < t_short)
    ebar_rel = np.where(take_long, t_long, t_short)
    o_e = Ow[ar, np.minimum(ebar_rel, valid - 1)]
    epx = np.where(take_long, px_long(o_e), px_short(o_e))
    res = {"dec": [], "ebar": [], "xbar": [], "side": [], "epx": [], "xpx": [], "reason": [], "both": [], "sig": []}
    free_from = -1                                            # first bar index at which a decision may be taken (flat again)
    for j in np.where(has)[0]:
        dj = int(d[j])
        if dj < free_from:
            continue
        e = dj + int(ebar_rel[j]); side = 1 if take_long[j] else -1; p0 = float(epx[j]); s0 = float(sg[j]); mj = float(mm[j])
        stop_px = None if stop_s is None else (p0 * (1 - stop_s * s0) if side == 1 else p0 * (1 + stop_s * s0))
        tgt = mj if family == "REV" else None
        xbar, xpx, why = _exit(O, Hi, Lo, e, Hh, side, stop_px, tgt)
        res["dec"].append(dj); res["ebar"].append(e); res["xbar"].append(xbar); res["side"].append(side); res["epx"].append(p0)
        res["xpx"].append(xpx); res["reason"].append(why); res["both"].append(bool(both[j])); res["sig"].append(s0)
        free_from = xbar if why == "time" else xbar + 1      # a time stop is the OPEN of its bar: a decision on that bar is allowed
    out = {k_: np.array(v) for k_, v in res.items()}
    out["ret"] = np.log(out["xpx"] / out["epx"]) * out["side"] if len(out["dec"]) else np.array([])
    return out


def _exit(O, Hi, Lo, e, Hh, side, stop_px, tgt):
    """First exit of a position entered in bar e, looking at bars e+1 .. e+Hh-1 and then the open of bar e+Hh. Returns (bar, price, reason)."""
    j0, j1 = e + 1, e + Hh                                     # bars j0 .. j1-1 are checked; the time stop is the open of bar j1
    ho, hh, ll = O[j0:j1], Hi[j0:j1], Lo[j0:j1]
    big = j1 - j0 + 1
    if side == 1:
        si = int(np.argmax(ll <= stop_px)) if stop_px is not None and (ll <= stop_px).any() else big
        ti = int(np.argmax(hh >= tgt * (1 + TT))) if tgt is not None and (hh >= tgt * (1 + TT)).any() else big
    else:
        si = int(np.argmax(hh >= stop_px)) if stop_px is not None and (hh >= stop_px).any() else big
        ti = int(np.argmax(ll <= tgt * (1 - TT))) if tgt is not None and (ll <= tgt * (1 - TT)).any() else big
    if si == big and ti == big:
        return j1, float(O[j1]), "time"
    if si <= ti:                                               # a stop and a target in one bar: the stop first
        o = ho[si]; px = min(stop_px, o) if side == 1 else max(stop_px, o)
        return j0 + si, float(px), "stop"
    o = ho[ti]; px = max(tgt, o) if side == 1 else min(tgt, o)
    return j0 + ti, float(px), "target"


# ------------------------------------------------------------------ metrics
def monthly_returns(tr, ta_entry_months, months, cost_bps):
    """Sum of net position returns by entry month (log return minus cost); 0 for a month with no trade."""
    net = tr["ret"] - cost_bps * 1e-4; s = pd.Series(net, index=pd.Index(ta_entry_months)) if len(net) else pd.Series(dtype=float)
    g = s.groupby(level=0).sum() if len(s) else s
    return np.array([float(g.get(mo, 0.0)) for mo in months])


def max_drawdown(net):
    """Largest peak-to-trough fall of the cumulative sum of position returns (>= 0; the start counts as a peak at 0)."""
    if len(net) == 0:
        return 0.0
    c = np.concatenate([[0.0], np.cumsum(net)]); return float((np.maximum.accumulate(c) - c).max())


def t_stat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def summarise(tr, entry_months, months):
    """The selection numbers of one configuration: trades, monthly mean net return at each cost, t-stat at 2 bps, hit, max drawdown, exit reasons."""
    out = {"n_trades": int(len(tr["ret"]))}
    for c in COSTS_BPS:
        mr = monthly_returns(tr, entry_months, months, c); out[f"mean_m_{int(c)}bps"] = float(mr.mean())
        if c == 2.0:
            out["t_2bps"] = t_stat(mr); out["months_pos_2bps"] = int((mr > 0).sum()); out["n_months"] = len(mr)
    net2 = tr["ret"] - 2e-4 if len(tr["ret"]) else np.array([])
    out["hit_2bps"] = float((net2 > 0).mean()) if len(net2) else None
    out["max_dd_2bps"] = max_drawdown(net2)
    out["reasons"] = {r: int((tr["reason"] == r).sum()) for r in ("target", "stop", "time")} if len(net2) else {}
    out["n_both_same_bar"] = int(tr["both"].sum()) if len(net2) else 0
    out["long_share"] = float((tr["side"] == 1).mean()) if len(net2) else None
    return out


def config_grid():
    return [{"family": f, "k": k, "H": H, "stop": s} for f in ("REV", "BRK") for k in (1.0, 1.5, 2.0) for H in HZ for s in (1.0, None)]


def cfg_name(c):
    return f"{c['family']}_k{c['k']}_H{c['H']}_s{'none' if c['stop'] is None else c['stop']}"
