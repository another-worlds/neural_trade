"""O9 phase 2 library (SPEC.md + addendum): arms, trade frames, the frozen artefact. Builds on o9_lib (phase 1, unchanged). CPU only."""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o9_lib as L  # noqa: E402
L8 = L.L8; P5 = L.P5; O3 = L.O3; T = L.T
H = L.H; N_S3 = L.N_S3
SEED = 0
ARMS9 = ("FULL", "NO_REG", "NO_DV", "NO_GEOM", "SHUF")
FIRST_SCORED = pd.Period("2026-11", freq="M")                  # SPEC: the first 3 complete calendar months AFTER 2026-10
BASE_CFG = {"k": 1.5, "delta": None, "g": None, "tau": None, "size": "1x"}       # O8's BRK_k1.5_H60_s1.0


def make_X(TF, ex):
    """The training dict of dv_fit / pup_fit from a frame TF (ta, XB, ok60) and train_extras output."""
    X = dict(ex); X["ta"] = TF["ta"]; X["S3"] = np.ascontiguousarray(TF["XB"][:, :N_S3]); X["ok60"] = TF["ok60"]
    X["dok"] = X["ok60"] & np.isfinite(X["dA"])
    X["pok"] = np.isfinite(X["ret60"]) & (np.abs(X["ret60"]) > L.DEAD) & np.isfinite(X["tb7"]).all(1) & TF["ok60"]
    return X


def _clean(out):
    bad = ~np.all([np.isfinite(out[k]) for k in out], 0)
    for k in out:
        out[k][bad] = np.nan
    return out


def signals_arms(TF, X, DF, DX, months, frozen_m0=None):
    """L1 and R0 sigma_60, dV and p_up at the decision rows of `months`. Monthly walk-forward, or (frozen_m0) one fit at that time for all months.
    nan where any of the four is invalid."""
    n = len(DF["ta"]); dv = np.full(n, np.nan); pu = np.full(n, np.nan)
    arms, _ = L8.sigma_arms(TF, DF, months, H, frozen_m0=frozen_m0)
    groups = [(frozen_m0, months)] if frozen_m0 is not None else [(P5.month_bounds(mo)[0], [mo]) for mo in months]
    for m0, ms in groups:
        rows = np.concatenate([L.month_rows(DF["ta"], mo) for mo in ms]); rows = rows[DF["ok"][rows]]
        if len(rows) == 0:
            continue
        sd = L.dv_fit(X, m0); sp = L.pup_fit(X, m0); assert sd["last_label_end"] <= m0 and sp["last_label_end"] <= m0
        dv[rows] = O3.ridge_pred(sd["m"], DF["XB"][rows][:, :N_S3]); pu[rows] = L.pup_predict(sp, DX["tb7"][rows])
    return _clean({"L1": arms["L1"].copy(), "R0": arms["R0"].copy(), "dv": dv, "p": pu})


def shuffle_month(x, ta, months, seed):
    """Permute the finite values of x within each calendar month (the multiset per month is kept)."""
    mo_of = pd.DatetimeIndex(ta).to_period("M").astype(str).values; out = x.copy()
    for mo in months:
        r = np.where((mo_of == mo) & np.isfinite(x))[0]
        if len(r):
            out[r] = L8.shuffle_within(x[r], seed, H, mo)
    return out


def arm_inputs(S, arm, ta, months):
    """(sigma, dv, p) of an arm: NO_GEOM uses R0's sigma; SHUF permutes sigma (L1), dV and p_up independently within each month (seeds 0, 1, 2)."""
    if arm == "NO_GEOM":
        return S["R0"], S["dv"], S["p"]
    if arm == "SHUF":
        return shuffle_month(S["L1"], ta, months, SEED), shuffle_month(S["dv"], ta, months, SEED + 1), shuffle_month(S["p"], ta, months, SEED + 2)
    return S["L1"], S["dv"], S["p"]


def arm_cfg(c, arm):
    """The configuration an arm trades: NO_REG drops the regression gate, NO_DV the volatility gate; the others keep the candidate's."""
    c = dict(c)
    if arm == "NO_REG":
        c["delta"] = None
    if arm == "NO_DV":
        c["g"] = None
    return c


def weights9(size, sig_r, ref_r):
    """w per trade: 1 for '1x'; min(2, ref / sigma) for 'vt'; 1 where the reference month is missing (the first month of a span)."""
    if size != "vt":
        return np.ones(len(sig_r))
    w = L.size_weights(sig_r, ref_r); return np.where(np.isfinite(w), w, 1.0)


TCOLS = ["decision_time", "entry_time", "exit_time", "side", "entry_px", "exit_px", "reason", "ret_gross", "w", "sigma", "p_up", "dV"]


def trades9(s, DF, rows, sigma, dv, p, ref_row, cfg, pess):
    """One configuration on decision rows `rows`: the trade frame (entry/exit time, side, prices, reason, gross return, w, signals at the decision)."""
    O, Hi, Lo, Cl = s["open"].values, s["high"].values, s["low"].values, s["close"].values
    tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    aL, aS = L.gate_masks(p[rows], dv[rows], cfg["delta"], cfg["g"])
    tr = L.simulate9(O, Hi, Lo, Cl, DF["a"][rows], DF["m"][rows], sigma[rows], cfg["k"], H, 1.0, allow_long=aL, allow_short=aS, tau=cfg["tau"], tmin=tmin, pess=pess)
    if len(tr["j"]) == 0:
        return pd.DataFrame({c: [] for c in TCOLS})
    r = rows[tr["j"]]
    return pd.DataFrame({"decision_time": s.index[tr["dec"]], "entry_time": s.index[tr["ebar"]], "exit_time": s.index[tr["xbar"]], "side": tr["side"], "entry_px": tr["epx"], "exit_px": tr["xpx"],
                         "reason": tr["reason"], "ret_gross": tr["ret"], "w": weights9(cfg["size"], sigma[r], ref_row[r]), "sigma": tr["sig"], "p_up": p[r], "dV": dv[r]})


def ref_rows(sigma, ta, months):
    """sigma_ref (previous calendar month's median sigma_60) broadcast to every decision row of the months."""
    ref = L.sigma_ref_by_month(sigma, ta, months); mo_of = pd.DatetimeIndex(ta).to_period("M").astype(str).values
    return np.array([ref.get(m, np.nan) for m in mo_of])


def monthly_tf(tf, months, c):
    em = tf["entry_time"].dt.to_period("M").astype(str).values if len(tf) else np.array([], dtype=str)
    return L.monthly9(tf["ret_gross"].values if len(tf) else np.array([]), tf["w"].values if len(tf) else np.array([]), em, months, c)


def daily_net(tf, days, cost_bps=2.0):
    """Net return per exit day (w x (ret - cost)), zero on days without an exit, over the index `days`."""
    if len(tf) == 0:
        return pd.Series(0.0, index=days)
    net = tf["w"].values * (tf["ret_gross"].values - cost_bps * 1e-4)
    return pd.Series(net, index=pd.DatetimeIndex(tf["exit_time"]).normalize()).groupby(level=0).sum().reindex(days, fill_value=0.0)


# ------------------------------------------------------------------ the frozen artefact
def pup_to_json(st):
    return {"mean": st["sc"].mean_.tolist(), "scale": st["sc"].scale_.tolist(), "coef": st["lr"].coef_[0].tolist(), "icpt": float(st["lr"].intercept_[0]),
            "n_tr": st["n_tr"], "last_label_end": str(st["last_label_end"]), "base": st["base"]}


def pup_predict_j(j, tb7):
    z = (np.asarray(tb7, np.float64) - np.array(j["mean"])) / np.array(j["scale"]); return 1 / (1 + np.exp(-(z @ np.array(j["coef"]) + j["icpt"])))


def freeze_fit(bars, m0, first=O3.FIRST):
    """Fit L1, R0 (S3), the dV ridge and the logistic p_up ONCE on every anchor of `bars` whose label ends at or before m0 (the Binance file's
    2025-10-01 .. 2026-09-30 for the real freeze). Returns the artefact dict (with sha256)."""
    F = P5.build_frame(bars, first); ex = L.train_extras(bars, F["ta"]); X = make_X(F, ex)
    st = P5.fit_h(F, H, m0, models=("S3", "L1")); sd = L.dv_fit(X, m0); sp = L.pup_fit(X, m0)
    assert pd.Timestamp(st["last_label_end"]) <= m0 and sd["last_label_end"] <= m0 and sp["last_label_end"] <= m0
    art = {"spec": "O9 SPEC.md freeze", "m0": str(m0), "last_bar": str(bars.index.max()), "first_bar": str(bars.index.min()), "n_bars": int(len(bars)),
           "stride": T.STRIDE, "warm_bars": int(first), "horizon": H, "dead_band": L.DEAD, "C": L.C_LR, "size_cap": L.SIZE_CAP,
           "sigma": P5.state_to_json(st),
           "dv": {"mu": sd["m"]["mu"].tolist(), "sd": sd["m"]["sd"].tolist(), "w": sd["m"]["w"].tolist(), "ym": float(sd["m"]["ym"]),
                  "n_tr": sd["n_tr"], "last_label_end": str(sd["last_label_end"])},
           "pup": pup_to_json(sp), "lib_note": "o9_lib / o9_p2 with o8_lib, o5_lib, o3_lib, o1_lib, t01_lib as in the repository at the freeze commit"}
    art["sha256"] = P5.sha_of(art); return art


def load_artefact(art):
    """Rebuild the model objects from the artefact; raises if the sha256 does not match."""
    if P5.sha_of(art) != art["sha256"]:
        raise ValueError("the frozen artefact does not match its sha256")
    d = art["dv"]
    return {"sigma": P5.state_from_json(art["sigma"]), "dv": {"mu": np.array(d["mu"]), "sd": np.array(d["sd"]), "w": np.array(d["w"]), "ym": float(d["ym"])}, "pup": art["pup"]}


def signals_frozen(models, DF, DX, months):
    """L1 / R0 sigma_60, dV and p_up with the frozen models at the decision rows of `months` (no refit)."""
    n = len(DF["ta"]); out = {k: np.full(n, np.nan) for k in ("L1", "R0", "dv", "p")}
    for mo in months:
        rows = L.month_rows(DF["ta"], mo); rows = rows[DF["ok"][rows]]
        if len(rows) == 0:
            continue
        pr = P5.predict(models["sigma"], DF, rows, models=("S3", "L1"))
        out["L1"][rows] = np.sqrt(np.exp(pr["L1"])); out["R0"][rows] = np.sqrt(np.exp(pr["S3"]))
        out["dv"][rows] = O3.ridge_pred(models["dv"], DF["XB"][rows][:, :N_S3]); out["p"][rows] = pup_predict_j(models["pup"], DX["tb7"][rows])
    return _clean(out)


def make_dframe_for(s, first):
    """Decision frame (the dict shape of load_dframe) and its tb7 for an arbitrary bar series."""
    D = L8.build_dframe(s, first)
    DF = {"XB": D["XB"], "G": D["G"], "ok": D["ok"], "a": D["a"], "ta": D["ta"], "m": D["m"], **{f"trail{h}": D["trail"][h] for h in L8.HZ}}
    return DF, {"tb7": L.tb7_at(s.values.astype(np.float64), DF["a"])}


# ------------------------------------------------------------------ prospective statistics
def block_means(x, B=10000, block=5, seed=0):
    """Moving-block bootstrap means of a daily series (block of `block` days, B draws)."""
    x = np.asarray(x, float); n = len(x); nb = int(np.ceil(n / block)); rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(n - block + 1, 1), size=(B, nb)); idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return x[idx].mean(1)


def boot_ci(x, B=10000, block=5, seed=0):
    """mean, 95% percentile interval and the two-sided bootstrap p-value of mean = 0 (centred at 0 by the share of draws on each side)."""
    bm = block_means(x, B, block, seed); lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(bm <= 0) + 1) / (B + 1), (np.sum(bm >= 0) + 1) / (B + 1)))
    return {"mean": float(np.mean(x)), "lo": float(lo), "hi": float(hi), "p_two_sided": float(p), "n_days": int(len(x))}


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run_
    return adj
