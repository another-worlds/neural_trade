# T13 COPY of runs/tactical/outlook/O5_geom_bestbase/o5_lib.py from origin/nt-tactical-outlook at ed870691ab56c2873660b8070d1e4838d4aaea5b; ONLY the sys.path/TAC lines changed (marked 'T13 copy').
"""O5 (SPEC.md): does indicator geometry add over the BEST volatility base (BANK ridge or boosted BANK)? Reuses o1_lib (G1, G15, HGB settings)
and o3_lib (bank, rows, ridge, QLIKE) read-only. CPU only. Anchor a sees bars < a only.

Models (columns over one frame: XB = [S3 (16) | ln bank (36)], G = [G1 | G15] (328)):
  S3 ridge on S3; L0 ridge on S3+bank; L1 = L0 + G; N0 HGB on S3+bank; N1 = N0 + G.
"""
import base64, hashlib, json, os, pickle, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = HERE  # T13 copy: the O1/O3 copies sit in this folder
sys.path.insert(0, OUTLOOK)
import o1_lib as O1  # noqa: E402
import o3_lib as O3  # noqa: E402
T = O3.T
HORIZONS = O3.HORIZONS

# ------------------------------------------------------------------ spans
LOAD_FROM = pd.Timestamp("2017-08-17 00:00")                 # as O4: the first bars used
LOAD_HI = pd.Timestamp("2024-07-01 03:59")                   # the 4-hour label tail of the last selection anchors; far before the cut
J_MONTHS = [str(m) for m in pd.period_range("2018-07", "2020-12", freq="M")]
S_MONTHS = list(O1.M_SEL)                                     # 2021-01 .. 2024-06
FIRST5 = 6145                                                 # >= O3.FIRST; 6145 % 7 == 6 puts the anchors on O3's grid (series start 2017-08-17)
FREEZE_FIRST = O3.FIRST
B_LO = pd.Timestamp("2025-10-01 00:00"); B_HI = pd.Timestamp("2026-09-30 23:59"); B_M0 = pd.Timestamp("2026-10-01 00:00")
N_S3 = len(O3.S3_NAMES); N_B = N_S3 + 36; N_G = 2 * len(O1.G.geom_names(None, O1.CROSS))
SPECS = {"S3": ("ridge", N_S3, False), "L0": ("ridge", N_B, False), "L1": ("ridge", N_B, True),
         "N0": ("hgb", N_B, False), "N1": ("hgb", N_B, True)}
SCRATCH = "D:/nt/nt_scratch/o5"
HGB_KW = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, early_stopping=False, random_state=0)   # O1's settings


def month_bounds(m):
    p = pd.Period(m, freq="M"); return p.start_time, (p + 1).start_time


# ------------------------------------------------------------------ frame
def build_frame(s, first, with_g=True):
    """Features of one bar frame: ta, XB [n, 52] float64, G [n, 328] float32 (or None), rv/ok per horizon."""
    F3 = O3.build_features(s, first)
    out = {"ta": F3["ta"], "XB": np.concatenate([F3["S3"], O3.log_bank(F3["bank"])], 1)}
    if with_g:
        g1, g15 = O1.window_features(F3["A"], F3["anch"])
        out["G"] = np.concatenate([g1, g15], 1).astype(np.float32)
    else:
        out["G"] = None
    for H in HORIZONS:
        out[f"rv{H}"] = F3[f"rv{H}"]; out[f"ok{H}"] = F3[f"ok{H}"]
    return out


def getX(F, rows, name):
    kind, nc, g = SPECS[name]; X = F["XB"][rows][:, :nc]
    if g:
        X = np.concatenate([X, F["G"][rows].astype(np.float64)], 1)
    return X


# ------------------------------------------------------------------ fit / predict
def fit_h(F, H, m0, models=tuple(SPECS), hgb_kw=None):
    """Every requested model for a test starting m0. Ridge: refit on all 365-day rows, QLIKE scale c from the first-85% fit on the hold rows (gap ceil(H/7)).
    HGB: one fit on the first 85% of the rows strided by 10 (O1), c from the hold rows."""
    tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, m0); fit, hold = O3.split_fit_hold(tr, H)
    y = np.log(F[f"rv{H}"][tr]); v = F[f"rv{H}"][tr]
    pos = {r: i for i, r in enumerate(tr)}; fi = np.array([pos[r] for r in fit]); hi = np.array([pos[r] for r in hold])
    st = {"H": H, "n_tr": int(len(tr)), "m": {}, "c": {},
          "first_ta": str(pd.Timestamp(F["ta"][tr].min())) if len(tr) else None,
          "last_label_end": str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=H)) if len(tr) else None}
    for nm in models:
        kind = SPECS[nm][0]; X = getX(F, tr, nm)
        if kind == "ridge":
            mf = O3.ridge_fit(X[fi], y[fi]); st["m"][nm] = O3.ridge_fit(X, y)
            st["c"][nm] = O3.qlike_c(v[hi], O3.ridge_pred(mf, X[hi]))
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            sub = fi[::10]; mdl = HistGradientBoostingRegressor(**(hgb_kw or HGB_KW)).fit(X[sub], y[sub])
            st["m"][nm] = mdl; st["c"][nm] = O3.qlike_c(v[hi], mdl.predict(X[hi]))
    return st


def predict(st, F, te, models=None):
    out = {}
    for nm in (models or st["m"]):
        X = getX(F, te, nm); m = st["m"][nm]
        out[nm] = O3.ridge_pred(m, X) if SPECS[nm][0] == "ridge" else m.predict(X)
    return out


def score(st, F, te, span, month, models=None):
    H = st["H"]; v = F[f"rv{H}"][te]; rows = []
    for nm, mu in predict(st, F, te, models).items():
        rows.append({"span": span, "month": month, "H": H, "model": nm, "n_te": int(len(te)), "n_tr": st["n_tr"],
                     "spearman": float(spearmanr(mu, v)[0]) if np.std(mu) > 0 else None, "qlike": O3.qlike(v, mu, st["c"][nm])})
    return rows


# ------------------------------------------------------------------ statistics
def paired(df, a, b, col, months, H, min_n=3):
    """a - b per month (rows of horizon H): mean, 95% t interval, two-sided t p, months positive."""
    d0 = df[df.H == H]
    pa = d0[d0.model == a].set_index("month")[col].astype(float); pb = d0[d0.model == b].set_index("month")[col].astype(float)
    d = (pa - pb).reindex(months).dropna()
    if len(d) < min_n:
        return None
    mu, lo, hi, t = T.ci(d.values)
    return {"mean": mu, "lo": lo, "hi": hi, "p": float(2 * stats.t.sf(abs(t), len(d) - 1)), "pos": int((d > 0).sum()), "n": int(len(d)),
            "min": float(d.min()), "max": float(d.max())}


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run_
    return adj


def criterion(sp, ql, need=20):
    """O5 SPEC, H = 60, candidate - B*: mean Spearman gain >= +0.003, 95% t lower bound > 0, positive months >= need, mean QLIKE diff <= 0."""
    c1 = sp["mean"] >= 0.003; c2 = sp["lo"] > 0; c3 = sp["pos"] >= need; c4 = ql["mean"] <= 0
    return {"c1_mean_ge_0.003": bool(c1), "c2_lo_gt_0": bool(c2), "c3_months": bool(c3), "need_positive": int(need), "c4_qlike_le_0": bool(c4),
            "pass": bool(c1 and c2 and c3 and c4)}


# ------------------------------------------------------------------ the frozen artefact
def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_to_json(nm, m):
    if SPECS[nm][0] == "ridge":
        return {"kind": "ridge", "mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "w": m["w"].tolist(), "ym": float(m["ym"])}
    return {"kind": "hgb", "pickle_b64": base64.b64encode(pickle.dumps(m, protocol=4)).decode()}


def model_from_json(j):
    if j["kind"] == "ridge":
        return {"mu": np.array(j["mu"]), "sd": np.array(j["sd"]), "w": np.array(j["w"]), "ym": float(j["ym"])}
    return pickle.loads(base64.b64decode(j["pickle_b64"]))


def state_to_json(st):
    return {"H": st["H"], "n_tr": st["n_tr"], "first_ta": st["first_ta"], "last_label_end": st["last_label_end"],
            "c": {k: float(v) for k, v in st["c"].items()}, "m": {k: model_to_json(k, v) for k, v in st["m"].items()}}


def state_from_json(j):
    return {"H": j["H"], "n_tr": j["n_tr"], "first_ta": j["first_ta"], "last_label_end": j["last_label_end"],
            "c": dict(j["c"]), "m": {k: model_from_json(v) for k, v in j["m"].items()}}


def freeze(bars, base, hgb_kw=None):
    """Fit B* (`base` in {L0, N0}) and the candidate (L1 / N1) ONCE per horizon on every anchor of the Binance bars with a label end <= 2026-10-01."""
    assert bars.index.max() <= B_HI
    F = build_frame(bars, FREEZE_FIRST); cand = {"L0": "L1", "N0": "N1"}[base]
    art = {"spec": "O5 SPEC.md freeze", "base": base, "candidate": cand, "last_bar": str(bars.index.max()), "n_bars": int(len(bars)),
           "file_first_bar": str(bars.index.min()), "stride": T.STRIDE, "warm_bars": FREEZE_FIRST, "ridge_alpha": 1.0, "horizons": list(HORIZONS),
           "taus": list(O3.TAUS), "eps_ch": O3.EPS_CH.tolist(), "s3_names": O3.S3_NAMES, "n_g": N_G, "hgb_kw": hgb_kw or HGB_KW,
           "features": "XB = S3 (16) + ln(EWMA_tau(channel)+eps_channel) (36, channel-major); G = o1_lib G1 + G15 (328)",
           "lib_note": "o1_lib.py 6cb5617, o3_lib.py 5e4ab42", "states": {}}
    for H in HORIZONS:
        art["states"][str(H)] = state_to_json(fit_h(F, H, B_M0, models=(base, cand), hgb_kw=hgb_kw))
    art["sha256"] = sha_of(art); return art


def load_freeze_bars(path=None):
    """Binance bars B_LO..B_HI only (reads exactly the rows up to B_HI when the file has no gaps)."""
    path = path or T.FWD_CSV; n = int((B_HI - B_LO).total_seconds() // 60) + 1
    df = pd.read_csv(path, usecols=T.COLS, nrows=n); ts = pd.to_datetime(df["timestamp"]); df.index = ts; df = df.drop(columns="timestamp")
    assert ts.iloc[0] == B_LO
    if ts.iloc[-1] < B_HI or len(df) < n:
        return T.load_bars(path, B_LO, B_HI)
    df = df[(df.index >= B_LO) & (df.index <= B_HI)]
    assert df.index.min() >= B_LO and df.index.max() <= B_HI
    assert not ((df.index >= T.FORBID_LO) & (df.index <= T.FORBID_HI)).any()
    return df.sort_index()
