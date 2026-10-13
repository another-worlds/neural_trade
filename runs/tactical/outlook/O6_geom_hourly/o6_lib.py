"""O6 (SPEC.md): does geometry of HOURLY bars (G60) add to the best volatility base at H = 240? Reuses o5_lib (frame definition, ridge, QLIKE, statistics,
walk-forward) and o1_lib / o3_lib unchanged. CPU only. Anchor a sees bars < a only.

Columns over one frame: XB = [S3 (16) | ln bank (36)] (52), G = [G1 (164) | G15 (164) | G60 (164)] (492).
  L1 = XB + G1 + G15; L2 = L1 + G60; L2sh = L2 with the G60 rows permuted (train and test independently).
"""
import json, os, sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUTLOOK, "O5_geom_bestbase"))
import o5_lib as P5  # noqa: E402
sys.path.remove(os.path.join(OUTLOOK, "O5_geom_bestbase"))   # O5 has its own score_B / run_o5: keep module names of this folder
sys.path.insert(0, HERE)
O1 = P5.O1; O3 = P5.O3; T = P5.T; G = O1.G
HORIZONS = P5.HORIZONS
SCRATCH = "D:/nt/nt_scratch/o6"
LOAD_FROM = P5.LOAD_FROM; LOAD_HI = T.CUTOFF                 # the whole cut file: spent spans run to 2025-07-20
FIRST6 = P5.FIRST5                                            # 6145 >= 3600 + 1 and puts the anchors on O3's / O5's grid
J_MONTHS = [str(m) for m in pd.period_range("2019-01", "2020-12", freq="M")]
S_SPENT1 = [str(m) for m in pd.period_range("2021-01", "2024-06", freq="M")]
S_SPENT2 = [str(m) for m in pd.period_range("2024-07", "2025-07", freq="M")]     # 2025-07 ends at the cut (07-20)
N_XB = P5.N_B; N_GS = len(G.geom_names(None, O1.CROSS)); N_G1G15 = 2 * N_GS
HOUR = 60; NB = 60
MODELS = ("L1", "L2", "L2sh")
B_LO, B_HI, B_M0 = P5.B_LO, P5.B_HI, P5.B_M0
FREEZE_FIRST = P5.FREEZE_FIRST
G60_NAMES = ["g60." + n for n in G.geom_names(None, O1.CROSS)]


# ------------------------------------------------------------------ windows
def aggm(A, anchors, m, nb=NB):
    """[N, nb, 5] bars of m minutes from the m*nb one-minute bars ending at a-1 (open of the first, max high, min low, close of the last, summed
    volume). The generalisation of o1_lib.agg15 (m = 15, nb = 60 reproduces it exactly)."""
    L = m * nb; a = np.asarray(anchors); s0 = int(a.min()) - L; seg = A[s0:int(a.max())]
    rmax = swv(seg[:, 1], m).max(1); rmin = swv(seg[:, 2], m).min(1); cv = np.concatenate([[0.0], np.cumsum(seg[:, 4])])
    st = (a - s0)[:, None] - L + m * np.arange(nb)[None, :]
    return np.stack([seg[st, 0], rmax[st], rmin[st], seg[st + m - 1, 3], cv[st + m] - cv[st]], -1)


def window_features6(A, anchors, chunk=4000):
    """(G1, G15, G60) for the anchors: geom_rich on the last 60 bars of 1, 15 and 60 minutes."""
    a = np.asarray(anchors); out = []
    for i in range(0, len(a), chunk):
        b = a[i:i + chunk]
        out.append(np.concatenate([G.geom_rich(np.stack([A[x - 60:x] for x in b])), G.geom_rich(aggm(A, b, 15)), G.geom_rich(aggm(A, b, HOUR))], 1))
    return np.concatenate(out).astype(np.float32)


def build_frame6(s, first):
    """ta, anch, XB [n, 52] float64, G [n, 492] float32, rv / ok per horizon. XB and G1/G15 are exactly o5_lib.build_frame's."""
    F3 = O3.build_features(s, first); assert first >= HOUR * NB + 1
    out = {"ta": F3["ta"], "anch": F3["anch"], "XB": np.concatenate([F3["S3"], O3.log_bank(F3["bank"])], 1),
           "G": window_features6(F3["A"], F3["anch"])}
    for H in HORIZONS:
        out[f"rv{H}"] = F3[f"rv{H}"]; out[f"ok{H}"] = F3[f"ok{H}"]
    return out


def getX(F, rows, name, g60_perm=None):
    """Design matrix of a model for `rows`. g60_perm: an index permutation of range(len(rows)) applied to the G60 block (L2sh only)."""
    X = F["XB"][rows]; g = F["G"][rows]
    if name == "L1":
        return np.concatenate([X, g[:, :N_G1G15].astype(np.float64)], 1)
    g60 = g[:, N_G1G15:]
    if name == "L2sh":
        g60 = g60[g60_perm]
    return np.concatenate([X, g[:, :N_G1G15].astype(np.float64), g60.astype(np.float64)], 1)


def perm_for(n, H, key):
    return np.random.default_rng(10_000 * H + key).permutation(n)


# ------------------------------------------------------------------ fit / predict / score
def fit_h(F, H, m0, models=MODELS, key=0):
    """Ridge alpha 1, refit on every 365-day row; QLIKE scale c from the first-85% fit on the hold rows (gap ceil(H/7)); as o5_lib.fit_h."""
    tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, m0); fit, hold = O3.split_fit_hold(tr, H)
    k = len(fit); fi = np.arange(k); hi = np.arange(len(tr) - len(hold), len(tr))
    assert np.array_equal(tr[fi], fit) and np.array_equal(tr[hi], hold)
    y = np.log(F[f"rv{H}"][tr]); v = F[f"rv{H}"][tr]
    st = {"H": H, "n_tr": int(len(tr)), "m": {}, "c": {},
          "first_ta": str(pd.Timestamp(F["ta"][tr].min())) if len(tr) else None,
          "last_label_end": str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=H)) if len(tr) else None}
    pt = perm_for(len(tr), H, key)
    for nm in models:
        X = getX(F, tr, nm, pt); mf = O3.ridge_fit(X[fi], y[fi]); st["m"][nm] = O3.ridge_fit(X, y)
        st["c"][nm] = O3.qlike_c(v[hi], O3.ridge_pred(mf, X[hi])); del X
    return st


def predict(st, F, te, models=None, key=0):
    out = {}; pe = perm_for(len(te), st["H"], key + 1)
    for nm in (models or st["m"]):
        out[nm] = O3.ridge_pred(st["m"][nm], getX(F, te, nm, pe))
    return out


def score(st, F, te, span, month, models=None, key=0):
    from scipy.stats import spearmanr
    H = st["H"]; v = F[f"rv{H}"][te]; rows = []
    for nm, mu in predict(st, F, te, models, key).items():
        rows.append({"span": span, "month": month, "H": H, "model": nm, "n_te": int(len(te)), "n_tr": st["n_tr"],
                     "spearman": float(spearmanr(mu, v)[0]) if np.std(mu) > 0 else None, "qlike": O3.qlike(v, mu, st["c"][nm])})
    return rows


# ------------------------------------------------------------------ frozen artefact (ridge only)
def model_to_json(m):
    return {"kind": "ridge", "mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "w": m["w"].tolist(), "ym": float(m["ym"])}


def model_from_json(j):
    return {"mu": np.array(j["mu"]), "sd": np.array(j["sd"]), "w": np.array(j["w"]), "ym": float(j["ym"])}


def state_to_json(st):
    return {"H": st["H"], "n_tr": st["n_tr"], "first_ta": st["first_ta"], "last_label_end": st["last_label_end"],
            "c": {k: float(v) for k, v in st["c"].items()}, "m": {k: model_to_json(v) for k, v in st["m"].items()}}


def state_from_json(j):
    return {"H": j["H"], "n_tr": j["n_tr"], "first_ta": j["first_ta"], "last_label_end": j["last_label_end"],
            "c": dict(j["c"]), "m": {k: model_from_json(v) for k, v in j["m"].items()}}


def freeze(bars):
    """Fit L1 and L2 ONCE per horizon on every anchor of the Binance bars with a label end <= 2026-10-01."""
    assert bars.index.max() <= B_HI
    F = build_frame6(bars, FREEZE_FIRST)
    art = {"spec": "O6 SPEC.md freeze", "base": "L1", "candidate": "L2", "last_bar": str(bars.index.max()), "n_bars": int(len(bars)),
           "file_first_bar": str(bars.index.min()), "stride": T.STRIDE, "warm_bars": FREEZE_FIRST, "ridge_alpha": 1.0, "horizons": list(HORIZONS),
           "taus": list(O3.TAUS), "eps_ch": O3.EPS_CH.tolist(), "s3_names": O3.S3_NAMES, "n_g_per_scale": N_GS, "g60_names": G60_NAMES,
           "features": "XB = S3 (16) + ln(EWMA_tau(channel)+eps_channel) (36); G = geom_rich of the last 60 bars of 1, 15 and 60 minutes (3 x 164); "
                       "L1 = XB + G1 + G15, L2 = L1 + G60",
           "lib_note": "o1_lib.py 6cb5617, o3_lib.py 5e4ab42, o5_lib.py 0ab6466, o6_lib.py", "states": {}}
    for H in HORIZONS:
        art["states"][str(H)] = state_to_json(fit_h(F, H, B_M0, models=("L1", "L2")))
    art["sha256"] = P5.sha_of(art); return art


paired = P5.paired; holm = P5.holm; criterion = P5.criterion; sha_of = P5.sha_of; month_bounds = P5.month_bounds
