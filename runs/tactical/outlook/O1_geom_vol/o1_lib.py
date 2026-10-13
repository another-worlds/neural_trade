"""O1 (SPEC.md): does indicator geometry (1-minute and 15-minute scale) add to the S3 volatility forecast? Library: features, the
walk-forward ridge/HGB fit and the monthly scoring. CPU only. Anchor a sees bars < a only."""
import os, sys, math
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "tasks", "T01_vol_forward")); sys.path.insert(0, os.path.join(TAC, "geom2")); sys.path.insert(0, os.path.join(TAC, "volnet"))
import t01_lib as T  # noqa: E402
import geomlib as G  # noqa: E402
import volnet_lib as V  # noqa: E402

HORIZONS = (15, 60, 240)
LOAD_FROM = pd.Timestamp("2019-12-01 00:00")
M_SEL = [str(m) for m in pd.period_range("2021-01", "2024-06", freq="M")]
M_LOOK = [str(m) for m in pd.period_range("2024-07", "2025-07", freq="M")]
FWD_FIRST = 1440 + 900
TRAIN_END = T.CUTOFF + pd.Timedelta(minutes=1)          # the frozen fit: labels end by here
SCRATCH = "D:/nt/nt_scratch/o1"
S3_NAMES = ["rv15", "rv60", "rv240", "rv1440", "tod_sin", "tod_cos", "vol15r", "vol60r", "vol240r", "rng15", "rng60", "rv5",
            "lrv_h2", "lrv_h3", "lrv_h4", "lrv_disp"]
CROSS = (3, 5, 10, 20)
GROUPS = {"price_bands": lambda n: n.startswith(("ema", "sma20", "bb_up", "bb_lo", "kel_", "don_", "vwap20")),
          "oscillators": lambda n: n.startswith(("rsi14", "macd_", "stoch14", "cci20", "mfi14", "dmi_diff")),
          "volatility": lambda n: n.startswith(("atr14", "bb_width", "adx14")),
          "volume": lambda n: n.startswith("obv")}


def group_of(channel):
    for g, f in GROUPS.items():
        if f(channel):
            return g
    return "other"


def feature_names():
    gn = G.geom_names(None, CROSS)
    return S3_NAMES + ["g1." + n for n in gn] + ["g15." + n for n in gn]


# ------------------------------------------------------------------ windows and geometry
def agg15(A, anchors):
    """[N, 60, 5] bars of 15 minutes from the 900 one-minute bars ending at a-1 (open of the first, max high, min low, close of the
    last, summed volume). A columns: open, high, low, close, volume."""
    a = np.asarray(anchors); s0 = int(a.min()) - 900; seg = A[s0:int(a.max())]
    rmax = swv(seg[:, 1], 15).max(1); rmin = swv(seg[:, 2], 15).min(1); cv = np.concatenate([[0.0], np.cumsum(seg[:, 4])])
    st = (a - s0)[:, None] - 900 + 15 * np.arange(60)[None, :]            # start offsets of the 60 bars inside seg
    out = np.stack([seg[st, 0], rmax[st], rmin[st], seg[st + 14, 3], cv[st + 15] - cv[st]], -1)
    return out


def window_features(A, anchors, chunk=4000):
    """(G1, G15) for the anchors: geom_rich on the last 60 one-minute bars and on the last 60 bars of 15 minutes."""
    a = np.asarray(anchors); g1, g15 = [], []
    for i in range(0, len(a), chunk):
        b = a[i:i + chunk]
        w1 = np.stack([A[x - 60:x] for x in b]); g1.append(G.geom_rich(w1))
        g15.append(G.geom_rich(agg15(A, b)))
    return np.concatenate(g1), np.concatenate(g15)


def build_features(s, first):
    """dict of the feature arrays of one bar frame: ta, S3, G1, G15, per-H realised RV, |r|, ok masks."""
    B = T.build(s, first); D = B["D"]; a = D.anch; cs = D.cs2; A = D.A
    S3 = B["F"]["S3"].astype(np.float64)
    prev_ok = np.all([(cs[a - 60 * k] - cs[a - 60 * (k + 1)]) > 0 for k in range(1, 5)] + [(cs[a] - cs[a - 60]) > 0], 0)
    okf = prev_ok & np.all(D.har[:, :4] > -20, 1) & np.isfinite(S3).all(1)
    g1, g15 = window_features(A, a)
    out = {"ta": D.ta.values.astype("datetime64[ns]").astype(np.int64), "S3": S3.astype(np.float32), "G1": g1, "G15": g15, "okf": okf}
    for j, H in enumerate(HORIZONS):
        rv = cs[a + H] - cs[a]
        out[f"rv{H}"] = rv; out[f"absr{H}"] = D.absr[:, j]; out[f"v2_{H}"] = D.v2[:, j]; out[f"ok{H}"] = okf & (rv > 0)
    return out


# ------------------------------------------------------------------ rows
def train_rows(ta, ok, H, m0, days=365):
    """rows available in the 365 days before m0 whose label ends at or before m0."""
    ta = np.asarray(ta).astype("datetime64[ns]"); lo = np.datetime64(m0 - pd.Timedelta(days=days), "ns"); hi = np.datetime64(m0 - pd.Timedelta(minutes=H), "ns")
    return np.where((ta >= lo) & (ta <= hi) & ok)[0]


def test_rows(ta, ok, m0, m1):
    ta = np.asarray(ta).astype("datetime64[ns]")
    return np.where((ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")) & ok)[0]


def split_fit_hold(tr, H, stride=7, frac=0.15):
    n = len(tr); k = int(n * (1 - frac)); gap = int(math.ceil(H / stride))
    return tr[:k], tr[k + gap:]


# ------------------------------------------------------------------ ridge by normal equations (alpha 1, standardised, free intercept)
class Ridge:
    """Standardises on its rows, one gram matrix; solves any column subset for any target (== sklearn Ridge(alpha=1) per subset)."""

    def __init__(self, X, alpha=1.0):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-12; Xs = (X - self.mu) / self.sd
        self.Xs = Xs; self.G = Xs.T @ Xs; self.alpha = alpha

    def fit(self, y, idx):
        ym = float(y.mean()); b = self.Xs[:, idx].T @ (y - ym)
        w = np.linalg.solve(self.G[np.ix_(idx, idx)] + self.alpha * np.eye(len(idx)), b)
        return {"idx": np.asarray(idx), "w": w, "ym": ym}

    def predict(self, m, X):
        i = m["idx"]; return ((X[:, i] - self.mu[i]) / self.sd[i]) @ m["w"] + m["ym"]


def sets_def():
    """column sets over [S3 | G1 | G15]."""
    n3, ng = len(S3_NAMES), len(G.geom_names(None, CROSS)); s3 = np.arange(n3); g1 = np.arange(n3, n3 + ng); g15 = np.arange(n3 + ng, n3 + 2 * ng)
    sets = {"R0": s3, "R1": np.r_[s3, g1], "R2": np.r_[s3, g1, g15], "Rsh": np.r_[s3, g1, g15]}
    gn = G.geom_names(None, CROSS)
    for g, f in GROUPS.items():
        sets["grp:" + g] = np.r_[s3, [n3 + i for i, n in enumerate(gn) if f(n.split(".")[0])]]
    return sets


def permute_geom(X, rng):
    """permute the geometry block (all columns after S3) across rows, the block moving together."""
    Xp = X.copy(); n3 = len(S3_NAMES); Xp[:, n3:] = X[rng.permutation(len(X)), n3:]; return Xp


def allX(F, rows):
    return np.concatenate([F["S3"][rows], F["G1"][rows], F["G15"][rows]], 1).astype(np.float64)


def target(F, kind, H, rows):
    """kind lnrv: ln RV(a,a+H); absr: ln(|r_H|+1e-5). Returns (y to fit, v used by Spearman and QLIKE)."""
    if kind == "lnrv":
        v = F[f"rv{H}"][rows]; return np.log(v), v
    return np.log(F[f"absr{H}"][rows] + V.EPS_Y), F[f"v2_{H}"][rows]


def qlike_c(v_hold, mu_hold, kind="lnrv"):
    """scale c on the hold rows. lnrv (mu = ln RV): c = mean(v / e^mu). absr (mu = ln|r|, v = r^2): c = mean(v / e^(2 mu))."""
    return float(np.mean(v_hold / np.exp(mu_hold))) if kind == "lnrv" else V.scale_c(v_hold, mu_hold)


def qlike(v, mu, c, kind="lnrv"):
    """QLIKE = mean(q - ln q - 1), q = v / (c e^mu) for the ln RV target (SPEC), v / (c e^(2 mu)) for the ln|r| target."""
    if kind == "absr":
        return V.qlike(v, mu, c)
    q = v / (c * np.exp(mu)); return float(np.mean(q - np.log(q) - 1))


def fit_state(F, H, m0, groups=False, hgb=True, seed=0):
    """Fit every model on the rows for a test starting m0. Returns a state used by score_state. Models: R0 R1 R2 Rsh (ridge, both targets),
    H0 / H2 (HistGradientBoosting on S3 / all, primary target only), optional group sets."""
    tr = train_rows(F["ta"], F[f"ok{H}"], H, m0); fit, hold = split_fit_hold(tr, H)
    sets = sets_def()
    if not groups:
        sets = {k: v for k, v in sets.items() if not k.startswith("grp:")}
    rng = np.random.default_rng(seed)
    perm_tr = rng.permutation(len(tr))                     # one permutation of the training rows' geometry
    Xtr = allX(F, tr); n3 = len(S3_NAMES)
    Xp = Xtr.copy(); Xp[:, n3:] = Xtr[perm_tr, n3:]
    pos = {r: i for i, r in enumerate(tr)}
    fi = np.array([pos[r] for r in fit]); hi = np.array([pos[r] for r in hold])
    st = {"H": H, "tr": tr, "sets": sets, "m": {}, "c": {}, "rng_seed": seed}
    Y = {k: target(F, k, H, tr) for k in ("lnrv", "absr")}
    for part, ix in (("full", slice(None)), ("fit", fi)):
        for X, names in ((Xtr, [k for k in sets if k != "Rsh"]), (Xp, ["Rsh"])):
            rr = Ridge(X[ix])
            for kind in ("lnrv", "absr"):
                for nm in names:
                    st["m"][(kind, part, nm)] = (rr, rr.fit(Y[kind][0][ix], sets[nm]))
            rr.Xs = None; rr.G = None
    for kind in ("lnrv", "absr"):
        v = Y[kind][1]
        for nm in sets:
            X = Xp if nm == "Rsh" else Xtr
            rr, m = st["m"][(kind, "fit", nm)]
            st["c"][(kind, nm)] = qlike_c(v[hi], rr.predict(m, X[hi]), kind)
    if hgb:
        from sklearn.ensemble import HistGradientBoostingRegressor
        y, v = target(F, "lnrv", H, tr); sub = fi[::10]
        for nm, cols in (("H0", sets["R0"]), ("H2", sets["R2"])):
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                                early_stopping=False, random_state=0).fit(Xtr[sub][:, cols], y[sub])
            st["m"][("lnrv", "hgb", nm)] = (mdl, cols); st["c"][("lnrv", nm)] = qlike_c(v[hi], mdl.predict(Xtr[hi][:, cols]))
    return st


def score_state(st, F, te, span, month, seed=0):
    """Metric rows for the test rows te (indices into F)."""
    H = st["H"]; Xte = allX(F, te); rng = np.random.default_rng(seed + 1)
    Xtp = permute_geom(Xte, rng); rows = []
    for kind in ("lnrv", "absr"):
        _, v = target(F, kind, H, te)
        for nm in st["sets"]:
            if kind == "absr" and nm.startswith("grp:"):
                continue
            rr, m = st["m"][(kind, "full", nm)]; mu = rr.predict(m, Xtp if nm == "Rsh" else Xte)
            ok = np.std(mu) > 0
            rows.append({"span": span, "month": month, "H": H, "target": kind, "model": nm, "n_te": int(len(te)),
                         "spearman": float(spearmanr(mu, v)[0]) if ok else None,
                         "qlike": None if nm.startswith("grp:") else qlike(v, mu, st["c"][(kind, nm)], kind)})
        if kind == "lnrv":
            for nm in ("H0", "H2"):
                if ("lnrv", "hgb", nm) in st["m"]:
                    mdl, cols = st["m"][("lnrv", "hgb", nm)]; mu = mdl.predict(Xte[:, cols])
                    rows.append({"span": span, "month": month, "H": H, "target": kind, "model": nm, "n_te": int(len(te)),
                                 "spearman": float(spearmanr(mu, v)[0]), "qlike": qlike(v, mu, st["c"][(kind, nm)], "lnrv")})
    return rows
