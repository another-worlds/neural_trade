"""T03 (SPEC.md): forward test of volatility-based return intervals. Reuses T01's guarded loaders and frozen S1 vol-change model. CPU only."""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "tasks", "T01_vol_forward"))
import t01_lib as T1  # noqa: E402  (loaders with the forbidden-span assertion, build, fit, predict)
V = T1.V

HZ = (15, 60, 240)
ALPHA = 0.10            # interval score / central 90% interval
VAR_P = 0.05
EPS_Y = 1e-5
SCORED = [str(m) for m in pd.period_range("2025-11", "2026-09", freq="M")]
FIRST_TRAIN = 1441


def make_data(s, first):
    """V.Data with the three horizons (anchors stride 7). Features, log-return targets and the trailing scale for each T."""
    D = V.Data(s.values.astype(np.float64), s.index, horizons=HZ, stride=T1.STRIDE, first=first, with_tb7=False)
    a = D.anch; c = D.A[:, 3]; cs = D.cs2
    out = {"D": D, "r": {}, "trail": {}}
    for H in HZ:
        out["r"][H] = np.log(c[a + H - 1]) - np.log(c[a - 1])                      # label bars a .. a+H-1
        out["trail"][H] = np.sqrt(np.maximum(cs[a] - cs[a - 4 * H], 0.0) / 4.0)   # sqrt(RV over 4H bars, scaled to H); bars < a only
    out["ok"] = np.all(D.har[:, :4] > -20, 1)
    return out


def row_ok(B, H):
    return B["ok"] & (B["trail"][H] > 0)


def train_rows(B, H, m0, days=T1.TRAIN_DAYS):
    ta = B["D"].ta
    m = np.asarray((ta >= m0 - pd.Timedelta(days=days)) & (ta + pd.Timedelta(minutes=H) <= m0))
    return np.where(m & row_ok(B, H))[0]


def s1_pred(s1model, B):
    mu, sd, coef, ic = s1model
    return ((B["D"].har - mu) / sd) @ coef + ic


def design(B, idx, s1m, plus):
    X = B["D"].har[idx]
    return np.column_stack([X, s1_pred(s1m, B)[idx]]) if plus else X


def fit_har(B, H, rows, s1m, plus):
    X = design(B, rows, s1m, plus); y = np.log(np.abs(B["r"][H][rows]) + EPS_Y)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    r = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    m = {"mu": mu, "sd": sd, "coef": r.coef_.copy(), "ic": float(r.intercept_), "plus": plus}
    e = np.exp(((X - mu) / sd) @ m["coef"] + m["ic"])
    m["c"] = float(np.mean(np.abs(B["r"][H][rows]) / e))
    return m


def sigma(model, B, idx, s1m):
    X = design(B, idx, s1m, model["plus"])
    return np.exp(((X - model["mu"]) / model["sd"]) @ model["coef"] + model["ic"]) * model["c"]


def sigma_all(frozen, B, H, idx):
    s1m = frozen["s1"]
    return {"trail": B["trail"][H][idx], "HAR": sigma(frozen[H]["HAR"], B, idx, s1m), "HAR+": sigma(frozen[H]["HAR+"], B, idx, s1m)}


def quantiles(z):
    """lower/upper bounds (5%, 95%), the 5% quantile and the ES factor (mean of z at or below it) of normalised returns z."""
    lo, hi = np.quantile(z, [ALPHA / 2, 1 - ALPHA / 2]); q = np.quantile(z, VAR_P)
    return {"lo": float(lo), "hi": float(hi), "q05": float(q), "es": float(z[z <= q].mean())}


def cal_rows(B, H, month):
    """rows that recalibrate for scored `month`: the previous calendar month, labels ending by the month start (strictly before it)."""
    m0 = month.start_time; p0 = (month - 1).start_time; ta = B["D"].ta
    m = np.asarray((ta >= p0) & (ta + pd.Timedelta(minutes=H) <= m0))
    return np.where(m & row_ok(B, H))[0]


def month_rows(B, H, month):
    m0 = month.start_time; m1 = (month + 1).start_time; ta = B["D"].ta
    te = np.asarray((ta >= m0) & (ta < m1))
    return np.where(te & row_ok(B, H))[0], int(te.sum())


def fz0(y, v, e):
    """Fissler-Ziegel FZ0 (Patton-Ziegel-Chen 2019) of the lower-tail (VaR v, ES e < 0) pair at level VAR_P for returns y."""
    return float(np.mean(-(y <= v).astype(float) * (v - y) / (VAR_P * e) + v / e + np.log(-e) - 1.0))


def score(r, sg, q):
    lo, hi = q["lo"] * sg, q["hi"] * sg; v = q["q05"] * sg; e = q["es"] * sg; a = ALPHA
    isc = (hi - lo) + (2 / a) * (lo - r) * (r < lo) + (2 / a) * (r - hi) * (r > hi)
    return {"n": int(len(r)), "cover": float(np.mean((r >= lo) & (r <= hi))), "width_bps": float(np.mean(hi - lo) * 1e4),
            "iscore_bps": float(np.mean(isc) * 1e4), "var_hit": float(np.mean(r < v)), "fz0": fz0(r, v, e), "n_in": int(np.sum((r >= lo) & (r <= hi)))}


def score_month(frozen, BF, H, month):
    """All forecasters x {frozen, cal} for one scored month. `frozen` is read, never modified."""
    idx, n_all = month_rows(BF, H, month); cidx = cal_rows(BF, H, month); r = BF["r"][H][idx]
    sg = sigma_all(frozen, BF, H, idx); cg = sigma_all(frozen, BF, H, cidx); out = []
    for f in ("trail", "HAR", "HAR+"):
        qc = quantiles(BF["r"][H][cidx] / cg[f])
        for ver, q in (("frozen", frozen[H]["q"][f]), ("cal", qc)):
            d = score(r, sg[f], q); d.update({"month": str(month), "H": H, "f": f, "ver": ver, "n_dropped": n_all - len(idx), "n_cal": int(len(cidx))})
            out.append(d)
    return out


def fit_frozen(BT, S1B):
    """Fit everything once on the 365 days ending at the cutoff. S1B = T01's build of the same training bars (for the S1 vol-change model)."""
    m0 = T1.CUTOFF + pd.Timedelta(minutes=1)
    tr = {tg: T1.train_idx(S1B, tg, m0) for tg in T1.TARGETS}
    s1m = T1.fit(S1B, tr)["A"]["S1"]
    fr = {"s1": s1m, "m0": m0}
    for H in HZ:
        rows = train_rows(BT, H, m0)
        fr[H] = {"n_train": int(len(rows)), "HAR": fit_har(BT, H, rows, s1m, False), "HAR+": fit_har(BT, H, rows, s1m, True)}
        sg = {"trail": BT["trail"][H][rows], "HAR": sigma(fr[H]["HAR"], BT, rows, s1m), "HAR+": sigma(fr[H]["HAR+"], BT, rows, s1m)}
        fr[H]["q"] = {f: quantiles(BT["r"][H][rows] / sg[f]) for f in sg}
        fr[H]["rows_last_label_end"] = str(BT["D"].ta[rows].max() + pd.Timedelta(minutes=H))
    return fr


def ci(x):
    """mean, 95% t-interval, one-sided p for mean < 0."""
    x = np.asarray(x, float); n = len(x); mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    t = mu / se if se > 0 else 0.0
    return float(mu), float(mu - h), float(mu + h), float(stats.t.cdf(t, n - 1))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj
