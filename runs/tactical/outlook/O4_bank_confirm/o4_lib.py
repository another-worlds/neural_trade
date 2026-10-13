"""O4 (SPEC.md): confirm the EWMA-bank volatility forecast (O3's BANK) against R0 on data that did not choose it. Reuses o3_lib read-only
(bank, rows, ridge, QLIKE); only R0 and BANK are fitted (no kernel model). CPU only."""
import hashlib, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "O3_synth_volind"))
import o3_lib as L  # noqa: E402
T = L.T

A_LO = pd.Timestamp("2017-08-17 00:00"); A_HI = pd.Timestamp("2018-12-31 23:59")
A_LOAD_HI = A_HI + pd.Timedelta(minutes=max(L.HORIZONS))           # the H-minute label tail of the last anchors; read only as labels
A_MONTHS = [str(m) for m in pd.period_range("2017-09", "2018-12", freq="M")]
B_LO = pd.Timestamp("2025-10-01 00:00"); B_HI = pd.Timestamp("2026-09-30 23:59"); B_M0 = pd.Timestamp("2026-10-01 00:00")
MODELS = ("R0", "BANK")


def month_bounds(m):
    p = pd.Period(m, freq="M"); return p.start_time, (p + 1).start_time


def fit_rb(F, H, m0):
    """R0 and BANK exactly as o3_lib.fit_month: the model is refit on all training rows, the QLIKE scale c comes from the 15% hold-out of the
    first-85% fit. Returns {'H', 'n_tr', 'm': {R0, BANK}, 'c': {R0, BANK}, first_ta, last_label_end}."""
    tr = L.train_rows(F["ta"], F[f"ok{H}"], H, m0); fit, hold = L.split_fit_hold(tr, H)
    y = np.log(F[f"rv{H}"]); v = F[f"rv{H}"]
    S3t = F["S3"][tr]; XBt = np.concatenate([S3t, L.log_bank(F["bank"][tr])], 1); yt = y[tr]; vt = v[tr]
    pos = {r: i for i, r in enumerate(tr)}; fi = np.array([pos[r] for r in fit]); hi = np.array([pos[r] for r in hold])
    st = {"H": H, "n_tr": int(len(tr)), "m": {}, "c": {}}
    for nm, Xt in (("R0", S3t), ("BANK", XBt)):
        mf = L.ridge_fit(Xt[fi], yt[fi]); st["m"][nm] = L.ridge_fit(Xt, yt)
        st["c"][nm] = L.qlike_c(vt[hi], L.ridge_pred(mf, Xt[hi]))
    st["first_ta"] = str(pd.Timestamp(F["ta"][tr].min())) if len(tr) else None
    st["last_label_end"] = str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=H)) if len(tr) else None
    return st


def predict_rb(st, F, te):
    S3 = F["S3"][te]; XB = np.concatenate([S3, L.log_bank(F["bank"][te])], 1)
    return {"R0": L.ridge_pred(st["m"]["R0"], S3), "BANK": L.ridge_pred(st["m"]["BANK"], XB)}


def score_rb(st, F, te, span, month):
    H = st["H"]; v = F[f"rv{H}"][te]; rows = []
    for nm, mu in predict_rb(st, F, te).items():
        rows.append({"span": span, "month": month, "H": H, "model": nm, "n_te": int(len(te)), "n_tr": st["n_tr"],
                     "spearman": float(spearmanr(mu, v)[0]) if np.std(mu) > 0 else None, "qlike": L.qlike(v, mu, st["c"][nm])})
    return rows


def paired(df, a, b, col, months, min_n=3):
    pa = df[df.model == a].set_index("month")[col]; pb = df[df.model == b].set_index("month")[col]
    d = (pa - pb).reindex(months).dropna()
    if len(d) < min_n:
        return None
    mu, lo, hi, _ = T.ci(d.values)
    return {"mean": mu, "lo": lo, "hi": hi, "p": float(stats.ttest_1samp(d.values, 0).pvalue), "pos": int((d > 0).sum()), "n": int(len(d))}


def criterion(sp, ql, need):
    """Criterion A / B at H = 60 (BANK - R0): mean Spearman gain >= 0.005, 95% t lower bound > 0, positive months >= need, mean QLIKE diff <= 0."""
    c1 = sp["mean"] >= 0.005; c2 = sp["lo"] > 0; c3 = sp["pos"] >= need; c4 = ql["mean"] <= 0
    return {"c1_mean_ge_0.005": bool(c1), "c2_lo_gt_0": bool(c2), "c3_months": bool(c3), "need_positive": int(need), "c4_qlike_le_0": bool(c4),
            "pass": bool(c1 and c2 and c3 and c4)}


def holm2(ps):
    o = np.argsort(ps); out = [None, None]
    out[o[0]] = min(1.0, ps[o[0]] * 2); out[o[1]] = min(1.0, max(ps[o[1]], out[o[0]]))
    return out


def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ridge_json(m):
    return {"mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "w": m["w"].tolist(), "ym": float(m["ym"])}


def _ridge_obj(j):
    return {"mu": np.array(j["mu"]), "sd": np.array(j["sd"]), "w": np.array(j["w"]), "ym": float(j["ym"])}


def state_to_json(st):
    return {"H": st["H"], "n_tr": st["n_tr"], "first_ta": st["first_ta"], "last_label_end": st["last_label_end"],
            "c": {k: float(v) for k, v in st["c"].items()}, "m": {k: _ridge_json(v) for k, v in st["m"].items()}}


def state_from_json(j):
    return {"H": j["H"], "n_tr": j["n_tr"], "first_ta": j["first_ta"], "last_label_end": j["last_label_end"],
            "c": dict(j["c"]), "m": {k: _ridge_obj(v) for k, v in j["m"].items()}}


def load_freeze_bars(path=None):
    """Binance bars B_LO..B_HI only; reads exactly the rows up to B_HI (nrows) when the file has no gaps, else the guarded chunked loader."""
    path = path or T.FWD_CSV; n = int((B_HI - B_LO).total_seconds() // 60) + 1
    df = pd.read_csv(path, usecols=T.COLS, nrows=n); ts = pd.to_datetime(df["timestamp"]); df.index = ts; df = df.drop(columns="timestamp")
    assert ts.iloc[0] == B_LO
    if ts.iloc[-1] < B_HI or len(df) < n:
        return T.load_bars(path, B_LO, B_HI)
    df = df[(df.index >= B_LO) & (df.index <= B_HI)]
    assert df.index.min() >= B_LO and df.index.max() <= B_HI
    assert not ((df.index >= T.FORBID_LO) & (df.index <= T.FORBID_HI)).any()
    return df.sort_index()


def freeze(bars):
    """Fit R0 and BANK ONCE per horizon on every anchor with a label end <= 2026-10-01 00:00 (the first anchor is the first warm-up-complete one)."""
    assert bars.index.max() <= B_HI
    F = L.build_features(bars)
    art = {"spec": "O4 SPEC.md Part B freeze", "last_bar": str(bars.index.max()), "n_bars": int(len(bars)), "file_first_bar": str(bars.index.min()),
           "stride": T.STRIDE, "warm_bars": L.FIRST, "ridge_alpha": 1.0, "horizons": list(L.HORIZONS), "taus": list(L.TAUS),
           "eps_ch": L.EPS_CH.tolist(), "channels": list(L.CHANNELS), "s3_names": L.S3_NAMES,
           "bank_features": "S3 (16) + ln(EWMA_tau(channel) + eps_channel), channel-major, 36", "o3_lib_note": "definitions of o3_lib.py at 5e4ab42",
           "states": {}}
    for H in L.HORIZONS:
        art["states"][str(H)] = state_to_json(fit_rb(F, H, B_M0))
    art["sha256"] = sha_of(art); return art
