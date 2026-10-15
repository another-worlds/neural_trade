"""T11 (SPEC.md): correct clocks. Reuses T01's guarded loaders, S3 features, target A (next-hour change in log RV), drop filter and ridge fit, and
T04's freeze / artefact pattern; adds the calendar variants V0..V3 from `common/timefix.py`, the secondary 15-minute |r| model, the monthly scoring
and the prospective verdict logic. CPU only.

Variants (ridge alpha 1, standardised; all on top of S3 = T01's 16 columns, whose columns 4 and 5 are the hour sin/cos of the last bar's FILE time):
  V0  S3 as T01.
  V1  S3 with columns 4 and 5 replaced by the hour sin/cos of the last bar's UTC time.
  V2  S3 + the New York basis (NY_NAMES, 14 columns: see the deviation note in the report: the SPEC lists 4+1+1+1+1+4+2 = 14 components, titled 13).
  V3  S3 + 96 UTC 15-minute slot dummies (T04-B).
Every calendar term other than V1's hour uses the anchor time `ta` (the minute the last bar closes = the start of the forecast hour), converted to UTC
(and to New York time). Secondary target: ln |r_15| (T03's HAR design: ln RV 15/60/240/1440 + the hour terms, same variants on that base)."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "common")); sys.path.insert(0, os.path.join(TAC, "tasks", "T01_vol_forward"))
import timefix as TF  # noqa: E402
import t01_lib as T  # noqa: E402
V = T.V

VARIANTS = ["V0", "V1", "V2", "V3"]
SEL_LO, SEL_HI = pd.Timestamp("2017-12-01 00:00"), pd.Timestamp("2024-07-02 00:00")     # the cut-file span the selection reads
SEL_MONTHS = [str(m) for m in pd.period_range("2019-01", "2024-06", freq="M")]
A_LO, A_HI = pd.Timestamp("2017-08-17 00:00"), pd.Timestamp("2018-12-31 23:59")            # report only (T04-A's span)
A_MONTHS = [str(m) for m in pd.period_range("2017-09", "2018-12", freq="M")]
FW_LO, FW_HI = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")          # forward file, freeze and report-only reads stop here
FW_MONTHS = [str(m) for m in pd.period_range("2025-11", "2026-09", freq="M")]
B_TRAIN_FROM = pd.Timestamp("2025-10-02 00:00"); B_M0 = pd.Timestamp("2026-10-01 00:00")  # freeze: rows from here, labels ending by B_M0
SCORE_START = pd.Timestamp("2026-10-02 00:00")                                             # prospective scoring starts here
POCOCK_ALPHA = 0.0294; LOOKS = (3, 6); NEED_POSITIVE = {3: 2, 6: 5}
MIN_DIFF = 0.01; MIN_SPEARMAN = 0.005
N_SLOT = 96; SEAM_PAD = 61; H15 = 15; EPS_R = 1e-5
NY_NAMES = (["ny_sin1", "ny_cos1", "ny_sin2", "ny_cos2", "ny_open", "ny_close", "weekend", "qhour"] +
            [f"wk_{n}" for n in ("ny_sin1", "ny_cos1", "ny_sin2", "ny_cos2")] + ["min_sin", "min_cos"])


# ---------------------------------------------------------------- calendar (pure functions of the anchor timestamps)
def calendar(ta, source):
    """All calendar arrays for anchors with close-time `ta` (file time of `source`): the last bar's UTC hour sin/cos (V1), the 14 New York terms
    (V2) and the UTC 15-minute slot (V3). Only the timestamps are read."""
    ta = pd.DatetimeIndex(ta); utc = TF.to_utc(ta, source); last = TF.to_utc(ta - pd.Timedelta(minutes=1), source)
    ny = TF.ny_time(utc); h = ny.hour.values + ny.minute.values / 60.0
    ph = lambda k: 2 * np.pi * k * h / 24.0
    harm = np.column_stack([np.sin(ph(1)), np.cos(ph(1)), np.sin(ph(2)), np.cos(ph(2))])
    wk = (utc.dayofweek.values >= 5).astype(float); m = utc.minute.values
    cols = [harm, ((h >= 9.5) & (h < 10.5))[:, None], ((h >= 15.5) & (h < 16.5))[:, None], wk[:, None], (np.isin(m % 15, (0, 1, 14)))[:, None],
            harm * wk[:, None], np.column_stack([np.sin(2 * np.pi * m / 60.0), np.cos(2 * np.pi * m / 60.0)])]
    hr = last.hour.values
    return {"hr_sc": np.column_stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)]), "ny": np.column_stack(cols).astype(float),
            "slot": (utc.hour.values * 60 + utc.minute.values) // 15}


def build(s, first, source):
    """T01's build plus the calendar arrays, the secondary target and the seam exclusion (bitstamp only). Returns the T01 dict extended with
    B['cal'], B['r15'], B['v15'], B['ok'] (T01's filter AND no seam within SEAM_PAD minutes) and B['drops']."""
    B = T.build(s, first); D = B["D"]; ta = D.ta
    B["cal"] = calendar(ta, source); B["source"] = source
    c = D.A[:, 3]; a = D.anch
    B["r15"] = np.log(c[a + H15 - 1]) - np.log(c[a - 1]); B["v15"] = B["r15"] ** 2 + V.EPS_V
    ok_t01 = B["ok"].copy(); seam = TF.seam_mask(ta, SEAM_PAD, source)
    B["ok"] = ok_t01 & ~seam
    B["drops"] = {"anchors": int(len(a)), "kept": int(B["ok"].sum()), "t01_drops": int((~ok_t01).sum()), "seam_excluded": int((ok_t01 & seam).sum()),
                  "seam_fraction_of_anchors": float(seam.mean())}
    return B


def X_A(B, name, idx):
    """Target-A design matrix of variant `name` for rows idx."""
    S3 = B["F"]["S3"][idx]; cal = B["cal"]
    if name == "V0":
        return S3
    if name == "V1":
        X = S3.copy(); X[:, 4:6] = cal["hr_sc"][idx]; return X
    if name == "V2":
        return np.hstack([S3, cal["ny"][idx]])
    if name == "V3":
        return np.hstack([S3, np.eye(N_SLOT)[cal["slot"][idx]]])
    raise KeyError(name)


def X_R(B, name, idx):
    """Secondary (|r_15|) design matrix: T03's HAR (ln RV 15/60/240/1440 + hour sin/cos) with the same variant additions."""
    H = B["D"].har[idx]; cal = B["cal"]
    if name == "V0":
        return H
    if name == "V1":
        X = H.copy(); X[:, 4:6] = cal["hr_sc"][idx]; return X
    if name == "V2":
        return np.hstack([H, cal["ny"][idx]])
    if name == "V3":
        return np.hstack([H, np.eye(N_SLOT)[cal["slot"][idx]]])
    raise KeyError(name)


# ---------------------------------------------------------------- fit / predict / score
def train_rows(B, m0, start=None, days=T.TRAIN_DAYS):
    """Rows (A, R) available before m0 whose label ends by m0: the 60-minute label for A, the 15-minute label for R."""
    ta = B["D"].ta; lo = (m0 - pd.Timedelta(days=days)) if start is None else start
    a = np.where(np.asarray((ta >= lo) & (ta + pd.Timedelta(minutes=60) <= m0)) & B["ok"])[0]
    r = np.where(np.asarray((ta >= lo) & (ta + pd.Timedelta(minutes=H15) <= m0)) & B["ok"])[0]
    return a, r


def _ridge(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12; r = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": r.coef_.tolist(), "intercept": float(r.intercept_)}


def _lin(m, X):
    return ((X - np.array(m["mu"])) / np.array(m["sd"])) @ np.array(m["coef"]) + m["intercept"]


def fit(B, tr_a, tr_r, names=VARIANTS):
    """ar (OLS on L0 - L1) and one ridge per variant for target A on rows tr_a; per variant a ridge of ln(|r_15| + 1e-5) with its variance scale c
    on rows tr_r. Plain dicts and lists, so the frozen artefact is JSON."""
    y = B["T"]["A"][tr_a]; out = {"A": {"ar": [float(v) for v in T.ols1(B["AR"]["A"][tr_a], y)]}, "R": {}}
    yr = np.log(np.abs(B["r15"][tr_r]) + EPS_R)
    for nm in names:
        out["A"][nm] = _ridge(X_A(B, nm, tr_a), y); m = _ridge(X_R(B, nm, tr_r), yr)
        mu_tr = _lin(m, X_R(B, nm, tr_r)); m["c"] = V.scale_c(B["v15"][tr_r], mu_tr); out["R"][nm] = m
    return out


def predict_A(models, B, idx):
    p = {"rw": np.zeros(len(idx))}; b = models["A"]["ar"]; p["ar"] = b[0] + b[1] * B["AR"]["A"][idx]
    for nm, m in models["A"].items():
        if nm != "ar":
            p[nm] = _lin(m, X_A(B, nm, idx))
    return p


def score_month(models, B, month, m_start=None):
    """One scored month (rows with ta in [max(month start, m_start), next month start)): target A mse and skill vs ar per model, and for the
    secondary target the Spearman of the prediction with |r_15| and the QLIKE of the variance forecast, per variant."""
    ta = B["D"].ta; m0 = month.start_time if m_start is None else max(month.start_time, m_start); m1 = (month + 1).start_time
    te = np.asarray((ta >= m0) & (ta < m1)); idx = np.where(te & B["ok"])[0]; yt = B["T"]["A"][idx]
    p = predict_A(models, B, idx); mse = {k: float(np.mean((yt - v) ** 2)) for k, v in p.items()}
    sec = {}
    for nm, m in models["R"].items():
        mu = _lin(m, X_R(B, nm, idx)); sec[nm] = {"spearman": float(spearmanr(mu, np.abs(B["r15"][idx]))[0]),
                                                  "qlike": V.qlike(B["v15"][idx], mu, m["c"])}
    return {"month": str(month), "n_te": int(len(idx)), "n_anchors": int(te.sum()), "mse": mse,
            "skill_ar": {k: 1 - mse[k] / mse["ar"] for k in mse}, "sec": sec}


def walk_forward(B, months, label, fixed_start=None, log=None):
    """Monthly refit: for each month, train on the 365 days before it (or from `fixed_start` on) with labels ending by the month start."""
    rows = []
    for m in pd.PeriodIndex(months, freq="M"):
        tra, trr = train_rows(B, m.start_time, start=fixed_start); mod = fit(B, tra, trr)
        r = score_month(mod, B, m); r.update(n_tr=int(len(tra)), set=label); rows.append(r)
        if log:
            log(m, r)
    return rows


# ---------------------------------------------------------------- statistics
def paired(rows, a, b="V0"):
    """Paired monthly differences (a minus b) of skill vs ar [target A] and of Spearman / QLIKE [secondary]; mean, 95% t-CI, months positive, one-sided p."""
    d = np.array([r["skill_ar"][a] - r["skill_ar"][b] for r in rows]); ds = np.array([r["sec"][a]["spearman"] - r["sec"][b]["spearman"] for r in rows])
    dq = np.array([r["sec"][a]["qlike"] - r["sec"][b]["qlike"] for r in rows]); rr = np.array([1 - r["mse"][a] / r["mse"][b] for r in rows])
    out = {"a": a, "b": b, "n_months": len(d)}
    for key, x in (("skill_ar_diff", d), ("mse_ratio_skill", rr), ("spearman_diff", ds), ("qlike_diff", dq)):
        mu, lo, hi, _ = T.ci(x) if len(x) > 1 else (float(x.mean()), float("nan"), float("nan"), 0.0)
        out[key] = {"mean": mu, "ci": [lo, hi], "positive": int((x > 0).sum())}
    return out


def pocock_ci(x, alpha=POCOCK_ALPHA):
    x = np.asarray(x, float); n = len(x); mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    p = float(1 - stats.t.cdf(mu / se, n - 1)) if se > 0 else (0.0 if mu > 0 else 1.0)
    return float(mu), float(mu - h), float(mu + h), p


def verdict(rows, chosen, look):
    """The prospective criterion at a look (the first `look` complete months): chosen minus V0 in monthly skill vs ar >= +0.01, Pocock CI above 0,
    at least 2 of 3 (5 of 6) months positive; secondary: Spearman of the 15-minute |r| >= +0.005 and QLIKE not worse; Holm over the two targets
    on the one-sided p-values at the look's Pocock alpha."""
    rows = rows[:look]; d = np.array([r["skill_ar"][chosen] - r["skill_ar"]["V0"] for r in rows])
    ds = np.array([r["sec"][chosen]["spearman"] - r["sec"]["V0"]["spearman"] for r in rows]); dq = np.array([r["sec"][chosen]["qlike"] - r["sec"]["V0"]["qlike"] for r in rows])
    mu, lo, hi, p1 = pocock_ci(d); mus, los, his, p2 = pocock_ci(ds)
    order = sorted([("A", p1), ("R", p2)], key=lambda t: t[1]); holm = {}; ok = True
    for k, (nm, p) in enumerate(order):
        ok = ok and (p <= POCOCK_ALPHA / (2 - k)); holm[nm] = bool(ok)
    prim = {"mean_diff": mu, "pocock_ci": [lo, hi], "p_one_sided": p1, "positive_months": int((d > 0).sum()), "need_positive": NEED_POSITIVE[look],
            "c1_mean_ge_0.01": bool(mu >= MIN_DIFF), "c2_ci_lo_gt_0": bool(lo > 0), "c3_months": bool((d > 0).sum() >= NEED_POSITIVE[look]), "holm": holm["A"]}
    prim["pass"] = bool(prim["c1_mean_ge_0.01"] and prim["c2_ci_lo_gt_0"] and prim["c3_months"] and prim["holm"])
    sec = {"spearman_diff_mean": mus, "pocock_ci": [los, his], "p_one_sided": p2, "qlike_diff_mean": float(dq.mean()), "c1_spearman_ge_0.005": bool(mus >= MIN_SPEARMAN),
           "c2_qlike_not_worse": bool(dq.mean() <= 0), "holm": holm["R"]}
    sec["pass"] = bool(sec["c1_spearman_ge_0.005"] and sec["c2_qlike_not_worse"] and sec["holm"])
    return {"look": look, "months": [r["month"] for r in rows], "chosen": chosen, "primary_A": prim, "secondary_R": sec, "pass_primary": prim["pass"]}


# ---------------------------------------------------------------- artefact
def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_freeze_bars(path):
    """Binance bars FW_LO..FW_HI only: reads exactly the rows up to FW_HI (nrows), so no later row is parsed when the file has no gaps."""
    n = int((FW_HI - FW_LO).total_seconds() // 60) + 1
    df = pd.read_csv(path, usecols=T.COLS, nrows=n); ts = pd.to_datetime(df["timestamp"]); df.index = ts; df = df.drop(columns="timestamp")
    assert ts.iloc[0] == FW_LO
    if ts.iloc[-1] < FW_HI:                                     # gaps: fall back to the guarded chunked loader
        return T.load_bars(path, FW_LO, FW_HI, chunk=100_000)
    df = df[(df.index >= FW_LO) & (df.index <= FW_HI)]
    assert df.index.min() >= FW_LO and df.index.max() <= FW_HI
    assert not ((df.index >= T.FORBID_LO) & (df.index <= T.FORBID_HI)).any()
    return df.sort_index()


def freeze(bars, chosen, selection_sha, train_from=B_TRAIN_FROM, m0=B_M0):
    """Fit V0, V1, the chosen variant and V3 ONCE on rows with ta >= train_from and label end <= m0; the artefact (with its sha256) is JSON-able."""
    assert chosen in VARIANTS
    names = sorted({"V0", "V1", "V3", chosen}); B = build(bars, T.WARM, "binance"); tra, trr = train_rows(B, m0, start=train_from)
    models = fit(B, tra, trr, names); ta = B["D"].ta[tra]
    art = {"spec": "T11 SPEC.md freeze", "chosen": chosen, "selection_json_sha256": selection_sha, "variants": names, "train_from": str(train_from),
           "train_label_end_max": str(ta.max() + pd.Timedelta(minutes=60)), "last_bar": str(bars.index.max()), "n_bars": int(len(bars)),
           "n_train_rows_A": int(len(tra)), "n_train_rows_R": int(len(trr)), "stride": T.STRIDE, "warm": T.WARM, "ridge_alpha": 1.0,
           "ny_names": NY_NAMES, "drops": B["drops"], "models": models}
    art["sha256"] = sha_of(art); return art


def assert_committed(path):
    """The order of the SPEC is binding: refuse unless `path` is tracked, unmodified against HEAD and HEAD's last commit touching it exists."""
    import subprocess
    cwd = os.path.dirname(os.path.abspath(path)); name = os.path.basename(path)
    g = lambda *a: subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if g("ls-files", "--error-unmatch", name).returncode != 0:
        raise RuntimeError(f"{name} is not tracked: commit selection.json before the report-only looks")
    if g("diff", "--quiet", "HEAD", "--", name).returncode != 0:
        raise RuntimeError(f"{name} differs from HEAD: commit it unchanged first")
    return g("log", "-1", "--format=%H", "--", name).stdout.strip()
