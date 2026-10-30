"""Paired ablations on identical origins: variant vs base, both scored on the variant's origin subset.

Answers "does X help?" directly (ctx 8192 vs 2048, covariates vs none, symmetric averaging off vs on, size),
instead of comparing two numbers that are each measured against EWMA-t on different origin sets.
Values: % change of mean loss (variant vs base); negative = variant better. CI: paired moving-block bootstrap.
QLIKE: per-origin difference of y^2/v + ln v (all origins), as % of the base's unit-free mean on y != 0 (as in
metrics2.py).
Writes RESULTS/matched.json and prints a markdown table. Set BENCH_DATA=post for the post-release set.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, HORIZONS, QLEVELS, RESULTS, load, origins  # noqa: E402

LV = np.array(QLEVELS)
DEC = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
EDGES = np.concatenate([[0.0], (LV[1:] + LV[:-1]) / 2, [1.0]])
W_MEAN = np.diff(EDGES)
NBOOT = 1000
rs = np.random.default_rng(0)

PAIRS = {
    "repo": [
        ("timestral-209M_ctx8192_price_sub6", "timestral-209M_ctx2048_price"),
        ("timestral-209M_ctx2048_price_cov_sub6", "timestral-209M_ctx2048_price"),
        ("timestral-28M_ctx8192_price_sub6", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx32768_price_sub6", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_price_cov", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_price_nosym", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_logret", "timestral-28M_ctx2048_price"),
        ("timestral-209M_ctx2048_logret", "timestral-209M_ctx2048_price"),
        ("timestral-209M_ctx2048_price", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_price", "timestral-28M_ctx512_price"),
        ("official-chronos2_ctx2048_logret", "official-chronos2_ctx2048_price"),
        ("official-chronos2_ctx512_logret", "official-chronos2_ctx512_price"),
        ("official-chronos2_ctx2048_logret", "timestral-28M_ctx2048_logret"),
        ("samay-timesfm25_ctx512_logret", "samay-timesfm25_ctx512_price"),
        ("neural_trade-rolling1440m-seed0_ctx60_price", "neural_trade-asis-seed0_ctx60_price"),
        ("neural_trade-rolling1440m-seed1_ctx60_price", "neural_trade-asis-seed1_ctx60_price"),
        ("neural_trade-rolling1440m-seed2_ctx60_price", "neural_trade-asis-seed2_ctx60_price"),
        ("official-chronos2_ctx512_aggret", "timestral-28M_ctx512_aggret"),
        ("official-chronos2_ctx512_aggret", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx512_aggret", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_aggret", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx2048_aggret_sub6", "timestral-28M_ctx2048_aggret"),
        ("official-chronos2_ctx2048_aggret_sub6", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx2048_aggret_sub6", "official-chronos2_ctx512_aggret"),
        ("official-chronos2_ctx2048_logret", "timestral-28M_ctx2048_price"),   # appended last: keeps earlier draws
    ],
    "post": [
        ("timestral-209M_ctx8192_price_sub6", "timestral-209M_ctx2048_price"),
        ("timestral-28M_ctx8192_price_sub6", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx32768_price_sub6", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_price_cov_sub3", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx512_price_nosym_sub6", "timestral-28M_ctx512_price"),
        ("timestral-28M_ctx2048_logret_sub6", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx8192_price_sub6", "official-chronos2_ctx2048_price"),
        ("timestral-209M_ctx2048_price", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_price", "timestral-28M_ctx512_price"),
        ("official-chronos2_ctx2048_logret", "official-chronos2_ctx2048_price"),
        ("official-chronos2_ctx512_logret", "official-chronos2_ctx512_price"),
        ("official-chronos2_ctx2048_logret", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx2048_logret", "timestral-28M_ctx2048_aggret"),
        ("samay-timesfm25_ctx512_logret", "samay-timesfm25_ctx512_price"),
        ("neural_trade-rolling1440m-seed0_ctx60_price", "neural_trade-asis-seed0_ctx60_price"),
        ("neural_trade-rolling1440m-seed1_ctx60_price", "neural_trade-asis-seed1_ctx60_price"),
        ("neural_trade-rolling1440m-seed2_ctx60_price", "neural_trade-asis-seed2_ctx60_price"),
        ("official-chronos2_ctx512_aggret", "timestral-28M_ctx512_aggret"),
        ("official-chronos2_ctx512_aggret", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx512_aggret", "timestral-28M_ctx2048_price"),
        ("timestral-28M_ctx2048_aggret", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx2048_aggret_sub6", "timestral-28M_ctx2048_aggret"),
        ("official-chronos2_ctx2048_aggret_sub6", "timestral-28M_ctx2048_price"),
        ("official-chronos2_ctx2048_aggret_sub6", "official-chronos2_ctx512_aggret"),
    ],
}[DATA]

df, _ = load()
lc = np.log(df["close"].to_numpy(np.float64))
ORG = origins(len(lc))


def pinball(q, y, idx):
    d = y[:, None] - q[:, idx]
    lv = LV[idx]
    return np.mean(np.maximum(lv * d, (lv - 1) * d), axis=1)


def qlike(q, y):
    m = q @ W_MEAN
    v = np.maximum((q ** 2) @ W_MEAN - m ** 2, 1e-6)
    out = np.full(len(y), np.nan)
    nz = y != 0
    r = y[nz] ** 2 / v[nz]
    out[nz] = r - np.log(r) - 1
    return out, y ** 2 / v + np.log(v)


def pct_ci(d, base, block, den=None):
    n = len(d)
    st = rs.integers(0, n - block + 1, size=(NBOOT, int(np.ceil(n / block))))
    idx = (st[:, :, None] + np.arange(block)).reshape(NBOOT, -1)[:, :n]
    den = abs(float(base.mean())) if den is None else den
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return round(100 * float(d.mean()) / den, 3), [round(100 * lo / den, 3), round(100 * hi / den, 3)]


def load_q(tag, org):
    z = np.load(f"{RESULTS}/{tag}.npz", allow_pickle=True)
    o = z["origins"]
    pos = np.searchsorted(o, org)
    assert np.array_equal(o[pos], org), f"{tag}: origins missing"
    q = z["q"].astype(np.float64)[pos]
    framing = str(z["framing"])
    out = {}
    for hi, h in enumerate(HORIZONS):
        if framing in ("price", "price_cov"):
            out[h] = (np.log(np.maximum(q[:, hi, :], 1e-9)) - lc[org][:, None]) * 1e4
        elif framing == "aggret":
            out[h] = q[:, hi, :]
        elif framing == "logret" and h == 1:
            out[h] = q[:, hi, :]          # multi-step log-return quantiles cannot be summed; h=1 only
    return out


rows = []
for var, base in PAIRS:
    if not (os.path.exists(f"{RESULTS}/{var}.npz") and os.path.exists(f"{RESULTS}/{base}.npz")):
        print(f"skip {var} vs {base}: file missing", file=sys.stderr)
        continue
    org = np.load(f"{RESULTS}/{var}.npz", allow_pickle=True)["origins"]
    step = int(np.median(np.diff(org)))
    block = max(5, int(round(60 / step)))
    qv, qb = load_q(var, org), load_q(base, org)
    r = {"variant": var, "base": base, "n": int(len(org)), "stride_min": step}
    for h in HORIZONS:
        if h not in qv or h not in qb:
            continue
        y = (lc[org + h] - lc[org]) * 1e4
        pv, pb = pinball(qv[h], y, DEC), pinball(qb[h], y, DEC)
        (_, sv), (ub, sb) = qlike(qv[h], y), qlike(qb[h], y)
        r[f"pinball9_h{h}"] = pct_ci(pv - pb, pb, block)
        r[f"qlike_h{h}"] = pct_ci(sv - sb, None, block, den=abs(float(np.nanmean(ub))))
    rows.append(r)

# Share of h=15 median-sign calls that equal the 15-min reversal sign (-sign of the last 15-min return), full origin set.
agree = {}
rev = -np.sign(lc[ORG] - lc[ORG - 15])
for tag in ["timestral-5M_ctx512_price", "timestral-28M_ctx512_price", "timestral-209M_ctx512_price",
            "timestral-28M_ctx2048_price", "timestral-209M_ctx2048_price", "samay-timesfm25_ctx512_price"]:
    if os.path.exists(f"{RESULTS}/{tag}.npz"):
        z = np.load(f"{RESULTS}/{tag}.npz", allow_pickle=True)
        if np.array_equal(z["origins"], ORG):
            s = np.sign(np.log(np.maximum(z["q"][:, HORIZONS.index(15), 10], 1e-9)) - lc[ORG])
            m = (s != 0) & (rev != 0)
            agree[tag] = round(float((s[m] == rev[m]).mean()), 4)

json.dump({"dataset": DATA, "rows": rows, "agree_h15_with_15min_reversal": agree},
          open(f"{RESULTS}/matched.json", "w"), indent=1)


def cell(v):
    if v is None:
        return "–"
    m, (lo, hi) = v
    s = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    return f"**{s}**" if lo > 0 or hi < 0 else s


print(f"\n#### T-J ({DATA}). Paired ablations on identical origins (variant vs base, % change; negative = variant better)\n")
print("| Variant | Base | n | Pinball h=1 / 5 / 15 | QLIKE h=1 / 5 / 15 |")
print("|---|---|---|---|---|")
for r in rows:
    pb = " / ".join(cell(r.get(f"pinball9_h{h}")) for h in HORIZONS)
    ql = " / ".join(cell(r.get(f"qlike_h{h}")) for h in HORIZONS)
    print(f"| {r['variant']} | {r['base']} | {r['n']} | {pb} | {ql} |")
