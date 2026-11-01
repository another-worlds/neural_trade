"""Calibrate a direction-trading rule on horizon alignment and confidence; evaluate it out of sample.

Rule family (one 1-unit position at a time, hold H minutes, no overlap):
  H       trade horizon 1, 5 or 15 min; direction = sign of the model's median h=H forecast
  align   none | agree3 (median signs at h = 1, 5, 15 agree) | path15 (all 15 median steps share one sign)
  conf    z = median / robust sigma (IQR / 1.349) | pm = P(up) - 0.5 | dp = neural_trade direction head minus its
          calibration median. Confidence counts only when it agrees with the trade direction.
  keep k  trade only origins whose confidence ranks in the top k of the trailing 480 forecasts (a causal,
          label-free percentile, so the rule carries over when the confidence scale drifts); k = 1 means no filter
  side    follow | fade
Accuracy = share of trades with the right sign; zero-return trades excluded.
Selection on calibration data only, then a selection-adjusted gate:
  W   maximise the Wilson 95% lower bound of trade accuracy (>= 30 decided trades)
  R   maximise raw accuracy (>= 100 decided trades)
  The same search runs on 39 circular label shifts of the calibration data. The chosen rule is deployed only if its
  score beats the shifted searches (p_sel <= 0.05); otherwise the answer is 'abstain: do not trade'.
Protocols:
  P-A  calibrate on pre-Test forecasts, evaluate on the full Test (A0 zero-shot, A1, A2, price-only baseline)
  P-B  calibrate on the first half of Test (labels purged at the split), evaluate on the second half (all models,
       incl. neural_trade seed ensembles, which have no pre-Test forecasts)
Descriptive: 'test_picked' = rule R searched on the evaluation data itself, with a permutation p-value.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

import train
from data import QLEVELS

LV = np.array(QLEVELS)
HS = (1, 5, 15)
KEEP = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)
DEADBAND = 0.01          # bps; a median within this of the last price is no call
COSTS = (4.0, 10.0)
RANK_WINDOW, RANK_MIN = 480, 60
N_NULL = 39
MIN_TEST = 30


# ------------------------------------------------------------------------------------------------ signals
def p_up(q):
    """P(return > 0) from monotone quantiles (N, 21), as metrics2.p_up, vectorised."""
    q = np.maximum.accumulate(q, axis=-1)
    j = (q < 0).sum(-1)
    out = np.empty(len(q))
    lo, hi = j == 0, j == 21
    mid = ~(lo | hi)
    jj = j[mid]
    qa, qb = q[mid, jj - 1], q[mid, jj]
    w = np.where(qb > qa, (0 - qa) / np.where(qb > qa, qb - qa, 1), 0.5)
    out[mid] = 1 - (LV[jj - 1] + w * (LV[jj] - LV[jj - 1]))
    out[lo], out[hi] = 1.0, 0.0
    return np.clip(out, 0.01, 0.99)


def rank_pct(c):
    """Causal percentile of each confidence within the trailing RANK_WINDOW origins (NaN = no call)."""
    return pd.Series(c).rolling(RANK_WINDOW, min_periods=RANK_MIN).rank(pct=True).to_numpy()


def finish(s):
    """Agreement-signed confidence -> causal rank per (conf, H); non-agreeing or no-call origins get NaN."""
    s["rank"] = {}
    for cname, per_h in s["conf"].items():
        s["rank"][cname] = {}
        for h in HS:
            a = s["dir"][h] * per_h[h]
            s["rank"][cname][h] = rank_pct(np.where((s["dir"][h] != 0) & (a > 0), a, np.nan))
    return s


def model_signals(q, dir_prob=None, dp_center=None):
    """q: (N, n_steps, 21) log-bps quantiles; n_steps 15 (steps 1..15) or 3 (h = 1, 5, 15)."""
    idx = {1: 0, 5: 4, 15: 14} if q.shape[1] == 15 else {1: 0, 5: 1, 15: 2}
    med = {h: np.where(np.abs(q[:, i, 10]) > DEADBAND, q[:, i, 10], 0.0) for h, i in idx.items()}
    sig = {h: np.maximum((q[:, i, 15] - q[:, i, 5]) / 1.349, 1e-6) for h, i in idx.items()}
    s = {"dir": {h: np.sign(med[h]) for h in HS},
         "conf": {"z": {h: med[h] / sig[h] for h in HS},
                  "pm": {h: p_up(q[:, idx[h]]) - 0.5 for h in HS}}}
    d = s["dir"]
    s["align"] = {"none": np.ones(len(q), bool), "agree3": (d[1] != 0) & (d[1] == d[5]) & (d[5] == d[15])}
    if q.shape[1] == 15:
        sg = np.sign(np.where(np.abs(q[..., 10]) > DEADBAND, q[..., 10], 0.0))
        s["align"]["path15"] = (sg[:, 0] != 0) & (sg == sg[:, :1]).all(1)
    if dir_prob is not None:
        s["conf"]["dp"] = {h: dir_prob[:, i] - dp_center[i] for i, h in enumerate(HS)}
    return finish(s)


def price_signals(lc, org, sigma):
    """Price-only baseline: direction = sign of the last H-minute return; conf = that return / (sigma sqrt H)."""
    r = {h: (lc[org] - lc[org - h]) * 1e4 for h in HS}
    d = {h: np.sign(np.where(np.abs(r[h]) > DEADBAND, r[h], 0.0)) for h in HS}
    return finish({"dir": d, "conf": {"z": {h: r[h] / (sigma[org] * np.sqrt(h)) for h in HS}},
                   "align": {"none": np.ones(len(org), bool), "agree3": (d[1] != 0) & (d[1] == d[5]) & (d[5] == d[15])}})


def sub(sig, m):
    return {"dir": {h: v[m] for h, v in sig["dir"].items()},
            "conf": {c: {h: v[m] for h, v in d.items()} for c, d in sig["conf"].items()},
            "rank": {c: {h: v[m] for h, v in d.items()} for c, d in sig["rank"].items()},
            "align": {a: v[m] for a, v in sig["align"].items()}}


# ------------------------------------------------------------------------------------------------ rules
def trade_index(sig, org, H, align, conf, keep, side):
    """Greedy non-overlapping entries (positions, signed directions). Depends on signals only, not on labels."""
    d = sig["dir"][H] * (1 if side == "follow" else -1)
    ok = sig["align"][align] & (d != 0)
    if keep < 1:
        r = sig["rank"][conf][H]
        ok &= np.nan_to_num(r, nan=-1.0) >= 1 - keep
    pos, nxt = [], -1
    for i in np.flatnonzero(ok):
        if org[i] >= nxt:
            pos.append(i)
            nxt = org[i] + H
    pos = np.array(pos, dtype=int)
    return pos, d[pos]


def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def score(pnl, rule):
    dec = pnl[pnl != 0]
    n, k = len(dec), int((dec > 0).sum())
    if rule == "W":
        return (wilson_lb(k, n), n) if n >= 30 else None
    return (k / n, n) if n >= 100 else None


def rules(sig):
    for H in HS:
        for align in sig["align"]:
            for conf in sig["conf"]:
                for side in ("follow", "fade"):
                    for k in KEEP:
                        if k == 1 and conf != list(sig["conf"])[0]:
                            continue           # k = 1 ignores confidence: count it once
                        yield H, align, conf, k, side


def search(sig, org, y, rule, seed=0):
    """Best rule on (sig, org, y) and a selection-adjusted p from N_NULL circular label shifts."""
    rng = np.random.default_rng(seed)
    n = len(org)
    shifts = rng.integers(n // 10, n - n // 10, N_NULL)
    y_null = {h: np.stack([np.roll(y[h], s) for s in shifts]) for h in HS}
    best, null_best = None, np.full(N_NULL, -np.inf)
    for H, align, conf, k, side in rules(sig):
        pos, d = trade_index(sig, org, H, align, conf, k, side)
        if len(pos) < 30:
            continue
        sc = score(d * y[H][pos], rule)
        if sc is not None and (best is None or sc > best[0]):
            best = (sc, dict(H=H, align=align, conf=conf, keep=k, side=side, calib_decided=sc[1],
                             calib_accuracy=None))
        pn = d[None, :] * y_null[H][:, pos]
        for j in range(N_NULL):
            s2 = score(pn[j], rule)
            if s2 is not None and s2[0] > null_best[j]:
                null_best[j] = s2[0]
    if best is None:
        return None
    p = best[1]
    pos, d = trade_index(sig, org, p["H"], p["align"], p["conf"], p["keep"], p["side"])
    dec = (d * y[p["H"]][pos])
    dec = dec[dec != 0]
    p["calib_accuracy"] = round(float((dec > 0).mean()), 4)
    p["calib_score"] = round(float(best[0][0]), 4)
    p["p_sel"] = round(float((1 + (null_best >= best[0][0]).sum()) / (1 + N_NULL)), 4)
    p["null_max_median"] = round(float(np.median(null_best)), 4)
    return p


def stats(pnl, days):
    dec = pnl[pnl != 0]
    n, k = len(dec), int((dec > 0).sum())
    out = {"trades": int(len(pnl)), "decided": int(n), "per_day": round(len(pnl) / days, 1)}
    if n == 0:
        return out
    out.update(accuracy=round(k / n, 4), acc_ci95=[round(wilson_lb(k, n), 4), round(1 - wilson_lb(n - k, n), 4)],
               p_one_sided=float(binomtest(k, n, 0.5, alternative="greater").pvalue),
               pnl_bps=round(float(pnl.sum()), 1), mean_bps=round(float(pnl.mean()), 3),
               t=round(float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))), 2) if len(pnl) > 1 else None,
               insufficient=bool(n < MIN_TEST))
    for c in COSTS:
        out[f"net_bps_cost{c:g}"] = round(float((pnl - c).sum()), 1)
    return out


def targets(lc, org):
    return {h: (lc[org + h] - lc[org]) * 1e4 for h in HS}


def apply(sig, org, y, p, days):
    pos, d = trade_index(sig, org, p["H"], p["align"], p["conf"], p["keep"], p["side"])
    return stats(d * y[p["H"]][pos], days)


def evaluate_model(sig_cal, org_cal, y_cal, sig_te, org_te, y_te, days):
    res = {"median_call_share": {h: round(float((sig_te["dir"][h] != 0).mean()), 4) for h in HS}}
    for rule in ("W", "R"):
        p = search(sig_cal, org_cal, y_cal, rule)
        if p is None:
            res[rule] = {"params": None, "decision": "abstain: no rule met the trade minimum"}
        elif p["p_sel"] > 0.05:
            res[rule] = {"params": p, "decision": "abstain: calibration edge not above chance (p_sel > 0.05)",
                         "test_if_deployed": apply(sig_te, org_te, y_te, p, days)}
        else:
            res[rule] = {"params": p, "decision": "deploy", "test": apply(sig_te, org_te, y_te, p, days)}
    p = search(sig_te, org_te, y_te, "R", seed=1)
    if p is not None:
        res["test_picked"] = {"params": p, "test": apply(sig_te, org_te, y_te, p, days)}
    res["unfiltered"] = {H: apply(sig_te, org_te, y_te, dict(H=H, align="none", conf=list(sig_te["conf"])[0],
                                                             keep=1.0, side="follow"), days) for H in HS}
    return res


def holm(ps):
    """Holm-adjusted p-values for a dict name -> p."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    return out


def apply_holm(res):
    """Holm within each protocol family over distinct deployed rules (both sets, both selection rules).
    W and R picking the same rule on the same model and set count once."""
    for proto in ("P-A", "P-B"):
        keys, ps = {}, {}
        for ds, d in res.items():
            if not isinstance(d, dict) or proto not in d:
                continue
            for name, r in d[proto].items():
                for rule in ("W", "R"):
                    x = r[rule]
                    if x.get("decision") != "deploy" or "p_one_sided" not in x["test"]:
                        continue
                    p = x["params"]
                    k = "|".join(str(v) for v in (ds, name, p["H"], p["align"], p["conf"], p["keep"], p["side"]))
                    keys.setdefault(k, []).append((ds, name, rule))
                    ps[k] = x["test"]["p_one_sided"]
        for k, adj in holm(ps).items():
            for ds, name, rule in keys[k]:
                res[ds][proto][name][rule]["test"]["p_holm"] = adj


# ------------------------------------------------------------------------------------------------ data
def lig_predictions(X, rung, org):
    """Seed-ensemble log-bps quantiles (N, 15, 21) from the pre-registered checkpoints (hashes verified)."""
    cj = json.load(open(train.wdir(X.D.dataset, "candidate.json")))
    ts = train.load_ts()
    qs = []
    for name, h in cj["checkpoints"].items():
        r, seed = name.rsplit("_s", 1)
        if r != rung:
            continue
        path = train.wdir(X.D.dataset, "ckpt", f"{name}.pt")
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == h
        m = train.make_model(X, r, int(seed), ts)
        train.load_state(m, torch.load(path)["state"])
        qs.append(train.predict(X, m, org))
    return np.mean(qs, 0)


def main(ds):
    X = train.Ctx(ds)
    D = X.D
    lc, sigma = D.lc, D.sigma
    te = D.test
    cal = np.unique(np.concatenate([D.train[::2], D.v1, D.v2]))
    v12 = np.unique(np.concatenate([D.v1, D.v2]))
    days_te = (te[-1] - te[0]) / 1440
    out = {"dataset": ds, "test_days": round(days_te, 2), "n_test": int(len(te)),
           "calibration": {"pretest_all": [int(cal[0]), int(cal[-1]), int(len(cal))], "v1v2": [int(v12[0]), int(v12[-1]), int(len(v12))]},
           "P-A": {}, "P-B": {}}
    q0_te = X.q0(te)
    A = {"A0 zero-shot": ((model_signals(X.q0(cal)), cal), model_signals(q0_te)),
         "A0 zero-shot (V1+V2 calib)": ((model_signals(X.q0(v12)), v12), model_signals(q0_te))}
    lig_te = {}
    for rung in ("A1", "A2"):
        lig_te[rung] = lig_predictions(X, rung, te)
        A[f"{rung} (calib partly in-sample)"] = ((model_signals(lig_predictions(X, rung, v12)), v12), model_signals(lig_te[rung]))
    A["price-only baseline"] = ((price_signals(lc, cal, sigma), cal), price_signals(lc, te, sigma))
    for name, ((sc, oc), st) in A.items():
        out["P-A"][name] = evaluate_model(sc, oc, targets(lc, oc), st, te, targets(lc, te), days_te)
        print(f"P-A {ds} {name}: W {out['P-A'][name]['W'].get('decision')}", flush=True)
    # P-B: first half of Test calibrates (labels purged at the split), second half evaluates
    half = te[len(te) // 2]
    m1, m2 = te + max(HS) < half, te >= half
    days2 = (te[m2][-1] - te[m2][0]) / 1440
    B = {"A0 zero-shot": model_signals(q0_te), "A1": model_signals(lig_te["A1"]), "A2": model_signals(lig_te["A2"]),
         "price-only baseline": price_signals(lc, te, sigma)}
    for kind in ("asis", "rolling1440m"):
        qs, dps = [], []
        for s in range(3):
            z = np.load(os.path.join(D.bench_results, f"neural_trade-{kind}-seed{s}_ctx60_price.npz"), allow_pickle=True)
            assert np.array_equal(z["origins"], te)
            qs.append((np.log(np.maximum(z["q"].astype(np.float64), 1e-9)) - lc[te][:, None, None]) * 1e4)
            dps.append(z["dir_prob"].astype(np.float64))
        q, dp = np.mean(qs, 0), np.mean(dps, 0)
        B[f"neural_trade {kind} (3-seed ensemble)"] = model_signals(q, dp, np.median(dp[m1], axis=0))
    for name, s_all in B.items():
        out["P-B"][name] = evaluate_model(sub(s_all, m1), te[m1], targets(lc, te[m1]),
                                          sub(s_all, m2), te[m2], targets(lc, te[m2]), days2)
        print(f"P-B {ds} {name}: W {out['P-B'][name]['W'].get('decision')}", flush=True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    res = {ds: main(ds) for ds in (sys.argv[1:] or ["repo", "post"])}
    apply_holm(res)
    src = open(__file__, "rb").read()
    for ds in res:
        res[ds]["code_sha256"] = hashlib.sha256(src).hexdigest()
    json.dump(res, open(os.path.join(train.WORK, "calibration.json"), "w"), indent=1)
