"""Direction accuracy and zero-commission PnL of the median forecasts on both Test sets.

Rules (as bench/metrics2.py, with cost 0):
- signal = sign of the median forecast of the h-minute log return (bps). |median| <= 0.01 bps counts as no call:
  the npz files store prices as float32, so a median shrunk exactly to the last price reads as +-0.0004 bps noise.
- accuracy = share of correct signs over all Test origins (every 3 min) with a call and a nonzero return.
- PnL = one position of 1 unit at a time: enter at an origin with a call, hold h minutes, then look again
  (non-overlapping). PnL is the sum of signed log returns in bps; no commission, no slippage, no funding.
Baselines: always long; momentum and reversal on the last h-minute return.
Run from this folder with the benchmark venv; reads ../tsfm_eval/work/results* and work results_lig*.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tsfm_eval", "bench"))
DEADBAND = 0.01
HORIZONS = (1, 5, 15)


def trades(point, y, org, h):
    pnl, nxt = [], -1
    for i, t in enumerate(org):
        if t < nxt or abs(point[i]) <= DEADBAND:
            continue
        pnl.append(np.sign(point[i]) * y[i])
        nxt = t + h
    pnl = np.array(pnl)
    if len(pnl) < 2:
        return {"n": int(len(pnl)), "total_bps": float(pnl.sum()) if len(pnl) else 0.0}
    eq = np.cumsum(pnl)
    return {"n": int(len(pnl)), "total_bps": round(float(pnl.sum()), 1), "mean_bps": round(float(pnl.mean()), 3),
            "t_stat": round(float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))), 2),
            "hit": round(float((pnl > 0).mean()), 4),
            "max_dd_bps": round(float((np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])).max()), 1)}


def evaluate(point, y, org, h):
    call = np.where(np.abs(point) > DEADBAND, np.sign(point), 0.0)
    m = (call != 0) & (y != 0)
    out = {"coverage": round(float((call != 0).mean()), 4),
           "accuracy": round(float((call[m] == np.sign(y[m])).mean()), 4) if m.any() else None,
           "n_calls": int(m.sum()), "up_share": round(float((call[m] > 0).mean()), 4) if m.any() else None}
    out.update(trades(point, y, org, h))
    return out


def main(ds, results, lig):
    os.environ["BENCH_DATA"] = ds
    import importlib
    import common
    importlib.reload(common)
    df, _ = common.load()
    lc = np.log(df["close"].to_numpy(np.float64))
    org = common.origins(len(lc))
    models = {"A0 zero-shot Timestral-28M": f"{results}/timestral-28M_ctx2048_price.npz",
              "A1 recalibration (candidate)": f"{lig}/lig-28M_ctx2048_A1_ens.npz",
              "A2 indicators + geometry": f"{lig}/lig-28M_ctx2048_A2_ens.npz",
              "C-FHS-hi (no Timestral)": f"{lig}/lig-28M_ctx2048_C-FHS-hi_ens.npz"}
    for kind in ("asis", "rolling1440m"):
        for s in range(3):
            models[f"neural_trade {kind} seed{s}"] = f"{results}/neural_trade-{kind}-seed{s}_ctx60_price.npz"
    out = {"dataset": ds, "n_origins": int(len(org)), "days": round(float((org[-1] - org[0]) / 1440), 2),
           "market_bps": round(float((lc[org[-1] + 15] - lc[org[0]]) * 1e4), 1), "models": {}}
    for name, path in models.items():
        z = np.load(path, allow_pickle=True)
        o = z["origins"]
        assert np.array_equal(o, org), name
        q = z["q"].astype(np.float64)
        res = {}
        for hi, h in enumerate(HORIZONS):
            y = (lc[org + h] - lc[org]) * 1e4
            point = (np.log(np.maximum(q[:, hi, 10], 1e-9)) - lc[org]) * 1e4
            res[h] = evaluate(point, y, org, h)
            if "dir_prob" in z.files:
                dp = z["dir_prob"][:, hi].astype(np.float64)
                m = y != 0
                res[h]["dirhead_accuracy"] = round(float(((dp[m] > 0.5) == (y[m] > 0)).mean()), 4)
                res[h]["dirhead_pnl"] = trades(np.where(dp > 0.5, 1.0, -1.0), y, org, h)
        out["models"][name] = res
    for name, fn in (("always long", lambda h: np.ones(len(org))),
                     ("momentum (last h min)", lambda h: (lc[org] - lc[org - h]) * 1e4),
                     ("reversal (last h min)", lambda h: -(lc[org] - lc[org - h]) * 1e4)):
        out["models"][name] = {h: evaluate(fn(h), (lc[org + h] - lc[org]) * 1e4, org, h) for h in HORIZONS}
    return out


if __name__ == "__main__":
    work = os.environ.get("TSFM_EVAL_WORK", os.path.join(HERE, "..", "tsfm_eval", "work"))
    res = {}
    for ds in ("repo", "post"):
        sfx = "" if ds == "repo" else f"_{ds}"
        res[ds] = main(ds, f"{work}/results{sfx}", os.environ.get("LIG_RESULTS_DIR", f"{work}/results_lig") + sfx)
    json.dump(res, open(os.path.join(HERE, "results", "trading_nocost.json"), "w"), indent=1)
    for ds, r in res.items():
        print(f"\n### {ds}: {r['n_origins']} origins, {r['days']} days, market move {r['market_bps']:+.0f} bps")
        print("| Model | h | accuracy | coverage | trades | total PnL bps | mean bps | t | max DD bps |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, rr in r["models"].items():
            for h in HORIZONS:
                v = rr[h]
                print(f"| {name} | {h} | {v['accuracy']} | {v['coverage']} | {v['n']} | {v.get('total_bps')} | "
                      f"{v.get('mean_bps')} | {v.get('t_stat')} | {v.get('max_dd_bps')} |")
