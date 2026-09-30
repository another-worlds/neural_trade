"""Applied (per-window) periods vs the logged global period, on each run's dev (oos) block."""
import glob, os, json
import numpy as np, pandas as pd, h5py, joblib

BASE = "D:/neural_trade/runs/scenarios/long_360d_stab"
csv = pd.read_csv("D:/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "close"])
csv["timestamp"] = pd.to_datetime(csv["timestamp"])
close = csv["close"].to_numpy(np.float64)
ts_index = pd.Index(csv["timestamp"])
NAMES = {"ma_period_%d": "alpha_ma_%d", "rsi_period_%d": "rsi_alpha_%d", "bb_period_%d": "bb_alpha_%d"}
order = ["ma_period_0", "ma_period_1", "ma_period_2"] + [f"macd_{i}_{p}" for i in range(3) for p in ("fast", "slow", "signal")] \
    + ["rsi_period_0", "rsi_period_1", "rsi_period_2", "bb_period_0", "bb_period_1", "bb_period_2"]

def wname(k):
    if k.startswith("ma_period"):
        return "alpha_ma_" + k[-1]
    if k.startswith("rsi_period"):
        return "rsi_alpha_" + k[-1]
    if k.startswith("bb_period"):
        return "bb_alpha_" + k[-1]
    return k

sig = lambda v: 1 / (1 + np.exp(-v))
res = {}
for d in sorted(glob.glob(BASE + "/2026*")):
    tag = d.split("default__")[1]
    f = h5py.File(d + "/weights.h5", "r")
    W = f["dense/dense/kernel:0"][()]; b = f["dense/dense/bias:0"][()]
    logits = np.array([f[f"learnable_indicators/learnable_indicators/{wname(k)}:0"][()] for k in order])
    scale = float(joblib.load(d + "/scaler.joblib").scale_[0])
    z = np.load(d + "/predictions_oos.npz")
    anchors = pd.to_datetime(z["extra__anchor_timestamp"])
    pos = ts_index.get_indexer(anchors)
    assert (pos >= 59).all()
    sel = pos[::5]
    win = np.stack([close[p - 59:p + 1] for p in sel])
    lc = win[:, -1:]
    assert np.allclose(lc[:, 0], z["last_close"][::5]), "last close mismatch"
    xn = (win - lc) / scale
    feat = np.stack([xn.mean(1), xn.max(1)], 1)
    shift = 0.5 * np.tanh(feat @ W + b)          # [N, 18]
    alpha = sig(logits[None, :] + shift)
    p_applied = 2 / alpha - 1
    p_global = 2 / sig(logits) - 1
    res[tag] = dict(p_global=p_global, med=np.median(p_applied, 0), q05=np.quantile(p_applied, 0.05, 0),
                    q95=np.quantile(p_applied, 0.95, 0), shift_mean=shift.mean(0), bias_shift=0.5 * np.tanh(b))
    print(tag, "scale", round(scale, 2), "n", len(sel))
rows = []
for i, k in enumerate(order):
    r = {"param": k}
    for tag, v in res.items():
        r[tag] = f"{v['p_global'][i]:.1f}|{v['med'][i]:.1f} [{v['q05'][i]:.1f},{v['q95'][i]:.1f}]"
    r["mean_shift"] = np.mean([v["shift_mean"][i] for v in res.values()])
    r["med_applied_mean"] = np.mean([v["med"][i] for v in res.values()])
    r["med_applied_sd"] = np.std([v["med"][i] for v in res.values()], ddof=1)
    r["global_mean"] = np.mean([v["p_global"][i] for v in res.values()])
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
print(df.round(3).to_string(index=False))
df.to_csv("D:/nt_math_scratch/applied.csv", index=False)
