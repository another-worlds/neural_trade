import numpy as np, pandas as pd, json
a = pd.read_csv("D:/nt/nt_fwd_data/BTCUSDT_1m_2025-06.csv", parse_dates=["timestamp"]).set_index("timestamp")
parts = []
for ch in pd.read_csv("D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv", chunksize=500_000):
    ch = ch[(ch["timestamp"] >= "2025-06-01") & (ch["timestamp"] < "2025-07-01")]
    if len(ch): parts.append(ch)
b = pd.concat(parts); b["timestamp"] = pd.to_datetime(b["timestamp"]); b = b.set_index("timestamp")
d = a["close"] - b["close"]
ra, rb = np.log(a["close"]).diff(), np.log(b["close"]).diff()
out = {"close_diff_mean": d.mean(), "close_diff_std": d.std(), "close_diff_median": d.median(), "ret1m_corr": ra.corr(rb),
 "ret1h_corr": np.log(a["close"].resample("1h").last()).diff().corr(np.log(b["close"].resample("1h").last()).diff()),
 "volume_sum_dl": a["volume"].sum(), "volume_sum_main": b["volume"].sum(), "volume_corr": a["volume"].corr(b["volume"]),
 "main_zero_vol_frac": float((b["volume"] == 0).mean()), "main_integer_price_frac": float((b["close"] % 1 == 0).mean()), "dl_integer_price_frac": float((a["close"] % 1 == 0).mean())}
out = {k: float(v) for k, v in out.items()}
json.dump(out, open("source_check_2025-06_diag.json", "w"), indent=1); print(out)
