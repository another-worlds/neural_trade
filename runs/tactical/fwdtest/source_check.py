"""Compare the downloaded 2025-06 month with the cut copy of the main file (rows of 2025-06 only)."""
import json
import numpy as np, pandas as pd
a = pd.read_csv("D:/nt/nt_fwd_data/BTCUSDT_1m_2025-06.csv", parse_dates=["timestamp"]).set_index("timestamp")
parts = []
for ch in pd.read_csv("D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv", chunksize=500_000):
    ch = ch[(ch["timestamp"] >= "2025-06-01") & (ch["timestamp"] < "2025-07-01")]
    if len(ch):
        parts.append(ch)
b = pd.concat(parts); b["timestamp"] = pd.to_datetime(b["timestamp"]); b = b.set_index("timestamp")
j = a.join(b, how="outer", lsuffix="_dl", rsuffix="_main")
res = {"rows_download": len(a), "rows_main": len(b), "rows_union": len(j), "missing_in_main": int(j["open_main"].isna().sum()), "missing_in_download": int(j["open_dl"].isna().sum())}
both = j.dropna()
for c in ["open", "high", "low", "close", "volume"]:
    d = (both[c + "_dl"] - both[c + "_main"]).abs()
    res[c] = {"max_abs_diff": float(d.max()), "match_rate_exact": float((d == 0).mean()), "match_rate_1e-8": float((d <= 1e-8).mean())}
res["all_ohlcv_exact_rate"] = float(np.all([(both[c + "_dl"] == both[c + "_main"]) for c in ["open", "high", "low", "close", "volume"]], axis=0).mean())
json.dump(res, open("D:/nt/nt_fwdtest/runs/tactical/fwdtest/source_check_2025-06.json", "w"), indent=1)
print(json.dumps(res, indent=1))
