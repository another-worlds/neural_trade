import json, numpy as np, pandas as pd
d = pd.read_csv("D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv", parse_dates=["timestamp"])
log = json.load(open("D:/nt/nt_fwd_data/download_log.json"))
ts = d["timestamp"]; full = pd.date_range(ts.min(), ts.max(), freq="min")
z = (d["volume"] == 0).values; runs = []; i = 0
while i < len(z):
    if z[i]:
        j = i
        while j + 1 < len(z) and z[j + 1]: j += 1
        runs.append((str(ts[i]), j - i + 1)); i = j + 1
    else: i += 1
out = {"rows": len(d), "first": str(ts.min()), "last": str(ts.max()), "expected_minutes": len(full), "missing_minutes": int(len(full.difference(pd.DatetimeIndex(ts)))),
 "duplicate_timestamps": int(ts.duplicated().sum()), "nan_cells": int(d.isna().sum().sum()), "zero_volume_minutes": int(z.sum()),
 "zero_volume_runs": len(runs), "longest_zero_volume_runs": sorted(runs, key=lambda r: -r[1])[:5], "ohlc_inconsistent": int(((d.high < d[["open","close","low"]].max(axis=1)) | (d.low > d[["open","close","high"]].min(axis=1))).sum()),
 "monthly_archives": sum(1 for r in log if r["kind"] == "monthly" and r.get("checksum_ok")), "daily_archives": sum(1 for r in log if r["kind"] == "daily" and r.get("checksum_ok")),
 "archives_404": [r["d"] for r in log if r.get("status") == "404"], "checksum_failures": sum(1 for r in log if r.get("checksum_ok") is False), "units": sorted({r.get("unit") for r in log if r.get("unit")}),
 "min_ts_in_protected_span": int(((ts >= "2025-07-21") & (ts < "2025-10-01")).sum()), "file_sha256": __import__("hashlib").sha256(open("D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv","rb").read()).hexdigest()}
json.dump(out, open("data_quality.json", "w"), indent=1); print(json.dumps(out, indent=1))
