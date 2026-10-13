import numpy as np, pandas as pd
a = pd.read_csv("D:/nt/nt_fwd_data/BTCUSDT_1m_2025-06.csv", parse_dates=["timestamp"]).set_index("timestamp")
parts = []
for ch in pd.read_csv("D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv", chunksize=500_000):
    ch = ch[(ch["timestamp"] >= "2025-05-31") & (ch["timestamp"] < "2025-07-02")]
    if len(ch): parts.append(ch)
b = pd.concat(parts); b["timestamp"] = pd.to_datetime(b["timestamp"]); b = b.set_index("timestamp")
print(a.head(3)); print(b.loc["2025-06-01 00:00":].head(3))
for lag in range(-600, 601, 1):
    bb = b["close"].copy(); bb.index = bb.index + pd.Timedelta(minutes=lag)
    j = pd.concat([a["close"], bb], axis=1, join="inner")
    m = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
    if (m < 0.011).mean() > 0.3:
        print("lag", lag, "n", len(j), "exact", float((m < 0.011).mean()), "max", float(m.max()))
# correlation at lag 0
j = pd.concat([a["close"], b["close"]], axis=1, join="inner"); print(j.corr().iloc[0, 1])
