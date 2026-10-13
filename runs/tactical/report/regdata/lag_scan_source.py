import json, urllib.request, pandas as pd, numpy as np, time
df = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", parse_dates=["timestamp"]).set_index("timestamp")
print("rows", len(df), "dup ts", df.index.duplicated().sum(), "monotonic", df.index.is_monotonic_increasing)
full = pd.date_range(df.index[0], df.index[-1], freq="min")
print("expected minutes", len(full), "missing", len(full.difference(df.index)))
oc = df["close"].reindex(full).ffill()
orr = np.log(oc).diff()

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def binance(t0, n=1000):
    ms = int(pd.Timestamp(t0, tz="UTC").timestamp() * 1000)
    k = get(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={ms}&limit={n}")
    return pd.Series([float(x[4]) for x in k], index=pd.to_datetime([x[0] for x in k], unit="ms"))

rows = []
for t0 in pd.date_range("2017-09-15", "2025-09-15", freq="MS") + pd.Timedelta(days=14, hours=6):
    try:
        x = binance(t0)
    except Exception as e:
        print(t0, "ERR", e); continue
    rx = np.log(x).diff().dropna()
    best = (-1, None)
    for lag in range(-420, 421):
        y = orr.reindex(rx.index + pd.Timedelta(minutes=lag)).values
        m = np.isfinite(y)
        if m.sum() < 500: continue
        c = np.corrcoef(rx.values[m], y[m])[0, 1]
        if c > best[0]: best = (c, lag)
    c, lag = best
    y = orr.reindex(rx.index + pd.Timedelta(minutes=lag))
    ac_b = rx.autocorr(1); ac_o = pd.Series(y.values).autocorr(1)
    lvl = np.nanmedian(oc.reindex(x.index + pd.Timedelta(minutes=lag)).values - x.values)
    sd_b = rx.std() * 1e4; sd_o = np.nanstd(y.values) * 1e4
    rows.append((t0.date(), lag, round(c, 3), round(ac_b, 3), round(ac_o, 3), round(sd_b, 2), round(sd_o, 2), round(lvl, 2)))
    print(rows[-1], flush=True)
    time.sleep(0.2)
pd.DataFrame(rows, columns=["date", "lag_min", "corr", "ac1_binance", "ac1_file", "sd1m_bin_bps", "sd1m_file_bps", "lvl_diff"]).to_csv("scan.csv", index=False)
