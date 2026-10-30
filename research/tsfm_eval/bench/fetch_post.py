"""Fetch BTCUSDT 1-min klines 2026-10-08..2026-10-30 UTC from Binance's public market-data mirror."""
import csv
import datetime as dt
import json
import time
import os
import urllib.request

OUT = os.path.join(os.environ.get("TSFM_EVAL_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")), "data", "btcusdt_1m_2026-09-20_2026-10-30.csv")
start = int(dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc).timestamp() * 1000)
end = int(dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
rows, t = [], start
while t < end:
    url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={t}&limit=1000"
    for attempt in range(4):
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=30).read())
            break
        except Exception:
            time.sleep(2 ** attempt)
    if not data:
        break
    rows += [r for r in data if r[0] < end]
    t = data[-1][0] + 60_000
    time.sleep(0.2)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "open", "high", "low", "close", "volume"])
    for r in rows:
        ts = dt.datetime.fromtimestamp(r[0] / 1000, tz=dt.timezone.utc)
        w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S+00:00"), r[1], r[2], r[3], r[4], r[5]])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(len(rows), "rows ->", OUT)
