"""Download Binance USD-M BTCUSDT perp 1-min klines and funding rates (data.binance.vision), build one compact store.

  python fetch.py            -> $PNL_DATA/raw/*.zip and $PNL_DATA/btcusdt_perp_1m.npz, funding.npz
Monthly archives for 2020-01 .. last full month; daily archives for the current month. Funding for months without a
monthly archive comes from the public REST endpoint /fapi/v1/fundingRate.
"""
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

DATA = os.environ.get("PNL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "data"))
BASE = "https://data.binance.vision/data/futures/um"
SYMBOL = "BTCUSDT"
START = (2020, 1)
END = (2026, 10)            # inclusive


def get(url, path, tries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(path + ".part", path)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 ** (i + 1))
        except Exception:
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed: {url}")


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def read_klines(path):
    with zipfile.ZipFile(path) as z:
        rows = []
        for name in z.namelist():
            for row in csv.reader(io.TextIOWrapper(z.open(name))):
                if not row or not row[0].isdigit():
                    continue                       # header line in newer files
                rows.append(row[:11])
    return rows


def main():
    raw = os.path.join(DATA, "raw")
    os.makedirs(raw, exist_ok=True)
    kl, fr, missing_fr = [], [], []
    for y, m in months():
        tag = f"{y}-{m:02d}"
        p = os.path.join(raw, f"{SYMBOL}-1m-{tag}.zip")
        if get(f"{BASE}/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{tag}.zip", p):
            kl.append(p)
        else:                                        # current month: daily archives
            d = dt.date(y, m, 1)
            while d.month == m and d < dt.date.today():
                pd_ = os.path.join(raw, f"{SYMBOL}-1m-{d}.zip")
                if get(f"{BASE}/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{d}.zip", pd_):
                    kl.append(pd_)
                d += dt.timedelta(days=1)
        pf = os.path.join(raw, f"{SYMBOL}-fundingRate-{tag}.zip")
        if get(f"{BASE}/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{tag}.zip", pf):
            fr.append(pf)
        else:
            missing_fr.append((y, m))
        print("fetched", tag, flush=True)
    # klines -> arrays
    rows = []
    for p in kl:
        rows.extend(read_klines(p))
    a = np.array([[float(x) for x in r] for r in rows])
    a = a[np.argsort(a[:, 0], kind="stable")]
    _, keep = np.unique(a[:, 0], return_index=True)
    a = a[keep]
    t = (a[:, 0] // 1000).astype(np.int64)            # open time, seconds
    np.savez(os.path.join(DATA, "btcusdt_perp_1m.npz"), t=t, open=a[:, 1], high=a[:, 2], low=a[:, 3], close=a[:, 4],
             volume=a[:, 5].astype(np.float32), quote_volume=a[:, 7].astype(np.float32), trades=a[:, 8].astype(np.float32),
             taker_buy_base=a[:, 9].astype(np.float32), taker_buy_quote=a[:, 10].astype(np.float32))
    # funding -> arrays
    ft, fv = [], []
    for p in fr:
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                for row in csv.reader(io.TextIOWrapper(z.open(name))):
                    if row and row[0].isdigit():
                        ft.append(int(row[0]) // 1000)
                        fv.append(float(row[2]))
    for y, m in missing_fr:
        s = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        e = int(dt.datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={SYMBOL}&startTime={s}&endTime={e}&limit=1000"
        try:
            for x in json.load(urllib.request.urlopen(url, timeout=60)):
                ft.append(int(x["fundingTime"]) // 1000)
                fv.append(float(x["fundingRate"]))
        except Exception as ex:
            print(f"funding {y}-{m:02d} unavailable: {ex}", file=sys.stderr)
    o = np.argsort(ft)
    ft, fv = np.array(ft)[o], np.array(fv)[o]
    _, keep = np.unique(ft, return_index=True)
    np.savez(os.path.join(DATA, "funding.npz"), t=ft[keep], rate=fv[keep])
    gaps = np.diff(t)
    print(json.dumps({"bars": int(len(t)), "first": str(dt.datetime.utcfromtimestamp(t[0])),
                      "last": str(dt.datetime.utcfromtimestamp(t[-1])), "gaps_gt_60s": int((gaps > 60).sum()),
                      "max_gap_min": float(gaps.max() / 60), "funding_prints": int(len(ft[keep]))}, indent=1))


if __name__ == "__main__":
    main()
