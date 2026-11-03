"""New-data backtest, step 1: hourly bars and funding for markets and periods the PnL study never scored.

  python nd_fetch.py      -> $ND_DATA (default work/nd)/universe.json, hourly/<SYMBOL>.npz

Sources (data.binance.vision only; the fapi REST endpoints are geo-blocked from this machine):
- universe: every USD-M USDT-margined perpetual in the archive listing (delisted symbols included, so the panel has no
  survivorship filter) whose first 1d bar is on or before 2020-12-31. BTCUSDT is kept apart (the study's own market).
- USD-M perps: 1h klines (monthly archives, daily archives for the month in progress) and funding prints (monthly).
  Hourly bars let the engine trim exposure inside the day (engine E trims at every 1-min close).
- BTC before the study: spot BTCUSDT 1h klines 2017-08 .. 2020-12 (no funding: a price-path test only).
- The monthly 1h archives leave out some days that the daily archives contain (e.g. 2022-02-26..28 and 2022-04-01..02
  for 30 panel markets). fill_days() fetches every day with fewer than 24 bars between a market's first and last day
  from the daily archive and merges it. A day missing from both archives (e.g. 2026-10-28) stays missing.
- manifest.json: SHA-256 of universe.json and of every hourly file; nd_run.py score checks the data against the
  manifest committed with the pre-registration (results/newdata_manifest.json).
Nothing here reads a price to choose anything: the universe rule uses listing dates only.
"""
import concurrent.futures as cf
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ND_DATA", os.path.join(HERE, "work", "nd"))
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/data"
LISTED_BY = (2020, 12)          # first 1d bar on or before 2020-12-31
LAST_MONTH = (2026, 10)         # last complete monthly archive
TODAY = dt.date(2026, 11, 3)    # daily archives exist up to the day before


def http(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** (i + 1))
        except Exception:
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed: {url}")


def s3_list(prefix, delimiter=True):
    """All CommonPrefixes (delimiter) or Keys under a prefix, following pagination."""
    out, marker = [], ""
    while True:
        enc = lambda v: urllib.parse.quote(v, safe="/")       # some newer symbols are not ASCII
        q = f"{S3}?prefix={enc(prefix)}" + ("&delimiter=/" if delimiter else "") + (f"&marker={enc(marker)}" if marker else "")
        x = http(q).decode()
        tag = "Prefix" if delimiter else "Key"
        found = re.findall(rf"<{tag}>([^<]+)</{tag}>", x)
        found = [f for f in found if f != prefix]
        out += found
        if "<IsTruncated>true</IsTruncated>" not in x:
            return out
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = nm[0] if nm else found[-1]


def months(a, b):
    y, m = a
    while (y, m) <= b:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def zip_rows(blob):
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            for row in csv.reader(io.TextIOWrapper(z.open(name))):
                if row and row[0].strip().lstrip("-").isdigit():
                    rows.append(row)
    return rows


def klines(market, sym, first):
    """1h klines from `first` (y, m) to today: monthly archives, then daily archives for the current month."""
    root = f"{BASE}/futures/um" if market == "um" else f"{BASE}/spot"
    last = LAST_MONTH if market == "um" else (2020, 12)
    urls = [f"{root}/monthly/klines/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip" for y, m in months(first, last)]
    d = dt.date(LAST_MONTH[0] + (LAST_MONTH[1] == 12), LAST_MONTH[1] % 12 + 1, 1)
    while d < TODAY and market == "um":
        urls.append(f"{root}/daily/klines/{sym}/1h/{sym}-1h-{d}.zip")
        d += dt.timedelta(days=1)
    rows = []
    for u in urls:
        blob = http(u)
        if blob:
            rows += zip_rows(blob)
    return to_arrays(rows)


def to_arrays(rows):
    if not rows:
        return None
    a = np.array([[float(x) for x in r[:11]] for r in rows])
    t = a[:, 0]
    t = np.where(t > 1e14, t / 1e3, t)                     # some newer spot files are in microseconds
    t = (t // 1000).astype(np.int64)
    o = np.argsort(t, kind="stable")
    a, t = a[o], t[o]
    _, keep = np.unique(t, return_index=True)
    return {"t": t[keep], "open": a[keep, 1], "high": a[keep, 2], "low": a[keep, 3], "close": a[keep, 4],
            "volume": a[keep, 5], "quote_volume": a[keep, 7]}


def fill_days(sym, market="um"):
    """Fetch every UTC day with fewer than 24 bars (between the first and the last full day of the file) from the
    daily archive and merge it into the file. Returns the days that gained bars."""
    path = os.path.join(OUT, "hourly", f"{sym}{'' if market == 'um' else '_spot'}.npz")
    z = dict(np.load(path))
    if "filled_days" in z:
        return sym, "filled before"
    root = f"{BASE}/futures/um" if market == "um" else f"{BASE}/spot"
    day = z["t"] // 86400
    d0, d1 = int(day[0]) + 1, int(day[-1]) - 1
    cnt = np.bincount(day - day[0], minlength=int(day[-1] - day[0] + 1))
    short = [d for d in range(d0, d1 + 1) if cnt[d - int(day[0])] < 24]
    rows, gained = [], []
    for d in short:
        ds = dt.date(1970, 1, 1) + dt.timedelta(days=d)
        blob = http(f"{root}/daily/klines/{sym}/1h/{sym}-1h-{ds}.zip")
        if blob:
            r = zip_rows(blob)
            if r:
                rows += r
                gained.append(str(ds))
    if rows:
        k = to_arrays(rows)
        t = np.r_[z["t"], k["t"]]
        o = np.argsort(t, kind="stable")
        _, keep = np.unique(t[o], return_index=True)      # the monthly archive's bar wins on a duplicate
        idx = o[keep]
        for f in ("open", "high", "low", "close", "volume", "quote_volume"):
            z[f] = np.r_[z[f], k[f]][idx]
        z["t"] = t[idx]
    z["filled_days"] = np.array(gained, dtype="U10")
    np.savez(path, **z)
    return sym, f"{len(short)} short days, {len(gained)} filled from the daily archive"


def manifest():
    import hashlib
    files = sorted(os.listdir(os.path.join(OUT, "hourly")))
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    m = {"universe.json": h(os.path.join(OUT, "universe.json")),
         **{f"hourly/{f}": h(os.path.join(OUT, "hourly", f)) for f in files if f.endswith(".npz")}}
    json.dump(m, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    return m


def funding(sym, first):
    ft, fv, missing = [], [], []
    for y, m in months(first, LAST_MONTH):
        blob = http(f"{BASE}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip")
        if blob is None:
            missing.append(f"{y}-{m:02d}")
            continue
        for r in zip_rows(blob):
            ft.append(int(r[0]) // 1000)
            fv.append(float(r[2]))
    ft, fv = np.array(ft, np.int64), np.array(fv)
    o = np.argsort(ft, kind="stable")
    ft, fv = ft[o], fv[o]
    _, keep = np.unique(ft, return_index=True)
    return ft[keep], fv[keep], missing


def first_month(sym):
    keys = s3_list(f"data/futures/um/monthly/klines/{sym}/1d/", delimiter=False)
    ym = sorted({tuple(int(v) for v in re.search(r"-(\d{4})-(\d{2})\.zip$", k).groups())
                 for k in keys if k.endswith(".zip")})
    return ym[0] if ym else None


def fetch_symbol(sym, first, market="um"):
    path = os.path.join(OUT, "hourly", f"{sym}{'' if market == 'um' else '_spot'}.npz")
    if os.path.exists(path):
        return sym, "cached"
    k = klines(market, sym, first)
    if k is None:
        return sym, "no klines"
    if market == "um":
        ft, fv, miss = funding(sym, first)
    else:
        ft, fv, miss = np.zeros(0, np.int64), np.zeros(0), []
    np.savez(path, **k, funding_t=ft, funding_rate=fv, funding_missing=np.array(miss, dtype="U7"))
    return sym, f"{len(k["t"])} bars, {len(ft)} prints, {len(miss)} months without funding"


def main():
    os.makedirs(os.path.join(OUT, "hourly"), exist_ok=True)
    uni_path = os.path.join(OUT, "universe.json")
    if os.path.exists(uni_path):
        uni = json.load(open(uni_path))
    else:
        pre = s3_list("data/futures/um/monthly/klines/")
        syms = sorted({p.rstrip("/").split("/")[-1] for p in pre})
        perps = [s for s in syms if s.endswith("USDT") and "_" not in s]
        with cf.ThreadPoolExecutor(16) as ex:
            fm = dict(zip(perps, ex.map(first_month, perps)))
        listed = sorted(s for s, f in fm.items() if f is not None and f <= LISTED_BY)
        uni = {"rule": "USD-M USDT perpetuals in the data.binance.vision listing (delisted included) whose first 1d "
                       "monthly archive is for a month <= 2020-12; BTCUSDT kept apart",
               "listing_symbols": len(syms), "usdt_perps": len(perps),
               "first_month": {s: f"{f[0]}-{f[1]:02d}" for s, f in fm.items() if f is not None},
               "panel": [s for s in listed if s != "BTCUSDT"]}
        json.dump(uni, open(uni_path, "w"), indent=1)
    print(f"panel: {len(uni['panel'])} symbols", flush=True)
    jobs = [(s, tuple(int(v) for v in uni["first_month"][s].split("-")), "um") for s in uni["panel"] + ["BTCUSDT"]]
    jobs.append(("BTCUSDT", (2017, 8), "spot"))
    with cf.ThreadPoolExecutor(12) as ex:
        for sym, msg in ex.map(lambda j: fetch_symbol(*j), jobs):
            print(sym, msg, flush=True)
    with cf.ThreadPoolExecutor(12) as ex:
        for sym, msg in ex.map(lambda j: fill_days(j[0], j[2]), jobs):
            print(sym, msg, flush=True)
    print(f"manifest: {len(manifest())} files", flush=True)


if __name__ == "__main__":
    sys.exit(main())
