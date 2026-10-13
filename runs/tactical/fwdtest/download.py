"""Download Binance spot BTCUSDT 1m klines (data.binance.vision), verify .CHECKSUM, convert to the main file's format.
Refuses to fetch any date in the MVP's protected span 2025-07-21..2025-09-30 (guard in fetch()).
    python download.py check2025_06   -> D:/nt/nt_fwd_data/BTCUSDT_1m_2025-06.csv  (source check month)
    python download.py forward        -> D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"""
import hashlib, io, json, os, sys, zipfile, datetime as dt
import urllib.request, urllib.error
import pandas as pd

BASE = "https://data.binance.vision/data/spot/{k}/klines/BTCUSDT/1m/BTCUSDT-1m-{d}.zip"
OUT = "D:/nt/nt_fwd_data"
PROT0, PROT1 = pd.Timestamp("2025-07-21"), pd.Timestamp("2025-10-01")   # protected [PROT0, PROT1)
COLS = ["open_time", "open", "high", "low", "close", "volume"]


def guard(d):
    a = pd.Timestamp(d + "-01") if len(d) == 7 else pd.Timestamp(d)
    b = (a + pd.offsets.MonthBegin(1)) if len(d) == 7 else a + pd.Timedelta(days=1)
    if a < PROT1 and b > PROT0:
        raise SystemExit(f"refused: {d} overlaps the protected span")


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch(kind, d, log):
    guard(d)
    url = BASE.format(k=kind, d=d)
    z = get(url)
    if z is None:
        log.append({"d": d, "kind": kind, "status": "404"}); return None
    cs = get(url + ".CHECKSUM")
    sha = hashlib.sha256(z).hexdigest()
    ok = None
    if cs is not None:
        ok = cs.decode().split()[0].lower() == sha
        if not ok:
            raise SystemExit(f"CHECKSUM MISMATCH {d}")
    log.append({"d": d, "kind": kind, "bytes": len(z), "sha256": sha, "checksum_published": cs is not None, "checksum_ok": ok})
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        raw = zf.read(zf.namelist()[0])
    head = raw[:20].decode()
    df = pd.read_csv(io.BytesIO(raw), header=0 if not head[0].isdigit() else None)
    df = df.iloc[:, :6]; df.columns = COLS
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.max() > 10 ** 14 else "ms"
    log[-1]["unit"] = unit
    df["timestamp"] = pd.to_datetime(ot, unit=unit)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main(mode):
    os.makedirs(OUT, exist_ok=True); log = []
    if mode == "check2025_06":
        df = fetch("monthly", "2025-06", log)
        df.to_csv(f"{OUT}/BTCUSDT_1m_2025-06.csv", index=False)
        json.dump(log, open(f"{OUT}/download_log_check.json", "w"), indent=1); print(log); return
    today = pd.Timestamp(dt.datetime.utcnow().date())
    parts = []
    for per in pd.period_range("2025-10", today.to_period("M"), freq="M"):
        if per < today.to_period("M"):
            d = fetch("monthly", str(per), log)
            if d is not None:
                parts.append(d); continue
        for day in pd.date_range(per.start_time, min(per.end_time.normalize(), today - pd.Timedelta(days=1))):
            d = fetch("daily", day.strftime("%Y-%m-%d"), log)
            if d is not None:
                parts.append(d)
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp")
    assert df["timestamp"].min() >= PROT1
    df.to_csv(f"{OUT}/BTCUSDT_1m_2025-10_onward.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    json.dump(log, open(f"{OUT}/download_log.json", "w"), indent=1)
    print(len(df), df["timestamp"].min(), df["timestamp"].max())


if __name__ == "__main__":
    main(sys.argv[1])
