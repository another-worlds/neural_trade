"""Static excerpt of the long BTC/USDT file for tactical tests and everyday work (owner, 2026-10-08: "prepare a static
cut of the big file for testing and the main tasks; the whole file only for giant runs").

What it keeps (rows of Bitcoin_BTCUSDT.csv, unchanged, same CSV format):
  * for each of the 12 tactical slices (the 6 climb slices = the candidate check's C2 / hill-climb slices, and the 6
    held-out final slices of runs/tactical/hc4/make_hc4.py), the CHUNK_DAYS days that end at that DATA_END - enough for
    the long block (MAX_SEQUENCE_COUNT 126,000 windows = 87.5 days) plus the window, horizons and trend periods;
  * the file's last TAIL_DAYS days (the protected dev/test span, D-020), so DATA_END_PROTECTED_DAYS refuses exactly the
    same dates as on the full file. Nothing in that tail is used for any choice.
The loader drops empty minutes and never fills gaps (data/preprocessors.resample_bars), so chunks just follow each
other; a window that would straddle a gap is older than the 126,000 most recent ones and is trimmed away.
Writes D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv (git-ignored like every *.csv) and a manifest with its sha256.
usage: python make_excerpt.py"""
import hashlib, json, os, sys
import pandas as pd

sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical\hc4"); sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical")
import make_hc4 as H

SRC = r"D:\nt\neural_trade\Bitcoin_BTCUSDT.csv"
DST = r"D:\nt\neural_trade\Bitcoin_BTCUSDT_tactical.csv"
MANIFEST = r"D:\nt\nt_tactical\runs\tactical\data\excerpt_manifest.json"
CHUNK_DAYS, TAIL_DAYS = 95, 70

slices = sorted(set(H.CLIMB) | set(H.FINAL))
df = pd.read_csv(SRC)
ts = pd.to_datetime(df["timestamp"])
keep = ts >= ts.max() - pd.Timedelta(days=TAIL_DAYS)
chunks = []
for s in slices:
    end = pd.Timestamp(s)
    m = (ts > end - pd.Timedelta(days=CHUNK_DAYS)) & (ts <= end)
    keep |= m
    chunks.append({"data_end": s, "from": str(end - pd.Timedelta(days=CHUNK_DAYS)), "rows": int(m.sum())})
out = df[keep]
out.to_csv(DST, index=False)
h = hashlib.sha256(open(DST, "rb").read()).hexdigest()
man = {"source": SRC, "source_rows": int(len(df)), "excerpt": DST, "rows": int(len(out)), "sha256": h,
       "bytes": os.path.getsize(DST), "chunk_days": CHUNK_DAYS, "tail_days": TAIL_DAYS, "tail_from": str(ts.max() - pd.Timedelta(days=TAIL_DAYS)),
       "slices": chunks, "climb_slices": H.CLIMB, "final_slices": H.FINAL}
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
json.dump(man, open(MANIFEST, "w"), indent=1)
print(json.dumps({k: man[k] for k in ("rows", "source_rows", "bytes", "sha256")}))
