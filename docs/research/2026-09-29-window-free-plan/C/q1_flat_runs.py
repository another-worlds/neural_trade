"""Q1 support: runs of flat zero-volume bars (close unchanged, volume 0) in the long file: forward-filled
outages that a missing-minute check cannot see. Read-only; CPU. Writes q1_flat_runs.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

df = pd.read_csv("D:/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
flat = (df["volume"].values == 0) & (df["high"].values == df["low"].values)
flat[1:] &= (df["close"].values[1:] == df["close"].values[:-1])
# run lengths of consecutive flat bars
idx = np.flatnonzero(np.diff(np.concatenate([[0], flat.astype(np.int8), [0]])))
starts, ends = idx[::2], idx[1::2]
lens = ends - starts
out = {"flat_bars": int(flat.sum()), "flat_share": float(flat.mean()), "runs": int(len(lens)),
       "runs_ge": {str(k): int((lens >= k).sum()) for k in (10, 60, 240, 1440)},
       "longest_runs": [{"start": str(df["timestamp"].iloc[s]), "minutes": int(l)}
                        for s, l in sorted(zip(starts, lens), key=lambda x: -x[1])[:8]]}
Path(__file__).with_name("q1_flat_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
