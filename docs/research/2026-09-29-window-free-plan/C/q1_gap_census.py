"""Q1 support: missing-minute census of the long history file, and the share of anchors a
reset-and-mask burn-in at every data gap would remove, for the burn-ins M(1e-3) of q1_purge_costs.

Reads D:/neural_trade/Bitcoin_BTCUSDT.csv (read-only; the owner's local file, not in git). CPU only.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_gap_census.py
Writes q1_gap_census.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("D:/neural_trade/Bitcoin_BTCUSDT.csv")
OUT = Path(__file__).with_name("q1_gap_census.json")
L_NET = 60
# M(1e-3) after the maximal shift, single EWMA (q1_purge_costs.json)
BURN = {"p60": 340, "p240": 1365, "p1440": 8198, "p10080": 57399}


def masked_fraction(seg_lengths, burn):
    """Anchors lost if the state resets at every segment start and the first burn + L_NET bars are masked."""
    lost = np.minimum(seg_lengths, burn + L_NET).sum()
    return float(lost / seg_lengths.sum())


def main():
    ts = pd.read_csv(SRC, usecols=["timestamp", "volume", "close"])
    t = pd.to_datetime(ts["timestamp"]).values.astype("datetime64[s]").astype(np.int64)
    d = np.diff(t)
    n = len(t)
    out = {"file": str(SRC), "bars": int(n), "first": str(ts["timestamp"].iloc[0]), "last": str(ts["timestamp"].iloc[-1])}
    out["non_monotonic_or_duplicate_steps"] = int((d <= 0).sum())
    gap_idx = np.where(d > 60)[0]
    missing = (d[gap_idx] // 60 - 1)
    out["gaps_n"] = int(len(gap_idx))
    out["missing_minutes_total"] = int(missing.sum())
    out["expected_minutes"] = int((t[-1] - t[0]) // 60 + 1)
    out["gap_length_minutes_quantiles"] = {q: float(np.quantile(missing, q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(missing) else {}
    # segments between gaps (contiguous runs of bars)
    starts = np.concatenate([[0], gap_idx + 1])
    ends = np.concatenate([gap_idx + 1, [n]])
    seg = ends - starts
    out["segments_n"] = int(len(seg))
    out["segment_length_bars_quantiles"] = {q: float(np.quantile(seg, q)) for q in (0.1, 0.5, 0.9)}
    out["masked_fraction_reset_at_every_gap"] = {k: masked_fraction(seg, v) for k, v in BURN.items()}
    # the same, ignoring gaps of a single missing minute (bridged by carrying the state across one bar)
    big = gap_idx[missing > 1]
    starts2 = np.concatenate([[0], big + 1]); ends2 = np.concatenate([big + 1, [n]])
    seg2 = ends2 - starts2
    out["gaps_over_1_minute_n"] = int(len(big))
    out["masked_fraction_reset_at_gaps_over_1_minute"] = {k: masked_fraction(seg2, v) for k, v in BURN.items()}
    # per year
    years = pd.to_datetime(ts["timestamp"].iloc[gap_idx + 1]).dt.year.value_counts().sort_index()
    out["gaps_per_year"] = {int(k): int(v) for k, v in years.items()}
    # zero-volume bars (possible forward-filled minutes): share per year
    vol0 = (ts["volume"].values == 0)
    yr = pd.to_datetime(ts["timestamp"]).dt.year.values
    out["zero_volume_share_per_year"] = {int(y): float(vol0[yr == y].mean()) for y in np.unique(yr)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
