"""Median sm% and fb MB of the dmon samples inside each run's steady-state training windows."""
import datetime as dt
import glob
import json
import os
import statistics

for js in sorted(glob.glob("D:/nt_bench_runs/*_bs*.json")):
    d = json.load(open(js))
    tag = d["tag"]
    path = f"D:/nt_bench_runs/logs/dmon_{tag}.txt"
    if not os.path.exists(path):
        print(tag, "no dmon file")
        continue
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # -o T gives HH:MM:SS as the first column
        try:
            hms = parts[0]
            sm = int(parts[2]); mem = int(parts[3]); fb = int(parts[8])
        except (ValueError, IndexError):
            continue
        rows.append((hms, sm, mem, fb))
    sm_all, fb_all = [], []
    for (t_begin, t_end) in d["steady_windows"]:
        if not t_end:
            continue
        b = dt.datetime.fromtimestamp(t_begin).strftime("%H:%M:%S")
        e = dt.datetime.fromtimestamp(t_end).strftime("%H:%M:%S")
        for hms, sm, mem, fb in rows:
            if b <= hms <= e:
                sm_all.append(sm); fb_all.append(fb)
    if sm_all:
        print(f"{tag}: steady samples={len(sm_all)} median sm%={statistics.median(sm_all)} "
              f"p10={sorted(sm_all)[len(sm_all)//10]} p90={sorted(sm_all)[9*len(sm_all)//10]} "
              f"median fb MB={statistics.median(fb_all)} max fb MB={max(fb_all)}")
    else:
        print(tag, "no samples in steady windows; rows:", len(rows))
