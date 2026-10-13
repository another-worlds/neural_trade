"""Precompute for the report (section 10): the 60-minute 90% return intervals of T03 (HAR vs trailing RV, 'cal' version = quantiles recalibrated
on the previous month) on three real forward days (calm / normal / stormy) of the Binance file 2025-11..2026-09.
It RE-USES the frozen fit code of runs/tactical/tasks/T03_vol_intervals/t03_lib.py UNCHANGED (fit_frozen, sigma_all, cal_rows, quantiles) and computes
bands only for the drawn days; nothing of the study is re-scored. Days are picked by the day's 1-minute realised volatility (quantiles 10% / 50% / 98%
over the scored days), never by interval performance.
usage: python runs/tactical/report/vol_windows.py   (CPU, ~20 s)  -> runs/tactical/report/vol_windows.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAC = HERE.parent
sys.path.insert(0, str(TAC / "tasks" / "T03_vol_intervals"))
import t03_lib as L  # noqa: E402

T1 = L.T1
H = 60


def main():
    tr_bars = T1.load_train()
    fw = T1.load_fwd()
    BT = L.make_data(tr_bars, L.FIRST_TRAIN)
    S1B = T1.build(tr_bars, 1441)
    BF = L.make_data(fw, T1.WARM)
    fr = L.fit_frozen(BT, S1B)
    D = BF["D"]
    c = fw["close"].to_numpy(np.float64)
    idx_time = fw.index
    # day volatility over the scored months
    lr = np.diff(np.log(c))
    day = pd.Series(lr, index=idx_time[1:]).groupby(lambda t: t.normalize())
    sd = day.std()
    cnt = day.size()
    ok = [d for d in sd.index if cnt[d] >= 1439 and pd.Timestamp("2025-11-02") <= d <= pd.Timestamp("2026-09-29")]
    sdv = sd.loc[ok]
    picks = []
    for lab, q in (("Спокойный день", 0.10), ("Обычный день", 0.50), ("Штормовой день", 0.98)):
        target = np.quantile(sdv.values, q)
        d = sdv.index[int(np.argmin(np.abs(sdv.values - target)))]
        picks.append((lab, d, float(sdv.loc[d]) * 1e4))
    out = {"days": [], "note": "bands = c0 * exp(q * sigma) with the cal quantiles of the month before; windows start >= 60 min apart"}
    ta = D.ta
    for lab, d, sd_bps in picks:
        month = pd.Period(d, "M")
        cidx = L.cal_rows(BF, H, month)
        cg = L.sigma_all(fr, BF, H, cidx)
        qs = {f: L.quantiles(BF["r"][H][cidx] / cg[f]) for f in ("trail", "HAR")}
        m = np.asarray((ta >= d) & (ta < d + pd.Timedelta(days=1)))
        rows = np.where(m & L.row_ok(BF, H))[0]
        sg = L.sigma_all(fr, BF, H, rows)
        chosen, last = [], None
        for k, r_ in enumerate(rows):
            if last is None or ta[r_] >= last + pd.Timedelta(minutes=H):
                chosen.append(k)
                last = ta[r_]
        wins = []
        for k in chosen:
            r_ = rows[k]
            a = int(D.anch[r_])
            c0 = float(c[a - 1])
            w = {"t0": str(ta[r_]), "c0": round(c0, 2), "end": round(float(c[a + H - 1]), 2)}
            for f in ("HAR", "trail"):
                lo, hi = qs[f]["lo"] * sg[f][k], qs[f]["hi"] * sg[f][k]
                w[f] = {"lo": round(c0 * float(np.exp(lo)), 2), "hi": round(c0 * float(np.exp(hi)), 2), "sigma_bps": round(float(sg[f][k]) * 1e4, 2)}
                r = float(np.log(w["end"] / c0))
                w[f]["inside"] = bool(lo <= r <= hi)
            wins.append(w)
        t0 = pd.Timestamp(d)
        sl = slice(int(np.searchsorted(idx_time, t0)), int(np.searchsorted(idx_time, t0 + pd.Timedelta(days=1, hours=1))))
        out["days"].append({"label": lab, "date": str(d.date()), "sd_bps_1m": round(sd_bps, 3), "month": str(month),
                            "t": [str(t) for t in idx_time[sl]], "close": [round(float(x), 2) for x in c[sl]], "windows": wins,
                            "width_bps": {f: round(float(np.mean([(w[f]["hi"] - w[f]["lo"]) / w["c0"] for w in wins])) * 1e4, 1) for f in ("HAR", "trail")},
                            "covered": {f: int(sum(w[f]["inside"] for w in wins)) for f in ("HAR", "trail")}, "n_windows": len(wins)})
        print(lab, d.date(), f"1m sd {sd_bps:.2f} bps, {len(wins)} windows, covered", out["days"][-1]["covered"], "width", out["days"][-1]["width_bps"], flush=True)
    json.dump(out, open(HERE / "vol_windows.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
