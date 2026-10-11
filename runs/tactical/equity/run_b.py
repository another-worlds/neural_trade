"""Strategy (b): H56 picks, monthly walk-forward over the look months 2024-07..2025-09; each month trades its 10% most confident
predictions (|P-0.5| >= the 90th percentile of the month's own predictions, as recent/wf.py): long if P > 0.5, else short, held for
the label horizon (1m: 15 bars after the entry; 1h: 1 bar). Same fit as wf.walk_forward (imported pieces), plus the predictions.
usage: python run_b.py 1m|1h   (1m: all:l1:365, 1h: rich:l1:365)"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REC = os.path.join(os.path.dirname(HERE), "recent")
sys.path.insert(0, REC); sys.path.insert(0, HERE)
import wf  # noqa: E402
import eqlib as E  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

bar = sys.argv[1]; name, kind, T = {"1m": ("all", "l1", 365), "1h": ("rich", "l1", 365)}[bar]
t0 = time.time(); wf.lowprio()
blocks, t, r = wf.load_cell(bar)
X = np.concatenate([np.asarray(blocks[b], np.float32) for b in wf.SETS[name]], 1) if len(wf.SETS[name]) > 1 else np.asarray(blocks[wf.SETS[name][0]])
f, l = wf.MONTHS["look"]; ti = pd.DatetimeIndex(t)
sel = np.asarray((ti >= pd.Period(f, "M").start_time - pd.Timedelta(days=T + 1)) & (ti < (pd.Period(l, "M") + 1).start_time))
X, ti, r = X[sel], ti[sel], r[sel]
H, bm = wf.HOR[bar], wf.BAR_MIN[bar]; db = wf.deadband(bar); mask = np.abs(r) > db; y = (r > 0).astype(int)
rows = []; chk = []
for m in pd.period_range(f, l, freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    tr, te = wf.split_masks(ti, m0, m1, H, bm, T); trm, tem = tr & mask, te & mask
    if trm.sum() < wf.MIN_TRAIN or tem.sum() < wf.MIN_TEST or len(np.unique(y[tem])) < 2 or len(np.unique(y[trm])) < 2:
        continue
    Xtr = np.asarray(X[trm], np.float32); sc = StandardScaler().fit(Xtr)
    g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
    mdl, info = wf._fit(kind, g(Xtr), y[trm], gap_rows=H + 1)
    p = mdl.predict_proba(g(X[te]))[:, 1]; conf = np.abs(p - 0.5); k = conf >= np.quantile(conf, 0.9)
    rt = r[te][k]; side = np.where(p[k] > 0.5, 1, -1)
    # entry = close of the anchor's last bar (t is that bar's open time) = t + 1 bar; exit = close of the label bar = t + (H+1) bars
    t_open = ti[te][k].values.astype("datetime64[ns]").astype(np.int64)
    bar_ns = bm * 60 * 10 ** 9
    rows.append(dict(entry=t_open + bar_ns, exit=t_open + (H + 1) * bar_ns, r=rt, side=side))
    mv = side * rt * 1e4; chk.append((str(m), float(np.mean(mv > 0)), float(np.mean(mv)), int(k.sum()), info.get("C")))
    print(chk[-1], f"{time.time() - t0:.0f}s", flush=True)
# consistency with recent/look_<bar>.jsonl (the pick's stored top-10% numbers)
stored = [json.loads(x) for x in open(f"{REC}/look_{bar}.jsonl", encoding="utf-8")]
st = [d for d in stored if (d["set"], d["model"], d["T"]) == (name, kind, T)][0]["months"]
# NB: the stored 1m look was fitted before wf._fit seeded the L1 solver (run-to-run noise of liblinear), so a rerun is not bit-identical;
# the differences are recorded, not asserted (1h is expected to match exactly).
dh = [h - s["top10_hit"] for (mo, h, b, n, _), s in zip(chk, st)]; db_ = [b - s["top10_bps"] for (mo, h, b, n, _), s in zip(chk, st)]
assert [c[0] for c in chk] == [s["month"] for s in st] and [c[3] for c in chk] == [s["n_top10"] for s in st]
repro = {"max_abs_hit_diff": float(np.max(np.abs(dh))), "max_abs_bps_diff": float(np.max(np.abs(db_))),
         "mean_bps_rerun": float(np.mean([c[2] for c in chk])), "mean_bps_stored": float(np.mean([s["top10_bps"] for s in st])),
         "mean_hit_rerun": float(np.mean([c[1] for c in chk])), "mean_hit_stored": float(np.mean([s["top10_hit"] for s in st]))}
print("rerun vs stored look rows:", json.dumps(repro), flush=True)
cat = {k: np.concatenate([x[k] for x in rows]) for k in ("entry", "exit", "r", "side")}
res = E.finish(f"b_h56_{bar}_{name}_{kind}_{T}", f"(b) H56 {bar} {name} {kind} {T} d, top-10% most confident, 2024-07..2025-09",
               cat["entry"], cat["exit"], cat["r"], -cat["r"], cat["side"], int(pd.Timestamp("2024-07-01").value), HERE,
               extra={"config": f"{bar} {name}:{kind}:{T}", "months": len(rows), "rerun_vs_stored_look": repro, "hold": f"{H} bars of {bm} min"})
print(json.dumps({c: v["stats"] for c, v in res["levels"].items()}, indent=1)); print(f"n_signals {res['n_signals']} trades {res['n_trades']} skipped {res['n_skipped']}; {time.time() - t0:.0f}s")
