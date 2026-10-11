"""Strategy (c): H38 hourly dev winner T12_tp1.5_sl1.5_rich_hgb_barrier_mag, meta selection (q 0.2), held-out 2024-07..2025-07-20 (recut).
Runs runs/tactical/hourly/hourly.py's own code (build, run_split; the only edits: work directory, and run_split also returns the side).
usage: python run_c.py"""
import json, os, sys, time, types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import eqlib as E  # noqa: E402

src = open(os.path.join(WT, "runs/tactical/hourly/hourly.py"), encoding="utf-8").read()
src = src.replace('os.chdir(r"D:' + chr(92) + 'nt' + chr(92) + 'nt_recut")', f'os.chdir(r"{WT}")')
assert f'os.chdir(r"{WT}")' in src
assert 'out[name]["_pl"] = pl[k]' in src
src = src.replace('out[name]["_pl"] = pl[k]', 'out[name]["_pl"] = pl[k]; out[name]["_side"] = side[k]')
H = types.ModuleType("hourly_copy"); H.__file__ = os.path.join(WT, "runs/tactical/hourly/hourly.py"); exec(compile(src, H.__file__, "exec"), H.__dict__)

t0 = time.time()
a = types.SimpleNamespace(T=12, tp=1.5, sl=1.5, q=0.2, feat="rich", primary="hgb", target="barrier", meta="mag")
D = H.build(a); idx = np.arange(len(D["F"])); rng = np.random.default_rng(0)
dev = idx[D["t_exit"] < H.HOLD]; held = idx[D["t_entry"] >= H.HOLD]
tr = dev[D["t_exit"][dev] < H.HOLD - pd.Timedelta(hours=2 * a.T)]
r = H.run_split(D, a, tr, held, rng); m = r["meta"]
print(f"meta: n {m['n']} hit {m['hit']:.4f} bps {m['bps']:.2f} (H38: 2013, 0.538, +9.83)", flush=True)
# rows of the selected trades in D (t_entry is unique, one anchor per hourly bar)
te_t = pd.DatetimeIndex(D["t_entry"]); rows = te_t.get_indexer(pd.DatetimeIndex(m["_t"])); assert (rows >= 0).all()
side = m["_side"]; assert np.allclose(np.where(side > 0, D["pl_long"][rows], D["pl_short"][rows]), m["_pl"])
# exit: first barrier touch (or the label's last bar), end of that bar
b = H.bars(); A = b.values.astype(np.float64); n = len(A); T = a.T
anch = np.arange(H.L + 168, n - T); c0 = A[anch - 1, 3]; unit = D["unit"]
Hf = np.lib.stride_tricks.sliding_window_view(A[:, 1], T)[anch]; Lf = np.lib.stride_tricks.sliding_window_view(A[:, 2], T)[anch]
lt, ls = H.first_touch(Hf >= (c0 * (1 + a.tp * unit))[:, None]), H.first_touch(Lf <= (c0 * (1 - a.sl * unit))[:, None])
st_, ss = H.first_touch(Lf <= (c0 * (1 - a.tp * unit))[:, None]), H.first_touch(Hf >= (c0 * (1 + a.sl * unit))[:, None])
rT = A[anch + T - 1, 3] / c0 - 1
assert np.allclose(np.where(ls <= lt, np.where(ls < T, -a.sl * unit, rT), np.where(lt < T, a.tp * unit, rT)), D["pl_long"])
jl, js = np.minimum(np.minimum(ls, lt), T - 1), np.minimum(np.minimum(ss, st_), T - 1)
j = np.where(side > 0, jl[rows], js[rows]); hour = 3600 * 10 ** 9
entry_ns = D["t_entry"].values.astype("datetime64[ns]").astype(np.int64)[rows] + hour
exit_ns = D["t_entry"].values.astype("datetime64[ns]").astype(np.int64)[rows] + (j + 2) * hour
res = E.finish("c_h38_hourly_barrier_meta", "(c) H38 hourly T12 tp/sl 1.5, rich HGB barrier+mag meta, 2024-07..2025-07-20 (recut)", entry_ns, exit_ns,
               D["pl_long"][rows], D["pl_short"][rows], side, int(H.HOLD.value), HERE,
               extra={"config": "T12 tp1.5 sl1.5 q0.2 rich hgb barrier mag", "hold_hours_median": float(np.median(j + 1)),
                      "reproduces_H38": {"n": m["n"], "hit": m["hit"], "bps": m["bps"]}})
print(json.dumps({c: v["stats"] for c, v in res["levels"].items()}, indent=1)); print(f"n_signals {res['n_signals']} trades {res['n_trades']} skipped {res['n_skipped']}; {time.time() - t0:.0f}s")
