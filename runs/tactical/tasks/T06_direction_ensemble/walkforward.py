"""T06 honest present: the monthly walk-forward 2024-07 .. 2025-07-20 (13 months, the last partial) for L0, L5, L6 (ladder.py, unchanged
logic via members_lab) and their E1 (equal-weight mean of P(up)). Next-open labels as H59 (decay_nextopen.py): anchors every 10 bars,
a 60-bar window A[i-60:i], r_h = close[i+H] / open[i] - 1 for H = 10, 15, 20, a 5-bps deadband, train = the 60 days before the month (labels
end before it: anchors earlier than the month start minus 20 minutes), score = the month. The top-10% threshold is the 90th percentile of
the member's (or the ensemble's) confidence on the inner-model predictions of the last 15% of the training rows (a training-block
quantity); hit / bps are read on h1 with the mean of the three horizons' P(up), rows with |r_h1| above the deadband.
CPU only; reads the CUT file only. usage: python walkforward.py"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "common")); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import cutoff as C
import members_lab as ML
import ladder
import lab
import ens_lib as E
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

L, STRIDE, DB, HS = 60, 10, 5e-4, (10, 15, 20)
FIRST, LAST = "2024-07", C.LAST_MONTH
CACHE = os.environ.get("SERIES_CACHE", "D:/nt/nt_t06_cache")
OUT = os.path.join(HERE, "walkforward.json")


def load():
    d = C.read_csv(usecols=["timestamp", "open", "high", "low", "close", "volume"])
    C.assert_cut(d["timestamp"], "walkforward")
    t = pd.DatetimeIndex(pd.to_datetime(d["timestamp"]))
    return d[["open", "high", "low", "close", "volume"]].to_numpy(np.float64), t


def make_S(Ftr, Ytr, Mtr, Fte_scaled):
    n = len(Ftr); cut = int(round((1 - ladder.INNER) * n))
    return dict(Ftr=Ftr, Fva=Fte_scaled, Y=Ytr, M=Mtr, itr=np.arange(0, cut - ladder.GAP), iva=np.arange(cut, n))


def month_rows(m, t, anch, r):
    m0, m1 = m.start_time, (m + 1).start_time
    tr = np.asarray((t[anch - 1] >= m0 - pd.Timedelta(days=60)) & (t[anch - 1] < m0 - pd.Timedelta(minutes=max(HS))))
    te = np.asarray((t[anch - 1] >= m0) & (t[anch - 1] < m1))
    return tr, te


def main():
    t0 = time.time(); A, t = load(); n = len(A)
    anch = np.arange(L, n - max(HS), STRIDE)
    keep = np.asarray(t[anch - 1] >= pd.Timestamp("2024-04-25")); anch = anch[keep]       # 60 days before 2024-07 plus slack
    R = np.stack([A[anch + h, 3] / A[anch, 0] - 1 for h in HS], 1)
    F_tb = np.nan_to_num(np.concatenate([lab.fs_tb7(np.stack([A[i - L:i] for i in anch[k:k + 5000]]).astype(np.float32)) for k in range(0, len(anch), 5000)]))
    F_rich = np.nan_to_num(np.concatenate([lab.fs_rich(np.stack([A[i - L:i] for i in anch[k:k + 5000]]).astype(np.float32)) for k in range(0, len(anch), 5000)]))
    print("anchors", len(anch), "features", f"{time.time() - t0:.0f}s", flush=True)
    Y = (R > 0).astype(np.float32); Mk = (np.abs(R) > DB).astype(np.float32)
    rows = []
    for m in pd.period_range(FIRST, LAST, freq="M"):
        tr, te = month_rows(m, t, anch, R)
        if tr.sum() < 2000 or te.sum() < 500:
            continue
        res = {}
        # --- L0 (tb7) ---
        tf.keras.utils.set_random_seed(0)
        sc = StandardScaler().fit(F_tb[tr]); S = make_S(sc.transform(F_tb[tr]).astype(np.float32), Y[tr], Mk[tr], sc.transform(F_tb[te]).astype(np.float32))
        res["L0"] = ML.l0(S)
        # --- L5 / L6 (rich) ---
        tf.keras.utils.set_random_seed(0)
        sc = StandardScaler().fit(F_rich[tr]); S = make_S(sc.transform(F_rich[tr]).astype(np.float32), Y[tr], Mk[tr], sc.transform(F_rich[te]).astype(np.float32))
        l5 = ladder.l5_stage(S); inner_m = ML.inner_model_from(S["Ftr"], l5[1], S["Ftr"].shape[1])
        res["L5"] = (ML.pred(l5[3], [S["Fva"]]), ML.pred(inner_m, [S["Ftr"][S["iva"]]]))
        Pv, Pt, be = ML.residual(S, l5, True, False); res["L6"] = (Pv, Pt)
        Pv_all = np.stack([res[k][0] for k in ("L0", "L5", "L6")]); Pt_all = np.stack([res[k][1] for k in ("L0", "L5", "L6")])
        res["E1"] = (E.e1(Pv_all), E.e1(Pt_all))
        rte = R[te]; rtr_dummy = R[tr]
        row = dict(month=str(m), n_train=int(tr.sum()), n_test=int(te.sum()), best_epoch_L6=int(be), C_L5=[float(c) for c in l5[0]])
        for k, (Pv, Pt) in res.items():
            sc_ = E.score(Pv, rte, rtr_dummy, DB); thr = E.conf_threshold(Pt); tp = E.top_read(Pv, rte, DB, thr)
            row[k] = dict(sc_, thr=thr, **{f"top10_{a}": b for a, b in tp.items()})
        rows.append(row)
        print(row["month"], {k: round(row[k]["auc3"], 4) for k in res}, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(dict(rows=rows, seconds=round(time.time() - t0, 1)), open(OUT, "w"), indent=1)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
