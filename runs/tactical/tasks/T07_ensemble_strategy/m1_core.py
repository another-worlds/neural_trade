"""T07 member M1: the minute reversal ensemble. E1 = the mean of P(up) of T06's walk-forward members L0, L5, L6 (members_lab.py / ladder.py,
logic unchanged), read at the 15-minute horizon (column 1). Features, labels and masks as T06 walkforward.py (60-bar window, anchors every
10 bars, next-open labels r_h = close[i+h] / open[i] - 1 for h = 10, 15, 20, a 5-bps deadband, train = the 60 days before the month with the
labels closed 20 minutes before it). Importing this module imports lab.py, which chdirs to the repo root: use absolute paths.
CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "tasks", "T06_direction_ensemble")); sys.path.insert(0, os.path.join(TAC, "rebuild"))
sys.path.insert(0, os.path.join(TAC, "lab"))
import numpy as np
import pandas as pd

L, STRIDE, DB, HS = 60, 10, 5e-4, (10, 15, 20)
TRAIN_DAYS = 60


def features(raw):
    """Anchor arrays of a raw 1-minute frame (rows in time order): t_sig (open time of the window's last bar), t_entry (open of the next bar),
    t_exit (end of the 15-minute label bar), tb7 / rich features, R [n, 3] next-open returns. Window = rows [i-60, i), entry row i."""
    import lab
    A = raw[["open", "high", "low", "close", "volume"]].to_numpy(np.float64); t = pd.DatetimeIndex(raw.index); n = len(A)
    anch = np.arange(L, n - max(HS), STRIDE)
    R = np.stack([A[anch + h, 3] / A[anch, 0] - 1 for h in HS], 1)
    ftb, fri = [], []
    for k in range(0, len(anch), 5000):
        a = anch[k:k + 5000]
        W = A[a[:, None] + np.arange(-L, 0)[None, :]].astype(np.float32)
        ftb.append(np.nan_to_num(lab.fs_tb7(W))); fri.append(np.nan_to_num(lab.fs_rich(W)))
    ex = t.values.astype("datetime64[m]")[anch + HS[1]] + np.timedelta64(1, "m")
    return dict(t_sig=t.values.astype("datetime64[m]")[anch - 1], t_entry=t.values.astype("datetime64[m]")[anch], t_exit=ex,
                tb=np.concatenate(ftb).astype(np.float32), rich=np.concatenate(fri).astype(np.float32), R=R)


def train_mask(t_sig, m0):
    """T06 month_rows: train = t_sig in [m0 - 60 d, m0 - 20 min)."""
    ts = pd.DatetimeIndex(t_sig)
    return np.asarray((ts >= m0 - pd.Timedelta(days=TRAIN_DAYS)) & (ts < m0 - pd.Timedelta(minutes=max(HS))))


def month_masks(t_sig, m):
    """train as train_mask at the month start; test = t_sig in the month."""
    ts = pd.DatetimeIndex(t_sig); m0, m1 = m.start_time, (m + 1).start_time
    return train_mask(t_sig, m0), np.asarray((ts >= m0) & (ts < m1))


def fit_predict(F, tr, Fte):
    """Fit L0, L5, L6 on the training rows of F (dict tb / rich / R) and predict the feature dict Fte (tb / rich). Returns P [3 members, n, 3]."""
    import ladder
    import members_lab as ML
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    R = F["R"][tr]; Y = (R > 0).astype(np.float32); Mk = (np.abs(R) > DB).astype(np.float32)

    def make_S(Ftr, Fte_s):
        n = len(Ftr); cut = int(round((1 - ladder.INNER) * n))
        return dict(Ftr=Ftr, Fva=Fte_s, Y=Y, M=Mk, itr=np.arange(0, cut - ladder.GAP), iva=np.arange(cut, n))
    out = []
    tf.keras.utils.set_random_seed(0)
    sc = StandardScaler().fit(F["tb"][tr]); S = make_S(sc.transform(F["tb"][tr]).astype(np.float32), sc.transform(Fte["tb"]).astype(np.float32))
    out.append(ML.l0(S)[0])
    tf.keras.utils.set_random_seed(0)
    sc = StandardScaler().fit(F["rich"][tr]); S = make_S(sc.transform(F["rich"][tr]).astype(np.float32), sc.transform(Fte["rich"]).astype(np.float32))
    l5 = ladder.l5_stage(S)
    out.append(ML.pred(l5[3], [S["Fva"]]))
    Pv, _Pt, _be = ML.residual(S, l5, True, False)
    out.append(Pv)
    return np.stack(out)


def month_task(args):
    """Worker: one walk-forward month from the saved feature file. Writes CACHE/m1_<stage>_<month>.npz (resumable)."""
    feat_path, month, out_path = args
    if os.path.exists(out_path):
        return month
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(1)
    z = np.load(feat_path); F = {k: z[k] for k in ("t_sig", "tb", "rich", "R")}
    m = pd.Period(month, "M"); tr, te = month_masks(F["t_sig"], m)
    if tr.sum() < 2000 or te.sum() < 100:
        np.savez(out_path, ok=False); return month
    P = fit_predict(F, tr, dict(tb=F["tb"][te], rich=F["rich"][te]))
    np.savez(out_path, ok=True, idx=np.flatnonzero(te), P=P.astype(np.float32), n_train=int(tr.sum()))
    return month


def frozen_task(args):
    """Worker: fit once on the 60 days before `cut_start` (feature file `train_path`) and predict every anchor of `test_path`."""
    train_path, cut_start, test_path, out_path = args
    if os.path.exists(out_path):
        return out_path
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4); tf.config.threading.set_inter_op_parallelism_threads(1)
    z = np.load(train_path); F = {k: z[k] for k in ("t_sig", "tb", "rich", "R")}
    tr = train_mask(F["t_sig"], pd.Timestamp(cut_start))
    zt = np.load(test_path)
    P = fit_predict(F, tr, dict(tb=zt["tb"], rich=zt["rich"]))
    np.savez(out_path, P=P.astype(np.float32), n_train=int(tr.sum()), train_last_sig=str(F["t_sig"][tr].max()))
    return out_path
