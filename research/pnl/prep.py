"""Shared data preparation: cached features, per-fold scalers, targets on the 1-min grid, decision masks.

All arrays are indexed by the dev bar index (0 = 2020-01-01 00:00). Nothing here reads the seal.
"""
import json
import os

import numpy as np

import data as D
import features as FE
import folds as FO

WORK = FE.WORK
GROUPS = {"A": ("z_",), "B": ("lnsig_", "park_"), "C": ("er_", "vr15", "dhi_", "dlo_"),
          "D": ("vol_", "trades_", "tsize_"), "E": ("imb_",), "F": ("fund_",),
          "G": ("hsin", "hcos", "dsin", "dcos", "weekend"), "daily": ("dz_", "ddd_", "dsma_", "lnvt")}


class Store:
    """Bars, features (memory-mapped), sigma, 1-min returns and funding, all on the dev grid."""

    def __init__(self, bars=None, X=None, names=None, sigma=None):
        self.b = bars if bars is not None else D.load(1)
        if X is None:
            X = np.load(os.path.join(WORK, "features_v1.npy"), mmap_mode="r")
            meta = json.load(open(os.path.join(WORK, "features_v1.json")))
            names = meta["columns"]
            sigma = np.load(os.path.join(WORK, "sigma_v1.npy")).astype(np.float64)
        self.X, self.names, self.sigma = X, list(names), sigma
        assert len(self.X) == len(self.b.t)
        c = self.b.c
        self.lc = np.log(c)
        self.n = len(c)
        r = np.r_[0.0, np.diff(self.lc)]
        self.sigma60 = np.maximum(np.sqrt(FE.ewm(r * r, 60)), 0.5e-4)
        self.R1 = np.r_[c[1:] / c[:-1] - 1.0, 0.0]                  # held over (t, t+1]
        self.F1 = np.r_[self.b.funding[1:], 0.0]                     # charged at bar t+1 to the position held at t
        self.cumF = np.r_[0.0, np.cumsum(self.b.funding)]             # cumF[i] = sum funding[0:i]
        self.day = (self.b.t // 86400).astype(np.int64)

    def idx(self, ts):
        return int(np.searchsorted(self.b.t, ts))

    def cols(self, track):
        return FE.track1_cols(self.names) if track == 1 else list(range(len(self.names)))

    def group_index(self, cols):
        """Group id per selected column (for feature-group dropout)."""
        gid = []
        keys = list(GROUPS)
        for j in cols:
            nm = self.names[j]
            g = next(k for k in keys if any(nm.startswith(p) for p in GROUPS[k]))
            gid.append(keys.index(g))
        return np.array(gid), keys

    def target(self, rows, h):
        """R = c(t+h)/c(t) - 1, F = funding charged in (t, t+h], sigma_hat = sigma_t sqrt(h); rows + h < n."""
        R = np.exp(self.lc[rows + h] - self.lc[rows]) - 1.0
        F = self.cumF[rows + h + 1] - self.cumF[rows + 1]
        return R, F, self.sigma[rows] * np.sqrt(h)

    def sigma_H(self, rows, H):
        return (self.sigma60[rows] if H <= 60 else self.sigma[rows]) * np.sqrt(H)


def decision_mask(t, h, phase=0):
    """Decision bars at cadence h minutes. phase 0 = closes at minute h-1 of each h-block (hh:59, 23:59 ...)."""
    m = (t // 60) % h
    return m == ((h - 1 - phase) % h)


class Scaler:
    """Per-fold median/IQR, fitted on training rows only (deterministic subsample), clip +-5, NaN -> 0."""

    def __init__(self, X, rows, cols, max_rows=400_000):
        if len(rows) > max_rows:
            rows = rows[:: int(np.ceil(len(rows) / max_rows))]
        A = np.asarray(X[rows][:, cols], dtype=np.float64)
        self.med = np.nanmedian(A, axis=0)
        q75, q25 = np.nanpercentile(A, [75, 25], axis=0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
        self.cols = np.asarray(cols)

    def __call__(self, X, rows):
        A = (np.asarray(X[rows][:, self.cols], dtype=np.float64) - self.med) / self.iqr
        A = np.clip(np.nan_to_num(A, nan=0.0), -5, 5)
        return A.astype(np.float32)

    def state(self):
        return {"med": self.med.tolist(), "iqr": self.iqr.tolist(), "cols": self.cols.tolist()}


def fold_rows(st, track, test_start, test_end, h):
    """Training rows [a, b) (purged), test rows [s, e) as dev bar indices."""
    a_ts, b_ts = FO.train_range(track, test_start, h)
    return (st.idx(a_ts), st.idx(b_ts)), (st.idx(test_start), st.idx(test_end) if test_end <= FO.SEAL_START else st.n)
