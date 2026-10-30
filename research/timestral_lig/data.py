"""Data for Timestral-LIG: series, leakage-safe Train / V1 / V2 / Test origins, batches.

Test origins are the benchmark origins (bench/common.py): every 3 min from the holdout start.
An origin t is the last context bar; its label uses closes t+1..t+15. Blocks are separated by
GAP = 16 (label purge) + 60 (embargo) bars, so every label of a block ends at least 60 bars before
the next block's first origin. V1 (1 day) only sets the loss weights; V2 (3 days) does early
stopping, promotion and gate reports; Test is scored once.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import torch

BENCH = os.environ.get("LIG_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tsfm_eval", "bench"))
CTX = 2048
PRED_LEN = 15
HORIZONS = (1, 5, 15)
GAP = PRED_LEN + 1 + 60
V1_LEN, V2_LEN = 1440, 3 * 1440
QLEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def ewma_sigma_bps(close, lam=0.94):
    """Causal EWMA standard deviation of 1-min log returns in bps at every index (uses r up to t), floored at 0.5."""
    r = np.zeros_like(close)
    r[1:] = np.diff(np.log(close)) * 1e4
    v = np.empty_like(r)
    v[0] = r[1] ** 2
    for i in range(1, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r[i] ** 2
    return np.maximum(np.sqrt(v), 0.5)


class Data:
    def __init__(self, dataset):
        os.environ["BENCH_DATA"] = dataset
        if BENCH not in sys.path:
            sys.path.insert(0, BENCH)
        import common
        importlib.reload(common)
        df, _ = common.load()
        self.dataset, self.bench_results = dataset, common.RESULTS
        self.results = common.RESULTS.replace("/results", "/results_lig")      # LIG Test npz never go to the bench dir
        self.close = df["close"].to_numpy(np.float64)
        self.high = df["high"].to_numpy(np.float64)
        self.low = df["low"].to_numpy(np.float64)
        ts = pd.to_datetime(df["datetime"], utc=True)
        ny = ts.dt.tz_convert("America/New_York")
        self.utc_min = (ts.dt.hour * 60 + ts.dt.minute).to_numpy(np.float64)
        self.ny_min = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.float64)
        self.weekend = (ts.dt.dayofweek >= 5).to_numpy(np.float64)
        self.lc = np.log(self.close)
        self.sigma = ewma_sigma_bps(self.close)
        self.test = common.origins(len(self.close))
        t0 = int(self.test[0])
        v2_end = t0 - GAP
        v2_start = v2_end - V2_LEN + 1
        v1_end = v2_start - GAP
        v1_start = v1_end - V1_LEN + 1
        tr_end = v1_start - GAP
        self.train = np.arange(CTX - 1, tr_end + 1)
        self.v1 = np.arange(v1_start, v1_end + 1)
        self.v2 = np.arange(v2_start, v2_end + 1)
        for a, b in ((self.train, self.v1), (self.v1, self.v2), (self.v2, self.test)):
            assert a[-1] + PRED_LEN <= b[0] - 60, "label of a block reaches into the next block's embargo"
        if os.environ.get("LIG_SMOKE"):                       # pipeline test only: few origins, other dirs
            self.train, self.v1, self.v2, self.test = self.train[::40], self.v1[::12], self.v2[::12], self.test[::12]
        if os.environ.get("LIG_RESULTS"):
            self.results = os.environ["LIG_RESULTS"] + ("" if dataset == "repo" else f"_{dataset}")

    def windows(self, origins, arr):
        return arr[origins[:, None] + np.arange(-CTX + 1, 1)[None, :]]

    def context(self, origins):
        """(B, CTX) float32 close prices (the Timestral input)."""
        return torch.tensor(self.windows(origins, self.close), dtype=torch.float32)

    def raw(self, origins):
        """float64 context windows for the indicator bank."""
        w = lambda a: torch.from_numpy(self.windows(origins, a))
        return {"close": w(self.close), "high": w(self.high), "low": w(self.low),
                "utc_min": w(self.utc_min), "ny_min": w(self.ny_min), "weekend": w(self.weekend)}

    def target(self, origins):
        """(B, 15) future log returns in bps and (B,) EWMA sigma in bps at the origin."""
        y = (self.lc[origins[:, None] + np.arange(1, PRED_LEN + 1)[None, :]] - self.lc[origins][:, None]) * 1e4
        return torch.tensor(y, dtype=torch.float64), torch.tensor(self.sigma[origins], dtype=torch.float64)

    def describe(self):
        r = lambda a: [int(a[0]), int(a[-1]), int(len(a))]
        return {"dataset": self.dataset, "train": r(self.train), "v1": r(self.v1), "v2": r(self.v2),
                "test": r(self.test)}
