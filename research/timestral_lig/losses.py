"""Losses on symmetric-averaged quantiles q (B, 15, 21) in log-bps from the last close; targets y (B, 15).

L = L_pin + lam_Q L_QL + sum_j lam_j L_j (geometry) + gate penalty + period prior.
L_pin: pinball over the 21 levels, divided by s_tk = sg(sigma_t) sqrt(k) (a positive weight known at t, so
each quantile score stays strictly consistent) and by its zero-shot Train mean P0_k (so L_pin starts at ~1).
L_QL: per-origin QLIKE y^2/v + ln v of the variance implied by the 21 quantiles with metrics2's midpoint
weights, minus constants (ln v0 Train mean per step, and 1); constants do not change gradients.
Geometry terms act on predictions (neural_trade's trend and coherence terms used only targets):
  fan   widths must not shrink with k (soft hinge, T = 0.005)
  term  log-log term structure of widths stays straight (kinks penalised)
  tail  tail ratio W98/W50 must not rise with k by more than 2% (aggregational Gaussianity)
  mcurv median path must not bend (neural_trade's cross-horizon coherence, rebuilt on predictions)
  mrw   random-walk prior on the median (no model shows price or direction skill)
"""
import math

import numpy as np
import torch

LV = torch.tensor((0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99), dtype=torch.float64)
_E = torch.cat([torch.zeros(1, dtype=torch.float64), (LV[1:] + LV[:-1]) / 2, torch.ones(1, dtype=torch.float64)])
W_MEAN = torch.diff(_E)
OMEGA = torch.full((15,), 1 / 24, dtype=torch.float64)
OMEGA[[0, 4, 14]] = 1 / 6
SQK = torch.arange(1, 16, dtype=torch.float64).sqrt()
INTERVALS = ((5, 15), (2, 18), (0, 20))          # 50%, 80%, 98%
T_SOFT = 0.005
GEOM = ("fan", "term", "tail", "mcurv", "mrw")
TERMS = ("QL",) + GEOM


def to_logbps(Q, close_T):
    """Price quantiles (B, 15, 21) -> log-bps from the last close (float64)."""
    return 1e4 * (torch.log(Q) - torch.log(close_T)[:, None, None])


def pinball_s(q, y, s):
    """(B, 15): mean over levels of the pinball loss divided by s sqrt(k)."""
    d = y[..., None] - q
    pb = torch.maximum(LV * d, (LV - 1) * d).mean(-1)
    return pb / (s[:, None] * SQK)


def qlike(q, y):
    """(B, 15): y^2 / v + ln v with v the midpoint-weight variance of the quantiles."""
    m = q @ W_MEAN
    v = torch.clamp((q ** 2) @ W_MEAN - m ** 2, min=1e-6)
    return y ** 2 / v + torch.log(v)


def geometry(q, s):
    lw = torch.stack([torch.log(torch.clamp(q[..., hi] - q[..., lo], min=1e-6)) for lo, hi in INTERVALS], -1)
    out = {}
    out["fan"] = (T_SOFT * torch.nn.functional.softplus((lw[:, :-1] - lw[:, 1:]) / T_SOFT)).mean()
    k = torch.arange(1, 15, dtype=torch.float64)
    sl = (lw[:, 1:] - lw[:, :-1]) / torch.log((k + 1) / k)[None, :, None]
    out["term"] = ((sl[:, 1:] - sl[:, :-1]) ** 2).mean()
    kap = lw[..., 2] - lw[..., 0]
    out["tail"] = (T_SOFT * torch.nn.functional.softplus((kap[:, 1:] - kap[:, :-1] - math.log(1.02)) / T_SOFT)).mean()
    m = q[..., 10]
    stk = s[:, None] * SQK
    out["mcurv"] = (((m[:, 2:] - 2 * m[:, 1:-1] + m[:, :-2]) / stk[:, 1:-1]) ** 2).mean()
    out["mrw"] = ((m / stk) ** 2).mean()
    return out


class Objective:
    """Holds the fixed constants (from zero-shot on Train) and the governed loss weights."""

    def __init__(self, P0, lnv0, lam0, geom_off=False):
        self.P0 = torch.as_tensor(P0, dtype=torch.float64)
        self.lnv0 = torch.as_tensor(lnv0, dtype=torch.float64)
        self.lam0 = dict(lam0)
        if geom_off:
            for j in GEOM:
                self.lam0[j] = 0.0
        self.loglam = {j: math.log(v) for j, v in self.lam0.items() if v > 0}
        self.ema = {j: 0.0 for j in self.loglam}
        self.history = []

    def lam(self, j):
        return math.exp(self.loglam[j]) if j in self.loglam else 0.0

    def bounds(self, j):
        return (0.025, 1.0) if j == "QL" else (self.lam0[j] / 10, self.lam0[j] * 10)

    def terms(self, q, y, s):
        t = {"pin": ((pinball_s(q, y, s).mean(0) / self.P0) * OMEGA).sum(),
             "QL": (((qlike(q, y) - self.lnv0).mean(0) - 1) * OMEGA).sum()}
        t.update(geometry(q, s))
        return t

    def total(self, t):
        return t["pin"] + sum(self.lam(j) * t[j] for j in self.loglam)

    def govern(self, cos):
        """Gradient alignment with held-out pinball: ln lam_j += 0.1 EMA_0.9(cos_j), clamped."""
        for j, c in cos.items():
            if j not in self.loglam:
                continue
            self.ema[j] = 0.9 * self.ema[j] + 0.1 * c
            lo, hi = self.bounds(j)
            self.loglam[j] = float(np.clip(self.loglam[j] + 0.1 * self.ema[j], math.log(lo), math.log(hi)))
        self.history.append({j: round(self.lam(j), 5) for j in self.loglam} | {f"cos_{j}": round(c, 4) for j, c in cos.items()})

    def state(self):
        return {"loglam": dict(self.loglam), "ema": dict(self.ema), "lam0": dict(self.lam0)}

    def load(self, st):
        self.loglam, self.ema, self.lam0 = dict(st["loglam"]), dict(st["ema"]), dict(st["lam0"])


# ---------------------------------------------------------------- evaluation metrics (as metrics2 / matched.py)
DEC9 = [2, 4, 6, 8, 10, 12, 14, 16, 18]


def eval_losses(q, y):
    """Per-origin pinball21 (log-bps) and QLIKE score at h = 1, 5, 15 (numpy, (N, 3) each), plus unit-free QLIKE."""
    q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
    lv = LV.numpy()
    wm = W_MEAN.numpy()
    pb, ql, uf = [], [], []
    for k in (0, 4, 14):
        d = y[:, k, None] - q[:, k]
        pb.append(np.maximum(lv * d, (lv - 1) * d).mean(1))
        m = q[:, k] @ wm
        v = np.maximum((q[:, k] ** 2) @ wm - m ** 2, 1e-6)
        ql.append(y[:, k] ** 2 / v + np.log(v))
        r = y[:, k] ** 2 / v
        nz = y[:, k] != 0
        uf.append(np.where(nz, r - np.log(np.where(nz, r, 1)) - 1, np.nan))
    return np.stack(pb, 1), np.stack(ql, 1), np.stack(uf, 1)


def agg_vs_base(q, q0, y):
    """S = mean over h of 0.5 (dQLIKE% + dPinball21%) vs base q0; per-origin aggregate differences (in %)."""
    pb, ql, _ = eval_losses(q, y)
    pb0, ql0, uf0 = eval_losses(q0, y)
    den_pb = pb0.mean(0)
    den_ql = np.abs(np.nanmean(uf0, 0))
    d = 100 * 0.5 * ((ql - ql0) / den_ql + (pb - pb0) / den_pb)        # (N, 3)
    cells = {"pin21_pct": (100 * (pb.mean(0) - pb0.mean(0)) / den_pb).round(4).tolist(),
             "qlike_pct": (100 * (ql.mean(0) - ql0.mean(0)) / den_ql).round(4).tolist()}
    return float(d.mean()), d.mean(1), cells


def block_ci(d, block, nboot=1000, seed=0):
    rs = np.random.default_rng(seed)
    n = len(d)
    st = rs.integers(0, n - block + 1, size=(nboot, int(np.ceil(n / block))))
    idx = (st[:, :, None] + np.arange(block)).reshape(nboot, -1)[:, :n]
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)
