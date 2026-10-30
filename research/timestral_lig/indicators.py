"""Learnable indicator and geometry bank: 62 unitless features in 11 gated groups, 40 learnable periods.

Base series per context row (float64, causal): u_t = 1e4 (c_t - c_{t-1}) / |c_{t-1}| (simple return in bps,
u_0 = 0), y = cumsum(u) (path in bps), sig_t = sqrt(EMA60_adj(u^2)) floored at 0.5 bps (the unit of most
features). EMA_P uses alpha = 2 / (P + 1) and the kernel alpha (1 - alpha)^j. EMAs of y start at y_0 = 0
(a pure causal convolution, so nested MACD signals use one composite kernel); EMAs of u-series use
adjust=True (conv(s, w) / conv(1, w)). Two evaluation modes: 'last' (origin only, a matmul with the kernels)
and 'seq' (every step, an exact blocked EMA scan with triangular matrices: a step never touches a later
value, not even through floating-point roundoff, which an FFT convolution would allow).

Fixes by design vs neural_trade LearnableIndicators: no x20 forward logit multiplier (D1) and no asinh
write-back of momentum periods (D7). Periods are P = 1 + exp(theta) (spans) or 1 + softplus(theta) (lags),
initialised at the exact inverse of neural_trade's Config values.
Every feature is odd or even under x -> -x; the mirror row of symmetric averaging is F * PARITY.
"""
import math

import numpy as np
import torch
from torch import nn

F64 = torch.float64
N = 2048

# (name, initial value, kind) for the 40 learnable scalars; kind 'span' (EMA) or 'lag' (momentum)
PERIODS = ([("g1_ema", p, "span") for p in (5, 15, 30)]
           + [(f"g2_{w}{i}", p, "span") for i, s in enumerate(((15, 30, 5), (45, 60, 30), (30, 45, 15)))
              for w, p in zip("fsg", s)]
           + [(f"g3_{w}{i}", p, "span") for i, s in enumerate(((30, 60, 9), (15, 30, 9), (25, 45, 9)))
              for w, p in zip("fsg", s)]
           + [("g4_rsi", p, "span") for p in (9, 21, 30)]
           + [("g5_bb", p, "span") for p in (10, 20, 50)]
           + [("g6_mom", p, "lag") for p in (1, 2, 3)]
           + [("g8_vol", p, "span") for p in (5, 15, 60, 240, 1440)] + [("g8_vr", 240, "span")]
           + [("g9_er", p, "span") for p in (15, 60)] + [("g9_mom", 240, "span")]
           + [("g10_park", 30, "span")])
assert len(PERIODS) == 40

GROUPS = ["ema_dist", "macd", "macd_pairs", "rsi", "bollinger", "momentum", "trend_geom", "vol_geom",
          "path_shape", "range", "calendar"]
SIZES = [3, 9, 9, 3, 6, 3, 6, 7, 5, 2, 9]
GROUP_OF = torch.tensor(np.repeat(np.arange(11), SIZES))
_O, _E = -1.0, 1.0
PARITY = torch.tensor([_O] * 3 + [_O] * 9 + [_O] * 9 + [_O] * 3 + [_O, _E] * 3 + [_O] * 3 + [_O] * 6
                      + [_E] * 7 + [_E, _E, _O, _E, _E] + [_E] * 2 + [_E] * 9, dtype=F64)
assert PARITY.numel() == sum(SIZES) == 62
P_MIN, P_MAX = 1.5, 1440.0


def ema_kernels(P, n=N):
    """(k,) spans -> (k, n) float64 EMA kernels alpha (1 - alpha)^j; entries below 1e-30 set to 0."""
    a = 2.0 / (P + 1.0)
    w = a[:, None] * torch.exp(torch.arange(n, dtype=F64)[None, :] * torch.log1p(-a)[:, None])
    return torch.where(w < 1e-30, torch.zeros_like(w), w)


class Ops:
    """Evaluate causal operators at the origin ('last') or at every step ('seq')."""

    def __init__(self, mode, n=N):
        assert mode in ("last", "seq")
        self.mode, self.n = mode, n
        self._memo = {}

    def _prep(self, s):
        """Flipped series ('last' mode), computed once per series per forward."""
        key = id(s)
        if key not in self._memo:
            self._memo[key] = (s, s.flip(-1))
        return self._memo[key][1]

    def at(self, s):
        return s[..., -1] if self.mode == "last" else s

    def shift(self, s, k):
        """s at t - k (index clamped at 0), evaluated at the output steps."""
        if self.mode == "last":
            return s[..., max(self.n - 1 - k, 0)]
        if k == 0:
            return s
        return torch.cat([s[..., :1].expand(*s.shape[:-1], min(k, self.n)), s[..., :max(self.n - k, 0)]], -1)

    def lag(self, s, L):
        """s at t - L for a fractional lag L (linear interpolation; exact at integer lags)."""
        l0 = int(torch.floor(L.detach()).item())
        f = L - l0
        return self.shift(s, l0) * (1 - f) + self.shift(s, l0 + 1) * f

    def ema(self, s, P, adjust):
        """EMA of series s (B, n) for spans P (k,) -> (B, k) or (B, n, k)."""
        if self.mode == "seq":
            return ema_scan(s, P, adjust)
        K = ema_kernels(P, self.n)
        out = self._prep(s) @ K.T
        return out / K.sum(-1) if adjust else out

    def macd(self, y, Pf, Ps, Pg):
        """MACD line EMA_f - EMA_s and signal EMA_g(line) of a series that starts at 0."""
        if self.mode == "seq":
            line = (ema_scan(y, Pf, False) - ema_scan(y, Ps, False))[..., 0]
            return line, ema_scan(line, Pg, False)[..., 0]
        Kf, Ks, Kg = (ema_kernels(p, self.n) for p in (Pf, Ps, Pg))
        return (self._prep(y) @ (Kf - Ks).T)[..., 0], (self._prep(y) @ composite(Kg, Kf, Ks, self.n).T)[..., 0]


def ema_scan(s, P, adjust, b=32):
    """Exact causal EMA e_t = a s_t + (1 - a) e_{t-1}, e_{-1} = 0, at every step: (B, n), (k,) -> (B, n, k).

    Within blocks of b steps a lower-triangular Toeplitz matmul; across blocks the same on block-end values."""
    B, n = s.shape
    nb, k = n // b, P.numel()
    a = 2.0 / (P + 1.0)
    lr = torch.log1p(-a)
    i = torch.arange(b, dtype=F64)
    di = i[:, None] - i[None, :]
    L = torch.where(di >= 0, a[:, None, None] * torch.exp(di.clamp_min(0)[None] * lr[:, None, None]), 0.0)
    local = torch.einsum("bnj,kij->bkni", s.view(B, nb, b), L)
    m = torch.arange(nb, dtype=F64)
    dm = m[:, None] - m[None, :]
    M = torch.where(dm >= 0, torch.exp(dm.clamp_min(0)[None] * (b * lr)[:, None, None]), 0.0)
    E = torch.einsum("bkm,klm->bkl", local[..., -1], M)
    E_prev = torch.cat([torch.zeros(B, k, 1, dtype=F64), E[..., :-1]], -1)
    e = local + E_prev[..., None] * torch.exp((i + 1)[None, :] * lr[:, None])[None, :, None, :]
    e = e.reshape(B, k, n).transpose(1, 2)
    if adjust:
        e = e / -torch.expm1((torch.arange(1, n + 1, dtype=F64)[:, None]) * lr[None, :])
    return e


def composite(Kg, Kf, Ks, n=N):
    """Kernel of EMA_g(EMA_f - EMA_s), truncated to n (exact for series that start at 0)."""
    m = 2 * n
    return torch.fft.irfft(torch.fft.rfft(Kg, m) * torch.fft.rfft(Kf - Ks, m), m)[..., :n]


class Bank(nn.Module):
    def __init__(self, sigma_ref=1.0, frame_ref=1.0):
        super().__init__()
        init = []
        for _, p0, kind in PERIODS:
            init.append(math.log(p0 - 1) if kind == "span" else math.log(math.expm1(max(p0 - 1, 0.05))))
        self.register_buffer("theta0", torch.tensor(init, dtype=F64))
        self.theta = nn.Parameter(self.theta0.clone())
        self.is_lag = torch.tensor([k == "lag" for _, _, k in PERIODS])
        self.register_buffer("sigma_ref", torch.tensor(float(sigma_ref), dtype=F64))
        self.register_buffer("frame_ref", torch.tensor(float(frame_ref), dtype=F64))
        self.idx = {}
        for i, (name, _, _) in enumerate(PERIODS):
            self.idx.setdefault(name, []).append(i)

    def periods(self):
        span = torch.clamp(1 + torch.exp(self.theta), P_MIN, P_MAX * (1 + 1e-9))
        lag = 1 + nn.functional.softplus(self.theta)
        return torch.where(self.is_lag, lag, span)

    def prior(self):
        return ((self.theta - self.theta0) ** 2).sum()

    def forward(self, raw, mode="last"):
        """raw: dict of (B, n) float64 windows -> unnormalised features (B, 62) or (B, n, 62)."""
        o = Ops(mode, raw["close"].shape[-1])
        P = self.periods()
        p = lambda name: P[self.idx[name]]
        c = raw["close"]
        u = torch.zeros_like(c)
        u[:, 1:] = 1e4 * (c[:, 1:] - c[:, :-1]) / c[:, :-1].abs()
        y = torch.cumsum(u, -1)
        yt = o.at(y)
        u2 = u ** 2
        sig = torch.sqrt(o.ema(u2, torch.tensor([60.0], dtype=F64), True)[..., 0].clamp_min(0.25))
        feats = []
        # G1 EMA distance
        feats.append((yt[..., None] - o.ema(y, p("g1_ema"), False)) / (sig[..., None] * p("g1_ema").sqrt()))
        # G2 / G3 MACD line, signal, hist (normalised by sig * sqrt(slow span))
        for g in ("g2", "g3"):
            for i in range(3):
                den = sig * p(f"{g}_s{i}").sqrt()
                line, signal = o.macd(y, *(p(f"{g}_{w}{i}") for w in "fsg"))
                feats.append(torch.stack([line / den, signal / den, (line - signal) / den], -1))
        # G4 RSI as (RSI - 50) / 50
        gu = o.ema(torch.clamp(u, min=0), p("g4_rsi"), True)
        gd = o.ema(torch.clamp(-u, min=0), p("g4_rsi"), True)
        feats.append((gu - gd) / (gu + gd + 1e-6))
        # G5 Bollinger: %B - 0.5 (odd) and log band width in units of sig sqrt(P) (even)
        mu = o.ema(y, p("g5_bb"), False)
        sd = torch.sqrt(torch.clamp(o.ema(y ** 2, p("g5_bb"), False) - mu ** 2, min=0) + 1e-4)
        pb = (yt[..., None] - mu) / (4 * sd)
        lw = torch.log(sd / (sig[..., None] * p("g5_bb").sqrt()))
        feats.append(torch.stack([pb, lw], -1).flatten(-2))
        # G6 momentum with learnable fractional lags
        L = p("g6_mom")
        feats.append(torch.stack([(yt - o.lag(y, L[i])) / (sig * L[i].sqrt()) for i in range(3)], -1))
        # G7 trend geometry (neural_trade EXTENDED_TREND_PERIODS 1, 5, 15): slope and curvature, fixed
        g7 = [(yt - o.shift(y, kk)) / (sig * math.sqrt(kk)) for kk in (1, 5, 15)]
        g7 += [(yt - 2 * o.shift(y, kk) + o.shift(y, 2 * kk)) / (sig * math.sqrt(2 * kk)) for kk in (1, 5, 15)]
        feats.append(torch.stack(g7, -1))
        # G8 volatility geometry: log EWMA vol at 5 spans, log variance ratios at k = 5, 15
        vol = torch.sqrt(o.ema(u2, p("g8_vol"), True).clamp_min(1e-6))
        Pvr = p("g8_vr")
        den_vr = o.ema(u2, Pvr, True)[..., 0].clamp_min(1e-6)
        vr = [torch.log(o.ema(_shift_series(y, kk) ** 2, Pvr, True)[..., 0].clamp_min(1e-6) / (kk * den_vr))
              for kk in (5, 15)]
        feats.append(torch.cat([torch.log(vol / self.sigma_ref), torch.stack(vr, -1)], -1))
        # G9 path shape: efficiency ratios, realised skew and log kurtosis, Timestral frame
        er = o.ema(u, p("g9_er"), True).abs() / (o.ema(u.abs(), p("g9_er"), True) + 1e-6)
        Pm = p("g9_mom")
        m2 = o.ema(u2, Pm, True)[..., 0].clamp_min(1e-6)
        skew = o.ema(u ** 3, Pm, True)[..., 0] / m2 ** 1.5
        kurt = torch.log(o.ema(u2 ** 2, Pm, True)[..., 0].clamp_min(1e-12) / m2 ** 2)
        frame = torch.log(1e4 * o.at(_running_std(c)) / o.at(c).abs() / self.frame_ref)
        feats.append(torch.cat([er, torch.stack([skew, kurt, frame], -1)], -1))
        # G10 range from OHLC (information the close-only backbone never sees)
        rho = 1e4 * torch.log(raw["high"] / raw["low"])
        park = torch.log(torch.sqrt(o.ema(rho ** 2, p("g10_park"), True)[..., 0].clamp_min(1e-6) / (4 * math.log(2))) / sig)
        rr = o.ema(rho, torch.tensor([30.0, 240.0], dtype=F64), True)
        feats.append(torch.stack([park, torch.log((rr[..., 0] + 1e-6) / (rr[..., 1] + 1e-6))], -1))
        # G11 calendar: UTC and New York clocks (the NY clock carries the DST change), weekend
        cal = []
        for key in ("utc_min", "ny_min"):
            m = o.at(raw[key])
            for j in (1, 2):
                cal += [torch.sin(2 * math.pi * j * m / 1440), torch.cos(2 * math.pi * j * m / 1440)]
        cal.append(o.at(raw["weekend"]))
        feats.append(torch.stack(cal, -1))
        return torch.cat(feats, -1)


def _shift_series(y, k):
    """Full series y_t - y_{t-k}, with y_{t-k} clamped at index 0."""
    return y - torch.cat([y[..., :1].expand(*y.shape[:-1], k), y[..., :-k]], -1)


def _running_std(c):
    """Causal running std of the context closes from the window start (Bessel), as Timestral's causal scaler."""
    x = c - c[..., :1]
    t = torch.arange(1, c.shape[-1] + 1, dtype=F64)
    s1, s2 = torch.cumsum(x, -1), torch.cumsum(x * x, -1)
    var = (s2 - s1 ** 2 / t) / torch.clamp(t - 1, min=1)
    return torch.sqrt(var.clamp_min(0)).clamp_min(1e-6 * c.abs())


def normalise(f):
    return torch.clamp(torch.asinh(f), -5, 5)
