"""Per-bar scale-free channels. Window-local: every scale (volatility, mean volume) comes from the CONTEXT window itself
(bars up to its last bar), so a window's channels never depend on the future and cached windows need no file lookup.
The future segment (pretraining target) is normalised with the context's scale, and its first return starts from the
context's last close."""
import numpy as np

N_CH = 5
CTX, FUT = 60, 15
EPS = 1e-9


def _channels(o, h, l, c, v, prev_c, vol, vmean):
    """arrays [B, T]; prev_c [B, T] previous close of every bar; vol, vmean [B, 1]."""
    ret = np.log(c / prev_c) / vol
    rng = np.log(np.maximum(h, l + EPS) / np.maximum(l, EPS)) / vol
    body = (c - o) / (h - l + EPS * c)
    pos = (c - l) / (h - l + EPS * c) - 0.5
    lv = np.log1p(v / (vmean + EPS))
    return np.stack([ret, rng, body, pos, lv], -1).astype(np.float32)


def context_channels(W):
    """W [B, 60, 5] raw OHLCV -> [B, 60, N_CH] raw (not yet standardised) channels, and the scales (vol, vmean) [B, 1]."""
    W = W.astype(np.float64)
    o, h, l, c, v = (W[..., i] for i in range(5))
    prev = np.concatenate([o[:, :1], c[:, :-1]], 1)             # bar 0 starts from its own open
    lr = np.log(c / prev)
    vol = np.maximum(lr.std(1, keepdims=True), 1e-6)
    vmean = v.mean(1, keepdims=True)
    return _channels(o, h, l, c, v, prev, vol, vmean), vol, vmean


def future_channels(F, last_close, vol, vmean):
    """F [B, 15, 5] raw OHLCV of the next bars, normalised with the context's scales."""
    F = F.astype(np.float64)
    o, h, l, c, v = (F[..., i] for i in range(5))
    prev = np.concatenate([last_close.reshape(-1, 1), c[:, :-1]], 1)
    return _channels(o, h, l, c, v, prev, vol, vmean)


class Standardiser:
    def __init__(self, mean, std): self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
    def __call__(self, x): return np.clip((x - self.mean) / self.std, -8, 8).astype(np.float32)
    def state(self): return {"mean": self.mean.tolist(), "std": self.std.tolist()}
    @staticmethod
    def fit(x): x = x.reshape(-1, x.shape[-1]); return Standardiser(x.mean(0), x.std(0) + 1e-6)
