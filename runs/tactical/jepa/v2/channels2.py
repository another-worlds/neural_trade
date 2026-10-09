"""v2 channels: v1's per-bar channels (reused) plus (1) a masked-context variant that guarantees a hidden patch's raw bars
influence no visible token, and (2) multi-segment future targets.

Masking leaks to close in the channels (not only in the encoder):
  * the window scale (std of returns, mean volume) is computed from the VISIBLE bars only;
  * the return of a visible bar whose previous bar is hidden would read the hidden close: it is set to 0.
With no hidden bar the result equals v1's context_channels exactly (test_jepa2.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from channels import N_CH, CTX, EPS, _channels, context_channels, future_channels, Standardiser  # noqa: F401,E402

PATCH = 5
N_PATCH = CTX // PATCH      # 12
SEG, N_SEG = 20, 3          # v2 future: 3 segments of 20 bars


def context_channels_masked(W, hid_patch):
    """W [B,60,5] raw OHLCV, hid_patch [B,12] bool (True = hidden). Returns channels [B,60,N_CH] (hidden bars' channels are
    computed from their own bars but must never be fed: the encoder replaces those tokens)."""
    W = W.astype(np.float64)
    o, h, l, c, v = (W[..., i] for i in range(5))
    hid = np.repeat(hid_patch, PATCH, axis=1)                        # [B,60] hidden bar
    vis = ~hid
    prev = np.concatenate([o[:, :1], c[:, :-1]], 1)
    lr = np.log(c / prev)
    prev_vis = np.concatenate([np.ones_like(vis[:, :1]), vis[:, :-1]], 1)
    ok = vis & prev_vis                                              # return uses only visible bars
    n = np.maximum(ok.sum(1, keepdims=True), 1)
    mu = (lr * ok).sum(1, keepdims=True) / n
    vol = np.maximum(np.sqrt((((lr - mu) ** 2) * ok).sum(1, keepdims=True) / n), 1e-6)
    vmean = (v * vis).sum(1, keepdims=True) / np.maximum(vis.sum(1, keepdims=True), 1)
    ch = _channels(o, h, l, c, v, prev, vol, vmean)
    ch[..., 0] = np.where(vis & ~prev_vis, 0.0, ch[..., 0])          # first visible bar after a hidden one: return unknown
    return ch


def segment_channels(F, last_close, vol, vmean, seg=SEG):
    """F [B, n_seg*seg, 5] -> [B, n_seg, seg, N_CH], normalised with the context's scales, first return from the context close."""
    ch = future_channels(F, last_close, vol, vmean)
    return ch.reshape(len(F), -1, seg, ch.shape[-1])


def block_masks(rng, n, lo=4, hi=6, n_patch=N_PATCH):
    """n masks [n,12] bool; 4..6 of 12 patches (33-50%) hidden in 1 or 2 contiguous blocks."""
    out = np.zeros((n, n_patch), bool)
    for i in range(n):
        k_hide = int(rng.integers(lo, hi + 1))
        for _ in range(50):
            blocks = int(rng.integers(1, 3))
            lens = [k_hide] if blocks == 1 else [k_hide // 2, k_hide - k_hide // 2]
            m = np.zeros(n_patch, bool); ok = True
            for L in lens:
                s = int(rng.integers(0, n_patch - L + 1))
                if m[max(s - 1, 0): s + L + 1].any(): ok = False; break   # no overlap, no touching (so two blocks stay two)
                m[s: s + L] = True
            if ok and m.sum() == k_hide:
                out[i] = m; break
        else:
            s = int(rng.integers(0, n_patch - k_hide + 1)); out[i, s: s + k_hide] = True
    return out
