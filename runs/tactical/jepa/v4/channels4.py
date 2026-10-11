"""v4 channels: v2's, with the context length (CTX bars) and the mask ratio as arguments. Context bars are the CTX bars before the
anchor; targets the 60 bars from the anchor on (3 segments of 20). No target bar is ever a context input (test_jepa4.py)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "v2"))
from channels2 import PATCH, SEG, N_SEG, context_channels_masked, segment_channels, block_masks as _bm  # noqa: F401,E402
from channels import N_CH, context_channels, future_channels, Standardiser  # noqa: F401,E402


def block_masks_ratio(rng, n, ratio, n_patch=12):
    """masks [n, n_patch]; k hidden patches, k in [round(r n)-1, round(r n)+1] (clipped to 1..n_patch-3), 1 or 2 blocks."""
    k0 = int(round(ratio * n_patch))
    return _bm(rng, n, max(1, k0 - 1), min(n_patch - 3, k0 + 1), n_patch)


def context_window(bars, idx, ctx):
    return bars[idx[:, None] + np.arange(-ctx, 0)]


def future_window(bars, idx, n=SEG * N_SEG):
    return bars[idx[:, None] + np.arange(n)]
