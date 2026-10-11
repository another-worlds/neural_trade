"""Arrow-of-time study (SPEC.md): window channels, the four reversal transforms, the small NEEP-style 1D-CNN, helpers.
Hard cutoff: no bar after 2025-07-20 (LAST_BAR). Every bar loader here drops later rows before anything else sees them.
Channels of a 60-bar window (all scaled by the window's own realised volatility, a time-symmetric scalar):
  body = ln(c/o), up = ln(h / max(o,c)), dn = ln(min(o,c) / l), lv = ln(1+volume) - mean over the window, gap = ln(o_k / c_(k-1)).
A time reversal of a window reverses the bar order and swaps each bar's open and close: body -> -body, up and dn unchanged,
gaps negated and re-attached to the new neighbours."""
import os
import numpy as np
import pandas as pd

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
CACHE_IN = "D:/nt/nt_recent_cache"
CACHE = "D:/nt/nt_arrow_cache"
LAST_BAR = pd.Timestamp("2025-07-20 23:59:00")           # hard cutoff: no bar after 2025-07-20
L = 60
H = 15
DEADBAND = 5e-4
CLIP = 20.0


def load_bars():
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    s = s[s.index <= LAST_BAR]
    assert s.index.max() <= LAST_BAR
    return s


# ---------------------------------------------------------------- channels and transforms
def channels(W):
    """W [N, L, 5] (o, h, l, c, v) float64 -> X [N, L, 5] float32 (body, up, dn, lv, gap) and sigma [N] (window RV, unscaled).
    gap_k = ln(o_k / c_(k-1)) (0 for the first bar). sigma = sqrt(mean(body^2 + gap^2)): the close-to-close path's RMS, invariant
    under time reversal. Windows with sigma == 0 are marked by sigma == 0 (no information in their order); the caller drops them."""
    o, h, l, c, v = (W[..., i] for i in range(5))
    o = np.maximum(o, 1e-9); h = np.maximum(h, 1e-9); l = np.maximum(l, 1e-9); c = np.maximum(c, 1e-9)
    body = np.log(c / o)
    up = np.log(h / np.maximum(o, c)); dn = np.log(np.minimum(o, c) / l)
    gap = np.zeros_like(body); gap[:, 1:] = np.log(o[:, 1:] / c[:, :-1])
    sig = np.sqrt(np.mean(body * body + gap * gap, axis=1))
    lv = np.log1p(np.maximum(v, 0.0)); lv = lv - lv.mean(axis=1, keepdims=True)
    s = np.where(sig > 0, sig, 1.0)[:, None]
    X = np.stack([body / s, up / s, dn / s, lv, gap / s], axis=2)
    X = np.clip(X, -CLIP, CLIP)
    return X.astype(np.float32), sig


def to_abs(X):
    """sign-free channels for the |r| classifier: (|body|, up + dn, lv, |gap|)."""
    return np.stack([np.abs(X[..., 0]), X[..., 1] + X[..., 2], X[..., 3], np.abs(X[..., 4])], axis=-1)


def _reverse(X, gi, gap_sign):
    """bar order reversed; the gap channel (between a bar and the previous one) is re-attached to the new neighbours:
    reversed gap_j = gap_sign * gap_(k+1) of the original bar k that became bar j."""
    Y = X[:, ::-1].copy()
    Y[..., gi] = 0.0; Y[:, 1:, gi] = gap_sign * X[:, 1:, gi][:, ::-1]
    return Y


def t_full(X):
    """true time reversal: bar order reversed, open and close swapped (body negated, gaps negated and re-attached; wicks, volume unchanged)."""
    Y = _reverse(X, 4, -1.0); Y[..., 0] *= -1.0
    return Y


def t_sign(X):
    """sign only: the mirror image of the price path (body and gap negated, up and dn swapped), order kept."""
    Y = X.copy(); Y[..., 0] *= -1.0; Y[..., 4] *= -1.0; Y[..., 1], Y[..., 2] = X[..., 2].copy(), X[..., 1].copy()
    return Y


def t_order(X, gi=4):
    """order only: bar order reversed, no sign change (gap channel at index gi)."""
    return _reverse(X, gi, 1.0)


def t_order_abs(A):
    return t_order(A, gi=3)


MODELS = {              # name -> (input builder, transform, n channels)
    "full": (lambda X: X, t_full, 5),
    "sign": (lambda X: X, t_sign, 5),
    "order": (lambda X: X, t_order, 5),
    "abs": (to_abs, t_order_abs, 4),
}


# ---------------------------------------------------------------- anchors
def anchor_table():
    """The tb7 cache's anchors (1m, stride 10, open time t of the window's last bar) restricted to what the cutoff allows:
    the label bar (pos + H) must be <= the last bar. Returns bars (array, DatetimeIndex), pos, row index into the cache."""
    s = load_bars(); A = s.values.astype(np.float64)
    t = np.load(f"{CACHE_IN}/1m_t.npy")
    pos = s.index.get_indexer(pd.DatetimeIndex(t.astype("datetime64[ns]")))
    ok = (pos >= L - 1) & (pos + H < len(A))
    idx = np.nonzero(ok)[0]
    return s, A, pos[idx], idx


def forward_returns_nextopen(A, pos):
    """entry at the open of the bar after the anchor's last bar, exit at the close H bars after it (tb7's next-open label)."""
    return A[pos + H, 3] / A[pos + 1, 0] - 1


# ---------------------------------------------------------------- the model
def build_model(tf, n_ch, width=32):
    k = tf.keras
    x = inp = k.Input((L, n_ch))
    x = k.layers.Conv1D(width, 5, padding="same", activation="relu")(x)
    x = k.layers.Conv1D(width, 5, strides=2, padding="same", activation="relu")(x)
    x = k.layers.Conv1D(width, 3, strides=2, padding="same", activation="relu")(x)
    x = k.layers.Concatenate()([k.layers.GlobalAveragePooling1D()(x), k.layers.GlobalMaxPooling1D()(x)])
    x = k.layers.Dense(16, activation="relu")(x)
    out = k.layers.Dense(1)(x)
    return k.Model(inp, out)


def setup_gpu(mem_mb=3000):
    """below-normal priority, GPU capped at mem_mb, TF imported after neural_trade (CUDA DLLs)."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    import neural_trade  # noqa: F401
    import tensorflow as tf
    g = tf.config.list_physical_devices("GPU")
    if g:
        tf.config.set_logical_device_configuration(g[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)])
    return tf


def bce_logits(z, y):
    """numpy, stable: -[y log s(z) + (1-y) log(1-s(z))]."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
