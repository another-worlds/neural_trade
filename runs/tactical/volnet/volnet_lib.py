"""volnet: features, models and the monthly walk-forward step (SPEC.md). Anchor a sees bars < a only (last bar a-1)."""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lab"))
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
HORIZONS = (15, 60, 240)
STRIDE, FIRST, T_SEQ, PATCH = 10, 1441, 240, 10
TRAIN_DAYS, HOLD_FRAC = 365, 0.15
EPS_Y, EPS_V = 1e-5, 1e-10
HAR_NAMES = ["rv15", "rv60", "rv240", "rv1440", "tod_sin", "tod_cos"]


def load_bars():
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    return s


class Data:
    """Bars, anchors and per-anchor features. Everything at anchor a uses bars with index < a."""

    def __init__(self, A, index, horizons=HORIZONS, stride=STRIDE, first=FIRST, with_tb7=True):
        self.A, self.horizons = A, tuple(horizons)
        n = len(A); c, h, l, v = A[:, 3], A[:, 1], A[:, 2], A[:, 4]
        self.lr = np.concatenate([[0.0], np.diff(np.log(c))])
        self.cs2 = np.concatenate([[0.0], np.cumsum(self.lr ** 2)])
        self.anch = np.arange(first, n - max(horizons), stride)
        a = self.anch
        self.ta = index[a - 1] + pd.Timedelta(minutes=1)             # the time the last bar closes = when the row is available
        self.r = {H: c[a + H - 1] / c[a - 1] - 1 for H in horizons}
        self.y = np.column_stack([np.log(np.abs(self.r[H]) + EPS_Y) for H in horizons])
        self.absr = np.column_stack([np.abs(self.r[H]) for H in horizons])
        self.v2 = np.column_stack([self.r[H] ** 2 + EPS_V for H in horizons])
        rv = lambda k: np.log(np.maximum(self.cs2[a] - self.cs2[a - k], 0) / k + 1e-12)
        hr = index[a - 1].hour.values
        self.har = np.column_stack([rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
        cv = np.concatenate([[0.0], np.cumsum(v)]); rng = (h - l) / c; cr = np.concatenate([[0.0], np.cumsum(rng)])
        lvm = lambda k: np.log(np.maximum(cv[a] - cv[a - k], 0) / k + 1e-8)
        lrg = lambda k: np.log(np.maximum(cr[a] - cr[a - k], 0) / k + 1e-8)
        self.extra = np.column_stack([lvm(15) - lvm(1440), lvm(60) - lvm(1440), lvm(240) - lvm(1440), lrg(15), lrg(60), rv(5)])
        self.tb7 = None
        if with_tb7:
            from lab import fs_tb7
            self.tb7 = np.concatenate([np.nan_to_num(fs_tb7(np.stack([A[i - 60:i] for i in a[k:k + 20000]]).astype(np.float32)))
                                       for k in range(0, len(a), 20000)])
        # per-bar channels for the sequence model (return, log high/low, log1p volume)
        self.ch = np.stack([self.lr, np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9)), np.log1p(v)], 1).astype(np.float32)
        self._swv = np.lib.stride_tricks.sliding_window_view(self.ch, T_SEQ, axis=0)   # [n-T+1, 3, T]

    def vec_full(self):
        parts = [self.har, self.extra] + ([self.tb7] if self.tb7 is not None else [])
        return np.column_stack(parts).astype(np.float64)

    def seq(self, idx):
        """Scale-free channels for anchors `idx` (rows of self.anch): [N, T, 3]. Bars a-T..a-1 only."""
        a = self.anch[idx]
        w = np.transpose(self._swv[a - T_SEQ], (0, 2, 1)).copy()           # [N, T, 3]
        s = np.sqrt((w[:, :, 0] ** 2).mean(1)) + 1e-6
        w[:, :, 0] /= s[:, None]; w[:, :, 1] /= s[:, None]
        w[:, :, 2] -= w[:, :, 2].mean(1, keepdims=True)
        return w.astype(np.float32)


def month_masks(ta, m0, m1, H):
    """train: available a year before m0 and the label ends by m0; test: the month itself."""
    tr = (ta >= m0 - pd.Timedelta(days=TRAIN_DAYS)) & (ta + pd.Timedelta(minutes=H) <= m0)
    te = (ta >= m0) & (ta < m1)
    return np.asarray(tr), np.asarray(te)


def fit_hold(tr_idx, H_max):
    """chronological split of the training rows into fit (first 85%) and hold (last 15%); a gap of H_max minutes of rows
    (H_max / STRIDE rows) is dropped between them so no hold label overlaps a fit label."""
    n = len(tr_idx); k = int(n * (1 - HOLD_FRAC)); gap = int(np.ceil(H_max / STRIDE))
    return tr_idx[:k], tr_idx[k + gap:]


def qlike(v, mu, c):
    s = c * np.exp(2 * mu); q = v / s
    return float(np.mean(q - np.log(q) - 1))


def scale_c(v_hold, mu_hold):
    return float(np.mean(v_hold / np.exp(2 * mu_hold)))


def metrics(mu_te, mu_ho, absr_te, v_te, v_ho):
    c = scale_c(v_ho, mu_ho)
    return {"spearman": float(spearmanr(mu_te, absr_te)[0]), "qlike": qlike(v_te, mu_te, c), "c": c}


# ------------------------------------------------------------------ models
def get_tf():
    import neural_trade  # noqa: F401  (CUDA DLLs on PATH before tensorflow)
    import tensorflow as tf
    return tf


def setup_device(mem_mb=4000, cpu=False):
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf = get_tf()
    tf.config.threading.set_intra_op_parallelism_threads(3); tf.config.threading.set_inter_op_parallelism_threads(2)
    if cpu:
        return "cpu"
    try:
        g = tf.config.list_physical_devices("GPU")
        if g:
            tf.config.set_logical_device_configuration(g[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)])
            return "gpu"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable:", e, flush=True)
    return "cpu"


def ridge_fit(X, Y, alpha=1.0):
    """standardise on X, Ridge on all horizons at once; returns (mean, std, coef [nH, n], intercept [nH])."""
    from sklearn.linear_model import Ridge
    mean, std = X.mean(0), X.std(0) + 1e-12
    m = Ridge(alpha=alpha).fit((X - mean) / std, Y)
    return mean, std, m.coef_, m.intercept_


def build_net(kind, n_vec, n_har, nH=3):
    """kind 'mlp': inputs [vec]; kind 'seq': inputs [har, seq]. Output mu [B, nH] = linear skip + residual (last layer zeros)."""
    tf = get_tf(); L = tf.keras.layers
    vec = L.Input((n_vec,), name="vec"); skip = L.Dense(nH, name="skip")
    if kind == "mlp":
        h = L.Dense(96, activation="gelu")(vec); h = L.Dense(96, activation="gelu")(h)
        res = L.Dense(nH, kernel_initializer="zeros", bias_initializer="zeros", name="residual")(h)
        return tf.keras.Model([vec], L.Add()([skip(vec), res]), name="mlp")
    seq = L.Input((T_SEQ, 3), name="seq")
    d, heads, key, ff = 32, 4, 8, 64
    x = L.Reshape((T_SEQ // PATCH, PATCH * 3))(seq)
    x = L.Dense(d, name="embed")(x)
    x = _addpos_cls()(name="pos")(x)
    for i in range(2):
        hh = L.LayerNormalization(epsilon=1e-5)(x)
        hh = L.MultiHeadAttention(heads, key, dropout=0.1)(hh, hh)
        x = L.Add()([x, hh])
        hh = L.LayerNormalization(epsilon=1e-5)(x)
        hh = L.Dense(d)(L.Dense(ff, activation="gelu")(hh))
        x = L.Add()([x, hh])
    x = L.GlobalAveragePooling1D()(L.LayerNormalization(epsilon=1e-5)(x))
    x = L.Dropout(0.1)(L.Dense(16, activation="gelu")(x))
    res = L.Dense(nH, kernel_initializer="zeros", bias_initializer="zeros", name="residual")(x)
    return tf.keras.Model([vec, seq], L.Add()([skip(vec), res]), name="seq")


_POS = []


def _addpos_cls():
    if _POS:
        return _POS[0]
    tf = get_tf()

    class AddPos(tf.keras.layers.Layer):
        def build(self, shape):
            self.pos = self.add_weight("pos", (shape[1], shape[2]), initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

        def call(self, x):
            return x + self.pos
    _POS.append(AddPos)
    return AddPos


def n_params(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))


def set_skip(model, coef, intercept):
    model.get_layer("skip").set_weights([np.asarray(coef, np.float32).T, np.asarray(intercept, np.float32)])


def train_net(model, Xfit, Yfit, Xho, Yho, seed, max_epochs=25, patience=4, lr=2e-3, bs=512):
    """Gaussian NLL on log|r| with one learned log-sigma per horizon. Epoch 0 (the init = the ridge) is a candidate.
    Returns (history dict). Best-hold weights are restored."""
    tf = get_tf(); nH = Yfit.shape[1]
    tf.random.set_seed(seed); rs = np.random.RandomState(seed)
    ls = tf.Variable(tf.zeros(nH), name="log_sigma"); opt = tf.keras.optimizers.Adam(lr)
    vars_ = model.trainable_weights + [ls]

    def nll(y, mu):
        return tf.reduce_sum(tf.reduce_mean(0.5 * tf.square(y - mu) * tf.exp(-2 * ls) + ls, 0))

    @tf.function(reduce_retracing=True)
    def step(xs, y):
        with tf.GradientTape() as tp:
            loss = nll(y, model(xs, training=True))
        opt.apply_gradients(zip(tp.gradient(loss, vars_), vars_)); return loss

    def predict(Xs, bsz=4096):
        n = len(Xs[0]); return np.concatenate([model([x[i:i + bsz] for x in Xs], training=False).numpy() for i in range(0, n, bsz)])

    def hold_loss():
        mu = predict(Xho); return float(np.mean(np.square(Yho - mu)))        # MSE; sigma is a nuisance scale per horizon
    best, best_w, bad, hist = hold_loss(), model.get_weights(), 0, [hold_loss()]
    n = len(Xfit[0])
    for ep in range(max_epochs):
        perm = rs.permutation(n)
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]; step([x[b] for x in Xfit], Yfit[b].astype(np.float32))
        hl = hold_loss(); hist.append(hl)
        if hl < best - 1e-6:
            best, best_w, bad = hl, model.get_weights(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.set_weights(best_w)
    return {"hold_mse_by_epoch": hist, "best_epoch": int(np.argmin(hist)), "predict": predict}
