"""Synthetic indicators + geometry (SPEC.md): data, models, arms, metrics. CPU only; see SPEC.md for the design.

Everything that touches data goes through common/cutoff.py (no bar after 2025-07-20 23:59). TF is imported lazily so the
numpy parts (data, baselines) stay importable without it."""
import os, sys, time, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(TAC))
for sub in ("common", "jepa", "lab", "geom2"):
    sys.path.insert(0, os.path.join(TAC, sub))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cutoff as C  # noqa: E402
from channels import context_channels  # noqa: E402
import geomlib  # noqa: E402

LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
L = 60
HZ = (10, 15, 20)
DB = 5e-4
RV_H = 60
K_PRIMARY = 12
_SERIES = {}


# ---------------------------------------------------------------- data
def series():
    """OHLCV [T, 5] float64 and the bar open times, from the CUT file only."""
    if "A" not in _SERIES:
        import pandas as pd
        d = C.read_csv(usecols=["timestamp", "open", "high", "low", "close", "volume"])
        t = pd.to_datetime(d["timestamp"])
        C.assert_cut(t, "series")
        _SERIES["A"] = d[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
        _SERIES["t"] = pd.DatetimeIndex(t)
        lr = np.diff(np.log(_SERIES["A"][:, 3]), prepend=np.nan)
        _SERIES["lr"] = lr
        sq = np.nan_to_num(lr) ** 2
        _SERIES["cs"] = np.concatenate([[0.0], np.cumsum(sq)])          # cs[k] = sum of lr^2 over bars < k
    return _SERIES["A"], _SERIES["t"]


def rv_next(e_idx):
    """log realised volatility of the RV_H one-minute returns after bar e (bars e+1 .. e+RV_H); NaN if they pass the file end."""
    series(); cs = _SERIES["cs"]; e = np.asarray(e_idx)
    ok = e + RV_H < len(_SERIES["A"])
    out = np.full(len(e), np.nan)
    ee = e[ok]
    out[ok] = np.log(np.sqrt(cs[ee + RV_H + 1] - cs[ee + 1]) + 1e-9)
    return out


def locate(W0):
    """Index in the cut series of the first bar of window W0 [60, 5] (float32 cache window)."""
    A, _ = series(); Af = A.astype(np.float32)
    for i in np.flatnonzero(Af[: len(Af) - L + 1, 3] == W0[0, 3]):
        if np.array_equal(Af[i:i + L], W0):
            return int(i)
    raise LookupError("window not found in the cut series")


def tod_features(minute_of_day):
    a = 2 * np.pi * np.asarray(minute_of_day, float) / 1440.0
    return np.stack([f(k * a) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)


def har_features(W, minute_of_day):
    """HAR+tod baseline features from the window alone: log RV over the last 5 / 15 / 59 returns + time of day."""
    c = W[..., 3].astype(np.float64); lr = np.diff(np.log(c), axis=1)
    f = [np.log(np.sqrt(np.mean(lr[:, -k:] ** 2, 1)) + 1e-9) for k in (5, 15, 59)]
    return np.concatenate([np.stack(f, 1), tod_features(minute_of_day)], 1)


def make_inputs(W, stats=None):
    """[N, 60, 5] OHLCV -> [N, 60, 6] network input: the 5 scale-free channels (standardised) + the cumulative-return path / 5."""
    ch, _, _ = context_channels(W)
    p = np.cumsum(ch[..., 0], axis=1, dtype=np.float64)[..., None] / 5.0
    if stats is None:
        stats = (ch.reshape(-1, ch.shape[-1]).mean(0), ch.reshape(-1, ch.shape[-1]).std(0) + 1e-6)
    x = np.concatenate([(ch - stats[0]) / stats[1], p], -1).astype(np.float32)
    return np.clip(x, -10, 10), stats


def lab_slice(path):
    """One lab slice -> dict of train / val windows, close-entry returns [N, 3], deadband, vol labels, har features."""
    D = np.load(path); out = {"data_end": str(D["data_end"]), "db": float(D["deadband"])}
    A, t = series()
    for part in ("tr", "va"):
        W = D["W" + part]
        assert np.array_equal(W[1, :-1], W[0, 1:]), "lab windows must be consecutive (stride 1)"
        s = locate(W[0]); n = len(W)
        e = s + np.arange(n) + L - 1
        assert np.array_equal(A[e[-1], :].astype(np.float32), W[-1, -1]), "located block does not end where the cache ends"
        C.assert_cut(t[e[-1]:e[-1] + 1], path)
        out["W" + part] = W; out["r" + part] = D["r" + part]
        out["v" + part] = rv_next(e)
        mod = (t[e].hour * 60 + t[e].minute).to_numpy()
        out["h" + part] = har_features(W, mod)
    return out


class WF:
    """Monthly walk-forward data: anchors every 10 bars; window A[i-60:i]; next-open labels; vol labels from bars i .. i+59."""
    def __init__(self, start="2019-12-01", end="2024-07-01"):
        import pandas as pd
        A, t = series(); self.A, self.t = A, t
        i0 = int(t.searchsorted(pd.Timestamp(start))) + L
        i1 = int(t.searchsorted(pd.Timestamp(end)))
        self.i = np.arange(i0, i1 - RV_H - 1, 10)
        self.ts = t[self.i - 1]                                    # open time of the window's last bar
        self.sw = swv(A, L, axis=0)                                # [T-59, 5, 60]
        self.r = np.stack([A[self.i + h, 3] / A[self.i, 0] - 1 for h in HZ], 1)
        self.v = rv_next(self.i - 1)
        self.mod = (self.ts.hour * 60 + self.ts.minute).to_numpy()
        C.assert_cut(self.ts, "wf anchors")

    def windows(self, sel):
        return np.ascontiguousarray(self.sw[self.i[sel] - L].transpose(0, 2, 1)).astype(np.float32)

    def month_masks(self, m0, m1, train_days=365, train_stride=2):
        import pandas as pd
        tr = (self.ts >= m0 - pd.Timedelta(days=train_days)) & (self.ts < m0 - pd.Timedelta(minutes=RV_H + 1))
        te = (self.ts >= m0) & (self.ts < m1)
        tr = np.asarray(tr); te = np.asarray(te)
        k = np.flatnonzero(tr)[::train_stride]; tr2 = np.zeros_like(tr); tr2[k] = True
        return tr2, te


# ---------------------------------------------------------------- TF models
def tf_setup(threads=1):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def _classes():
    """Keras classes built on first use (so importing this module does not import TF)."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from neural_trade.models.layers.indicator_geometry import IndicatorGeometry

    class PairGeometry(layers.Layer):
        """Pairwise crossings between K synthetic series: asinh of the standardised difference at the window end and its smooth
        cross_bars-bar crossing. Reads bars of the window only; differentiable."""
        def __init__(self, k, cross_bars=5, **kw):
            super().__init__(**kw)
            self.k, self.cb = int(k), int(cross_bars)
            ii, jj = np.triu_indices(self.k, 1)
            self.ii, self.jj = tf.constant(ii), tf.constant(jj)
            self.norm = layers.LayerNormalization(name="pair_norm")

        def call(self, s):
            d = tf.gather(s, self.ii, axis=2) - tf.gather(s, self.jj, axis=2)          # [B, L, P]
            sd = tf.sqrt(tf.math.reduce_variance(d[:, 1:] - d[:, :-1], axis=1) + 1e-8)       # [B, P]
            z = d / tf.expand_dims(sd, 1)
            f1 = tf.math.asinh(z[:, -1]); sg = tf.tanh(z)
            f2 = (sg[:, -1] - sg[:, -1 - self.cb]) / 2.0
            return self.norm(tf.concat([f1, f2], -1))

        def get_config(self):
            c = super().get_config(); c.update({"k": self.k, "cross_bars": self.cb}); return c

    return tf, layers, IndicatorGeometry, PairGeometry


def build_generator(k=K_PRIMARY, hidden=24, dils=(1, 2, 4, 8, 16), seed=None):
    tf, layers, _, _ = _classes()
    if seed is not None:
        tf.random.set_seed(seed)
    inp = layers.Input((L, 6), name="x")
    h = inp
    for d in dils:
        h = layers.Conv1D(hidden, 3, dilation_rate=d, padding="causal", activation="tanh")(h)
    out = layers.Conv1D(k, 1)(h)
    skip = layers.Conv1D(k, 3, dilation_rate=4, padding="causal", use_bias=False)(inp)
    return tf.keras.Model(inp, layers.Add()([out, skip]), name="generator")


def geometry_features(series_t, x_t, k):
    """[B, 60, K] synthetic series and the [B, 60, 6] input (its last channel is the price path) -> [B, 5K + K(K-1)] features."""
    tf, layers, IG, PG = _classes()
    p = x_t[:, :, 5:6]
    g = IG(name="geometry")(tf.concat([series_t, p], -1))                 # [B, 5 (K + 1)]: blocks of K+1, the price is the last column
    g = tf.reshape(g, (-1, 5, k + 1))[:, :, :k]                          # drop the price's own features
    g = tf.reshape(g, (-1, 5 * k))
    return layers.Concatenate(name="geom_concat")([g, PG(k, name="pairs")(series_t)])


def build_net(gen, k, mode, n_out):
    """mode 'geom': geometry features -> readout; mode 'last': the K last values -> readout. Returns (net, feature_model)."""
    tf, layers, _, _ = _classes()
    x = layers.Input((L, 6), name="x")
    s = gen(x)
    f = geometry_features(s, x, k) if mode == "geom" else layers.Lambda(lambda t: t[:, -1, :], name="last_values")(s)
    z = layers.BatchNormalization(name="readout_bn")(f)
    out = layers.Dense(n_out, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="readout")(z)
    return tf.keras.Model(x, out), tf.keras.Model(x, f)


def _masked_bce(yt, lg):
    tf = _classes()[0]
    y, m = yt[:, :3], yt[:, 3:]
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=lg) * m) / (tf.reduce_sum(m) + 1.0)


def _mse(yt, p):
    tf = _classes()[0]
    return tf.reduce_mean(tf.square(yt[:, :1] - p))


def _best_restore(patience):
    tf = _classes()[0]

    class Best(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.best, self.w, self.wait, self.ep = np.inf, None, 0, 0

        def on_epoch_end(self, epoch, logs=None):
            v = logs["val_loss"]
            if v < self.best:
                self.best, self.w, self.wait, self.ep = v, self.model.get_weights(), 0, epoch + 1
            else:
                self.wait += 1
                if self.wait >= patience:
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.w is not None:
                self.model.set_weights(self.w)
    return Best()


def pack_dir(r, db=DB):
    """returns [N, 3] -> y/mask packed [N, 6]."""
    m = (np.abs(r) > db).astype(np.float32)
    return np.concatenate([(r > 0).astype(np.float32), m], 1)


def fit_e2e(Xtr, Ytr, target, mode, k, seed, max_epochs=12, batch=256, gen=None):
    """Train generator + (geometry or last values) + linear readout jointly. Ytr is pack_dir(...) for direction, [y_std, 1] for vol."""
    tf = _classes()[0]
    tf.keras.utils.set_random_seed(seed)
    gen = gen or build_generator(k)
    net, fm = build_net(gen, k, mode, 3 if target == "dir" else 1)
    net.compile(tf.keras.optimizers.Adam(2e-3), loss=_masked_bce if target == "dir" else _mse)
    n = len(Xtr); nv = int(0.15 * n); nf = n - nv - 80
    cb = _best_restore(3)
    net.fit(Xtr[:nf], Ytr[:nf], validation_data=(Xtr[n - nv:], Ytr[n - nv:]), epochs=max_epochs, batch_size=batch, verbose=0,
            callbacks=[cb], shuffle=True)
    return net, fm, cb.ep


def pretrain_generator(Xtr, k, seed, epochs=6, batch=256):
    """B: from s_t a linear decoder predicts the 5 channels of bar t+1 (MSE, inside the training windows)."""
    tf, layers, _, _ = _classes()
    tf.keras.utils.set_random_seed(seed)
    gen = build_generator(k)
    x = layers.Input((L, 6)); dec = layers.Dense(5, name="decoder")(gen(x)[:, :-1])
    m = tf.keras.Model(x, dec); m.compile(tf.keras.optimizers.Adam(2e-3), loss="mse")
    m.fit(Xtr, Xtr[:, 1:, :5], epochs=epochs, batch_size=batch, verbose=0, shuffle=True)
    return gen


def frozen_features(gen, k, mode, X, batch=4096):
    tf, layers, _, _ = _classes()
    x = layers.Input((L, 6)); s = gen(x)
    f = geometry_features(s, x, k) if mode == "geom" else s[:, -1, :]
    return tf.keras.Model(x, f).predict(X, batch_size=batch, verbose=0).astype(np.float32)


def predict(net, X):
    return net.predict(X, batch_size=4096, verbose=0)


# ---------------------------------------------------------------- readouts on fixed features and metrics
def fit_predict_dir(Ftr, rtr, Fva_list, C_=0.1, db=DB):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr); a = sc.transform(np.nan_to_num(Ftr))
    outs = [np.zeros((len(F), 3)) for F in Fva_list]
    for i in range(3):
        m = np.abs(rtr[:, i]) > db
        clf = LogisticRegression(C=C_, max_iter=1000).fit(a[m], (rtr[m, i] > 0).astype(int))
        for o, F in zip(outs, Fva_list):
            o[:, i] = clf.predict_proba(sc.transform(np.nan_to_num(F)))[:, 1]
    return outs


def fit_predict_vol(Ftr, ytr, Fva, alpha=30.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ok = np.isfinite(ytr); sc = StandardScaler().fit(Ftr[ok]); mu, sd = ytr[ok].mean(), ytr[ok].std()
    m = Ridge(alpha=alpha).fit(sc.transform(np.nan_to_num(Ftr[ok])), (ytr[ok] - mu) / sd)
    return m.predict(sc.transform(np.nan_to_num(Fva))) * sd + mu


def auc_dir(P, r, db=DB):
    from sklearn.metrics import roc_auc_score
    a = []
    for i in range(3):
        m = np.abs(r[:, i]) > db
        a.append(roc_auc_score((r[m, i] > 0).astype(int), P[m, i]))
    return float(np.mean(a)), [float(x) for x in a]


def vol_scores(pred, y, ytr_mean=None):
    from scipy.stats import spearmanr
    ok = np.isfinite(y)
    return {"spearman": float(spearmanr(pred[ok], y[ok])[0]), "r2": float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))}


# ---------------------------------------------------------------- all arms on one (train, val) block pair
def run_block(tr, va, seed, k=K_PRIMARY, arms=("tb7", "c3", "A", "B", "C1", "C2a", "C2b"), targets=("dir", "vol"), log=None):
    """tr / va: dicts with W [N,60,5], r [N,3] (labels), v [N] (log RV labels), h [N,F] (HAR+tod), minute-of-day inside h.
    Returns {arm: {target: metrics, 'sec': s}}. Direction metric: mean AUC; volatility: Spearman and R2."""
    from lab import fs_tb7
    res = {}
    def note(a, t0):
        res.setdefault(a, {})["sec"] = round(time.time() - t0, 1)
        if log:
            log(f"  {a} {res[a]}")
    Xtr, st = make_inputs(tr["W"]); Xva, _ = make_inputs(va["W"], st)
    okv = np.isfinite(tr["v"]); vmu, vsd = tr["v"][okv].mean(), tr["v"][okv].std()
    ytr_v = np.where(okv, (tr["v"] - vmu) / vsd, 0.0).astype(np.float32)

    def score_frozen(arm, Ftr, Fva, t0):
        if "dir" in targets:
            P = fit_predict_dir(Ftr, tr["r"], [Fva])[0]; res.setdefault(arm, {})["dir"] = auc_dir(P, va["r"])
        if "vol" in targets:
            res.setdefault(arm, {})["vol"] = vol_scores(fit_predict_vol(Ftr, tr["v"], Fva), va["v"])
        note(arm, t0)

    if "tb7" in arms or "har" in arms:
        t0 = time.time()
        if "dir" in targets:
            Ftr, Fva = fs_tb7(tr["W"]), fs_tb7(va["W"]); P = fit_predict_dir(Ftr, tr["r"], [Fva])[0]
            res.setdefault("tb7", {})["dir"] = auc_dir(P, va["r"])
        if "vol" in targets:
            res.setdefault("har", {})["vol"] = vol_scores(fit_predict_vol(tr["h"], tr["v"], va["h"], alpha=1.0), va["v"])
        note("tb7", t0)
    if "c3" in arms:
        t0 = time.time(); score_frozen("c3", geomlib.geom_rich(tr["W"]), geomlib.geom_rich(va["W"]), t0)
    if "C1" in arms:
        t0 = time.time(); acc = []
        for j in range(3):
            g = build_generator(k, seed=1000 * (seed + 1) + j)
            Ftr, Fva = frozen_features(g, k, "geom", Xtr), frozen_features(g, k, "geom", Xva)
            tmp = {}
            if "dir" in targets:
                tmp["dir"] = auc_dir(fit_predict_dir(Ftr, tr["r"], [Fva])[0], va["r"])[0]
            if "vol" in targets:
                tmp["vol"] = vol_scores(fit_predict_vol(Ftr, tr["v"], Fva), va["v"])
            acc.append(tmp)
        if "dir" in targets:
            res.setdefault("C1", {})["dir"] = [float(np.mean([a["dir"] for a in acc]))]
        if "vol" in targets:
            res.setdefault("C1", {})["vol"] = {m: float(np.mean([a["vol"][m] for a in acc])) for m in ("spearman", "r2")}
        note("C1", t0)
    if "B" in arms or "C2b" in arms:
        t0 = time.time(); gen = pretrain_generator(Xtr, k, seed); tp = round(time.time() - t0, 1)
        if "B" in arms:
            t1 = time.time(); score_frozen("B", frozen_features(gen, k, "geom", Xtr), frozen_features(gen, k, "geom", Xva), t1)
            res["B"]["sec_pretrain"] = tp
        if "C2b" in arms:
            t1 = time.time(); score_frozen("C2b", frozen_features(gen, k, "last", Xtr), frozen_features(gen, k, "last", Xva), t1)
    for arm, mode in (("A", "geom"), ("C2a", "last")):
        if arm not in arms:
            continue
        t0 = time.time()
        if "dir" in targets:
            net, _, ep = fit_e2e(Xtr, pack_dir(tr["r"]), "dir", mode, k, seed)
            P = 1 / (1 + np.exp(-predict(net, Xva))); res.setdefault(arm, {})["dir"] = auc_dir(P, va["r"]); res[arm]["epochs_dir"] = ep
        if "vol" in targets:
            net, _, ep = fit_e2e(Xtr, np.stack([ytr_v, np.ones_like(ytr_v)], 1), "vol", mode, k, seed)
            pv = predict(net, Xva)[:, 0] * vsd + vmu; res.setdefault(arm, {})["vol"] = vol_scores(pv, va["v"]); res[arm]["epochs_vol"] = ep
        note(arm, t0)
    return res
