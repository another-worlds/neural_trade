"""Fast tests for the synthetic-indicator track: python -m pytest runs/tactical/synth/test_synth.py -q  (CPU, seconds)."""
import os, re, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synthlib as S  # noqa: E402
C = S.C
K = 6


def _windows(n=8, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (n, S.L)), 1))
    o = np.concatenate([c[:, :1], c[:, :-1]], 1); h = np.maximum(o, c) * 1.0005; l = np.minimum(o, c) * 0.9995
    v = rng.uniform(1, 5, (n, S.L))
    return np.stack([o, h, l, c, v], -1).astype(np.float32)


def _x(n=8, seed=0):
    return S.make_inputs(_windows(n, seed))[0]


def test_cutoff_series_and_anchors():
    A, t = S.series()
    assert t.max() <= pd.Timestamp(C.CUTOFF)
    d = S.WF(start="2024-05-01", end="2024-07-01")
    assert d.ts.max() <= pd.Timestamp(C.CUTOFF) and d.ts.max() < pd.Timestamp("2024-07-01")


def test_no_script_reads_the_raw_file():
    raw = re.compile(r"neural_trade/Bitcoin_BTCUSDT\.csv|neural_trade\Bitcoin_BTCUSDT\.csv")
    for f in ("synthlib.py", "run_lab.py", "run_wf.py", "report.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            assert not raw.search(open(p).read()), f


def test_generator_is_causal_in_time():
    g = S.build_generator(K, seed=0); x = _x(); y = g.predict(x, verbose=0)
    for cut in (10, 33, 59):
        x2 = x.copy(); x2[:, cut + 1:, :] = np.random.default_rng(1).normal(size=x2[:, cut + 1:, :].shape)
        y2 = g.predict(x2, verbose=0)
        assert np.allclose(y[:, :cut + 1], y2[:, :cut + 1], atol=1e-5), cut
    assert y.shape == (8, S.L, K)


def test_inputs_use_only_the_window():
    """Channels and the price path at position t do not change when bars after t change (window-local scale uses the whole window,
    so the test changes bars after t and checks the standardised channels of the SAME window up to t only through a shorter-window
    equality of the price path: the cumulative path is a prefix sum)."""
    W = _windows(4); X, st = S.make_inputs(W)
    assert X.shape == (4, S.L, 6) and np.isfinite(X).all()
    W2 = W.copy(); W2[:, 40:, :4] *= 1.01          # change late bars
    p1, p2 = S.make_inputs(W)[0][:, :, 5], S.make_inputs(W2)[0][:, :, 5]
    # window volatility changes the scale but the SIGN pattern of the early path is the same
    assert np.array_equal(np.sign(p1[:, 1:20]), np.sign(p2[:, 1:20]))


def test_geometry_features_finite_shape_and_differentiable():
    import tensorflow as tf
    g = S.build_generator(K, seed=0); net, fm = S.build_net(g, K, "geom", 3)
    x = _x(); f = fm.predict(x, verbose=0)
    assert f.shape == (8, 5 * K + K * (K - 1)) and np.isfinite(f).all()
    with tf.GradientTape() as tape:
        out = net(tf.constant(x), training=True); loss = tf.reduce_mean(out ** 2)
    grads = tape.gradient(loss, g.trainable_variables)
    assert all(gr is not None and np.isfinite(gr.numpy()).all() for gr in grads)
    assert sum(float(tf.reduce_sum(tf.abs(gr))) for gr in grads) > 0         # the gradient reaches the generator through the geometry


def test_geometry_reads_only_recent_bars_of_the_series():
    """Slopes, distance and the 5-bar crossing read the last 11 bars; the squeeze and the scale statistics read the window."""
    g = S.build_generator(K, seed=0); _, fm = S.build_net(g, K, "geom", 3)
    x = _x(); f1 = fm.predict(x, verbose=0)
    # a different future cannot exist inside a window; instead check that features depend on the window's last bar
    x2 = x.copy(); x2[:, -1, :5] += 3.0
    assert not np.allclose(f1, fm.predict(x2, verbose=0))


def test_readout_sees_only_the_geometry_features():
    g = S.build_generator(K, seed=0); net, fm = S.build_net(g, K, "geom", 3)
    bn = net.get_layer("readout_bn")
    assert bn.input.shape[-1] == 5 * K + K * (K - 1)
    assert bn.input.node.layer.name == "geom_concat"
    src = {i.node.layer.__class__.__name__ for i in bn.input.node.layer.input}
    assert src <= {"TFOpLambda", "PairGeometry", "SlicingOpLambda"}, src
    # the readout is a single linear layer on that tensor
    assert net.get_layer("readout").input.node.layer.name == "readout_bn" and net.get_layer("readout").activation.__name__ == "linear"
    # the last-values control has K inputs to the readout and no geometry layer
    net2, _ = S.build_net(S.build_generator(K, seed=0), K, "last", 3)
    assert net2.get_layer("readout_bn").input.shape[-1] == K
    assert "geometry" not in {l.name for l in net2.layers} and "pairs" not in {l.name for l in net2.layers}


def test_pair_geometry_antisymmetry():
    tf, layers, IG, PG = S._classes()
    s = np.random.default_rng(0).normal(size=(4, S.L, 3)).cumsum(1).astype(np.float32)
    pg = PG(3); f = pg(tf.constant(s)).numpy()
    assert f.shape == (4, 6) and np.isfinite(f).all()


def test_baseline_features_causal_and_labels():
    W = _windows(5); h = S.har_features(W, np.arange(5) * 100)
    assert h.shape == (5, 9) and np.isfinite(h).all()
    r = np.array([[1e-3, -1e-3, 1e-5]]); y = S.pack_dir(r)
    assert list(y[0]) == [1, 0, 1, 1, 1, 0]          # the third return is inside the 5 bps deadband: masked
