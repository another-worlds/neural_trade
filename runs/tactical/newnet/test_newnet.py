"""Tests for the new network (CPU, seconds): python -m pytest runs/tactical/newnet/test_newnet.py -q -p no:cacheprovider"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lab"))
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import data as dm  # noqa: E402
import lab  # noqa: E402

HAVE_CACHE = os.path.exists(f"{dm.CACHE}/series_close.npy") and os.path.isdir(dm.LAB_CACHE)


def synthetic(n=1500, seed=0):
    r = np.random.RandomState(seed)
    c = 30000 * np.exp(np.cumsum(r.randn(n) * 5e-4))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(r.randn(n)) * 2e-4)
    l = np.minimum(o, c) * (1 - np.abs(r.randn(n)) * 2e-4); v = np.abs(r.randn(n)) * 10 + 1
    t = np.arange(n, dtype=np.float64)
    return np.stack([o, h, l, c, v], 1), t % 1440, (t // 1440) % 7


def test_features_are_causal():
    A, tod, dow = synthetic()
    v0, s0 = dm.bar_arrays(A, tod, dow, lab)
    t0 = 900
    B = A.copy(); rng = np.random.RandomState(1); B[t0 + 1:] *= rng.uniform(0.5, 2.0, (len(A) - t0 - 1, 5))
    v1, s1 = dm.bar_arrays(B, tod, dow, lab)
    assert np.array_equal(v0[:t0 + 1], v1[:t0 + 1]), "vec features at or before t0 changed when later bars changed"
    assert np.array_equal(s0[:t0 + 1], s1[:t0 + 1]), "seq features at or before t0 changed when later bars changed"
    assert not np.array_equal(v0[t0 + 1:], v1[t0 + 1:])
    assert v0.shape == (len(A), dm.N_VEC) and s0.shape == (len(A), dm.N_SEQ)


def test_rich_columns_equal_the_labs_window_features():
    A, tod, dow = synthetic()
    v, _ = dm.bar_arrays(A, tod, dow, lab)
    t = np.array([100, 400, 1000, 1499]); W = np.stack([A[i - 59:i + 1] for i in t]).astype(np.float32)
    ref = np.nan_to_num(lab.fs_rich(W))
    assert np.allclose(v[t, :dm.N_RICH].astype(np.float32), ref, rtol=2e-3, atol=2e-3)


def test_labels_and_windows_index_the_series():
    A, tod, dow = synthetic(400)
    vec, seq = dm.bar_arrays(A, tod, dow, lab)
    s = dm.Series(vec, seq, A[:, 3])
    r = s.labels(np.array([10])); i = 10 + 59
    assert np.allclose(r[0], [A[i + h, 3] / A[i, 3] - 1 for h in dm.HZ], rtol=1e-4, atol=1e-7)
    assert s.raw_seq(np.array([10])).shape == (1, 60, dm.N_SEQ)
    assert np.array_equal(s.raw_seq(np.array([10]))[0], seq[10:70].astype(np.float32))
    assert np.array_equal(s.raw_vec(np.array([10]))[0], vec[69].astype(np.float32))


@pytest.mark.skipif(not HAVE_CACHE, reason="series cache not built (python data.py --build)")
@pytest.mark.parametrize("name", ["2021-10-13T18", "2017-05-02T12"])
def test_val_block_equals_the_lab_cache(name):
    series = dm.Series(); A, _, _ = dm.load_series()
    for span in ("11.5d", "90d"):
        sp = dm.Split(name, span, series)
        assert sp.check_val_block(A)
        # the same gap as the cache and longtrain: the last training window starts 81 bars before the first val window
        assert sp.s0 + sp.n - 1 == sp.vs - dm.GAPB
        last = sp.s0 + sp.n - 1 + dm.WIN - 1 + max(dm.HZ)
        assert last < sp.vs, "a training label reaches into the val block"
        assert sp.itr[-1] + dm.INNER_GAP < sp.iva[0] + 1
    sp = dm.Split(name, "11.5d", series); D = np.load(f"{dm.LAB_CACHE}/{name}.npz")
    if sp.n == len(D["Wtr"]):                                   # the 11.5-day span is the cache's train block
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(A[sp.s0:sp.s0 + sp.n + 59], 60, axis=0).transpose(0, 2, 1).astype(np.float32)
        assert np.allclose(W, D["Wtr"], rtol=1e-5, atol=1e-6)
        assert np.allclose(series.labels(sp.tr_starts), D["rtr"], rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("arch", ["patch", "tcn"])
def test_zero_init_residual_equals_the_regression_and_param_count(arch):
    import tensorflow as tf
    import model as mdl
    rs = np.random.RandomState(0); coef = rs.randn(3, dm.N_VEC).astype(np.float32); b = rs.randn(3).astype(np.float32)
    net = mdl.build_direction(arch, dm.N_VEC); lin = mdl.build_direction("linear", dm.N_VEC)
    mdl.set_linear(net, coef, b); mdl.set_linear(lin, coef, b)
    x = rs.randn(32, dm.N_VEC).astype(np.float32); q = rs.randn(32, 60, dm.N_SEQ).astype(np.float32)
    assert np.allclose(net([x, q], training=False).numpy(), lin(x).numpy(), atol=1e-6)
    assert np.allclose(lin(x).numpy(), x @ coef.T + b, atol=1e-5)
    vol = mdl.build_volatility(dm.N_VEC, np.zeros(3))
    n = mdl.n_params(net) + mdl.n_params(vol) + 3
    print(arch, "params", mdl.n_params(net), mdl.n_params(vol), n)
    lo, hi = (20_000, 30_000) if arch == "patch" else (10_000, 30_000)
    assert lo <= n <= hi, n
    # the residual branch must be trainable and receive a gradient once the output layer moves off zero
    net.get_layer("residual").set_weights([rs.randn(32, 3).astype(np.float32) * 0.1, np.zeros(3, np.float32)])
    with tf.GradientTape() as t:
        loss = tf.reduce_sum(net([x, q], training=False) ** 2)
    g = t.gradient(loss, [v for v in net.trainable_variables if v.name.startswith("embed") or v.name.startswith("tcn_in")])
    assert all(gi is not None and float(tf.reduce_sum(tf.abs(gi))) > 0 for gi in g)


def test_volatility_gradient_does_not_reach_the_direction_branch():
    import tensorflow as tf
    import model as mdl
    net = mdl.build_direction("patch", dm.N_VEC); vol = mdl.build_volatility(dm.N_VEC, np.zeros(3))
    log_b = tf.Variable(np.zeros(3, np.float32))
    x = tf.constant(np.random.RandomState(0).randn(16, dm.N_VEC).astype(np.float32)); y = tf.constant(np.random.RandomState(1).randn(16, 3).astype(np.float32))
    with tf.GradientTape(persistent=True) as t:
        loss = tf.reduce_sum(mdl.vol_nll(y, vol(x), log_b))
    assert all(g is None for g in t.gradient(loss, net.trainable_variables))
    assert all(g is not None for g in t.gradient(loss, vol.trainable_variables))
    assert not {id(v) for v in net.variables} & {id(v) for v in vol.variables}
