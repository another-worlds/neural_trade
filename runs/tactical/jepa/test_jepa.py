import os, sys, glob
import numpy as np, pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "lab"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _bars(n=400, seed=0):
    r = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(r.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.0005; l = np.minimum(o, c) * 0.9995
    return np.stack([o, h, l, c, r.uniform(1, 5, n)], 1)


def test_context_channels_and_embedding_ignore_the_future():
    import tensorflow as tf
    from channels import context_channels, N_CH
    import model as M
    bars = _bars(); a = 200
    b2 = bars.copy(); b2[a:] *= np.random.default_rng(1).uniform(0.5, 2.0, (len(bars) - a, 5))   # rewrite everything from the anchor on
    c1, _, _ = context_channels(bars[None, a - 60:a]); c2, _, _ = context_channels(b2[None, a - 60:a])
    assert np.array_equal(c1, c2)
    tf.random.set_seed(0); enc = M.Encoder(N_CH); z1 = enc(c1).numpy(); z2 = enc(c2).numpy()
    assert np.array_equal(z1, z2)


def test_future_channels_start_from_the_context_close():
    from channels import context_channels, future_channels
    bars = _bars(); a = 200
    _, vol, vm = context_channels(bars[None, a - 60:a])
    f = future_channels(bars[None, a:a + 15], bars[None, a - 1, 3], vol, vm)
    assert np.isclose(f[0, 0, 0], np.log(bars[a, 3] / bars[a - 1, 3]) / vol[0, 0], rtol=1e-4)


def test_ema_update_is_the_exponential_average():
    import tensorflow as tf
    import model as M
    online, target = M.Encoder(5), M.Encoder(5)
    x = tf.zeros((2, 60, 5)); online(x); target(tf.zeros((2, 15, 5)))
    for w in target.weights: w.assign(tf.zeros_like(w))
    for w in online.weights: w.assign(tf.ones_like(w))
    M.ema_update(target, online, 0.9)
    assert all(np.allclose(w.numpy(), 0.1) for w in target.weights)
    M.ema_update(target, online, 0.9)
    assert all(np.allclose(w.numpy(), 0.19) for w in target.weights)


def test_vicreg_flags_a_collapsed_batch_and_accepts_a_spread_one():
    import tensorflow as tf
    import model as M
    var, _ = M.vicreg(tf.ones((64, M.D)) * 0.3)
    assert float(var) > 0.9
    var, cov = M.vicreg(tf.random.stateless_normal((512, M.D), seed=[1, 2]))
    assert float(var) < 0.05 and float(cov) < 0.2


def test_probe_pipeline_reproduces_sklearn_on_a_slice():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import lab, probe
    p = "D:/nt/nt_tactical/runs/tactical/lab/cache/2020-03-30T04.npz"
    if not os.path.exists(p): pytest.skip("lab cache not on this machine")
    D = np.load(p); db = float(D["deadband"]); n = 3000
    Ft, Fv = np.nan_to_num(lab.fs_tb7(D["Wtr"][:n])), np.nan_to_num(lab.fs_tb7(D["Wva"][:2000]))
    P = probe.logit_probs(*probe.standardise(Ft, Fv), D["rtr"][:n], db)
    sc = StandardScaler().fit(Ft); r = D["rtr"][:n, 1]; m = np.abs(r) > db
    ref = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(Ft)[m], (r[m] > 0).astype(int)).predict_proba(sc.transform(Fv))[:, 1]
    assert np.allclose(P[:, 1], ref)
    rv = D["rva"][:2000, 1]; mv = np.abs(rv) > db
    assert np.isclose(roc_auc_score(rv[mv] > 0, P[mv, 1]), roc_auc_score(rv[mv] > 0, ref[mv]))
