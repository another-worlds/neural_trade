"""CPU tests (seconds) for v3: nothing the trade filter uses at an entry bar reads a later bar. run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_tradefilter.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "v2")); sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tradefilter as TF  # noqa: E402

T, STRIDE = 15, 5


def _bars(n=4200, seed=0):
    r = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(r.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.0005; l = np.minimum(o, c) * 0.9995
    return np.stack([o, h, l, c, r.uniform(1, 5, n)], 1)


def _perturbed(A, t):
    B = A.copy(); r = np.random.default_rng(5)
    B[t + 1:] *= r.uniform(0.5, 2.0, (len(A) - t - 1, 5)); B[t + 1:, 1] = np.maximum(B[t + 1:, 1], B[t + 1:, :4].max(1))
    return B


def test_embedding_at_entry_bar_ignores_later_bars():
    import tensorflow as tf
    from channels import N_CH, Standardiser
    import model2 as M2
    tf.random.set_seed(0)
    enc = M2.Encoder2(N_CH); enc(np.zeros((2, 60, N_CH), np.float32))
    std = Standardiser(np.zeros(N_CH), np.ones(N_CH))
    A = _bars(); anch = TF.make_anchors(len(A), T, STRIDE); t = int(A.shape[0] * 0.7)
    B = _perturbed(A, t)
    e1, e2 = TF.embed_anchors(enc, std, A, anch), TF.embed_anchors(enc, std, B, anch)
    before = (anch - 1) <= t
    assert before.sum() > 50 and (~before).sum() > 50
    assert np.array_equal(e1[before], e2[before])
    assert not np.allclose(e1[~before], e2[~before])          # the perturbation is real


def test_causal_features_and_window_features_ignore_later_bars():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "lab"))
    import lab
    A = _bars(); anch = TF.make_anchors(len(A), T, STRIDE); t = int(A.shape[0] * 0.7); B = _perturbed(A, t)
    s1, v1 = TF.causal_features(A, anch); s2, v2 = TF.causal_features(B, anch)
    before = (anch - 1) <= t
    assert np.array_equal(s1[before], s2[before]) and np.array_equal(v1[before], v2[before])
    assert not np.allclose(v1[~before], v2[~before])
    f1, f2 = TF.window_features(A, anch, lab.fs_tb7), TF.window_features(B, anch, lab.fs_tb7)
    assert np.array_equal(f1[before], f2[before])


def test_har_label_reads_the_future_but_inputs_do_not():
    A = _bars(); anch = TF.make_anchors(len(A), T, STRIDE); t = int(A.shape[0] * 0.7); B = _perturbed(A, t)
    y1, y2 = TF.har_target(A, anch, T), TF.har_target(B, anch, T)
    last_full = (anch + T - 1) <= t
    assert np.array_equal(y1[last_full], y2[last_full]) and not np.allclose(y1[(anch - 1) <= t], y2[(anch - 1) <= t])


def test_folds_train_labels_end_before_test_entries():
    anch = TF.make_anchors(40000, T, STRIDE)
    for tr, te in TF.fold_splits(len(anch), STRIDE, T):
        assert anch[tr[-1]] + T - 1 < anch[te[0]] - 1 and tr[0] == 0 and len(te) > 100


def test_pnl_triple_barrier_conventions():
    # long, TP touched at bar 2 -> +barrier; short on the same path, stop touched -> -barrier; both in one bar -> stop
    barrier = np.array([0.01]); pnl = TF.make_pnl(T, barrier, np.array([2]), np.array([T + 1]), np.array([0.003]))
    assert pnl(np.array([1]), np.array([0]))[0] == 0.01 and pnl(np.array([-1]), np.array([0]))[0] == -0.01
    pnl = TF.make_pnl(T, barrier, np.array([3]), np.array([3]), np.array([0.003]))
    assert pnl(np.array([1]), np.array([0]))[0] == -0.01
    pnl = TF.make_pnl(T, barrier, np.array([T + 1]), np.array([T + 1]), np.array([0.003]))
    assert abs(pnl(np.array([-1]), np.array([0]))[0] + 0.003) < 1e-12
