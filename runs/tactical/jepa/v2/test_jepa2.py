"""CPU tests (seconds) for the v2 masking and target segments. run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_jepa2.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _bars(n=600, seed=0):
    r = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(r.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.0005; l = np.minimum(o, c) * 0.9995
    return np.stack([o, h, l, c, r.uniform(1, 5, n)], 1)


def _perturb_hidden(W, hid_patch, seed=3):
    W2 = W.copy(); r = np.random.default_rng(seed)
    for b in range(len(W)):
        for p in np.where(hid_patch[b])[0]:
            W2[b, p * 5:(p + 1) * 5] *= r.uniform(0.3, 3.0, (5, 5))          # rewrite every hidden bar completely
    return W2


def test_block_masks_hide_4_to_6_patches_in_at_most_two_blocks():
    from channels2 import block_masks
    m = block_masks(np.random.default_rng(0), 2000)
    assert set(m.sum(1)) <= {4, 5, 6} and set(m.sum(1)) == {4, 5, 6}
    runs = (np.diff(np.pad(m.astype(int), ((0, 0), (1, 0))), axis=1) == 1).sum(1)
    assert runs.max() <= 2 and runs.min() >= 1


def test_masked_channels_of_visible_bars_ignore_hidden_bars():
    from channels2 import block_masks, context_channels_masked
    bars = _bars(); a = 300
    W = np.stack([bars[a - 60 + i: a + i] for i in range(8)])
    hid = block_masks(np.random.default_rng(1), 8)
    c1 = context_channels_masked(W, hid); c2 = context_channels_masked(_perturb_hidden(W, hid), hid)
    vis = ~np.repeat(hid, 5, axis=1)
    assert np.array_equal(c1[vis], c2[vis])


def test_masked_channels_without_a_mask_equal_v1():
    from channels import context_channels
    from channels2 import context_channels_masked
    W = np.stack([_bars()[100 + i: 160 + i] for i in range(6)])
    v1, _, _ = context_channels(W)
    assert np.allclose(context_channels_masked(W, np.zeros((6, 12), bool)), v1, atol=1e-6)


def test_encoder_output_does_not_depend_on_hidden_patches_and_gets_no_gradient_from_them():
    import tensorflow as tf
    from channels2 import block_masks, context_channels_masked
    import model2 as M
    tf.random.set_seed(0)
    W = np.stack([_bars()[100 + i: 160 + i] for i in range(8)]); hid = block_masks(np.random.default_rng(2), 8)
    enc = M.Encoder2(5); enc(tf.zeros((2, 60, 5)))
    x1 = context_channels_masked(W, hid); x2 = context_channels_masked(_perturb_hidden(W, hid), hid)
    # the hidden tokens' own channels differ; the encoder output must not
    assert not np.array_equal(x1, x2)
    vis = tf.constant(~hid)
    o1, o2 = enc(x1, vis).numpy(), enc(x2, vis).numpy()
    assert np.array_equal(o1, o2)
    t1, t2 = enc.tokens_out(x1, vis).numpy(), enc.tokens_out(x2, vis).numpy()
    assert np.array_equal(t1[~hid], t2[~hid])
    x = tf.constant(x1)
    with tf.GradientTape() as g:
        g.watch(x); l = tf.reduce_sum(enc(x, vis))
    gr = g.gradient(l, x).numpy()
    assert np.all(gr[np.repeat(hid, 5, axis=1)] == 0) and np.abs(gr[~np.repeat(hid, 5, axis=1)]).max() > 0
    # and the mask matters: with everything visible the output differs from the masked one
    assert not np.allclose(enc(x1).numpy(), o1)


def test_target_segments_are_strictly_after_the_context_and_independent_of_it():
    from channels import context_channels
    from channels2 import segment_channels
    bars = _bars(); a = 300
    ctx = bars[None, a - 60:a]; fut = bars[None, a:a + 60]
    _, vol, vm = context_channels(ctx)
    s = segment_channels(fut, bars[None, a - 1, 3], vol, vm, 20)
    assert s.shape == (1, 3, 20, 5)
    # the first return of segment 0 starts from the context's last close (bar a-1) and ends at bar a
    assert np.isclose(s[0, 0, 0, 0], np.log(bars[a, 3] / bars[a - 1, 3]) / vol[0, 0], rtol=1e-4)
    # rewriting the context's bars except its last close leaves the segments unchanged given the same scales
    b2 = bars.copy(); b2[a - 60:a - 1] *= 1.7
    s2 = segment_channels(b2[None, a:a + 60], b2[None, a - 1, 3], vol, vm, 20)
    assert np.array_equal(s, s2)
    # and no segment bar index reaches back into the context
    idx = a + np.arange(60).reshape(3, 20)
    assert idx.min() >= a


def test_pretrain_batches_take_the_future_from_after_the_anchor():
    import pretrain2 as P
    from channels import Standardiser
    bars = _bars(800); std = Standardiser(np.zeros(5), np.ones(5))
    idx = np.array([200, 250])
    b = P.make_batch(bars, idx, std, P.SPEC["c"], np.zeros((4, 12), bool), np.random.default_rng(0))
    assert b["ctx"].shape == (2, 60, 5) and b["seg"].shape == (2, 3, 20, 5) and b["mctx"].shape == (2, 60, 5)
    bb = bars.copy(); bb[idx[0]:] *= 3.0                   # rewrite the future (from the anchor on)
    b2 = P.make_batch(bb, idx[:1], std, P.SPEC["c"], np.zeros((4, 12), bool), np.random.default_rng(0))
    assert np.array_equal(b["ctx"][0], b2["ctx"][0]) and np.array_equal(b["mctx"][0], b2["mctx"][0])
    assert not np.array_equal(b["seg"][0], b2["seg"][0])


def test_one_training_step_runs_for_every_variant_without_nan():
    import subprocess
    for v in ("ctl", "a", "b", "c"):
        r = subprocess.run([sys.executable, os.path.join(HERE, "pretrain2.py"), "--variant", v, "--steps", "500", "--bs", "32",
                            "--out", os.path.join(os.environ.get("TMP", "."), f"jepa2_t_{v}")], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        assert r.returncode == 0, r.stderr[-800:]
        assert "nan" not in r.stdout.lower()
