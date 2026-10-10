"""CPU tests for v4. run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_jepa4.py"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import channels4 as C4
import pretrain4 as P4


def _bars(n=900, seed=0):
    r = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(r.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.0005; l = np.minimum(o, c) * 0.9995
    return np.stack([o, h, l, c, r.uniform(1, 10, n)], 1)


def test_no_future_in_any_context_input():
    """changing every bar at or after the anchor leaves the unmasked and the masked context channels unchanged (ctx 60 and 120)."""
    for ctx in (60, 120):
        bars = _bars()
        std = C4.Standardiser(np.zeros(5), np.ones(5)); pool = C4.block_masks_ratio(np.random.default_rng(0), 64, 0.5, ctx // 5)
        for ix in (300, 500):
            bb = bars.copy(); bb[ix:] = bb[ix:] * 1.5
            c = P4.make_batch(bb, np.array([ix]), std, ctx, pool, np.random.default_rng(3))
            a1 = P4.make_batch(bars, np.array([ix]), std, ctx, pool, np.random.default_rng(3))
            assert np.array_equal(c["ctx"], a1["ctx"]) and np.array_equal(c["mctx"], a1["mctx"])
            assert not np.array_equal(c["seg"], a1["seg"])        # the targets do see the future (sanity)


def test_mask_ratio_levels():
    for r in (0.3, 0.5, 0.7):
        m = C4.block_masks_ratio(np.random.default_rng(0), 4000, r, 12)
        assert abs(m.mean() - r) < 0.06, (r, m.mean())


def test_encoder_shapes_big_and_long():
    import model4 as M
    enc = M.Encoder4(5, d=64, heads=4, ffn=256, layers=3, max_tok=24)
    assert enc(np.zeros((2, 120, 5), np.float32)).shape == (2, 64)
    vis = np.ones((2, 24), bool); vis[:, 5:9] = False
    assert enc(np.zeros((2, 120, 5), np.float32), vis).shape == (2, 64)
    print("params", sum(int(np.prod(w.shape)) for w in enc.trainable_weights))


def test_finetune_head_equals_regression_at_init():
    """at init the fine-tune model's output equals the sklearn linear model on the tb features (direction: logit; volatility: Ridge)."""
    import finetune4 as F, model4 as M
    from sklearn.linear_model import LogisticRegression, Ridge
    r = np.random.default_rng(0); X = r.normal(size=(500, 7)).astype(np.float32); C = r.normal(size=(500, 60, 5)).astype(np.float32)
    y = (X[:, 0] + r.normal(size=500) > 0).astype(int)
    lg = LogisticRegression(C=0.1).fit(X, y)
    w = np.tile(lg.coef_.T, (1, 3)); b = np.repeat(lg.intercept_, 3)
    enc, lin, res, fwd = F.build(lambda: M.Encoder4(5, d=16, heads=2, ffn=32, layers=1, max_tok=12), 7, w, b)
    np.testing.assert_allclose(fwd(X, C).numpy()[:, 0], lg.decision_function(X), atol=1e-4)
    Y = r.normal(size=(500, 3)); rg = Ridge(alpha=1.0).fit(X, Y)
    enc, lin, res, fwd = F.build(lambda: M.Encoder4(5, d=16, heads=2, ffn=32, layers=1, max_tok=12), 7, rg.coef_.T, rg.intercept_)
    np.testing.assert_allclose(fwd(X, C).numpy(), rg.predict(X), atol=1e-4)
