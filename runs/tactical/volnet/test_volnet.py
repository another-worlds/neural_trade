"""volnet tests (seconds, CPU): causal inputs, the walk-forward never trains on the scored month, the seq model equals the HAR ridge at init."""
import os, sys
import numpy as np
import pandas as pd
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import volnet_lib as V  # noqa: E402


def synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    v = np.abs(rng.normal(5, 2, n)); idx = pd.date_range("2022-01-01", periods=n, freq="1min")
    return np.column_stack([o, h, l, c, v]), idx


def test_inputs_are_causal():
    A, idx = synth(); D = V.Data(A, idx, horizons=(15, 60), first=300, stride=10)
    a0 = D.anch[200]                                   # corrupt every bar from a0 on: rows of anchors <= a0 must not change
    A2 = A.copy(); A2[a0:] *= np.random.default_rng(1).uniform(0.5, 2, (len(A) - a0, 1))
    D2 = V.Data(A2, idx, horizons=(15, 60), first=300, stride=10)
    k = 201                                            # anchors 0..200 have anch <= a0
    np.testing.assert_array_equal(D.har[:k], D2.har[:k]); np.testing.assert_array_equal(D.extra[:k], D2.extra[:k])
    np.testing.assert_array_equal(D.tb7[:k], D2.tb7[:k]); np.testing.assert_array_equal(D.seq(np.arange(k)), D2.seq(np.arange(k)))
    assert not np.array_equal(D.y[k:], D2.y[k:])       # the labels do change (they are the future)


def test_walk_forward_never_trains_on_the_scored_month():
    t = pd.date_range("2021-01-01", "2023-01-01", freq="10min") + pd.Timedelta(minutes=1)
    for H in V.HORIZONS:
        for m in pd.period_range("2022-01", "2022-12", freq="M"):
            m0, m1 = m.start_time, (m + 1).start_time
            tr, te = V.month_masks(t, m0, m1, H)
            assert tr.any() and te.any() and not (tr & te).any()
            assert (t[tr] + pd.Timedelta(minutes=H)).max() <= m0 and t[tr].max() < m0 and t[te].min() >= m0 and t[te].max() < m1
            assert t[tr].min() >= m0 - pd.Timedelta(days=V.TRAIN_DAYS)
            fit, ho = V.fit_hold(np.where(tr)[0], H)
            assert fit.max() < ho.min() and (ho.min() - fit.max()) * V.STRIDE >= H     # no label overlap between fit and hold


def test_seq_model_equals_har_ridge_at_init():
    A, idx = synth(); D = V.Data(A, idx, horizons=(15, 60, 240), first=1441, stride=10, with_tb7=False)
    ii = np.arange(len(D.anch)); mean, std, coef, ic = V.ridge_fit(D.har[ii[:400]], D.y[ii[:400]])
    net = V.build_net("seq", D.har.shape[1], D.har.shape[1], 3); V.set_skip(net, coef, ic)
    Z = (D.har - mean) / std; got = net([Z[ii[:300]], D.seq(ii[:300])], training=False).numpy()
    want = Z[:300] @ coef.T + ic
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert 15_000 < V.n_params(net) < 35_000, V.n_params(net)
