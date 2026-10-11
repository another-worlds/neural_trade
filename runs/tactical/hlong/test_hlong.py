"""Seconds. Inputs are causal; the walk-forward never trains on the scored month; the nets equal the linear model at init."""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hfeats as F  # noqa: E402
import hwf  # noqa: E402


def _bars(n=1500, seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * (1 + rng.uniform(0, .002, n)); l = np.minimum(o, c) * (1 - rng.uniform(0, .002, n))
    v = rng.uniform(1, 5, n); idx = pd.date_range("2020-01-01", periods=n, freq="60min")
    return np.stack([o, h, l, c, v], 1), idx.hour.to_numpy(float), idx.dayofweek.to_numpy(float)


def test_features_at_a_bar_do_not_change_when_later_bars_change():
    A, hr, dw = _bars(); vec, seq = F.per_bar(A, hr, dw); j = 1100
    B = A.copy(); B[j + 1:] *= np.random.default_rng(1).uniform(0.5, 2, (len(A) - j - 1, 5))
    vec2, seq2 = F.per_bar(B, hr, dw)
    np.testing.assert_allclose(vec[:j + 1], vec2[:j + 1]); np.testing.assert_allclose(seq[:j + 1], seq2[:j + 1])
    assert not np.allclose(vec[j + 5], vec2[j + 5])                       # the check is not vacuous


def test_window_of_an_anchor_ends_before_the_label_bar():
    seq = np.arange(1000, dtype=np.float32)[:, None].repeat(7, 1); anch = np.array([800, 900])
    w = hwf.windows(seq, anch, 168)
    assert w[0, -1, 0] == 799 and w[0, 0, 0] == 800 - 168 and w[1, -1, 0] == 899


def _synth(n=24 * 700, seed=0):
    t = pd.date_range("2021-01-01", periods=n, freq="60min"); rng = np.random.default_rng(seed)
    return {"t": t, "r": rng.normal(0, 3e-3, n), "anch": np.arange(n), "vec": rng.normal(size=(n, 20)).astype(np.float32),
            "tb7": rng.normal(size=(n, 13)).astype(np.float32), "seq": rng.normal(size=(n + 10, 7)).astype(np.float32)}


def test_walk_forward_never_trains_on_the_scored_month():
    D = _synth(); t = D["t"]; seen = []

    def fake(arch, L, Vtr, Str, ytr, Vte, Ste, seed):
        seen.append((len(ytr), float(np.abs(Vtr).max()), len(Vte))); z = np.zeros(len(Vte)); return z, z, {"best_epoch": 0, "epochs_run": 1}
    rows = hwf.walk_forward(D, 365, "2022-03", "2022-03", [("patch", 168)], fit_net_fn=fake, log=lambda s: None)
    assert rows and seen
    m0 = pd.Timestamp("2022-03-01"); tr, te = hwf.wf.split_masks(t, m0, pd.Timestamp("2022-04-01"), 1, 60, 365)
    assert not (tr & te).any() and (t[tr] + pd.Timedelta(hours=2)).max() <= m0 and t[tr].max() < m0
    # corrupting the scored month's labels and features changes nothing the training saw
    D2 = dict(D, r=np.where(te, -D["r"], D["r"]), vec=np.where(te[:, None], 9.0, D["vec"]).astype(np.float32))
    first = seen[0]; seen.clear()
    hwf.walk_forward(D2, 365, "2022-03", "2022-03", [("patch", 168)], fit_net_fn=fake, log=lambda s: None)
    assert seen[0][:2] == first[:2]


def test_nets_start_at_the_linear_model():
    import hmodel
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0); V = rng.normal(size=(300, 20)).astype(np.float32); y = (V[:, 0] + rng.normal(size=300) > 0).astype(int)
    lr = LogisticRegression(C=0.1).fit(V, y)
    for arch in ("patch", "tcn"):
        for L in (168, 336):
            m = hmodel.build(arch, 20, 7, L); hmodel.set_linear(m, lr.coef_, lr.intercept_)
            S = rng.normal(size=(300, L, 7)).astype(np.float32)
            np.testing.assert_allclose(m([V, S], training=False).numpy()[:, 0], lr.decision_function(V), atol=1e-4)
            assert 15_000 <= hmodel.n_params(m) - 21 <= 32_000, (arch, L, hmodel.n_params(m))


def test_loader_drops_the_spent_period():
    if not os.path.exists(f"{F.CACHE}/h.npz"):
        return
    D = hwf.load(); assert D["t"].max() < pd.Timestamp("2024-07-01") and len(D["r"]) == len(D["vec"]) == len(D["tb7"])
