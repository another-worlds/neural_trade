"""Evaluation protocol (plan C2): metric definitions, baselines, confidence gap, n_eff."""
from __future__ import annotations

import math

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.evaluation.baselines import BaselineSet, lag_features
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.evaluation.report import confidence_gap, evaluate, gaussian_crps
from neural_trade.metrics import numpy_metrics as npm

SCALE, LC = 250.0, 110_000.0


def _frame(n=6000, signal=0.5, seed=0, calibrated_sigma=True):
    """y = a + e with a predictable part a (share `signal`^2 of the variance); the model predicts a
    and reports the honest residual sigma. Closes follow a random walk on BTC's monthly scale."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(100, 300, (n, 3))
    a = rng.normal(0, 1, (n, 3)) * sig * signal
    resid_sd = sig * math.sqrt(1 - signal ** 2)
    y = a + rng.normal(0, 1, (n, 3)) * resid_sd
    if not calibrated_sigma:
        resid_sd = np.full_like(sig, 200.0)
    prob = 1 / (1 + np.exp(-a / 60.0))
    close = LC + np.cumsum(rng.normal(0, 100, n + 60))
    X = np.stack([close[i:i + 60] for i in range(n)])
    lc = X[:, -1]
    return PredictionFrame(y, lc, {h: a[:, i] for i, h in enumerate(HORIZONS)},
                           {h: prob[:, i] for i, h in enumerate(HORIZONS)},
                           {h: (resid_sd[:, i] / SCALE) ** 2 for i, h in enumerate(HORIZONS)}, SCALE, 0.0,
                           (10, 15, 20), "test", X_raw=X)


def test_ece_positive_class_calibrated_is_zero_and_top_label_is_not_used():
    rng = np.random.default_rng(0)
    p = np.full(100_000, 0.05)
    labels = (rng.uniform(size=p.size) < 0.05).astype(float)
    assert npm.ece_pos(labels, p) < 0.005  # a top-label ECE mix reports ~0.9 here


def test_crps_closed_form_matches_monte_carlo():
    rng = np.random.default_rng(1)
    y, mu, s = 1.3, 0.2, 0.7
    draws = rng.normal(mu, s, 2_000_000)
    mc = np.mean(np.abs(draws - y)) - 0.5 * np.mean(np.abs(draws - rng.permutation(draws)))
    assert abs(float(gaussian_crps(np.array([y]), mu, s)[0]) - mc) < 5e-3


def test_informative_model_beats_baselines_and_n_eff_is_reported():
    train, test = _frame(seed=1), _frame(seed=2)
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    rep = evaluate(test, Config(), baselines=base)
    h1 = rep.model["horizons"]["h1"]
    assert h1["delta"]["ev"] > 0.15 and h1["direction"]["auc"] > 0.6 and h1["variance"]["pit_ks"] < 0.05
    assert h1["n_eff"] == len(test) // 15
    assert rep.beats_baseline["zero_delta"]["delta/rmse"]["h1"]
    assert rep.beats_baseline["class_prior"]["direction/auc"]["h1"]
    assert rep.model["horizons"]["h1"]["variance"]["crpss"] > 0  # beats the constant-variance baseline


def test_ev_price_trap_is_not_reported():
    """Predicting no change scores EV ~0.999 on price LEVELS; only delta EV may be reported."""
    train, test = _frame(seed=3), _frame(seed=4)
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    zero = base.predict(test)["zero_delta"]
    price_ev = npm.explained_variance(test.last_close + test.y[:, 1], test.last_close + zero.delta["h1"])
    assert price_ev > 0.99  # the trap
    rep = evaluate(zero, Config())
    assert abs(rep.model["horizons"]["h1"]["delta"]["ev"]) < 1e-9
    assert not any("price" in k for k in rep.flat())


def test_confidence_gap_separates_signal_from_noise():
    rng = np.random.default_rng(5)
    n = 8000
    conf = rng.uniform(0, 0.5, n)
    real = (rng.uniform(size=n) < 0.5 + conf * 0.4).astype(float)  # accuracy rises with confidence
    noise = (rng.uniform(size=n) < 0.55).astype(float)
    assert confidence_gap(real, conf, np.median(conf))["verdict"] == "WORKS"
    assert confidence_gap(noise, conf, np.median(conf))["verdict"] == "NOISE"


def test_logreg_baseline_builds_lag_features_and_predicts():
    rng = np.random.default_rng(6)
    n = 5000
    steps = rng.normal(0, 20, n + 80)
    close = LC + np.cumsum(steps)
    X = np.stack([close[i:i + 60] for i in range(n)])
    future = np.stack([close[i + 60 + h - 1] for i in range(n) for h in (10, 15, 20)]).reshape(n, 3)
    y = future - X[:, -1:]
    feats = lag_features(X)
    assert feats.shape == (n, 8)
    frame = PredictionFrame(y, X[:, -1], {h: np.zeros(n) for h in HORIZONS}, {h: np.full(n, 0.5) for h in HORIZONS},
                            {h: np.ones(n) for h in HORIZONS}, SCALE, X_raw=X)
    base = BaselineSet.fit(X[:3000], y[:3000], X[:3000, -1], 5.0)
    assert "logreg_lags" in base.predict(frame)
