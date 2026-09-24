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


# ------------------------------------------------------------------ the numbers the old notebooks printed
def _shrunk(frame, betas):
    """The same frame with served deltas = beta x raw (as the calibration's delta shrink does)."""
    import copy

    served = copy.deepcopy(frame)
    raw = {h: frame.delta[h].copy() for h in HORIZONS}
    served.delta = {h: betas[h] * raw[h] for h in HORIZONS}
    return served, raw


def test_direction_block_reports_the_confusion_matrix_and_every_rate():
    from neural_trade.evaluation.report import HIGHER_IS_BETTER, LOWER_IS_BETTER, direction_block

    # 6 up, 4 down; the head calls up on 3 ups and 1 down -> tp 3, fn 3, fp 1, tn 3
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], float)
    prob = np.array([.9, .8, .7, .2, .3, .4, .6, .1, .2, .3, .99])
    mask = np.array([True] * 10 + [False])  # the last sample sits inside the deadband
    d = direction_block(labels, mask, prob)
    assert (d["tp"], d["fn"], d["fp"], d["tn"], d["n_masked"]) == (3, 3, 1, 3, 10)
    assert d["precision"] == 3 / 4 and d["recall"] == 3 / 6 and d["specificity"] == 3 / 4
    assert math.isclose(d["f1"], 2 * 3 / (2 * 3 + 1 + 3)) and math.isclose(d["bal_acc"], (0.5 + 0.75) / 2)
    assert math.isclose(d["acc"], 0.6) and math.isclose(d["pred_up_rate"], 0.4)
    # a constant baseline has an empty "beats" on these, so they are never compared
    assert not {"precision", "recall", "specificity", "f1", "tp"} & (HIGHER_IS_BETTER | LOWER_IS_BETTER)
    none_up = direction_block(labels, mask, np.full(11, 0.2))
    assert math.isnan(none_up["precision"]) and none_up["recall"] == 0.0 and none_up["specificity"] == 1.0


def test_delta_block_reports_dollar_errors_against_zero_and_the_bias():
    from neural_trade.evaluation.report import delta_block

    y = np.array([10.0, -20.0, 30.0, -40.0])
    d = np.array([5.0, -5.0, -5.0, -5.0])
    b = delta_block(y, d)
    assert math.isclose(b["rmse_zero"], math.sqrt(np.mean(y ** 2))) and math.isclose(b["mae_zero"], 25.0)
    assert math.isclose(b["rmse"], math.sqrt(np.mean((y - d) ** 2))) and math.isclose(b["mean_pred"], -2.5)
    assert math.isclose(b["mean_true"], -5.0) and b["share_pred_up"] == 0.25 and b["share_true_up"] == 0.5
    assert math.isclose(b["skill_vs_zero"], 1 - b["rmse"] ** 2 / b["rmse_zero"] ** 2)


def test_coherence_scores_the_trained_ordering_on_the_raw_heads():
    """A per-horizon shrink reorders the served magnitudes; the report must say so and score the raw heads."""
    rng = np.random.default_rng(7)
    frame = _frame(n=3000, seed=7)
    ordered = np.sort(np.abs(rng.normal(0, 50, (3000, 3))), axis=1) * np.sign(rng.normal(size=(3000, 1)))
    frame.delta = {h: ordered[:, i] for i, h in enumerate(HORIZONS)}
    betas = {"h0": 0.2, "h1": 0.02, "h2": 0.25}
    served, raw = _shrunk(frame, betas)
    rep = evaluate(served, Config(), raw_delta=raw, delta_scale=betas)
    c = rep.model["coherence"]
    assert c["mag_order_full_raw"] == 1.0 and c["mag_h0_le_h1_raw"] == 1.0
    assert c["mag_order_full"] < 0.2 and c["mag_h0_le_h1"] < 0.2          # the served (shrunk) ordering
    p = np.stack([served.prob(h) for h in HORIZONS], 1) > 0.5
    for i, h in enumerate(HORIZONS):
        assert math.isclose(c[f"delta_dir_align_{h}"], np.mean((raw[h] > 0) == p[:, i]))
    assert math.isclose(c["coherence_primary"], c["delta_dir_align_h1"])
    assert 0 <= c["delta_dir_align_indep_all"] <= min(c[f"delta_dir_align_indep_{h}"] for h in HORIZONS) + 1e-12
    # the raw heads are scored too, and the betas are recorded
    h1 = rep.model["horizons"]["h1"]
    assert math.isclose(h1["delta_raw"]["rmse"], math.sqrt(np.mean((frame.y[:, 1] - raw["h1"]) ** 2)))
    assert math.isclose(h1["delta"]["rmse"], math.sqrt(np.mean((frame.y[:, 1] - 0.02 * raw["h1"]) ** 2)))
    assert rep.meta["delta_scale"] == betas
    assert "h1/delta_raw/rmse" in rep.flat()


def test_evaluate_recovers_the_raw_heads_from_the_betas_or_estimates_the_betas():
    frame = _frame(n=2000, seed=8)
    betas = {"h0": 0.5, "h1": 0.1, "h2": 0.3}
    served, raw = _shrunk(frame, betas)
    by_beta = evaluate(served, Config(), delta_scale=betas).model
    by_raw = evaluate(served, Config(), raw_delta=raw)
    for h in HORIZONS:
        assert math.isclose(by_beta["horizons"][h]["delta_raw"]["rmse"], by_raw.model["horizons"][h]["delta_raw"]["rmse"])
        assert math.isclose(by_raw.meta["delta_scale"][h], betas[h])      # least-squares served / raw
    plain = evaluate(served, Config())
    assert "delta_raw" not in plain.model["horizons"]["h1"] and "mag_order_full_raw" not in plain.model["coherence"]


def test_a_zero_beta_is_flagged_and_the_sign_checks_use_the_raw_heads():
    frame = _frame(n=2000, seed=10)
    betas = {"h0": 0.3, "h1": 0.0, "h2": 0.3}
    served, raw = _shrunk(frame, betas)
    rep = evaluate(served, Config(), raw_delta=raw, delta_scale=betas)
    p1 = served.prob("h1") > 0.5
    assert math.isclose(rep.model["coherence"]["delta_dir_align_h1"], np.mean((raw["h1"] > 0) == p1))
    assert "beta = 0 for h1" in rep.to_markdown()


def test_legacy_report_keys_keep_their_meaning():
    frame = _frame(n=2000, seed=9)
    served, raw = _shrunk(frame, {"h0": 0.2, "h1": 0.02, "h2": 0.25})
    rep = evaluate(served, Config(), raw_delta=raw)
    legacy_direction = {"mcc", "auc", "brier", "ece_pos", "acc", "bal_acc", "pred_up_rate", "true_up_rate", "n_masked"}
    legacy_delta = {"rmse", "mae", "ev", "corr", "skill_vs_zero"}
    for h in HORIZONS:
        row = rep.model["horizons"][h]
        assert legacy_direction <= set(row["direction"]) and legacy_delta <= set(row["delta"])
    c = rep.model["coherence"]
    assert {"mag_order_full", "unanimity", "delta_dir_align_all", "coherence_primary"} <= set(c)
    d = np.abs(np.stack([served.delta[h] for h in HORIZONS], 1))
    assert math.isclose(c["mag_order_full"], np.mean((d[:, 0] <= d[:, 1]) & (d[:, 1] <= d[:, 2])))  # still served


def test_baseline_margins_carry_both_values_and_flag_wins_inside_the_noise():
    train, test = _frame(seed=11), _frame(seed=12)
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    rep = evaluate(test, Config(), baselines=base)
    mg = rep.baseline_margins["zero_delta"]["delta/rmse"]["h1"]
    assert mg["model"] == rep.model["horizons"]["h1"]["delta"]["rmse"]
    assert mg["baseline"] == rep.baselines["zero_delta"]["horizons"]["h1"]["delta"]["rmse"]
    assert math.isclose(mg["margin"], mg["baseline"] - mg["model"]) and mg["margin"] > 0   # lower is better
    assert mg["dm_z"] > 3                                                                  # a real edge
    auc = rep.baseline_margins["class_prior"]["direction/auc"]["h1"]
    assert math.isclose(auc["margin"], auc["model"] - auc["baseline"]) and auc["dm_z"] is None  # no per-sample loss
    # a "model" that is the zero prediction plus a whisper: it beats or loses by nothing, and says so
    tiny = _frame(seed=12)
    rng = np.random.default_rng(3)
    tiny.delta = {h: rng.normal(0, 0.5, len(tiny)) for h in HORIZONS}
    tiny_rep = evaluate(tiny, Config(), baselines=base)
    z = tiny_rep.baseline_margins["zero_delta"]["delta/rmse"]
    assert all(abs(z[h]["dm_z"]) < 1.96 for h in HORIZONS)
    assert "noise (DM z" in tiny_rep.to_markdown()


def test_markdown_shows_the_classification_rates_dollar_errors_and_baseline_values():
    import re

    train, test = _frame(seed=13), _frame(seed=14)
    served, raw = _shrunk(test, {"h0": 0.3, "h1": 0.05, "h2": 0.4})
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    md = evaluate(served, Config(), baselines=base, raw_delta=raw).to_markdown()
    for label in ("n scored (outside the deadband)", "true up-rate", "calls up (predicted up-rate)", "accuracy",
                  "balanced accuracy", "precision (up)", "recall / sensitivity (up)", "specificity (down)",
                  "F1 (up)", "| Brier |", "TP / FP / TN / FN", "RMSE ($), served", "RMSE ($), raw heads",
                  "RMSE ($), zero prediction", "MAE ($), zero prediction", "mean predicted ($), raw heads",
                  "mean realised ($)", "shrink beta", "raw price heads (the trained ordering)",
                  "expected if the two signs were independent", "Gaussian readout: ECE"):
        assert label in md, label
    rmse_row = next(line for line in md.splitlines() if line.startswith("| RMSE ($), served |"))
    assert re.fullmatch(r"\| RMSE \(\$\), served \|( \d+\.\d{2} \|){3}", rmse_row)   # dollars, 2 decimals
    tp_row = next(line for line in md.splitlines() if line.startswith("| TP / FP / TN / FN |"))
    assert re.search(r"\| \d+ / \d+ / \d+ / \d+ \|", tp_row)
    base_row = next(line for line in md.splitlines() if line.startswith("| logreg_lags | direction/auc |"))
    assert base_row.count(" vs ") == 3 and "beat" in base_row          # values, not only a verdict
    assert "| zero_delta | delta/ev |" not in md                         # restates the RMSE verdict
    assert "| zero_delta | delta/rmse |" in md


def test_markdown_of_an_older_report_without_the_new_fields_still_renders():
    from neural_trade.evaluation.report import EvalReport

    old_direction = {"mcc": 0.02, "auc": 0.51, "brier": 0.25, "ece_pos": 0.01, "acc": 0.51, "bal_acc": 0.51,
                     "pred_up_rate": 0.5, "true_up_rate": 0.5, "n_masked": 900}
    row = {"direction": old_direction, "gauss_direction": old_direction,
           "delta": {"rmse": 200.0, "mae": 150.0, "ev": 0.0, "corr": 0.01, "skill_vs_zero": 0.0},
           "variance": {"crps": 100.0, "nll": 6.0, "pit_ks": 0.05, "corr_var_err2_spearman": 0.2},
           "n": 1000, "n_eff": 100, "confidence_gap": {"gap": 0.0, "ci": [-0.01, 0.01], "verdict": "NOISE"}}
    rep = EvalReport("old", "test", 5.0, 1000, {"horizons": {h: row for h in HORIZONS},
                                                "coherence": {"mag_order_full": 0.07, "unanimity": 0.3,
                                                              "delta_dir_align_all": 0.2,
                                                              "coherence_primary": 0.6}},
                     baselines={"zero_delta": {"horizons": {h: {"delta": {"rmse": 201.0}} for h in HORIZONS}}},
                     beats_baseline={"zero_delta": {"delta/rmse": {h: True for h in HORIZONS}}})
    md = rep.to_markdown()
    assert "| precision (up) | n/a | n/a | n/a |" in md
    assert "200.00 vs 201.00 (+1.00, +0.50%): beats" in md


# ------------------------------------------------------------------ noise tests of the baseline margins
def test_dm_z_uses_a_newey_west_long_run_variance_of_the_loss_difference():
    """Overlapping h-bar targets make the loss difference an MA(h-1): the z must use its long-run variance."""
    from neural_trade.evaluation.report import DM_LAG_PER_STEP, dm_z, long_run_variance

    rng = np.random.default_rng(20)
    n, h = 8000, 15
    e = rng.normal(0, 1, n + h)
    diff = 0.02 + np.convolve(e, np.ones(h) / h, mode="valid")[:n]      # overlapping sums, a small real edge
    base, model = diff, np.zeros(n)                                      # loss_base - loss_model = diff
    lag = DM_LAG_PER_STEP * h
    z_c = diff - diff.mean()
    lrv = z_c @ z_c / n + 2 * sum((1 - k / (lag + 1)) * (z_c[k:] @ z_c[:-k]) / n for k in range(1, lag + 1))
    assert math.isclose(long_run_variance(diff, lag), lrv)
    assert math.isclose(dm_z(model, base, h), diff.mean() / math.sqrt(lrv / n))
    naive = diff.mean() / (diff.std(ddof=1) / math.sqrt(n))             # pretends the samples are independent
    on_n_eff = diff.mean() / (diff.std(ddof=1) / math.sqrt(n // h))     # the old, conservative statistic
    assert on_n_eff < dm_z(model, base, h) < naive
    # iid differences: the long-run variance is the plain variance, whatever the lag
    iid = rng.normal(0.05, 1, n)
    assert dm_z(np.zeros(n), iid, 1, lag=0) == pytest.approx(iid.mean() / (iid.std() / math.sqrt(n)), rel=1e-9)
    assert dm_z(np.zeros(n), iid, 10) == pytest.approx(iid.mean() / (iid.std() / math.sqrt(n)), rel=0.1)
    # unscored bars (NaN) keep their place in time instead of pulling distant bars together
    gappy = diff.copy()
    gappy[::3] = np.nan
    ok = np.isfinite(gappy)
    zc = np.where(ok, gappy - np.nanmean(gappy), 0.0)
    m = ok.sum()
    lrv_gap = zc @ zc / m + 2 * sum((1 - k / (lag + 1)) * (zc[k:] @ zc[:-k]) / m for k in range(1, lag + 1))
    assert math.isclose(dm_z(model, gappy, h), np.nanmean(gappy) / math.sqrt(lrv_gap / m))
    assert dm_z(np.zeros(20), np.ones(20), 15) is None                   # < 2 non-overlapping outcomes
    assert dm_z(np.zeros(100), np.ones(100), 1) is None                  # a constant difference


def test_block_bootstrap_metrics_reproduce_the_report_at_unit_weights():
    """With every bar once, each resampled metric equals the report's value: the bootstrap scores the same thing."""
    from neural_trade.evaluation.report import _boot_metric, block_bootstrap_counts, score_frame
    from neural_trade.metrics.direction_labels import direction_labels_np

    train, test = _frame(n=3000, seed=21), _frame(n=3000, seed=22)
    rng = np.random.default_rng(0)
    test.direction_prob = {h: np.round(test.direction_prob[h], 2) for h in HORIZONS}   # ties in P(up) and ranks
    test.variance_scaled = {h: np.round(test.variance_scaled[h], 1) + 0.05 for h in HORIZONS}
    frames = {"model": test, **BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0).predict(test)}
    labels = direction_labels_np(test.y, test.last_close, 5.0)
    ones = np.ones((1, len(test)))
    for name, fr in frames.items():
        scored = score_frame(fr, 5.0)
        for i, h in enumerate(HORIZONS):
            for group, metric in (("direction", "mcc"), ("direction", "auc"), ("direction", "bal_acc"),
                                  ("direction", "ece_pos"), ("delta", "corr"), ("delta", "ev"),
                                  ("variance", "pit_ks"), ("variance", "corr_var_err2_spearman")):
                want = scored["horizons"][h][group][metric]
                got = float(_boot_metric(fr, i, group, metric, labels, ones)[0])
                assert got == pytest.approx(want, abs=1e-9), (name, h, group, metric)
    W = block_bootstrap_counts(1000, block=80, n_boot=50, seed=rng.integers(100))
    assert W.shape == (50, 1000) and (W.sum(1) == 1000).all()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, W[0] > 0, 0])))      # resamples are made of whole blocks
    assert runs.max() >= 80


def test_ranked_metrics_get_a_paired_bootstrap_noise_flag():
    train, test = _frame(seed=23), _frame(seed=24)
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    rep = evaluate(test, Config(), baselines=base)
    auc = rep.baseline_margins["class_prior"]["direction/auc"]["h1"]
    assert auc["dm_z"] is None and auc["boot_z"] > 5 and auc["boot_se"] > 0      # a real edge (AUC ~0.75)
    for key in ("direction/mcc", "direction/bal_acc", "direction/ece_pos"):
        assert rep.baseline_margins["logreg_lags"][key]["h1"]["boot_se"] > 0
    assert rep.meta["noise_tests"]["dm_z"].startswith("Diebold-Mariano")
    # a model that is the logreg baseline plus a whisper: every ranked margin is inside the noise
    twin = base.predict(test)["logreg_lags"]
    twin.direction_prob = {h: np.clip(twin.prob(h) + np.random.default_rng(1).normal(0, 1e-3, len(twin)), 0, 1)
                           for h in HORIZONS}
    twin_rep = evaluate(twin, Config(), baselines=base)
    for key in ("direction/mcc", "direction/auc", "direction/bal_acc"):
        for h in HORIZONS:
            z = twin_rep.baseline_margins["logreg_lags"][key][h]["boot_z"]
            assert z is None or abs(z) < 1.96, (key, h, z)
    assert "noise (boot z" in twin_rep.to_markdown()


def test_a_significant_loss_is_called_significantly_worse():
    train, test = _frame(seed=25), _frame(seed=26)
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    worse = _frame(seed=26)
    worse.delta = {h: -2 * test.delta[h] for h in HORIZONS}              # confidently wrong prices
    rep = evaluate(worse, Config(), baselines=base)
    assert rep.baseline_margins["zero_delta"]["delta/rmse"]["h1"]["dm_z"] < -1.96
    row = next(line for line in rep.to_markdown().splitlines() if line.startswith("| zero_delta | delta/rmse |"))
    assert "does not beat, significantly worse (DM z -" in row


def test_plain_report_labels_the_frame_deltas_served_and_says_the_raw_heads_are_missing():
    rep = evaluate(_frame(n=2000, seed=27), Config())
    md = rep.to_markdown()
    assert "| RMSE ($), served |" in md and "| mean predicted ($), served |" in md
    assert "predicted ($), predicted" not in md and "), predicted |" not in md
    assert "raw_delta=" in md and "RMSE ($), raw heads" not in md


def test_a_frame_can_carry_its_raw_heads_and_betas_to_evaluate():
    frame = _frame(n=2000, seed=28)
    betas = {"h0": 0.4, "h1": 0.05, "h2": 0.3}
    served, raw = _shrunk(frame, betas)
    served.meta["delta_raw"], served.meta["delta_scale"] = raw, betas
    rep = evaluate(served, Config())
    assert rep.meta["delta_scale"] == betas
    assert math.isclose(rep.model["horizons"]["h1"]["delta_raw"]["rmse"],
                        math.sqrt(np.mean((frame.y[:, 1] - raw["h1"]) ** 2)))
    assert "mag_order_full_raw" in rep.model["coherence"]


def test_markdown_direction_table_counts_the_effective_size_of_the_scored_moves():
    frame = _frame(n=3000, seed=29)
    rep = evaluate(frame, Config())
    md = rep.to_markdown()
    row = next(line for line in md.splitlines() if line.startswith("| n_eff of the scored moves"))
    want = [rep.model["horizons"][h]["direction"]["n_masked"] // s for h, s in zip(HORIZONS, (10, 15, 20))]
    assert row == "| n_eff of the scored moves (n scored // bars ahead) | " + " | ".join(map(str, want)) + " |"
    assert "| n_eff (non-overlapping outcomes) |" not in md
