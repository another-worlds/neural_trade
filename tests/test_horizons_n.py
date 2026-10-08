"""NT-042 (D-022): a variable number of horizons, N = 2, 3 and 4, on CPU.

* the one source of horizon keys and its layout rules (core/horizons.py);
* Config accepts N >= 2; every registered architecture builds 3N + 1 named heads;
* the training objective's per-horizon wiring, term by term (each per-horizon component reads its own
  target column, price / direction / variance head, extended-trend column and point-loss weight; the
  pairwise terms run over every neighbouring pair, scaled so N = 3 is the old two-pair sum);
* calibration, serving post-processing, the evaluation report and the strategies for N horizons;
* one tiny end-to-end training run per N (train -> calibrate -> bundle -> Predictor -> evaluate ->
  SignalFrame -> backtest -> figures).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from neural_trade.core import horizons as HZ

tf = pytest.importorskip("tensorflow")

N_ALL = (2, 3, 4)
B = 64


# ------------------------------------------------------------------ core/horizons.py
def test_horizon_keys_primary_roles_pairs_and_scale():
    assert HZ.horizon_keys(4) == ("h0", "h1", "h2", "h3")
    assert HZ.REFERENCE_HORIZONS == ("h0", "h1", "h2") and HZ.PRIMARY_HORIZON == "h1"
    assert [HZ.primary_index(n) for n in (2, 3, 4)] == [1, 1, 2]
    assert [HZ.point_weight_role(i, 3) for i in range(3)] == ["short", "point", "long"]
    assert [HZ.point_weight_role(i, 2) for i in range(2)] == ["short", "long"]
    assert [HZ.point_weight_role(i, 4) for i in range(4)] == ["short", "point", "point", "long"]
    assert HZ.neighbour_pairs(4) == [(0, 1), (1, 2), (2, 3)] and HZ.neighbour_pairs(1) == []
    assert HZ.pair_sum_scale(3) == 1.0 and HZ.pair_sum_scale(2) == 2.0
    assert HZ.pair_sum_scale(4) == pytest.approx(2.0 / 3.0) and HZ.pair_sum_scale(1) == 0.0
    assert HZ.keys_of({"h10": 1, "h2": 2, "x": 3}) == ("h2", "h10") and HZ.keys_of({}) == HZ.REFERENCE_HORIZONS
    with pytest.raises(ValueError):
        HZ.horizon_index("weighted")


def test_namedtuples_are_the_reference_classes_at_three_and_extend_otherwise():
    from neural_trade.core.outputs import LossComponents, PredictiveOutputs

    assert HZ.predictive_outputs_type(3) is PredictiveOutputs and HZ.loss_components_type(3) is LossComponents
    assert HZ.predictive_output_fields(3) == PredictiveOutputs._fields
    assert HZ.loss_component_fields(3) == LossComponents._fields
    for n in (2, 4):
        po, lc = HZ.predictive_outputs_type(n), HZ.loss_components_type(n)
        assert len(po._fields) == 3 * n + 1 and po._fields[-1] == "vacuum_overflow"
        assert len(lc._fields) == 37 + 8 * (n - 3)       # point, local, global, extended, dir, nll, crps, soft_ece
        assert f"point_h{n - 1}" in lc._fields and f"soft_ece_h{n - 1}" in lc._fields
        assert HZ.n_from_outputs(3 * n + 1) == n


def _steps(n):
    return [10 + 5 * i for i in range(n)]


def _cfg(n, **kw):
    from neural_trade.core.config import Config

    return Config(HORIZON_STEPS=_steps(n), EXTENDED_TREND_PERIODS=_steps(n), **kw)


@pytest.mark.parametrize("n", (2, 4, 5))
def test_config_accepts_any_number_of_horizons_from_two(n):
    assert len(_cfg(n).HORIZON_STEPS) == n


# ------------------------------------------------------------------ the models
@pytest.mark.parametrize("n", (2, 4))
@pytest.mark.parametrize("name", ["gru_attention", "gru_small", "linear_indicators"])
def test_every_architecture_builds_three_named_heads_per_horizon(name, n):
    from neural_trade.registries.models import Models

    tf.keras.utils.set_random_seed(0)
    cfg = _cfg(n, LOOKBACK=32, INPUT_SERIES=["close"], INDICATOR_FAMILIES={}, MA_SPANS=[5], RSI_PERIODS=[14],
               BB_PERIODS=[20], MACD_SETTINGS=[{"fast": 12, "slow": 26, "signal": 9}])
    model = Models.build(name, cfg)
    fields = HZ.predictive_output_fields(n)
    assert len(model.outputs) == 3 * n + 1
    assert all(o == f or o.startswith(f + "_") for o, f in zip(model.output_names, fields)), model.output_names
    outs = HZ.predictive_outputs_type(n)(*model(tf.random.normal([4, 32]), training=False))
    last = getattr(outs, f"variance_h{n - 1}")
    assert last.shape == (4, 1) and float(tf.reduce_min(last)) > 0.0


def test_the_output_contract_checks_names_and_counts():
    from neural_trade.core.exceptions import ComponentValidationError
    from neural_trade.registries.models import ensure_predictive_outputs

    inp = tf.keras.Input((8,))

    def heads(names):
        return tf.keras.Model(inp, [tf.keras.layers.Dense(1, name=nm)(inp) for nm in names])

    good = list(HZ.predictive_output_fields(2))
    ensure_predictive_outputs(heads(good), n_horizons=2)
    swapped = good.copy()
    swapped[0], swapped[3] = swapped[3], swapped[0]           # price_h0 <-> price_h1
    with pytest.raises(ComponentValidationError, match="output names"):
        ensure_predictive_outputs(heads(swapped), n_horizons=2)
    with pytest.raises(ComponentValidationError, match="expected 10 outputs"):
        ensure_predictive_outputs(heads(good))                # the reference three by default
    clip = [f + "_clip" if f != "vacuum_overflow" else f for f in HZ.predictive_output_fields(4)]
    ensure_predictive_outputs(heads(clip), n_horizons=4)


# ------------------------------------------------------------------ the training objective
def _loss_model(n, **cfg_kw):
    from neural_trade.training.custom_model import CustomTrainModel

    inp = tf.keras.Input(shape=(60,), name="close_window")
    outs = [tf.keras.layers.Dense(1, name=f"head_{i}")(inp) for i in range(3 * n + 1)]
    base = tf.keras.Model(inp, outs, name="tiny_base")
    cfg = _cfg(n, **cfg_kw)
    return CustomTrainModel(base_model=base, pred_scale=261.0, pred_mean=3.2, config=cfg,
                            lambda_point=cfg.LAMBDA_POINT, inputs=base.inputs, outputs=base.outputs)


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (B, 60)).astype(np.float32)
    y = rng.normal(0.0, 1.0, (B, n)).astype(np.float32)
    lc = (110_000.0 + rng.normal(0.0, 500.0, (B, 1))).astype(np.float32)
    ext = rng.normal(0.0, 200.0, (B, n)).astype(np.float32)
    price = [rng.normal(0.0, 1.0, (B, 1)).astype(np.float32) for _ in range(n)]
    dirs = [rng.uniform(0.05, 0.95, (B, 1)).astype(np.float32) for _ in range(n)]
    var = [rng.uniform(0.3, 2.0, (B, 1)).astype(np.float32) for _ in range(n)]
    ov = rng.uniform(0.0, 0.5, (B, 1)).astype(np.float32)
    return dict(x=x, y=y, lc=lc, ext=ext, price=price, dirs=dirs, var=var, ov=ov)


def _loss(m, d):
    heads = []
    for p, q, v in zip(d["price"], d["dirs"], d["var"]):
        heads += [tf.constant(p), tf.constant(q), tf.constant(v)]
    return m.custom_loss(tf.constant(d["x"]), tf.constant(d["y"]), heads, tf.constant(d["lc"]),
                         tf.constant(d["ext"]), vacuum_overflow=tf.constant(d["ov"]))


def _vals(c):
    return {f: float(v) for f, v in zip(c._fields, c)}


PER_H = ("point", "extended", "dir", "nll", "crps", "soft_ece")
ALL_ON = dict(LAMBDA_CRPS=1.0, LAMBDA_SOFT_ECE=1.0, LAMBDA_T_PERP=1.0, LAMBDA_CASIMIR=1.0, LAMBDA_HD=1.0,
              LAMBDA_IFE=1.0, LAMBDA_VAC_OVERFLOW=1.0, LAMBDA_VOL=1.0)


def _changed(a, b, rtol=1e-6):
    return {k for k in a if not math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=1e-9)}


@pytest.mark.parametrize("n", N_ALL)
def test_loss_components_type_finite_and_the_total_is_the_sum_of_its_terms(n):
    m = _loss_model(n, **ALL_ON)
    c = _loss(m, _inputs(n))
    assert type(c) is HZ.loss_components_type(n)
    vals = _vals(c)
    assert all(np.isfinite(v) for v in vals.values()), vals
    contrib = sum(float(v) for v in m._contrib_terms(c).values())
    assert contrib == pytest.approx(vals["total"], rel=1e-5)


def _per_h_changed(a, b, n):
    return {(name, i) for name in PER_H for i in range(n)
            if not math.isclose(a[f"{name}_h{i}"], b[f"{name}_h{i}"], rel_tol=1e-6, abs_tol=1e-9)}


@pytest.mark.parametrize("n", N_ALL)
def test_each_target_column_feeds_only_its_own_horizon(n):
    m = _loss_model(n, **ALL_ON)
    d = _inputs(n)
    base = _vals(_loss(m, d))
    for j in range(n):
        e = {**d, "y": d["y"].copy()}
        e["y"][:, j] = -3.0 * e["y"][:, j] + 0.7          # flips signs too: the direction labels move
        moved = _per_h_changed(base, _vals(_loss(m, e)), n)
        assert moved == {("point", j), ("dir", j), ("nll", j), ("crps", j), ("soft_ece", j)}, (j, moved)


@pytest.mark.parametrize("n", N_ALL)
def test_each_head_feeds_only_its_own_horizon(n):
    m = _loss_model(n, **ALL_ON)
    d = _inputs(n)
    base = _vals(_loss(m, d))
    for j in range(n):
        e = {**d, "price": list(d["price"])}
        e["price"][j] = e["price"][j] * 1.7 + 0.3
        assert _per_h_changed(base, _vals(_loss(m, e)), n) == {("point", j), ("extended", j), ("nll", j),
                                                                 ("crps", j)}, ("price", j)
        e = {**d, "dirs": list(d["dirs"])}
        e["dirs"][j] = np.clip(1.0 - e["dirs"][j], 0.01, 0.99)
        assert _per_h_changed(base, _vals(_loss(m, e)), n) == {("dir", j), ("soft_ece", j)}, ("dir", j)
        e = {**d, "var": list(d["var"])}
        e["var"][j] = e["var"][j] * 2.5
        assert _per_h_changed(base, _vals(_loss(m, e)), n) == {("nll", j), ("crps", j)}, ("var", j)
        e = {**d, "ext": d["ext"].copy()}
        e["ext"][:, j] = -e["ext"][:, j] + 50.0
        assert _per_h_changed(base, _vals(_loss(m, e)), n) == {("extended", j)}, ("ext", j)


@pytest.mark.parametrize("n", N_ALL)
def test_point_weights_follow_the_short_point_long_roles(n):
    d = _inputs(n)
    base = _vals(_loss(_loss_model(n), d))
    for role, field in (("short", "LAMBDA_SHORT"), ("point", "LAMBDA_POINT"), ("long", "LAMBDA_LONG")):
        got = _vals(_loss(_loss_model(n, **{field: 3.0}), d))
        for i in range(n):
            factor = 3.0 if HZ.point_weight_role(i, n) == role else 1.0
            assert got[f"point_h{i}"] == pytest.approx(factor * base[f"point_h{i}"], rel=1e-5), (role, i)


@pytest.mark.parametrize("n", N_ALL)
def test_per_horizon_terms_match_numpy(n):
    """point (log-cosh), NLL (Gaussian), the extended-trend prior and the direction BCE per horizon."""
    m = _loss_model(n)
    d = _inputs(n)
    v = _vals(_loss(m, d))
    scale, mean = 261.0, 3.2
    y_raw = d["y"] * scale + mean
    lc = d["lc"][:, 0]
    lam = {"short": m.config.LAMBDA_SHORT, "point": m.config.LAMBDA_POINT, "long": m.config.LAMBDA_LONG}
    for i in range(n):
        p, var, y = d["price"][i][:, 0], np.maximum(d["var"][i][:, 0], 1e-4), d["y"][:, i]
        r = y - p
        logcosh = np.mean(np.abs(r) + np.log1p(np.exp(-2 * np.abs(r))) - np.log(2.0))
        assert v[f"point_h{i}"] == pytest.approx(lam[HZ.point_weight_role(i, n)] * logcosh, rel=1e-4)
        nll = np.mean(0.5 * (np.log(2 * np.pi) + np.log(var + 1e-8)) + 0.5 * r ** 2 / (var + 1e-8))
        assert v[f"nll_h{i}"] == pytest.approx(nll, rel=1e-4)
        past = (d["ext"][:, i] - mean) / (scale + 1e-8)
        z = p - past
        ext = np.mean(np.abs(z) + np.log1p(np.exp(-2 * np.abs(z))) - np.log(2.0))
        assert v[f"extended_h{i}"] == pytest.approx(m.lambda_extended_trend.numpy() * ext, rel=1e-4)
        ret = y_raw[:, i] / (lc + 1e-8)
        mask = np.abs(ret) > m.config.DIR_DEADBAND_BPS / 1e4
        lab = (ret > m.config.DIR_DEADBAND_BPS / 1e4).astype(float)
        q = np.clip(d["dirs"][i][:, 0], 1e-7, 1 - 1e-7)
        bce = -(lab * np.log(q) + (1 - lab) * np.log(1 - q))
        assert v[f"dir_h{i}"] == pytest.approx((bce * mask).sum() / (mask.sum() + 1e-8), rel=1e-4)


@pytest.mark.parametrize("n", N_ALL)
def test_vol_penalty_reads_the_primary_horizon(n):
    m = _loss_model(n, LAMBDA_VOL=1.0)
    d = _inputs(n)
    pi = HZ.primary_index(n)
    v = _vals(_loss(m, d))
    expect = min(abs(np.std(d["price"][pi][:, 0]) - np.std(d["y"][:, pi])), 10.0) * float(m.lambda_vol.numpy())
    assert v["vol_loss"] == pytest.approx(expect, rel=1e-4)
    for j in range(n):
        e = {**d, "price": list(d["price"])}
        e["price"][j] = e["price"][j] * 3.0
        moved = not math.isclose(_vals(_loss(m, e))["vol_loss"], v["vol_loss"], rel_tol=1e-6)
        assert moved == (j == pi), j


def _np_pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.mean(a * b) / ((a.std() + 1e-8) * (b.std() + 1e-8)))


@pytest.mark.parametrize("n", N_ALL)
def test_pairwise_terms_average_every_neighbouring_pair_scaled_to_the_reference(n):
    """Casimir, IFE and the coherence magnitude ordering: (2 / (N - 1)) x the sum over (h_i, h_i+1)."""
    rng = np.random.default_rng(3)
    d = _inputs(n)
    # alternating signs (Casimir interference), low variance (its hinge), strongly correlated heads (IFE)
    common = rng.normal(0.0, 1.0, (B, 1)).astype(np.float32)
    d["price"] = [((-1) ** i * (1.0 + 0.3 * i) * common + 0.05 * rng.normal(0, 1, (B, 1))).astype(np.float32)
                  for i in range(n)]
    d["var"] = [np.full((B, 1), 0.2 + 0.1 * i, np.float32) for i in range(n)]
    m = _loss_model(n, LAMBDA_CASIMIR=1.0, LAMBDA_IFE=1.0, RHO_MAX=0.5, COHERENCE_MAGNITUDE_ONLY=True)
    v = _vals(_loss(m, d))
    p = [x[:, 0] for x in d["price"]]
    vv = [np.maximum(x[:, 0], 1e-4) for x in d["var"]]
    s = [np.tanh(x / 0.5) for x in p]
    pairs = HZ.neighbour_pairs(n)
    cas = sum(np.mean(np.maximum(-s[a] * s[b], 0) * np.maximum(-np.log(0.5 * (vv[a] + vv[b]) + 1e-8), 0))
              for a, b in pairs)
    ife = sum(max(abs(_np_pearson(p[a], p[b])) - 0.5, 0.0) for a, b in pairs)
    mag = sum(np.mean(np.maximum(np.abs(p[a]) - np.abs(p[b]), 0)) for a, b in pairs)
    scale = 2.0 / (n - 1)
    assert cas > 0 and ife > 0 and mag > 0
    assert v["casimir_val"] == pytest.approx(scale * cas, rel=1e-4)
    assert v["ife_val"] == pytest.approx(scale * ife, rel=1e-4)
    assert v["coherence_penalty_val"] == pytest.approx(scale * mag / 3.0, rel=1e-4)


@pytest.mark.parametrize("n", N_ALL)
def test_horizon_mean_terms_hd_overflow_and_vacuum(n):
    """HD reads the mean variance over the horizons, the overflow term the mean residual, the vacuum term
    the cross-horizon std: none of them grows with N."""
    m = _loss_model(n, LAMBDA_HD=1.0, LAMBDA_VAC_OVERFLOW=1.0, LAMBDA_VAC=0.1)
    d = _inputs(n)
    v = _vals(_loss(m, d))
    vv = np.mean([np.maximum(x[:, 0], 1e-4) for x in d["var"]], axis=0)
    log_vol = np.log(np.std(d["x"], axis=1) + 1e-3)
    log_var = np.log(vv + 1e-3)

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-3)

    assert v["hd_val"] == pytest.approx(1.0 - np.mean(z(log_vol) * z(log_var)), rel=1e-3, abs=1e-5)
    res = np.mean([np.abs(d["y"][:, i] - d["price"][i][:, 0]) for i in range(n)], axis=0)
    mean_ov, mean_res = d["ov"].mean(), res.mean()
    assert v["vac_overflow_val"] == pytest.approx((mean_ov - mean_res) ** 2 / (mean_res ** 2 + 1e-8), rel=1e-4)
    cross = np.std(np.stack([x[:, 0] for x in d["price"]], 1), axis=1)
    assert v["vac_val"] == pytest.approx(np.mean(np.maximum(cross - 0.1, 0)), rel=1e-4)


def test_two_horizons_have_no_interior_smoothness_term_and_three_reproduce_it():
    """target_smoothness compares an interior horizon with both neighbours: none at N = 2 (it is 0 there);
    at N = 3 it is the h1 term of the reference coherence penalty."""
    for n in (2, 3):
        d = _inputs(n)
        m = _loss_model(n)
        c = _vals(_loss(m, d))
        p = [x[:, 0] for x in d["price"]]
        pairs = HZ.neighbour_pairs(n)
        dd = 1.0 - np.mean([np.mean(np.sign(p[a]) == np.sign(p[b])) for a, b in pairs])
        mag = (2.0 / (n - 1)) * sum(np.mean(np.maximum(np.abs(p[a]) - np.abs(p[b]), 0)) for a, b in pairs)
        sgn = np.sign(d["y"] * 261.0 + 3.2)
        ts = 0.0 if n == 2 else float(np.mean((sgn[:, 1] == sgn[:, 0]) ^ (sgn[:, 1] == sgn[:, 2])))
        assert c["coherence_penalty_val"] == pytest.approx((dd + mag + ts) / 3.0, rel=1e-4), n


@pytest.mark.parametrize("n", N_ALL)
def test_direction_labels_per_horizon_and_pnl_utility(n):
    from neural_trade.losses.functions import pnl_utility
    from neural_trade.metrics.tf_direction import direction_labels_tf

    d = _inputs(n)
    y_raw = tf.constant(d["y"] * 261.0 + 3.2)
    out = direction_labels_tf(y_raw, tf.constant(d["lc"][:, 0]), 1.0)
    assert len(out) == 2 * n
    for i in range(n):
        ret = (d["y"][:, i] * 261.0 + 3.2) / (d["lc"][:, 0] + 1e-8)
        np.testing.assert_array_equal(out[i].numpy(), (np.abs(ret) > 1e-4).astype(np.float32))
        np.testing.assert_array_equal(out[n + i].numpy(), (ret > 1e-4).astype(np.float32))
    m = _loss_model(n, LAMBDA_PNL=1.0)
    heads = []
    for p, q, v in zip(d["price"], d["dirs"], d["var"]):
        heads += [tf.constant(p), tf.constant(q), tf.constant(v)]
    c = pnl_utility(m, tf.constant(d["x"]), tf.constant(d["y"]), heads, tf.constant(d["lc"]), tf.constant(d["ext"]))
    assert type(c) is HZ.loss_components_type(n) and np.isfinite(float(c.pnl_val)) and float(c.pnl_val) != 0.0


@pytest.mark.parametrize("n", N_ALL)
def test_train_and_test_step_log_every_horizon(n):
    m = _loss_model(n)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    d = _inputs(n)
    data = (tf.constant(d["x"]), tf.constant(d["y"]), tf.constant(d["lc"]), tf.constant(d["ext"]))
    m.train_step(data)
    logs = {k: float(v) for k, v in m.train_epoch_logs().items()}
    m.test_step(data)
    val = {k: float(v) for k, v in m._epoch_results("", "gauss_", False).items()}
    for i in range(n):
        h = f"h{i}"
        for key in (f"point_{h}", f"nll_{h}", f"train_dir_mcc_{h}", f"pit_ks_{h}", f"dir_n_{h}",
                    f"var_at_floor_{h}", f"masked_head_price_{h}"):
            assert key in logs, key
        assert f"dir_mcc_{h}" in val and f"gauss_dir_mcc_{h}" in val
    assert f"point_h{n}" not in logs


# ------------------------------------------------------------------ calibration and serving (arrays)
def _preds(n, rng, size=3000):
    keys = HZ.horizon_keys(n)
    y = rng.normal(0.0, 100.0, (size, n))
    d = {h: 0.3 * y[:, i] + rng.normal(0, 80.0, size) for i, h in enumerate(keys)}
    p = {h: 1 / (1 + np.exp(-(y[:, i] / 100.0 + rng.normal(0, 1.0, size)))) for i, h in enumerate(keys)}
    v = {h: np.full(size, 1.2) for h in keys}
    return y, {"delta": d, "direction_prob": p, "variance": v}


@pytest.mark.parametrize("n", (2, 4))
def test_calibration_pipeline_fits_saves_and_loads_n_horizons(n, tmp_path):
    from neural_trade.calibration.pipeline import CalibrationPipeline

    rng = np.random.default_rng(0)
    y, preds = _preds(n, rng)
    lc = np.full(len(y), 100_000.0)
    pipe = CalibrationPipeline(conformal_scale="sigma", shrink_delta=True).fit_from_arrays(
        preds, y, lc, deadband_bps=1.0, pred_scale=100.0, horizon_steps=_steps(n))
    keys = HZ.horizon_keys(n)
    assert pipe.horizons == keys and list(pipe.temperature_scaler.temperatures) == list(keys)
    out = pipe.apply(preds)
    assert list(out["intervals"]) == list(keys) and all(0 <= pipe.delta_scale[h] <= 1 for h in keys)
    pipe.save(str(tmp_path / "cal"))
    back = CalibrationPipeline.load(str(tmp_path / "cal"))
    assert back.horizons == keys
    again = back.apply(preds)
    for h in keys:
        np.testing.assert_allclose(again["direction_prob"][h], out["direction_prob"][h])
        np.testing.assert_allclose(again["intervals"][h][0], out["intervals"][h][0])


@pytest.mark.parametrize("n", (2, 4))
def test_heads_to_predictions_and_the_prediction_batch(n):
    from neural_trade.core.config import Config
    from neural_trade.core.postprocess import heads_to_predictions
    from neural_trade.serving.predictor import PredictionBatch

    rng = np.random.default_rng(1)
    heads = [rng.normal(0, 1, (7, 1)).astype(np.float32) for _ in range(3 * n + 1)]
    preds = heads_to_predictions(heads, 5, 100.0, 1.0, Config())
    keys = HZ.horizon_keys(n)
    assert all(list(preds[k]) == list(keys) for k in ("delta", "direction_prob", "variance"))
    np.testing.assert_allclose(preds["delta"][f"h{n - 1}"], heads[3 * (n - 1)][:5, 0] * 100.0 + 1.0, rtol=1e-6)
    lc = np.full(5, 100_000.0)
    iv = {h: (preds["delta"][h] - 10, preds["delta"][h] + 10) for h in keys}
    batch = PredictionBatch(preds["delta"], preds["direction_prob"], preds["direction_prob"],
                            {h: np.ones(5) for h in keys}, preds["variance"], preds["direction_prob"], iv, lc,
                            tuple(_steps(n)))
    assert batch.horizons == keys
    cols = batch.to_frame().columns
    assert f"h{n - 1}_p_up" in cols and f"h{n}_p_up" not in cols
    frame = batch.to_prediction_frame(100.0)
    assert frame.horizons == keys and frame.y.shape == (5, n)


# ------------------------------------------------------------------ one tiny end-to-end run per N
@pytest.fixture(scope="module", params=(2, 4), ids=("N2", "N4"))
def trained_n(request, tmp_path_factory):
    import os

    import pandas as pd

    from neural_trade.training.trainer import train_and_evaluate

    n = request.param
    tmp = tmp_path_factory.mktemp(f"n{n}")
    rng = np.random.default_rng(0)
    close = 110_000.0 * np.exp(np.cumsum(rng.normal(0.0, 8e-4, 3000)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 15.0, close.shape))
    pd.DataFrame({"datetime": pd.date_range("2025-10-11 02:30:00", periods=len(close), freq="1min", tz="UTC"),
                  "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
                  "close": close, "volume": rng.gamma(2.0, 20.0, close.shape)}).to_csv(tmp / "bars.csv", index=False)
    cfg = _cfg(n, EPOCHS=1, BATCH_SIZE=64, MAX_SEQUENCE_COUNT=1500, PATIENCE=1, EARLY=1, INPUT_SERIES=["close"],
               INDICATOR_FAMILIES={}, MA_SPANS=[5], RSI_PERIODS=[14], BB_PERIODS=[20],
               MACD_SETTINGS=[{"fast": 12, "slow": 26, "signal": 9}],
               CSV_PATH=str(tmp / "bars.csv"), MODEL_PATH=str(tmp / "w.h5"), SCALER_PATH=str(tmp / "s.joblib"),
               ARTIFACTS_DIR=str(tmp / "artifacts"))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        tf.keras.utils.set_random_seed(0)
        result = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=True, fit_calibration=True,
                                    save_artifacts=True)
    finally:
        os.chdir(cwd)
    return n, cfg, result


def test_tiny_training_run_end_to_end(trained_n):
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import evaluate
    from neural_trade.serving.predictor import Predictor
    from neural_trade.strategy import Bars, SignalFrame, backtest, build_backtest_config, build_strategy
    from neural_trade.visualization import model_analytics as MA
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.training_dashboard import training_dashboard

    n, cfg, result = trained_n
    keys = HZ.horizon_keys(n)
    assert all(list(result.predictions[k]) == list(keys) for k in ("delta", "direction_prob", "variance"))
    assert list(result.predictions_calibrated["intervals"]) == list(keys)
    assert result.calibration_pipeline.horizons == keys
    assert set(result.calibration_report) == set(keys)
    hist = result.history.history
    for h in keys:
        assert f"val_dir_mcc_{h}" in hist and f"point_{h}" in hist and f"val_crps_{h}" in hist
    assert all(np.isfinite(v) for v in hist["val_loss"])
    # the calibration pass measured the per-role point weights
    assert {"lambda_short", "lambda_long"} <= set(result.calibration_lambdas or {})

    # serving: the saved bundle reproduces the reported raw heads
    p = Predictor.from_artifacts(result.artifacts_dir)
    batch = p.predict(result.windows_test, result.last_close_test, calibrated=False)
    assert batch.horizons == keys
    for h in keys:
        np.testing.assert_array_equal(batch.direction_prob[h], result.predictions["direction_prob"][h])

    # evaluation, signals, a strategy, figures
    frame = PredictionFrame.from_result(result)
    assert frame.horizons == keys and frame.y.shape[1] == n
    rep = evaluate(frame, cfg)
    assert rep.horizons == keys and "mag_order_full" in rep.model["coherence"]
    sig = SignalFrame.build(frame, 1.0)
    assert sig.p.shape[1] == n and sig.primary == n // 2
    strat = build_strategy("calibrated_quantile", None, calibration=sig)
    res = backtest(sig, Bars.from_close(frame.last_close), strat, build_backtest_config({"random_seeds": 0}))
    assert np.isfinite(res.summary["net_pnl"])
    fig = MA.direction_analytics(frame, cfg)
    assert T.horizon_color(keys[-1]).lower() in fig.to_json().lower()
    td = training_dashboard(result.history.history, cfg)
    assert T.horizon_color(keys[-1]).lower() in td.to_json().lower()


# ------------------------------------------------------------------ stored three-horizon runs still load
LEGACY_BUNDLE = "tests/data/legacy_bundle_426de4f"
STORED_RUN = "runs/scenarios/reference_default/20260929T094723Z-82a848f-7d9a6fef-default__f-1__s2"


def test_a_stored_three_horizon_bundle_loads_with_three_horizons(repo_root):
    from neural_trade.training.artifacts import ArtifactBundle

    bundle = ArtifactBundle.load(repo_root / LEGACY_BUNDLE)
    assert len(bundle.config.HORIZON_STEPS) == 3
    assert bundle.calibration_pipeline.horizons == HZ.REFERENCE_HORIZONS
    model = bundle.build_model()
    assert len(model.outputs) == 10          # the weights file of a pre-NT-042 run fits the rebuilt towers


def test_stored_metrics_and_eval_report_read_as_three_horizons(repo_root):
    import json

    from neural_trade.evaluation.report import EvalReport, health_block
    from neural_trade.visualization.training_dashboard import history_frame, history_horizons

    run = repo_root / STORED_RUN
    if not (run / "eval_report_test.json").is_file():
        pytest.skip(f"{STORED_RUN} is not in this checkout")
    rep = EvalReport.from_json(run / "eval_report_test.json")
    assert rep.horizons == HZ.REFERENCE_HORIZONS
    md = rep.to_markdown()
    assert "| | h0 | h1 | h2 | all 3 |" in md and "full chain h0 <= h1 <= h2" in md
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert history_horizons(history_frame(rows)) == HZ.REFERENCE_HORIZONS
    health = health_block(rows)
    assert {"dir_n_h0", "dir_n_h1", "dir_n_h2"} <= set(health["per_epoch"][0])
