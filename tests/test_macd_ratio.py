"""MACD ratio parametrisation (NT-106; D-045 B_model_indicators.md 4.1, 7 item 8).

Config.MACD_PARAM = "ratio" replaces the independently-learned fast leg by
fast = 1 + r * (slow_eff - 1), r its own learned logit; this file pins (1) the config switch
and its default, (2) the structural fast < slow invariant for extreme logits and meta shifts,
(3) that the layer and the model build and run under the switch, (4) that the applied-period
report (NT-097, evaluation/applied_periods.py) and get_learned_parameters() still read sensible
fast/slow/signal periods under it, and (5) that the default ("independent") path is untouched
(scripts/golden_run.py pins the bit-for-bit default-config claim end to end).
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.evaluation.applied_periods import applied_period_samples, applied_period_stats
from neural_trade.indicators import Indicators
from neural_trade.indicators.families import _macd_ratio_periods
from neural_trade.models.layers import LearnableIndicators
from neural_trade.models.layers.learnable_indicators import _RATIO_CLIP_MAX, _RATIO_CLIP_MIN
from neural_trade.models.registry import Models

OLD = dict(INPUT_SERIES=["close"], INDICATOR_FAMILIES={})  # the pre-NT-047 four families
B, L = 8, 60


def _x(seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(np.cumsum(rng.normal(0, 1, (B, L)), axis=1).astype(np.float32))


def _model(cfg):
    return Models.build(getattr(cfg, "MODEL_NAME", None), cfg)


# ---------------------------------------------------------------- the config switch


def test_macd_param_default_is_independent():
    cfg = Config()
    assert cfg.MACD_PARAM == "independent"


def test_macd_param_accepts_ratio_and_rejects_garbage():
    Config(MACD_PARAM="ratio")  # does not raise
    Config(MACD_PARAM="RATIO")  # ignore_case
    with pytest.raises(InvalidConfigurationError):
        Config(MACD_PARAM="something_else")


def test_macd_ratio_family_is_registered_and_contract_valid():
    fam = Indicators.get("macd_ratio")
    assert fam.name == "macd_ratio"
    assert set(p.name for p in fam.params) == {"slow", "ratio", "signal"}
    assert len(fam.channels) == 4


# ---------------------------------------------------------------- the structural invariant


def _extreme_alphas(slow_logit, ratio_logit, meta_shift, signal_logit=0.0):
    return {
        "slow": tf.sigmoid(tf.constant([slow_logit + meta_shift], dtype=tf.float32)),
        "ratio": tf.sigmoid(tf.constant([ratio_logit + meta_shift], dtype=tf.float32)),
        "signal": tf.sigmoid(tf.constant([signal_logit + meta_shift], dtype=tf.float32)),
    }


LOGITS = (-1e6, -1e4, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1e4, 1e6)
META_SHIFTS = (-1e6, -1.0, 0.0, 1.0, 1e6)


@pytest.mark.parametrize("slow_logit", LOGITS)
@pytest.mark.parametrize("ratio_logit", LOGITS)
@pytest.mark.parametrize("meta_shift", META_SHIFTS)
def test_fast_less_than_slow_always_holds_under_extreme_logits_and_meta_shifts(
        slow_logit, ratio_logit, meta_shift):
    alphas = _extreme_alphas(slow_logit, ratio_logit, meta_shift)
    fast, slow_eff, signal = _macd_ratio_periods(alphas)
    f, s = float(fast.numpy()[0]), float(slow_eff.numpy()[0])
    assert np.isfinite(f) and np.isfinite(s)
    assert f < s, (slow_logit, ratio_logit, meta_shift, f, s)
    assert f >= 1.0 - 1e-6, (slow_logit, ratio_logit, meta_shift, f)
    assert np.isfinite(float(signal.numpy()[0]))


def test_fast_reaches_close_to_p_equals_1_as_r_shrinks():
    """As r -> 0 the fast leg approaches the raw price (period 1), which 'independent' mode's
    hard floor at MOMENTUM_CLIP_MIN (2 bars) can never reach (B_model_indicators.md 7 item 8)."""
    alphas = _extreme_alphas(slow_logit=0.0, ratio_logit=-1e6, meta_shift=0.0)
    fast, slow_eff, _ = _macd_ratio_periods(alphas)
    assert float(fast.numpy()[0]) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- layer / model wiring


def test_layer_builds_and_runs_under_ratio_mode():
    cfg = Config(**OLD, MACD_PARAM="ratio")
    layer = LearnableIndicators(cfg)
    meta = tf.zeros((B, 18), dtype=tf.float32)  # OLD's 18 logits (ma 3 + macd 9 + rsi 3 + bb 3)
    out = layer((_x(), meta))
    assert out.shape == (B, L, 31)  # ma 3 + macd 3x4 + rsi 3 + bb 3x4 + raw close, unchanged by NT-106


def test_model_builds_and_predicts_under_ratio_mode():
    cfg = Config(**OLD, MACD_PARAM="ratio")
    model = _model(cfg)
    x = _x()
    out = model(x, training=False)
    assert out is not None


# --------------------------------- repair round 1: the ratio leg must be clipped AS A FRACTION


def _vars_by_kind(layer):
    """``{"ratio": [(family, name, var), ...], "period": [...]}`` for every learned parameter."""
    out = {"ratio": [], "period": []}
    for family, _insts, varmaps in layer._families:
        kind_of = {p.name: getattr(p, "kind", "period") for p in family.params}
        for vm in varmaps:
            for name, var in vm.items():
                out[kind_of[name]].append((family.name, name, var))
    return out


@pytest.mark.parametrize("extreme_logit", [-1e6, 1e6])
def test_clip_learned_periods_and_bound_applied_clip_ratio_as_a_fraction_not_a_period(extreme_logit):
    """QA of NT-106 found that clip_learned_periods (and INDICATOR_BOUND_APPLIED's per-window
    clip in ``_alpha``) treated the 'ratio' parameter as a period, re-flooring r into
    [0.0328, 0.6667] at the MOMENTUM_CLIP_MIN/MAX defaults and never letting the fast leg below
    about 1.82 - defeating the item's purpose (B_model_indicators.md 4.1: fast legs pinned at
    the floor of 2). This pins the fix: pushing the ratio logit to +-1e6, then running
    clip_learned_periods and a forward pass with INDICATOR_BOUND_APPLIED on, must (a) let
    fast reach close to p = 1 when the ratio logit was driven very negative (r -> 0), (b) keep
    fast < slow regardless, and (c) leave every PERIOD-kind logit's clip exactly as it always
    was (inside [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX])."""
    cfg = Config(**OLD, MACD_PARAM="ratio", INDICATOR_BOUND_APPLIED=True,
                 MOMENTUM_CLIP_MIN=2.0, MOMENTUM_CLIP_MAX=60.0)
    layer = LearnableIndicators(cfg)
    meta = tf.zeros((B, 18), dtype=tf.float32)
    layer([_x(), meta])  # builds the weights

    by_kind = _vars_by_kind(layer)
    ratio_vars = [v for fam, name, v in by_kind["ratio"] if fam == "macd_ratio" and name == "ratio"]
    period_vars = by_kind["period"]
    assert len(ratio_vars) == 3  # the three macd_ratio instances

    for var in ratio_vars:
        var.assign(extreme_logit)

    layer.clip_learned_periods(cfg.MOMENTUM_CLIP_MIN, cfg.momentum_clip_max)

    # (c) every PERIOD-kind logit is still clipped into [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX],
    # exactly as before this repair (the ratio vars were never touched by this assertion before
    # either, so period behaviour for 'independent' mode and macd_ratio's 'slow'/'signal' is
    # provably unaffected by the kind-aware branching)
    for _fam, _name, var in period_vars:
        p = float(layer._period_from_logit(var).numpy())
        assert cfg.MOMENTUM_CLIP_MIN - 1e-3 <= p <= cfg.MOMENTUM_CLIP_MAX + 1e-3

    # the ratio var itself is clipped as a FRACTION, not re-floored into the period-implied
    # alpha interval [2/61, 2/3] ~= [0.033, 0.667]
    for var in ratio_vars:
        r = float(tf.sigmoid(var).numpy())
        assert _RATIO_CLIP_MIN - 1e-6 <= r <= _RATIO_CLIP_MAX + 1e-6
        if extreme_logit < 0:
            assert r == pytest.approx(_RATIO_CLIP_MIN, abs=1e-6)
        else:
            assert r == pytest.approx(_RATIO_CLIP_MAX, abs=1e-6)

    # the forward pass (applied_period_samples, which reuses _alpha exactly as call() does,
    # INDICATOR_BOUND_APPLIED on) shows the same effect per window
    applied = layer.applied_period_samples(meta)
    for i in range(3):
        fast = applied[f"macd_{i}_fast"]
        slow = applied[f"macd_{i}_slow"]
        assert np.all(fast < slow), i
        if extreme_logit < 0:
            assert np.all(fast < 1.1), (i, fast)  # (a): p = 1 is reachable, not floored near 2


def test_bound_applied_clips_ratio_as_a_fraction_without_clip_learned_periods():
    """The per-window INDICATOR_BOUND_APPLIED clip in ``_alpha`` (not just the post-optimizer-
    step clip_learned_periods) must not period-clip the ratio parameter either."""
    cfg = Config(**OLD, MACD_PARAM="ratio", INDICATOR_BOUND_APPLIED=True,
                 MOMENTUM_CLIP_MIN=2.0, MOMENTUM_CLIP_MAX=60.0)
    layer = LearnableIndicators(cfg)
    meta = tf.zeros((B, 18), dtype=tf.float32)
    layer([_x(), meta])
    ratio_vars = [v for fam, name, v in _vars_by_kind(layer)["ratio"]
                 if fam == "macd_ratio" and name == "ratio"]
    for var in ratio_vars:
        var.assign(-1e6)  # r -> 0, no clip_learned_periods call this time
    applied = layer.applied_period_samples(meta)
    for i in range(3):
        fast, slow = applied[f"macd_{i}_fast"], applied[f"macd_{i}_slow"]
        assert np.all(fast < slow), i
        assert np.all(fast < 1.1), (i, fast)


# ---------------------------------------------------------------- reporting (NT-097 + NT-106)


def test_get_learned_parameters_reports_sensible_fast_slow_signal_under_ratio_mode():
    cfg = Config(**OLD, MACD_PARAM="ratio")
    layer = LearnableIndicators(cfg)
    layer([_x(), tf.zeros((B, 18), dtype=tf.float32)])  # builds the layer's weights
    learned = layer.get_learned_parameters()
    for i in range(3):
        fast = learned[f"macd_{i}_fast"]
        slow = learned[f"macd_{i}_slow"]
        signal = learned[f"macd_{i}_signal"]
        assert 1.0 <= fast < slow, (i, fast, slow)
        assert signal > 0
        assert all("ratio" not in k for k in learned if k.startswith(f"macd_{i}_"))


def test_applied_period_samples_reports_fast_less_than_slow_on_real_windows():
    cfg = Config(**OLD, MACD_PARAM="ratio")
    model = _model(cfg)
    x = _x(seed=5).numpy()
    samples = applied_period_samples(model, x)
    for i in range(3):
        fast = samples[f"macd_{i}_fast"]
        slow = samples[f"macd_{i}_slow"]
        assert fast.shape == (B,)
        assert np.all(np.isfinite(fast)) and np.all(np.isfinite(slow))
        assert np.all(fast < slow), i


def test_applied_period_stats_p5_p50_p95_ordered_under_ratio_mode():
    cfg = Config(**OLD, MACD_PARAM="ratio")
    model = _model(cfg)
    x = _x(seed=6).numpy()
    stats = applied_period_stats(model, x)
    for name in (f"macd_{i}_{p}" for i in range(3) for p in ("fast", "slow", "signal")):
        assert name in stats
        s = stats[name]
        assert s["p5"] <= s["p50"] <= s["p95"], (name, s)


# ---------------------------------------------------------------- the default is untouched


def test_independent_mode_report_keys_and_formula_unchanged():
    """The 'independent' family has no applied_report hook: get_learned_parameters still uses
    the identity 2/alpha - 1 map, exactly as before NT-106 (bit-for-bit; the golden run is the
    full-pipeline pin, scripts/golden_run.py record/verify)."""
    cfg = Config(**OLD)  # MACD_PARAM defaults to "independent"
    layer = LearnableIndicators(cfg)
    layer([_x(), tf.zeros((B, 18), dtype=tf.float32)])  # builds the layer's weights
    learned = layer.get_learned_parameters()
    assert learned["macd_0_fast"] == pytest.approx(12.0, abs=1e-3)
    assert learned["macd_0_slow"] == pytest.approx(26.0, abs=1e-3)
    assert learned["macd_0_signal"] == pytest.approx(9.0, abs=1e-3)
    assert not hasattr(Indicators.get("macd"), "applied_report")


def test_building_with_default_macd_param_never_touches_the_ratio_family():
    """build() resolves the registry name from Config.MACD_PARAM; off (default) must resolve
    'macd' (today's family), never 'macd_ratio', even though both are registered."""
    cfg = Config(**OLD)
    layer = LearnableIndicators(cfg)
    layer([_x(), tf.zeros((B, 18), dtype=tf.float32)])  # builds the layer's weights
    families_used = {family.name for family, _insts, _varmaps in layer._families}
    assert "macd" in families_used and "macd_ratio" not in families_used
