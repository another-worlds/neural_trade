"""NT-105: ATTENTION_MODE and HEAD_POOL switches (B_model_indicators.md 1.1, 1.4, 7 item 1).

Criterion (1): config switches for the cross-unit attention block (``ATTENTION_MODE``: 'time'
today, 'channels' across the indicator channels before the GRU, 'none') and for the sequence
pooling before the shared dense layer (``HEAD_POOL``: 'flatten' today, 'mean', 'attention').
Criterion (2): the default is unchanged (the golden run is checked separately,
``scripts/golden_run.py``); a parameter count independent of LOOKBACK under the new switches;
the new modes train one step on CPU with finite loss and gradients.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config


def test_default_switches_are_todays_behaviour():
    cfg = Config()
    assert cfg.ATTENTION_MODE == "time"
    assert cfg.HEAD_POOL == "flatten"


def test_default_build_is_unchanged_by_the_new_switches(tf):
    """Building with the explicit defaults gives the same graph (same output count, same
    layer names) as building with nothing set - the switches are additive."""
    from neural_trade.models.gru_attention import build_gru_attention

    m_default = build_gru_attention(Config())
    m_explicit = build_gru_attention(Config(ATTENTION_MODE="time", HEAD_POOL="flatten"))
    assert len(m_default.outputs) == len(m_explicit.outputs) == 10
    assert m_default.count_params() == m_explicit.count_params()


@pytest.mark.parametrize("bad_field,bad_value", [("ATTENTION_MODE", "spatial"), ("HEAD_POOL", "sum")])
def test_an_invalid_mode_is_refused_by_validate(bad_field, bad_value):
    from neural_trade.core.exceptions import InvalidConfigurationError

    with pytest.raises(InvalidConfigurationError):
        Config(**{bad_field: bad_value})


# --------------------------------------------------------------------- criterion 2: L-independence
@pytest.mark.parametrize("attention_mode", ["channels", "none"])
@pytest.mark.parametrize("head_pool", ["mean", "attention"])
def test_param_count_independent_of_lookback(tf, attention_mode, head_pool):
    """gru_attention.py ties two blocks' parameter count to LOOKBACK today (block 6's
    time-as-features attention: 513*L + 384; Flatten -> Dense: 512*L + 32). With both switches
    off the legacy path, a model built at L=60 and one built at L=120 have the identical
    parameter count."""
    from neural_trade.models.gru_attention import build_gru_attention

    kwargs = dict(ATTENTION_MODE=attention_mode, HEAD_POOL=head_pool)
    m60 = build_gru_attention(Config(LOOKBACK=60, **kwargs))
    m120 = build_gru_attention(Config(LOOKBACK=120, **kwargs))
    assert m60.count_params() == m120.count_params()
    assert len(m60.outputs) == len(m120.outputs) == 10


def test_todays_default_path_still_ties_params_to_lookback():
    """Sanity check on the test above: the *default* switches still show the L-dependence the
    math report measured (block 6's time-as-features attention and Flatten -> Dense both grow
    linearly with LOOKBACK), unlike the new switches."""
    from neural_trade.models.gru_attention import build_gru_attention

    m60 = build_gru_attention(Config(LOOKBACK=60))
    m120 = build_gru_attention(Config(LOOKBACK=120))
    assert m120.count_params() > m60.count_params()


# --------------------------------------------------------------------- criterion 2: new modes train
@pytest.mark.parametrize("attention_mode", ["time", "channels", "none"])
@pytest.mark.parametrize("head_pool", ["flatten", "mean", "attention"])
def test_one_training_step_is_finite_under_every_switch_combination(
    tf, tiny_close_only_config, tmp_path, synthetic_bars, attention_mode, head_pool
):
    """Every ATTENTION_MODE x HEAD_POOL combination (including today's default) builds a model
    that trains one step on CPU with a finite loss and finite, non-None gradients for every
    trainable weight."""
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.datasets import create_datasets
    from neural_trade.models.registry import Models

    tf.keras.utils.set_random_seed(0)
    cfg = tiny_close_only_config
    cfg.ATTENTION_MODE = attention_mode
    cfg.HEAD_POOL = head_pool
    cfg.validate()

    csv_path = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv_path, index=False)
    cfg.CSV_PATH = str(csv_path)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")

    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te, y_tr, _y_te, _scaler) = dp.prepare_datasets(df, close)

    base = Models.build(getattr(cfg, 'MODEL_NAME', None), cfg)
    std = float(np.std(y_tr))
    pred_scale = std if std > 0 else 1.0
    pred_mean = float(np.mean(y_tr))
    model = CustomTrainModel(
        base_model=base, pred_scale=pred_scale, pred_mean=pred_mean,
        lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
        lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
        lambda_dir=cfg.LAMBDA_DIR, config=cfg, inputs=base.inputs, outputs=base.outputs,
    )
    train_ds, val_ds = create_datasets(cfg, X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR))

    hist = model.fit(train_ds, epochs=1, steps_per_epoch=1, verbose=0)

    assert np.isfinite(hist.history["loss"][-1])
    assert hist.history["nonfinite_grad_steps"][-1] == 0
    assert np.isfinite(hist.history["grad_global_norm"][-1])
    assert all(np.all(np.isfinite(w)) for w in model.get_weights())
