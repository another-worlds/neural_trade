"""Typed Config (plan section B2): legacy names/defaults preserved, validation, override,
YAML round trip, per-instance mutable defaults."""
from __future__ import annotations

import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError

# Every setting of the pre-package model.Config with its default (extracted from the
# class before the move). The typed Config must keep all of them, unchanged.
LEGACY_DEFAULTS = {
    "CSV_PATH": "binance_btcusdt_1min_ccxt.csv", "LOOKBACK": 60, "WINDOW_STEP": 1, "RESAMPLE_MINUTES": 1,
    "BATCH_SIZE": 256, "EPOCHS": 20,  # BATCH_SIZE: 64 before the GPU-speed work
    "LR": 0.001, "PATIENCE": 3, "EARLY": 6, "MAX_SEQUENCE_COUNT": 53280,
    "VAL_FRACTION": 0.066, "CAL_FRACTION": 0.066, "N_FOLDS": 5, "EXTENDED_TREND_PERIODS": [10, 15, 20],
    "HORIZON_STEPS": [10, 15, 20], "DAMPING": 0.5, "CALIB_WARMUP_FRACTION": 0.05,
    "CALIB_SAMPLE_FRACTION": 0.1, "CALIB_LAMBDA_MIN": 0.1, "CALIB_LAMBDA_MAX": 20.0, "CALIB_DAMPING": 1,
    "CALIB_DAMPING_POINT": None, "CALIB_DAMPING_TREND": 0.0, "CALIB_DAMPING_DIR": None,
    "CALIB_DAMPING_VAR": None, "CALIB_DAMPING_CRPS": None, "CALIB_DAMPING_ECE": None,
    "CALIB_DAMPING_VOL": None, "CALIB_DAMPING_PHYSICS": 0.0, "CALIB_OUTER": False,
    "LAMBDA_LOCAL_TREND": 1.0, "LAMBDA_GLOBAL_TREND": 1.0, "LAMBDA_EXTENDED_TREND": 0.1,
    "LAMBDA_QUANTILE": 1.0, "REG_MOMENTUM_L2": 0, "INDICATOR_L2": 0, "INDICATOR_LR_MULT": 5.0,
    "MOMENTUM_CLIP_MIN": 2.0, "EWMA_IMPL": "matrix", "MOMENTUM_CLIP_MAX": 60, "USE_HUBER": True,
    "LAMBDA_SHORT": 1.0, "LAMBDA_POINT": 1.0, "LAMBDA_LONG": 1.0, "LAMBDA_DIR": 1.0, "LAMBDA_INTER": 1.0,
    "LAMBDA_VOL": 1.0, "LAMBDA_VAR": 1.0, "LAMBDA_TREND_OUTER": 1.0, "LAMBDA_DIR_OUTER": 1.0,
    "LAMBDA_DIR_ALIGN_OUTER": 0.0, "LAMBDA_COHERENCE": 1.0, "LAMBDA_NLL_OUTER": 1.0, "LAMBDA_CRPS": 1.0,
    "LAMBDA_SOFT_ECE": 1.0, "T_PERP_DIM": 16, "LAMBDA_T_PERP": 0.1, "LAMBDA_CASIMIR": 0.1,
    "LAMBDA_VAC": 0.0, "LAMBDA_HD": 0.1, "LAMBDA_IFE": 0.1, "RHO_MAX": 0.95, "VACUUM_E_MAX": 1.0,
    "LAMBDA_VAC_OVERFLOW": 0.1, "MODEL_PATH": "nn_learnable_indicators_v3.weights.h5",
    "SCALER_PATH": "scaler_v3.joblib", "MA_SPANS": [5, 10, 30],
    "MACD_SETTINGS": [{"fast": 12, "slow": 26, "signal": 9}, {"fast": 5, "slow": 35, "signal": 5},
                      {"fast": 8, "slow": 17, "signal": 9}],
    "RSI_PERIODS": [9, 14, 21], "BB_PERIODS": [10, 20, 25], "TANH_SCALE": 1.0, "HUBER_DELTA": 1.0,
    "SIGMOID_SCALE": 1.0, "INDICATOR_GRAD_MULT": 5.0, "GRAD_CLIP_NORM": 20.0, "FOCAL_ALPHA": 0.5,
    "FOCAL_GAMMA": 2, "DIR_DEADBAND_BPS": 5.0, "VAR_FLOOR": 0.0001, "VAR_CAP": 1000.0,
    "DELTA_MAPE_MIN_ABS": 1.0, "LAMBDA_DIR_ALIGN": 0.7,
}


def test_every_legacy_setting_keeps_its_name_and_default():
    cfg = Config()
    missing = [k for k in LEGACY_DEFAULTS if not hasattr(cfg, k)]
    assert not missing
    changed = {k: (v, getattr(cfg, k)) for k, v in LEGACY_DEFAULTS.items() if getattr(cfg, k) != v}
    assert not changed, changed
    assert Config.HOUR == 60 and Config.DAY == 1440


def test_keyword_construction_and_real_fields():
    cfg = Config(EPOCHS=5, LR=3e-4)
    assert cfg.EPOCHS == 5 and cfg.LR == 3e-4
    assert len(cfg.to_dict()) >= len(LEGACY_DEFAULTS)


def test_mutable_defaults_are_per_instance():
    a, b = Config(), Config()
    a.MA_SPANS.append(99)
    a.MACD_SETTINGS[0]["fast"] = 1
    assert b.MA_SPANS == [5, 10, 30] and b.MACD_SETTINGS[0]["fast"] == 12


def test_validation_raises_a_value_error():
    with pytest.raises(ValueError):
        Config(LR=0.0)
    with pytest.raises(InvalidConfigurationError, match="length must match"):
        Config(HORIZON_STEPS=[10, 15, 20], EXTENDED_TREND_PERIODS=[10, 15])
    with pytest.raises(InvalidConfigurationError, match="EWMA_IMPL"):
        Config(EWMA_IMPL="fft")
    with pytest.raises(InvalidConfigurationError, match=">= 0"):
        Config(LAMBDA_HD=-1.0)


def test_override_rejects_unknown_names_with_suggestions():
    cfg = Config()
    with pytest.raises(InvalidConfigurationError, match="did you mean LAMBDA_HD"):
        cfg.override(LAMBDA_HDD=0.0)
    cfg.override(LAMBDA_HD=0.0, EPOCHS="7")
    assert cfg.LAMBDA_HD == 0.0 and cfg.EPOCHS == 7
    with pytest.raises(InvalidConfigurationError):
        cfg.override(EPOCHS=0)


def test_momentum_clip_max_derives_from_lookback():
    assert Config(LOOKBACK=32).MOMENTUM_CLIP_MAX == 32
    assert Config(LOOKBACK=32, MOMENTUM_CLIP_MAX=20).MOMENTUM_CLIP_MAX == 20


def test_yaml_round_trip_including_scientific_notation_strings(tmp_path):
    cfg = Config(EPOCHS=3, ABLATE_LAMBDAS=["LAMBDA_HD"], LOSS_WEIGHT_SCHEDULE={"lambda_hd": {0: 0.0, 2: 0.1}})
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert Config.from_yaml(path).to_dict() == cfg.to_dict()

    path.write_text("LR: 1e-3\nEPOCHS: 4\nSGD_NESTEROV: 'false'\nCALIB_DAMPING_DIR: null\n", encoding="utf-8")
    loaded = Config.from_yaml(path)  # PyYAML reads 1e-3 as the STRING '1e-3'
    assert loaded.LR == 1e-3 and isinstance(loaded.LR, float)
    assert loaded.EPOCHS == 4 and loaded.SGD_NESTEROV is False and loaded.CALIB_DAMPING_DIR is None


def test_lambda_weights_and_ablation_names():
    lw = Config().lambda_weights()
    assert lw["LAMBDA_HD"] == 0.1 and "ABLATE_LAMBDAS" not in lw
    with pytest.raises(InvalidConfigurationError, match="ABLATE_LAMBDAS"):
        Config(ABLATE_LAMBDAS=["LAMBDA_NOPE"])


def test_legacy_import_surface_is_the_same_class():
    import neural_trade.compat as compat

    assert compat.Config is Config


@pytest.mark.parametrize("overrides, match", [
    ({"LOOKBACK": 0}, "LOOKBACK must be positive"),
    ({"LOOKBACK": 2000}, "exceed 1 day"),
    ({"BATCH_SIZE": 8}, "BATCH_SIZE"),
    ({"LR": 2.0}, "LR"),
    ({"EPOCHS": 0}, "EPOCHS"),
    ({"HORIZON_STEPS": [10, 0, 20]}, "positive"),
    ({"HORIZON_STEPS": [20, 15, 10], "EXTENDED_TREND_PERIODS": [10, 15, 20]}, "ascending"),
    ({"EXTENDED_TREND_PERIODS": [20, 15, 10]}, "ascending"),
    ({"HORIZON_STEPS": [10, 20], "EXTENDED_TREND_PERIODS": [10, 20]}, "three horizon towers"),
    ({"VAR_CAP": 1e-5}, "VAR_CAP"),
    ({"VAL_FRACTION": 0.6}, "VAL_FRACTION"),
    ({"FOLD_INDEX": 7}, "FOLD_INDEX"),
    ({"DIR_DEADBAND_BPS": -1.0}, "DIR_DEADBAND_BPS"),
    ({"MOMENTUM_CLIP_MIN": 80}, "MOMENTUM_CLIP_MIN"),
    ({"CONFORMAL_SCALE": "iqr"}, "CONFORMAL_SCALE"),
])
def test_validate_rejects_structurally_invalid_settings(overrides, match):
    """(Ported from the old test_model_math_consistency.py, which asserted the DEFAULTS were
    'reasonable' instead of checking that validate() rejects bad values.)"""
    with pytest.raises(InvalidConfigurationError, match=match):
        Config(**overrides)


def test_values_from_yaml_or_the_cli_are_coerced_to_the_field_type():
    cfg = Config().override(USE_HUBER="off", EPOCHS="12", LR="5e-4", HORIZON_STEPS="[5, 10, 20]",
                            EXTENDED_TREND_PERIODS=(5, 10, 20), CALIB_DAMPING_DIR="null",
                            LOSS_WEIGHT_SCHEDULE="{lambda_hd: {0: 0.0}}")
    assert cfg.USE_HUBER is False and cfg.EPOCHS == 12 and cfg.LR == 5e-4
    assert cfg.HORIZON_STEPS == [5, 10, 20] and cfg.EXTENDED_TREND_PERIODS == [5, 10, 20]
    assert cfg.CALIB_DAMPING_DIR is None and cfg.LOSS_WEIGHT_SCHEDULE == {"lambda_hd": {0: 0.0}}
    for bad in ({"USE_HUBER": "maybe"}, {"EPOCHS": 2.5}, {"HORIZON_STEPS": "7"}, {"LOSS_WEIGHT_SCHEDULE": "[1, 2]"}):
        with pytest.raises(InvalidConfigurationError, match="cannot interpret"):
            Config().override(**bad)


def test_copy_is_independent():
    a = Config()
    b = a.copy().override(EPOCHS=3)
    b.MA_SPANS.append(1)
    assert a.EPOCHS == 20 and a.MA_SPANS == [5, 10, 30]
