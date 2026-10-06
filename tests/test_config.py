"""Typed Config (plan section B2): legacy names/defaults preserved, validation, override,
YAML round trip, per-instance mutable defaults."""
from __future__ import annotations

from pathlib import Path

import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError

# Every setting of the pre-package model.Config with its default (extracted from the
# class before the move). The typed Config must keep all of them, except LAMBDA_VOL and
# LAMBDA_SOFT_ECE, which D-057 set to 0 (NT-117).
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
    "MOMENTUM_CLIP_MIN": 2.0, "EWMA_IMPL": "matrix", "MOMENTUM_CLIP_MAX": None, "USE_HUBER": True,
    "LAMBDA_SHORT": 1.0, "LAMBDA_POINT": 1.0, "LAMBDA_LONG": 1.0, "LAMBDA_DIR": 1.0, "LAMBDA_INTER": 1.0,
    "LAMBDA_VOL": 0.0, "LAMBDA_VAR": 1.0, "LAMBDA_TREND_OUTER": 1.0, "LAMBDA_DIR_OUTER": 1.0,
    "LAMBDA_DIR_ALIGN_OUTER": 0.0, "LAMBDA_COHERENCE": 1.0, "LAMBDA_NLL_OUTER": 1.0, "LAMBDA_CRPS": 1.0,
    "LAMBDA_SOFT_ECE": 0.0, "T_PERP_DIM": 16, "LAMBDA_T_PERP": 0.1, "LAMBDA_CASIMIR": 0.1,
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


def test_loss_prune_defaults_ship_at_zero():
    """NT-117 (D-057): LAMBDA_VOL and LAMBDA_SOFT_ECE are 0 in Config and configs/default.yaml."""
    import yaml
    from pathlib import Path

    cfg = Config()
    assert cfg.LAMBDA_VOL == 0.0
    assert cfg.LAMBDA_SOFT_ECE == 0.0
    repo = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((repo / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert float(data["LAMBDA_VOL"]) == 0.0
    assert float(data["LAMBDA_SOFT_ECE"]) == 0.0


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
    assert Config(LOOKBACK=32).momentum_clip_max == 32
    assert Config(LOOKBACK=32, MOMENTUM_CLIP_MAX=20).momentum_clip_max == 20
    assert Config(LOOKBACK=32).MOMENTUM_CLIP_MAX is None  # resolved at use, not stored (NT-125)


def test_momentum_clip_max_follows_lookback_on_every_override_path():
    """NT-125: the ceiling is resolved at use, so every path that changes LOOKBACK moves it; an
    explicit value survives each path."""
    from neural_trade.cli import _load_config

    default_yaml = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    paths = {
        "override": lambda **kw: Config().override(**kw),
        "copy": lambda **kw: Config().copy(**kw),
        "yaml+override": lambda **kw: Config.from_yaml(default_yaml).override(**kw),
        "cli": lambda **kw: _load_config(str(default_yaml), kw),
        "cli-nofile": lambda **kw: _load_config(None, kw),
    }
    for name, make in paths.items():
        assert make(LOOKBACK=120).momentum_clip_max == 120, name
        assert make(LOOKBACK=120, MOMENTUM_CLIP_MAX=20).momentum_clip_max == 20, name
    explicit = Config(MOMENTUM_CLIP_MAX=20)
    assert explicit.copy(LOOKBACK=120).momentum_clip_max == 20
    assert Config().override(LOOKBACK=120).override(LOOKBACK=90).momentum_clip_max == 90




def test_default_yaml_has_no_drift_from_config_defaults():
    """NT-125: every default.yaml value equals Config()'s, key by key, and no key is missing or
    unknown; MOMENTUM_CLIP_MAX is null so the ceiling follows LOOKBACK."""
    import yaml

    repo = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((repo / "configs" / "default.yaml").read_text(encoding="utf-8"))
    defaults = Config().to_dict()
    assert data["MOMENTUM_CLIP_MAX"] is None
    assert set(data) <= set(defaults), set(data) - set(defaults)  # a missing key means the default
    drift = {k: (data[k], defaults[k]) for k in data if data[k] != defaults[k]}
    assert not drift, drift
    assert Config.from_yaml(repo / "configs" / "default.yaml").override(LOOKBACK=120).momentum_clip_max == 120


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


# ----------------------------------------------------------------------------- NT-141: validate gaps
class _Cap:
    """The package logger does not propagate (core/logging.py), so pytest's caplog never sees it."""

    def __init__(self):
        import logging

        self.handler = logging.Handler()
        self.records = []
        self.handler.emit = self.records.append
        self.log = logging.getLogger("neural_trade.core.config")

    def __enter__(self):
        import logging

        self.old = self.log.level
        self.log.addHandler(self.handler)
        self.log.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.log.removeHandler(self.handler)
        self.log.setLevel(self.old)


@pytest.fixture
def caplog():
    with _Cap() as cap:
        yield cap


@pytest.mark.parametrize("overrides, match", [
    # period bounds: [MOMENTUM_CLIP_MIN, the resolved ceiling]
    ({"MA_SPANS": [1, 10, 30]}, "below MOMENTUM_CLIP_MIN"),
    ({"BB_PERIODS": [0.5, 20, 25]}, "below MOMENTUM_CLIP_MIN"),
    ({"INDICATOR_FAMILIES": {"stoch": [{"k_period": 14, "d_period": 1}]}}, "below MOMENTUM_CLIP_MIN"),
    # MACD fast >= slow
    ({"MACD_SETTINGS": [{"fast": 26, "slow": 12, "signal": 9}]}, "fast"),
    ({"MACD_SETTINGS": [{"fast": 12, "slow": 12, "signal": 9}]}, "fast"),
    # MOMENTUM_CLIP_MIN must exceed 1
    ({"MOMENTUM_CLIP_MIN": 1.0}, "MOMENTUM_CLIP_MIN"),
    ({"MOMENTUM_CLIP_MIN": 0.5}, "MOMENTUM_CLIP_MIN"),
    # schedules
    ({"LOSS_WEIGHT_SCHEDULE": {"lambda_nope": {0: 0.1}}}, "unknown loss weight"),
    ({"LOSS_WEIGHT_SCHEDULE": {"hd": {0: 0.1}}}, "unknown loss weight"),
    ({"LOSS_WEIGHT_SCHEDULE": {"lambda_hd": {1.5: 0.1}}}, "integer"),
    ({"LOSS_WEIGHT_SCHEDULE": {"lambda_hd": {"soon": 0.1}}}, "integer"),
    ({"LOSS_WEIGHT_SCHEDULE": {"lambda_hd": {0: "x"}}}, "number"),
    # ties, thresholds, clamps
    ({"HORIZON_STEPS": [10, 10, 20], "EXTENDED_TREND_PERIODS": [10, 15, 20]}, "strictly ascending"),
    ({"RHO_MAX": 1.0}, "RHO_MAX"),
    ({"CALIB_LAMBDA_MIN": 5.0, "CALIB_LAMBDA_MAX": 2.0}, "CALIB_LAMBDA_MIN"),
    # a partial LAYERS dict names the missing roles
    ({"LAYERS": {"indicators": "learnable_indicators"}}, "missing .*energy_gate"),
])
def test_validate_refuses_the_nt141_gaps(overrides, match):
    with pytest.raises(InvalidConfigurationError, match=match):
        Config(**overrides)


def test_valid_edge_values_of_the_nt141_rules_still_validate():
    Config(MA_SPANS=[2, 30, 60], MACD_SETTINGS=[{"fast": 12, "slow": 26, "signal": 9}],
           MOMENTUM_CLIP_MIN=1.5, CALIB_LAMBDA_MIN=1.0, CALIB_LAMBDA_MAX=1.0, RHO_MAX=0.999,
           LOSS_WEIGHT_SCHEDULE={"lambda_hd": {0: 0.0, "3": 0.1}})
    # the ceiling is resolved at use (NT-125): periods up to a larger LOOKBACK are fine
    Config(LOOKBACK=120, MA_SPANS=[5, 100, 120])


def test_a_period_above_the_ceiling_warns_but_loads(caplog):
    """Small-LOOKBACK configs keep the default periods and tests set small explicit ceilings; the clip moves
    the period, so the config is warned, not refused (NT-141 report)."""
    if True:
        Config(LOOKBACK=20)
        Config(LOOKBACK=32, MOMENTUM_CLIP_MAX=20)
    assert sum("period ceiling" in r.getMessage() for r in caplog.records) >= 2


def test_schedulable_keys_match_the_training_lambdas():
    from neural_trade.core.config import _LAYER_ROLES, _SCHEDULABLE_LAMBDA_KEYS
    from neural_trade.training.lambdas import _LAMBDA_VARIABLE_KEYS

    assert tuple(_SCHEDULABLE_LAMBDA_KEYS) == tuple(_LAMBDA_VARIABLE_KEYS)
    assert set(_LAYER_ROLES) == set(Config().LAYERS)


def test_lambda_schedule_callback_refuses_unknown_names():
    from neural_trade.training.callbacks import LambdaScheduleCallback

    assert LambdaScheduleCallback({"lambda_hd": {0: 0.1}}).schedule == {"lambda_hd": {0: 0.1}}
    with pytest.raises(KeyError, match="lambda_typo"):
        LambdaScheduleCallback({"lambda_typo": {0: 0.1}})


def test_coercion_refuses_a_fraction_for_an_int_and_none_for_a_str():
    assert Config().override(EPOCHS="12", SEED="1e1").SEED == 10
    with pytest.raises(InvalidConfigurationError, match="cannot interpret"):
        Config().override(EPOCHS="1.5")
    with pytest.raises(InvalidConfigurationError, match="cannot interpret"):
        Config().override(DATA_LOADER=None)
    assert Config().override(CALIB_DAMPING_DIR=None).CALIB_DAMPING_DIR is None  # Optional stays None


def test_var_floor_warns_through_logging_not_a_hidden_deprecation_warning(caplog):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a DeprecationWarning would raise here
        Config(VAR_FLOOR=1e-5)
    assert any("VAR_FLOOR" in r.getMessage() for r in caplog.records)


def test_indicator_l2_is_not_tunable():
    assert Config.field_specs()["INDICATOR_L2"].tunable is False


def test_every_committed_config_and_scenario_still_validates():
    """default.yaml, every scenario and every screen spec: the base config of each cell goes through validate."""
    root = Path(__file__).resolve().parents[1] / "configs"
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.screen import ScreenSpec

    checked = 0
    for path in sorted(root.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if path.name == "default.yaml":
            Config.from_yaml(path)
        elif "screens" in path.parts:
            spec = ScreenSpec.from_yaml(path)
            assert isinstance(spec.base(), Config)  # builds (and validates) the base Config
        elif path.parent.name == "scenarios" and "schema_version" in text:
            Scenario.from_yaml(path).validate()
        else:
            continue  # an ablation spec, a compare, a strategy file: not a Config carrier
        checked += 1
    assert checked >= 20
