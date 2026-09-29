"""Config field metadata (NT-029): every field has a doc, a unit, its range or choices, a tunable
and a deprecated flag; validate enforces the ranges; the config reference is generated from them."""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, fields
from pathlib import Path

import pytest
import yaml

from neural_trade.core.config import UNITS, Config, FieldSpec, _f, metadata_problems
from neural_trade.core.exceptions import InvalidConfigurationError

REPO = Path(__file__).resolve().parent.parent

# The fields that had no doc on 2026-09-28 (NT-029, counted with dataclasses.fields).
FORMERLY_UNDOCUMENTED = [
    "EPOCHS", "CALIB_LAMBDA_MIN", "CALIB_LAMBDA_MAX", "CALIB_DAMPING_POINT", "CALIB_DAMPING_DIR",
    "CALIB_DAMPING_VAR", "CALIB_DAMPING_CRPS", "CALIB_DAMPING_ECE", "CALIB_DAMPING_VOL", "LAMBDA_TREND_OUTER",
    "LAMBDA_DIR_OUTER", "LAMBDA_COHERENCE", "LAMBDA_NLL_OUTER", "LAMBDA_CRPS", "LAMBDA_SOFT_ECE", "MA_SPANS",
    "MACD_SETTINGS", "RSI_PERIODS", "BB_PERIODS", "FOCAL_GAMMA", "MODEL_PATH", "SCALER_PATH", "ADAM_BETA1",
    "ADAM_BETA2", "SGD_MOMENTUM", "SGD_NESTEROV",
]


def test_every_field_has_a_doc_a_unit_flags_and_a_default_inside_its_own_range():
    assert metadata_problems(Config) == []
    specs = Config.field_specs()
    assert list(specs) == [f.name for f in fields(Config)]
    for spec in specs.values():
        assert spec.unit in UNITS and spec.doc.strip() and "\n" not in spec.doc
        assert isinstance(spec.tunable, bool) and isinstance(spec.deprecated, bool)
    assert len(FORMERLY_UNDOCUMENTED) == 26
    assert all(specs[name].doc.strip() for name in FORMERLY_UNDOCUMENTED)


@dataclass
class _BrokenMetadata(Config):
    NO_DOC: int = _f(1, "ops", unit="count")
    NO_UNIT: int = _f(1, "ops", "a doc")
    UNKNOWN_UNIT: int = _f(1, "ops", "a doc", unit="furlongs")
    DEFAULT_BELOW_RANGE: int = _f(0, "ops", "a doc", unit="count", ge=1)
    DEFAULT_NOT_A_CHOICE: str = _f("c", "ops", "a doc", unit="name", choices=("a", "b"))
    TUNABLE_AND_DEPRECATED: float = _f(1.0, "ops", "a doc", unit="weight", ge=0.0, tunable=True, deprecated=True)


def test_the_metadata_check_fails_on_a_missing_doc_or_unit_and_on_a_default_outside_its_range():
    problems = "\n".join(metadata_problems(_BrokenMetadata))
    for name in ("NO_DOC: no doc", "NO_UNIT: unit None", "UNKNOWN_UNIT: unit 'furlongs'",
                 "DEFAULT_BELOW_RANGE: default", "DEFAULT_NOT_A_CHOICE: default", "TUNABLE_AND_DEPRECATED"):
        assert name in problems
    assert len(metadata_problems(_BrokenMetadata)) == 6


def test_resample_minutes_is_not_tunable_until_nt040():
    """The annualisation ignores the bar size until NT-040 (which lifts this), so a sweep must not vary it."""
    spec = Config.field_specs()["RESAMPLE_MINUTES"]
    assert spec.unit == "minutes" and not spec.tunable and "NT-040" in spec.doc


def test_tunable_fields_are_searchable_hyperparameters():
    """A sweep can build a search space from every tunable field; data, blocks, targets, paths, seeds,
    registry keys and deprecated fields are never tunable."""
    specs = Config.field_specs()
    tunable = {n for n, s in specs.items() if s.tunable}
    assert {"LR", "BATCH_SIZE", "LAMBDA_HD", "INDICATOR_LR_MULT"} <= tunable
    for name in tunable:
        s = specs[name]
        assert s.has_range or s.choices, name
        assert s.unit not in ("path", "key", "seed", "index", "mapping"), name
        assert s.group not in ("data", "horizons", "paths", "registries"), name
        assert not s.deprecated, name
    for name in ("RESAMPLE_MINUTES", "SEED", "CSV_PATH", "MODEL_PATH", "MODEL_NAME", "LOOKBACK", "HORIZON_STEPS",
                 "FOLD_INDEX", "EPOCHS", "DIRECTION_LOSS"):
        assert name not in tunable, name


def test_deprecated_fields_are_the_ones_the_code_calls_legacy_or_unused():
    deprecated = {n for n, s in Config.field_specs().items() if s.deprecated}
    assert deprecated == {"DAMPING", "LAMBDA_LOCAL_TREND", "LAMBDA_GLOBAL_TREND", "LAMBDA_QUANTILE", "TANH_SCALE",
                          "SIGMOID_SCALE", "USE_HUBER", "HUBER_DELTA"}


def test_a_deprecated_field_set_away_from_default_warns_and_still_loads():
    """NT-028: none of the deprecated fields is removed, so an old config still loads; setting one
    to a non-default value warns instead of failing."""
    with pytest.warns(DeprecationWarning, match="Config.DAMPING is deprecated"):
        Config(DAMPING=0.9)
    with pytest.warns(DeprecationWarning, match="Config.HUBER_DELTA is deprecated"):
        Config(HUBER_DELTA=2.0)
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        Config()  # every deprecated field at its default: no "is deprecated" warning
    assert not any("is deprecated and has no effect" in str(w.message) for w in caught)


def test_a_field_spec_checks_values_on_its_own():
    below = FieldSpec(name="X", group="ops", doc="d", unit="count", type="float", default=1.0,
                      maximum=5.0, max_inclusive=False)
    assert below.range_text() == "< 5" and below.check(4.0) is None
    assert "outside its range < 5" in below.check(5.0) and "not a number" in below.check("five")
    at_most = FieldSpec(name="X", group="ops", doc="d", unit="count", type="float", default=1.0, maximum=5.0)
    assert at_most.range_text() == "<= 5" and at_most.check(5.0) is None
    free = FieldSpec(name="X", group="ops", doc="d", unit="path", type="str", default="a")
    assert free.range_text() == "" and free.check(None) is None and free.check(object()) is None
    names = FieldSpec(name="X", group="ops", doc="d", unit="name", type="str", default="a", choices=("a", "b"),
                      ignore_case=True)
    assert names.range_text() == "one of a, b (any case)" and names.check("B") is None and names.check("c")


def test_ranges_are_machine_readable():
    specs = Config.field_specs()
    lr = specs["LR"]
    assert isinstance(lr, FieldSpec)
    assert (lr.minimum, lr.min_inclusive, lr.maximum, lr.max_inclusive, lr.log) == (0.0, False, 1.0, True, True)
    assert lr.range_text() == "(0, 1]"
    assert specs["BATCH_SIZE"].range_text() == "[16, 2048]" and specs["BATCH_SIZE"].step == 1
    assert specs["HORIZON_STEPS"].range_text() == "each >= 1"
    assert specs["DIRECTION_LOSS"].choices == ("bce", "focal_dice")
    assert specs["CALIB_DAMPING_DIR"].nullable and not specs["CALIB_DAMPING"].nullable
    assert specs["MACD_SETTINGS"].default == Config().MACD_SETTINGS


def test_the_old_field_helper_signature_still_works():
    f = _f(3, "ops", "a doc")
    assert f.default == 3 and f.metadata["group"] == "ops" and f.metadata["doc"] == "a doc"
    assert f.metadata["tunable"] is False and f.metadata["deprecated"] is False and f.metadata["unit"] is None
    assert _f([1, 2], "ops").default_factory() == [1, 2]
    with pytest.raises(ValueError):
        _f(1.0, "ops", "a doc", ge=0.0, gt=0.0)


@pytest.mark.parametrize("overrides, name", [
    ({"WINDOW_STEP": 0}, "WINDOW_STEP"),                    # int
    ({"SEED": -1}, "SEED"),                                  # int, lower bound
    ({"ADAM_BETA1": 1.0}, "ADAM_BETA1"),                     # float, exclusive upper bound
    ({"VACUUM_E_MAX": 0.0}, "VACUUM_E_MAX"),                 # float, exclusive lower bound
    ({"CALIB_LAMBDA_MIN": -0.5}, "CALIB_LAMBDA_MIN"),        # float, inclusive lower bound
    ({"RHO_MAX": 1.5}, "RHO_MAX"),                           # float, inclusive upper bound
    ({"LR": math.nan}, "LR"),                                # NaN is never in range
    ({"CALIB_DAMPING_DIR": 1.5}, "CALIB_DAMPING_DIR"),       # Optional[float]
    ({"MA_SPANS": [5, 0, 30]}, "MA_SPANS"),                  # List[int]: each entry
    ({"MACD_SETTINGS": [{"fast": 12, "slow": 26, "signal": 0}]}, "MACD_SETTINGS"),  # list of dicts
    ({"WINDOW_NORMALIZER": "zscore"}, "WINDOW_NORMALIZER"),  # invalid choice
    ({"SGD_MOMENTUM": None}, "SGD_MOMENTUM"),                # None where not nullable
])
def test_validate_refuses_values_outside_the_declared_range_or_choices(overrides, name):
    with pytest.raises(InvalidConfigurationError, match=name):
        Config().override(**overrides)
    with pytest.raises(InvalidConfigurationError, match=name):
        Config(**overrides)


def test_validate_refuses_an_out_of_range_value_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("SGD_MOMENTUM: 1.5\n", encoding="utf-8")
    with pytest.raises(InvalidConfigurationError, match="SGD_MOMENTUM"):
        Config.from_yaml(path)


def test_boundary_values_stay_valid():
    cfg = Config().override(MAX_SEQUENCE_COUNT=0, GRAD_CLIP_NORM=0.0, CALIB_DAMPING=1.0, CALIB_DAMPING_POINT=None,
                            EARLY=0, PATIENCE=0, ADAM_BETA1=0.0, LAMBDA_HD=0.0, MOMENTUM_CLIP_MAX=None,
                            EWMA_IMPL="SCAN", PLUGINS_DIR=None, LOSS_WEIGHT_SCHEDULE=None)
    assert cfg.EWMA_IMPL == "SCAN" and cfg.MAX_SEQUENCE_COUNT == 0


def test_window_normalizer_choices_match_the_data_package():
    from neural_trade.data.scaling import WINDOW_NORMALIZERS

    assert Config.field_specs()["WINDOW_NORMALIZER"].choices == tuple(WINDOW_NORMALIZERS)


def _config_files():
    return sorted((REPO / "configs").glob("*.yaml"))


def test_every_committed_config_file_validates():
    """A file whose keys are Config fields loads with Config.from_yaml; the physics ablation spec's
    Config overrides validate as overrides. default.yaml loads to exactly the defaults."""
    names = set(Config.field_names())
    checked = []
    for path in _config_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if any(k in names for k in data):
            Config.from_yaml(path)
            checked.append(path.name)
        elif path.name == "ablation_physics.yaml":
            for scale in data["scales"].values():
                Config().override(**data.get("base_overrides", {}), **scale, **data["terms"])
            for fold in data["periods"].values():
                Config().override(FOLD_INDEX=fold)
            checked.append(path.name)
    assert {"default.yaml", "ci.yaml", "ablation_physics.yaml"} <= set(checked)
    assert Config.from_yaml(REPO / "configs" / "default.yaml").to_dict() == Config().to_dict()


def _generator():
    spec = importlib.util.spec_from_file_location("gen_config_reference", REPO / "scripts" / "gen_config_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_config_reference_is_generated_from_the_metadata():
    gen = _generator()
    committed = (REPO / "docs" / "guide" / "config-reference.md").read_text(encoding="utf-8")
    assert committed == gen.render(), "stale: run python scripts/gen_config_reference.py"
    assert gen.main(["--check"]) == 0
    for name in Config.field_names():
        assert f"| `{name}` |" in committed
