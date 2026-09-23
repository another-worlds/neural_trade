"""Typed configuration: one flat dataclass (remediation plan section B2).

Flat on purpose: the code base reads ``cfg.LAMBDA_HD``-style names in hundreds of
places; grouping lives in field metadata (``group``), which drives the grouped YAML and
the docs, not in nested sub-objects.

    cfg = Config()                                 # defaults, validated
    cfg = Config(EPOCHS=5, LR=3e-4)                # keyword construction
    cfg = cfg.override(LAMBDA_HD=0.0)              # validated update, unknown keys rejected
    cfg = Config.from_yaml("configs/default.yaml") # flat keys == field names
    cfg.to_yaml("run/config.yaml")

Every setting of the pre-package ``model.Config`` keeps its name and default value.
"""
from __future__ import annotations

import difflib
import logging
import typing
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from .exceptions import InvalidConfigurationError

_log = logging.getLogger(__name__)

GROUPS = (
    "data", "horizons", "training", "calibration", "loss_weights", "physics",
    "indicators", "architecture", "stability", "direction", "variance", "paths",
    "registries", "optimizers", "ops",
)


def _f(default, group: str, doc: str = "", **kw):
    """A dataclass field with its group and one-line documentation in metadata."""
    if isinstance(default, (list, dict)):
        value = default
        return field(default_factory=lambda: _copy(value), metadata={"group": group, "doc": doc}, **kw)
    return field(default=default, metadata={"group": group, "doc": doc}, **kw)


def _copy(v):
    if isinstance(v, list):
        return [_copy(x) for x in v]
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    return v


_DEFAULT_MACD = [
    {"fast": 12, "slow": 26, "signal": 9},
    {"fast": 5, "slow": 35, "signal": 5},
    {"fast": 8, "slow": 17, "signal": 9},
]


@dataclass
class Config:
    HOUR: ClassVar[int] = 60
    DAY: ClassVar[int] = 60 * 24

    # ------------------------------------------------------------------ data
    CSV_PATH: str = _f("binance_btcusdt_1min_ccxt.csv", "data", "OHLCV CSV (timestamp/datetime, open..volume)")
    LOOKBACK: int = _f(60, "data", "input window length in bars")
    WINDOW_STEP: int = _f(1, "data", "stride between consecutive training windows")
    RESAMPLE_MINUTES: int = _f(1, "data", "aggregate to coarser bars (1 = native minute bars)")
    MAX_SEQUENCE_COUNT: int = _f(1440 * 37, "data", "keep only the most recent N sequences")
    VAL_FRACTION: float = _f(0.066, "data", "validation block size (fraction of sequences)")
    CAL_FRACTION: float = _f(0.066, "data", "calibration block size (fraction of sequences)")
    N_FOLDS: int = _f(5, "data", "TimeSeriesSplit folds; the last fold's test block is reported")
    FOLD_INDEX: int = _f(-1, "data", "which purged fold to train/evaluate on (-1 = the latest; walk-forward varies it)")

    # ------------------------------------------------------------------ horizons
    EXTENDED_TREND_PERIODS: List[int] = _f([10, 15, 20], "horizons",
                                           "lags (bars) of the past-delta momentum features, one per horizon")
    HORIZON_STEPS: List[int] = _f([10, 15, 20], "horizons", "forecast horizons in bars (h0, h1, h2)")

    # ------------------------------------------------------------------ training
    BATCH_SIZE: int = _f(64, "training")
    EPOCHS: int = _f(20, "training")
    LR: float = _f(1e-3, "training", "main optimizer learning rate")
    PATIENCE: int = _f(3, "training", "ReduceLROnPlateau patience (was EPOCHS: disabled)")
    EARLY: int = _f(6, "training", "EarlyStopping patience on val_loss (was EPOCHS: disabled)")

    # ------------------------------------------------------------------ calibration (pre-training lambda pass)
    DAMPING: float = _f(0.5, "calibration", "legacy alias; use CALIB_DAMPING")
    CALIB_WARMUP_FRACTION: float = _f(0.05, "calibration", "warm-up forward passes (fraction of an epoch)")
    CALIB_SAMPLE_FRACTION: float = _f(0.1, "calibration", "loss-magnitude sampling (fraction of an epoch)")
    CALIB_LAMBDA_MIN: float = _f(0.1, "calibration")
    CALIB_LAMBDA_MAX: float = _f(20.0, "calibration")
    CALIB_DAMPING: float = _f(1.0, "calibration", "0 = no change, 1 = full magnitude equalisation")
    CALIB_DAMPING_POINT: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_TREND: Optional[float] = _f(0.0, "calibration", "0: the trend prior is a regulariser, not rescaled")
    CALIB_DAMPING_DIR: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_VAR: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_CRPS: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_ECE: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_VOL: Optional[float] = _f(None, "calibration")
    CALIB_DAMPING_PHYSICS: Optional[float] = _f(0.0, "calibration",
                                                "0: bounded physics regularisers are never rescaled")
    CALIB_OUTER: bool = _f(False, "calibration", "also calibrate the outer group multipliers")
    CONFORMAL_SCALE: str = _f("realized_vol", "calibration",
                              "conformal interval scale: 'realized_vol' (window), 'sigma' (variance head) or 'none'")

    # ------------------------------------------------------------------ loss weights
    LAMBDA_LOCAL_TREND: float = _f(1.0, "loss_weights", "retired term (always 0 in the objective)")
    LAMBDA_GLOBAL_TREND: float = _f(1.0, "loss_weights", "retired term (always 0 in the objective)")
    LAMBDA_EXTENDED_TREND: float = _f(0.1, "loss_weights", "momentum prior; a regulariser, kept small")
    LAMBDA_QUANTILE: float = _f(1.0, "loss_weights", "unused (no quantile term in the objective)")
    LAMBDA_SHORT: float = _f(1.0, "loss_weights", "point loss weight, h0")
    LAMBDA_POINT: float = _f(1.0, "loss_weights", "point loss weight, h1")
    LAMBDA_LONG: float = _f(1.0, "loss_weights", "point loss weight, h2")
    LAMBDA_DIR: float = _f(1.0, "loss_weights", "direction loss (focal + dice)")
    LAMBDA_INTER: float = _f(1.0, "loss_weights", "weight of model.losses (layer regularisers)")
    LAMBDA_VOL: float = _f(1.0, "loss_weights", "prediction-spread vs target-spread penalty")
    LAMBDA_VAR: float = _f(1.0, "loss_weights", "Gaussian NLL of the variance heads")
    LAMBDA_TREND_OUTER: float = _f(1.0, "loss_weights")
    LAMBDA_DIR_OUTER: float = _f(1.0, "loss_weights")
    LAMBDA_DIR_ALIGN_OUTER: float = _f(0.0, "loss_weights", "direction head vs Gaussian readout alignment")
    LAMBDA_COHERENCE: float = _f(1.0, "loss_weights")
    LAMBDA_NLL_OUTER: float = _f(1.0, "loss_weights")
    LAMBDA_CRPS: float = _f(1.0, "loss_weights")
    LAMBDA_SOFT_ECE: float = _f(1.0, "loss_weights")
    LAMBDA_DIR_ALIGN: float = _f(0.7, "loss_weights", "inner weight of the alignment term")

    # ------------------------------------------------------------------ physics-inspired terms (T-perp / QBOX)
    T_PERP_DIM: int = _f(16, "physics", "width of the perpendicular projection")
    LAMBDA_T_PERP: float = _f(0.1, "physics", "batch variance tracks batch residual energy")
    LAMBDA_CASIMIR: float = _f(0.1, "physics", "disagreeing horizons need variance")
    LAMBDA_VAC: float = _f(0.0, "physics", "vacuum bandwidth threshold (0 = off)")
    LAMBDA_HD: float = _f(0.1, "physics", "variance ordered like realised volatility")
    LAMBDA_IFE: float = _f(0.1, "physics", "cross-horizon correlation hinge")
    RHO_MAX: float = _f(0.95, "physics", "max allowed cross-horizon correlation")
    VACUUM_E_MAX: float = _f(1.0, "physics", "per-dimension energy ceiling of the vacuum layer")
    LAMBDA_VAC_OVERFLOW: float = _f(0.1, "physics", "overflow tracks residual magnitude")

    # ------------------------------------------------------------------ learnable indicators
    MA_SPANS: List[int] = _f([5, 10, 30], "indicators")
    MACD_SETTINGS: List[Dict[str, int]] = _f(_DEFAULT_MACD, "indicators")
    RSI_PERIODS: List[int] = _f([9, 14, 21], "indicators")
    BB_PERIODS: List[int] = _f([10, 20, 25], "indicators")
    INDICATOR_L2: float = _f(0.0, "indicators", "L2 on the indicator logits")
    INDICATOR_LR_MULT: float = _f(5.0, "indicators", "indicator optimizer LR = LR * this")
    INDICATOR_GRAD_MULT: float = _f(5.0, "indicators", "straight-through gradient scale")
    MOMENTUM_CLIP_MIN: float = _f(2.0, "indicators", "period floor (1.0 saturated the logit)")
    MOMENTUM_CLIP_MAX: Optional[float] = _f(None, "indicators", "period ceiling; None -> LOOKBACK")
    EWMA_IMPL: str = _f("matrix", "indicators", "'matrix' (batched einsum) or 'scan' (reference)")

    # ------------------------------------------------------------------ architecture / activations
    REG_MOMENTUM_L2: float = _f(0.0, "architecture", "L2 on the dense towers")
    TANH_SCALE: float = _f(1.0, "architecture", "unused")
    SIGMOID_SCALE: float = _f(1.0, "architecture", "unused")
    HUBER_DELTA: float = _f(1.0, "architecture", "delta of CustomTrainModel.huber (not the point loss)")
    USE_HUBER: bool = _f(True, "architecture", "legacy flag; the point loss is log(cosh)")

    # ------------------------------------------------------------------ stability
    GRAD_CLIP_NORM: float = _f(20.0, "stability", "global-norm clip, per optimizer group")

    # ------------------------------------------------------------------ direction
    FOCAL_ALPHA: float = _f(0.5, "direction", "weight of the DOWN class")
    FOCAL_GAMMA: float = _f(2.0, "direction")
    DIRECTION_LOSS: str = _f("bce", "direction",
                             "'bce' (proper scoring rule) or 'focal_dice' (legacy: its optimum is a constant extreme)")
    DIR_DEADBAND_BPS: float = _f(5.0, "direction", "|return| below this is neutral and masked")

    # ------------------------------------------------------------------ variance
    VAR_FLOOR: float = _f(1e-4, "variance", "variance floor (scaled units^2)")
    VAR_CAP: float = _f(1e3, "variance", "variance cap for metrics (not applied in the loss)")
    DELTA_MAPE_MIN_ABS: float = _f(1.0, "variance", "min |y| ($) for the delta safe-MAPE")

    # ------------------------------------------------------------------ paths
    MODEL_PATH: str = _f("nn_learnable_indicators_v3.weights.h5", "paths")
    SCALER_PATH: str = _f("scaler_v3.joblib", "paths")
    ARTIFACTS_DIR: str = _f("artifacts", "paths", "where Trainer writes the serving bundle")
    PLUGINS_DIR: Optional[str] = _f(None, "paths", "directory of plugin modules to load at startup")

    # ------------------------------------------------------------------ registries (component selection)
    MODEL_NAME: str = _f("gru_attention", "registries", "Models registry key")
    LOSS_NAME: str = _f("custom_loss", "registries", "Losses objective key")
    OPTIMIZER_NAME: str = _f("adam", "registries", "Optimizers key, main network")
    INDICATOR_OPTIMIZER_NAME: str = _f("adam", "registries", "Optimizers key, indicator logits")
    DATA_LOADER: str = _f("csv", "registries", "DataLoaders key")
    PREPROCESSORS: List[str] = _f(["standardize_ohlcv", "sort_dedupe", "resample_bars", "drop_missing_close"],
                                  "registries", "Preprocessors keys, applied in order")
    WINDOW_NORMALIZER: str = _f("window_relative", "registries", "input normalisation of the windows")
    LAYERS: Dict[str, str] = _f({"indicators": "learnable_indicators", "positional_encoding": "positional_encoding",
                                 "vacuum_noise": "vacuum_saturation_noise", "energy_gate": "energy_gate"},
                                "registries", "Layers keys by role in the architecture")
    METRICS: List[str] = _f(["mse", "rmse", "mae", "explained_variance", "corr", "r2", "safe_mape", "smape",
                             "wape", "direction_accuracy", "direction_f1", "mcc", "brier", "ece_pos",
                             "pit_ks", "coverage"], "registries", "Metrics keys, numpy tier (evaluation)")
    STEP_METRICS: List[str] = _f(["dir_acc", "dir_sensitivity", "dir_specificity", "dir_bal_acc", "dir_f1",
                                  "dir_mcc", "dir_brier", "dir_ece", "pred_up_rate", "true_up_rate",
                                  "mean_dir_prob"], "registries", "Metrics keys, TF tier (train/test step)")
    CALLBACKS: List[str] = _f(["csv_logger", "early_stopping", "model_checkpoint", "tqdm_progress",
                               "params_logger", "reduce_lr_on_plateau"], "registries",
                              "Callbacks keys, in order")
    VISUALIZATION: str = _f("plotly_interactive", "registries", "Visualizations key")

    # ------------------------------------------------------------------ optimizers
    ADAM_BETA1: float = _f(0.9, "optimizers")
    ADAM_BETA2: float = _f(0.999, "optimizers")
    ADAM_EPSILON: float = _f(1e-7, "optimizers", "Keras default")
    WEIGHT_DECAY: float = _f(0.004, "optimizers", "adamw only")
    SGD_MOMENTUM: float = _f(0.9, "optimizers")
    SGD_NESTEROV: bool = _f(True, "optimizers")

    # ------------------------------------------------------------------ ops
    LOSS_WEIGHT_SCHEDULE: Optional[Dict[str, Any]] = _f(None, "ops",
                                                   "{'lambda_hd': {epoch: value, ...}, ...} for the lambda_schedule callback")
    ABLATE_LAMBDAS: List[str] = _f([], "ops", "LAMBDA_* names forced to 0 after calibration (ablation)")
    SEED: int = _f(42, "ops", "global seed set by train_and_evaluate")

    # ================================================================== behaviour
    def __post_init__(self):
        for name in ("EXTENDED_TREND_PERIODS", "HORIZON_STEPS"):
            if isinstance(getattr(self, name), tuple):
                setattr(self, name, list(getattr(self, name)))
        if self.MOMENTUM_CLIP_MAX is None:
            self.MOMENTUM_CLIP_MAX = self.LOOKBACK
        self.validate()

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise :class:`InvalidConfigurationError` (a ``ValueError``) on invalid settings."""
        def bad(msg):
            raise InvalidConfigurationError(msg)

        if self.LOOKBACK <= 0:
            bad("LOOKBACK must be positive")
        if self.LOOKBACK > 1440:
            bad("LOOKBACK should not exceed 1 day for minute data")
        if self.BATCH_SIZE < 16 or self.BATCH_SIZE > 2048:
            bad("BATCH_SIZE should be between 16 and 2048")
        if self.LR <= 0 or self.LR > 1.0:
            bad("LR should be >0 and <=1.0")
        if self.EPOCHS < 1:
            bad("EPOCHS must be >=1")
        if not self.HORIZON_STEPS or any(h <= 0 for h in self.HORIZON_STEPS):
            bad("HORIZON_STEPS must be non-empty list of positive ints")
        if not self.EXTENDED_TREND_PERIODS or any(t <= 0 for t in self.EXTENDED_TREND_PERIODS):
            bad("EXTENDED_TREND_PERIODS must be non-empty list of positive ints")
        if len(self.HORIZON_STEPS) != len(self.EXTENDED_TREND_PERIODS):
            bad(f"HORIZON_STEPS ({self.HORIZON_STEPS}) length must match "
                f"EXTENDED_TREND_PERIODS ({self.EXTENDED_TREND_PERIODS}) "
                "for semantic consistency (DataProcessor CRITICAL + extended_trend_loss).")
        if len(self.HORIZON_STEPS) != 3:
            bad("the architecture has exactly three horizon towers: HORIZON_STEPS needs 3 entries")
        if self.HORIZON_STEPS != sorted(self.HORIZON_STEPS):
            bad("HORIZON_STEPS should be ascending")
        if self.EXTENDED_TREND_PERIODS != sorted(self.EXTENDED_TREND_PERIODS):
            bad("EXTENDED_TREND_PERIODS should be ascending")
        if self.VAR_FLOOR != 1e-4:
            warnings.warn(f"Config.VAR_FLOOR={self.VAR_FLOOR} (expected 1e-4). Using provided value.",
                          DeprecationWarning, stacklevel=2)
        if self.VAR_CAP <= self.VAR_FLOOR:
            bad("VAR_CAP must be > VAR_FLOOR")
        if self.LAMBDA_VAC > 0:
            _log.warning("Config.LAMBDA_VAC > 0: vacuum_bandwidth_loss is active (default is off).")
        if not 0.0 < self.VAL_FRACTION < 0.5 or not 0.0 < self.CAL_FRACTION < 0.5:
            bad("VAL_FRACTION and CAL_FRACTION must be in (0, 0.5)")
        if self.N_FOLDS < 2:
            bad("N_FOLDS must be >= 2")
        if not -self.N_FOLDS <= self.FOLD_INDEX < self.N_FOLDS:
            bad(f"FOLD_INDEX must be in [-{self.N_FOLDS}, {self.N_FOLDS - 1}]")
        if self.DIR_DEADBAND_BPS < 0:
            bad("DIR_DEADBAND_BPS must be >= 0")
        if self.DIRECTION_LOSS not in ("bce", "focal_dice"):
            bad(f"DIRECTION_LOSS must be 'bce' or 'focal_dice', got {self.DIRECTION_LOSS!r}")
        if self.CONFORMAL_SCALE not in ("none", "sigma", "realized_vol"):
            bad(f"CONFORMAL_SCALE must be 'none', 'sigma' or 'realized_vol', got {self.CONFORMAL_SCALE!r}")
        if str(self.EWMA_IMPL).lower() not in ("matrix", "scan"):
            bad(f"EWMA_IMPL must be 'matrix' or 'scan', got {self.EWMA_IMPL!r}")
        if not (0 < self.MOMENTUM_CLIP_MIN < (self.MOMENTUM_CLIP_MAX or self.LOOKBACK)):
            bad("need 0 < MOMENTUM_CLIP_MIN < MOMENTUM_CLIP_MAX")
        negative = [k for k, v in self.lambda_weights().items() if v < 0]
        if negative:
            bad(f"loss weights must be >= 0: {negative}")
        unknown_ablate = [n for n in self.ABLATE_LAMBDAS if n not in self.lambda_weights()]
        if unknown_ablate:
            bad(f"ABLATE_LAMBDAS names unknown LAMBDA_* fields: {unknown_ablate}")

    # --------------------------------------------------------------- derived
    @property
    def momentum_clip_max(self) -> float:
        return float(self.MOMENTUM_CLIP_MAX if self.MOMENTUM_CLIP_MAX is not None else self.LOOKBACK)

    @property
    def effective_var_floor(self) -> float:
        return self.VAR_FLOOR

    def lambda_weights(self) -> Dict[str, float]:
        """Every ``LAMBDA_*`` loss weight (the thresholds LAMBDA_VAC / RHO_MAX included as-is)."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)
                if f.name.startswith("LAMBDA_")}

    # --------------------------------------------------------------- updates
    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def override(self, **changes) -> "Config":
        """Apply ``changes`` in place (unknown names are an error), re-validate, return self."""
        names = set(self.field_names())
        unknown = [k for k in changes if k not in names]
        if unknown:
            hints = {k: difflib.get_close_matches(k, names, n=3) for k in unknown}
            raise InvalidConfigurationError(
                "unknown Config field(s): " + "; ".join(
                    f"{k}" + (f" (did you mean {', '.join(v)}?)" if v else "") for k, v in hints.items()))
        hints = typing.get_type_hints(type(self))
        for k, v in changes.items():
            setattr(self, k, _coerce(v, hints[k], k))
        self.validate()
        return self

    def copy(self, **changes) -> "Config":
        """A deep copy, optionally overridden."""
        return type(self).from_dict(self.to_dict()).override(**changes)

    # --------------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _copy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = True) -> "Config":
        names = set(cls.field_names())
        unknown = [k for k in data if k not in names]
        if unknown and strict:
            hints = {k: difflib.get_close_matches(k, names, n=3) for k in unknown}
            raise InvalidConfigurationError(f"unknown Config field(s) {unknown}; suggestions: {hints}")
        hints = typing.get_type_hints(cls)
        kwargs = {k: _coerce(v, hints[k], k) for k, v in data.items() if k in names}
        return cls(**kwargs)

    def to_yaml(self, path=None) -> str:
        """Flat YAML (keys = field names), ordered and commented by group."""
        import yaml

        by_group: Dict[str, list] = {g: [] for g in GROUPS}
        for f in fields(self):
            by_group.setdefault(f.metadata.get("group", "ops"), []).append(f)
        out = ["# neural_trade Config - flat keys; see neural_trade/core/config.py for every field"]
        for group, fs in by_group.items():
            if not fs:
                continue
            out.append(f"\n# ---- {group}")
            for f in fs:
                dumped = yaml.safe_dump({f.name: getattr(self, f.name)}, default_flow_style=True,
                                        sort_keys=False, width=10_000).strip()
                if dumped.startswith("{") and dumped.endswith("}"):
                    dumped = dumped[1:-1]
                doc = f.metadata.get("doc")
                out.append(dumped + (f"  # {doc}" if doc else ""))
        text = "\n".join(out) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path) -> "Config":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise InvalidConfigurationError(f"{path}: expected a mapping of Config fields")
        return cls.from_dict(data)


def _coerce(value, hint, name):
    """Coerce YAML/CLI values to the annotated type (PyYAML reads '1e-3' as a string)."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union and type(None) in args:  # Optional[X]
        if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "")):
            return None
        hint = next(a for a in args if a is not type(None))
        origin, args = typing.get_origin(hint), typing.get_args(hint)
    try:
        if hint is bool:
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1", "yes", "on"):
                    return True
                if v in ("false", "0", "no", "off"):
                    return False
                raise ValueError(value)
            return bool(value)
        if hint is int:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(value)
            return int(float(value)) if isinstance(value, str) else int(value)
        if hint is float:
            return float(value)
        if hint is str:
            return str(value)
        if origin in (list, List):
            if isinstance(value, str):
                import yaml
                value = yaml.safe_load(value)
            if isinstance(value, tuple):
                value = list(value)
            if not isinstance(value, list):
                raise ValueError(value)
            inner = args[0] if args else Any
            return [_coerce(v, inner, name) if inner in (int, float, str, bool) else v for v in value]
        if origin in (dict, Dict):
            if isinstance(value, str):
                import yaml
                value = yaml.safe_load(value)
            if not isinstance(value, dict):
                raise ValueError(value)
            return dict(value)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"Config.{name}: cannot interpret {value!r} as {hint}") from exc
    return value


__all__ = ["Config", "GROUPS"]
