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

Every field also declares, in its metadata, a unit (``UNITS``), a range (``ge`` / ``gt`` /
``le`` / ``lt``) or a set of ``choices`` where one applies, whether a sweep may tune it and
whether it is deprecated. ``Config.field_specs()`` returns them as :class:`FieldSpec` objects
(for the control panel and the sweep search spaces), ``Config.validate`` enforces the ranges and
choices, and ``docs/guide/config-reference.md`` is generated from them by
``scripts/gen_config_reference.py``.
"""
from __future__ import annotations

import difflib
import logging
import math
import numbers
import typing
import warnings
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from .exceptions import InvalidConfigurationError

_log = logging.getLogger(__name__)

GROUPS = (
    "data", "horizons", "training", "calibration", "loss_weights", "physics",
    "indicators", "architecture", "stability", "direction", "variance", "paths",
    "registries", "optimizers", "ops",
)

# The unit vocabulary of the field metadata. Time is either "bars" (of the resampled series) or
# wall-clock "minutes", so fields configured in wall-clock time (NT-041) can say which they are.
UNITS: Dict[str, str] = {
    "bars": "bars of the (resampled) price series",
    "minutes": "wall-clock minutes",
    "sequences": "input windows (one sequence per anchor bar)",
    "epochs": "training epochs",
    "steps": "optimizer steps (training batches)",
    "count": "a plain count",
    "fraction": "a share between 0 and 1",
    "bps": "basis points (1/10,000 of the price)",
    "weight": "a dimensionless multiplier of a loss term",
    "dimensionless": "a dimensionless number",
    "scaled": "target-scaler units (the standardised price delta)",
    "scaled^2": "squared target-scaler units (a variance)",
    "quote": "quote currency of the instrument (USDT on the reference setup)",
    "index": "a position in a list (negative counts from the end)",
    "seed": "a random seed",
    "flag": "true or false",
    "name": "one of the listed names",
    "key": "a registry key or a Config field name, checked when the registries load",
    "path": "a file or directory path",
    "mapping": "a nested mapping (see the doc)",
    "timestamp": "an ISO-8601 timestamp (naive timestamps are read as UTC)",
    "days": "a number of calendar days",
}


@dataclass(frozen=True)
class FieldSpec:
    """The machine-readable metadata of one Config field.

    ``minimum`` / ``maximum`` bound a number, or every number inside a list or dict value (for
    example each entry of HORIZON_STEPS); ``None`` means unbounded. ``step`` and ``log`` are hints
    for widgets and search spaces and are not enforced. ``choices`` lists the valid names of a
    string field (compared case-insensitively when ``ignore_case``).
    """
    name: str
    group: str
    doc: str
    unit: Optional[str]
    type: str
    default: Any
    nullable: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    step: Optional[float] = None
    log: bool = False
    choices: Optional[Tuple[str, ...]] = None
    ignore_case: bool = False
    tunable: bool = False
    deprecated: bool = False

    @property
    def has_range(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    @property
    def per_item(self) -> bool:
        """True when the range applies to each number inside a list or dict value."""
        return self.type.startswith(("List", "Dict", "list", "dict"))

    def range_text(self) -> str:
        """The range or the choices as text: ``[1, 1440]``, ``(0, 0.5)``, ``>= 0``, ``one of a, b``."""
        if self.choices is not None:
            return "one of " + ", ".join(self.choices) + (" (any case)" if self.ignore_case else "")
        if not self.has_range:
            return ""
        lo, hi = self.minimum, self.maximum
        if lo is not None and hi is not None:
            text = (f"{'[' if self.min_inclusive else '('}{_num(lo)}, "
                    f"{_num(hi)}{']' if self.max_inclusive else ')'}")
        elif lo is not None:
            text = f"{'>=' if self.min_inclusive else '>'} {_num(lo)}"
        else:
            text = f"{'<=' if self.max_inclusive else '<'} {_num(hi)}"
        return ("each " if self.per_item else "") + text

    def check(self, value) -> Optional[str]:
        """``None`` when ``value`` satisfies the declared range or choices, otherwise the reason
        (naming the field). A field without a range or choices accepts any value here."""
        if self.choices is None and not self.has_range:
            return None
        if value is None:
            return None if self.nullable else f"{self.name}=None is not allowed ({self.range_text()})"
        if self.choices is not None:
            valid = [c.lower() for c in self.choices] if self.ignore_case else list(self.choices)
            if (str(value).lower() if self.ignore_case else value) not in valid:
                return f"{self.name}={value!r} is not one of {', '.join(self.choices)}"
            return None
        for x in _numbers_in(value):
            if x is None:
                return f"{self.name}={value!r} is not a number"
            if not self._in_range(x):
                return f"{self.name}={value!r} is outside its range {self.range_text()}"
        return None

    def _in_range(self, x) -> bool:
        if math.isnan(float(x)):
            return False
        if self.minimum is not None and (x < self.minimum if self.min_inclusive else x <= self.minimum):
            return False
        if self.maximum is not None and (x > self.maximum if self.max_inclusive else x >= self.maximum):
            return False
        return True


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _numbers_in(value):
    """Every number inside ``value`` (a number, or a list / dict of them); ``None`` for a non-number."""
    if isinstance(value, numbers.Real):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _numbers_in(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numbers_in(v)
    else:
        yield None


def _f(default, group: str, doc: str = "", *, unit: Optional[str] = None, ge=None, gt=None, le=None, lt=None,
       step=None, log: bool = False, choices=None, ignore_case: bool = False, tunable: bool = False,
       deprecated: bool = False, **kw):
    """A dataclass field with its group, one-line doc and :class:`FieldSpec` metadata.

    ``ge`` / ``gt`` (lower bound) and ``le`` / ``lt`` (upper bound) declare the valid range,
    ``choices`` the valid names; ``tunable`` marks what a sweep may search and ``deprecated`` a
    legacy field. The metadata arguments are keyword-only, so ``_f(default, group, doc)`` still works.
    """
    if (ge is not None and gt is not None) or (le is not None and lt is not None):
        raise ValueError("give at most one of ge / gt and at most one of le / lt")
    meta = {"group": group, "doc": doc, "unit": unit,
            "minimum": gt if gt is not None else ge, "min_inclusive": gt is None,
            "maximum": lt if lt is not None else le, "max_inclusive": lt is None,
            "step": step, "log": bool(log), "choices": tuple(choices) if choices is not None else None,
            "ignore_case": bool(ignore_case), "tunable": bool(tunable), "deprecated": bool(deprecated)}
    if isinstance(default, (list, dict)):
        value = default
        return field(default_factory=lambda: _copy(value), metadata=meta, **kw)
    return field(default=default, metadata=meta, **kw)


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

# The valid WINDOW_NORMALIZER names; equal to neural_trade.data.scaling.WINDOW_NORMALIZERS (tested),
# repeated here so that importing the Config does not import scikit-learn.
_WINDOW_NORMALIZERS = ("window_relative", "per_lag_standard")


@dataclass
class Config:
    HOUR: ClassVar[int] = 60
    DAY: ClassVar[int] = 60 * 24

    # Tunable: a model or training hyperparameter a sweep may search (NT-030). Not tunable: data,
    # blocks, targets and folds (trials must be scored on the same blocks), the training budget,
    # component choices (registry keys, loss and calibration switches settled in DECISIONS), paths,
    # seeds, diagnostics, numerical guards, deprecated fields, and knobs without an effect under the
    # default components (FOCAL_* with bce, SGD_* and WEIGHT_DECAY with adam).

    # ------------------------------------------------------------------ data
    CSV_PATH: str = _f("binance_btcusdt_1min_ccxt.csv", "data", "OHLCV CSV (timestamp/datetime, open..volume)",
                       unit="path")
    LOOKBACK: int = _f(60, "data", "input window length in bars", unit="bars", ge=1, le=1440, step=1)
    INPUT_SERIES: List[str] = _f(["open", "high", "low", "close", "volume"], "data",
                                 "which bar series each input window carries, in this fixed order (a "
                                 "subsequence of open, high, low, close, volume that includes 'close'; "
                                 "['close'] is the pre-NT-047 close-only input, and the model input is then "
                                 "[B, LOOKBACK] instead of [B, LOOKBACK, len(INPUT_SERIES)]); OHLC channels "
                                 "are window-relative, volume has its own train-fit scale "
                                 "(neural_trade.data.scaling)", unit="name")
    WINDOW_STEP: int = _f(1, "data", "stride between consecutive training windows", unit="bars", ge=1, step=1)
    RESAMPLE_MINUTES: int = _f(1, "data", "aggregate to coarser bars (1 = native minute bars); not tunable until "
                               "NT-040, because the annualisation ignores the bar size until then",
                               unit="minutes", ge=1, step=1)
    MAX_SEQUENCE_COUNT: int = _f(1440 * 37, "data", "keep only the most recent N sequences (0 = keep all)",
                                 unit="sequences", ge=0, step=1)
    VAL_FRACTION: float = _f(0.066, "data", "validation block size (fraction of sequences)",
                             unit="fraction", gt=0.0, lt=0.5)
    CAL_FRACTION: float = _f(0.066, "data", "calibration block size (fraction of sequences)",
                             unit="fraction", gt=0.0, lt=0.5)
    N_FOLDS: int = _f(5, "data", "TimeSeriesSplit folds; the last fold's test block is reported", unit="count",
                      ge=2, step=1)
    FOLD_INDEX: int = _f(-1, "data", "which purged fold to train/evaluate on (-1 = the latest; walk-forward varies it)",
                         unit="index")

    # NT-088 (screen mode): lets a trial's data end anywhere in history instead of only at the file's
    # newest bar, so a screen can slice quiet/volatile regimes for its mass 6-hour training blocks.
    # The slice is taken from load_and_prepare_data, before MAX_SEQUENCE_COUNT trims from the end of
    # that slice. None (default) is today's behaviour: no slicing, the newest bars are used.
    DATA_END: Optional[str] = _f(None, "data", "end the prepared data at this timestamp instead of the file's "
                                 "newest bar (None = today's behaviour); refused when the slice would reach into "
                                 "the protected dev/test span (DATA_END_PROTECTED_DAYS) of the full file (D-020)",
                                 unit="timestamp")
    DATA_END_PROTECTED_DAYS: float = _f(64.0, "data", "DATA_END is refused when it falls within this many days of "
                                        "the full file's last bar, so a screen trial cannot slice into the "
                                        "long file's dev/test period (D-020)", unit="days", ge=0.0)

    # ------------------------------------------------------------------ horizons
    EXTENDED_TREND_PERIODS: List[int] = _f([10, 15, 20], "horizons",
                                           "lags (bars) of the past-delta momentum features, one per horizon",
                                           unit="bars", ge=1, step=1)
    HORIZON_STEPS: List[int] = _f([10, 15, 20], "horizons", "forecast horizons in bars (h0, h1, h2)",
                                  unit="bars", ge=1, step=1)

    # ------------------------------------------------------------------ training
    BATCH_SIZE: int = _f(256, "training", "256: a step costs about the same at 64 or 256 on the GPU (launch-bound), so ~3.7x faster epochs",
                         unit="count", ge=16, le=2048, step=1, log=True, tunable=True)
    EPOCHS: int = _f(20, "training", "maximum training epochs; EarlyStopping (EARLY) may stop sooner, and the "
                     "best-validation epoch is served (D-011)", unit="epochs", ge=1, step=1)
    LR: float = _f(1e-3, "training", "main optimizer learning rate", unit="dimensionless", gt=0.0, le=1.0, log=True,
                   tunable=True)
    PATIENCE: int = _f(3, "training", "ReduceLROnPlateau patience (was EPOCHS: disabled)", unit="epochs", ge=0, step=1,
                       tunable=True)
    EARLY: int = _f(6, "training", "EarlyStopping patience on val_loss (was EPOCHS: disabled)", unit="epochs", ge=0,
                    step=1)
    SHUFFLE_BUFFER: int = _f(2048, "training", "shuffle buffer of the training dataset, in windows (0 = the whole "
                             "training block: a full reshuffle every epoch); validation is never shuffled",
                             unit="sequences", ge=0, step=1)

    # ------------------------------------------------------------------ calibration (pre-training lambda pass)
    DAMPING: float = _f(0.5, "calibration", "legacy alias; use CALIB_DAMPING", unit="dimensionless", ge=0.0, le=1.0,
                        deprecated=True)
    CALIB_WARMUP_FRACTION: float = _f(0.05, "calibration", "warm-up forward passes (fraction of an epoch)",
                                      unit="fraction", ge=0.0, le=1.0)
    CALIB_SAMPLE_FRACTION: float = _f(0.1, "calibration", "loss-magnitude sampling (fraction of an epoch)",
                                      unit="fraction", ge=0.0, le=1.0)
    CALIB_LAMBDA_MIN: float = _f(0.1, "calibration", "lower clamp of every loss weight the calibration pass rescales",
                                 unit="weight", ge=0.0)
    CALIB_LAMBDA_MAX: float = _f(20.0, "calibration", "upper clamp of every loss weight the calibration pass rescales",
                                 unit="weight", gt=0.0)
    CALIB_DAMPING: float = _f(1.0, "calibration", "0 = no change, 1 = full magnitude equalisation",
                              unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_POINT: Optional[float] = _f(None, "calibration", "damping of the point-loss weights LAMBDA_SHORT, "
                                              "LAMBDA_POINT and LAMBDA_LONG (None = CALIB_DAMPING)",
                                              unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_TREND: Optional[float] = _f(0.0, "calibration", "0: the trend prior is a regulariser, not rescaled",
                                              unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_DIR: Optional[float] = _f(None, "calibration", "damping of LAMBDA_DIR (None = CALIB_DAMPING)",
                                            unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_VAR: Optional[float] = _f(None, "calibration", "damping of LAMBDA_VAR (None = CALIB_DAMPING)",
                                            unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_CRPS: Optional[float] = _f(None, "calibration", "damping of LAMBDA_CRPS, rescaled only when it is "
                                             "> 0 (None = CALIB_DAMPING)", unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_ECE: Optional[float] = _f(None, "calibration", "damping of LAMBDA_SOFT_ECE, rescaled only when it "
                                            "is > 0 (None = CALIB_DAMPING)", unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_VOL: Optional[float] = _f(None, "calibration", "damping of LAMBDA_VOL (None = CALIB_DAMPING)",
                                            unit="dimensionless", ge=0.0, le=1.0)
    CALIB_DAMPING_PHYSICS: Optional[float] = _f(0.0, "calibration",
                                                "0: bounded physics regularisers are never rescaled",
                                                unit="dimensionless", ge=0.0, le=1.0)
    CALIB_OUTER: bool = _f(False, "calibration", "also calibrate the outer group multipliers", unit="flag")
    CALIB_MODE: str = _f("value", "calibration",
                         "'value': rescale weights so each term's median value matches the reference "
                         "(today's behaviour, lambda_calibration.py:207-210); 'gradient': rescale so each "
                         "term's gradient norm on the shared trunk (main-group variables minus the price/"
                         "direction/variance head Dense layers) matches the reference (GradNorm-style, "
                         "NT-101), clipped to [CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX]",
                         unit="name", choices=("value", "gradient"))
    DELTA_SHRINKAGE: bool = _f(True, "calibration",
                               "serve beta x price-head delta, beta = clip(E[yd]/E[d^2], 0, 1) on the calibration block",
                               unit="flag")
    CONFORMAL_SCALE: str = _f("realized_vol", "calibration",
                              "conformal interval scale: 'realized_vol' (window), 'sigma' (variance head) or 'none'",
                              unit="name", choices=("realized_vol", "sigma", "none"))

    # ------------------------------------------------------------------ loss weights
    LAMBDA_LOCAL_TREND: float = _f(1.0, "loss_weights", "retired term (always 0 in the objective)", unit="weight",
                                   ge=0.0, deprecated=True)
    LAMBDA_GLOBAL_TREND: float = _f(1.0, "loss_weights", "retired term (always 0 in the objective)", unit="weight",
                                    ge=0.0, deprecated=True)
    LAMBDA_EXTENDED_TREND: float = _f(0.1, "loss_weights", "momentum prior; a regulariser, kept small", unit="weight",
                                      ge=0.0, tunable=True)
    LAMBDA_QUANTILE: float = _f(1.0, "loss_weights", "unused (no quantile term in the objective)", unit="weight",
                                ge=0.0, deprecated=True)
    LAMBDA_SHORT: float = _f(1.0, "loss_weights", "point loss weight, h0", unit="weight", ge=0.0, tunable=True)
    LAMBDA_POINT: float = _f(1.0, "loss_weights", "point loss weight, h1", unit="weight", ge=0.0, tunable=True)
    LAMBDA_LONG: float = _f(1.0, "loss_weights", "point loss weight, h2", unit="weight", ge=0.0, tunable=True)
    LAMBDA_DIR: float = _f(1.0, "loss_weights", "direction loss (DIRECTION_LOSS), summed over the horizons",
                           unit="weight", ge=0.0, tunable=True)
    LAMBDA_INTER: float = _f(1.0, "loss_weights", "weight of model.losses (layer regularisers)", unit="weight", ge=0.0,
                             tunable=True)
    LAMBDA_VOL: float = _f(1.0, "loss_weights", "prediction-spread vs target-spread penalty", unit="weight", ge=0.0,
                           tunable=True)
    LAMBDA_VAR: float = _f(1.0, "loss_weights", "Gaussian NLL of the variance heads", unit="weight", ge=0.0,
                           tunable=True)
    LAMBDA_TREND_OUTER: float = _f(1.0, "loss_weights", "outer multiplier of the summed extended-trend terms; "
                                   "rescaled by the calibration pass only with CALIB_OUTER", unit="weight", ge=0.0,
                                   tunable=True)
    LAMBDA_DIR_OUTER: float = _f(1.0, "loss_weights", "outer multiplier of the direction loss (LAMBDA_DIR term); "
                                 "rescaled by the calibration pass only with CALIB_OUTER", unit="weight", ge=0.0,
                                 tunable=True)
    LAMBDA_DIR_ALIGN_OUTER: float = _f(0.0, "loss_weights", "direction head vs Gaussian readout alignment",
                                       unit="weight", ge=0.0, tunable=True)
    LAMBDA_COHERENCE: float = _f(1.0, "loss_weights", "weight of the cross-horizon coherence penalty (sign "
                                 "disagreement and magnitude ordering of the price heads)", unit="weight", ge=0.0,
                                 tunable=True)
    LAMBDA_NLL_OUTER: float = _f(1.0, "loss_weights", "outer multiplier of the variance-head NLL (LAMBDA_VAR term); "
                                 "rescaled by the calibration pass only with CALIB_OUTER", unit="weight", ge=0.0,
                                 tunable=True)
    LAMBDA_CRPS: float = _f(1.0, "loss_weights", "weight of the Gaussian CRPS of the price and variance heads, summed "
                            "over the horizons (0 = off)", unit="weight", ge=0.0, tunable=True)
    LAMBDA_SOFT_ECE: float = _f(1.0, "loss_weights", "weight of the differentiable ECE of the direction heads, summed "
                                "over the horizons (0 = off)", unit="weight", ge=0.0, tunable=True)
    LAMBDA_DIR_ALIGN: float = _f(0.7, "loss_weights", "inner weight of the alignment term", unit="weight", ge=0.0,
                                 tunable=True)
    LAMBDA_PNL: float = _f(0.0, "loss_weights", "weight of the mean-variance P&L utility on the direction heads' "
                           "implied positions (0 = off; NT-087)", unit="weight", ge=0.0, tunable=True)
    PNL_GAMMA: float = _f(1.0, "loss_weights", "risk-aversion coefficient of the pnl_utility objective's quadratic "
                          "penalty", unit="dimensionless", gt=0.0, tunable=True)
    PNL_COST_BPS: float = _f(0.0, "loss_weights", "round-trip trading cost assumed by the pnl_utility objective "
                             "(matches strategy/variance_strategies.py's DEFAULT_COST; 0 by default, D-044)",
                             unit="bps", ge=0.0, tunable=True)
    PNL_SIGMA_SOURCE: str = _f("realized_vol", "loss_weights", "volatility scale for the pnl_utility objective's "
                               "r~ = r_H / sigma_H: 'realized_vol' (causal, equal-weighted std of the input "
                               "window's bar-to-bar RAW-price returns, reconstructed from the normalised window "
                               "via last_close and pred_scale) or 'model' (the variance head, stop-gradient). Not "
                               "an EWMA despite the name once used here (NT-087 repair round 1): every bar in the "
                               "window is weighted equally.", unit="name", choices=("realized_vol", "model"))

    # ------------------------------------------------------------------ physics-inspired terms (T-perp / QBOX)
    T_PERP_DIM: int = _f(16, "physics", "width of the perpendicular projection", unit="count", ge=1, step=1,
                         tunable=True)
    LAMBDA_T_PERP: float = _f(0.1, "physics", "batch variance tracks batch residual energy", unit="weight", ge=0.0,
                              tunable=True)
    LAMBDA_CASIMIR: float = _f(0.1, "physics", "disagreeing horizons need variance", unit="weight", ge=0.0,
                               tunable=True)
    LAMBDA_VAC: float = _f(0.0, "physics", "vacuum bandwidth threshold (0 = off)", unit="scaled", ge=0.0, tunable=True)
    LAMBDA_HD: float = _f(0.1, "physics", "variance ordered like realised volatility", unit="weight", ge=0.0,
                          tunable=True)
    LAMBDA_IFE: float = _f(0.1, "physics", "cross-horizon correlation hinge", unit="weight", ge=0.0, tunable=True)
    RHO_MAX: float = _f(0.95, "physics", "max allowed cross-horizon correlation", unit="dimensionless", ge=0.0, le=1.0,
                        tunable=True)
    VACUUM_E_MAX: float = _f(1.0, "physics", "per-dimension energy ceiling of the vacuum layer",
                             unit="dimensionless", gt=0.0, tunable=True)
    LAMBDA_VAC_OVERFLOW: float = _f(0.1, "physics", "overflow tracks residual magnitude", unit="weight", ge=0.0,
                                    tunable=True)

    # ------------------------------------------------------------------ learnable indicators
    MA_SPANS: List[int] = _f([5, 10, 30], "indicators", "initial EWMA periods (bars) of the learnable moving "
                             "averages, one learned period each", unit="bars", ge=1, step=1)
    MACD_SETTINGS: List[Dict[str, int]] = _f(_DEFAULT_MACD, "indicators", "initial fast / slow / signal EWMA "
                                             "periods (bars) of each learnable MACD, three learned periods each",
                                             unit="bars", ge=1, step=1)
    RSI_PERIODS: List[int] = _f([9, 14, 21], "indicators", "initial smoothing periods (bars) of the learnable RSIs",
                                unit="bars", ge=1, step=1)
    BB_PERIODS: List[int] = _f([10, 20, 25], "indicators", "initial periods (bars) of the learnable Bollinger bands",
                               unit="bars", ge=1, step=1)
    INDICATOR_FAMILIES: Dict[str, List] = _f({
        "atr": [7, 14, 28],
        "stoch": [{"k_period": 14, "d_period": 3}, {"k_period": 9, "d_period": 3},
                  {"k_period": 21, "d_period": 5}],
        "willr": [7, 14, 28],
        "keltner": [{"period": 20, "atr_period": 10}, {"period": 10, "atr_period": 10},
                    {"period": 40, "atr_period": 20}],
        "obv": [10, 20, 40],
        "vwap": [10, 20, 40],
        "mfi": [7, 14, 28],
        "adx": [7, 14, 28],
        "cci": [10, 20, 40],
        "donchian": [10, 20, 55],
    }, "indicators", "instances of further Indicators-registry families "
                                             "(family name -> list of instances, each a starting period in bars or a "
                                             "dict of parameter periods); naming ma / macd / rsi / bb here overrides "
                                             "the four fields above (neural_trade.indicators.indicator_instances); "
                                             "the default lists the ten OHLCV families of D-031 with 3 instances "
                                             "each ({} = the four close-only families alone); families that read "
                                             "high / low / volume need those series in INPUT_SERIES",
                                             unit="mapping", ge=1)
    ADAPTIVE_INDICATORS: bool = _f(True, "indicators", "shift each learned period per window through the meta_adjust "
                                   "network; off, every applied period in every window equals the family's learned "
                                   "global value (the frozen-twin switch, NT-033/NT-046)", unit="flag")
    INDICATOR_L2: float = _f(0.0, "indicators", "L2 on the indicator logits", unit="dimensionless", ge=0.0,
                             tunable=True)
    INDICATOR_LR_MULT: float = _f(5.0, "indicators", "indicator optimizer LR = LR * this", unit="dimensionless",
                                  gt=0.0, log=True, tunable=True)
    INDICATOR_GRAD_MULT: float = _f(5.0, "indicators", "straight-through gradient scale", unit="dimensionless", gt=0.0,
                                    tunable=True)
    MOMENTUM_CLIP_MIN: float = _f(2.0, "indicators", "period floor (1.0 saturated the logit)", unit="bars", gt=0.0)
    MOMENTUM_CLIP_MAX: Optional[float] = _f(None, "indicators", "period ceiling; None -> LOOKBACK", unit="bars",
                                            gt=0.0)
    EWMA_IMPL: str = _f("matrix", "indicators", "'matrix' (batched einsum) or 'scan' (reference)", unit="name",
                        choices=("matrix", "scan"), ignore_case=True)

    # ------------------------------------------------------------------ architecture / activations
    REG_MOMENTUM_L2: float = _f(0.0, "architecture", "L2 on the dense towers", unit="dimensionless", ge=0.0,
                                tunable=True)
    TRAIN_METRICS_EVERY: int = _f(10, "training",
                                  "update the training-set diagnostics every N steps (1 = every step); the "
                                  "training loss and all validation metrics are always exact", unit="steps", ge=1,
                                  step=1)
    TANH_SCALE: float = _f(1.0, "architecture", "unused", unit="dimensionless", deprecated=True)
    SIGMOID_SCALE: float = _f(1.0, "architecture", "unused", unit="dimensionless", deprecated=True)
    HUBER_DELTA: float = _f(1.0, "architecture", "unused (CustomTrainModel.huber has no caller; the point "
                            "loss is log(cosh), NT-028)", unit="scaled", gt=0.0, deprecated=True)
    USE_HUBER: bool = _f(True, "architecture", "legacy flag; the point loss is log(cosh)", unit="flag",
                         deprecated=True)

    # ------------------------------------------------------------------ stability
    GRAD_CLIP_NORM: float = _f(20.0, "stability", "global-norm clip, per optimizer group", unit="dimensionless",
                               ge=0.0)
    SEEDED_STOCHASTIC_LAYERS: bool = _f(
        False, "stability",
        "dropout (models/gru_attention.py) and VacuumSaturationNoise draw from a resettable "
        "tf.random.Generator per layer instead of TF's legacy stateful random ops, whose per-op "
        "state cannot be reset from Python. False (default) is today's behaviour everywhere the "
        "normal training path runs (scenario run, the golden run): unaffected. Screen mode "
        "(NT-092 phase 2) forces this True on every trial it builds, fresh or reused, so a reused "
        "trial's stochastic layers restart from its own seed instead of continuing whatever trial "
        "ran through the same persistent model before it - the only way phase 2 can match a fresh "
        "run bit-for-bit (docs/RUNBOOK.md 'Screen mode').", unit="flag")
    STRICT_LOSS_MASKS: bool = _f(
        False, "stability",
        "turn off every non-finite mask in losses/functions.py (about 46 sites, including the "
        "total itself and pnl_utility's own extra total, not only the ones that feed `total` "
        "directly): a non-finite value anywhere then makes the total non-finite, so train_step's "
        "finite-gradient guard (custom_model.py) sees it and counts the step, instead of the term "
        "being silently zeroed (D-026, NT-036). False (default) is today's behaviour, bit-for-bit "
        "(scripts/golden_run.py verify). True is for the CI stability tests and any run that "
        "wants a loud failure instead of a silent zero.", unit="flag")
    PROBE_GRADIENTS: bool = _f(
        False, "stability",
        "per-loss-term gradient probe (NT-037, D-026 'about 10%'): every PROBE_EVERY training "
        "steps, an extra persistent-tape backward pass (training=False, so it never perturbs "
        "dropout/noise) measures, per term and per variable group ('trunk', 'head', "
        "'indicator' - CustomTrainModel._probe_groups_of), its share of that group's gradient "
        "norm, its cosine with the group's total gradient, and (once per group) the mean and "
        "worst pairwise cosine conflict between any two terms; the value share is reported once, "
        "ungrouped. Logged into metrics.jsonl as probe_*. Off by default: the probe subgraph is "
        "never built when this is False (no probe_* key is written, no per-step cost).",
        unit="flag")
    PROBE_EVERY: int = _f(50, "stability",
                          "run the per-loss-term gradient probe every N training steps when "
                          "PROBE_GRADIENTS is on", unit="steps", ge=1, step=1)

    # ------------------------------------------------------------------ direction
    FOCAL_ALPHA: float = _f(0.5, "direction", "weight of the DOWN class", unit="fraction", ge=0.0, le=1.0)
    FOCAL_GAMMA: float = _f(2.0, "direction", "focusing exponent of the focal loss (only with DIRECTION_LOSS "
                            "'focal_dice')", unit="dimensionless", ge=0.0)
    DIRECTION_SKIP: bool = _f(True, "direction",
                              "add a linear logit from trailing-return features of the window to each direction head",
                              unit="flag")
    DIRECTION_SKIP_L2: float = _f(1e-4, "direction", "L2 on the direction skip weights", unit="dimensionless",
                                  ge=0.0, tunable=True)
    DIRECTION_DEEP_ZERO_INIT: bool = _f(False, "direction",
                                        "zero-initialise the deep (tower) direction logit's Dense kernel, so each "
                                        "direction head starts exactly at the DIRECTION_SKIP linear logit "
                                        "(has no effect when DIRECTION_SKIP is off: the bias is already 0, and "
                                        "without a skip there is nothing for the deep logit to start from - the "
                                        "head keeps its usual glorot-initialised Dense(1, sigmoid)); default off "
                                        "keeps the golden run bit-for-bit (NT-104)", unit="flag")
    DIRECTION_LOSS: str = _f("bce", "direction",
                             "'bce' (proper scoring rule) or 'focal_dice' (legacy: its optimum is a constant extreme)",
                             unit="name", choices=("bce", "focal_dice"))
    DIR_DEADBAND_BPS: float = _f(5.0, "direction", "|return| below this is neutral and masked", unit="bps", ge=0.0)

    # ------------------------------------------------------------------ variance
    VAR_FLOOR: float = _f(1e-4, "variance", "variance floor (scaled units^2)", unit="scaled^2", gt=0.0, log=True)
    VAR_CAP: float = _f(1e3, "variance", "variance cap for metrics (not applied in the loss)", unit="scaled^2",
                        gt=0.0, log=True)
    DELTA_MAPE_MIN_ABS: float = _f(1.0, "variance", "min |y| ($) for the delta safe-MAPE", unit="quote", ge=0.0)

    # ------------------------------------------------------------------ paths
    MODEL_PATH: str = _f("nn_learnable_indicators_v3.weights.h5", "paths", "weights file: model_checkpoint writes "
                         "the best-validation weights here, and training loads it first when it exists (a warm start, "
                         "NT-049); a RunContext points it into the run directory", unit="path")
    SCALER_PATH: str = _f("scaler_v3.joblib", "paths", "target-scaler file (joblib) written by the data processor and "
                          "after training; an input scaler goes next to it as *_input.joblib; a RunContext points it "
                          "into the run directory", unit="path")
    ARTIFACTS_DIR: str = _f("artifacts", "paths", "where Trainer writes the serving bundle", unit="path")
    PLUGINS_DIR: Optional[str] = _f(None, "paths", "directory of plugin modules to load at startup", unit="path")

    # ------------------------------------------------------------------ registries (component selection)
    MODEL_NAME: str = _f("gru_attention", "registries", "Models registry key", unit="key")
    LOSS_NAME: str = _f("custom_loss", "registries", "Losses objective key", unit="key")
    OPTIMIZER_NAME: str = _f("adam", "registries", "Optimizers key, main network", unit="key")
    INDICATOR_OPTIMIZER_NAME: str = _f("adam", "registries", "Optimizers key, indicator logits", unit="key")
    DATA_LOADER: str = _f("csv", "registries", "DataLoaders key", unit="key")
    PREPROCESSORS: List[str] = _f(["standardize_ohlcv", "sort_dedupe", "resample_bars", "drop_missing_close"],
                                  "registries", "Preprocessors keys, applied in order", unit="key")
    WINDOW_NORMALIZER: str = _f("window_relative", "registries", "input normalisation of the windows", unit="name",
                                choices=_WINDOW_NORMALIZERS)
    LAYERS: Dict[str, str] = _f({"indicators": "learnable_indicators", "positional_encoding": "positional_encoding",
                                 "vacuum_noise": "vacuum_saturation_noise", "energy_gate": "energy_gate"},
                                "registries", "Layers keys by role in the architecture", unit="key")
    METRICS: List[str] = _f(["mse", "rmse", "mae", "explained_variance", "corr", "r2", "safe_mape", "smape",
                             "wape", "direction_accuracy", "direction_f1", "mcc", "brier", "ece_pos",
                             "pit_ks", "coverage"], "registries", "Metrics keys, numpy tier (evaluation)", unit="key")
    STEP_METRICS: List[str] = _f(["dir_acc", "dir_sensitivity", "dir_specificity", "dir_bal_acc", "dir_f1",
                                  "dir_mcc", "dir_brier", "dir_ece", "pred_up_rate", "true_up_rate",
                                  "mean_dir_prob"], "registries", "Metrics keys, TF tier (train/test step)", unit="key")
    CALLBACKS: List[str] = _f(["csv_logger", "early_stopping", "model_checkpoint", "tqdm_progress",
                               "params_logger", "reduce_lr_on_plateau"], "registries",
                              "Callbacks keys, in order", unit="key")
    VISUALIZATION: str = _f("plotly_interactive", "registries", "Visualizations key", unit="key")

    # ------------------------------------------------------------------ optimizers
    ADAM_BETA1: float = _f(0.9, "optimizers", "first-moment decay of adam, adamw and nadam", unit="dimensionless",
                           ge=0.0, lt=1.0, tunable=True)
    ADAM_BETA2: float = _f(0.999, "optimizers", "second-moment decay of adam, adamw and nadam",
                           unit="dimensionless", ge=0.0, lt=1.0, tunable=True)
    ADAM_EPSILON: float = _f(1e-7, "optimizers", "Keras default", unit="dimensionless", gt=0.0, log=True)
    WEIGHT_DECAY: float = _f(0.004, "optimizers", "adamw only", unit="dimensionless", ge=0.0)
    SGD_MOMENTUM: float = _f(0.9, "optimizers", "momentum of the sgd_momentum optimizer", unit="dimensionless",
                             ge=0.0, le=1.0)
    SGD_NESTEROV: bool = _f(True, "optimizers", "Nesterov momentum for the sgd_momentum optimizer", unit="flag")

    # ------------------------------------------------------------------ ops
    LOSS_WEIGHT_SCHEDULE: Optional[Dict[str, Any]] = _f(None, "ops",
                                                   "{'lambda_hd': {epoch: value, ...}, ...} for the lambda_schedule callback",
                                                   unit="mapping")
    ABLATE_LAMBDAS: List[str] = _f([], "ops", "LAMBDA_* names forced to 0 after calibration (ablation)", unit="key")
    SEED: int = _f(42, "ops", "global seed set by train_and_evaluate", unit="seed", ge=0, step=1)

    # ================================================================== behaviour
    def __post_init__(self):
        for name in ("EXTENDED_TREND_PERIODS", "HORIZON_STEPS"):
            if isinstance(getattr(self, name), tuple):
                setattr(self, name, list(getattr(self, name)))
        if self.MOMENTUM_CLIP_MAX is None:
            self.MOMENTUM_CLIP_MAX = self.LOOKBACK
        self.validate()

    # --------------------------------------------------------------- metadata
    @classmethod
    def field_specs(cls) -> Dict[str, FieldSpec]:
        """Every field's :class:`FieldSpec` (group, doc, unit, range or choices, tunable, deprecated), in
        declaration order."""
        cached = _SPECS.get(cls)
        if cached is None:
            hints = typing.get_type_hints(cls)
            cached = {}
            for f in fields(cls):
                m = f.metadata
                default = f.default if f.default is not MISSING else f.default_factory()
                hint = hints[f.name]
                nullable = typing.get_origin(hint) is typing.Union and type(None) in typing.get_args(hint)
                cached[f.name] = FieldSpec(
                    name=f.name, group=m.get("group", "ops"), doc=m.get("doc", ""), unit=m.get("unit"),
                    type=str(f.type), default=default, nullable=nullable,
                    minimum=m.get("minimum"), maximum=m.get("maximum"),
                    min_inclusive=m.get("min_inclusive", True), max_inclusive=m.get("max_inclusive", True),
                    step=m.get("step"), log=m.get("log", False), choices=m.get("choices"),
                    ignore_case=m.get("ignore_case", False), tunable=m.get("tunable", False),
                    deprecated=m.get("deprecated", False))
            _SPECS[cls] = cached
        return dict(cached)

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise :class:`InvalidConfigurationError` (a ``ValueError``) on invalid settings: the
        cross-field rules below, then every field's declared range or choices (``field_specs``)."""
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
        if int(self.TRAIN_METRICS_EVERY) < 1:
            bad("TRAIN_METRICS_EVERY must be >= 1")
        if self.CONFORMAL_SCALE not in ("none", "sigma", "realized_vol"):
            bad(f"CONFORMAL_SCALE must be 'none', 'sigma' or 'realized_vol', got {self.CONFORMAL_SCALE!r}")
        if self.PNL_GAMMA <= 0:
            bad("PNL_GAMMA must be > 0")
        if self.PNL_COST_BPS < 0:
            bad("PNL_COST_BPS must be >= 0")
        if self.PNL_SIGMA_SOURCE not in ("realized_vol", "model"):
            bad(f"PNL_SIGMA_SOURCE must be 'realized_vol' or 'model', got {self.PNL_SIGMA_SOURCE!r}")
        if str(self.EWMA_IMPL).lower() not in ("matrix", "scan"):
            bad(f"EWMA_IMPL must be 'matrix' or 'scan', got {self.EWMA_IMPL!r}")
        canonical = ("open", "high", "low", "close", "volume")
        series = list(self.INPUT_SERIES or [])
        if "close" not in series:
            bad(f"INPUT_SERIES must include 'close', got {series}")
        if [s for s in canonical if s in series] != series or len(set(series)) != len(series):
            bad(f"INPUT_SERIES must be a subsequence of {list(canonical)} without repeats, got {series}")
        if not (0 < self.MOMENTUM_CLIP_MIN < (self.MOMENTUM_CLIP_MAX or self.LOOKBACK)):
            bad("need 0 < MOMENTUM_CLIP_MIN < MOMENTUM_CLIP_MAX")
        negative = [k for k, v in self.lambda_weights().items() if v < 0]
        if negative:
            bad(f"loss weights must be >= 0: {negative}")
        unknown_ablate = [n for n in self.ABLATE_LAMBDAS if n not in self.lambda_weights()]
        if unknown_ablate:
            bad(f"ABLATE_LAMBDAS names unknown LAMBDA_* fields: {unknown_ablate}")
        # The declared range or choices of every field (FieldSpec), after the rules above so their
        # established messages still win where both apply.
        problems = [msg for spec in self.field_specs().values()
                    if (msg := spec.check(getattr(self, spec.name))) is not None]
        if problems:
            bad("; ".join(problems))
        # Deprecated fields have no effect on the objective or the training path (NT-028, D-029):
        # a config that still sets one away from its default is warned, not refused, so old
        # configs keep loading.
        for spec in self.field_specs().values():
            if spec.deprecated and getattr(self, spec.name) != spec.default:
                warnings.warn(
                    f"Config.{spec.name} is deprecated and has no effect (default {spec.default!r})",
                    DeprecationWarning, stacklevel=2)

    # --------------------------------------------------------------- derived
    @property
    def momentum_clip_max(self) -> float:
        return float(self.MOMENTUM_CLIP_MAX if self.MOMENTUM_CLIP_MAX is not None else self.LOOKBACK)

    @property
    def effective_var_floor(self) -> float:
        return self.VAR_FLOOR

    def input_series(self) -> Tuple[str, ...]:
        """The bar series of each input window (NT-047); ``('close',)`` = the legacy 2-D input."""
        return tuple(self.INPUT_SERIES or ["close"])

    def close_channel(self) -> int:
        """Index of the close channel inside a multi-series input window."""
        return self.input_series().index("close")

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


_SPECS: Dict[type, Dict[str, FieldSpec]] = {}


def metadata_problems(cls=Config) -> List[str]:
    """What is missing or inconsistent in a Config class's field metadata (empty = complete):
    a field without a doc or a unit from ``UNITS``, a default outside its own range or choices, a
    log hint without a positive lower bound, or a field that is both tunable and deprecated."""
    problems = []
    for spec in cls.field_specs().values():
        if not str(spec.doc).strip():
            problems.append(f"{spec.name}: no doc")
        if spec.unit not in UNITS:
            problems.append(f"{spec.name}: unit {spec.unit!r} is not in UNITS")
        msg = spec.check(spec.default)
        if msg is not None:
            problems.append(f"{spec.name}: default {msg}")
        if spec.log and (spec.minimum is None or spec.minimum < 0 or (spec.minimum == 0 and spec.min_inclusive)):
            problems.append(f"{spec.name}: a log-scale hint needs a positive lower bound")
        if spec.tunable and spec.deprecated:
            problems.append(f"{spec.name}: tunable and deprecated")
    return problems


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


__all__ = ["Config", "FieldSpec", "GROUPS", "UNITS", "metadata_problems"]
