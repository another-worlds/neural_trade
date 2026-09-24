"""Training record: the per-epoch dashboard, the health card, the epoch table and two companion figures.

The trainer logs ~260 values per epoch (metrics.jsonl): total and component losses for train and
validation, direction metrics per horizon for both the direction head and the Gaussian readout of
the price head, calibration (ECE, PIT-KS), prediction bias, the physics terms, the loss weights,
gradient norm, learning rates and the learned indicator periods (drawn by indicator_evolution).

* :func:`training_dashboard_figure` - 12 panels: total loss (best epoch and SERVED epoch marked),
  what the validation loss is made of (each term as it enters the total), direction-head and
  price-head MCC and balanced accuracy with their 95% chance bands, Brier skill against the base
  rate with the ceiling a forecast without information stays under, ECE and PIT-KS of the raw heads
  with the level a calibrated head stays under, up-rate bias, the physics terms (exact zeros on
  their own rows), learning rates and the gradient norm with its clip level. Validation solid,
  training dotted, horizons in fixed colours.
* :func:`training_health` / :func:`training_health_html` - the verdicts a person scans first, each
  with its meaning in plain text; the per-horizon verdicts are graded at the served epoch against
  the chance band of the validation block, trends against their Student-t level. The HTML ends
  with :func:`epoch_table_html`.
* :func:`epoch_table_html` - the exact numbers at the served epoch: every loss term with its weight
  and share of the validation loss, and every direction metric per horizon (val and train) with
  its reference and chance range.
* :func:`direction_detail_figure` - accuracy, sensitivity / specificity, F1, Brier and mean P(up)
  per horizon, each against its no-skill reference and that reference's 95% noise range.

Chance ranges use n_eff = validation samples // horizon bars (overlapping targets, as in the
evaluation report) and are withheld below 10 effective samples.
* :func:`loss_terms_figure` - each loss term per horizon on its own linear axis.
* :func:`batch_loss_figure` - the live within-epoch loss (and, when given, the running predicted
  up-rate per horizon, the class-collapse signal).

The SERVED epoch is the one whose weights were evaluated on TEST, calibrated and bundled
(``artifacts/meta.json`` ``weights_epoch``; see :func:`served_epoch` for runs saved before it
was recorded).

``history``: a list of epoch dicts (metrics.jsonl rows or Keras epoch logs), a Keras History,
a {key: [values]} dict, a DataFrame, or a path to metrics.jsonl (whose run directory is then read
for the served epoch and the validation size).
"""
from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

LOSS_COMPONENTS: Tuple[Tuple[str, str], ...] = (
    ("point_loss", "point"), ("trend_loss", "trend"), ("dir_loss", "direction"), ("nll_loss", "NLL"),
    ("crps_loss", "CRPS"), ("soft_ece_loss", "soft ECE"), ("vol_loss", "volatility"), ("reg_loss", "regulariser"),
)
PHYSICS_TERMS: Tuple[Tuple[str, str], ...] = (
    ("t_perp_loss", "T-perp"), ("casimir_loss", "Casimir"), ("hd_loss", "HD"), ("ife_loss", "IFE"),
    ("vac_loss", "vacuum"), ("vac_overflow_loss", "vac overflow"),
)
_PHYSICS_LAMBDA = {"t_perp_loss": "lambda_t_perp", "casimir_loss": "lambda_casimir", "hd_loss": "lambda_hd",
                   "ife_loss": "lambda_ife", "vac_overflow_loss": "lambda_vac_overflow"}
LN2 = math.log(2.0)

PANELS = (
    "Total loss",
    "Val loss by term, as each enters the total (stacked)",
    "Direction head MCC (validation / training)",
    "Price head MCC (Gaussian readout)",
    "Direction head balanced accuracy",
    "Direction Brier skill vs base rate (shade: by chance)",
    "Direction ECE, raw head (shade: calibrated, by chance)",
    "PIT-KS, raw price head (shade: calibrated, by chance)",
    "Up-call bias: predicted − true up-rate (0 = unbiased)",
    "Physics terms × λ (log)",
    "Learning rate used in each epoch (log)",
    "Gradient norm: epoch mean, both groups, pre-clip",
)
# Titles stay under ~55 characters: a subplot title wider than its column overlaps its neighbour.
_Y_TITLES = ("loss", "loss", "MCC", "MCC", "balanced accuracy", "1 − Brier / p(1 − p)", "ECE", "KS statistic",
             "pred − true up-rate", "loss × λ", "learning rate", "global norm")

# Series colours here skip T.OTHER_SERIES[2] (#008300): next to the status green T.GOOD (#0ca30c) it
# would read as "good". Pairs of related terms share a colour, the second one hatched.
_SERIES = tuple(c for i, c in enumerate(T.OTHER_SERIES) if i != 2)
# Loss-term contributions, bottom to top of the stack: (key, label, colour, fill pattern)
_CONTRIB = (
    ("point", "point", _SERIES[0], ""),
    ("trend", "trend", _SERIES[0], "/"),
    ("direction", "direction BCE", _SERIES[1], ""),
    ("nll", "NLL", _SERIES[2], ""),
    ("crps", "CRPS", _SERIES[2], "/"),
    ("soft_ece", "soft ECE", _SERIES[3], ""),
    ("volatility", "volatility", _SERIES[4], ""),
    ("physics", "physics", _SERIES[5], ""),
    ("other", "other: coherence (not logged)", _SERIES[5], "x"),
)
_FORECAST_TERMS = ("point", "trend", "direction")
_CALIBRATION_TERMS = ("nll", "crps", "soft_ece", "volatility", "physics")
# physics colours: distinct neighbours (Casimir and IFE sit close together on the axis; HD is not the
# salmon of soft ECE, which reads too close to IFE's pink)
_PHYSICS_COLORS = {"t_perp_loss": _SERIES[0], "casimir_loss": _SERIES[2], "hd_loss": _SERIES[4],
                   "ife_loss": _SERIES[1], "vac_loss": _SERIES[3], "vac_overflow_loss": _SERIES[5]}

_CFG_LAMBDA = {
    "lambda_short": "LAMBDA_SHORT", "lambda_point": "LAMBDA_POINT", "lambda_long": "LAMBDA_LONG",
    "lambda_extended_trend": "LAMBDA_EXTENDED_TREND", "lambda_dir": "LAMBDA_DIR", "lambda_var": "LAMBDA_VAR",
    "lambda_vol": "LAMBDA_VOL", "lambda_crps": "LAMBDA_CRPS", "lambda_soft_ece": "LAMBDA_SOFT_ECE",
    "lambda_t_perp": "LAMBDA_T_PERP", "lambda_casimir": "LAMBDA_CASIMIR", "lambda_hd": "LAMBDA_HD",
    "lambda_ife": "LAMBDA_IFE", "lambda_vac_overflow": "LAMBDA_VAC_OVERFLOW",
}
# outer weights: (key in meta.json calibration_lambdas when CALIB_OUTER calibrated them, Config field, default)
_OUTER = {"trend": ("lambda_trend_outer", "LAMBDA_TREND_OUTER", 1.0), "dir": ("lambda_dir_outer", "LAMBDA_DIR_OUTER", 1.0),
          "nll": ("lambda_nll_outer", "LAMBDA_NLL_OUTER", 1.0), "coherence": (None, "LAMBDA_COHERENCE", 1.0),
          "dir_align": (None, "LAMBDA_DIR_ALIGN_OUTER", 0.0), "inter": (None, "LAMBDA_INTER", 1.0)}
# Fixed factors in losses.functions.custom_loss: ``0.1 * inter_reg`` and ``0.1 * vol_loss``.
_VOL_FACTOR = 0.1
_REG_FACTOR = 0.1

_ROW_PX = 18            # one row of a panel legend
_HEAD_PX = 20           # a subplot heading
_MIN_FIG_WIDTH = 1000   # panel legends are packed to fit their column in a figure this wide (a notebook cell)
_MARGIN = dict(l=64, r=24, b=56)
_SUB_CHARS = 120        # a subtitle line this long fits a 900 px notebook cell (12 px font, ~6 px a character)
_SUB_LINE_PX = 17       # height of one subtitle line

# ------------------------------------------------------------------ chance ranges on the validation block
# Every range uses n_eff = validation samples / horizon bars (overlapping targets), as the evaluation
# report does. Below _MIN_N_EFF effective samples no range is drawn and nothing is graded against one.
_MIN_N_EFF = 10
# Kolmogorov 95% critical value: a calibrated head's PIT-KS stays below 1.358 / sqrt(n_eff).
_KS95 = 1.358
# 95th percentile of ECE * sqrt(n_eff / p(1 - p)) for a CALIBRATED head whose P(up) falls in ~3 of the
# 10 ECE bins (simulated: 1.96 for 1 bin, 2.24 for 2, 2.47 for 3, 2.66 for 4). The direction head's
# P(up) sits in 2-3 bins around 0.5 (effective bin count 2.7-3.0 on the reference run).
_ECE95 = 2.47


# ------------------------------------------------------------------ data
def history_frame(history) -> pd.DataFrame:
    """Normalise any history-like input to a DataFrame with one row per epoch."""
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        if isinstance(history, str) or hasattr(history, "read_text"):
            from neural_trade.telemetry.epoch_logger import read_metrics

            history = read_metrics(str(history))
        if hasattr(history, "history") and not isinstance(history, (list, dict)):
            history = history.history
        if isinstance(history, dict):
            df = pd.DataFrame(history)
        else:
            df = pd.DataFrame(list(history or []))
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(len(df))
    df = df.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    if "seconds" in df.columns and "epoch_seconds" not in df.columns:
        df["epoch_seconds"] = df["seconds"]
    return df


def _col(df, key) -> Optional[np.ndarray]:
    if key not in df.columns:
        return None
    v = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
    return v if np.isfinite(v).any() else None


def run_meta(path) -> Optional[dict]:
    """The serving bundle's meta.json for a run directory, its artifacts directory, or a file inside the
    run directory (e.g. metrics.jsonl). None when there is no bundle (a live run, a bare history)."""
    if path is None:
        return None
    p = Path(path)
    base = p.parent if p.suffix else p
    for cand in (base / "artifacts" / "meta.json", base / "meta.json"):
        try:
            if cand.is_file():
                meta = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and ("format_version" in meta or "pred_scale" in meta):
                    return meta   # the run-level meta.json (RunContext) is a different file
        except (OSError, ValueError):
            continue
    return None


def _configured(config, name: str) -> bool:
    """Whether callback ``name`` is in Config.CALLBACKS (the Config() default list when unknown)."""
    if config is None:
        return name in ("early_stopping", "reduce_lr_on_plateau", "model_checkpoint")   # Config() defaults
    return name in list(getattr(config, "CALLBACKS", None) or [])


def _early_stopping_configured(config) -> bool:
    return _configured(config, "early_stopping")


def _older_logger_rows(df: pd.DataFrame) -> bool:
    """Rows written by the jsonl epoch logger of a trainer from before the served epoch was recorded.

    Such a logger (identified by its ``run_id`` / ``sec_per_step`` columns) ran after
    ReduceLROnPlateau, and its trainer did not restore the best weights when a run reached EPOCHS.
    Current trainers add ``lr_used`` (the rate at the epoch's start) to every row; a logger that
    snapshots the rate itself writes ``lr_next``."""
    cols = set(df.columns)
    return bool(cols & {"run_id", "sec_per_step"}) and not (cols & {"lr_used", "lr_next"})


def served_epoch(history, config=None, *, meta: Optional[dict] = None) -> Tuple[Optional[int], str]:
    """The 1-based epoch whose weights were evaluated on TEST, calibrated and bundled, and how it is known.

    1. ``meta["weights_epoch"]`` (artifacts/meta.json, written by the trainer since it restores the
       best-validation weights itself).
    2. A bundle without that key, or a metrics.jsonl from such a trainer without a bundle (see
       :func:`_older_logger_rows`), was written before the fix, when Keras 2.10 EarlyStopping restored
       the best weights ONLY when it stopped the run: the best epoch if it fired (``epochs since the
       best >= EARLY``), otherwise the last epoch.
    3. No bundle (a live run or a bare history): the best validation epoch when early_stopping is
       configured (the trainer restores it when training ends), else the last epoch.
    """
    if meta and meta.get("weights_epoch") is not None:
        return int(meta["weights_epoch"]), "recorded in artifacts/meta.json"
    if meta and "weights_epoch" in meta:     # recorded as null: no training ran, the weights were loaded
        return None, "unknown (the bundle's weights were loaded, not trained)"
    df = history_frame(history)
    val = _col(df, "val_loss")
    n = len(df)
    if n == 0 or val is None:
        return None, "unknown (no validation loss)"
    best = int(np.nanargmin(val))
    es = _early_stopping_configured(config)
    if meta is not None or _older_logger_rows(df):
        what = "older bundle" if meta is not None else "older run"
        early = int(getattr(config, "EARLY", 0) or 0) if config is not None else 0
        if es and early and n - 1 - best >= early:
            return best + 1, f"{what}: EarlyStopping fired and restored the best epoch"
        return n, (f"{what}: EarlyStopping did not stop the run and Keras 2.10 then keeps the last "
                   "epoch's weights")
    if es:
        return best + 1, "best validation epoch (restored when training ends)"
    return n, "last epoch (no early_stopping callback)"


@dataclass
class _Ctx:
    df: pd.DataFrame
    x: List[int]
    n: int
    val: Optional[np.ndarray]
    loss: Optional[np.ndarray]
    best: Optional[int]          # 0-based row of the lowest val loss
    served: Optional[int]        # 0-based row of the served epoch
    served_epoch: Optional[int]  # 1-based
    served_note: str
    n_val: Optional[int]
    meta: Optional[dict]
    config: Any


def _resolve(history, config, weights_epoch=None, n_val=None, meta=None, run_dir=None) -> _Ctx:
    df = history_frame(history)
    if meta is None:
        src = run_dir if run_dir is not None else (history if isinstance(history, (str, Path)) else None)
        meta = run_meta(src)
    x = (df["epoch"].to_numpy() + 1).astype(int).tolist()
    val, loss = _col(df, "val_loss"), _col(df, "loss")
    best = int(np.nanargmin(val)) if val is not None else None
    if weights_epoch is not None:
        ep = int(weights_epoch)
        note = ("recorded in artifacts/meta.json" if meta and meta.get("weights_epoch") == ep
                else "given by the caller")
    else:
        ep, note = served_epoch(df, config, meta=meta)
    served = None
    if ep is not None and x:
        served = x.index(ep) if ep in x else int(np.clip(ep - x[0], 0, len(x) - 1))
    if n_val is None and meta:
        n_val = ((meta.get("fold") or {}).get("val"))
    return _Ctx(df, x, len(df), val, loss, best, served, ep, note, int(n_val) if n_val else None, meta, config)


def _n_eff(ctx: _Ctx, h: str) -> Optional[int]:
    if not ctx.n_val or ctx.config is None or getattr(ctx.config, "HORIZON_STEPS", None) is None:
        return None
    return int(S.n_eff(ctx.n_val, S.horizon_steps(ctx.config, h)))


def _enough(ctx: _Ctx, h: str) -> Optional[int]:
    """n_eff of horizon ``h`` when it is large enough for a chance range, else None."""
    n = _n_eff(ctx, h)
    return n if n is not None and n >= _MIN_N_EFF else None


def _too_few(ctx: _Ctx) -> bool:
    return any(_n_eff(ctx, h) is not None and _enough(ctx, h) is None for h in T.HORIZONS)


def _chance(ctx: _Ctx, h: str) -> Optional[float]:
    """Half-width of the 95% band an MCC stays inside by chance on the validation block (n_eff = n / h bars),
    in MCC (r) units, so below 1 (not the Fisher-z half-width: n_eff 10 gives 0.63, not 0.74); None when
    the validation size is unknown or n_eff < _MIN_N_EFF."""
    n = _enough(ctx, h)
    if n is None:
        return None
    return min(1.0, S.corr_null_r(n))


def _ks_crit(ctx: _Ctx, h: str) -> Optional[float]:
    """PIT-KS a calibrated price head stays below in 95% of validation blocks."""
    n = _enough(ctx, h)
    return None if n is None else min(1.0, _KS95 / math.sqrt(n))


def _ece_crit(ctx: _Ctx, h: str, p: float) -> Optional[float]:
    """ECE a calibrated direction head stays below in 95% of validation blocks (P(up) over ~3 bins)."""
    n = _enough(ctx, h)
    if n is None or not (0 < p < 1):
        return None
    return min(1.0, _ECE95 * math.sqrt(p * (1 - p) / n))


def _skill_ceiling(ctx: _Ctx, h: str) -> Optional[float]:
    """Brier skill a forecast WITHOUT information exceeds with probability at most 2.5%: z^2 / n_eff.

    Brier skill against the block's own up-rate p is (2 cov(q, y) - E(q - p)^2) / p(1 - p). Without
    information cov(q, y) ~ N(0, var(q) p(1 - p) / n_eff); with r = sd(q) / sd(y) the skill stays below
    2 r z / sqrt(n_eff) - r^2, whose largest value over r is z^2 / n_eff."""
    n = _enough(ctx, h)
    return None if n is None else min(1.0, S.Z95 ** 2 / n)


def _ref_noise(ctx: _Ctx, h: str, kind: str, p: float) -> Optional[float]:
    """95% half-width of the sampling noise around a no-skill reference on the validation block (up-rate
    ``p``): how far a head whose true value equals the reference lands from it by chance.

    acc: binomial on n_eff at the majority-class accuracy; f1: F1 of always-up, binomial on the n_eff up
    moves (close to the delta-method spread of random classifiers calling up 50-100% of the time);
    sens / spec: a recall of 0.5 on the up / down moves; bias: mean P(up) of a calibrated head vs the rate."""
    n = _enough(ctx, h)
    if n is None or not (0 < p < 1):
        return None
    z = S.Z95
    if kind == "acc":
        a = max(p, 1 - p)
        return z * math.sqrt(a * (1 - a) / n)
    if kind == "f1":
        f = 2 * p / (1 + p)
        return z * math.sqrt(f * (1 - f) / (n * p))
    if kind == "sens":
        return z * math.sqrt(0.25 / (n * p))
    if kind == "spec":
        return z * math.sqrt(0.25 / (n * (1 - p)))
    if kind == "bias":
        return z * math.sqrt(p * (1 - p) / n)
    raise ValueError(kind)


def _val_rate(ctx: _Ctx, h: str, prefix: str = "val_") -> float:
    """The block's up-rate at the served (else last) epoch (constant for the fixed validation block)."""
    r = _col(ctx.df, f"{prefix}true_up_rate_{h}")
    return _at(r, ctx.served if ctx.served is not None else ctx.n - 1)


def _at(a: Optional[np.ndarray], i: Optional[int]) -> float:
    if a is None or i is None or i >= len(a):
        return float("nan")
    return float(a[i])


def _weight(df, key, config) -> Optional[np.ndarray]:
    """A per-term loss weight per epoch: the logged lambda_* column, else the Config value."""
    v = _col(df, key)
    if v is not None:
        return pd.Series(v).ffill().bfill().to_numpy()
    name = _CFG_LAMBDA.get(key)
    if config is not None and name and hasattr(config, name):
        return np.full(len(df), float(getattr(config, name)))
    return None


def _outer(name, config, meta) -> float:
    key, cfg_name, default = _OUTER[name]
    cal = (meta or {}).get("calibration_lambdas") or {}
    if key and cal.get(key) is not None:
        return float(cal[key])
    if config is not None and hasattr(config, cfg_name):
        return float(getattr(config, cfg_name))
    return default


def loss_contributions(history, config=None, *, prefix: str = "val_", meta: Optional[dict] = None
                       ) -> Optional[pd.DataFrame]:
    """Each loss term per epoch exactly as it enters the total in ``losses.functions.custom_loss``.

    Columns: point, trend, direction, nll, crps, soft_ece, volatility, physics, regulariser, other
    (= the logged total minus the known terms: mostly the coherence penalty, which is not logged),
    total. The logged values are on mixed bases: point, trend, volatility and the physics terms carry
    their lambda already; direction, NLL, CRPS and soft ECE are raw sums and are multiplied here by
    their lambda (per-epoch ``lambda_*`` columns, else the Config value) and outer weight; volatility
    and the regulariser enter at 0.1. Exact for validation; training components are sampled every
    TRAIN_METRICS_EVERY steps, so there the parts only approximately add up. None when the terms or
    their weights are not available.
    """
    df = history_frame(history)
    n = len(df)

    def c(k):
        return _col(df, prefix + k)

    total = c("loss")
    lam = {k: _weight(df, k, config) for k in ("lambda_dir", "lambda_var", "lambda_crps", "lambda_soft_ece")}
    raw = {k: c(k) for k in ("point_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss")}
    if total is None or any(v is None for v in raw.values()) or any(v is None for v in lam.values()):
        return None
    ext = [c(f"extended_h{i}") for i in range(3)]
    trend = np.sum(ext, axis=0) if all(e is not None for e in ext) else c("trend_loss")
    zero = np.zeros(n)
    out = pd.DataFrame({
        "point": raw["point_loss"],
        "trend": _outer("trend", config, meta) * (trend if trend is not None else zero),
        "direction": _outer("dir", config, meta) * lam["lambda_dir"] * raw["dir_loss"],
        "nll": _outer("nll", config, meta) * lam["lambda_var"] * raw["nll_loss"],
        "crps": lam["lambda_crps"] * raw["crps_loss"],
        "soft_ece": lam["lambda_soft_ece"] * raw["soft_ece_loss"],
        "volatility": _VOL_FACTOR * np.nan_to_num(c("vol_loss") if c("vol_loss") is not None else zero),
        "physics": np.sum([np.nan_to_num(c(k)) for k, _ in PHYSICS_TERMS if c(k) is not None] or [zero], axis=0),
        "regulariser": _REG_FACTOR * np.nan_to_num(c("inter_reg") if c("inter_reg") is not None else zero),
    })
    out["other"] = total - out.sum(axis=1)
    out["total"] = total
    return out


def _weight_labels(ctx: _Ctx) -> Dict[str, str]:
    """How each term is weighted in the total, with the values at the served (else last) epoch."""
    df, cfg, meta = ctx.df, ctx.config, ctx.meta
    i = ctx.served if ctx.served is not None else ctx.n - 1

    def lam(k):
        w = _weight(df, k, cfg)
        return _at(w, i) if w is not None else float("nan")

    def f(v):
        return "?" if not np.isfinite(v) else f"{v:.3g}"

    o = {k: _outer(k, cfg, meta) for k in _OUTER}
    phys = [lam(k) for k in _PHYSICS_LAMBDA.values()]
    phys_txt = f(phys[0]) if phys and np.allclose(phys, phys[0], equal_nan=True) else "per term"
    return {
        "point": f"λ {f(lam('lambda_short'))} / {f(lam('lambda_point'))} / {f(lam('lambda_long'))} (inside)",
        "trend": f"λ_ext {f(lam('lambda_extended_trend'))} (inside) × {f(o['trend'])}",
        "direction": f"λ_dir {f(lam('lambda_dir'))} × {f(o['dir'])}",
        "nll": f"λ_var {f(lam('lambda_var'))} × {f(o['nll'])}",
        "crps": f"λ_crps {f(lam('lambda_crps'))}",
        "soft_ece": f"λ_soft_ece {f(lam('lambda_soft_ece'))}",
        "volatility": f"λ_vol {f(lam('lambda_vol'))} (inside) × {_VOL_FACTOR:g}",
        "physics": f"λ {phys_txt} (inside)",
        "regulariser": f"LAMBDA_INTER {f(o['inter'])} (inside) × {_REG_FACTOR:g}",
        "other": f"LAMBDA_COHERENCE {f(o['coherence'])}" + (
            f", dir-align × {f(o['dir_align'])}" if o["dir_align"] > 0 else "") + " (not logged)",
    }


def _weight_factors(ctx: _Ctx) -> Dict[str, float]:
    """The multiplier the dashboard applies to a logged term (terms logged with λ inside are absent)."""
    df, cfg, meta = ctx.df, ctx.config, ctx.meta
    i = ctx.served if ctx.served is not None else ctx.n - 1
    out = {"volatility": _VOL_FACTOR}
    for k, lam_key, outer in (("direction", "lambda_dir", "dir"), ("nll", "lambda_var", "nll"),
                              ("crps", "lambda_crps", None), ("soft_ece", "lambda_soft_ece", None)):
        w = _weight(df, lam_key, cfg)
        if w is not None:
            out[k] = _at(w, i) * (_outer(outer, cfg, meta) if outer else 1.0)
    trend_outer = _outer("trend", cfg, meta)
    if trend_outer != 1.0:
        out["trend"] = trend_outer
    return out


def _lr_in_effect(ctx: _Ctx, key: str = "lr") -> Tuple[Optional[np.ndarray], bool]:
    """The learning rate each epoch was trained with (``key`` 'lr' or 'lr_indicator').

    1. ``lr_used`` / ``lr_indicator_used``: read at the epoch's start by the trainer, exact.
    2. A logger that writes ``lr_next`` records the epoch's own rate in ``lr``.
    3. Rows of an older jsonl logger (:func:`_older_logger_rows`) with ReduceLROnPlateau configured
       and no bundle from a current trainer: the logger ran after the scheduler, so ``lr`` is the NEXT
       epoch's rate; it is shifted back one epoch, the first epoch taking Config.LR. Only the network
       rate is scheduled, so only ``lr`` is shifted.
    Returns (lr, shifted)."""
    df = ctx.df
    used = _col(df, f"{key}_used")
    if used is not None:
        return used, False
    lr = _col(df, key)
    if lr is None:
        return None, False
    current_bundle = ctx.meta is not None and "weights_epoch" in ctx.meta
    legacy = (key == "lr" and _older_logger_rows(df) and not current_bundle
              and _configured(ctx.config, "reduce_lr_on_plateau"))
    if legacy and len(lr) > 1:
        first = float(getattr(ctx.config, "LR", lr[0])) if ctx.config is not None else float(lr[0])
        return np.r_[first, lr[:-1]], True
    return lr, False


# ------------------------------------------------------------------ small statistics
def _t_crit(df: int, q: float = 0.975) -> float:
    """Student-t quantile ``q`` for ``df`` degrees of freedom; the default is the two-sided 95% level
    (4.30 at 2, 2.45 at 6, 2.10 at 18): a trend fitted on a handful of epochs needs a larger t."""
    from scipy.stats import t as student_t

    return float(student_t.ppf(q, max(1, int(df))))


def _ols(y) -> Optional[dict]:
    """Least-squares trend of the finite values of ``y`` against their index, with its t statistic and
    the 95% level it is graded against (``crit``, Student t on k - 2 degrees of freedom)."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    x, y = x[m], y[m]
    k = len(y)
    if k < 4:
        return None
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid ** 2) / (k - 2) / sxx)) if sxx > 0 else float("nan")
    if se > 0:
        t = float(slope / se)
    else:
        t = 0.0 if slope == 0 else math.copysign(float("inf"), slope)
    return dict(slope=float(slope), se=se, t=t, resid=resid, mean=float(np.mean(y)), k=k,
                change=float(slope * (x[-1] - x[0])), crit=_t_crit(k - 2))


def _trend(fit: Optional[dict]) -> str:
    """'falling' / 'rising' when the least-squares trend is beyond its 95% t level, else 'no clear trend'."""
    if fit is None:
        return "too few epochs"
    return "falling" if fit["t"] <= -fit["crit"] else ("rising" if fit["t"] >= fit["crit"] else "no clear trend")


def _wrap(parts: Sequence[str], width: int = _SUB_CHARS) -> List[str]:
    """Join ``parts`` with ' · ' into lines of at most ``width`` characters; a part longer than a line is
    broken at spaces."""
    def size(s):
        return len(re.sub("<[^>]+>", "", s))

    lines, cur = [], ""
    for p in parts:
        if not p:
            continue
        if cur and size(cur) + 3 + size(p) > width:
            lines.append(cur)
            cur = ""
        if size(p) <= width:
            cur = f"{cur} · {p}" if cur else p
            continue
        for word in p.split(" "):          # a long part: fill lines word by word
            if cur and size(cur) + 1 + size(word) > width:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


def _fmt(v, spec=".4g") -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return "0" if v == 0 else format(v, spec)


_RANK = {"good": 0, "info": 0, "warning": 1, "critical": 2}


def _worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


# ------------------------------------------------------------------ health
def training_health(history, config=None, *, weights_epoch: Optional[int] = None, n_val: Optional[int] = None,
                    meta: Optional[dict] = None, run_dir=None) -> Dict[str, dict]:
    """Named checks, each ``{"value", "status" (good|warning|critical|info), "detail", "label", "note"}``.

    ``weights_epoch``: the served epoch (1-based); by default from the run's artifacts/meta.json, else
    see :func:`served_epoch`. ``n_val``: validation samples (default: meta.json fold.val), which sets the
    chance band the per-horizon verdicts are graded against. ``meta`` / ``run_dir``: the bundle metadata
    or where to find it (found automatically when ``history`` is a metrics.jsonl path).
    """
    return _health(_resolve(history, config, weights_epoch, n_val, meta, run_dir))


def _health(ctx: _Ctx) -> Dict[str, dict]:
    df, n, val, loss, config = ctx.df, ctx.n, ctx.val, ctx.loss, ctx.config
    out: Dict[str, dict] = {}
    planned = int(getattr(config, "EPOCHS", n) or n) if config is not None else n
    out["epochs"] = dict(value=f"{n} / {planned}", status="info", detail="epochs finished / planned")
    if n == 0 or val is None:
        return out
    loss_ok = loss is not None
    bad_val = np.flatnonzero(~np.isfinite(val))
    bad_loss = np.flatnonzero(~np.isfinite(loss)) if loss_ok else np.array([], int)
    last_bad = (not np.isfinite(val[-1])) or (loss_ok and not np.isfinite(loss[-1]))
    if len(bad_val) or len(bad_loss):
        parts = []
        if len(bad_val):
            parts.append(f"val loss not finite at epoch {ctx.x[bad_val[-1]]} ({len(bad_val)} epochs)")
        if len(bad_loss):
            parts.append(f"train loss not finite at epoch {ctx.x[bad_loss[-1]]} ({len(bad_loss)} epochs)")
        out["divergence"] = dict(value="; ".join(parts), status="critical",
                                 detail="a non-finite loss: the run diverged (the checks below use finite epochs only)")
    b = ctx.best
    if ctx.served is None:
        best_detail = "lowest validation loss"
    elif ctx.served == b:
        best_detail = "lowest validation loss; the served weights are this epoch's"
    else:
        best_detail = (f"lowest validation loss; NOT the served weights, which are epoch {ctx.served_epoch} "
                       "(see 'served weights')")
    out["best"] = dict(label="best validation loss", value=f"{val[b]:.4f} @ epoch {ctx.x[b]}", status="info",
                       detail=best_detail)
    if ctx.served is not None:
        s = ctx.served
        same = s == b
        out["served"] = dict(
            label="served weights" if "restored when training ends" not in ctx.served_note else "weights to be served",
            value=f"epoch {ctx.served_epoch}" + (" = best" if same else f" ≠ best ({ctx.x[b]})"),
            status="good" if same else "warning",
            note=f"val loss {_fmt(val[s], '.4f')} ({ctx.served_note})",
            detail="the weights evaluated on TEST, calibrated and saved in the bundle; the per-horizon tiles "
                   "and the table below describe this epoch")
    last = f"val {_fmt(val[-1], '.4f')}" + (f" · train {_fmt(loss[-1], '.4f')}" if loss_ok else "")
    out["last epoch"] = dict(label=f"last epoch ({ctx.x[-1]})", value=last, status="critical" if last_bad else "info",
                             detail="epoch-end losses; train is the dropout-on running mean over the epoch")
    patience = int(getattr(config, "EARLY", 0) or 0) if config is not None else 0
    since = n - 1 - b
    if patience:
        frac = since / patience
        out["patience"] = dict(value=f"{since} / {patience}",
                               status="good" if frac < 0.5 else ("warning" if frac < 1 else "critical"),
                               detail="epochs since the best val loss / early-stopping patience")
    k = min(n, 8)
    fit = _ols(val[-k:])
    if last_bad:
        out["convergence"] = dict(value="n/a (non-finite loss)", status="critical",
                                  detail="val-loss trend: not computed on a diverged epoch")
    elif fit is not None:
        rel = fit["slope"] / abs(fit["mean"]) if fit["mean"] else 0.0
        state = {"falling": "improving", "rising": "worsening"}.get(_trend(fit), "no clear trend")
        out["convergence"] = dict(
            value=f"{100 * rel:+.2f}% / epoch ({state})",
            status={"improving": "good", "no clear trend": "warning", "worsening": "critical"}[state],
            note=f"t = {fit['t']:+.1f} over the last {fit['k']} epochs (95% level ±{fit['crit']:.2f})",
            detail="least-squares val-loss trend; a verdict needs |t| beyond the 95% Student-t level")
        noise = float(np.std(fit["resid"]) / abs(fit["mean"])) if fit["mean"] else float("nan")
        out["stability"] = dict(label="epoch-to-epoch noise", value=f"±{100 * noise:.2f}% around the trend",
                                status="info",
                                detail=f"scatter of the last {fit['k']} val losses around their trend: how large a "
                                       "one-epoch move has to be before it means anything")
    if loss_ok:
        gap = val - loss
        if last_bad:
            out["gap"] = dict(label="val − train loss", value="n/a (non-finite loss)", status="critical",
                              detail="not computed on a diverged epoch")
        else:
            gfit = _ols(gap[-k:])
            state, status = "steady", "info"
            if _trend(gfit) == "rising":
                state, status = "val rising vs train", "warning"
            elif _trend(gfit) == "falling":
                state = "val falling vs train"
            out["gap"] = dict(label="val − train loss", value=f"{gap[-1]:+.4f} ({state})", status=status,
                              note=None if gfit is None else f"trend t = {gfit['t']:+.1f} over {gfit['k']} epochs",
                              detail="the level is not comparable (train is a dropout-on running mean on another "
                                     "block); only its trend is a signal. The per-horizon tiles show overfitting")
        kk = min(n - 1, 10)
        if kk >= 2:
            d1, d2 = np.diff(loss[-kk - 1:]), np.diff(val[-kk - 1:])
            ok = np.isfinite(d1) & np.isfinite(d2)
            if ok.sum() >= 2:
                agree = int(np.sum(np.sign(d1[ok]) == np.sign(d2[ok])))
                out["agreement"] = dict(
                    label="train / val agreement", value=f"{agree} of {int(ok.sum())} moves ({100 * agree / ok.sum():.0f}%)",
                    status="info",
                    detail="epoch-to-epoch moves where train and val loss went the same way; train falls almost "
                           "every epoch, so this mostly counts the epochs in which val fell")
    fin = np.flatnonzero(np.isfinite(val))
    if len(fin) >= 2:
        out["progress"] = _progress_tile(ctx, fin, last_bad)
    s = ctx.served if ctx.served is not None else n - 1
    for key, up_key, name, label in (("dir", "", "direction", "direction head"),
                                     ("gauss_dir", "gauss_", "price head", "price head (Gaussian readout)")):
        for h in T.HORIZONS:
            tile = _head_tile(ctx, s, key, up_key, h)
            if tile is not None:
                tile["label"] = f"{label} · {T.horizon_label(h, config)}"
                out[f"{name} {h}"] = tile
    skill = _skill_tile(ctx, s)
    if skill is not None:
        out["skill"] = skill
    nf = _col(df, "nonfinite_grad_steps")
    if nf is not None:
        total = float(np.nansum(nf))
        gn = _col(df, "grad_global_norm")
        clip = float(getattr(config, "GRAD_CLIP_NORM", 0) or 0) if config is not None else 0.0
        status = "good" if total == 0 else "critical"
        note = None
        if gn is not None:
            note = f"epoch-mean norm {_fmt(np.nanmin(gn), '.3g')}–{_fmt(np.nanmax(gn), '.3g')}" + (
                f" (clip {clip:g} per group)" if clip else "")
            if clip and np.nanmax(gn) >= clip:
                status = _worse(status, "warning")
        out["gradients"] = dict(value=f"{int(total)} non-finite steps", status=status, note=note,
                                detail="training steps whose gradient had a NaN/inf (skipped by the guard)")
    lr, _ = _lr_in_effect(ctx)
    if lr is not None:
        cuts = int(np.sum(np.diff(lr[np.isfinite(lr)]) < 0))
        out["learning rate"] = dict(value=f"{lr[-1]:.2e} ({cuts} reduction{'' if cuts == 1 else 's'})", status="info",
                                    detail="rate of the last epoch; ReduceLROnPlateau cuts so far")
    sec = _col(df, "epoch_seconds")
    if sec is not None:
        left = max(0, planned - n)
        out["time"] = dict(value=f"{np.nanmedian(sec):.0f} s / epoch" + (f", ~{left * np.nanmedian(sec) / 60:.0f} min left"
                                                                         if left else ""),
                           status="info", detail="median epoch time")
    return out


def _progress_tile(ctx: _Ctx, fin: np.ndarray, last_bad: bool) -> dict:
    """Val loss since epoch 1, graded on least-squares trends (t statistics), never on the sign of an
    endpoint difference: the total, and what drove it (forecast terms vs calibration terms)."""
    val = ctx.val
    i0, i1 = fin[0], fin[-1]
    rel = (val[i1] - val[i0]) / abs(val[i0]) if val[i0] else 0.0
    head = f"{val[i0]:.3f} → {val[i1]:.3f} ({100 * rel:+.1f}%)"
    detail = ("least-squares trend of each part over all epochs: falling or rising only beyond the 95% "
              "Student-t level. Good only when the forecast terms fall too, not just the calibration terms")
    label = "val loss since epoch 1"
    if last_bad:
        return dict(label=label, value=head, status="critical", note="the last epoch's loss is not finite",
                    detail=detail)
    fit = _ols(val)
    if fit is None:
        return dict(label=label, value=head, status="info", note="too few epochs for a trend (4 needed)",
                    detail=detail)
    total = _trend(fit)
    notes = [f"total {fit['change']:+.3f} over the run (t = {fit['t']:+.1f})"]
    forecast = None
    comp = loss_contributions(ctx.df, ctx.config, meta=ctx.meta)
    if comp is not None:
        for name, cols in (("forecast terms (point, trend, direction)", list(_FORECAST_TERMS)),
                           ("calibration / variance", list(_CALIBRATION_TERMS)),
                           ("other", ["other", "regulariser"])):
            gf = _ols(comp[cols].sum(axis=1).to_numpy())
            if gf is None:
                continue
            notes.append(f"{name} {gf['change']:+.3f} (t = {gf['t']:+.1f})")
            if cols == list(_FORECAST_TERMS):
                forecast = _trend(gf)
    if total == "rising":
        status, verdict = "critical", "rising"
    elif total == "no clear trend":
        status, verdict = "warning", "no clear trend"
    elif forecast in (None, "falling"):
        status, verdict = "good", "falling" + (", forecast terms too" if forecast else "")
    elif forecast == "rising":
        status, verdict = "warning", "falling, but the forecast terms rose"
    else:
        status, verdict = "warning", "falling on the calibration terms only"
    return dict(label=label, value=f"{head}: {verdict}", status=status, note=" · ".join(notes), detail=detail)


def _head_tile(ctx: _Ctx, s: int, key: str, up_key: str, h: str) -> Optional[dict]:
    df = ctx.df
    v = _col(df, f"val_{key}_mcc_{h}")
    if v is None:
        return None
    t = _col(df, f"train_{key}_mcc_{h}")
    up, true = _col(df, f"val_{up_key}pred_up_rate_{h}"), _col(df, f"val_{up_key}true_up_rate_{h}")
    bal = _col(df, f"val_{key}_bal_acc_{h}")
    m = _at(v, s)
    band = _chance(ctx, h)
    u = _at(up, s)
    collapsed = np.isfinite(u) and (u < 0.03 or u > 0.97)
    if collapsed:
        status, verdict = "critical", f"collapsed: predicts up {100 * u:.0f}%"
    elif not np.isfinite(m):
        status, verdict = "critical", "not finite"
    elif band is None:
        n = _n_eff(ctx, h)
        status = "info"
        verdict = ("no chance band (validation size unknown)" if n is None
                   else f"too few validation samples for a chance band (n_eff {n})")
    elif m > band:
        status, verdict = "good", f"above chance (±{band:.2f})"
    elif m < -band:
        status, verdict = "critical", f"below chance (±{band:.2f})"
    else:
        status, verdict = "info", f"within chance (±{band:.2f})"
    u_last = _at(up, ctx.n - 1)
    if s != ctx.n - 1 and np.isfinite(u_last) and (u_last < 0.03 or u_last > 0.97):   # collapsing now (live)
        verdict += f"; last epoch collapsed (predicts up {100 * u_last:.0f}%)"
        status = _worse(status, "warning")
    notes = [f"epoch {ctx.x[s]}"]
    if s != ctx.n - 1:
        notes.append(f"last {_fmt(v[-1], '+.3f')}")
    if t is not None:
        notes.append(f"train {_fmt(_at(t, s), '+.3f')}")
    if bal is not None:
        notes.append(f"bal acc {_fmt(_at(bal, s), '.3f')}")
    if np.isfinite(u):
        notes.append(f"predicts up {100 * u:.0f}%" + (f" (true {100 * _at(true, s):.0f}%)" if true is not None else ""))
    # Overfitting: the train − val MCC gap widened, and not by noise. Both are required: a rising
    # least-squares trend of the gap over all epochs beyond its one-sided 99.5% t level (the check is
    # repeated every epoch on six series, so it is stricter than the 95% verdicts), and a gap in the last
    # epochs that exceeds the one in the first epochs by more than the val MCC chance band. The windows
    # never overlap.
    if t is not None and band is not None and ctx.n >= 6:
        g = t - v
        w = max(2, min(5, ctx.n // 3))
        early, late = float(np.nanmean(g[:w])), float(np.nanmean(g[-w:]))
        gfit = _ols(g)
        if gfit is not None and gfit["t"] >= _t_crit(gfit["k"] - 2, 0.995) and late - early > band:
            verdict += (f"; overfitting: train − val gap {early:+.2f} → {late:+.2f} (first / last {w} epochs, "
                        f"t = {gfit['t']:+.1f})")
            status = _worse(status, "warning")
    return dict(value=f"MCC {_fmt(m, '+.3f')} · {verdict}", status=status, note=" · ".join(notes),
                detail="val MCC at the served epoch vs its 95% chance band. Overfitting: the train − val gap "
                       "rose over the run (a significant trend) by more than that band")


def _skill_tile(ctx: _Ctx, s: int) -> Optional[dict]:
    parts, skills, ceilings, bce = [], [], [], []
    for h in T.HORIZONS:
        br, r = _col(ctx.df, f"val_dir_brier_{h}"), _col(ctx.df, f"val_true_up_rate_{h}")
        if br is None or r is None:
            continue
        rr = _at(r, s)
        ref = rr * (1 - rr)
        sk = 1 - _at(br, s) / ref if ref > 0 else float("nan")
        c = _skill_ceiling(ctx, h)
        skills.append(sk)
        ceilings.append(c)
        mark = "" if c is None or not np.isfinite(sk) else (" beyond chance" if sk > c else "")
        parts.append(f"{h} {100 * sk:+.1f}%{mark}")
        d = _col(ctx.df, f"val_dir_loss_{h}")
        if d is not None:
            bce.append(f"{_at(d, s):.3f}")
    if not skills:
        return None
    pairs = [(sk, c) for sk, c in zip(skills, ceilings) if np.isfinite(sk)]
    known = pairs and all(c is not None for _, c in pairs)
    if pairs and all(sk <= 0 for sk, _ in pairs):
        status = "warning"
    elif known and all(sk > c for sk, c in pairs):
        status = "good"
    else:
        status = "info"
    notes = []
    if known:
        notes.append("chance ceiling " + " / ".join(f"+{100 * c:.1f}%" for _, c in pairs))
    elif ctx.n_val:
        notes.append("too few validation samples for a chance ceiling")
    else:
        notes.append("no chance ceiling (validation size unknown)")
    if bce:
        notes.append(f"direction BCE {' / '.join(bce)} vs ln 2 = {LN2:.3f} (a constant 0.5)")
    return dict(label="direction head vs base rate", value="Brier skill " + " · ".join(parts), status=status,
                note=" · ".join(notes),
                detail="1 − Brier / p(1 − p), p = the validation block's up-rate: > 0 beats always forecasting that "
                       "rate. A forecast with no information stays below the chance ceiling z² / n_eff (97.5%); "
                       "good only when every horizon is beyond it, warning when none is above 0")


_ICON = {"good": "&#10003;", "warning": "!", "critical": "&#10007;", "info": "i"}
_STATUS_COLOR = {"good": T.GOOD, "warning": T.WARNING, "critical": T.CRITICAL, "info": T.MUTED}


def training_health_html(history, config=None, *, title: str = "Training health", weights_epoch: Optional[int] = None,
                         n_val: Optional[int] = None, meta: Optional[dict] = None, run_dir=None,
                         table: bool = True) -> str:
    """The checks from :func:`training_health` as a grid of tiles (icon + label + colour, then the
    numbers behind the verdict and what the check means), followed by :func:`epoch_table_html`."""
    ctx = _resolve(history, config, weights_epoch, n_val, meta, run_dir)
    checks = _health(ctx)
    tiles = []
    small = f"color:{T.MUTED};font-size:11px;margin-top:3px;line-height:1.35"
    for name, c in checks.items():
        color = _STATUS_COLOR[c["status"]]
        note = f"<div style='{small};color:{T.INK_2}'>{html.escape(c['note'])}</div>" if c.get("note") else ""
        tiles.append(
            f"<div style='background:{T.SURFACE};border:1px solid {T.AXIS};"
            f"border-left:3px solid {color};border-radius:6px;padding:8px 10px;min-width:0'>"
            f"<div style='color:{T.MUTED};font-size:11px;letter-spacing:.04em;text-transform:uppercase'>"
            f"{html.escape(c.get('label') or name)}</div>"
            f"<div style='color:{T.INK};font-size:14px;margin-top:2px;font-variant-numeric:tabular-nums'>"
            f"<span style='color:{color};font-weight:700;margin-right:6px'>{_ICON[c['status']]}</span>"
            f"{html.escape(c['value'])}</div>{note}"
            f"<div style='{small}'>{html.escape(c['detail'])}</div></div>")
    sub = _context_line(ctx)
    out = (f"<div style='font-family:{T.FONT};background:{T.PAPER};padding:12px;border-radius:8px'>"
           f"<div style='color:{T.INK};font-weight:600'>{html.escape(title)}</div>"
           f"<div style='color:{T.MUTED};font-size:12px;margin:2px 0 8px'>{html.escape(sub)}</div>"
           f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px'>"
           + "".join(tiles) + "</div></div>")
    if table:
        out += _epoch_table(ctx)
    return out


def _context_parts(ctx: _Ctx, served: bool = True) -> List[str]:
    parts = []
    if ctx.n_val:
        effs = [f"{h} {_n_eff(ctx, h)}" for h in T.HORIZONS if _n_eff(ctx, h)]
        parts.append(f"validation block {ctx.n_val:,} samples" + (f" (n_eff {' / '.join(effs)})" if effs else ""))
        if _too_few(ctx):
            few = [h for h in T.HORIZONS if _n_eff(ctx, h) is not None and _enough(ctx, h) is None]
            parts.append(f"fewer than {_MIN_N_EFF} effective samples for {' / '.join(few)}: no chance range there")
    else:
        parts.append("validation size unknown: no chance ranges")
    if served and ctx.served_epoch is not None:
        parts.append(f"served epoch {ctx.served_epoch} ({ctx.served_note})")
    return parts


def _context_line(ctx: _Ctx) -> str:
    return " · ".join(_context_parts(ctx))


# ------------------------------------------------------------------ epoch table
def epoch_table_html(history, config=None, *, weights_epoch: Optional[int] = None, n_val: Optional[int] = None,
                     meta: Optional[dict] = None, run_dir=None) -> str:
    """Exact numbers at the served epoch (and the last epoch when it differs): every loss term as it
    enters the total with its weight and its share of the validation loss, the physics terms (exact
    zeros shown as 0), and every direction metric per horizon for validation (training in brackets)."""
    return _epoch_table(_resolve(history, config, weights_epoch, n_val, meta, run_dir))


def _epoch_table(ctx: _Ctx) -> str:
    if ctx.n == 0:
        return ""
    s = ctx.served if ctx.served is not None else ctx.n - 1
    last = ctx.n - 1
    df, config = ctx.df, ctx.config
    cv = loss_contributions(df, config, prefix="val_", meta=ctx.meta)
    ct = loss_contributions(df, config, prefix="", meta=ctx.meta)
    wl = _weight_labels(ctx)
    # colours and sizes on every cell: tables do not inherit them in quirks-mode pages
    th = (f"text-align:right;padding:3px 8px;color:{T.MUTED};font-weight:500;font-size:12px;"
          f"border-bottom:1px solid {T.AXIS}")
    td = f"text-align:right;padding:2px 8px;font-variant-numeric:tabular-nums;color:{T.INK};font-size:12px"
    tl = f"text-align:left;padding:2px 8px;color:{T.INK_2};font-size:12px"
    tbl = f"border-collapse:collapse;min-width:640px;font-family:{T.FONT};color:{T.INK};font-size:12px"
    box = (f"font-family:{T.FONT};background:{T.PAPER};padding:4px 12px 12px;border-radius:8px;color:{T.INK};"
           "font-size:12px;overflow-x:auto")

    def num(v, spec=".4g"):
        return _fmt(v, spec)

    parts = [f"<div style='{box}'>"]
    # --- loss terms
    head = [f"epoch {ctx.x[s]} (served): val", "share of val", "train"]
    if last != s:
        head.append(f"epoch {ctx.x[last]} (last): val")
    rows = []
    if cv is not None:
        tot = cv["total"].iloc[s]
        names = {k: lab for k, lab, _, _ in _CONTRIB}
        names["regulariser"] = "regulariser"
        for key in ("point", "trend", "direction", "nll", "crps", "soft_ece", "volatility", "physics", "regulariser",
                    "other"):
            vv = cv[key].iloc[s]
            cells = [num(vv), f"{100 * vv / tot:.1f}%" if tot else "n/a",
                     num(ct[key].iloc[s]) if ct is not None and key != "other" else "–"]
            if last != s:
                cells.append(num(cv[key].iloc[last]))
            rows.append((names[key], wl.get(key, ""), cells))
            if key == "physics":
                for pk, plab in PHYSICS_TERMS:
                    pv, pt = _col(df, f"val_{pk}"), _col(df, pk)
                    if pv is None and pt is None:
                        continue
                    cells = [num(_at(pv, s)), "", num(_at(pt, s))] + ([num(_at(pv, last))] if last != s else [])
                    rows.append((f"&nbsp;&nbsp;&nbsp;{plab}", "", cells))
        cells = [num(tot), "100%", num(_at(ctx.loss, s))] + ([num(cv["total"].iloc[last])] if last != s else [])
        rows.append(("<b>total loss</b>", "", cells))
        caption = ("Loss terms as they enter the total (losses.functions.custom_loss). Validation is exact; "
                   "training terms are sampled every TRAIN_METRICS_EVERY steps and the train total is a "
                   "dropout-on running mean, so they do not add up exactly. 'other' = total − known terms "
                   "(the coherence penalty is not logged).")
    else:
        for key, lab in LOSS_COMPONENTS + PHYSICS_TERMS:
            pv, pt = _col(df, f"val_{key}"), _col(df, key)
            if pv is None and pt is None:
                continue
            cells = [num(_at(pv, s)), "", num(_at(pt, s))] + ([num(_at(pv, last))] if last != s else [])
            rows.append((lab, "as logged", cells))
        cells = [num(_at(ctx.val, s)), "", num(_at(ctx.loss, s))] + ([num(_at(ctx.val, last))] if last != s else [])
        rows.append(("<b>total loss</b>", "", cells))
        caption = "Loss terms as logged (the loss weights were not logged, so their shares are not computed)."
    parts.append(f"<div style='color:{T.MUTED};margin:8px 0 4px'>{html.escape(caption)}</div>")
    parts.append(f"<table style='{tbl}'><tr>"
                 f"<th style='{th};text-align:left'>term</th><th style='{th};text-align:left'>weight in the total</th>"
                 + "".join(f"<th style='{th}'>{html.escape(x)}</th>" for x in head) + "</tr>")
    for lab, w, cells in rows:
        parts.append(f"<tr><td style='{tl}'>{lab}</td><td style='{tl};color:{T.MUTED}'>{html.escape(w)}</td>"
                     + "".join(f"<td style='{td}'>{c}</td>" for c in cells) + "</tr>")
    parts.append("</table>")
    # --- direction metrics per horizon
    metric_rows = []

    def pair(key, spec=".3f", h=None):
        v, t = _col(df, f"val_{key}_{h}"), _col(df, f"train_{key}_{h}")
        if v is None:
            return None
        txt = num(_at(v, s), spec)
        return txt + (f" <span style='color:{T.MUTED}'>({num(_at(t, s), spec)})</span>" if t is not None else "")

    def row(label, fn):
        cells = [fn(h) for h in T.HORIZONS]
        if any(c is not None for c in cells):
            metric_rows.append((label, [c or "–" for c in cells]))

    def mcc(key):
        def f(h):
            p = pair(key, "+.3f", h)
            band = _chance(ctx, h)
            return None if p is None else p + (f" <span style='color:{T.MUTED}'>±{band:.2f}</span>" if band else "")
        return f

    def muted(text):
        return f" <span style='color:{T.MUTED}'>{text}</span>"

    def with_ref(key, ref_fn, spec=".3f", noise=None):
        def f(h):
            p = pair(key, spec, h)
            r = _col(df, f"val_true_up_rate_{h}")
            if p is None or r is None:
                return p
            rr = _at(r, s)
            nz = _ref_noise(ctx, h, noise, rr) if noise else None
            return p + muted(f"ref {ref_fn(rr):{spec}}" + (f" ±{nz:.3f}" if nz else ""))
        return f

    def bal_acc(h):
        p = pair("dir_bal_acc", ".3f", h)
        band = _chance(ctx, h)
        return None if p is None else p + (muted(f"±{band / 2:.3f}") if band else "")

    def recall(key, kind):
        def f(h):
            p = pair(key, ".3f", h)
            nz = _ref_noise(ctx, h, kind, _val_rate(ctx, h))
            return None if p is None else p + (muted(f"ref 0.5 ±{nz:.3f}") if nz else "")
        return f

    def below(key, crit_fn, train_key=None):
        def f(h):
            v = _col(df, f"val_{key}_{h}")
            t = _col(df, train_key.format(h=h) if train_key else f"train_{key}_{h}")
            if v is None:
                return None
            c = crit_fn(h)
            return (num(_at(v, s), ".3f") + (muted(f"({num(_at(t, s), '.3f')})") if t is not None else "")
                    + (muted(f"calibrated &lt; {c:.3f}") if c else ""))
        return f

    def brier_skill(h):
        out = []
        for pre in ("val_", "train_"):
            br, r = _col(df, f"{pre}dir_brier_{h}"), _col(df, f"{pre}true_up_rate_{h}")
            if br is None or r is None:
                out.append(None)
                continue
            rr = _at(r, s)
            out.append(1 - _at(br, s) / (rr * (1 - rr)) if 0 < rr < 1 else float("nan"))
        if out[0] is None:
            return None
        c = _skill_ceiling(ctx, h)
        return (f"{100 * out[0]:+.1f}%" + (muted(f"({100 * out[1]:+.1f}%)") if out[1] is not None else "")
                + (muted(f"chance &lt; +{100 * c:.1f}%") if c else ""))

    def bce(h):
        v, t = _col(df, f"val_dir_loss_{h}"), _col(df, f"dir_loss_{h}")
        if v is None:
            return None
        return (num(_at(v, s), ".3f") + (f" <span style='color:{T.MUTED}'>({num(_at(t, s), '.3f')})</span>"
                                         if t is not None else "") + f" <span style='color:{T.MUTED}'>ln 2 {LN2:.3f}</span>")

    def rate(h):
        up, tr = _col(df, f"val_pred_up_rate_{h}"), _col(df, f"val_true_up_rate_{h}")
        if up is None:
            return None
        return f"{100 * _at(up, s):.1f}%" + (f" <span style='color:{T.MUTED}'>true {100 * _at(tr, s):.1f}%</span>"
                                             if tr is not None else "")

    def mean_p(h):
        m, tr = _col(df, f"val_mean_dir_prob_{h}"), _col(df, f"val_true_up_rate_{h}")
        if m is None:
            return None
        if tr is None:
            return num(_at(m, s), ".3f")
        nz = _ref_noise(ctx, h, "bias", _at(tr, s))
        return num(_at(m, s), ".3f") + muted(f"true {_at(tr, s):.3f}" + (f" ±{nz:.3f}" if nz else ""))

    row("direction head MCC (± chance)", mcc("dir_mcc"))
    row("price head MCC, Gaussian readout (± chance)", mcc("gauss_dir_mcc"))
    row("balanced accuracy (± chance)", bal_acc)
    row("accuracy (ref: always the majority class ± noise)", with_ref("dir_acc", lambda p: max(p, 1 - p), noise="acc"))
    row("F1, up class (ref: always up ± noise)", with_ref("dir_f1", lambda p: 2 * p / (1 + p), noise="f1"))
    row("sensitivity: up moves called up", recall("dir_sensitivity", "sens"))
    row("specificity: down moves called down", recall("dir_specificity", "spec"))
    row("Brier (ref: base rate p(1 − p))", with_ref("dir_brier", lambda p: p * (1 - p), ".4f"))
    row("Brier skill vs base rate (chance ceiling)", brier_skill)
    row("direction BCE (ref: constant 0.5)", bce)
    row("ECE, raw head (a calibrated head stays below)",
        below("dir_ece", lambda h: _ece_crit(ctx, h, _val_rate(ctx, h))))
    row("mean P(up) vs true up-rate (± noise)", mean_p)
    row("share of up calls vs true up-rate", rate)
    row("PIT-KS, raw price head (a calibrated head stays below)",
        below("pit_ks", lambda h: _ks_crit(ctx, h), train_key="pit_ks_{h}"))
    if metric_rows:
        if ctx.n_val:
            ranges = (f"Chance ranges: 95% on n_eff = {ctx.n_val:,} validation samples / horizon bars"
                      + (f" (none below {_MIN_N_EFF} effective samples)" if _too_few(ctx) else "")
                      + "; the direction metrics skip moves inside the deadband (not logged), so their true n_eff "
                        "is smaller and their ranges somewhat wider than shown.")
        else:
            ranges = "Validation size unknown: no chance ranges."
        cap = f"Direction metrics at epoch {ctx.x[s]} (served): validation, training in brackets. " + ranges
        parts.append(f"<div style='color:{T.MUTED};margin:12px 0 4px'>{html.escape(cap)}</div>")
        parts.append(f"<table style='{tbl}'><tr>"
                     f"<th style='{th};text-align:left'>metric</th>"
                     + "".join(f"<th style='{th};color:{T.HORIZON_COLORS[h]}'>{html.escape(T.horizon_label(h, config))}</th>"
                               for h in T.HORIZONS) + "</tr>")
        for lab, cells in metric_rows:
            parts.append(f"<tr><td style='{tl}'>{html.escape(lab)}</td>"
                         + "".join(f"<td style='{td}'>{c}</td>" for c in cells) + "</tr>")
        parts.append("</table>")
    parts.append("</div>")
    return "".join(parts)


# ------------------------------------------------------------------ figure helpers
def _line(fig, x, y, *, row, col, name, color, dash=T.VAL_DASH, group=None, show=True, width=2, hover=None,
          legend="legend", shape=None, symbol="circle", msize=5):
    """A per-epoch series. Every finite value is drawn: isolated points (NaN neighbours) always keep
    a marker, also in long runs where the other markers are dropped."""
    import plotly.graph_objects as go

    if y is None:
        return
    y = np.asarray(y, dtype=float)
    fin = np.isfinite(y)
    iso = fin & ~np.r_[False, fin[:-1]] & ~np.r_[fin[1:], False]
    size = msize if len(x) <= 40 else np.where(iso, msize, 0).tolist()
    fig.add_trace(go.Scatter(x=x, y=y.astype(np.float32), mode="lines+markers", name=name,
                             legendgroup=group or name, showlegend=show, legend=legend,
                             line=dict(color=color, dash=dash, width=width, shape=shape or "linear"),
                             marker=dict(size=size, color=color, symbol=symbol),
                             hovertemplate=hover or f"{name}: %{{y:.4g}}<extra></extra>"),
                  row=row, col=col)


def _log_ticks(fig, row, col, values):
    """Label a log axis at the values actually present (e.g. the learning-rate steps)."""
    v = np.unique(np.round(values[np.isfinite(values) & (values > 0)], 12))
    if len(v) and len(v) <= 6:
        fig.update_yaxes(tickvals=v, ticktext=[f"{a:.1e}" for a in v], row=row, col=col)


def _legend_width(name: str) -> float:
    """Generous width of one horizontal legend entry at font size 10 (symbol + text + gap), in px."""
    return 44.0 + 5.2 * len(re.sub("<[^>]+>", "", name or ""))


def _grid_layout(fig, n_rows: int, n_cols: int, panel_px: float, legends: Dict[str, Tuple[int, int]],
                 titles: Dict[Tuple[int, int], str]) -> Tuple[float, Dict[int, int]]:
    """Lay the panels out so every panel legend and heading has room, whatever it holds.

    Plotly wraps a horizontal legend only at the full plot width, never at its column, so a long panel
    legend used to push the right margin (every panel narrower) or, at notebook widths, collide with
    the next panel. Each panel legend is split into rows that fit its column in a ``_MIN_FIG_WIDTH``
    figure, each row its own legend (ids legend<N>) stacked above the panel; the subplot heading is
    kept above the rows, and the gap above every subplot row is sized for its tallest legend, the
    headings and the x tick labels of the row above. Returns (plot area height px, {row: legend rows}).
    """
    used = [int(k[6:]) for k in fig.layout.to_plotly_json() if re.fullmatch(r"legend\d+", k)]
    next_id = max([1] + used + [int(lid[6:]) for lid in legends if lid[6:].isdigit()]) + 1
    avail = _MIN_FIG_WIDTH - _MARGIN["l"] - _MARGIN["r"]
    plan, rows_of = [], {}
    for lid, (r, c) in legends.items():
        xd = fig.get_subplot(r, c).xaxis.domain
        order, names = [], {}
        for t in fig.data:
            if t.legend == lid and t.showlegend is not False and t.legendgroup not in names:
                order.append(t.legendgroup)
                names[t.legendgroup] = t.name
        chunks, cur, w = [], [], 0.0
        for g in order:
            gw = _legend_width(names[g])
            if cur and w + gw > (xd[1] - xd[0]) * avail:
                chunks.append(cur)
                cur, w = [], 0.0
            cur.append(g)
            w += gw
        if cur:
            chunks.append(cur)
        rows_of[r] = max(rows_of.get(r, 0), len(chunks))
        plan.append((lid, r, c, xd, chunks))
    # y domains: gap above row r = x tick labels of row r - 1 + heading + legend rows
    gaps = {r: 24 + _HEAD_PX + rows_of.get(r, 0) * _ROW_PX + 12 for r in range(2, n_rows + 1)}
    plot_px = n_rows * panel_px + sum(gaps.values())
    old_top, top = {}, plot_px
    for r in range(1, n_rows + 1):
        top -= gaps.get(r, 0)
        for c in range(1, n_cols + 1):
            sp = fig.get_subplot(r, c)
            if sp is None:
                continue
            old_top[(r, c)] = sp.yaxis.domain[1]
            sp.yaxis.domain = [max(0.0, (top - panel_px) / plot_px), top / plot_px]
        top -= panel_px
    for lid, r, c, xd, chunks in plan:
        y_top = fig.get_subplot(r, c).yaxis.domain[1]
        n = len(chunks)
        for k, groups in enumerate(chunks):
            new = lid
            if k:
                new, next_id = f"legend{next_id}", next_id + 1
                for t in fig.data:
                    if t.legend == lid and t.legendgroup in groups:
                        t.legend = new
            fig.update_layout({new: dict(
                orientation="h", x=xd[0], xanchor="left", y=y_top + (3 + (n - 1 - k) * _ROW_PX) / plot_px,
                yanchor="bottom", font=dict(size=10, color=T.INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
                itemwidth=30, groupclick="togglegroup", tracegroupgap=0, title=dict(text=""))})
    for a in fig.layout.annotations or ():
        for (r, c), text in titles.items():
            if a.text == text and (r, c) in old_top and a.y is not None and abs(a.y - old_top[(r, c)]) < 0.02:
                a.y = fig.get_subplot(r, c).yaxis.domain[1]
                a.yshift = rows_of.get(r, 0) * _ROW_PX + 2
    return plot_px, rows_of


def _shade(fig, spans, row: int, col: int) -> int:
    """Nested translucent neutral bands, one per horizon (``spans``: (y0, y1) or None each): where all
    of them overlap is darkest. Returns how many were drawn."""
    k = 0
    for span in spans:
        if span is None:
            continue
        fig.add_hrect(y0=span[0], y1=span[1], fillcolor=T.NEUTRAL, opacity=0.09, line_width=0, layer="below",
                      row=row, col=col)
        k += 1
    return k


def _horizon_limits(fig, edges, row: int, col: int) -> None:
    """Each horizon's own limit(s) as a dashed line in its colour (``edges``: (horizon, y) pairs), so a
    horizon's line is read against its own band rather than against whichever shaded tier it falls in."""
    for h, y in edges:
        fig.add_hline(y=y, line=dict(color=T.HORIZON_COLORS[h], dash="dash", width=1), opacity=0.7, row=row, col=col)


def _chance_bands(fig, ctx: _Ctx, row: int, col: int, *, center=0.0, scale=1.0):
    """Nested 95% chance bands of an MCC (scale 1, capped at ±1) or a balanced accuracy (centre 0.5,
    scale 0.5, so within [0, 1]), one per horizon (n_eff = n / h bars); the widest is h2's. Both edges
    of each horizon's band are also drawn dashed in that horizon's colour, as on the one-sided panels:
    the neutral tiers alone do not say which band is whose."""
    spans, edges = [], []
    for h in T.HORIZONS:
        band = _chance(ctx, h)
        spans.append(None if band is None else (center - scale * band, center + scale * band))
        if band is not None:
            edges += [(h, center - scale * band), (h, center + scale * band)]
    k = _shade(fig, spans, row, col)
    _horizon_limits(fig, edges, row, col)
    return k


def _style_key(fig, entries, row=1, col=1):
    """Figure-wide key entries (style or reference meaning, not data)."""
    import plotly.graph_objects as go

    for name, kw in entries:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legendgroup=f"key-{name}", hoverinfo="skip",
                                 legend="legend", **kw), row, col)


# ------------------------------------------------------------------ the dashboard
def training_dashboard_figure(history, config=None, *, title: Optional[str] = None, height: Optional[int] = None,
                              weights_epoch: Optional[int] = None, n_val: Optional[int] = None,
                              meta: Optional[dict] = None, run_dir=None):
    """The per-epoch record in 12 panels (see :data:`PANELS`).

    ``weights_epoch`` (1-based served epoch), ``n_val`` (validation samples, for the chance bands),
    ``meta`` / ``run_dir`` (the bundle's meta.json): all optional and read from the run directory when
    ``history`` is a metrics.jsonl path. ``height`` overrides the computed height.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ctx = _resolve(history, config, weights_epoch, n_val, meta, run_dir)
    df, x, val = ctx.df, ctx.x, ctx.val
    first = T.legend_once()

    # physics: which terms are drawn, their weights, which are off (decided before the headings exist)
    phys = _physics_plan(ctx)
    titles = list(PANELS)
    titles[9] = phys["title"]
    lr, lr_shifted = _lr_in_effect(ctx)

    # panel heights; the gaps are sized later by _grid_layout (x ticks, heading, legend rows)
    n_rows, panel_px = 6, 230
    if height:
        panel_px = max(150, (height - 190 - _MARGIN["b"] - (n_rows - 1) * 80) / n_rows)
    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.04,
                        horizontal_spacing=0.08)

    # figure-wide key: line styles, the chance band, the served-epoch marker (the horizons join below)
    key = [("validation", dict(mode="lines", line=dict(color=T.INK_2, dash=T.VAL_DASH))),
           ("training", dict(mode="lines", line=dict(color=T.INK_2, dash=T.TRAIN_DASH)))]
    if any(_chance(ctx, h) for h in T.HORIZONS):
        key.append(("95% chance range (n_eff = val / h bars)",
                    dict(mode="markers", marker=dict(symbol="square", size=12, color=T.rgba(T.NEUTRAL, 0.3)))))
    _style_key(fig, key, 2, 1)

    # 1: total loss, best and served epochs
    _line(fig, x, ctx.loss, row=1, col=1, name="train", color=T.INK_2, dash=T.TRAIN_DASH, group="total",
          legend="legend2")
    _line(fig, x, val, row=1, col=1, name="validation", color=T.INK, group="total-val", legend="legend2")
    if val is not None:
        b = ctx.best
        fig.add_trace(go.Scatter(x=[x[b]], y=[val[b]], mode="markers", name=f"best val (epoch {x[b]})", legend="legend2",
                                 legendgroup="best", marker=dict(symbol="star", size=13, color=T.INK,
                                                                 line=dict(color=T.PAPER, width=1)),
                                 hovertemplate=f"best val loss {val[b]:.4f} at epoch {x[b]}<extra></extra>"), 1, 1)
        if ctx.served is not None and np.isfinite(val[ctx.served]):
            s = ctx.served
            fig.add_trace(go.Scatter(
                x=[x[s]], y=[val[s]], mode="markers", legend="legend2", legendgroup="served",
                name=f"served weights (epoch {x[s]})" + ("" if s == b else " ≠ best"),
                marker=dict(symbol="circle-open", size=21, color=T.WARNING if s != b else T.INK, line=dict(width=2)),
                hovertemplate=f"served weights: epoch {x[s]}, val loss {val[s]:.4f} ({ctx.served_note})<extra></extra>"),
                1, 1)

    # 2: composition of the validation loss
    comp = loss_contributions(df, config, meta=ctx.meta)
    if comp is not None:
        factor = _weight_factors(ctx)
        tot = comp["total"].to_numpy()
        for k, label, color, pattern in _CONTRIB:
            y = comp[k].to_numpy() + (comp["regulariser"].to_numpy() if k == "other" else 0.0)
            if k != "other" and not np.any(np.abs(y) > 0):
                continue
            share = np.where(tot != 0, 100 * y / tot, np.nan)
            name = label + (f" ×{factor[k]:.3g}" if k in factor else "")
            fill = dict(fillcolor=T.rgba(color, 0.78)) if not pattern else dict(
                fillcolor="rgba(0,0,0,0)" if k == "other" else T.rgba(color, 0.35),
                fillpattern=dict(shape=pattern, fgcolor=color, size=7, solidity=0.35))
            fig.add_trace(go.Scatter(
                x=x, y=np.maximum(y, 0).astype(np.float32), name=name, legend="legend3", legendgroup=f"c-{k}",
                stackgroup="val-loss", mode="lines", line=dict(color=color, width=0.6), **fill,
                customdata=np.c_[y, share].astype(np.float32),
                hovertemplate=f"{label}: %{{customdata[0]:.4g}} (%{{customdata[1]:.1f}}% of val loss)<extra></extra>"),
                1, 2)

    # 3-9: per-horizon metrics, validation solid, training dotted
    def horizons(row, col, val_key, train_key=None, *, transform=None):
        for h in T.HORIZONS:
            c = T.HORIZON_COLORS[h]
            v = transform(h, "val_") if transform else _col(df, val_key.format(h=h))
            _line(fig, x, v, row=row, col=col, name=T.horizon_label(h, config), color=c, group=h,
                  show=v is not None and first(h), hover=f"{h} val: %{{y:.4g}}<extra></extra>")
            if train_key or transform:
                t = transform(h, "train_") if transform else _col(df, train_key.format(h=h))
                _line(fig, x, t, row=row, col=col, name=f"{h} train", color=c, dash=T.TRAIN_DASH, group=h, show=False,
                      width=1.5, hover=f"{h} train: %{{y:.4g}}<extra></extra>")

    def skill(h, pre):
        br, r = _col(df, f"{pre}dir_brier_{h}"), _col(df, f"{pre}true_up_rate_{h}")
        if br is None or r is None:
            return None
        ref = r * (1 - r)
        return np.where(ref > 0, 1 - br / np.where(ref > 0, ref, 1), np.nan)

    def bias(h, pre):
        up, tr = _col(df, f"{pre}pred_up_rate_{h}"), _col(df, f"{pre}true_up_rate_{h}")
        return None if up is None or tr is None else up - tr

    horizons(2, 1, "val_dir_mcc_{h}", "train_dir_mcc_{h}")
    horizons(2, 2, "val_gauss_dir_mcc_{h}", "train_gauss_dir_mcc_{h}")
    horizons(3, 1, "val_dir_bal_acc_{h}", "train_dir_bal_acc_{h}")
    horizons(3, 2, "", transform=skill)
    horizons(4, 1, "val_dir_ece_{h}", "train_dir_ece_{h}")
    horizons(4, 2, "val_pit_ks_{h}", "pit_ks_{h}")
    horizons(5, 1, "", transform=bias)
    _chance_bands(fig, ctx, 2, 1)
    _chance_bands(fig, ctx, 2, 2)
    _chance_bands(fig, ctx, 3, 1, center=0.5, scale=0.5)
    # what chance produces: Brier skill of a forecast without information (below its ceiling), ECE and
    # PIT-KS of a calibrated head (below their 95% levels); the validation lines are read against them
    for (r, c), limit in (((3, 2), lambda h: _skill_ceiling(ctx, h)),
                          ((4, 1), lambda h: _ece_crit(ctx, h, _val_rate(ctx, h))),
                          ((4, 2), lambda h: _ks_crit(ctx, h))):
        lims = {h: limit(h) for h in T.HORIZONS}
        _shade(fig, [(0.0, v) if v else None for v in lims.values()], r, c)
        # one-sided: each horizon's limit, so its own line can be read against it
        _horizon_limits(fig, [(h, v) for h, v in lims.items() if v], r, c)
    for r, c in ((4, 1), (4, 2)):
        fig.update_yaxes(rangemode="tozero", row=r, col=c)
    for r, c, y0 in ((2, 1, 0.0), (2, 2, 0.0), (3, 1, 0.5), (3, 2, 0.0), (5, 1, 0.0)):
        fig.add_hline(y=y0, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=r, col=c)

    # 10: physics terms (as logged: already × λ)
    _draw_physics(fig, ctx, phys, 5, 2)

    # 11: learning rates, as steps
    lri, _ = _lr_in_effect(ctx, "lr_indicator")
    _line(fig, x, lr, row=6, col=1, name="network" + (" (logged one epoch late here; re-aligned)" if lr_shifted else ""),
          color=T.OTHER_SERIES[3], group="lr", legend="legend5", shape="hv")
    _line(fig, x, lri, row=6, col=1, name="indicator periods", color=T.OTHER_SERIES[0], group="lri",
          legend="legend5", shape="hv")
    fig.update_yaxes(type="log", row=6, col=1)
    present = [a for a in (lr, lri) if a is not None]
    if present:
        _log_ticks(fig, 6, 1, np.concatenate(present))

    # 12: gradient norm with the clip level; non-finite steps flagged
    gn = _col(df, "grad_global_norm")
    _line(fig, x, gn, row=6, col=2, name="grad norm (epoch mean)", color=T.INK_2, group="gn", show=False)
    clip = float(getattr(config, "GRAD_CLIP_NORM", 0) or 0) if config is not None else 0.0
    if gn is not None and clip:
        fig.add_hline(y=clip, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=6, col=2,
                      annotation_text=f"clip {clip:g} (per optimizer group)", annotation_position="top left",
                      annotation_font=dict(size=10, color=T.MUTED))
    nf = _col(df, "nonfinite_grad_steps")
    if nf is not None and gn is not None and np.nansum(nf) > 0:
        bad = np.where(nf > 0)[0]
        fig.add_trace(go.Scatter(x=[x[i] for i in bad], y=gn[bad], mode="markers", name="non-finite steps",
                                 showlegend=False, marker=dict(symbol="x", size=11, color=T.CRITICAL),
                                 hovertext=[f"{int(nf[i])} non-finite steps (skipped)" for i in bad]), 6, 2)

    # the served epoch, as a thin vertical line through every panel
    if ctx.served is not None:
        fig.add_vline(x=x[ctx.served], line=dict(color=T.INK_2, width=1), opacity=0.35, row="all", col="all")

    for i, t in enumerate(_Y_TITLES):
        fig.update_yaxes(title_text=t, row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(showticklabels=True, tickformat="d")
    for c in (1, 2):
        fig.update_xaxes(title_text="epoch", row=n_rows, col=c)
    head_titles = {(i // 2 + 1, i % 2 + 1): t for i, t in enumerate(titles)}
    plot_px, rows_of = _grid_layout(fig, n_rows, 2, panel_px, {"legend2": (1, 1), "legend3": (1, 2), "legend4": (5, 2),
                                                               "legend5": (6, 1)}, head_titles)
    T.note_on_empty(fig)
    sub = _title_lines(ctx)
    T.apply(fig, title=title or "Training dashboard", subtitle="<br>".join(sub))
    top_block = rows_of.get(1, 0) * _ROW_PX + _HEAD_PX + 8       # row-1 headings and legends
    margin_t = _title_px(len(sub)) + 26 + top_block              # title + subtitle, figure key, row-1 headings
    fig.update_layout(hovermode="x unified", height=int(plot_px + margin_t + _MARGIN["b"]),
                      margin=dict(t=margin_t, l=_MARGIN["l"], r=_MARGIN["r"], b=_MARGIN["b"]),
                      legend=dict(y=1 + (top_block + 4) / plot_px, yanchor="bottom", x=0, xanchor="left",
                                  orientation="h", font=dict(size=11)))
    return fig


def _physics_plan(ctx: _Ctx) -> dict:
    """Which physics terms to draw (and how), with the panel heading built from their weights."""
    df, cfg = ctx.df, ctx.config
    terms, off = [], []
    for key, label in PHYSICS_TERMS:
        v, t = _col(df, f"val_{key}"), _col(df, key)
        v_any = v is not None and np.any(np.nan_to_num(v) > 0)
        t_any = t is not None and np.any(np.nan_to_num(t) > 0)
        if not (v_any or t_any):
            if key == "vac_loss":
                lam = float(getattr(cfg, "LAMBDA_VAC", float("nan"))) if cfg is not None else float("nan")
            else:
                lam = _at(_weight(df, _PHYSICS_LAMBDA[key], cfg), ctx.n - 1)
            if v is not None or t is not None:
                off.append(f"{label}: off (λ = 0)" if lam == 0 else f"{label}: 0 in every epoch")
            continue
        eval_zero = key == "vac_overflow_loss" and not v_any   # zero at evaluation by construction
        terms.append(dict(key=key, label=label, v=None if eval_zero else v, t=t, eval_zero=eval_zero))
    lams = [_at(_weight(df, k, cfg), ctx.n - 1) for k in _PHYSICS_LAMBDA.values() if _weight(df, k, cfg) is not None]
    lam_txt = ""
    if lams:
        lam_txt = f"λ = {lams[0]:.3g} each, " if np.allclose(lams, lams[0]) else "λ inside, "
    return dict(terms=terms, off=off, title=f"Physics terms × λ ({lam_txt}log; ▼ ▽ = exactly 0)")


def _tick(v: float) -> str:
    return f"{v:g}" if 1e-3 <= v < 1e4 else f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _draw_physics(fig, ctx: _Ctx, plan: dict, row: int, col: int):
    import plotly.graph_objects as go

    x = ctx.x
    pos = np.concatenate([a[np.isfinite(a) & (a > 0)] for tm in plan["terms"] for a in (tm["v"], tm["t"])
                          if a is not None] or [np.array([])])
    if not len(pos):
        return
    lo, hi = math.log10(float(pos.min())), math.log10(float(pos.max()))
    # exact zeros sit on two labelled rows below the data: validation (filled ▼), training (open ▽)
    rows = {False: lo - 0.45, True: lo - 0.9}
    used = set()
    for tm in plan["terms"]:
        color = _PHYSICS_COLORS[tm["key"]]
        group = f"p-{tm['key']}"
        drawn_v = tm["v"] is not None
        n_zero = {k: int(np.sum(a == 0)) for k, a in ((False, tm["v"]), (True, tm["t"])) if a is not None}
        counts = [f"{n_zero[k]}/{len(x)} {'train' if k else 'val'}" for k in (False, True) if n_zero.get(k)]
        zero_txt = f" (= 0 in {', '.join(counts)} epochs)" if counts else ""
        for arr, is_train in ((tm["v"], False), (tm["t"], True)):
            if arr is None:
                continue
            zeros = np.flatnonzero(arr == 0)
            if is_train:
                name = (f"{tm['label']} (train; 0 at eval by design){zero_txt}" if tm["eval_zero"]
                        else f"{tm['label']} train")
            else:
                name = tm["label"] + zero_txt
            _line(fig, x, T.positive(arr), row=row, col=col, name=name,
                  color=color, dash=T.TRAIN_DASH if is_train else T.VAL_DASH, group=group,
                  show=(not is_train) or not drawn_v, width=1.5 if is_train else 2, legend="legend4",
                  hover=f"{tm['label']}{' train' if is_train else ''}: %{{y:.3g}}<extra></extra>")
            if len(zeros):
                used.add(is_train)
                fig.add_trace(go.Scatter(
                    x=[x[i] for i in zeros], y=np.full(len(zeros), 10 ** rows[is_train], dtype=np.float32),
                    mode="markers", legend="legend4", legendgroup=group, showlegend=False,
                    marker=dict(symbol="triangle-down-open" if is_train else "triangle-down", size=7, color=color),
                    hovertemplate=f"{tm['label']} {'train' if is_train else 'val'}: exactly 0<extra></extra>"), row, col)
    bottom = min(rows[k] for k in used) if used else lo
    decades = [10.0 ** k for k in range(math.ceil(bottom), math.floor(hi + 0.3) + 1)
               if k >= lo - 0.25 and all(abs(k - rows[u]) > 0.38 for u in used)]
    if len(decades) < 2:
        decades = sorted({float(f"{10 ** lo:.1g}"), float(f"{10 ** hi:.1g}")})
    vals = decades + [10 ** rows[k] for k in sorted(used)]
    text = [_tick(v) for v in decades] + [f"= 0 {'train' if k else 'val'}" for k in sorted(used)]
    fig.update_yaxes(type="log", range=[bottom - 0.3, hi + 0.3], tickvals=vals, ticktext=text, row=row, col=col)
    for text in plan["off"]:   # a key entry, so a term that is off is visibly off rather than missing
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=text, legend="legend4",
                                 legendgroup=f"off-{text}", hoverinfo="skip",
                                 marker=dict(symbol="x-thin-open", size=8, color=T.MUTED, line=dict(width=1.5))),
                      row, col)


def _title_px(n_lines: int) -> int:
    """Height of the figure title and ``n_lines`` subtitle lines (T.apply: pad 14, title 15 px)."""
    return 40 + _SUB_LINE_PX * n_lines


def _title_lines(ctx: _Ctx) -> List[str]:
    """The dashboard's subtitle, wrapped to fit a 900 px cell: the run's state, then how to read it."""
    checks = _health(ctx)
    parts = [f"epoch {checks['epochs']['value']}"]
    if "best" in checks:
        parts.append(f"best val {checks['best']['value']}")
    if "served" in checks:
        parts.append(f"served weights: {checks['served']['value']}")
    if ctx.val is not None:
        parts.append(f"last val {_fmt(ctx.val[-1], '.4f')}")
    for k, lab in (("patience", "patience "), ("learning rate", "lr "), ("time", "")):
        if k in checks:
            parts.append(f"{lab}{checks[k]['value']}")
    how = ["solid = validation, dotted = training", "train metrics sampled every TRAIN_METRICS_EVERY steps",
           "ECE, PIT-KS: raw heads (before temperature scaling / β shrink)"]
    if any(_chance(ctx, h) for h in T.HORIZONS):
        # two parts, so _wrap breaks between the phrases rather than inside 'dashed = each horizon's limit'
        how += ["shaded = 95% of what chance gives on the validation block with no skill (ECE, PIT-KS: "
                "a calibrated head)", "dashed = each horizon's limit, in its colour"]
    return _wrap(parts) + _wrap(_context_parts(ctx, served=False) + how)


# ------------------------------------------------------------------ companion figures
def _key_rows(names: Sequence[str], fig_px: int = 900) -> int:
    """Rows a horizontal figure key needs in a ``fig_px`` wide cell (plotly wraps it at the plot width)."""
    avail = fig_px - _MARGIN["l"] - _MARGIN["r"] - 20
    rows, w = 1, 0.0
    for n in names:
        nw = _legend_width(n)
        if w and w + nw > avail:
            rows, w = rows + 1, 0.0
        w += nw
    return rows


def _key_top(fig, plot_px: float, n_sub: int, gap_px: int = 34, b: int = 50) -> None:
    """Place the figure-wide key ``gap_px`` above the plot area and size the top margin for the title,
    ``n_sub`` subtitle lines and the key (as many rows as it needs in a 900 px cell: a wrapped key
    grows upwards from its bottom anchor)."""
    names: Dict[str, str] = {}
    for t in fig.data:
        if (t.legend or "legend") == "legend" and t.showlegend is not False:
            names.setdefault(t.legendgroup or t.name, t.name)
    margin_t = _title_px(n_sub) + 26 + 20 * (_key_rows(list(names.values())) - 1) + gap_px
    fig.update_layout(height=int(plot_px + margin_t + b), margin=dict(t=margin_t, b=b),
                      legend=dict(y=1 + gap_px / plot_px, yanchor="bottom"))


def direction_detail_figure(history, config=None, *, weights_epoch: Optional[int] = None,
                            n_val: Optional[int] = None, meta: Optional[dict] = None, run_dir=None,
                            height: Optional[int] = None):
    """The direction head per horizon (one column each), every metric against its no-skill reference:
    accuracy vs the majority class, sensitivity / specificity vs 0.5, F1 vs always-up, Brier vs the base
    rate p(1 − p), and calibration in the large (mean P(up) − true up-rate).

    Validation solid, training dotted. Each reference is drawn for the block it belongs to: dashed from
    the validation up-rate, with its 95% sampling-noise range on n_eff = val / h bars shaded (see
    :func:`_ref_noise`; the Brier shade is where a forecast without information lands), and dotted
    from the training block's own up-rate for the training lines."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ctx = _resolve(history, config, weights_epoch, n_val, meta, run_dir)
    df, x = ctx.df, ctx.x
    rows = ("accuracy", "sensitivity ▲ / specificity ▼", "F1 (up class)", "Brier (lower is better)",
            "mean P(up) − true up-rate")
    # short headings (a column is ~270 px in a 1000 px cell); the first row names the horizon in full
    titles = [f"{T.horizon_label(h, config) if i == 0 else h} · {r}" for i, r in enumerate(rows) for h in T.HORIZONS]
    fig = make_subplots(rows=len(rows), cols=3, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06,
                        horizontal_spacing=0.06)
    banded = any(_enough(ctx, h) for h in T.HORIZONS)
    key = [("validation", dict(mode="lines", line=dict(color=T.INK_2, dash=T.VAL_DASH))),
           ("training", dict(mode="lines", line=dict(color=T.INK_2, dash=T.TRAIN_DASH))),
           ("no-skill reference", dict(mode="lines", line=dict(color=T.NEUTRAL, dash="dash"))),
           ("same, training", dict(mode="lines", line=dict(color=T.NEUTRAL, dash=T.TRAIN_DASH, width=1)))]
    if banded:
        key.append(("its 95% noise range",
                    dict(mode="markers", marker=dict(symbol="square", size=12, color=T.rgba(T.NEUTRAL, 0.3)))))
    key += [("sensitivity ▲", dict(mode="lines+markers", line=dict(color=T.UP_COLOR),
                                   marker=dict(symbol="triangle-up", size=9, color=T.UP_COLOR))),
            ("specificity ▼", dict(mode="lines+markers", line=dict(color=T.DOWN_COLOR),
                                   marker=dict(symbol="triangle-down", size=9, color=T.DOWN_COLOR)))]
    _style_key(fig, key, 1, 1)

    def ref(r, c, y, name, train=False):
        if y is not None and np.isfinite(y).any():
            fig.add_trace(go.Scatter(x=x, y=np.asarray(y, np.float32), mode="lines", showlegend=False,
                                     line=dict(color=T.NEUTRAL, dash=T.TRAIN_DASH if train else "dash",
                                               width=1 if train else 1.2),
                                     hovertemplate=f"{name} ({'training' if train else 'validation'} block): "
                                                   f"%{{y:.4g}}<extra></extra>"), r, c)

    def pair(k, h):
        return _col(df, f"val_{k}_{h}"), _col(df, f"train_{k}_{h}")

    for j, h in enumerate(T.HORIZONS):
        c, col = T.HORIZON_COLORS[h], j + 1
        p, pt = _col(df, f"val_true_up_rate_{h}"), _col(df, f"train_true_up_rate_{h}")
        p0 = _val_rate(ctx, h)
        for r, k in ((1, "dir_acc"), (3, "dir_f1"), (4, "dir_brier")):
            v, t = pair(k, h)
            _line(fig, x, v, row=r, col=col, name=f"{h} val", color=c, group=h, show=False)
            _line(fig, x, t, row=r, col=col, name=f"{h} train", color=c, dash=T.TRAIN_DASH, group=h, show=False,
                  width=1.5)
        for q, train in ((p, False), (pt, True)):
            if q is not None:
                ref(1, col, np.maximum(q, 1 - q), "majority-class accuracy", train)
                ref(3, col, 2 * q / (1 + q), "F1 of always up", train)
                ref(4, col, q * (1 - q), "base-rate Brier", train)
        for k, color, sym in (("dir_sensitivity", T.UP_COLOR, "triangle-up"), ("dir_specificity", T.DOWN_COLOR,
                                                                               "triangle-down")):
            v, t = pair(k, h)
            lab = "sensitivity" if "sens" in k else "specificity"
            _line(fig, x, v, row=2, col=col, name=f"{h} {lab} val", color=color, group=f"{h}-{lab}", show=False,
                  symbol=sym, msize=8)
            _line(fig, x, t, row=2, col=col, name=f"{h} {lab} train", color=color, dash=T.TRAIN_DASH,
                  group=f"{h}-{lab}", show=False, width=1.5, symbol=sym, msize=6)
        fig.add_hline(y=0.5, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=2, col=col)
        for pre, dash, w in (("val_", T.VAL_DASH, 2), ("train_", T.TRAIN_DASH, 1.5)):
            m, tr = _col(df, f"{pre}mean_dir_prob_{h}"), _col(df, f"{pre}true_up_rate_{h}")
            if m is not None and tr is not None:
                _line(fig, x, m - tr, row=5, col=col, name=f"{h} {pre[:-1]}", color=c, dash=dash, group=h, show=False,
                      width=w)
        fig.add_hline(y=0.0, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=5, col=col)
        if np.isfinite(p0):   # after the lines: plotly skips shapes on a subplot that has no trace yet
            acc, f1, brier = max(p0, 1 - p0), 2 * p0 / (1 + p0), p0 * (1 - p0)
            na, nf = _ref_noise(ctx, h, "acc", p0), _ref_noise(ctx, h, "f1", p0)
            ceil = _skill_ceiling(ctx, h)
            _shade(fig, [None if na is None else (max(0.0, acc - na), min(1.0, acc + na))], 1, col)
            _shade(fig, [None if nf is None else (max(0.0, f1 - nf), min(1.0, f1 + nf))], 3, col)
            _shade(fig, [None if ceil is None else (brier * (1 - ceil), brier)], 4, col)
            _shade(fig, [None if n is None else (max(0.0, 0.5 - n), min(1.0, 0.5 + n))
                         for n in (_ref_noise(ctx, h, "sens", p0), _ref_noise(ctx, h, "spec", p0))], 2, col)
            nb = _ref_noise(ctx, h, "bias", p0)
            _shade(fig, [None if nb is None else (-nb, nb)], 5, col)
    if ctx.served is not None:
        fig.add_vline(x=x[ctx.served], line=dict(color=T.INK_2, width=1), opacity=0.35, row="all", col="all")
    for r, t in enumerate(("accuracy", "recall", "F1", "Brier", "mean P(up) − rate"), start=1):
        fig.update_yaxes(title_text=t, row=r, col=1)
    fig.update_xaxes(showticklabels=True, tickformat="d")
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="epoch", row=len(rows), col=col)
    T.note_on_empty(fig)
    effs = [f"{h} {_n_eff(ctx, h)}" for h in T.HORIZONS if _n_eff(ctx, h)]
    parts = ["dashed = the no-skill reference on the validation block (up-rate p), dotted grey = the same on the "
             "training block",
             "accuracy max(p, 1 − p)", "F1 of always up 2p / (1 + p)", "Brier p(1 − p)", "recall 0.5", "bias 0"]
    if banded:
        parts += ["shaded = where a no-skill head lands by chance (95%, n_eff " + " / ".join(effs)
                  + "; Brier: no information)"]
    elif ctx.n_val:
        parts += [f"too few validation samples for noise ranges (n_eff {' / '.join(effs)})"]
    else:
        parts += ["validation size unknown: no noise ranges"]
    parts += ["sensitivity = share of up moves called up, specificity = share of down moves called down "
              "(deadband moves excluded)"]
    if ctx.served_epoch:
        parts.append(f"thin vertical line = served epoch {ctx.served_epoch}")
    sub = _wrap(parts)
    plot_px = 230 * len(rows) / (1 - 0.06 * (len(rows) - 1)) if not height else max(400, height - 250)
    T.apply(fig, title="Direction head per horizon", subtitle="<br>".join(sub))
    fig.update_layout(hovermode="x unified")
    _key_top(fig, plot_px, len(sub))
    return fig


def loss_terms_figure(history, config=None, *, weights_epoch: Optional[int] = None, meta: Optional[dict] = None,
                      run_dir=None, height: Optional[int] = None):
    """Each loss term per horizon on its own linear axis (validation solid, training dotted), with the
    weight it enters the total with in its heading; the direction BCE against ln 2 (a constant 0.5)."""
    from plotly.subplots import make_subplots

    ctx = _resolve(history, config, weights_epoch, None, meta, run_dir)
    df, x = ctx.df, ctx.x
    fac = _weight_factors(ctx)

    def times(k):
        return f" · enters × {fac[k]:.3g}" if k in fac else ""

    specs = (("point_h{i}", "point (Huber, λ per horizon inside)"),
             ("dir_loss_h{i}", f"direction BCE{times('direction')} · dashed = ln 2"),
             ("nll_h{i}", f"Gaussian NLL{times('nll')}"),
             ("crps_h{i}", f"CRPS{times('crps')}"),
             ("soft_ece_h{i}", f"soft ECE{times('soft_ece')}"),
             ("extended_h{i}", "trend (extended, λ_ext inside)" + times("trend")))
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, subplot_titles=[t for _, t in specs],
                        vertical_spacing=0.09, horizontal_spacing=0.08)
    first = T.legend_once()
    _style_key(fig, [("validation", dict(mode="lines", line=dict(color=T.INK_2, dash=T.VAL_DASH))),
                     ("training", dict(mode="lines", line=dict(color=T.INK_2, dash=T.TRAIN_DASH)))], 1, 1)
    for n_, (k, _) in enumerate(specs):
        r, c = n_ // 2 + 1, n_ % 2 + 1
        for i, h in enumerate(T.HORIZONS):
            v, t = _col(df, "val_" + k.format(i=i)), _col(df, k.format(i=i))
            _line(fig, x, v, row=r, col=c, name=T.horizon_label(h, config), color=T.HORIZON_COLORS[h], group=h,
                  show=v is not None and first(h))
            _line(fig, x, t, row=r, col=c, name=f"{h} train", color=T.HORIZON_COLORS[h], dash=T.TRAIN_DASH, group=h,
                  show=False, width=1.5)
        fig.update_yaxes(title_text="loss", row=r, col=c)
    fig.add_hline(y=LN2, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=1, col=2)
    if ctx.served is not None:
        fig.add_vline(x=x[ctx.served], line=dict(color=T.INK_2, width=1), opacity=0.35, row="all", col="all")
    fig.update_xaxes(showticklabels=True, tickformat="d")
    for c in (1, 2):
        fig.update_xaxes(title_text="epoch", row=3, col=c)
    T.note_on_empty(fig)
    parts = ["linear axes, each on its own scale", "validation exact, training sampled every TRAIN_METRICS_EVERY steps",
             "direction BCE above ln 2 (dashed) = worse than a constant 0.5"]
    if ctx.served_epoch:
        parts.append(f"thin vertical line = served epoch {ctx.served_epoch}")
    sub = _wrap(parts)
    T.apply(fig, title="Loss terms per horizon (as logged)", subtitle="<br>".join(sub))
    fig.update_layout(hovermode="x unified")
    _key_top(fig, max(400, (height or 900) - 170), len(sub))
    return fig


def batch_loss_figure(points: Sequence[Tuple[float, float]], val_points: Sequence[Tuple[float, float]] = (),
                      *, height: Optional[int] = None, dir_points: Optional[Sequence[tuple]] = None):
    """Live strip: the running-mean training loss after each batch (x in fractional epochs) and the
    validation loss at each epoch end. With ``dir_points`` = [(x, {h: (pred_up, true_up, ...)})] a second
    row shows the running share of up calls per horizon over the current epoch (training, sampled
    steps), the class-collapse signal, with the collapse zones (< 3% / > 97%) marked."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = 2 if dir_points else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.2,
                        subplot_titles=("",) + (("Share of up calls per horizon, running over the current epoch "
                                                  "(training, sampled steps)",) if rows == 2 else ()))

    def per_epoch(xs, ys):
        """Break the line where an epoch starts: the running means are reset there."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ep = np.ceil(xs - 1e-9) - 1
        cut = np.flatnonzero(np.diff(ep) != 0) + 1
        return (np.insert(xs, cut, np.nan).astype(np.float32), np.insert(ys, cut, np.nan).astype(np.float32))

    if points:
        xs, ys = per_epoch(*zip(*points))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="train (running mean over the epoch, per batch)",
                                 line=dict(color=T.INK_2, width=1.5, dash=T.TRAIN_DASH)), 1, 1)
    if val_points:
        xs, ys = zip(*val_points)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="validation (epoch end)",
                                 line=dict(color=T.INK, width=2)), 1, 1)
    if rows == 2:
        for h in T.HORIZONS:
            ys = [(p[1].get(h) or (None,))[0] for p in dir_points]
            xs, ys = per_epoch([p[0] for p in dir_points], [np.nan if v is None else v for v in ys])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{h} up calls (train)",
                                     line=dict(color=T.HORIZON_COLORS[h], dash=T.TRAIN_DASH, width=1.5),
                                     marker=dict(size=4, color=T.HORIZON_COLORS[h])), 2, 1)
        for y0, y1, lab in ((0.0, 0.03, "collapse: never up"), (0.97, 1.0, "collapse: always up")):
            fig.add_hrect(y0=y0, y1=y1, fillcolor=T.CRITICAL, opacity=0.18, line_width=0, row=2, col=1,
                          annotation_text=lab, annotation_position="top left" if y0 else "bottom left",
                          annotation_font=dict(size=10, color=T.MUTED))
        fig.add_hline(y=0.5, line=dict(color=T.NEUTRAL, dash="dash", width=1), row=2, col=1)
        fig.update_yaxes(range=[0, 1], title_text="share", row=2, col=1)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_xaxes(showticklabels=True)
    fig.update_xaxes(title_text="epoch", row=rows, col=1)
    h_px = height or (260 if rows == 1 else 460)
    T.apply(fig, title="Loss, batch by batch", height=h_px)
    fig.update_layout(margin=dict(t=76, b=44), legend=dict(y=1 + 8 / (h_px - 120)))
    return fig


# ------------------------------------------------------------------ registry entries (data, config)
def training_dashboard(data, config=None, **kw):
    return training_dashboard_figure(data, config, **kw)


def training_direction_detail(data, config=None, **kw):
    return direction_detail_figure(data, config, **kw)


def training_loss_terms(data, config=None, **kw):
    return loss_terms_figure(data, config, **kw)
