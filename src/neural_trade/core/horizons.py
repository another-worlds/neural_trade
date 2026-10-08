"""The one source of horizon keys and of the per-horizon layout rules (NT-042, D-022).

A run forecasts ``N = len(Config.HORIZON_STEPS)`` horizons, keyed ``h0 .. h{N-1}`` in ascending
order. The reference setup has N = 3 (10, 15 and 20 bars); every rule here reproduces today's
three-horizon layout exactly at N = 3, so the golden run (``scripts/golden_run.py verify``) is
unchanged.

* ``horizon_keys(n)``: the keys. ``REFERENCE_HORIZONS`` is ``horizon_keys(3)``, the default for
  callers that have no config and no data to read the count from (stored three-horizon runs).
* ``primary_index(n)``: the horizon the single-horizon read-outs use (the variance-penalty term,
  the strategies' entry line, the trade figures). ``n // 2``: h1 at N = 2 and N = 3, h2 at N = 4.
* ``point_weight_role(i, n)``: which configured point-loss weight horizon ``i`` takes: ``short``
  (LAMBDA_SHORT) for h0, ``long`` (LAMBDA_LONG) for the last horizon, ``point`` (LAMBDA_POINT)
  for the ones between. At N = 3 that is today's h0 short, h1 point, h2 long.
* ``neighbour_pairs(n)`` and ``pair_sum_scale(n)``: the pairwise physics terms (Casimir, IFE, the
  coherence magnitude ordering) run over every neighbouring pair (h_i, h_i+1); their per-pair
  values are averaged and multiplied by 2 (the reference pair count), so N = 3 gives today's sum
  of the two pairs exactly (lead's reading of D-022).
* ``predictive_outputs_type(n)`` / ``loss_components_type(n)``: the namedtuple classes for N
  horizons. At N = 3 they are the existing :data:`~neural_trade.core.outputs.PredictiveOutputs`
  and :data:`~neural_trade.core.outputs.LossComponents` classes themselves.

No TensorFlow import: every layer (models, losses, serving, figures) may read this module.
"""
from __future__ import annotations

import re
from collections import namedtuple
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

#: The reference setup's horizon count (D-019, D-022).
REFERENCE_N_HORIZONS = 3

_KEY_RE = re.compile(r"^h(\d+)$")


def horizon_keys(n: int) -> Tuple[str, ...]:
    """``('h0', ..., 'h{n-1}')``."""
    n = int(n)
    if n < 1:
        raise ValueError(f"a run needs at least one horizon, got {n}")
    return tuple(f"h{i}" for i in range(n))


REFERENCE_HORIZONS: Tuple[str, ...] = horizon_keys(REFERENCE_N_HORIZONS)


def n_horizons(config=None, default: int = REFERENCE_N_HORIZONS) -> int:
    """The horizon count of ``config`` (``len(HORIZON_STEPS)``), ``default`` without one."""
    steps = getattr(config, "HORIZON_STEPS", None) if config is not None else None
    return len(steps) if steps else int(default)


def keys_for(config=None) -> Tuple[str, ...]:
    """The horizon keys of ``config`` (the reference three without a config)."""
    return horizon_keys(n_horizons(config))


def horizon_index(h: str) -> int:
    """``'h2'`` -> 2. Raises ValueError for anything that is not a horizon key."""
    m = _KEY_RE.match(str(h))
    if m is None:
        raise ValueError(f"{h!r} is not a horizon key (h0, h1, ...)")
    return int(m.group(1))


def keys_of(mapping: Optional[Mapping], default: Sequence[str] = REFERENCE_HORIZONS) -> Tuple[str, ...]:
    """The horizon keys present in a ``{'h0': ..., 'h1': ...}`` mapping, in index order.

    Keys that are not horizon keys are ignored; ``default`` when the mapping holds none. Used by
    every reader of stored predictions (bundles, frames, reports), so a three-horizon run written
    before NT-042 reads back as h0-h2 and an N-horizon run as h0..h{N-1}.
    """
    if not mapping:
        return tuple(default)
    keys = [k for k in mapping if isinstance(k, str) and _KEY_RE.match(k)]
    return tuple(sorted(keys, key=horizon_index)) if keys else tuple(default)


def primary_index(n: int) -> int:
    """The primary horizon's index: ``n // 2`` (h1 at N = 2 and 3, h2 at N = 4)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"a run needs at least one horizon, got {n}")
    return n // 2


def primary_key(n_or_keys) -> str:
    """The primary horizon's key, for a count or a sequence of keys."""
    if isinstance(n_or_keys, int):
        return f"h{primary_index(n_or_keys)}"
    keys = tuple(n_or_keys)
    return keys[primary_index(len(keys))]


#: The reference setup's primary horizon (h1).
PRIMARY_HORIZON: str = primary_key(REFERENCE_N_HORIZONS)


def point_weight_role(i: int, n: int) -> str:
    """``'short'``, ``'point'`` or ``'long'``: the configured point-loss weight horizon i takes."""
    if n <= 1:
        return "point"
    if i == 0:
        return "short"
    if i == n - 1:
        return "long"
    return "point"


def neighbour_pairs(n: int) -> List[Tuple[int, int]]:
    """``[(0, 1), (1, 2), ..., (n-2, n-1)]``; empty for one horizon."""
    return [(i, i + 1) for i in range(max(int(n) - 1, 0))]


def pair_sum_scale(n: int) -> float:
    """The factor on the SUM over neighbouring pairs of a pairwise term: 2 / (n - 1).

    Averages the per-pair values and scales the average by the reference pair count (2), so
    N = 3 returns exactly 1.0 (today's plain sum of the (h0, h1) and (h1, h2) terms). 0.0 when
    there is no pair.
    """
    pairs = max(int(n) - 1, 0)
    if pairs == 0:
        return 0.0
    if pairs == REFERENCE_N_HORIZONS - 1:
        return 1.0
    return float(REFERENCE_N_HORIZONS - 1) / float(pairs)


# ------------------------------------------------------------------ named views of N horizons
def predictive_output_fields(n: int) -> Tuple[str, ...]:
    """The model's output names: (price, direction, variance) per horizon, then the overflow."""
    out: List[str] = []
    for h in horizon_keys(n):
        out += [f"price_{h}", f"direction_{h}", f"variance_{h}"]
    return tuple(out) + ("vacuum_overflow",)


def loss_component_fields(n: int) -> Tuple[str, ...]:
    """The training objective's component names for N horizons (the N = 3 order is today's)."""
    hs = horizon_keys(n)
    out: List[str] = ["total"]
    out += [f"point_{h}" for h in hs]
    for h in hs:
        out += [f"local_{h}", f"global_{h}", f"extended_{h}"]
    out += [f"dir_{h}" for h in hs]
    out += [f"nll_{h}" for h in hs]
    out += ["reg_loss", "inter_reg", "vol_loss"]
    out += [f"crps_{h}" for h in hs]
    out += [f"soft_ece_{h}" for h in hs]
    out += ["t_perp_total", "casimir_val", "vac_val", "hd_val", "ife_val", "vac_overflow_val", "pnl_val",
            "dir_align_val", "coherence_penalty_val"]
    return tuple(out)


@lru_cache(maxsize=None)
def predictive_outputs_type(n: int):
    """The PredictiveOutputs namedtuple class for N horizons (the reference class at N = 3)."""
    from neural_trade.core.outputs import PredictiveOutputs

    n = int(n)
    if n == REFERENCE_N_HORIZONS:
        return PredictiveOutputs
    return namedtuple(f"PredictiveOutputs{n}", predictive_output_fields(n))


@lru_cache(maxsize=None)
def loss_components_type(n: int):
    """The LossComponents namedtuple class for N horizons (the reference class at N = 3)."""
    from neural_trade.core.outputs import LossComponents

    n = int(n)
    if n == REFERENCE_N_HORIZONS:
        return LossComponents
    return namedtuple(f"LossComponents{n}", loss_component_fields(n))


def n_from_outputs(n_outputs: int) -> int:
    """The horizon count of a model with ``n_outputs`` outputs (3 heads per horizon + overflow)."""
    n_outputs = int(n_outputs)
    if n_outputs < 4 or (n_outputs - 1) % 3:
        raise ValueError(f"{n_outputs} outputs is not 3 heads per horizon plus vacuum_overflow")
    return (n_outputs - 1) // 3


def per_horizon(components, prefix: str, keys: Iterable[str]) -> list:
    """``[components.<prefix>h0, ...]`` (a LossComponents of any N)."""
    return [getattr(components, f"{prefix}{h}") for h in keys]


def components_horizons(components) -> Tuple[str, ...]:
    """The horizon keys of a LossComponents instance of any N (read from its ``point_h*`` fields)."""
    keys = [f[len("point_"):] for f in components._fields if f.startswith("point_")]
    return tuple(sorted(keys, key=horizon_index))


__all__ = [
    "REFERENCE_N_HORIZONS", "REFERENCE_HORIZONS", "PRIMARY_HORIZON", "horizon_keys", "n_horizons", "keys_for",
    "horizon_index", "keys_of", "primary_index", "primary_key", "point_weight_role", "neighbour_pairs",
    "pair_sum_scale", "predictive_output_fields", "loss_component_fields", "predictive_outputs_type",
    "loss_components_type", "n_from_outputs", "per_horizon", "components_horizons",
]
