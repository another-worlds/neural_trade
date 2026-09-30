"""Per-term loss weights as live, non-trainable tf.Variables (Phase A S7; moved in B10).

``model.lambda_x`` returns a tf.Variable usable directly inside the traced train step;
``model.lambda_x = v`` assigns in place, so calibration, schedules and ablations take effect on
the next step without retracing. The variables live in ``model._lambda_vars`` (created with
attribute tracking off), so they are not part of the Keras weights file; they are persisted in
the artifact bundle's meta.json instead.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import tensorflow as tf

_LAMBDA_VARIABLE_KEYS = ('short', 'point', 'long', 'extended_trend', 'dir', 'var', 'vol',
                         'crps', 'soft_ece', 't_perp', 'casimir', 'hd', 'ife', 'vac_overflow', 'pnl',
                         # Four of the five outer multipliers (NT-092): CustomTrainModel.__init__
                         # already assigns them through `self.lambda_<key> = ...`, so adding them here
                         # turns that assignment into the same tf.Variable property as the 15 above,
                         # with no other code change - losses/functions.py's
                         # `model.lambda_trend_outer * x` etc. already work on a tf.Variable exactly
                         # as they did on a plain float. Needed so screen phase 2 (a group of trials
                         # that reuse one traced graph) can change these values between trials without
                         # retracing. NOT 'dir_align_outer': losses/functions.py:675 does
                         # `if float(getattr(model, 'lambda_dir_align_outer', 0.0)) > 0.0:` - a
                         # Python-level branch that SKIPS THE OP ENTIRELY when the value is 0, traced
                         # once at graph-build time. `float()` on a Variable read INSIDE the traced
                         # function raises (it is a symbolic Tensor there, not a concrete Python
                         # value); turning it into a Variable does not just fail to help, it breaks
                         # training outright. Fixing that call site is outside this item's files
                         # (losses/), so LAMBDA_DIR_ALIGN_OUTER stays a plain float and, for screen
                         # phase 2, a STRUCTURAL field (see screen.py's STRUCTURAL_ONLY_LAMBDA_FIELDS).
                         'trend_outer', 'dir_outer', 'nll_outer', 'coherence_outer')


def _make_lambda_property(key):
    name = f'lambda_{key}'

    def _get(self):
        return self._lambda_vars[key]

    def _set(self, value):
        var = self._lambda_vars.get(key)
        if var is None:
            self._lambda_vars[key] = tf.Variable(float(value), trainable=False, dtype=tf.float32, name=name)
        else:
            var.assign(float(value))

    return property(_get, _set, doc=f"Non-trainable tf.Variable weight for the '{key}' loss term.")

# Config field -> model key, for the 15 weights that are variables.
CONFIG_NAME_OF_KEY: Dict[str, str] = {
    "short": "LAMBDA_SHORT", "point": "LAMBDA_POINT", "long": "LAMBDA_LONG",
    "extended_trend": "LAMBDA_EXTENDED_TREND", "dir": "LAMBDA_DIR", "var": "LAMBDA_VAR", "vol": "LAMBDA_VOL",
    "crps": "LAMBDA_CRPS", "soft_ece": "LAMBDA_SOFT_ECE", "t_perp": "LAMBDA_T_PERP",
    "casimir": "LAMBDA_CASIMIR", "hd": "LAMBDA_HD", "ife": "LAMBDA_IFE", "vac_overflow": "LAMBDA_VAC_OVERFLOW",
    "pnl": "LAMBDA_PNL",
    # Four of the five outer multipliers: LAMBDA_COHERENCE (no "_OUTER" suffix in Config) is the one
    # irregular name; the rest are LAMBDA_<KEY_UPPER>. LAMBDA_DIR_ALIGN_OUTER is deliberately absent
    # (see _LAMBDA_VARIABLE_KEYS above); `ablate()`'s explicit `elif name == "LAMBDA_DIR_ALIGN_OUTER"`
    # branch below still handles it.
    "trend_outer": "LAMBDA_TREND_OUTER", "dir_outer": "LAMBDA_DIR_OUTER", "nll_outer": "LAMBDA_NLL_OUTER",
    "coherence_outer": "LAMBDA_COHERENCE",
}
KEY_OF_CONFIG_NAME: Dict[str, str] = {v: k for k, v in CONFIG_NAME_OF_KEY.items()}


def _get_lambda_values(self):
    """Current per-term loss weights as plain floats (for logging, ablation and export)."""
    return {f'lambda_{k}': float(v.numpy()) for k, v in self._lambda_vars.items()}


def _set_lambda_values(self, **weights):
    """Assign per-term loss weights in place, e.g. model.set_lambda_values(lambda_hd=0.0)."""
    for name, value in weights.items():
        if not name.startswith('lambda_') or name[len('lambda_'):] not in _LAMBDA_VARIABLE_KEYS:
            raise KeyError(f"unknown loss weight {name!r}; known: {[f'lambda_{k}' for k in _LAMBDA_VARIABLE_KEYS]}")
        setattr(self, name, value)


def install_lambda_properties(cls) -> None:
    """Give ``cls`` a ``lambda_<key>`` property per key plus get/set_lambda_values."""
    for key in _LAMBDA_VARIABLE_KEYS:
        setattr(cls, f"lambda_{key}", _make_lambda_property(key))
    cls.get_lambda_values = _get_lambda_values
    cls.set_lambda_values = _set_lambda_values


def ablate(model, config_names: Iterable[str]) -> List[str]:
    """Force the named loss weights to 0 on ``model`` (after calibration, for ablations).

    Names are Config fields (``LAMBDA_HD``). Weights that are variables are assigned in place;
    ``LAMBDA_VAC`` (a threshold read from the config inside the loss) is zeroed on
    ``model.config``. Returns the names that were applied.
    """
    applied = []
    for name in config_names or []:
        key = KEY_OF_CONFIG_NAME.get(name)
        if key is not None:
            setattr(model, f"lambda_{key}", 0.0)
        elif name == "LAMBDA_VAC":
            model.config.LAMBDA_VAC = 0.0
        elif name == "LAMBDA_DIR_ALIGN_OUTER":
            model.lambda_dir_align_outer = 0.0
        else:
            raise ValueError(f"cannot ablate {name}: not a per-term loss weight")
        applied.append(name)
    return applied
