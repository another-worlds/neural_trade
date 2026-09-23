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
                         'crps', 'soft_ece', 't_perp', 'casimir', 'hd', 'ife', 'vac_overflow')


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

# Config field -> model key, for the 14 weights that are variables.
CONFIG_NAME_OF_KEY: Dict[str, str] = {
    "short": "LAMBDA_SHORT", "point": "LAMBDA_POINT", "long": "LAMBDA_LONG",
    "extended_trend": "LAMBDA_EXTENDED_TREND", "dir": "LAMBDA_DIR", "var": "LAMBDA_VAR", "vol": "LAMBDA_VOL",
    "crps": "LAMBDA_CRPS", "soft_ece": "LAMBDA_SOFT_ECE", "t_perp": "LAMBDA_T_PERP",
    "casimir": "LAMBDA_CASIMIR", "hd": "LAMBDA_HD", "ife": "LAMBDA_IFE", "vac_overflow": "LAMBDA_VAC_OVERFLOW",
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
