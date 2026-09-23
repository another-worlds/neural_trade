"""Reproducibility: one call seeds every random source a training run touches.

``seed_everything(seed)`` seeds Python's ``random``, NumPy's global generator and TensorFlow
(``tf.keras.utils.set_random_seed``: graph-level seed for initialisers, dropout and noise).
The training dataset's shuffle takes ``Config.SEED`` explicitly (data.datasets).

``TF_DETERMINISTIC_OPS`` is set by ``import neural_trade`` (before TensorFlow loads).
``PYTHONHASHSEED`` only takes effect at interpreter start, so scripts that need it re-exec
themselves (``ensure_hash_seed``). ``deterministic=True`` additionally enables
``tf.config.experimental.enable_op_determinism`` - only for the experiment runner, never in a
notebook mid-session (ops without a deterministic GPU kernel then raise).
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - numpy-only use
        return seed
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    return seed


def ensure_hash_seed(value: str = "0") -> None:
    """Re-exec the current script with PYTHONHASHSEED set, if it is not already."""
    if os.environ.get("PYTHONHASHSEED") == value:
        return
    os.environ["PYTHONHASHSEED"] = value
    os.execv(sys.executable, [sys.executable] + sys.argv)
