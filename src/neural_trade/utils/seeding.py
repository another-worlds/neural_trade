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


def set_arithmetic_rewrite(config) -> bool:
    """Set Grappler's arithmetic rewrite EXPLICITLY for ``config``'s model (NT-047); returns it.

    At the multi-series (OHLCV) input the graph is large enough that the rewrite reassociates
    float sums non-reproducibly between graph builds (hash-ordered; measured 1-ulp weight noise
    that training amplifies to ~1e-5 relative metric noise between same-seed runs in one
    process), so it is OFF for those configs. Close-only configs run with it ON, TensorFlow's
    default, so every pre-NT-047 number (scripts/golden_run.py records included) stays
    bit-for-bit. The option is process-wide and applies to graphs traced afterwards, so every
    training run (trainer.train_and_evaluate) and every served prediction (Predictor.predict)
    sets it for its own config: nothing is left over from an earlier run in the same process
    (tests/test_reproducibility.py), and a bundle is served with the rewrite it was trained
    and evaluated with."""
    import tensorflow as tf

    on = len(getattr(config, "INPUT_SERIES", None) or ["close"]) <= 1
    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": on})
    return on


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
