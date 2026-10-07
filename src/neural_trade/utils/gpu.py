"""Opt-in per-process GPU memory cap (NT_GPU_MEMORY_LIMIT_MB), so two trainings can share one GPU."""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)
ENV_VAR = "NT_GPU_MEMORY_LIMIT_MB"


def apply_gpu_memory_limit() -> bool:
    """Cap this process's GPU memory when ``NT_GPU_MEMORY_LIMIT_MB`` is a positive integer.

    Must run before the GPU is initialised. Unset, empty, 0 or invalid: does nothing. Never raises.
    Returns True only when the cap was applied.
    """
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return False
    try:
        limit = int(raw)
    except ValueError:
        LOG.warning("%s=%r is not an integer; no GPU memory cap", ENV_VAR, raw)
        return False
    if limit <= 0:
        return False
    import tensorflow as tf  # after `import neural_trade` (puts the CUDA DLLs on PATH)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        LOG.info("%s=%d: no GPU visible; no cap applied", ENV_VAR, limit)
        return False
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=limit)])
    except RuntimeError as exc:  # GPU already initialised
        LOG.warning("%s=%d not applied: %s", ENV_VAR, limit, exc)
        return False
    LOG.info("GPU memory capped at %d MB for this process (%s)", limit, ENV_VAR)
    return True
