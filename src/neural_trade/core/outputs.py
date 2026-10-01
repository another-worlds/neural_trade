"""Named views over the model's outputs and the loss's components (no TensorFlow import).

Both are namedtuples, i.e. tuples: positional unpacking at existing call sites keeps
working, and new code can use attribute access.
"""
from __future__ import annotations

from collections import namedtuple

#: The 10 model outputs, in output order (3 horizons x (price, direction, variance) + overflow).
#: ``variance_*`` is in SCALED units^2; ``vacuum_overflow`` is 0 at inference.
PredictiveOutputs = namedtuple(
    "PredictiveOutputs",
    [
        "price_h0", "direction_h0", "variance_h0",
        "price_h1", "direction_h1", "variance_h1",
        "price_h2", "direction_h2", "variance_h2",
        "vacuum_overflow",
    ],
)

#: The 37 components returned by the training objective (``losses.custom_loss`` and any objective
#: that wraps it, e.g. ``pnl_utility``, NT-087).
#: ``local_*``/``global_*`` are retired trend terms kept as exact zeros so the contract holds.
#: ``pnl_val`` is the lambda-weighted mean-variance P&L utility term (0.0 for ``custom_loss`` itself).
#: ``dir_align_val``/``coherence_penalty_val`` (NT-037, D-045 amendment) are the two addends of
#: ``total`` that had no field before: the direction/Gaussian alignment term (already multiplied
#: by its inner ``LAMBDA_DIR_ALIGN``; ``total`` applies ``lambda_dir_align_outer`` on top, default
#: 0) and the raw coherence penalty (``total`` applies ``lambda_coherence_outer`` on top). Added at
#: the end so every existing positional unpack keeps working; the two call sites that unpack the
#: full tuple (``training/custom_model.py`` train_step/test_step) were updated in the same change.
LossComponents = namedtuple(
    "LossComponents",
    [
        "total",
        "point_h0", "point_h1", "point_h2",
        "local_h0", "global_h0", "extended_h0",
        "local_h1", "global_h1", "extended_h1",
        "local_h2", "global_h2", "extended_h2",
        "dir_h0", "dir_h1", "dir_h2",
        "nll_h0", "nll_h1", "nll_h2",
        "reg_loss", "inter_reg", "vol_loss",
        "crps_h0", "crps_h1", "crps_h2",
        "soft_ece_h0", "soft_ece_h1", "soft_ece_h2",
        "t_perp_total", "casimir_val", "vac_val", "hd_val", "ife_val",
        "vac_overflow_val",
        "pnl_val",
        "dir_align_val", "coherence_penalty_val",
    ],
)

__all__ = ["PredictiveOutputs", "LossComponents"]
