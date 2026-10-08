"""Pre-training loss-weight calibration (extracted from train_and_evaluate in B10).

Measures the natural magnitude of each loss component over a few batches with every weight at
1.0, then rescales the weights so the components start on a common scale:

    new_lambda = clip(orig_lambda * (ref / median_component) ** damping, CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX)

with ``ref`` the mean of the non-zero medians. Components with damping 0 (the trend prior and
the bounded physics regularisers, by default) keep their configured weight. If anything fails,
the configured weights are restored and ``None`` is returned.

``Config.CALIB_MODE`` (NT-101) selects the measurement:

- ``"value"`` (default): the above, unchanged (golden-run bit-for-bit).
- ``"gradient"``: ``median_component`` is replaced by the term's mean gradient norm, on the trunk
  (incl. per-horizon towers: every main-group trainable variable except the indicator logits and
  the price/direction/variance head Dense layers), of the term exactly **as it enters `total`**
  in ``losses.functions.custom_loss`` — horizon sums (not an average) times whatever outer
  multiplier `total` applies (``lambda_trend_outer``/``lambda_dir_outer``/``lambda_nll_outer``,
  or ``0.1`` for the volatility penalty); see ``_total_term_tensors``. Sampled over the same
  calibration steps (GradNorm-style; A_losses.md recommendation 2). A term this pass never
  assigns a weight to (``vac``, ``vac_overflow``) or whose rescale would be a no-op (damping 0,
  or the term's own config weight is 0) is skipped entirely — no backward pass is run for it. The
  chosen weights, plus each measured term's pre- and post-calibration gradient norm and its share
  of the pre-calibration total, are added to the returned dict (and so to ``artifacts/meta.json``,
  via ``TrainResult.calibration_lambdas``).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

#: The terms the calibration pass measures and (mostly) rescales, in a fixed order. Matches the
#: 13 rescaled lambdas of ``_CALIB_LAMBDA_NAMES`` plus ``vac``/``vac_overflow``, which only feed the
#: shared reference (their weight is a threshold, never rescaled).
_CALIB_TERM_NAMES = ('short', 'point', 'long', 'ext', 'dir', 'var', 'vol', 'crps', 'ece',
                     't_perp', 'casimir', 'vac', 'hd', 'ife', 'vac_overflow')

#: short term-key -> the ``lambda_*`` attribute it calibrates (the 13 rescaled lambdas; ``vac``
#: and ``vac_overflow`` are never assigned a weight by this pass and have no entry here).
_LAMBDA_NAME_OF = {
    'short': 'lambda_short', 'point': 'lambda_point', 'long': 'lambda_long',
    'ext': 'lambda_extended_trend', 'dir': 'lambda_dir', 'var': 'lambda_var', 'vol': 'lambda_vol',
    'crps': 'lambda_crps', 'ece': 'lambda_soft_ece', 't_perp': 'lambda_t_perp',
    'casimir': 'lambda_casimir', 'hd': 'lambda_hd', 'ife': 'lambda_ife',
}


def _term_values(loss_components):
    """The calibration pass's 15 measured quantities, read from ``LossComponents`` **by attribute**,
    not by position (NT-101): forward-compatible with fields a later change appends after
    ``pnl_val`` (for example NT-037's ``dir_align_val``/``coherence_penalty_val``), since attribute
    access does not depend on the tuple's length or field order.
    """
    lc = loss_components
    ext = (lc.extended_h0 + lc.extended_h1 + lc.extended_h2) / 3.0
    dirn = (lc.dir_h0 + lc.dir_h1 + lc.dir_h2) / 3.0
    var = (lc.nll_h0 + lc.nll_h1 + lc.nll_h2) / 3.0
    crps = (lc.crps_h0 + lc.crps_h1 + lc.crps_h2) / 3.0
    ece = (lc.soft_ece_h0 + lc.soft_ece_h1 + lc.soft_ece_h2) / 3.0
    return {
        'short': lc.point_h0, 'point': lc.point_h1, 'long': lc.point_h2,
        'ext': ext, 'dir': dirn, 'var': var, 'vol': lc.vol_loss,
        'crps': crps, 'ece': ece,
        't_perp': lc.t_perp_total, 'casimir': lc.casimir_val, 'vac': lc.vac_val,
        'hd': lc.hd_val, 'ife': lc.ife_val, 'vac_overflow': lc.vac_overflow_val,
    }


def _total_term_tensors(custom_model, loss_components):
    """``CALIB_MODE=gradient``'s measured quantities (NT-101 QA repair round 1): each term
    exactly as it enters ``losses.functions.custom_loss``'s ``total`` (see that function,
    building its ``total =`` expression, roughly :771-789) — horizon **sums**, not
    ``_term_values``'s ``/3`` averages, and multiplied by whatever outer multiplier ``total``
    applies to that group (``lambda_trend_outer`` for the extended-trend sum, ``lambda_dir_outer``
    for the direction sum, ``lambda_nll_outer`` for the NLL sum, ``0.1`` for the volatility
    penalty; CRPS and soft ECE have no outer multiplier). Call with the term's OWN weight already
    reset to 1.0 (as ``calibrate_loss_weights`` does for every sampled component) so each tensor
    reads "this term, at weight 1, exactly as `total` would otherwise combine it" — the model's
    *current* (not reset) outer multipliers are used, matching how `total` actually weights them
    during this pass. ``value`` mode is unaffected: it keeps using ``_term_values``'s pre-existing
    ``/3`` averages unchanged (golden-run bit-for-bit); that mode's own inconsistency with how its
    measured quantity enters `total` is a separate, filed issue (NT-039 note, QA of NT-101), not
    something this item changes.

    ``vac`` and ``vac_overflow`` are not included here: this pass never assigns either a weight
    (see ``_LAMBDA_NAME_OF``), so there is nothing to equalise them against; the caller skips their
    backward pass entirely rather than measuring an unused quantity.
    """
    lc = loss_components
    m = custom_model
    return {
        'short': lc.point_h0, 'point': lc.point_h1, 'long': lc.point_h2,
        'ext': m.lambda_trend_outer * (lc.extended_h0 + lc.extended_h1 + lc.extended_h2),
        'dir': m.lambda_dir_outer * (lc.dir_h0 + lc.dir_h1 + lc.dir_h2),
        'var': m.lambda_nll_outer * (lc.nll_h0 + lc.nll_h1 + lc.nll_h2),
        'vol': 0.1 * lc.vol_loss,
        'crps': lc.crps_h0 + lc.crps_h1 + lc.crps_h2,
        'ece': lc.soft_ece_h0 + lc.soft_ece_h1 + lc.soft_ece_h2,
        't_perp': lc.t_perp_total, 'casimir': lc.casimir_val,
        'hd': lc.hd_val, 'ife': lc.ife_val,
    }


def rescale_weight(orig: float, measured: float, damping: float, ref: float, lam_min: float,
                    lam_max: float, eps: float = 1e-8, *, name: str = '', quiet: bool = False) -> float:
    """The calibration pass's damped rescale-and-clamp, shared by every term and both
    ``CALIB_MODE`` values (NT-101): ``measured`` is a median value in 'value' mode or a mean
    gradient norm in 'gradient' mode; either way, ``damping=1`` makes ``orig * measured`` (the
    weighted quantity, or its gradient) equal to ``ref`` for every term with ``measured > eps``.
    A component with ``measured <= eps`` (inactive, or disconnected from the measured graph) keeps
    its original weight. NT-118: a component with ``damping == 0`` is not rescaled and keeps its
    configured weight too, unclamped (a configured 0 stays 0, a value outside the clamp stays
    as configured); only a rescaled weight (``damping > 0``) is clipped to
    ``[lam_min, lam_max]``, and a clip that binds logs a WARNING (``quiet`` lowers it to INFO for
    the one documented lift, ``CALIB_VOL_ZERO_TO_FLOOR``).
    """
    if damping == 0.0 or measured <= eps:
        return orig
    raw = orig * (ref / (measured + eps)) ** damping
    out = float(np.clip(raw, lam_min, lam_max))
    if out != raw:
        logger.log(logging.INFO if quiet else logging.WARNING,
                   "[calib] %s weight %.6g clipped to %.6g (clamp [%g, %g])",
                   name or 'a loss', raw, out, lam_min, lam_max)
    return out


def _trunk_variables(custom_model):
    """The trunk, incl. per-horizon towers — the variables gradient-norm calibration equalises
    against (NT-101): every main-group trainable variable (the indicator logits, routed to the
    second optimizer, are excluded) except the price/direction/variance head Dense layers,
    identified by their layer name (``price_h*``, ``direction_h*[_logit|_skip]``, ``variance_h*``;
    see ``models/gru_attention.py``). This keeps each horizon's own ``tower_h*`` Dense layer (the
    16-unit block that feeds that horizon's heads) in the "trunk": it is not literally shared
    across horizons, but it is not a head either, and excluding it would leave several of the
    larger per-horizon variable groups out of every gradient-norm measurement. A variable's layer
    name is the part of its TF name before the first ``/`` (or the whole name for an unscoped
    variable)."""
    head_prefixes = ('price_h', 'direction_h', 'variance_h')
    indicator_ids = getattr(custom_model, '_indicator_var_ids', set())
    out = []
    for v in custom_model.trainable_variables:
        if id(v) in indicator_ids:
            continue
        layer_name = v.name.split('/', 1)[0]
        if layer_name.startswith(head_prefixes):
            continue
        out.append(v)
    return out


def calibrate_loss_weights(custom_model, train_ds, cfg, n_train: int) -> Optional[Dict[str, float]]:
    """Run the calibration pass on ``custom_model``; return the calibrated weights (or None).

    On failure (NT-111): the configured lambdas are always restored (never left at the 1.0 reset
    used for sampling). What happens next depends on ``CALIB_MODE`` and ``Config.CALIB_FAIL_LOUD``:

    - ``CALIB_MODE='gradient'``: always returns a dict with ``calib_failed: true`` and
      ``calib_error`` (never ``None``), so a trained run's ``calibration_lambdas`` (and so
      ``artifacts/meta.json``) records the failure instead of looking like an ordinary
      calibration; ``experiments.scorer.score_result`` refuses to score such a run (NT-111).
    - ``CALIB_MODE='value'`` with ``CALIB_FAIL_LOUD`` False (the default): unchanged, golden-run
      bit-for-bit — logs a warning and returns ``None``.
    - ``CALIB_MODE='value'`` with ``CALIB_FAIL_LOUD`` True: the same loud ``calib_failed`` dict as
      the gradient branch.
    """
    calib_mode_for_failure = str(getattr(cfg, 'CALIB_MODE', 'value')).lower()  # safe even if Phase 1/2 never run
    _calib_lambdas: Optional[Dict[str, float]] = None  # set below if calibrate=True
    _CALIB_LAMBDA_NAMES = (
        'lambda_short', 'lambda_point', 'lambda_long', 'lambda_extended_trend', 'lambda_dir',
        'lambda_var', 'lambda_vol', 'lambda_crps', 'lambda_soft_ece',
        'lambda_t_perp', 'lambda_casimir', 'lambda_hd', 'lambda_ife',
    )
    _calib_saved: Dict[str, float] = {}  # populated after the originals are read; restored on failure
    if True:
        try:
            # ----------------------------------------------------------------
            # Read calibration knobs from config (with safe fallbacks)
            # ----------------------------------------------------------------
            # Derive batch counts from actual training dataset size so the
            # calibration overhead scales with the training set, not a hardcoded number.
            train_batches  = math.ceil(n_train / cfg.BATCH_SIZE)
            warmup_frac    = float(getattr(cfg, 'CALIB_WARMUP_FRACTION', 0.15))
            sample_frac    = float(getattr(cfg, 'CALIB_SAMPLE_FRACTION', 0.35))
            n_warmup       = max(1, round(train_batches * warmup_frac))
            n_sample       = max(1, round(train_batches * sample_frac))
            lam_min    = float(getattr(cfg, 'CALIB_LAMBDA_MIN', 0.1))
            lam_max    = float(getattr(cfg, 'CALIB_LAMBDA_MAX', 20.0))
            # NT-028: the DAMPING fallback (`getattr(cfg, 'DAMPING', ...)`) was dead for a real
            # Config, which always has CALIB_DAMPING; DAMPING itself is deprecated (Config.validate
            # warns if it is set). Read CALIB_DAMPING directly.
            d_global   = float(getattr(cfg, 'CALIB_DAMPING', 0.5))
            calib_outer = bool(getattr(cfg, 'CALIB_OUTER', False))
            # NT-101: 'value' (default) measures each term's median value, as before; 'gradient'
            # measures each term's gradient norm on the shared trunk instead (GradNorm-style).
            calib_mode = str(getattr(cfg, 'CALIB_MODE', 'value')).lower()
            calib_mode_for_failure = calib_mode  # keep in sync now that Phase 1 knows the real value

            def _d(attr):
                """Resolve per-component damping, falling back to global."""
                v = getattr(cfg, attr, None)
                return float(v) if v is not None else d_global

            d_point = _d('CALIB_DAMPING_POINT')
            d_trend = _d('CALIB_DAMPING_TREND')
            d_dir   = _d('CALIB_DAMPING_DIR')
            d_var   = _d('CALIB_DAMPING_VAR')
            d_crps  = _d('CALIB_DAMPING_CRPS')
            d_ece   = _d('CALIB_DAMPING_ECE')
            d_vol   = _d('CALIB_DAMPING_VOL')
            d_physics = _d('CALIB_DAMPING_PHYSICS')  # 0.0 by default: bounded regularisers are excluded from equalisation

            # ----------------------------------------------------------------
            # Save originals and reset all per-component lambdas to 1.0
            # so that natural magnitudes are measured without existing weights.
            # ----------------------------------------------------------------
            orig_short = float(custom_model.lambda_short)
            orig_point = float(custom_model.lambda_point)
            orig_long  = float(custom_model.lambda_long)
            orig_ext   = float(custom_model.lambda_extended_trend)
            orig_dir   = float(custom_model.lambda_dir)
            orig_var   = float(custom_model.lambda_var)
            orig_vol   = float(custom_model.lambda_vol)
            orig_crps  = float(custom_model.lambda_crps)
            orig_ece   = float(custom_model.lambda_soft_ece)
            orig_t_perp   = float(custom_model.lambda_t_perp)
            orig_casimir  = float(custom_model.lambda_casimir)
            orig_hd       = float(custom_model.lambda_hd)
            orig_ife      = float(custom_model.lambda_ife)
            _calib_saved.update({_n: float(getattr(custom_model, _n)) for _n in _CALIB_LAMBDA_NAMES})

            custom_model.lambda_short            = 1.0
            custom_model.lambda_point            = 1.0
            custom_model.lambda_long             = 1.0
            custom_model.lambda_extended_trend   = 1.0
            custom_model.lambda_dir              = 1.0
            custom_model.lambda_var              = 1.0
            custom_model.lambda_vol              = 1.0
            custom_model.lambda_crps             = 1.0
            custom_model.lambda_soft_ece         = 1.0
            custom_model.lambda_t_perp           = 1.0
            custom_model.lambda_casimir          = 1.0
            custom_model.lambda_hd               = 1.0
            custom_model.lambda_ife              = 1.0

            # Whether each config-gated term is active at all (its OWN configured weight is
            # > 0), independent of damping. Needed before Phase 2 in gradient mode (an inactive
            # term, like a damping-0 one, is never rescaled, so its backward pass is skipped —
            # see "to_measure" below); value mode only reads these afterwards, in Phase 3, as before.
            crps_active = float(getattr(cfg, 'LAMBDA_CRPS', 0.0)) > 0.0
            ece_active  = float(getattr(cfg, 'LAMBDA_SOFT_ECE', 0.0)) > 0.0
            t_perp_active  = float(getattr(cfg, 'LAMBDA_T_PERP',  0.0)) > 0.0
            casimir_active = float(getattr(cfg, 'LAMBDA_CASIMIR', 0.0)) > 0.0
            hd_active      = float(getattr(cfg, 'LAMBDA_HD',      0.0)) > 0.0
            ife_active     = float(getattr(cfg, 'LAMBDA_IFE',     0.0)) > 0.0
            # NT-118: a configured LAMBDA_VOL of 0 is lifted to CALIB_LAMBDA_MIN only while
            # CALIB_VOL_ZERO_TO_FLOOR is on (the default: the arm NT-099 tested, D-058); with it
            # off, 0 means off, as for soft ECE.
            vol_zero_to_floor = bool(getattr(cfg, 'CALIB_VOL_ZERO_TO_FLOOR', True))
            vol_active = (float(getattr(cfg, 'LAMBDA_VOL', 0.0)) > 0.0) or vol_zero_to_floor
            # PRICE_HEAD='none': the point, extended-trend, vol, Casimir and IFE terms are exactly 0
            # (no price head). They are neither measured nor rescaled, and vol is never lifted to
            # CALIB_LAMBDA_MIN; their configured weights stay as they are.
            price_off = str(getattr(cfg, 'PRICE_HEAD', 'on')) == 'none'
            if price_off:
                vol_zero_to_floor = False
                vol_active = casimir_active = ife_active = False

            # ----------------------------------------------------------------
            # Phase 1 — warm-up forward passes (no sampling, no gradient). There is no BatchNorm in the graph; this builds the graph and model.losses before sampling.
            # ----------------------------------------------------------------
            logger.info(f"[calib] Warm-up forward passes over {n_warmup}/{train_batches} batches ({warmup_frac:.0%} of epoch) to build the graph and layer losses before sampling...")
            for batch in train_ds.take(n_warmup):
                x_batch, _, _, _ = batch
                _ = custom_model(x_batch, training=True)

            # ----------------------------------------------------------------
            # Phase 2 — Sample loss magnitudes (CALIB_MODE=value) or per-term gradient norms on
            # the shared trunk (CALIB_MODE=gradient, NT-101). Either way this phase produces one
            # "med_*" measurement per term; phases 3+ (reference, damped rescale, clamp, report)
            # read only those and do not care which measurement produced them.
            # ----------------------------------------------------------------
            if calib_mode == 'gradient':
                # NT-101 QA repair round 1: equalise each term's gradient norm AS IT ENTERS
                # `total` (horizon sums times outer multipliers; see _total_term_tensors), not
                # _term_values' /3-per-horizon-average reading (that reading stays correct for
                # value mode, where it is the pre-existing, unchanged behaviour).
                #
                # Performance: a term that this pass never assigns a weight to (vac,
                # vac_overflow — see _LAMBDA_NAME_OF) is never measured at all; a term whose
                # effective damping is 0, or whose own config weight is 0 (inactive), is rescaled
                # as a no-op regardless of what we measure (rescale_weight(d=0) always returns
                # orig; "measured <= eps" also returns orig), so its backward pass is skipped too
                # ("report the cost of the pass" below is for the terms that are actually
                # equalised). At the CALIB_* defaults this skips ext/t_perp/casimir/hd/ife
                # (CALIB_DAMPING_TREND and CALIB_DAMPING_PHYSICS both default to 0) and always
                # skips vac/vac_overflow, leaving short/point/long/dir/var/vol/crps/ece measured.
                _damping_of_term = {
                    'short': d_point, 'point': d_point, 'long': d_point, 'ext': d_trend,
                    'dir': d_dir, 'var': d_var, 'vol': d_vol, 'crps': d_crps, 'ece': d_ece,
                    't_perp': d_physics, 'casimir': d_physics, 'hd': d_physics, 'ife': d_physics,
                }
                _active_of_term = {
                    'vol': vol_active, 'crps': crps_active, 'ece': ece_active, 't_perp': t_perp_active,
                    'casimir': casimir_active, 'hd': hd_active, 'ife': ife_active,
                }
                _price_terms = ('short', 'point', 'long', 'ext', 'vol', 'casimir', 'ife')
                to_measure = [name for name, d in _damping_of_term.items()
                             if d != 0.0 and _active_of_term.get(name, True)
                             and not (price_off and name in _price_terms)]
                logger.info(f"[calib] Sampling the trunk gradient norm of {sorted(to_measure)} "
                            f"(as each enters `total`) over {n_sample}/{train_batches} batches "
                            f"({sample_frac:.0%} of epoch); skipped (never weighted, or a no-op "
                            f"at damping 0 / inactive): "
                            f"{sorted(set(_damping_of_term) - set(to_measure))} and vac/vac_overflow...")
                trunk_vars = _trunk_variables(custom_model)
                if not trunk_vars:
                    raise RuntimeError("CALIB_MODE=gradient: no shared-trunk variables found "
                                       "(every trainable variable was an indicator or a head)")
                norm_bufs = {name: [] for name in to_measure}
                for batch in train_ds.take(n_sample):
                    x_batch, y_batch, last_batch, ext_batch = batch
                    with tf.GradientTape(persistent=True) as tape:
                        _y_pred_raw = custom_model(x_batch, training=True)
                        (*y_pred_batch, _vac_overflow_batch) = _y_pred_raw
                        loss_components = custom_model.custom_loss(
                            x_batch, y_batch, y_pred_batch, last_batch, ext_batch,
                            vacuum_overflow=_vac_overflow_batch
                        )
                        # NT-101 fix (QA of 119319f): the measured tensors must be built INSIDE
                        # the tape's `with` block. Built outside (an earlier version of this
                        # branch did, via _term_values called after the block had closed), the
                        # +/x ops are not recorded, so tape.gradient() on the resulting (real,
                        # nonzero-valued) tensor finds no path back into the traced graph at all
                        # and returns None for every trunk variable (reproduced with a plain
                        # tf.Variable sum built outside a `with tf.GradientTape()` block: even
                        # x0 + x0 loses its gradient). Single-field terms with no extra arithmetic
                        # were unaffected, which is what made the bug look like batch noise in a
                        # subset of terms rather than a tape-scope bug in every combined one.
                        terms = _total_term_tensors(custom_model, loss_components)
                    for name in to_measure:
                        grads = tape.gradient(terms[name], trunk_vars)
                        present = [g for g in grads if g is not None]
                        gnorm = float(tf.linalg.global_norm(present)) if present else 0.0
                        norm_bufs[name].append(gnorm)
                    del tape  # persistent tapes must be released explicitly

                def _mean(buf):
                    return float(np.mean(np.array(buf))) if buf else 0.0

                med_short, med_point, med_long = (_mean(norm_bufs.get(n, [])) for n in ('short', 'point', 'long'))
                med_ext, med_dir, med_var, med_vol = (_mean(norm_bufs.get(n, [])) for n in ('ext', 'dir', 'var', 'vol'))
                med_crps, med_ece = _mean(norm_bufs.get('crps', [])), _mean(norm_bufs.get('ece', []))
                med_t_perp, med_casimir = (_mean(norm_bufs.get(n, [])) for n in ('t_perp', 'casimir'))
                med_hd, med_ife = (_mean(norm_bufs.get(n, [])) for n in ('hd', 'ife'))
                med_vac = 0.0             # never measured: no weight is ever derived from it
                med_vac_overflow = 0.0    # never measured: not one of the 13 rescaled lambdas
            else:
                logger.info(f"[calib] Sampling loss magnitudes over {n_sample}/{train_batches} batches ({sample_frac:.0%} of epoch)...")
                short_buf, point_buf, long_buf = [], [], []
                ext_buf, dir_buf, var_buf, vol_buf = [], [], [], []
                crps_buf, ece_buf = [], []
                t_perp_buf, casimir_buf, vac_buf, hd_buf, ife_buf, vac_overflow_buf = [], [], [], [], [], []

                for batch in train_ds.take(n_sample):
                    x_batch, y_batch, last_batch, ext_batch = batch
                    _y_pred_raw = custom_model(x_batch, training=True)
                    # Strip 10th output (vacuum_overflow) before passing to custom_loss
                    (*y_pred_batch, _vac_overflow_batch) = _y_pred_raw
                    (total,
                     point_h0, point_h1, point_h2,
                     local_h0, global_h0, ext_h0,
                     local_h1, global_h1, ext_h1,
                     local_h2, global_h2, ext_h2,
                     dir_h0, dir_h1, dir_h2,
                     nll_h0, nll_h1, nll_h2,
                     reg_val, inter_reg, vol_loss_val,
                     crps_h0_c, crps_h1_c, crps_h2_c,
                     soft_ece_h0_c, soft_ece_h1_c, soft_ece_h2_c,
                     t_perp_c, casimir_c, vac_c, hd_c, ife_c,
                     vac_overflow_c, _pnl_c, _dir_align_c, _coherence_c) = custom_model.custom_loss(
                        x_batch, y_batch, y_pred_batch, last_batch, ext_batch,
                        vacuum_overflow=_vac_overflow_batch
                    )

                    short_buf.append(float(point_h0))
                    point_buf.append(float(point_h1))
                    long_buf.append(float(point_h2))
                    ext_buf.append(float((ext_h0 + ext_h1 + ext_h2) / 3.0))
                    dir_buf.append(float((dir_h0 + dir_h1 + dir_h2) / 3.0))
                    var_buf.append(float((nll_h0 + nll_h1 + nll_h2) / 3.0))
                    vol_buf.append(float(vol_loss_val))
                    crps_buf.append(float((crps_h0_c + crps_h1_c + crps_h2_c) / 3.0))
                    ece_buf.append(float((soft_ece_h0_c + soft_ece_h1_c + soft_ece_h2_c) / 3.0))
                    t_perp_buf.append(float(t_perp_c))
                    casimir_buf.append(float(casimir_c))
                    vac_buf.append(float(vac_c))
                    hd_buf.append(float(hd_c))
                    ife_buf.append(float(ife_c))
                    vac_overflow_buf.append(float(vac_overflow_c))

                def _med(buf):
                    return float(np.median(np.array(buf))) if buf else 0.0

                med_short = _med(short_buf)
                med_point = _med(point_buf)
                med_long  = _med(long_buf)
                med_ext   = _med(ext_buf)
                med_dir   = _med(dir_buf)
                med_var   = _med(var_buf)
                med_vol   = _med(vol_buf)
                med_crps  = _med(crps_buf)
                med_ece   = _med(ece_buf)
                med_t_perp  = _med(t_perp_buf)
                med_casimir = _med(casimir_buf)
                med_vac     = _med(vac_buf)
                med_hd      = _med(hd_buf)
                med_ife     = _med(ife_buf)
                med_vac_overflow = _med(vac_overflow_buf)

            # Reference = mean of all active (non-zero) component medians (CALIB_MODE=value) or
            # mean gradient norms (CALIB_MODE=gradient).
            # CRPS and ECE are included only when their config lambda is active (crps_active etc.
            # were computed earlier, before Phase 2, so gradient mode can also use them there).
            candidate_meds = [med_short, med_point, med_long, med_ext, med_dir, med_var]
            if vol_active:
                candidate_meds.append(med_vol)
            if crps_active:
                candidate_meds.append(med_crps)
            if ece_active:
                candidate_meds.append(med_ece)
            if t_perp_active:
                candidate_meds.append(med_t_perp)
            if casimir_active:
                candidate_meds.append(med_casimir)
            if hd_active:
                candidate_meds.append(med_hd)
            if ife_active:
                candidate_meds.append(med_ife)
            vac_overflow_active = float(getattr(cfg, 'LAMBDA_VAC_OVERFLOW', 0.0)) > 0.0
            if vac_overflow_active and med_vac_overflow > 1e-8:
                candidate_meds.append(med_vac_overflow)
            # vac is always added (vacuum bandwidth self-limiting is always active)
            if med_vac > 1e-8:
                candidate_meds.append(med_vac)
            non_zero = [m for m in candidate_meds if m > 1e-8]
            ref_loss = float(np.mean(non_zero)) if non_zero else 1.0

            # ----------------------------------------------------------------
            # Phase 3 — Damped rescaling and clamping
            # ----------------------------------------------------------------
            eps = 1e-8

            def _rescale(orig, med, damping, name='', quiet=False):
                return rescale_weight(orig, med, damping, ref_loss, lam_min, lam_max, eps,
                                      name=name, quiet=quiet)

            if price_off:  # zeroed terms keep their configured weights
                new_short, new_point, new_long, new_ext = orig_short, orig_point, orig_long, orig_ext
            else:
                new_short = _rescale(orig_short, med_short, d_point, 'lambda_short')
                new_point = _rescale(orig_point, med_point, d_point, 'lambda_point')
                new_long  = _rescale(orig_long,  med_long,  d_point, 'lambda_long')
                new_ext   = _rescale(orig_ext,   med_ext,   d_trend, 'lambda_extended_trend')
            new_dir   = _rescale(orig_dir,   med_dir,   d_dir, 'lambda_dir')
            new_var   = _rescale(orig_var,   med_var,   d_var, 'lambda_var')
            new_vol   = (_rescale(orig_vol, med_vol, d_vol, 'lambda_vol',
                                  quiet=(orig_vol == 0.0 and vol_zero_to_floor))
                         if vol_active else orig_vol)
            new_crps  = _rescale(orig_crps,  med_crps,  d_crps, 'lambda_crps') if crps_active else orig_crps
            new_ece   = _rescale(orig_ece,   med_ece,   d_ece, 'lambda_soft_ece')  if ece_active  else orig_ece
            new_t_perp  = _rescale(orig_t_perp,  med_t_perp,  d_physics, 'lambda_t_perp') if t_perp_active  else orig_t_perp
            new_casimir = _rescale(orig_casimir, med_casimir, d_physics, 'lambda_casimir') if casimir_active else orig_casimir
            new_hd      = _rescale(orig_hd,      med_hd,      d_physics, 'lambda_hd') if hd_active      else orig_hd
            new_ife     = _rescale(orig_ife,     med_ife,     d_physics, 'lambda_ife') if ife_active     else orig_ife

            custom_model.lambda_short          = new_short
            custom_model.lambda_point          = new_point
            custom_model.lambda_long           = new_long
            custom_model.lambda_extended_trend = new_ext
            custom_model.lambda_dir            = new_dir
            custom_model.lambda_var            = new_var
            custom_model.lambda_vol            = new_vol
            custom_model.lambda_crps           = new_crps
            custom_model.lambda_soft_ece       = new_ece
            custom_model.lambda_t_perp         = new_t_perp
            custom_model.lambda_casimir        = new_casimir
            custom_model.lambda_hd             = new_hd
            custom_model.lambda_ife            = new_ife

            # Post-calibration weighted gradient norms (CALIB_MODE=gradient only; NT-101 QA repair
            # round 1, acceptance (1)'s "equal after calibration" check and meta.json's "post"
            # record). Computed analytically, not by a second backward pass: _total_term_tensors
            # scales each measured term by a *scalar* (its own weight — reset to 1.0 for Phase 2 —
            # times the current outer multiplier), and gradient is linear in a scalar factor,
            # so ``||grad(new_lambda * term)|| == new_lambda * ||grad(term)||`` exactly (new_lambda
            # >= 0 always, by its [0.1, 20] clamp). This reuses the Phase 2 measurement instead of
            # re-running the sampled batches with the new weights, at no accuracy cost.
            if calib_mode == 'gradient':
                _new_of_term = {
                    'short': new_short, 'point': new_point, 'long': new_long, 'ext': new_ext,
                    'dir': new_dir, 'var': new_var, 'vol': new_vol, 'crps': new_crps, 'ece': new_ece,
                    't_perp': new_t_perp, 'casimir': new_casimir, 'hd': new_hd, 'ife': new_ife,
                }
                _pre_of_term = {
                    'short': med_short, 'point': med_point, 'long': med_long, 'ext': med_ext,
                    'dir': med_dir, 'var': med_var, 'vol': med_vol, 'crps': med_crps, 'ece': med_ece,
                    't_perp': med_t_perp, 'casimir': med_casimir, 'hd': med_hd, 'ife': med_ife,
                }
                post_of_term = {name: _new_of_term[name] * _pre_of_term[name] for name in to_measure}
                # Caveat: this reflects Phase 3's weights. CALIB_OUTER (Phase 4, below, default
                # False) may further change lambda_trend_outer/dir_outer/nll_outer, which would
                # move ext/dir/var's "as it enters total" value again; re-deriving post_of_term
                # after Phase 4 is not done here (CALIB_OUTER=False is the default this item
                # targets; combining the two is left for whoever next changes CALIB_OUTER's design).

            # ----------------------------------------------------------------
            # Phase 4 — Optional outer-multiplier calibration (CALIB_OUTER)
            # Calibrates lambda_trend_outer, lambda_dir_outer, lambda_nll_outer
            # so that the already-rescaled per-component group sums are equalized.
            # Uses same damping logic (d_global) and same clamp bounds.
            # ----------------------------------------------------------------
            if calib_outer:
                med_trend_group = new_ext * med_ext          # post-rescale magnitude proxy
                med_dir_group   = new_dir * med_dir
                med_nll_group   = new_var * med_var
                outer_meds = [m for m in [med_trend_group, med_dir_group, med_nll_group] if m > eps]
                ref_outer = float(np.mean(outer_meds)) if outer_meds else 1.0

                def _rescale_outer(orig_outer, med_g):
                    if med_g > eps:
                        return float(np.clip(orig_outer * (ref_outer / (med_g + eps)) ** d_global, lam_min, lam_max))
                    return orig_outer

                custom_model.lambda_trend_outer = _rescale_outer(custom_model.lambda_trend_outer, med_trend_group)
                custom_model.lambda_dir_outer   = _rescale_outer(custom_model.lambda_dir_outer,   med_dir_group)
                custom_model.lambda_nll_outer   = _rescale_outer(custom_model.lambda_nll_outer,   med_nll_group)

            # ----------------------------------------------------------------
            # Print report
            # ----------------------------------------------------------------
            _measure_label = "||g||" if calib_mode == 'gradient' else "med"

            def _fmt_row(name, orig, med, new, active=True):
                skip = "" if active else " [skipped — inactive]"
                arrow = f"{orig:.4f} → {new:.4f}"
                return f"  {name:<14} {_measure_label}={med:.6f}  {arrow}{skip}"

            logger.info("[calib] Sampled %s and updated lambdas (CALIB_MODE=%s):",
                        "gradient norms" if calib_mode == 'gradient' else "medians", calib_mode)
            logger.info('%s', _fmt_row("λ_short",  orig_short, med_short, new_short))
            logger.info('%s', _fmt_row("λ_point",  orig_point, med_point, new_point))
            logger.info('%s', _fmt_row("λ_long",   orig_long,  med_long,  new_long))
            logger.info('%s', _fmt_row("λ_trend",  orig_ext,   med_ext,   new_ext))
            logger.info('%s', _fmt_row("λ_dir",    orig_dir,   med_dir,   new_dir))
            logger.info('%s', _fmt_row("λ_var",    orig_var,   med_var,   new_var))
            logger.info('%s', _fmt_row("λ_vol",    orig_vol,   med_vol,   new_vol, active=vol_active))
            logger.info('%s', _fmt_row("λ_crps",   orig_crps,  med_crps,  new_crps,  active=crps_active))
            logger.info('%s', _fmt_row("λ_ece",    orig_ece,   med_ece,   new_ece,   active=ece_active))
            logger.info('%s', _fmt_row("λ_t_perp", orig_t_perp,  med_t_perp,  new_t_perp,  active=t_perp_active))
            logger.info('%s', _fmt_row("λ_casimir",orig_casimir, med_casimir, new_casimir, active=casimir_active))
            logger.info('%s', _fmt_row("λ_hd",     orig_hd,      med_hd,      new_hd,      active=hd_active))
            logger.info('%s', _fmt_row("λ_ife",    orig_ife,     med_ife,     new_ife,     active=ife_active))
            lambda_vac_orig = float(getattr(cfg, 'LAMBDA_VAC', 0.0))
            logger.info('%s', _fmt_row("Λ_vac(thr)", lambda_vac_orig, med_vac, lambda_vac_orig, active=True) + "  (threshold, not rescaled)  # P0-2: default now 0 (opt-in)")
            if calib_outer:
                # float(): NT-092 turned these three into tf.Variable-backed properties (screen
                # phase 2 needs them settable without retracing); a bare ResourceVariable has no
                # ':.4f' formatter.
                logger.info(f"  [outer] λ_trend_outer={float(custom_model.lambda_trend_outer):.4f}  "
                      f"λ_dir_outer={float(custom_model.lambda_dir_outer):.4f}  "
                      f"λ_nll_outer={float(custom_model.lambda_nll_outer):.4f}")
            logger.info(f"[calib] ref_loss={ref_loss:.6f}  d_global={d_global}  "
                  f"warmup={n_warmup}/{train_batches}  sample={n_sample}/{train_batches}  clamp=[{lam_min}, {lam_max}]")

            _calib_lambdas = {
                'lambda_short':          new_short,
                'lambda_point':          new_point,
                'lambda_long':           new_long,
                'lambda_extended_trend': new_ext,
                'lambda_dir':            new_dir,
                'lambda_var':            new_var,
                'lambda_vol':            new_vol,
                'lambda_crps':           new_crps,
                'lambda_soft_ece':       new_ece,
                'lambda_t_perp':         new_t_perp,
                'lambda_casimir':        new_casimir,
                'lambda_hd':             new_hd,
                'lambda_ife':            new_ife,
                'ref_loss':              ref_loss,
            }
            if calib_mode == 'gradient':
                # NT-101 acceptance (2): the chosen weights are the lambda_* entries above; also
                # record the mode plus, for every term this pass actually measured and equalised
                # (``to_measure`` — damping 0 and inactive terms are no-ops and were never
                # measured, see Phase 2 above; "equalised", not "rescaled": a damping-0 term is
                # technically still passed through rescale_weight, but as a no-op, so calling it
                # "rescaled" overstates what happened to it), its gradient norm before calibration
                # (lambda reset to 1.0, as it enters `total`) and after (the now-assigned weight
                # times that same measurement — see the "Post-calibration" comment above for why
                # this is exact without a second backward pass), plus each term's share of the
                # pre-calibration total, so a run's meta.json shows what the equalisation pass saw
                # and that it worked (post norms should all be close to ref_loss). The 'calib_mode'
                # key is added only here (not for the default 'value' branch) so every value in the
                # dict stays a plain float there, as scripts/golden_run.py (and any other consumer
                # of calibration_lambdas) assumes.
                _calib_lambdas['calib_mode'] = calib_mode
                _grad_norms_pre = {_LAMBDA_NAME_OF[name]: _pre_of_term[name] for name in to_measure}
                _grad_norms_post = {_LAMBDA_NAME_OF[name]: post_of_term[name] for name in to_measure}
                _total_grad_norm_pre = sum(_grad_norms_pre.values()) or 1.0
                _calib_lambdas['grad_norms_pre'] = _grad_norms_pre
                _calib_lambdas['grad_norms_post'] = _grad_norms_post
                _calib_lambdas['grad_shares'] = {k: v / _total_grad_norm_pre for k, v in _grad_norms_pre.items()}
            if calib_outer:
                # float(): see the log line above (NT-092, these are now tf.Variable-backed).
                _calib_lambdas.update({
                    'lambda_trend_outer': float(custom_model.lambda_trend_outer),
                    'lambda_dir_outer':   float(custom_model.lambda_dir_outer),
                    'lambda_nll_outer':   float(custom_model.lambda_nll_outer),
                })

        except Exception as e:
            import traceback
            # Restore the lambdas that were reset to 1.0 for sampling. Without this, a
            # failure after the reset silently trained with every lambda at 1.0 while
            # the message claimed "default lambdas".
            for _name, _value in _calib_saved.items():
                setattr(custom_model, _name, _value)
            logger.warning(f"[calib] Calibration pass failed — restored configured lambdas and continuing: {e}")
            traceback.print_exc()
            # NT-111: a failure must not look like an ordinary, successful calibration downstream.
            # CALIB_MODE='gradient' always records it (the gradient pass is newer and the one QA
            # caught silently falling back); CALIB_MODE='value' only when CALIB_FAIL_LOUD is set
            # (default False keeps this branch's pre-NT-111 "return None" byte-for-byte).
            fail_loud = calib_mode_for_failure == 'gradient' or bool(getattr(cfg, 'CALIB_FAIL_LOUD', False))
            if fail_loud:
                _calib_lambdas = dict(_calib_saved)
                _calib_lambdas['calib_failed'] = True
                _calib_lambdas['calib_mode'] = calib_mode_for_failure
                _calib_lambdas['calib_error'] = {'type': type(e).__name__, 'message': str(e)}

    return _calib_lambdas
