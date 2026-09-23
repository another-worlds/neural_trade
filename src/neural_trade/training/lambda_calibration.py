"""Pre-training loss-weight calibration (extracted from train_and_evaluate in B10).

Measures the natural magnitude of each loss component over a few batches with every weight at
1.0, then rescales the weights so the components start on a common scale:

    new_lambda = clip(orig_lambda * (ref / median_component) ** damping, CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX)

with ``ref`` the mean of the non-zero medians. Components with damping 0 (the trend prior and
the bounded physics regularisers, by default) keep their configured weight. If anything fails,
the configured weights are restored and ``None`` is returned.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calibrate_loss_weights(custom_model, train_ds, cfg, n_train: int) -> Optional[Dict[str, float]]:
    """Run the calibration pass on ``custom_model``; return the calibrated weights (or None)."""
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
            d_global   = float(getattr(cfg, 'CALIB_DAMPING', getattr(cfg, 'DAMPING', 0.5)))
            if 'DAMPING' in dir(cfg) and not hasattr(cfg, 'CALIB_DAMPING'):
                logging.getLogger(__name__).warning("Config.DAMPING is legacy (P1-1); prefer CALIB_DAMPING")
            calib_outer = bool(getattr(cfg, 'CALIB_OUTER', False))

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

            # ----------------------------------------------------------------
            # Phase 1 — warm-up forward passes (no sampling, no gradient). There is no BatchNorm in the graph; this builds the graph and model.losses before sampling.
            # ----------------------------------------------------------------
            logger.info(f"[calib] Warm-up forward passes over {n_warmup}/{train_batches} batches ({warmup_frac:.0%} of epoch) to build the graph and layer losses before sampling...")
            for batch in train_ds.take(n_warmup):
                x_batch, _, _, _ = batch
                _ = custom_model(x_batch, training=True)

            # ----------------------------------------------------------------
            # Phase 2 — Sample loss magnitudes
            # ----------------------------------------------------------------
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
                 vac_overflow_c) = custom_model.custom_loss(
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

            # Reference = mean of all active (non-zero) component medians.
            # CRPS and ECE are included only when their config lambda is active.
            crps_active = float(getattr(cfg, 'LAMBDA_CRPS', 0.0)) > 0.0
            ece_active  = float(getattr(cfg, 'LAMBDA_SOFT_ECE', 0.0)) > 0.0
            t_perp_active  = float(getattr(cfg, 'LAMBDA_T_PERP',  0.0)) > 0.0
            casimir_active = float(getattr(cfg, 'LAMBDA_CASIMIR', 0.0)) > 0.0
            hd_active      = float(getattr(cfg, 'LAMBDA_HD',      0.0)) > 0.0
            ife_active     = float(getattr(cfg, 'LAMBDA_IFE',     0.0)) > 0.0
            candidate_meds = [med_short, med_point, med_long, med_ext, med_dir, med_var, med_vol]
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

            def _rescale(orig, med, damping):
                if med > eps:
                    return float(np.clip(orig * (ref_loss / (med + eps)) ** damping, lam_min, lam_max))
                return orig  # component inactive — keep original

            new_short = _rescale(orig_short, med_short, d_point)
            new_point = _rescale(orig_point, med_point, d_point)
            new_long  = _rescale(orig_long,  med_long,  d_point)
            new_ext   = _rescale(orig_ext,   med_ext,   d_trend)
            new_dir   = _rescale(orig_dir,   med_dir,   d_dir)
            new_var   = _rescale(orig_var,   med_var,   d_var)
            new_vol   = _rescale(orig_vol,   med_vol,   d_vol)
            new_crps  = _rescale(orig_crps,  med_crps,  d_crps) if crps_active else orig_crps
            new_ece   = _rescale(orig_ece,   med_ece,   d_ece)  if ece_active  else orig_ece
            new_t_perp  = _rescale(orig_t_perp,  med_t_perp,  d_physics) if t_perp_active  else orig_t_perp
            new_casimir = _rescale(orig_casimir, med_casimir, d_physics) if casimir_active else orig_casimir
            new_hd      = _rescale(orig_hd,      med_hd,      d_physics) if hd_active      else orig_hd
            new_ife     = _rescale(orig_ife,     med_ife,     d_physics) if ife_active     else orig_ife

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
            def _fmt_row(name, orig, med, new, active=True):
                skip = "" if active else " [skipped — inactive]"
                arrow = f"{orig:.4f} → {new:.4f}"
                return f"  {name:<14} med={med:.6f}  {arrow}{skip}"

            logger.info("[calib] Sampled medians and updated lambdas:")
            logger.info('%s', _fmt_row("λ_short",  orig_short, med_short, new_short))
            logger.info('%s', _fmt_row("λ_point",  orig_point, med_point, new_point))
            logger.info('%s', _fmt_row("λ_long",   orig_long,  med_long,  new_long))
            logger.info('%s', _fmt_row("λ_trend",  orig_ext,   med_ext,   new_ext))
            logger.info('%s', _fmt_row("λ_dir",    orig_dir,   med_dir,   new_dir))
            logger.info('%s', _fmt_row("λ_var",    orig_var,   med_var,   new_var))
            logger.info('%s', _fmt_row("λ_vol",    orig_vol,   med_vol,   new_vol))
            logger.info('%s', _fmt_row("λ_crps",   orig_crps,  med_crps,  new_crps,  active=crps_active))
            logger.info('%s', _fmt_row("λ_ece",    orig_ece,   med_ece,   new_ece,   active=ece_active))
            logger.info('%s', _fmt_row("λ_t_perp", orig_t_perp,  med_t_perp,  new_t_perp,  active=t_perp_active))
            logger.info('%s', _fmt_row("λ_casimir",orig_casimir, med_casimir, new_casimir, active=casimir_active))
            logger.info('%s', _fmt_row("λ_hd",     orig_hd,      med_hd,      new_hd,      active=hd_active))
            logger.info('%s', _fmt_row("λ_ife",    orig_ife,     med_ife,     new_ife,     active=ife_active))
            lambda_vac_orig = float(getattr(cfg, 'LAMBDA_VAC', 0.0))
            logger.info('%s', _fmt_row("Λ_vac(thr)", lambda_vac_orig, med_vac, lambda_vac_orig, active=True) + "  (threshold, not rescaled)  # P0-2: default now 0 (opt-in)")
            if calib_outer:
                logger.info(f"  [outer] λ_trend_outer={custom_model.lambda_trend_outer:.4f}  "
                      f"λ_dir_outer={custom_model.lambda_dir_outer:.4f}  "
                      f"λ_nll_outer={custom_model.lambda_nll_outer:.4f}")
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
            if calib_outer:
                _calib_lambdas.update({
                    'lambda_trend_outer': custom_model.lambda_trend_outer,
                    'lambda_dir_outer':   custom_model.lambda_dir_outer,
                    'lambda_nll_outer':   custom_model.lambda_nll_outer,
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

    return _calib_lambdas
