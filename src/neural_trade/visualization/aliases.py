"""Plot-friendly aliases for Keras epoch logs (moved from model.py in B12)."""
from __future__ import annotations


def _first_present(mapping, keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def _sum_present(mapping, keys):
    total = None
    for k in keys:
        if k not in mapping or mapping[k] is None:
            continue
        total = mapping[k] if total is None else (total + mapping[k])
    return total


def _mean_present(mapping, keys):
    total = None
    count = 0
    for k in keys:
        if k not in mapping or mapping[k] is None:
            continue
        total = mapping[k] if total is None else (total + mapping[k])
        count += 1
    if total is None or count == 0:
        return None
    return total / float(count)


def add_plot_aliases(logs, primary_horizon="h1", prefer_gauss=True):
    """Add plotting-friendly aliases into a Keras `logs` dict.

    The training code emits per-horizon metrics (e.g., `val_dir_f1_h1`, `train_dir_acc_h1`)
    and uses `nll_loss` for variance NLL. The notebook historically plotted legacy keys
    like `val_f1`, `val_dir_acc`, `var_nll`, and aggregated `*_trend_loss` fields.

    This helper keeps plotting code stable by:
    - Mapping per-horizon direction metrics to horizon-agnostic keys.
    - Computing legacy aggregate trend-loss keys from available components.
    - Aliasing `var_nll` -> `nll_loss`.

    It is safe to call on batch logs or epoch logs.
    """
    if logs is None:
        return {}
    out = dict(logs)

    def set_if_missing(key, value):
        if key not in out and value is not None:
            out[key] = value

    def set_agg(key, value):
        """Set aggregate-style aliases robustly.
        If value is None, remove any stale key from output to avoid carrying old values forward.
        Converts tensors/arrays to scalar floats by taking mean when needed.
        """
        if value is None:
            # Remove stale aggregate if no data present
            out.pop(key, None)
            return
        # Normalize to a scalar float when possible
        try:
            # TensorFlow tensors
            if hasattr(value, "numpy"):
                v = value.numpy()
            else:
                v = value
            v = np.asarray(v)
            if v.size == 1:
                out[key] = float(v.item())
            else:
                out[key] = float(np.mean(v))
        except Exception:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = value

    # --- Loss aliases (legacy plotting names) ---
    set_if_missing("var_nll", _first_present(out, ["nll_loss"]))
    set_if_missing("val_var_nll", _first_present(out, ["val_nll_loss"]))

    set_if_missing("local_trend_loss", _sum_present(out, ["local_h0", "local_h1", "local_h2"]))
    set_if_missing("val_local_trend_loss", _sum_present(out, ["val_local_h0", "val_local_h1", "val_local_h2"]))

    set_if_missing("global_trend_loss", _sum_present(out, ["global_h0", "global_h1", "global_h2"]))
    set_if_missing("val_global_trend_loss", _sum_present(out, ["val_global_h0", "val_global_h1", "val_global_h2"]))

    set_if_missing("extended_trend_loss", _sum_present(out, ["extended_h0", "extended_h1", "extended_h2"]))
    set_if_missing("val_extended_trend_loss", _sum_present(out, ["val_extended_h0", "val_extended_h1", "val_extended_h2"]))

    # --- Direction metric aliases (primary horizon, head vs gauss preference) ---
    train_pref = "train_gauss_" if prefer_gauss else "train_"
    train_fallback = "train_" if prefer_gauss else "train_gauss_"

    val_pref = "val_gauss_" if prefer_gauss else "val_"
    val_fallback = "val_" if prefer_gauss else "val_gauss_"

    # Average across horizons (h0/h1/h2)
    horizons = ("h0", "h1", "h2")

    train_acc_keys = [f"{train_pref}dir_acc_{h}" for h in horizons]
    train_f1_keys = [f"{train_pref}dir_f1_{h}" for h in horizons]
    train_sens_keys = [f"{train_pref}dir_sensitivity_{h}" for h in horizons]
    train_spec_keys = [f"{train_pref}dir_specificity_{h}" for h in horizons]

    train_acc_fb = [f"{train_fallback}dir_acc_{h}" for h in horizons]
    train_f1_fb = [f"{train_fallback}dir_f1_{h}" for h in horizons]
    train_sens_fb = [f"{train_fallback}dir_sensitivity_{h}" for h in horizons]
    train_spec_fb = [f"{train_fallback}dir_specificity_{h}" for h in horizons]

    val_acc_keys = [f"{val_pref}dir_acc_{h}" for h in horizons]
    val_f1_keys = [f"{val_pref}dir_f1_{h}" for h in horizons]
    val_sens_keys = [f"{val_pref}dir_sensitivity_{h}" for h in horizons]
    val_spec_keys = [f"{val_pref}dir_specificity_{h}" for h in horizons]

    val_acc_fb = [f"{val_fallback}dir_acc_{h}" for h in horizons]
    val_f1_fb = [f"{val_fallback}dir_f1_{h}" for h in horizons]
    val_sens_fb = [f"{val_fallback}dir_sensitivity_{h}" for h in horizons]
    val_spec_fb = [f"{val_fallback}dir_specificity_{h}" for h in horizons]

    # MCC, Brier, ECE keys for averaging
    train_mcc_keys = [f"{train_pref}dir_mcc_{h}" for h in horizons]
    train_mcc_fb = [f"{train_fallback}dir_mcc_{h}" for h in horizons]
    train_brier_keys = [f"{train_pref}dir_brier_{h}" for h in horizons]
    train_brier_fb = [f"{train_fallback}dir_brier_{h}" for h in horizons]
    train_ece_keys = [f"{train_pref}dir_ece_{h}" for h in horizons]
    train_ece_fb = [f"{train_fallback}dir_ece_{h}" for h in horizons]
    # Balanced Accuracy keys for averaging
    train_bal_acc_keys = [f"{train_pref}dir_bal_acc_{h}" for h in horizons]
    train_bal_acc_fb = [f"{train_fallback}dir_bal_acc_{h}" for h in horizons]

    val_mcc_keys = [f"{val_pref}dir_mcc_{h}" for h in horizons]
    val_mcc_fb = [f"{val_fallback}dir_mcc_{h}" for h in horizons]
    val_brier_keys = [f"{val_pref}dir_brier_{h}" for h in horizons]
    val_brier_fb = [f"{val_fallback}dir_brier_{h}" for h in horizons]
    val_ece_keys = [f"{val_pref}dir_ece_{h}" for h in horizons]
    val_ece_fb = [f"{val_fallback}dir_ece_{h}" for h in horizons]
    # Balanced Accuracy keys for validation
    val_bal_acc_keys = [f"{val_pref}dir_bal_acc_{h}" for h in horizons]
    val_bal_acc_fb = [f"{val_fallback}dir_bal_acc_{h}" for h in horizons]

    set_agg("dir_acc_avg", _first_present({"v": _mean_present(out, train_acc_keys), "v2": _mean_present(out, train_acc_fb)}, ["v", "v2"]))
    set_agg("f1_avg", _first_present({"v": _mean_present(out, train_f1_keys), "v2": _mean_present(out, train_f1_fb)}, ["v", "v2"]))
    set_agg("dir_sensitivity_avg", _first_present({"v": _mean_present(out, train_sens_keys), "v2": _mean_present(out, train_sens_fb)}, ["v", "v2"]))
    set_agg("dir_specificity_avg", _first_present({"v": _mean_present(out, train_spec_keys), "v2": _mean_present(out, train_spec_fb)}, ["v", "v2"]))
    # MCC, Brier, ECE averages (class-imbalance robust metrics)
    set_agg("mcc_avg", _first_present({"v": _mean_present(out, train_mcc_keys), "v2": _mean_present(out, train_mcc_fb)}, ["v", "v2"]))
    set_agg("brier_avg", _first_present({"v": _mean_present(out, train_brier_keys), "v2": _mean_present(out, train_brier_fb)}, ["v", "v2"]))
    set_agg("ece_avg", _first_present({"v": _mean_present(out, train_ece_keys), "v2": _mean_present(out, train_ece_fb)}, ["v", "v2"]))
    # Balanced Accuracy average (class-imbalance robust, 50% = random, range [0,1])
    set_agg("bal_acc_avg", _first_present({"v": _mean_present(out, train_bal_acc_keys), "v2": _mean_present(out, train_bal_acc_fb)}, ["v", "v2"]))

    set_agg("val_dir_acc_avg", _first_present({"v": _mean_present(out, val_acc_keys), "v2": _mean_present(out, val_acc_fb)}, ["v", "v2"]))
    set_agg("val_f1_avg", _first_present({"v": _mean_present(out, val_f1_keys), "v2": _mean_present(out, val_f1_fb)}, ["v", "v2"]))
    set_agg("val_dir_sensitivity_avg", _first_present({"v": _mean_present(out, val_sens_keys), "v2": _mean_present(out, val_sens_fb)}, ["v", "v2"]))
    set_agg("val_dir_specificity_avg", _first_present({"v": _mean_present(out, val_spec_keys), "v2": _mean_present(out, val_spec_fb)}, ["v", "v2"]))
    # Validation MCC, Brier, ECE averages
    set_agg("val_mcc_avg", _first_present({"v": _mean_present(out, val_mcc_keys), "v2": _mean_present(out, val_mcc_fb)}, ["v", "v2"]))
    set_agg("val_brier_avg", _first_present({"v": _mean_present(out, val_brier_keys), "v2": _mean_present(out, val_brier_fb)}, ["v", "v2"]))
    set_agg("val_ece_avg", _first_present({"v": _mean_present(out, val_ece_keys), "v2": _mean_present(out, val_ece_fb)}, ["v", "v2"]))
    # Validation Balanced Accuracy average
    set_agg("val_bal_acc_avg", _first_present({"v": _mean_present(out, val_bal_acc_keys), "v2": _mean_present(out, val_bal_acc_fb)}, ["v", "v2"]))

    # Batch-level aliases used by batch plot
    set_if_missing(
        "dir_acc",
        _first_present(
            out,
            [
                f"{train_pref}dir_acc_{primary_horizon}",
                f"{train_fallback}dir_acc_{primary_horizon}",
                f"dir_acc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "f1",
        _first_present(
            out,
            [
                f"{train_pref}dir_f1_{primary_horizon}",
                f"{train_fallback}dir_f1_{primary_horizon}",
                f"dir_f1_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "dir_mcc",
        _first_present(
            out,
            [
                f"{train_pref}dir_mcc_{primary_horizon}",
                f"{train_fallback}dir_mcc_{primary_horizon}",
                f"dir_mcc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "dir_sensitivity",
        _first_present(out, [f"{train_pref}dir_sensitivity_{primary_horizon}", f"{train_fallback}dir_sensitivity_{primary_horizon}"]),
    )
    set_if_missing(
        "dir_specificity",
        _first_present(out, [f"{train_pref}dir_specificity_{primary_horizon}", f"{train_fallback}dir_specificity_{primary_horizon}"]),
    )

    # Epoch-level aliases used by validation metrics plot
    set_if_missing(
        "val_dir_acc",
        _first_present(
            out,
            [
                f"{val_pref}dir_acc_{primary_horizon}",
                f"{val_fallback}dir_acc_{primary_horizon}",
                f"dir_acc_{primary_horizon}",
                "val_dir_acc",
            ],
        ),
    )
    set_if_missing(
        "val_f1",
        _first_present(
            out,
            [
                f"{val_pref}dir_f1_{primary_horizon}",
                f"{val_fallback}dir_f1_{primary_horizon}",
                f"dir_f1_{primary_horizon}",
                "val_f1",
            ],
        ),
    )
    set_if_missing(
        "val_dir_mcc",
        _first_present(
            out,
            [
                f"{val_pref}dir_mcc_{primary_horizon}",
                f"{val_fallback}dir_mcc_{primary_horizon}",
                f"dir_mcc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "val_dir_sensitivity",
        _first_present(
            out,
            [
                f"{val_pref}dir_sensitivity_{primary_horizon}",
                f"{val_fallback}dir_sensitivity_{primary_horizon}",
                f"dir_sensitivity_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "val_dir_specificity",
        _first_present(
            out,
            [
                f"{val_pref}dir_specificity_{primary_horizon}",
                f"{val_fallback}dir_specificity_{primary_horizon}",
                f"dir_specificity_{primary_horizon}",
            ],
        ),
    )

    # Back-compat: treat recall as sensitivity for UP class
    set_if_missing("val_recall", out.get("val_dir_sensitivity"))

    # Prefer avg metrics for legacy val_* keys if present
    set_if_missing("val_f1", out.get("val_f1_avg"))
    set_if_missing("val_dir_acc", out.get("val_dir_acc_avg"))
    set_if_missing("val_dir_sensitivity", out.get("val_dir_sensitivity_avg"))
    set_if_missing("val_dir_specificity", out.get("val_dir_specificity_avg"))

    # --- QBOX / T_⊥ aggregate alias ---
    # Sum all active T_⊥ loss components so the dashboard can plot a single trend line.
    # Components with lambda=0 contribute 0, so this is safe even when losses are off.
    _qbox_train_keys = ['t_perp_loss', 'casimir_loss', 'vac_loss', 'hd_loss', 'ife_loss', 'vac_overflow_loss']
    _qbox_val_keys   = [f'val_{k}' for k in _qbox_train_keys]
    _qbox_train_sum  = sum(float(out[k]) for k in _qbox_train_keys if k in out)
    _qbox_val_sum    = sum(float(out[k]) for k in _qbox_val_keys   if k in out)
    set_if_missing('qbox_loss',     _qbox_train_sum if _qbox_train_sum > 0 else None)
    set_if_missing('val_qbox_loss', _qbox_val_sum   if _qbox_val_sum   > 0 else None)

    return out

