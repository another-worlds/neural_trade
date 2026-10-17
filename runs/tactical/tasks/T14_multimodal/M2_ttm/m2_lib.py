"""M2 TTM r2 view: torch-free helpers (inputs, columns, causal debias). Reuses M1's 512-bar five-minute log-RV windows."""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN
_spec = importlib.util.spec_from_file_location("m1_inputs", os.path.join(HERE, "..", "M1_chronos2", "inputs.py"))
IN = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(IN)
CTX, HOR, EPS = IN.CTX, IN.HOR, AN.EPS_RV


def context(frame, ta):
    """Last 512 five-minute ln(rv5+1e-12) bars before ta (M1's window; None if < 256 finite). NaNs are forward-filled then back-filled inside the window.
    Returns (idx, x) with x float64 [512]."""
    w = IN.window(frame, ta)
    if w is None or len(w["target"]) < CTX:
        return None
    x = pd.Series(w["target"]).ffill().bfill().to_numpy(float)
    return w["idx"], x


def normalise(x):
    """x [n, 512] -> (z, mu, sd) per-window standardisation (sd floored)."""
    mu = x.mean(1, keepdims=True); sd = x.std(1, keepdims=True) + 1e-6
    return (x - mu) / sd, mu, sd


def hourly_from_bars(bars):
    return IN.hourly_from_bars(bars)


def to_columns(fc, last_obs, debias):
    """fc [n, 12] forecast ln(rv5) bars (denormalised), last_obs [n] last observed bar, debias scalar -> view columns (spec names)."""
    raw = hourly_from_bars(fc)
    return {"ttm_mean12": fc.mean(1), "ttm_last": fc[:, -1], "ttm_minus_lastbar": fc.mean(1) - last_obs,
            "ttm_logsum_raw": raw, "ttm_logsum": raw + debias, "point_log_rv": raw + debias}


def debias_constant(ta, implied_raw, y, ok, cap_before=None):
    """mean(target - implied) over TRAINING anchors only: the first 365 days of the source (cut: 2023-06-20.., all before Look 1), but never past `cap_before`
    (for fwd the first 365 days would be the whole look, so the cap = start of Look 2's first month). Rows with ok False or non-finite are excluded."""
    ta = pd.DatetimeIndex(ta); lo = ta.min()
    hi = lo + pd.Timedelta(days=365)
    if cap_before is not None:
        hi = min(hi, pd.Timestamp(cap_before))
    m = np.asarray((ta >= lo) & (ta < hi)) & np.asarray(ok, bool) & np.isfinite(implied_raw) & np.isfinite(y)
    return float(np.mean(np.asarray(y)[m] - np.asarray(implied_raw)[m])), int(m.sum()), (str(lo), str(hi))
