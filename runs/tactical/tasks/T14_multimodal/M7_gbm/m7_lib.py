"""T14 candidate M7: a nonlinear regime stage. CPU only; run with D:/nt/nt_mm_env/python.exe.

Monthly walk-forward over the 12 M6 regime keys. For every calendar month m (UTC) a model is fitted on the POOL of month m: anchors j with
  ok_j, finite keys / labels,  m_start - 365 days <= ta_j  and  ta_j + 60 min <= m_start   (label window ended before the month starts),
and predicts the anchors of month m. So a prediction never sees a label of its own month or later; the same procedure gives the view for the
earlier anchors, so the harness fits its ridge / logistic on honest out-of-month predictions. A month is served only when the pool has >= MIN_POOL anchors.

  gbm_fit_predict    LightGBM (num_leaves 15, 200 rounds, lr 0.05, min_data_in_leaf 200, bagging 0.8), regressor for A or classifier for B
  nw_fit_predict     Nadaraya-Watson kernel smoother, Gaussian kernel on the pool-standardised inputs, bandwidth by leave-one-month-out CV in the pool
"""
import ctypes
import os

import numpy as np
import pandas as pd

MIN_POOL = 5000
POOL_DAYS = 365
LABEL_MIN = 60
H_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CV_Q = 1500


def below_normal():
    """lower this process' CPU priority (Windows)."""
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def gbm_params(seed=0):
    return dict(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1,
                random_state=seed, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)


def gbm_fit_predict(Xtr, ytr, Xte, binary, seed=0):
    import lightgbm as lgb
    m = (lgb.LGBMClassifier if binary else lgb.LGBMRegressor)(**gbm_params(seed))
    m.fit(Xtr, ytr)
    f = (lambda X: m.predict_proba(X)[:, 1]) if binary else m.predict
    return f(Xte), f(Xtr), m


def _d2(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def _nw(D2, y, h, allow=None):
    """Nadaraya-Watson prediction from a squared-distance matrix [q, n]; `allow` [q, n] bool masks pairs out. Row-shifted for stability."""
    if allow is not None:
        D2 = np.where(allow, D2, np.inf)
    sh = D2.min(1, keepdims=True)
    W = np.exp(-(D2 - sh) / (2.0 * h * h)); s = W.sum(1)
    return (W @ y) / s


def nw_fit_predict(Xtr, ytr, Xte, binary, month_tr, seed=0):
    """returns (pred_te, pred_tr in-sample (leave-own-row-out is not used: plain in-sample smoother, only for the rank copy), info)."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    A = ((Xtr - mu) / sd).astype(np.float64); T = ((Xte - mu) / sd).astype(np.float64)
    rng = np.random.default_rng(seed); q = rng.choice(len(A), size=min(CV_Q, len(A)), replace=False)
    D2 = _d2(A[q], A); allow = month_tr[q][:, None] != month_tr[None, :]          # leave-one-month-out: never a neighbour of the query's own month
    errs = []
    for h in H_GRID:
        p = _nw(D2, ytr, h, allow); errs.append(float(np.mean((p - ytr[q]) ** 2)))
    h = H_GRID[int(np.argmin(errs))]
    pte = _nw(_d2(T, A), ytr, h)
    ptr = _nw(_d2(A[q], A), ytr, h)                                                # rank reference: smoother values on the CV sample
    return pte, ptr, {"h": h, "cv_mse": errs}


def rank_copy(pte, ptr):
    """empirical CDF of the prediction among the model's own training-pool predictions (causal: uses the same fit only)."""
    s = np.sort(ptr)
    return np.searchsorted(s, pte, side="right") / max(len(s), 1)


def monthly_view(ta, X, y, ok, source=None, same_source=False, binary=False, kind="gbm", min_pool=MIN_POOL, log=None, return_models=False):
    """Walk-forward over calendar months. `ta` DatetimeIndex (sorted or not), X [N, d] inputs, y [N] labels, ok [N] bool (pool and prediction mask,
    inputs and label finite). Returns (pred [N], rank [N], info per month); NaN where a row is not served."""
    ta = pd.DatetimeIndex(ta); N = len(ta); pred = np.full(N, np.nan); rnk = np.full(N, np.nan); info = []; models = {}
    per = np.asarray(ta.to_period("M")); end = ta + pd.Timedelta(minutes=LABEL_MIN); mon_id = np.asarray(ta.year * 12 + ta.month)
    for mo in sorted(set(per[ok])):
        m0 = mo.start_time; te = np.where(ok & (per == mo))[0]
        tr_mask = ok & (ta >= m0 - pd.Timedelta(days=POOL_DAYS)) & (end <= m0)
        if same_source:
            tr_mask &= (source == source[te[0]])
        tr = np.where(tr_mask)[0]
        if len(tr) < min_pool:
            info.append({"month": str(mo), "served": False, "n_pool": int(len(tr)), "n_te": int(len(te))}); continue
        if kind == "gbm":
            pte, ptr, mdl = gbm_fit_predict(X[tr], y[tr], X[te], binary); ex = {}
            if return_models:
                models[str(mo)] = mdl
        else:
            pte, ptr, ex = nw_fit_predict(X[tr], y[tr], X[te], binary, mon_id[tr])
        pred[te] = pte; rnk[te] = rank_copy(pte, ptr)
        info.append({"month": str(mo), "served": True, "n_pool": int(len(tr)), "n_te": int(len(te)), **ex})
        if log:
            log(f"    {mo} pool {len(tr)} te {len(te)} {ex.get('h', '')}")
    return (pred, rnk, info, models) if return_models else (pred, rnk, info)
