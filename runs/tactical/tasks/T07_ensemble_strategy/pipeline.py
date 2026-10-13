"""T07: builds the six members' minute positions (+ the HAR forecast) for a stage.
  sel / look : the cut main file, monthly walk-forward for M1 - M3, HAR refitted monthly; months 2018-12 (threshold month) .. last.
  fwd        : the Binance forward file; M1 - M3 and HAR frozen as fitted on data to 2025-07-20 23:59; thresholds / sigma* from 2025-10.
Caches go to D:/nt/nt_t07_cache (heavy arrays, not tracked)."""
import concurrent.futures as cf
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import t07_lib as T
import members as MB

FIRST_PRED_MONTH = "2018-12"


def log(*a):
    print(*a, flush=True)


def m1_features(raw, tag):
    import m1_core
    path = f"{T.CACHE}/m1_feat_{tag}.npz"
    if not os.path.exists(path):
        t0 = time.time(); f = m1_core.features(raw)
        np.savez(path, **f); log(f"  M1 features {tag}: {len(f['t_sig'])} anchors, {time.time() - t0:.0f}s")
    z = np.load(path)
    return {k: z[k] for k in z.files}, path


def m1_months(feat_path, months, workers):
    import m1_core
    os.makedirs(f"{T.CACHE}/m1", exist_ok=True)
    jobs = [(feat_path, m, f"{T.CACHE}/m1/{m}.npz") for m in months if not os.path.exists(f"{T.CACHE}/m1/{m}.npz")]
    if jobs:
        t0 = time.time()
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for m in ex.map(m1_core.month_task, jobs):
                log(f"  M1 month {m} done {time.time() - t0:.0f}s")
    return {m: f"{T.CACHE}/m1/{m}.npz" for m in months}


def build_main(tag, end_excl, last_month, workers=6):
    """Positions on the main-file grid from its first full hour to the end of the data. Returns a dict."""
    os.makedirs(T.CACHE, exist_ok=True)
    raw = T.read_main(end_excl)
    if end_excl is not None:
        assert raw.index.max() < pd.Timestamp(end_excl)
    grid = T.Grid(raw); log(f"[{tag}] bars {len(raw)}, grid {grid.n} minutes ({grid.n_missing} filled), last {raw.index.max()}")
    months = [str(m) for m in pd.period_range(FIRST_PRED_MONTH, last_month, freq="M")]
    # M1
    feat, fpath = m1_features(raw, tag)
    mfiles = m1_months(fpath, months, workers)
    pos1, meta1 = MB.m1_position(grid, feat, mfiles, months)
    # M2, M3
    F = MB.hourly_anchors(raw)
    p2 = MB.hourly_month_preds(F, MB.H2, "logreg", 730, months); pos2, meta2 = MB.hourly_member_position(grid, F, p2, MB.H2, months)
    p3 = MB.hourly_month_preds(F, MB.H3, "l1", 365, months); pos3, meta3 = MB.hourly_member_position(grid, F, p3, MB.H3, months)
    # M4 - M6
    pos456 = MB.rule_positions(grid)
    # HAR
    sig, med = MB.har_monthly(grid, ["2019-01", last_month])
    pos = np.stack([pos1, pos2, pos3] + pos456).astype(np.int8)
    log(f"[{tag}] trades M1 {meta1['n_trades']}, M2 {meta2['n_trades']}, M3 {meta3['n_trades']}")
    return dict(grid=grid, pos=pos, sig=sig, med=med, meta=dict(M1=meta1, M2=meta2, M3=meta3), months=months, m1_feat=feat, m1_feat_path=fpath)


def build_fwd(main_look, workers=4):
    """Forward positions on the forward grid with frozen models. `main_look` is the look-stage build (provides the training features and the
    main grid for the frozen HAR). Calibration month 2025-10 gives the V_prev thresholds and sigma*; scored months 2025-11 .. 2026-09."""
    import m1_core
    raw = T.read_fwd()
    assert raw.index.min() >= T.FWD_START and raw.index.max() <= T.FWD_LAST
    assert not ((raw.index >= T.NEVER_LO) & (raw.index <= T.NEVER_HI)).any()
    grid = T.Grid(raw); log(f"[fwd] bars {len(raw)}, grid {grid.n} minutes ({grid.n_missing} filled), {raw.index.min()} .. {raw.index.max()}")
    months = [T.FWD_CAL] + T.FWD_SCORED
    end = pd.Timestamp(T.C.CUTOFF) + pd.Timedelta(minutes=1)                   # 2025-07-21 00:00: training labels closed by it
    # M1 frozen
    feat, fpath = m1_features(raw, "fwd")
    out = f"{T.CACHE}/m1/frozen.npz"
    m1_core.frozen_task((main_look["m1_feat_path"], str(end), fpath, out))
    P = np.load(out)["P"]
    pos1, meta1 = MB.m1_position_frozen(grid, feat, P, months)
    # M2, M3 frozen
    F_tr = MB.hourly_anchors(T.read_main(None)); F = MB.hourly_anchors(raw)
    ms = {}
    for nm, H, kind, Td in (("M2", MB.H2, "logreg", 730), ("M3", MB.H3, "l1", 365)):
        pred, ntr, last_lab = MB.hourly_frozen(F_tr, H, kind, Td, end)
        p = pred(F["X"]); month = MB.month_labels(F["t_sig"])
        conf = {m: np.abs(p[month == m] - 0.5) for m in months}
        thr = T.month_thresholds(conf, months)
        t_entry = (pd.DatetimeIndex(F["t_sig"]) + pd.Timedelta(hours=1)).values.astype("datetime64[m]")
        v = F["valid"]
        pos, tr = MB.trades_to_position(grid, p[v], t_entry[v], month[v], thr, H * 60)
        ms[nm] = (pos, dict(thr=thr, n_trades=len(tr[2]), n_train=ntr, last_label_end=last_lab))
    pos456 = MB.rule_positions(grid)
    sig, sig_star = MB.har_frozen(main_look["grid"], grid, end, T.FWD_CAL)
    pos = np.stack([pos1, ms["M2"][0], ms["M3"][0]] + pos456).astype(np.int8)
    log(f"[fwd] trades M1 {meta1['n_trades']}, M2 {ms['M2'][1]['n_trades']}, M3 {ms['M3'][1]['n_trades']}; sigma* {sig_star:.3e}")
    return dict(grid=grid, pos=pos, sig=sig, sig_star=sig_star, meta=dict(M1=meta1, M2=ms["M2"][1], M3=ms["M3"][1]), months=months)
