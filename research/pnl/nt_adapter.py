"""neural_trade adapter: fit the fixed model.py (SPEC model_py_fixes M1-M11) per fold and write causal forecasts.

Folds (UTC; SPEC walk_forward): train on the trailing 12 months before the test start (never before 2020-02-01,
the end of the warm-up month), minus a 1-day purge; model.train_and_evaluate() takes the last 15% of that range
as its validation block (purged 15 bars) for early stopping. WINDOW_STEP 20, EPOCHS 10, 1 seed.
  Y2021..Y2024: test = the calendar year;  Y2025: test 2025-01-01..2025-10-31;  S: bundle only (seal later).

Outputs under $PNL_WORK/nt (PNL_WORK defaults to research/pnl/work, gitignored):
  <fold>.npz       t (unix s of the decision bar), mu (N,3) bps [tau=0.5 head], pup (N,3), scale (N,3) bps
                   [Student-t scale, DF_VAR dof], at EVERY 1-min bar of the test block, h = 1, 5, 15
  <fold>_val.npz   the same at every 1-min bar of the fold's validation block (for the in-fold choice of h)
  <fold>.json      config, fold ranges, timing, validation metrics, bundle path
  bundles/<fold>/bundle_<fold>_<config hash>/   M10 bundle (weights, both scalers, config.json)
  summary.json     per fold and h: Spearman IC, AUC (masked |r| >= 7 bps and unmasked), P(up) centring,
                   QLIKE against EWMA(0.94), share |mu| > 8 bps; day-block bootstrap CIs
  <fold>.log, run_all.log

Usage (TF venv; PNL_DATA must point at the data store; at most NT_THREADS=2 threads):
  python nt_adapter.py smoke        # tiny slice: shapes, bundle reload, predict_bundle consistency
  python nt_adapter.py fold Y2021   # one fold in this process
  python nt_adapter.py all          # every fold, one child process each, sequentially, then the summary
  python nt_adapter.py summary
Only load(seal=False) is ever called here. Seal inference: predict_bundle(bundle_dir, bars, start_ts, end_ts).
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

THREADS = int(os.environ.get("NT_THREADS", "2"))
for _k, _v in (("OMP_NUM_THREADS", THREADS), ("OPENBLAS_NUM_THREADS", THREADS), ("MKL_NUM_THREADS", THREADS),
               ("TF_NUM_INTRAOP_THREADS", THREADS), ("TF_NUM_INTEROP_THREADS", 1)):
    os.environ[_k] = str(_v)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

WORK = os.path.join(os.environ.get("PNL_WORK", os.path.join(HERE, "work")), "nt")
BUNDLES = os.path.join(WORK, "bundles")
HORIZONS = (1, 5, 15)
DEADBAND_BPS = 7.0
MU_TRADE_BPS = 8.0
WINDOW_STEP, EPOCHS, SEED = 20, 10, 42
CUT_WINDOW_STEP, CUT_EPOCHS = 30, 6           # SPEC compute_plan budget cut 3
CUT_IF_FIT_MIN = 40.0                          # cut when one fit takes / projects above this
WARMUP_DAYS = 7                                # causal-feature warm-up before the first training origin


def _ts(y, m, d):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


EARLIEST_TRAIN = _ts(2020, 2, 1)               # 2020-01 is warm-up only (SPEC data_and_features)
FOLDS = {                                      # test block [start, end)
    "Y2021": (_ts(2021, 1, 1), _ts(2022, 1, 1)),
    "Y2022": (_ts(2022, 1, 1), _ts(2023, 1, 1)),
    "Y2023": (_ts(2023, 1, 1), _ts(2024, 1, 1)),
    "Y2024": (_ts(2024, 1, 1), _ts(2025, 1, 1)),
    "Y2025": (_ts(2025, 1, 1), _ts(2025, 11, 1)),
    "S": (D.SEAL_START, None),                 # bundle only; the seal script predicts later
}
ORDER = ["Y2021", "Y2022", "Y2023", "Y2024", "Y2025", "S"]


def fold_ranges(fold):
    """(train_start, train_end_excl, test_start, test_end_excl). Training labels end before train_end_excl."""
    test_start, test_end = FOLDS[fold]
    d = D.utc(test_start)
    train_start = max(int(dt.datetime(d.year - 1, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()),
                      EARLIEST_TRAIN)
    return train_start, test_start - 86400, test_start, test_end


def _iso(ts):
    return D.utc(ts).isoformat() if ts is not None else None


def bars_frame(bars, a=0, b=None):
    """The M5 frame (timestamp, open, high, low, close, volume, taker_buy_base, trades) for bars[a:b]."""
    import pandas as pd
    sl = slice(a, b)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(bars.t[sl], unit="s"), "open": bars.o[sl], "high": bars.h[sl],
        "low": bars.l[sl], "close": bars.c[sl], "volume": bars.v[sl], "taker_buy_base": bars.tbb[sl],
        "trades": bars.n[sl]})


def _tf():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, REPO)
    import model as M
    assert os.path.abspath(M.__file__) == os.path.join(REPO, "model.py")
    return tf, M


def predict_bundle(bundle_dir, bars, start_ts, end_ts, bundle=None, frame=None):
    """Causal forecasts of a saved bundle at every bar with start_ts <= t < end_ts.

    `bars` is a data.Bars (for the seal: load(seal=True), done by the seal script, not here). Features are computed
    over all of `bars` from its first bar, exactly as for the dev folds, so dev rows are reproduced bit for bit.
    Windows that span a data gap are NOT suppressed here (the policy's gap rule does that).
    Returns {'t', 'mu', 'pup', 'scale'}: t in unix seconds of the decision bar; arrays (N, 3) for h = 1, 5, 15.
    """
    _, M = _tf()
    b = bundle if bundle is not None else M.load_bundle(bundle_dir)
    if frame is None:
        frame = bars_frame(bars)
    idx = np.flatnonzero((bars.t >= start_ts) & ((bars.t < end_ts) if end_ts is not None else True))
    idx = idx[idx >= b.config.LOOKBACK - 1]
    out = M.predict_at(b, frame, idx)
    out["t"] = bars.t[idx].astype(np.int64)
    return out


def _fold_config(M, fold, window_step, epochs):
    cfg = M.Config()
    cfg.WINDOW_STEP = int(window_step)
    cfg.EPOCHS = int(epochs)
    cfg.SEED = SEED
    cfg.FOLD_ID = fold
    cfg.FEATURE_WARMUP_BARS = WARMUP_DAYS * 1440 + 1   # first origin = first training bar
    cfg.ARTIFACT_DIR = cfg.LOG_DIR = os.path.join(BUNDLES, fold)
    return cfg


def _epoch_logger(tf):
    class EpochLog(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            keep = ("loss", "val_loss", "learning_rate", "val_gauss_dir_acc_h1", "val_dir_acc_h1")
            msg = " ".join(f"{k}={float(logs[k]):.4f}" for k in keep if k in logs)
            print(f"[nt] epoch {epoch + 1}: {time.time() - self.t0:.1f}s {msg}", flush=True)
    return EpochLog()


def run_fold(fold, window_step=WINDOW_STEP, epochs=EPOCHS, cut=False, bars=None, out_name=None):
    """Train one fold, save its bundle, predict at every test (and validation) minute, write npz + json."""
    tf, M = _tf()
    os.makedirs(WORK, exist_ok=True)
    out_name = out_name or fold
    t_all = time.time()
    bars = bars if bars is not None else D.load(1)
    tr_start, tr_end, te_start, te_end = fold_ranges(fold)
    i0 = max(0, bars.index_of(tr_start) - WARMUP_DAYS * 1440)
    i1 = bars.index_of(tr_end)
    assert bars.t[i1 - 1] < tr_end <= te_start and bars.t[i0] < tr_start
    train_frame = bars_frame(bars, i0, i1)
    print(f"[nt] {fold}: train {_iso(tr_start)} .. {_iso(tr_end)} (excl), {i1 - i0} rows incl. "
          f"{WARMUP_DAYS}-day warm-up; test {_iso(te_start)} .. {_iso(te_end)} (excl); "
          f"WINDOW_STEP {window_step}, EPOCHS {epochs}, threads {THREADS}", flush=True)

    cfg = _fold_config(M, fold, window_step, epochs)
    t0 = time.time()
    res = M.train_and_evaluate(config=cfg, frame=train_frame, force=True, extra_callbacks=[_epoch_logger(tf)])
    train_s = time.time() - t0
    n_epochs = len(res.history.history["loss"])
    meta = M._read_bundle_meta(res.bundle_dir)
    n_train = meta["split"]["n_train"]
    print(f"[nt] {fold}: fit {train_s / 60:.1f} min, {n_epochs} epochs, best epoch {meta.get('best_epoch', -1) + 1}, "
          f"{n_train} train seq -> {1e3 * train_s / max(1, n_train * n_epochs):.2f} ms per sequence-epoch "
          f"(incl. validation and overhead)", flush=True)
    bundle_dir = res.bundle_dir
    del res
    bundle = M.load_bundle(bundle_dir)  # predict with exactly what is on disk
    full_frame = bars_frame(bars)

    # validation block, every minute (dev data; for the in-fold choice of h)
    import pandas as pd
    v0 = int(pd.Timestamp(meta["split"]["val_first_origin"]).tz_localize("UTC").timestamp())
    v1 = int(pd.Timestamp(meta["split"]["val_last_origin"]).tz_localize("UTC").timestamp()) + 60
    tp = time.time()
    val = predict_bundle(None, bars, v0, v1, bundle=bundle, frame=full_frame)
    val_pred_s = time.time() - tp
    np.savez(os.path.join(WORK, f"{out_name}_val.npz"), **val)
    timing = {"train_s": round(train_s, 1), "epochs_run": n_epochs, "n_train_seq": n_train,
              "ms_per_seq_epoch": round(1e3 * train_s / max(1, n_train * n_epochs), 3),
              "val_predict_s": round(val_pred_s, 1), "n_val_minutes": int(len(val["t"]))}

    n_test, test_pred_s = 0, 0.0
    if te_end is not None:
        tp = time.time()
        test = predict_bundle(None, bars, te_start, te_end, bundle=bundle, frame=full_frame)
        test_pred_s = time.time() - tp
        n_test = int(len(test["t"]))
        assert n_test == bars.index_of(te_end) - bars.index_of(te_start), "a test minute is missing"
        np.savez(os.path.join(WORK, f"{out_name}.npz"), **test)
        timing.update({"test_predict_s": round(test_pred_s, 1), "n_test_minutes": n_test,
                       "ms_per_sample": round(1e3 * test_pred_s / max(1, n_test), 3)})
    timing["total_s"] = round(time.time() - t_all, 1)

    side = {
        "fold": fold, "out_name": out_name,
        "train_range_utc": [_iso(tr_start), _iso(tr_end)], "test_range_utc": [_iso(te_start), _iso(te_end)],
        "purge": "training labels end before test_start - 1 day; validation = last 15% of training, purged 15 bars",
        "window_step": int(window_step), "epochs": int(epochs), "seed": SEED, "budget_cut_applied": bool(cut),
        "threads": THREADS, "warmup_days": WARMUP_DAYS,
        "bundle_dir": bundle.path, "config_hash": meta["config_hash"], "config": meta["config"],
        "split": meta["split"], "best_epoch": meta.get("best_epoch"), "history": meta.get("history"),
        "val_metrics_stride": meta.get("val_metrics"), "timing": timing,
        "outputs": {"test": os.path.join(WORK, f"{out_name}.npz") if te_end is not None else None,
                    "val": os.path.join(WORK, f"{out_name}_val.npz")},
        "units": "mu, scale in bps of log return; pup = direction head P(r > 7 bps | |r| > 7 bps); "
                 "t = unix seconds (open time) of the decision bar, decision at its close",
        "pnl_data": D.DATA, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(WORK, f"{out_name}.json"), "w") as fh:
        json.dump(side, fh, indent=1, default=str)
    print(f"[nt] {fold}: done. timing {json.dumps(timing)}", flush=True)
    return side


# ----------------------------------------------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------------------------------------------
def _realised(bars, t):
    """r_h = 1e4 ln(c[o+h]/c[o]) for decision bars at times t; NaN when o+h is past the data or not h min later."""
    idx = np.searchsorted(bars.t, t)
    lc = np.log(bars.c.astype(np.float64))
    out = np.full((len(t), len(HORIZONS)), np.nan)
    for j, h in enumerate(HORIZONS):
        k = idx + h
        ok = k < len(bars.t)
        ok[ok] &= bars.t[k[ok]] - bars.t[idx[ok]] == 60 * h
        out[ok, j] = 1e4 * (lc[k[ok]] - lc[idx[ok]])
    return out


def _ewma_var(bars, lam=0.94):
    """Causal EWMA of squared 1-min log returns (bps^2), through the decision bar."""
    from scipy.signal import lfilter
    r = np.r_[0.0, 1e4 * np.diff(np.log(bars.c.astype(np.float64)))]
    r2 = r * r
    zi = np.array([lam * float(np.mean(r2[1:1441]))])
    v, _ = lfilter([1 - lam], [1.0, -lam], r2, zi=zi)
    return v


def _ranks(x):
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


def _block_ci(day, stats_fn, parts, n_boot=1000, block=5, seed=0):
    """Moving-block bootstrap over days. parts: dict of per-row arrays; per-day sums are resampled."""
    days, inv = np.unique(day, return_inverse=True)
    sums = {k: np.bincount(inv, weights=v, minlength=len(days)) for k, v in parts.items()}
    nd = len(days)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, nd - block + 1), nb)
        sel = (starts[:, None] + np.arange(block)[None, :]).ravel()[:nd]
        vals.append(stats_fn({k: s[sel].sum() for k, s in sums.items()}))
    vals = np.asarray(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def _pearson_from_sums(s):
    n = s["n"]
    cov = s["xy"] / n - (s["x"] / n) * (s["y"] / n)
    vx = s["xx"] / n - (s["x"] / n) ** 2
    vy = s["yy"] / n - (s["y"] / n) ** 2
    return cov / np.sqrt(vx * vy)


def measure(pred, real, ewma_v, day, cal=None, n_boot=1000):
    """Per-h measurement of one prediction set. `cal`: per-h scale^2 multipliers fitted on validation."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, h in enumerate(HORIZONS):
        y = real[:, j]
        ok = np.isfinite(y)
        y, mu, pup, sc, ev, dj = y[ok], pred["mu"][ok, j], pred["pup"][ok, j], pred["scale"][ok, j], ewma_v[ok] * h, \
            day[ok]
        r = {"n": int(ok.sum())}
        # Spearman IC (CI: block bootstrap of the Pearson correlation of full-sample ranks)
        rx, ry = _ranks(mu), _ranks(y)
        r["ic_spearman"] = float(np.corrcoef(rx, ry)[0, 1])
        parts = {"n": np.ones_like(rx), "x": rx, "y": ry, "xy": rx * ry, "xx": rx * rx, "yy": ry * ry}
        r["ic_ci95"] = _block_ci(dj, _pearson_from_sums, parts, n_boot)
        # AUC of P(up): unmasked (every non-zero move) and masked (|r| >= 7 bps)
        for name, m in (("auc_unmasked", y != 0), ("auc_masked7", np.abs(y) >= DEADBAND_BPS)):
            lab = y[m] > 0
            r[name] = float(roc_auc_score(lab, pup[m]))
            # AUC = 0.5 + mean_pos(F) - mean_neg(F), F = rank fraction within the subset; resampled by day
            rp = _ranks(pup[m])
            parts = {"np": lab.astype(float), "nn": (~lab).astype(float), "fp": rp * lab, "fn": rp * ~lab}
            r[name + "_ci95"] = _block_ci(dj[m], lambda s: 0.5 + s["fp"] / s["np"] - s["fn"] / s["nn"], parts,
                                          n_boot)
        # P(up) centring
        r["mean_pup"] = float(pup.mean())
        r["base_rate_masked7"] = float((y[np.abs(y) >= DEADBAND_BPS] > 0).mean())
        r["base_rate_unmasked"] = float((y[y != 0] > 0).mean())
        r["pup_minus_base_masked7"] = r["mean_pup"] - r["base_rate_masked7"]
        r["share_pup_gt_0.5"] = float((pup > 0.5).mean())
        # QLIKE vs EWMA(0.94): unit-free level y2/v - ln(y2/v) - 1 on y != 0; paired diff of y2/v + ln v on all rows
        y2 = y * y
        nz = y2 > 0

        def qlevel(v):
            q = y2[nz] / v[nz]
            return float(np.mean(q - np.log(q) - 1.0))
        q_e = qlevel(ev)
        r["qlike_ewma094"] = q_e
        variants = {"s2": sc ** 2, "tvar": sc ** 2 * 3.0}   # tvar = DF/(DF-2) s^2 with DF = 3
        if cal is not None:
            variants["s2_valcal"] = sc ** 2 * cal[j]
        for name, v in variants.items():
            r[f"qlike_{name}"] = qlevel(v)
            d = (y2 / v + np.log(v)) - (y2 / ev + np.log(ev))
            r[f"qlike_{name}_vs_ewma_pct"] = float(100.0 * d.mean() / q_e)
            r[f"qlike_{name}_vs_ewma_pct_ci95"] = [100.0 * x / q_e for x in _block_ci(
                dj, lambda s: s["d"] / s["n"], {"d": d, "n": np.ones_like(d)}, n_boot)]
        r["mean_y2_over_tvar"] = float(np.mean(y2 / variants["tvar"]))
        # share of origins with |mu| > 8 bps (the T1-NT entry threshold, theta = 1)
        r["share_abs_mu_gt8"] = float((np.abs(mu) > MU_TRADE_BPS).mean())
        r["mean_abs_mu"] = float(np.abs(mu).mean())
        r["mean_scale"] = float(sc.mean())
        out[f"h{h}"] = r
    return out


def summary(n_boot=1000):
    bars = D.load(1)
    ev = _ewma_var(bars)
    res = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "n_boot": n_boot,
           "bootstrap": "moving blocks of 5 UTC days over per-day sufficient statistics; IC and AUC use "
                        "full-sample ranks", "folds": {}}
    pooled = {k: [] for k in ("mu", "pup", "scale", "real", "ev", "day")}
    for fold in ORDER:
        jp = os.path.join(WORK, f"{fold}.json")
        if not os.path.exists(jp):
            continue
        side = json.load(open(jp))
        ent = {"timing": side["timing"], "budget_cut_applied": side["budget_cut_applied"],
               "bundle_dir": side["bundle_dir"]}
        cal = None
        vp = os.path.join(WORK, f"{fold}_val.npz")
        if os.path.exists(vp):
            v = dict(np.load(vp))
            rv = _realised(bars, v["t"])
            iv = np.searchsorted(bars.t, v["t"])
            cal = [float(np.nanmean(rv[:, j] ** 2 / v["scale"][:, j] ** 2)) for j in range(len(HORIZONS))]
            ent["val_scale2_calibration"] = cal
            ent["validation"] = measure(v, rv, ev[iv], v["t"] // 86400, None, n_boot=200)
        tp = os.path.join(WORK, f"{fold}.npz")
        if os.path.exists(tp):
            p = dict(np.load(tp))
            rr = _realised(bars, p["t"])
            it = np.searchsorted(bars.t, p["t"])
            ent["test"] = measure(p, rr, ev[it], p["t"] // 86400, cal, n_boot=n_boot)
            for k, a in (("mu", p["mu"]), ("pup", p["pup"]), ("scale", p["scale"]), ("real", rr),
                         ("ev", ev[it]), ("day", p["t"] // 86400)):
                pooled[k].append(a)
        res["folds"][fold] = ent
        print(f"[nt] summary {fold} done", flush=True)
    if pooled["mu"]:
        P = {k: np.concatenate(v) for k, v in pooled.items()}
        res["dev_oos_pooled"] = measure({"mu": P["mu"], "pup": P["pup"], "scale": P["scale"]}, P["real"], P["ev"],
                                        P["day"], None, n_boot=n_boot)
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[nt] wrote {os.path.join(WORK, 'summary.json')}", flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------------------------------------------
def smoke():
    """Tiny slice: 3 weeks of 2020-02 training, 1 epoch, 1 day of test inference. Checks shapes and reload."""
    global WORK, BUNDLES
    WORK = os.path.join(WORK, "smoke")
    BUNDLES = os.path.join(WORK, "bundles")
    bars = D.load(1)
    # pretend test starts 2020-02-22: training 2020-02-01 .. 02-21 (12-month rule clipped at EARLIEST_TRAIN)
    FOLDS["SMOKE"] = (_ts(2020, 2, 22), _ts(2020, 2, 23))
    side = run_fold("SMOKE", window_step=20, epochs=1, bars=bars, out_name="SMOKE")
    p = dict(np.load(os.path.join(WORK, "SMOKE.npz")))
    n = 1440
    assert p["t"].shape == (n,) and all(p[k].shape == (n, 3) for k in ("mu", "pup", "scale")), \
        {k: v.shape for k, v in p.items()}
    assert p["t"][0] == _ts(2020, 2, 22) and np.all(np.diff(p["t"]) == 60)
    assert np.all(np.isfinite(p["mu"])) and np.all((p["pup"] > 0) & (p["pup"] < 1)) and np.all(p["scale"] > 0)
    # predict_bundle from the saved bundle reproduces the saved predictions. A different batch composition changes
    # float32 rounding inside oneDNN kernels (relative ~1e-6), so this comparison is relative; identical batches
    # are bit-identical (next check).
    q = predict_bundle(side["bundle_dir"], bars, _ts(2020, 2, 22), _ts(2020, 2, 22) + 3600)
    # a Bars object truncated right after the last origin gives the same forecasts bit for bit (causal end to end)
    cut = bars.slice(0, bars.index_of(_ts(2020, 2, 22) + 3600))
    q2 = predict_bundle(side["bundle_dir"], cut, _ts(2020, 2, 22), _ts(2020, 2, 22) + 3600)
    for k in ("mu", "pup", "scale"):
        np.testing.assert_allclose(q[k], p[k][:60], rtol=1e-5, atol=1e-5)
        assert np.array_equal(q2[k], q[k]), k
    print("[nt] smoke: 1-hour re-prediction bit-identical to the 1-day run:",
          all(np.array_equal(q[k], p[k][:60]) for k in ("mu", "pup", "scale")), flush=True)
    print("[nt] smoke OK:", {k: v.shape for k, v in p.items()}, "mean |mu|", np.abs(p["mu"]).mean(0),
          "mean pup", p["pup"].mean(0), "mean scale", p["scale"].mean(0), flush=True)


def _child(fold, cut):
    args = [sys.executable, os.path.abspath(__file__), "fold", fold] + (["--cut"] if cut else [])
    print(f"[nt] {dt.datetime.now(dt.timezone.utc).isoformat()} start {fold}{' (cut)' if cut else ''}", flush=True)
    with open(os.path.join(WORK, f"{fold}.log"), "w") as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)
    side = json.load(open(os.path.join(WORK, f"{fold}.json")))
    tm = side["timing"]
    print(f"[nt] {dt.datetime.now(dt.timezone.utc).isoformat()} {fold}: fit {tm['train_s'] / 60:.1f} min, "
          f"total {tm['total_s'] / 60:.1f} min", flush=True)
    return side


def _done(fold, cut):
    jp = os.path.join(WORK, f"{fold}.json")
    return os.path.exists(jp) and json.load(open(jp))["budget_cut_applied"] == cut


def run_all():
    """Each fold in its own child process (TF does not free memory between fits), sequentially; then summary.

    Y2021 runs first and is timed. If its fit took more than CUT_IF_FIT_MIN minutes, the SPEC budget cut
    (WINDOW_STEP 30, EPOCHS 6) is applied to every fold, Y2021 included, and recorded in each sidecar.
    """
    os.makedirs(WORK, exist_ok=True)
    jp = os.path.join(WORK, "Y2021.json")
    cut = os.path.exists(jp) and json.load(open(jp))["budget_cut_applied"]
    for fold in ORDER:
        if _done(fold, cut):
            print(f"[nt] {fold}: already done, skipped", flush=True)
            continue
        side = _child(fold, cut)
        if fold == "Y2021":
            tm = side["timing"]
            fit_min, tot_min = tm["train_s"] / 60, tm["total_s"] / 60
            print(f"[nt] projection after Y2021: {fit_min:.1f} min per fit, {tot_min:.1f} min per dev fold -> "
                  f"Y2022-Y2025 + S need about {4 * tot_min + fit_min + 2:.0f} min more", flush=True)
            if not cut and fit_min > CUT_IF_FIT_MIN:
                print(f"[nt] fit above {CUT_IF_FIT_MIN} min: SPEC cut (WINDOW_STEP {CUT_WINDOW_STEP}, EPOCHS "
                      f"{CUT_EPOCHS}) for every fold; refitting Y2021", flush=True)
                cut = True
                _child("Y2021", cut)
    summary()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "smoke":
        smoke()
    elif cmd == "fold":
        f = sys.argv[2]
        c = "--cut" in sys.argv
        run_fold(f, window_step=CUT_WINDOW_STEP if c else WINDOW_STEP, epochs=CUT_EPOCHS if c else EPOCHS, cut=c)
    elif cmd == "all":
        run_all()
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
