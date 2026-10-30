"""Train neural_trade (model.py, imported unmodified from the repo) on data up to the first benchmark origin,
then predict at every benchmark origin and save price-level quantiles in the bench npz format.

Usage: nice -n 19 venv/bin/python run_nt.py --data repo|post --seed K

Only IO is redirected: CSV_PATH (header-renamed copy, training slice), MODEL_PATH / SCALER_PATH and the
cwd-relative logs (training_log.csv, indicator_params_history.csv) all go to runs/<data>_seed<K>/.
One non-IO override: train_and_evaluate() hard-codes tf.keras.utils.set_random_seed(42); that call is
redirected to set_random_seed(K) so seeds 0/1/2 differ (everything else unchanged).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
SCR_WORK = _os.environ.get("TSFM_EVAL_WORK", _os.path.join(_HERE, "..", "work"))
REPO_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

import argparse
import json
import os
import shutil
import sys
import time

for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True     # never write __pycache__ into the repo dir

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["repo", "post"], required=True)
ap.add_argument("--seed", type=int, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from scipy.stats import t as student_t  # noqa: E402

SCR = SCR_WORK
H2H = f"{SCR}/nt_h2h"
REPO = REPO_ROOT
os.environ["BENCH_DATA"] = args.data
sys.path.insert(0, _os.path.join(_HERE, "..", "bench"))
import common  # noqa: E402

sys.path.insert(0, REPO)
import model as M  # noqa: E402  (unmodified <repo>/model.py)

assert os.path.abspath(M.__file__) == f"{REPO}/model.py"

df_c, info = common.load()
ORG = common.origins(len(df_c))
close64 = df_c["close"].to_numpy(np.float64)
t0 = int(ORG[0])

run_dir = f"{H2H}/runs/{args.data}_seed{args.seed}"
shutil.rmtree(run_dir, ignore_errors=True)
os.makedirs(run_dir)
os.chdir(run_dir)   # CSVLogger / ParamsLogger write cwd-relative files

# ---- seed: replace the hard-coded 42 inside train_and_evaluate with args.seed ----
_orig_set_seed = tf.keras.utils.set_random_seed
_seed_calls = []


def _set_seed_override(s):
    _seed_calls.append(s)
    _orig_set_seed(args.seed)


tf.keras.utils.set_random_seed = _set_seed_override
_orig_set_seed(args.seed)

cfg = M.Config()
overrides = {
    "CSV_PATH": f"{H2H}/data/{args.data}_train_upto_{t0}.csv",
    "MODEL_PATH": f"{run_dir}/nn_learnable_indicators_v3.weights.h5",
    "SCALER_PATH": f"{run_dir}/scaler_v3.joblib",
}
t_train = time.time()
res = M.train_and_evaluate(config=cfg, config_overrides=overrides, force=False, calibrate=True)
train_s = time.time() - t_train
tf.keras.utils.set_random_seed = _orig_set_seed
assert _seed_calls == [42], _seed_calls
print(f"[h2h] training+eval wall {train_s:.1f}s")

model = res.model
tsc = res.target_scaler
isc = res.input_scaler
# keep the exact in-memory weights that model.py's own evaluation used (extra file, IO only)
model.base_model.save_weights(f"{run_dir}/final_base.weights.h5")

# ---- check that our prediction path reproduces model.py's own test-set predictions ----
yp = model.predict(tf.data.Dataset.from_tensor_slices(res.X_test_seq).batch(cfg.BATCH_SIZE), verbose=0)
own = res.predictions
chk_delta = max(float(np.max(np.abs(tsc.inverse_transform(yp[3 * k][:, :1]).ravel() - own["delta"][f"h{k}"])))
                for k in range(3))
print(f"[h2h] reproduce model.py test predictions: max |delta diff| = {chk_delta:.3e} USD")

# ---- full series via model.py's own loader (header-renamed full CSV) ----
cfg_full = M.Config()
cfg_full.CSV_PATH = f"{H2H}/data/{args.data}_full.csv"
_, close32 = M.DataProcessor(cfg_full).load_and_prepare_data()
assert len(close32) == len(close64)
assert np.array_equal(close32, close64.astype(np.float32))

L = cfg.LOOKBACK
# window used by model.py for sequence i is close[i-L:i] with last_close = close[i-1]; origin t <=> i = t+1
# check against the last test sequence of the training slice (i = t0 - 14 -> t = t0 - 15)
t_last = t0 - (max(cfg.HORIZON_STEPS) - 1) - 1
w_last = isc.transform(close32[t_last - L + 1:t_last + 1][None, :]).astype(np.float32)
assert np.allclose(w_last, res.X_test_seq[-1:], atol=1e-6), "window alignment mismatch"
assert np.isclose(res.last_close_test[-1], close32[t_last])

# ---- predict at every origin ----
idx = ORG[:, None] + np.arange(-L + 1, 1)[None, :]
t_pred = time.time()
X = isc.transform(close32[idx]).reshape(len(ORG), L).astype(np.float32)
ya = model.predict(tf.data.Dataset.from_tensor_slices(X).batch(cfg.BATCH_SIZE), verbose=0)
delta_usd = np.stack([tsc.inverse_transform(ya[3 * k][:, :1]).ravel() for k in range(3)], axis=1)
dir_prob = np.stack([np.asarray(ya[3 * k + 1]).reshape(-1) for k in range(3)], axis=1)
var_raw = np.stack([np.asarray(ya[3 * k + 2]).reshape(-1) for k in range(3)], axis=1)
elapsed = time.time() - t_pred

# ---- predictive distribution as in the NLL: Student-t(nu=DF_VAR), loc = price head, scale^2 = clip(v) ----
s_usd = float(tsc.scale_[0])
var_c = np.clip(var_raw, cfg.VAR_FLOOR, cfg.VAR_CAP)
scale_usd = np.sqrt(var_c) * s_usd
z = student_t.ppf(np.array(common.QLEVELS), df=cfg.DF_VAR)
q = close64[ORG][:, None, None] + delta_usd[:, :, None] + z[None, None, :] * scale_usd[:, :, None]
med = q[:, :, 10]
med_path = np.stack([np.interp(np.arange(1, 16), common.HORIZONS, m) for m in med])

out_dir = f"{H2H}/out/{args.data}"
os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/neural_trade-asis-seed{args.seed}_ctx60_price.npz"
np.savez(path, q=q.astype(np.float32), med_path=med_path.astype(np.float32), origins=ORG.astype(np.int64),
         framing="price", elapsed=float(elapsed), n=int(len(ORG)), threads=1,
         dir_prob=dir_prob.astype(np.float32), delta_usd=delta_usd.astype(np.float32),
         var_raw=var_raw.astype(np.float32))

# ---- sanity summary ----
real = np.stack([close64[ORG + h] - close64[ORG] for h in common.HORIZONS], axis=1)
summ = {
    "data": args.data, "seed": args.seed, "n": int(len(ORG)), "t0": t0,
    "train_wall_s": round(train_s, 1), "predict_wall_s": round(elapsed, 2),
    "history_epochs": len(res.history.history["loss"]) if res.history is not None else None,
    "val_loss": [round(float(v), 4) for v in res.history.history.get("val_loss", [])] if res.history else None,
    "lambdas": {k: float(getattr(model, k)) for k in ["lambda_point", "lambda_extended_trend", "lambda_dir",
                                                      "lambda_var", "lambda_short", "lambda_long"]},
    "target_scaler_mean": float(tsc.mean_[0]), "target_scaler_scale": s_usd,
    "pred_scale": float(model.pred_scale.numpy()), "pred_mean": float(model.pred_mean.numpy()),
    "reproduce_max_abs_diff_usd": chk_delta,
    "frac_dir_gt_0.5": [round(float((dir_prob[:, k] > 0.5).mean()), 4) for k in range(3)],
    "dir_prob_mean": [round(float(dir_prob[:, k].mean()), 4) for k in range(3)],
    "dir_prob_std": [round(float(dir_prob[:, k].std()), 5) for k in range(3)],
    "mean_abs_delta_usd": [round(float(np.abs(delta_usd[:, k]).mean()), 3) for k in range(3)],
    "mean_delta_usd": [round(float(delta_usd[:, k].mean()), 3) for k in range(3)],
    "std_delta_usd": [round(float(delta_usd[:, k].std()), 3) for k in range(3)],
    "frac_delta_gt_0": [round(float((delta_usd[:, k] > 0).mean()), 4) for k in range(3)],
    "mean_abs_real_delta_usd": [round(float(np.abs(real[:, k]).mean()), 3) for k in range(3)],
    "std_real_delta_usd": [round(float(real[:, k].std()), 3) for k in range(3)],
    "var_raw_mean": [float(var_raw[:, k].mean()) for k in range(3)],
    "var_raw_min": [float(var_raw[:, k].min()) for k in range(3)],
    "var_raw_max": [float(var_raw[:, k].max()) for k in range(3)],
    "frac_var_clipped": [float(((var_raw[:, k] < cfg.VAR_FLOOR) | (var_raw[:, k] > cfg.VAR_CAP)).mean())
                         for k in range(3)],
    "mean_scale_usd": [round(float(scale_usd[:, k].mean()), 3) for k in range(3)],
    "cov80_usd": [round(float(((real[:, k] >= q[:, k, 2] - close64[ORG]) & (real[:, k] <= q[:, k, 18] - close64[ORG])).mean()), 4)
                  for k in range(3)],
    "any_nan": bool(np.isnan(q).any() or np.isnan(dir_prob).any()),
    "corr_delta_real": [round(float(np.corrcoef(delta_usd[:, k], real[:, k])[0, 1]), 4) for k in range(3)],
    "npz": path,
}
json.dump(summ, open(f"{run_dir}/summary.json", "w"), indent=1)
print("[h2h] SUMMARY " + json.dumps(summ))
