"""Zero-shot forecasts through the AdityaLab Samay wrappers, same origins/horizons as run_timestral.py.

Usage: python run_samay.py <model> <context>
  model: chronos2 | bolt_small | bolt_base | timesfm2 | moirai2_small
Samay's wrapper classes expose no raw-array forecast() for Chronos models, so this calls
the wrapped object exactly as Samay's own evaluate() does (model(...) / pipeline.predict_quantiles).
Models with only deciles (Chronos-Bolt, TimesFM) are mapped to the 21 levels by clamping the
tails to q0.1 / q0.9, so their cov98 and tail pinball are not comparable.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, PRED_LEN, QLEVELS, RESULTS, load, origins  # noqa: E402

name, ctx = sys.argv[1], int(sys.argv[2])
batch = int(os.environ.get("BENCH_BATCH", "64"))
limit = int(os.environ.get("BENCH_LIMIT", "0"))
tag = f"samay-{name}_ctx{ctx}_price"
os.makedirs(RESULTS, exist_ok=True)

df, _ = load()
close = df["close"].to_numpy(np.float64)
org = origins(len(df))
sub = int(os.environ.get("BENCH_SUBSTRIDE", "1"))
if sub > 1:
    org = org[::sub]
    tag += f"_sub{sub}"
if limit:
    org = org[:limit]
    tag += f"_limit{limit}"
LV = np.array(QLEVELS)
DECILES = np.linspace(0.1, 0.9, 9)


def to21(q9):
    """(B, H, 9) deciles -> (B, H, 21), tails clamped."""
    return np.stack([np.stack([np.interp(LV, DECILES, row) for row in b]) for b in q9])


if name == "chronos2":
    from samay.model import Chronos_2_Model
    m = Chronos_2_Model(repo="amazon/chronos-2")
    m.model.eval()
    qlev = np.array(m.model.chronos_config.quantiles)
    patch = m.model.chronos_config.output_patch_size

    def predict(x):
        out = m.model(context=torch.tensor(x, dtype=torch.float32),
                      num_output_patches=(PRED_LEN + patch - 1) // patch)
        q = out.quantile_preds.transpose(1, 2).numpy()[:, :PRED_LEN]   # (B, H, Q)
        if len(qlev) != 21 or not np.allclose(qlev, LV):
            q = np.stack([np.stack([np.interp(LV, qlev, row) for row in b]) for b in q])
        return q
elif name.startswith("bolt"):
    from samay.model import ChronosBoltModel
    m = ChronosBoltModel(repo=f"amazon/chronos-bolt-{name.split('_')[1]}")

    def predict(x):
        q, _ = m.pipeline.predict_quantiles(context=torch.tensor(x, dtype=torch.float32),
                                            prediction_length=PRED_LEN, quantile_levels=list(DECILES))
        return to21(q.numpy())
elif name == "timesfm2":
    from samay.model import TimesfmModel
    cfg = {"context_len": ctx, "horizon_len": 128, "backend": "cpu", "per_core_batch_size": batch,
           "input_patch_len": 32, "output_patch_len": 128, "num_layers": 50, "model_dims": 1280,
           "use_positional_embedding": False, "quantiles": list(DECILES)}
    m = TimesfmModel(config=cfg, repo="google/timesfm-2.0-500m-pytorch")

    def predict(x):
        _, full = m.forecast(list(x))          # (B, 128, 1 + 9): mean then deciles
        return to21(np.asarray(full)[:, :PRED_LEN, 1:])
elif name == "timesfm25":
    from samay.model import TimesFM_2p5_Model
    cfg = {"max_context": ctx, "max_horizon": 128, "normalize_inputs": True, "per_core_batch_size": batch,
           "use_continuous_quantile_head": True, "force_flip_invariance": True, "infer_is_positive": True,
           "fix_quantile_crossing": True}
    m = TimesFM_2p5_Model(config=cfg, repo="google/timesfm-2.5-200m-pytorch")
    m.model.model.eval()

    def predict(x):
        xt = torch.tensor(x, dtype=torch.float32)
        _, qf = m.model.compiled_decode(PRED_LEN, xt, torch.zeros_like(xt))   # (B, H, 1 + 9): mean, q0.1..q0.9
        return to21(np.asarray(qf)[:, :PRED_LEN, 1:])
else:
    raise SystemExit(f"unknown model {name}")

hidx = [h - 1 for h in HORIZONS]
out_q = np.zeros((len(org), len(HORIZONS), 21), np.float32)
out_med = np.zeros((len(org), PRED_LEN), np.float32)
t0 = time.perf_counter()
with torch.inference_mode():
    for i in range(0, len(org), batch):
        ob = org[i:i + batch]
        x = close[np.stack([np.arange(t - ctx + 1, t + 1) for t in ob])]
        q = predict(x)
        out_q[i:i + len(ob)] = q[:, hidx, :]
        out_med[i:i + len(ob)] = q[:, :, 10]
        if i == 0:
            print(f"{tag}: first batch {time.perf_counter() - t0:.1f}s; sample median h1 {q[0, 0, 10]:.2f} "
                  f"last {x[0, -1]:.2f}", flush=True)
elapsed = time.perf_counter() - t0
np.savez_compressed(f"{RESULTS}/{tag}.npz", q=out_q, med_path=out_med, origins=org, framing="price",
                    elapsed=elapsed, n=len(org), threads=torch.get_num_threads())
print(f"{tag}: {len(org)} origins in {elapsed:.1f}s ({1000 * elapsed / len(org):.1f} ms/series batched)")
