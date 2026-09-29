"""Q1 helper: rebuild gru_attention with the trainer's seeding (seed 42) and read the INITIAL meta_adjust
Dense kernel, to see how far training moved it in the saved runs (read-only; CPU).

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_meta_init.py
"""
import glob
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401  (CUDA DLL path before TF)
import h5py
import numpy as np
from neural_trade.core.config import Config
from neural_trade.registries.models import Models
from neural_trade.utils.seeding import seed_everything

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1_meta_init.json")
RUNS = "D:/neural_trade/runs"


def init_kernel(seed=42):
    seed_everything(seed)
    m = Models.build(None, Config())
    dense = [l for l in m.layers if l.__class__.__name__ == "Dense" and tuple(l.kernel.shape) == (2, 18)]
    k, b = dense[0].get_weights()
    return k, b, dense[0].name


def main():
    k0, b0, name = init_kernel(42)
    k0b, _, _ = init_kernel(42)
    out = {"layer": name, "init_repeatable_max_abs_diff": float(np.abs(k0 - k0b).max()),
           "init_kernel_max_abs": float(np.abs(k0).max()), "glorot_limit": float(np.sqrt(6 / 20)), "runs": {}}
    for r in sorted(glob.glob(f"{RUNS}/2026*/")):
        fn = r + "artifacts/weights.h5" if os.path.exists(r + "artifacts/weights.h5") else r + "weights.h5"
        with h5py.File(fn, "r") as f:
            k = np.array(f["dense/dense/kernel:0"]); b = np.array(f["dense/dense/bias:0"])
        seed = json.load(open(r + "meta.json")).get("seed")
        out["runs"][os.path.basename(r.rstrip("/\\"))] = {
            "seed": seed, "weights": os.path.relpath(fn, RUNS),
            "max_abs_kernel_change_from_seed42_init": float(np.abs(k - k0).max()),
            "median_abs_kernel_change": float(np.median(np.abs(k - k0))),
            "rel_change_fro": float(np.linalg.norm(k - k0) / np.linalg.norm(k0)),
            "bias_range": [float(b.min()), float(b.max())]}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
