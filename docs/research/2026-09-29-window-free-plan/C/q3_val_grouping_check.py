"""Q3 (0): is the logged val CRPS / NLL independent of the validation batch grouping, and is val_loss not?

Loads one v1 dev-fold run's checkpoint weights on the CPU (as q2_block_scaling.py does), then evaluates
the SAME weights on the SAME fold -2 validation block with batch 256 and batch 1024, through the model's
own test_step (custom_model.evaluate). Nothing in D:/neural_trade is written (cwd and paths in ./scratch).
Run: cd D:/nt_research/wfp/C/scratch && CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 \
     C:/Users/Step/miniforge3/envs/nt/python ../q3_val_grouping_check.py
Writes q3_val_grouping_check.json.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "scratch"
OUT = HERE / "q3_val_grouping_check.json"
RUN = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full/runs/20260923T195537Z-6dec27a-53cbe6af-all_on__s0__P1")


def main():
    SCRATCH.mkdir(exist_ok=True)
    os.chdir(SCRATCH)
    import neural_trade  # noqa: F401
    import numpy as np
    from neural_trade.core.config import Config
    from neural_trade.data.datasets import create_datasets
    from neural_trade.data.processor import DataProcessor
    from neural_trade.training.trainer import train_and_evaluate

    wdir = SCRATCH / ("grouping_" + RUN.name)
    wdir.mkdir(exist_ok=True)
    shutil.copyfile(RUN / "weights.h5", wdir / "weights.h5")
    cfg = Config.from_yaml(RUN / "config.yaml")
    cfg.CSV_PATH = "D:/neural_trade/binance_btcusdt_1min_ccxt.csv"
    cfg.MODEL_PATH = str(wdir / "weights.h5")
    cfg.SCALER_PATH = str(wdir / "scaler.joblib")
    cfg.ARTIFACTS_DIR = str(wdir / "artifacts")
    res = train_and_evaluate(config=cfg, force=False, calibrate=False, fit_calibration=False, save_artifacts=False)
    model = res.model
    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    dp.prepare_datasets(df, close)
    vb = dp.val_block
    out = {"run": RUN.name, "n_val": int(len(vb["y_scaled"]))}
    for bs in (256, 1024, 2866):
        cfg.BATCH_SIZE = bs
        _, val_ds = create_datasets(cfg, vb["X"], vb["y_scaled"], vb["last_close"], vb["extended_trends"],
                                    vb["X"], vb["y_scaled"], vb["last_close"], vb["extended_trends"])
        r = model.evaluate(val_ds, return_dict=True, verbose=0)
        out[str(bs)] = {k: float(r[k]) for k in ("loss", "crps_h0", "crps_h1", "crps_h2", "nll_h0", "nll_h1", "nll_h2",
                                                 "soft_ece_h1", "t_perp_loss", "hd_loss", "ife_loss", "vol_loss", "dir_loss_h1")
                        if k in r}
    keys = [k for k in out["256"]]
    out["rel_diff_1024_vs_256"] = {k: (out["1024"][k] - out["256"][k]) / (abs(out["256"][k]) + 1e-12) for k in keys}
    out["rel_diff_fullblock_vs_256"] = {k: (out["2866"][k] - out["256"][k]) / (abs(out["256"][k]) + 1e-12) for k in keys}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
