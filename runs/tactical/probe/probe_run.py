"""Gradient probe of the default network (PROBE_GRADIENTS, NT-037): per loss term, its share of the gradient norm
and its cosine with the total gradient, per variable group (trunk / head / indicator), per epoch.
Same layout as the hill-climb base (cand_c2_base: ~11.5-day block, batch 256, 8 epochs, no calibration pass).
usage: python probe_run.py <DATA_END> <seed> [KEY=VALUE ...]  -> runs/tactical/probe/probe_<slice>_s<seed>[_tag].json"""
import json, os, sys, time
import neural_trade  # noqa: F401  (before tensorflow)
import numpy as np
import tensorflow as tf
import yaml
from neural_trade.core.config import Config
from neural_trade.experiments import screen as S
from neural_trade.registries.models import Models
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.optim import build_optimizers
from neural_trade.training.reset import reset_stateful_rngs, seeded_stochastic_layers
from neural_trade.utils.gpu import apply_gpu_memory_limit
from neural_trade.utils.seeding import seed_everything, set_arithmetic_rewrite

apply_gpu_memory_limit()
de, seed = sys.argv[1], int(sys.argv[2])
kv = dict(a.split("=", 1) for a in sys.argv[3:])
over = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 126000, "FOLD_INDEX": -2, "BATCH_SIZE": 256, "TRAIN_METRICS_EVERY": 10,
        "EPOCHS": 8, "PROBE_GRADIENTS": True, "PROBE_EVERY": 10, "SEEDED_STOCHASTIC_LAYERS": True}
over.update({k: yaml.safe_load(v) for k, v in kv.items()})
cfg = Config.from_yaml("configs/default.yaml").override(**over, DATA_END=de, SEED=seed)
seed_everything(seed); set_arithmetic_rewrite(cfg)
prep = S._prepare_trial_data(cfg, {})
with seeded_stochastic_layers():
    base = Models.build(cfg.MODEL_NAME, cfg)
opt = build_optimizers(cfg)
ys = np.std(prep.y_train) or 1.0
m = CustomTrainModel(base_model=base, pred_scale=ys, pred_mean=np.mean(prep.y_train), lambda_point=cfg.LAMBDA_POINT,
                     lambda_local_trend=cfg.LAMBDA_LOCAL_TREND, lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
                     lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND, lambda_dir=cfg.LAMBDA_DIR, config=cfg,
                     indicator_optimizer=opt.indicator, inputs=base.inputs, outputs=base.outputs)
reset_stateful_rngs(m, seed)
m.compile(optimizer=opt.main)
t0 = time.time()
h = m.fit(prep.train_ds, validation_data=prep.val_ds, epochs=int(cfg.EPOCHS), verbose=0)
probe = {k: [float(x) for x in v] for k, v in h.history.items() if k.startswith("probe_")}
other = {k: [float(x) for x in v] for k, v in h.history.items() if not k.startswith("probe_") and not k.startswith("val_probe")}
tag = ("_" + "_".join(f"{k}{v}" for k, v in kv.items())) if kv else ""
os.makedirs("runs/tactical/probe", exist_ok=True)
out = f"runs/tactical/probe/probe_{de[:10]}_s{seed}{tag}.json"
json.dump({"data_end": de, "seed": seed, "overrides": kv, "train_s": time.time() - t0, "probe": probe, "history": other},
          open(out, "w"), indent=1)
print("wrote", out, "probe keys", len(probe))
