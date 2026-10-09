"""Graph-mode gradient probe on the hc5_noprice3 configuration (CPU), every PROBE_EVERY-th batch over all epochs.
usage: python gprobe_run.py <DATA_END> <seed>  -> runs/tactical/probe/graph/gprobe_<slice>_s<seed>.json"""
import json, sys, time
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf
import yaml
from neural_trade.core.config import Config
from neural_trade.experiments import screen as S
from neural_trade.registries.models import Models
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.optim import build_optimizers
from neural_trade.training.reset import reset_stateful_rngs, seeded_stochastic_layers
from neural_trade.utils.seeding import seed_everything, set_arithmetic_rewrite

de, seed = sys.argv[1], int(sys.argv[2])
spec = yaml.safe_load(open("configs/tactical/hc5_noprice3.yaml"))
cfg = Config.from_yaml("configs/default.yaml").override(**spec["overrides"], DATA_END=de, SEED=seed, EPOCHS=int(spec["run"]["epochs"]),
        PROBE_GRADIENTS=True, PROBE_EVERY=10, SEEDED_STOCHASTIC_LAYERS=True)
seed_everything(seed); set_arithmetic_rewrite(cfg)
prep = S._prepare_trial_data(cfg, {})
with seeded_stochastic_layers():
    base = Models.build(cfg.MODEL_NAME, cfg)
opt = build_optimizers(cfg)
m = CustomTrainModel(base_model=base, pred_scale=np.std(prep.y_train) or 1.0, pred_mean=np.mean(prep.y_train),
    lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND, lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
    lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND, lambda_dir=cfg.LAMBDA_DIR, config=cfg,
    indicator_optimizer=opt.indicator, inputs=base.inputs, outputs=base.outputs)
reset_stateful_rngs(m, seed)
m.compile(optimizer=opt.main)
t0 = time.time(); stamps = []
class _Ep(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        stamps.append(time.time() - t0)
        print(f"PROBE_EPOCH {epoch + 1}/{int(cfg.EPOCHS)} t={stamps[-1]:.0f}s {time.strftime('%H:%M:%S')}", flush=True)
h = m.fit(prep.train_ds, validation_data=prep.val_ds, epochs=int(cfg.EPOCHS), verbose=0, callbacks=[_Ep()])
probe = {k: [float(x) for x in v] for k, v in h.history.items() if k.startswith("probe_")}
other = {k: [float(x) for x in v] for k, v in h.history.items() if not k.startswith(("probe_", "val_probe"))}
out = f"runs/tactical/probe/graph/gprobe_{de[:10]}_s{seed}.json"
json.dump({"data_end": de, "seed": seed, "probe_every": 10, "steps_per_epoch": int(prep.train_ds.cardinality()),
           "train_s": time.time() - t0, "epoch_end_s": stamps, "probe": probe, "history": other}, open(out, "w"), indent=1)
print("wrote", out, "probe keys", len(probe), flush=True)
