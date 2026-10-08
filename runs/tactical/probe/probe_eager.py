"""Gradient shares per loss term WITHOUT the in-graph probe (the owner: the graph-built probe spent 12-19 minutes compiling).
Trains the default network normally (PROBE_GRADIENTS off: no extra graph), then calls the model's own
_run_gradient_probe EAGERLY on N training batches, before training and after it, and reads the accumulators.
Same layout as the hill-climb base (cand_c2_base). Prints per-epoch progress lines for the dashboard.
usage: python probe_eager.py <DATA_END> <seed> [EPOCHS=3] [BATCHES=8]"""
import json, os, sys, time
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf
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
epochs, nb = int(kv.get("EPOCHS", 3)), int(kv.get("BATCHES", 8))
cfg = Config.from_yaml("configs/default.yaml").override(
    CSV_PATH="D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", N_FOLDS=2, VAL_FRACTION=0.1, CAL_FRACTION=0.1, MAX_SEQUENCE_COUNT=126000,
    FOLD_INDEX=-2, BATCH_SIZE=256, TRAIN_METRICS_EVERY=10, EPOCHS=epochs, SEEDED_STOCHASTIC_LAYERS=True, DATA_END=de, SEED=seed)
t0 = time.time()
print(f"PROBE_STAGE data {time.strftime('%H:%M:%S')}", flush=True)
seed_everything(seed); set_arithmetic_rewrite(cfg)
prep = S._prepare_trial_data(cfg, {})
with seeded_stochastic_layers():
    base = Models.build(cfg.MODEL_NAME, cfg)
opt = build_optimizers(cfg)
m = CustomTrainModel(base_model=base, pred_scale=np.std(prep.y_train) or 1.0, pred_mean=np.mean(prep.y_train),
                     lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
                     lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
                     lambda_dir=cfg.LAMBDA_DIR, config=cfg, indicator_optimizer=opt.indicator, inputs=base.inputs, outputs=base.outputs)
reset_stateful_rngs(m, seed)
m.compile(optimizer=opt.main)


def measure(tag):
    for acc in m._probe_means.values():
        acc.reset_state()
    for i, batch in enumerate(prep.train_ds.take(nb)):
        x, y, lc, ext = batch
        m._run_gradient_probe(x, y, lc, ext)   # eager: no graph to compile
    out = {}
    for k, acc in m._probe_means.items():
        try:
            out[k] = float(acc.result())
        except Exception:
            pass
    print(f"PROBE_STAGE measured_{tag} {time.strftime('%H:%M:%S')}", flush=True)
    return out


print(f"PROBE_STAGE measure_init {time.strftime('%H:%M:%S')}", flush=True)
init = measure("init")


class _EpochLine(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"PROBE_EPOCH {epoch + 1}/{epochs} {time.strftime('%H:%M:%S')}", flush=True)


h = m.fit(prep.train_ds, validation_data=prep.val_ds, epochs=epochs, verbose=0, callbacks=[_EpochLine()])
trained = measure("trained")
out = f"runs/tactical/probe/probe_{de[:10]}_s{seed}_eager.json"
json.dump({"data_end": de, "seed": seed, "epochs": epochs, "batches": nb, "seconds": time.time() - t0,
           "probe": {k: [init.get(k), trained.get(k)] for k in trained}, "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}},
          open(out, "w"), indent=1)
print("wrote", out, flush=True)
