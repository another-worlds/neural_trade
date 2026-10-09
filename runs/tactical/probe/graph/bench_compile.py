"""Compile / per-step time of the graph-mode gradient probe on CPU (hc5_noprice3 config, a few batches).
usage: python bench_compile.py <PROBE 0|1> [steps]"""
import sys, time, yaml
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf
from neural_trade.core.config import Config
from neural_trade.experiments import screen as S
from neural_trade.registries.models import Models
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.optim import build_optimizers
from neural_trade.training.reset import reset_stateful_rngs, seeded_stochastic_layers
from neural_trade.utils.seeding import seed_everything, set_arithmetic_rewrite

probe, steps = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6
spec = yaml.safe_load(open("configs/tactical/hc5_noprice3.yaml"))
cfg = Config.from_yaml("configs/default.yaml").override(**spec["overrides"], DATA_END="2021-10-13T18:00:00", SEED=0, EPOCHS=1,
        PROBE_GRADIENTS=bool(probe), PROBE_EVERY=1, SEEDED_STOCHASTIC_LAYERS=True)
seed_everything(0); set_arithmetic_rewrite(cfg)
prep = S._prepare_trial_data(cfg, {})
with seeded_stochastic_layers():
    base = Models.build(cfg.MODEL_NAME, cfg)
opt = build_optimizers(cfg)
m = CustomTrainModel(base_model=base, pred_scale=np.std(prep.y_train) or 1.0, pred_mean=np.mean(prep.y_train),
    lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND, lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
    lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND, lambda_dir=cfg.LAMBDA_DIR, config=cfg,
    indicator_optimizer=opt.indicator, inputs=base.inputs, outputs=base.outputs)
reset_stateful_rngs(m, 0)
m.compile(optimizer=opt.main)
ds = prep.train_ds.take(steps)
t0 = time.time(); m.fit(ds.take(1), epochs=1, verbose=0); t1 = time.time()
m.fit(ds, epochs=1, verbose=0); t2 = time.time()
print(f"RESULT probe={probe} first_fit(compile+1 step)={t1-t0:.1f}s next_{steps}_steps={t2-t1:.2f}s per_step={(t2-t1)/steps:.3f}s", flush=True)
