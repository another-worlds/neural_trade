"""CPU per-term gradient norms of the leader model (runs/scenarios/long_360d_stab/...f-2__s0) on one
batch of the bundled 30-day file. Read-only on the repo."""
import json, os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import neural_trade  # noqa: F401  (before tensorflow)
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.registries.models import Models
from neural_trade.training.custom_model import CustomTrainModel

RUN = 'D:/neural_trade/runs/scenarios/long_360d_stab/20260930T094257Z-dce15ed-e3669618-default__f-2__s0'
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

cfg = Config.from_yaml(RUN + '/config.yaml')
meta = json.load(open(RUN + '/artifacts/meta.json'))
cfg = cfg.override(CSV_PATH='D:/neural_trade/binance_btcusdt_1min_ccxt.csv', MAX_SEQUENCE_COUNT=0,
                   MODEL_PATH='D:/nt_math_scratch/none.h5', SCALER_PATH='D:/nt_math_scratch/none.joblib',
                   ARTIFACTS_DIR='D:/nt_math_scratch/art')
tf.random.set_seed(SEED)
dp = DataProcessor(cfg)
df, close = dp.load_and_prepare_data()
X, y, lc, ext = dp.build_windows(close)
ps, pm = float(meta['pred_scale']), float(meta['pred_mean'])
rng = np.random.default_rng(SEED)
idx = rng.choice(X.shape[0], size=B, replace=False)
lcb = lc[idx].reshape(-1, 1).astype(np.float32)
xb = ((X[idx] - lcb) / ps).astype(np.float32)          # window_relative
yb = ((y[idx] - pm) / ps).astype(np.float32)
eb = ext[idx].astype(np.float32)
print('pred_scale', ps, 'pred_mean', pm, 'X', X.shape, 'y scaled std per h', yb.std(0))

base = Models.build(cfg.MODEL_NAME, cfg)
m = CustomTrainModel(base_model=base, pred_scale=ps, pred_mean=pm, lambda_point=cfg.LAMBDA_POINT,
                     lambda_local_trend=cfg.LAMBDA_LOCAL_TREND, lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
                     lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND, lambda_dir=cfg.LAMBDA_DIR, config=cfg,
                     inputs=base.inputs, outputs=base.outputs)
m(xb[:2], training=False)
m.load_weights(RUN + '/weights.h5')
L = meta['lambda_values_final']
for k, v in L.items():
    try:
        setattr(m, k, v)
    except Exception as e:
        print('skip', k, e)

x_t, y_t, lc_t, e_t = map(tf.constant, (xb, yb, lcb, eb))
t0 = time.time()
with tf.GradientTape(persistent=True) as tape:
    out = m(x_t, training=True)
    heads = list(out[:9])
    for h in heads:
        tape.watch(h)
    c = m.custom_loss(x_t, y_t, heads, lc_t, e_t, vacuum_overflow=out[9])
    lam = {k: float(v) for k, v in L.items()}
    terms = {
        'point': c.point_h0 + c.point_h1 + c.point_h2,
        'trend': m.lambda_trend_outer * (c.extended_h0 + c.extended_h1 + c.extended_h2),
        'dir_bce': m.lambda_dir_outer * m.lambda_dir * (c.dir_h0 + c.dir_h1 + c.dir_h2),
        'nll': m.lambda_nll_outer * m.lambda_var * (c.nll_h0 + c.nll_h1 + c.nll_h2),
        'crps': m.lambda_crps * (c.crps_h0 + c.crps_h1 + c.crps_h2),
        'soft_ece': m.lambda_soft_ece * (c.soft_ece_h0 + c.soft_ece_h1 + c.soft_ece_h2),
        'vol': 0.1 * c.vol_loss,
        'inter_reg': 0.1 * c.inter_reg,
        't_perp': c.t_perp_total, 'casimir': c.casimir_val, 'hd': c.hd_val, 'ife': c.ife_val,
        'vac_overflow': c.vac_overflow_val,
    }
    terms['coherence(resid)'] = c.total - tf.add_n(list(terms.values()))
unw = {'point': None, 'trend': lam['lambda_extended_trend'], 'dir_bce': lam['lambda_dir'], 'nll': lam['lambda_var'],
       'crps': lam['lambda_crps'], 'soft_ece': lam['lambda_soft_ece'], 'vol': 0.1 * lam['lambda_vol'],
       't_perp': lam['lambda_t_perp'], 'casimir': lam['lambda_casimir'], 'hd': lam['lambda_hd'], 'ife': lam['lambda_ife'],
       'vac_overflow': lam['lambda_vac_overflow']}
tv = m.trainable_variables
ind_ids = m._indicator_var_ids
print(f'total loss {float(c.total):.4f}; n trainable vars {len(tv)}, indicator vars {len(ind_ids)}')
g_tot = tape.gradient(c.total, tv)
def gn(gs, which=None):
    sel = [g for g, v in zip(gs, tv) if g is not None and (which is None or ((id(v) in ind_ids) == which))]
    return float(tf.linalg.global_norm(sel)) if sel else 0.0
print(f"TOTAL: value {float(c.total):.4f} |g| all {gn(g_tot):.3f} nn {gn(g_tot, False):.3f} ind {gn(g_tot, True):.3f}")
rows = []
for k, v in terms.items():
    gs = tape.gradient(v, tv)
    gh = tape.gradient(v, heads, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # per-head-type norms of dL/d(head output), i.e. sum over batch of per-example grads (loss is a mean)
    hp = float(tf.linalg.global_norm([gh[0], gh[3], gh[6]]))
    hd_ = float(tf.linalg.global_norm([gh[1], gh[4], gh[7]]))
    hv = float(tf.linalg.global_norm([gh[2], gh[5], gh[8]]))
    # cosine with total gradient
    flat = lambda gs_: tf.concat([tf.reshape(g if g is not None else tf.zeros_like(v_), [-1]) for g, v_ in zip(gs_, tv)], 0)
    a, b = flat(gs), flat(g_tot)
    cos = float(tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + 1e-30))
    w = unw.get(k)
    rows.append((k, float(v), gn(gs), gn(gs, False), gn(gs, True), (gn(gs) / w) if w else float('nan'), cos, hp, hd_, hv))
print(f"{'term':18s} {'value':>9s} {'|g|w':>9s} {'|g|nn':>9s} {'|g|ind':>9s} {'|g|/lam':>9s} {'cos(g,gtot)':>11s}  |dL/dprice| |dL/ddir| |dL/dvar|")
for r in rows:
    print(f"{r[0]:18s} {r[1]:9.4f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:9.4f} {r[5]:9.4f} {r[6]:11.3f}  {r[7]:9.4g} {r[8]:9.4g} {r[9]:9.4g}")
# head statistics
pv = np.concatenate([heads[i].numpy() for i in (2, 5, 8)], 1)
pp = np.concatenate([heads[i].numpy() for i in (0, 3, 6)], 1)
pd_ = np.concatenate([heads[i].numpy() for i in (1, 4, 7)], 1)
print('var head min/median/max per h', pv.min(0), np.median(pv, 0), pv.max(0), 'share at floor 1e-4', (pv <= 1e-4).mean(0))
print('price head std per h', pp.std(0), 'target std', yb.std(0), 'resid^2 mean', ((yb - pp) ** 2).mean(0))
print('dir head min/max/std', pd_.min(0), pd_.max(0), pd_.std(0))
print('elapsed', round(time.time() - t0, 1), 's')
