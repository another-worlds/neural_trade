"""The batched gradient probe (one tape.gradient per term, Gram matrix) against the original per-group
implementation, eager and inside tf.function, plus the new pairwise-cosine keys (tactical nt-tactical-gprobe)."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.training.custom_model import CustomTrainModel

B = 32


def _model():
    inp = tf.keras.Input(shape=(60,), name="close_window")
    h = tf.keras.layers.Dense(8, activation="tanh", name="trunk_dense")(inp)
    kinds = ["price_h0", "price_h1", "price_h2", "direction_h0", "direction_h1", "direction_h2",
             "variance_h0", "variance_h1", "variance_h2", "variance_extra"]
    outs = [tf.keras.layers.Dense(1, name=k)(h) for k in kinds]
    base = tf.keras.Model(inp, outs, name="tiny_probe_base")
    cfg = Config(PROBE_GRADIENTS=True, PROBE_EVERY=1, LAMBDA_CRPS=0.2, LAMBDA_SOFT_ECE=0.2, LAMBDA_T_PERP=0.1,
                 LAMBDA_CASIMIR=0.1, LAMBDA_HD=0.1, LAMBDA_IFE=0.1, LAMBDA_VAC_OVERFLOW=0.1)
    m = CustomTrainModel(base_model=base, pred_scale=261.0, pred_mean=3.2, config=cfg,
                         inputs=base.inputs, outputs=base.outputs)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return m


def _batch():
    rng = np.random.default_rng(3)
    x = tf.constant(rng.normal(0, 1, (B, 60)).astype(np.float32))
    y = tf.constant(rng.normal(0, 1, (B, 3)).astype(np.float32))
    lc = tf.constant((110_000 + rng.normal(0, 500, (B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0, 200, (B, 3)).astype(np.float32))
    return x, y, lc, ext


def _reference(self, x_window, y_true, last_close, extended_trends):
    """The original per-term-and-group implementation, verbatim (returns plain dicts)."""
    groups = self._probe_groups_of()
    with tf.GradientTape(persistent=True) as tape:
        y_pred_list = self(x_window, training=False)
        heads = PredictiveOutputs(*y_pred_list[:len(PredictiveOutputs._fields)])
        c = self.custom_loss(x_window, y_true, y_pred_list[:9], last_close, extended_trends,
                             vacuum_overflow=heads.vacuum_overflow)
        terms = {
            'point': c.point_h0 + c.point_h1 + c.point_h2,
            'trend': self.lambda_trend_outer * (c.extended_h0 + c.extended_h1 + c.extended_h2),
            'dir': self.lambda_dir_outer * self.lambda_dir * (c.dir_h0 + c.dir_h1 + c.dir_h2),
            'dir_align': self.lambda_dir_align_outer * c.dir_align_val,
            'nll': self.lambda_nll_outer * self.lambda_var * (c.nll_h0 + c.nll_h1 + c.nll_h2),
            'crps': self.lambda_crps * (c.crps_h0 + c.crps_h1 + c.crps_h2),
            'soft_ece': self.lambda_soft_ece * (c.soft_ece_h0 + c.soft_ece_h1 + c.soft_ece_h2),
            'vol': 0.1 * c.vol_loss, 'inter_reg': 0.1 * c.inter_reg, 't_perp': c.t_perp_total,
            'casimir': c.casimir_val, 'hd': c.hd_val, 'ife': c.ife_val, 'vac_overflow': c.vac_overflow_val,
            'vac': c.vac_val, 'pnl': c.pnl_val, 'coherence': self.lambda_coherence_outer * c.coherence_penalty_val,
        }

    def flat(target, vs):
        gs = tape.gradient(target, vs)
        return tf.concat([tf.reshape(g if g is not None else tf.zeros_like(v), [-1]) for g, v in zip(gs, vs)], 0)

    out = {}
    for gname, vs in groups.items():
        if not vs:
            continue
        tot = flat(c.total, vs)
        fl = {n: flat(v, vs) for n, v in terms.items()}
        nm = {n: tf.norm(f) for n, f in fl.items()}
        ns = tf.add_n(list(nm.values())) + self.eps
        for n in terms:
            out[f'probe_cos_{n}_{gname}'] = float(tf.reduce_sum(fl[n] * tot) / (nm[n] * tf.norm(tot) + self.eps))
            out[f'probe_grad_share_{n}_{gname}'] = float(nm[n] / ns)
        names = list(terms)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                out[f'probe_pcos_{a}__{b}_{gname}'] = float(
                    tf.reduce_sum(fl[a] * fl[b]) / (nm[a] * nm[b] + self.eps))
    return out


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "tf.function"])
def test_batched_probe_equals_per_group_reference(graph):
    m = _model()
    batch = _batch()
    ref = _reference(m, *batch)
    groups = m._probe_groups_of()
    assert all(groups[g] for g in ("trunk", "head")) and not groups["indicator"]
    fn = tf.function(m._run_gradient_probe) if graph else m._run_gradient_probe
    fn(*batch)
    logs = m.train_epoch_logs()
    checked = 0
    for k, v in ref.items():
        assert logs[k] == pytest.approx(v, abs=1e-4), k
        checked += 1
    assert checked > 17 * 2 * 2 + 136  # shares + cosines for two groups, and pair cosines
    pair = [v for k, v in ref.items() if k.startswith("probe_pcos_") and k.endswith("_head")]
    assert logs["probe_conflict_min_head"] == pytest.approx(min(pair), abs=1e-4)
    assert logs["probe_conflict_mean_head"] == pytest.approx(float(np.mean(pair)), abs=1e-4)


def test_pairwise_keys_absent_when_probe_off():
    inp = tf.keras.Input(shape=(60,))
    outs = [tf.keras.layers.Dense(1, name=f"price_h{i}")(inp) for i in range(10)]
    base = tf.keras.Model(inp, outs)
    m = CustomTrainModel(base_model=base, pred_scale=1.0, pred_mean=0.0, config=Config(),
                         inputs=base.inputs, outputs=base.outputs)
    assert m._probe_mats == {}
