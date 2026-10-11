"""Long-context direction nets (exploratory, dev only). One P(up) logit for the 1-bar horizon.
logit = Dense(1)(vec)  [the linear skip, set to a logistic solution, frozen]  +  residual(seq)  [last layer zero-initialised]
 patch: patches of 6 bars -> Dense(d) + learned positions -> pre-LN transformer layers -> mean pool -> Dense(mlp) -> Dense(1, zeros)
 tcn:   causal dilated residual Conv1D blocks (dilations 1..32) -> [last step, mean] -> Dense(mlp) -> Dense(1, zeros)
Both about 20-30k parameters. The untrained network IS the linear model."""
import numpy as np
import tensorflow as tf

L = tf.keras.layers
PATCH = 6
CFG = {"patch": dict(d=32, layers=2, heads=4, key=8, ff=64, drop=0.3, mlp=24),
       "tcn": dict(ch=24, dil=(1, 2, 4, 8, 16, 32), drop=0.3, mlp=24)}


class Positions(L.Layer):
    def build(self, shape):
        self.pos = self.add_weight("pos", (shape[1], shape[2]), initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos


def _encoder(x, name, c):
    h = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln1")(x)
    h = L.MultiHeadAttention(c["heads"], c["key"], dropout=c["drop"], name=f"{name}_att")(h, h)
    x = L.Add()([x, L.Dropout(c["drop"])(h)])
    h = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln2")(x)
    h = L.Dense(c["d"], name=f"{name}_ff2")(L.Dense(c["ff"], activation="gelu", name=f"{name}_ff1")(h))
    return L.Add()([x, L.Dropout(c["drop"])(h)])


def build(arch, n_vec, n_seq, seq_len):
    c = CFG[arch]
    vec = L.Input((n_vec,), name="vec"); seq = L.Input((seq_len, n_seq), name="seq")
    lin = L.Dense(1, name="lin"); logit = lin(vec)
    if arch == "patch":
        assert seq_len % PATCH == 0
        x = L.Reshape((seq_len // PATCH, PATCH * n_seq))(seq)
        x = Positions(name="pos")(L.Dense(c["d"], name="embed")(x))
        for i in range(c["layers"]):
            x = _encoder(x, f"enc{i}", c)
        h = L.GlobalAveragePooling1D()(L.LayerNormalization(epsilon=1e-5, name="enc_out")(x))
    else:
        x = L.Conv1D(c["ch"], 1, name="tcn_in")(seq)
        for i, dil in enumerate(c["dil"]):
            h = L.Conv1D(c["ch"], 3, padding="causal", dilation_rate=dil, activation="gelu", name=f"tcn{i}a")(x)
            h = L.Dropout(c["drop"])(h)
            h = L.Conv1D(c["ch"], 3, padding="causal", dilation_rate=dil, name=f"tcn{i}b")(h)
            x = L.Add()([x, h])
        h = L.Concatenate()([x[:, -1, :], L.GlobalAveragePooling1D()(x)])
    h = L.Dropout(c["drop"])(L.Dense(c["mlp"], activation="gelu", name="mlp")(h))
    res = L.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="residual")(h)
    m = tf.keras.Model([vec, seq], L.Add(name="logit")([logit, res]), name=arch)
    lin.trainable = False                       # the skip stays at the logistic solution; only the residual branch learns
    return m


def set_linear(model, coef, intercept):
    model.get_layer("lin").set_weights([np.asarray(coef, np.float32).reshape(-1, 1), np.asarray(intercept, np.float32).reshape(1)])


def n_params(m):
    return int(sum(int(np.prod(w.shape)) for w in m.weights))
