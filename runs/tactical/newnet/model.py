"""New network, models (PLAN 10). Keras / TF 2.10, no registries, no old-model code.

  direction model  [vec (60), seq (60, 8)] -> logits [B, 3] (one P(up) logit per horizon)
      logit = Dense(3)(vec)                       the regression: initialised at sklearn's logistic solution (set_linear)
            + residual(seq)                       a branch whose LAST layer is zero-initialised, so the untrained model IS the regression
      patch: 5-bar patches (12 tokens x 40) -> Dense(32) + learned positions -> 2 pre-LN transformer encoder layers
             (4 heads, key_dim 8, FF 64, dropout 0.2) -> final LN -> mean pool -> Dense(32, gelu) -> dropout -> Dense(3, zeros)
      tcn:   4 residual blocks of 2 causal dilated Conv1D(20, k 3; dilations 1, 2, 4, 8) -> [last step, mean] -> Dense(32, gelu)
             -> dropout -> Dense(3, zeros)
      linear: no residual (the regression alone)
  volatility model [vec] -> predicted log|r| [B, 3]: its OWN small tower (Dense 32 -> 16 -> 3) on the vector, no weights shared
      with the direction model, so its gradient cannot reach the direction branch; Laplace NLL on log|r| with one learned
      scale per horizon (loss in train.py; `vol_nll`).
"""
import numpy as np
import tensorflow as tf

L = tf.keras.layers
D_MODEL, HEADS, KEY, FF, DROP, PATCH = 32, 4, 8, 64, 0.2, 5


def _encoder(x, name):
    h = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln1")(x)
    h = L.MultiHeadAttention(HEADS, KEY, dropout=DROP, name=f"{name}_att")(h, h)
    x = L.Add()([x, L.Dropout(DROP)(h)])
    h = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln2")(x)
    h = L.Dense(D_MODEL, name=f"{name}_ff2")(L.Dense(FF, activation="gelu", name=f"{name}_ff1")(h))
    return L.Add()([x, L.Dropout(DROP)(h)])


class Positions(L.Layer):
    """Learned position embedding added to [B, T, d]."""
    def build(self, shape):
        self.pos = self.add_weight("pos", (shape[1], shape[2]), initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos


def patch_branch(seq, n_seq):
    x = L.Reshape((60 // PATCH, PATCH * n_seq))(seq)
    x = Positions(name="pos")(L.Dense(D_MODEL, name="embed")(x))
    for i in range(2):
        x = _encoder(x, f"enc{i}")
    return L.GlobalAveragePooling1D()(L.LayerNormalization(epsilon=1e-5, name="enc_out")(x))


def tcn_branch(seq, ch=20):
    x = L.Conv1D(ch, 1, name="tcn_in")(seq)
    for i, dil in enumerate((1, 2, 4, 8)):
        h = L.Conv1D(ch, 3, padding="causal", dilation_rate=dil, activation="gelu", name=f"tcn{i}a")(x)
        h = L.Dropout(DROP)(h)
        h = L.Conv1D(ch, 3, padding="causal", dilation_rate=dil, name=f"tcn{i}b")(h)
        x = L.Add()([x, h])
    return L.Concatenate()([x[:, -1, :], L.GlobalAveragePooling1D()(x)])


def build_direction(arch, n_vec, n_seq=8):
    """Returns (model, inputs-needed). arch in {'linear', 'patch', 'tcn'}."""
    vec = L.Input((n_vec,), name="vec"); lin = L.Dense(3, name="lin")
    logit = lin(vec)
    if arch == "linear":
        return tf.keras.Model([vec], logit, name="linear")
    seq = L.Input((60, n_seq), name="seq")
    h = patch_branch(seq, n_seq) if arch == "patch" else tcn_branch(seq)
    h = L.Dropout(DROP)(L.Dense(32, activation="gelu", name="mlp")(h))
    res = L.Dense(3, kernel_initializer="zeros", bias_initializer="zeros", name="residual")(h)
    return tf.keras.Model([vec, seq], L.Add(name="logit")([logit, res]), name=arch)


def build_volatility(n_vec, mu):
    """Predicted log|r| per horizon; the output bias starts at the training mean of log|r|."""
    vec = L.Input((n_vec,), name="vvec")
    h = L.Dense(32, activation="gelu")(vec); h = L.Dense(16, activation="gelu")(h)
    out = L.Dense(3, bias_initializer=tf.constant_initializer(np.asarray(mu, np.float32)))(h)
    return tf.keras.Model(vec, out, name="volatility")


def set_linear(model, coef, intercept):
    """coef [3, n_vec] and intercept [3] from sklearn (one logistic regression per horizon) -> the Dense(3) 'lin' layer."""
    model.get_layer("lin").set_weights([np.asarray(coef, np.float32).T, np.asarray(intercept, np.float32)])


def vol_nll(y, mu, log_b):
    """Laplace NLL (up to a constant) of y = log|r| given the predicted location mu and one learned log-scale per horizon."""
    return tf.reduce_mean(tf.abs(y - mu) * tf.exp(-log_b) + log_b, 0)


def n_params(m):
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))
