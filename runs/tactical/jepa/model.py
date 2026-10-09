"""Small patch transformer, EMA target, predictor, VICReg terms."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

PATCH, D, HEADS, FFN, LAYERS, MAX_TOK = 5, 32, 4, 128, 2, 12


class Block(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ln1, self.ln2 = L.LayerNormalization(), L.LayerNormalization()
        self.att = L.MultiHeadAttention(HEADS, D // HEADS)
        self.ff = keras.Sequential([L.Dense(FFN, activation="gelu"), L.Dense(D)])

    def call(self, x):
        y = self.ln1(x); x = x + self.att(y, y)
        return x + self.ff(self.ln2(x))


class Encoder(keras.Model):
    """[B, T, C] (T a multiple of PATCH) -> mean-pooled embedding [B, D]. Shape-agnostic in T up to MAX_TOK patches."""
    def __init__(self, n_ch, **kw):
        super().__init__(**kw)
        self.n_ch = n_ch
        self.proj = L.Dense(D)
        self.pos = self.add_weight("pos", (MAX_TOK, D), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.blocks = [Block() for _ in range(LAYERS)]
        self.ln = L.LayerNormalization()

    def tokens(self, x):
        b, t = tf.shape(x)[0], x.shape[1]
        n = t // PATCH
        return tf.reshape(x, (b, n, PATCH * self.n_ch))

    def call(self, x):
        tok = self.tokens(x)
        h = self.proj(tok) + self.pos[: tok.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return tf.reduce_mean(self.ln(h), 1)


def make_predictor():
    return keras.Sequential([L.Dense(64, activation="gelu"), L.Dense(D)], name="predictor")


def ema_update(target, online, m):
    for t, o in zip(target.weights, online.weights):
        t.assign(m * t + (1.0 - m) * o)


def vicreg(z, gamma=1.0):
    """variance hinge on per-dim std and covariance off-diagonal penalty; z [B, D]."""
    z = z - tf.reduce_mean(z, 0)
    std = tf.sqrt(tf.math.reduce_variance(z, 0) + 1e-4)
    var = tf.reduce_mean(tf.nn.relu(gamma - std))
    cov = tf.matmul(z, z, transpose_a=True) / tf.cast(tf.shape(z)[0] - 1, tf.float32)
    off = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    return var, tf.reduce_sum(tf.square(off)) / D


def effective_rank(z):
    """exp of the entropy of the normalised singular values of the centred batch."""
    z = z - tf.reduce_mean(z, 0)
    with tf.device('/CPU:0'):
        s = tf.linalg.svd(z, compute_uv=False)
    p = s / (tf.reduce_sum(s) + 1e-12)
    return tf.exp(-tf.reduce_sum(p * tf.math.log(p + 1e-12)))
