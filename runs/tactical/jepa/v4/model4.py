"""v4 model: v2's masked patch transformer with the size and the context length as constructor arguments (v2 hard-codes D 32,
2 layers, 12 patches). With arch=(32,4,128,2) and max_tok 12 it is v2's Encoder2 (same weights shapes). Fine-tune heads at the end."""
import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import ema_update, effective_rank  # noqa: F401,E402

PATCH = 5


def vicreg(z, gamma=1.0):
    d = z.shape[-1]
    z = z - tf.reduce_mean(z, 0)
    std = tf.sqrt(tf.math.reduce_variance(z, 0) + 1e-4)
    var = tf.reduce_mean(tf.nn.relu(gamma - std))
    cov = tf.matmul(z, z, transpose_a=True) / tf.cast(tf.shape(z)[0] - 1, tf.float32)
    off = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    return var, tf.reduce_sum(tf.square(off)) / d


class Block4(keras.layers.Layer):
    def __init__(self, d, heads, ffn, **kw):
        super().__init__(**kw)
        self.ln1, self.ln2 = L.LayerNormalization(), L.LayerNormalization()
        self.att = L.MultiHeadAttention(heads, d // heads)
        self.ff = keras.Sequential([L.Dense(ffn, activation="gelu"), L.Dense(d)])

    def call(self, x, amask=None):
        y = self.ln1(x); x = x + self.att(y, y, attention_mask=amask)
        return x + self.ff(self.ln2(x))


class Encoder4(keras.Model):
    """x [B,T,C], T a multiple of PATCH; vis [B,T/PATCH] bool or None. A hidden patch becomes a learned mask token before the
    transformer and is not an attention key (v2 semantics)."""
    def __init__(self, n_ch, d=32, heads=4, ffn=128, layers=2, max_tok=12, **kw):
        super().__init__(**kw)
        self.n_ch, self.d = n_ch, d
        self.proj = L.Dense(d)
        self.pos = self.add_weight("pos", (max_tok, d), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.mask_tok = self.add_weight("mask_tok", (d,), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.blocks = [Block4(d, heads, ffn) for _ in range(layers)]
        self.ln = L.LayerNormalization()

    def tokens_out(self, x, vis=None):
        b, n = tf.shape(x)[0], x.shape[1] // PATCH
        h = self.proj(tf.reshape(x, (b, n, PATCH * self.n_ch)))
        amask = None
        if vis is not None:
            v = tf.cast(vis, tf.bool)
            h = tf.where(v[..., None], h, tf.zeros_like(h) + self.mask_tok)
            amask = tf.tile(v[:, None, :], (1, n, 1))
        h = h + self.pos[:n]
        for blk in self.blocks:
            h = blk(h, amask)
        return self.ln(h)

    def call(self, x, vis=None):
        out = self.tokens_out(x, vis)
        if vis is None:
            return tf.reduce_mean(out, 1)
        w = tf.cast(vis, tf.float32)[..., None]
        return tf.reduce_sum(out * w, 1) / tf.maximum(tf.reduce_sum(w, 1), 1.0)


class MaskPredictor4(keras.Model):
    def __init__(self, d, heads, ffn, max_tok, **kw):
        super().__init__(**kw)
        self.tok = self.add_weight("mask_tok_p", (d,), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.pos = self.add_weight("pos_p", (max_tok, d), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.block = Block4(d, heads, ffn); self.ln = L.LayerNormalization(); self.out = L.Dense(d)

    def call(self, h, vis):
        n = h.shape[1]
        h = tf.where(tf.cast(vis, tf.bool)[..., None], h, tf.zeros_like(h) + self.tok) + self.pos[:n]
        return self.out(self.ln(self.block(h)))


class SegPredictor4(keras.Model):
    def __init__(self, n_seg, d, **kw):
        super().__init__(**kw)
        self.n_seg = n_seg
        self.seg = self.add_weight("seg_tok", (n_seg, d), initializer=keras.initializers.RandomNormal(stddev=0.5))
        self.net = keras.Sequential([L.Dense(64, activation="gelu"), L.Dense(d)])

    def call(self, z):
        zz = tf.tile(z[:, None, :], (1, self.n_seg, 1))
        return self.net(tf.concat([zz, tf.zeros_like(zz) + self.seg], -1))
