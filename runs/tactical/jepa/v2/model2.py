"""v2 model: v1's patch transformer (same sizes) with optional patch masking, a masked-token predictor and a segment-conditioned
future predictor. Reuses v1's EMA update, VICReg and effective rank."""
import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import PATCH, D, HEADS, FFN, LAYERS, MAX_TOK, ema_update, vicreg, effective_rank  # noqa: F401,E402


class Block2(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ln1, self.ln2 = L.LayerNormalization(), L.LayerNormalization()
        self.att = L.MultiHeadAttention(HEADS, D // HEADS)
        self.ff = keras.Sequential([L.Dense(FFN, activation="gelu"), L.Dense(D)])

    def call(self, x, amask=None):
        y = self.ln1(x); x = x + self.att(y, y, attention_mask=amask)
        return x + self.ff(self.ln2(x))


class Encoder2(keras.Model):
    """x [B,T,C]; vis [B,T/PATCH] bool or None (None = everything visible). A hidden patch is replaced by a learned mask token
    BEFORE the transformer and is excluded as an attention key, so its raw values cannot reach any visible token's output.
    call() returns the mean over visible tokens; tokens_out() the per-token outputs."""
    def __init__(self, n_ch, **kw):
        super().__init__(**kw)
        self.n_ch = n_ch
        self.proj = L.Dense(D)
        self.pos = self.add_weight("pos", (MAX_TOK, D), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.mask_tok = self.add_weight("mask_tok", (D,), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.blocks = [Block2() for _ in range(LAYERS)]
        self.ln = L.LayerNormalization()

    def tokens_out(self, x, vis=None):
        b, n = tf.shape(x)[0], x.shape[1] // PATCH
        h = self.proj(tf.reshape(x, (b, n, PATCH * self.n_ch)))
        amask = None
        if vis is not None:
            v = tf.cast(vis, tf.bool)
            h = tf.where(v[..., None], h, tf.zeros_like(h) + self.mask_tok)
            amask = tf.tile(v[:, None, :], (1, n, 1))                       # keys: visible patches only
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


class MaskPredictor(keras.Model):
    """visible-token outputs [B,12,D] + vis -> predicted target embeddings at every position [B,12,D]."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.tok = self.add_weight("mask_tok_p", (D,), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.pos = self.add_weight("pos_p", (MAX_TOK, D), initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.block = Block2(); self.ln = L.LayerNormalization(); self.out = L.Dense(D)

    def call(self, h, vis):
        n = h.shape[1]
        h = tf.where(tf.cast(vis, tf.bool)[..., None], h, tf.zeros_like(h) + self.tok) + self.pos[:n]
        return self.out(self.ln(self.block(h)))


class SegPredictor(keras.Model):
    """pooled context embedding z [B,D] -> predicted embeddings of n_seg future segments [B,n_seg,D], conditioned on a learned
    segment token."""
    def __init__(self, n_seg, **kw):
        super().__init__(**kw)
        self.n_seg = n_seg
        self.seg = self.add_weight("seg_tok", (n_seg, D), initializer=keras.initializers.RandomNormal(stddev=0.5))
        self.net = keras.Sequential([L.Dense(64, activation="gelu"), L.Dense(D)])

    def call(self, z):
        zz = tf.tile(z[:, None, :], (1, self.n_seg, 1))
        return self.net(tf.concat([zz, tf.zeros_like(zz) + self.seg], -1))
