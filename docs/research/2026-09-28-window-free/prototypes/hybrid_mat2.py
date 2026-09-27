"""hybrid_proto.SeriesIndicators with the two-level matrix recurrence (C=64) instead of Hillis-Steele."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
import hybrid_proto as hp
from mat2_proto import scan_mat2


def linrec_mat2(a, b, C=64):
    T = a.shape[-1]; pad = (-T) % C
    if pad:  # pad the FUTURE end: causal, so earlier outputs are unaffected
        a = tf.concat([a, tf.ones_like(a[..., :pad])], -1); b = tf.concat([b, tf.zeros_like(b[..., :pad])], -1)
    return scan_mat2(a, b, C)[..., :T]


hp.linrec = linrec_mat2
hp.main()
