"""Window assembly D6b (NT-059; window-free plan README "Window assembly D6b").

Forward: ``tf.gather`` of ``[B, L, C]`` windows at the anchors. Backward: a custom gradient that
transposes by gather - a host-computed inverse map (bar -> batch row; anchors distinct within a batch,
asserted by the caller, not here), one gather of the padded upstream gradient, and a ``reduce_sum``
over L. No segment, scatter, sparse or matmul op
(``docs/research/2026-09-29-window-free-plan/A/FINDINGS.md`` Q2): ``tf.gather``'s own gradient goes
through ``UnsortedSegmentSum``, which raises under op determinism on this TF 2.10 Windows build
(``scripts/bench/common.py``'s ``RAISE_ON_GPU``).

Adapted from ``docs/research/2026-09-29-window-free-plan/A/q2_assembly.py``'s ``D6b`` (the research
prototype also benchmarks D0-D8 for comparison; this module keeps only the recommended design).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def inverse_map(anchors, n_bars, L):
    """Host (numpy) inverse map: ``inv[bar] = b`` where ``anchors[b] == bar``, else ``B`` (a sentinel
    row of zeros). Built outside the graph so it introduces no graph op; the caller must ensure the
    anchors in a batch are distinct (the data pipeline asserts this, not this function)."""
    anchors = np.asarray(anchors)
    B = anchors.shape[0]
    inv = np.full(n_bars + L - 1, B, np.int32)
    inv[anchors] = np.arange(B, dtype=np.int32)
    return inv


def _transpose_by_gather(dy, inv, N, L):
    """dF[n] = sum_l dy[inv[n + L - 1 - l], l] (inv = the batch row of the anchor feeding bar n via
    offset l, or B for "no anchor")."""
    C = dy.shape[-1]
    B = dy.shape[0]
    nl = tf.range(N)[:, None] + (L - 1) - tf.range(L)[None, :]        # [N, L]: anchor bar feeding bar n via l
    flat = tf.gather(inv, nl) * L + tf.range(L)[None, :]
    dyp = tf.concat([tf.reshape(dy, [B * L, C]), tf.zeros([L, C], dy.dtype)], 0)   # rows B*L.. are zero
    return tf.reduce_sum(tf.gather(dyp, flat), axis=1)


def assemble(F, anchors, inv, L):
    """``[B, L, C]`` windows: ``F[anchors[b] - (L - 1) + l, :]`` for ``l`` in ``0..L-1``.

    ``F``: ``[N, C]`` (the series indicator output). ``anchors``: ``[B]`` int32 bar indices
    (``>= L - 1``). ``inv``: ``[N + L - 1]`` int32, from `inverse_map` (a host array fed as a graph
    input, never recomputed inside the graph).
    """
    N = F.shape[0]
    idx = anchors[:, None] - (L - 1) + tf.range(L)[None, :]

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            return _transpose_by_gather(dy, inv, N, L)
        return tf.gather(F, idx), grad
    return g(F)
