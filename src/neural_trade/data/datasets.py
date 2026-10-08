"""tf.data pipelines for training/validation (moved from PricePredictor.create_datasets in B9).

Elements are ``(window [B, L], y_scaled [B, 3], last_close [B, 1], extended [B, 3])``.

Only the training dataset is shuffled, with a buffer of ``Config.SHUFFLE_BUFFER`` windows (default
2048; 0 = the whole training block, a full reshuffle every epoch), seeded by ``Config.SEED`` and
reshuffled each epoch. On a long training block a 2048-window buffer draws every batch from about
1.4 days of consecutive one-minute windows; the full buffer mixes the whole block. Measured on the
CPU with 518,428 windows of 60 bars (NT-082): one pass of the pipeline alone takes about 0.7-0.8 s
with the 2048 buffer and about 1.4-1.5 s with the full buffer, at batch 256 or 2048.
"""
from __future__ import annotations

import tensorflow as tf

DEFAULT_SHUFFLE_BUFFER = 2048


def shuffle_buffer_size(config, n_train: int) -> int:
    """The training shuffle buffer in windows: ``Config.SHUFFLE_BUFFER``, 0 meaning all ``n_train``."""
    size = int(getattr(config, "SHUFFLE_BUFFER", DEFAULT_SHUFFLE_BUFFER))
    return max(int(n_train), 1) if size == 0 else size


def create_datasets(config, X_train, y_train, last_close_train, extended_trends_train,
                    X_test, y_test, last_close_test, extended_trends_test, *,
                    path_train=None, path_test=None):
    """Train/val ``tf.data`` pipelines of ``(X, y, last_close, extended_trends)`` batches. With
    ``path_train`` / ``path_test`` (Config.PATH_HEAD: the scaled future-path targets ``[N, P]``) every
    batch carries a fifth element; without them (the default) the batches are the 4-tuples of before."""
    seed = int(getattr(config, "SEED", 42))

    def make_tf_dataset(Xseq, yseq, last_close, extended_trends, batch_size, shuffle=False, path=None):
        parts = (Xseq, yseq, last_close.reshape(-1,1), extended_trends)
        if path is not None:
            parts = parts + (path,)
        ds = tf.data.Dataset.from_tensor_slices(parts)
        if shuffle:
            # Explicit seed: the shuffle order no longer depends on how many random ops were
            # created before it (the implicit op seed does).
            ds = ds.shuffle(buffer_size=shuffle_buffer_size(config, len(Xseq)), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_tf_dataset(X_train, y_train, last_close_train, extended_trends_train,
                               config.BATCH_SIZE, shuffle=True, path=path_train)
    val_ds = make_tf_dataset(X_test, y_test, last_close_test, extended_trends_test,
                             config.BATCH_SIZE, shuffle=False, path=path_test)
    return train_ds, val_ds
