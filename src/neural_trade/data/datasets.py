"""tf.data pipelines for training/validation (moved from PricePredictor.create_datasets in B9).

Elements are ``(window [B, L], y_scaled [B, 3], last_close [B, 1], extended [B, 3])``.
"""
from __future__ import annotations

import tensorflow as tf


def create_datasets(config, X_train, y_train, last_close_train, extended_trends_train,
                    X_test, y_test, last_close_test, extended_trends_test):
    seed = int(getattr(config, "SEED", 42))

    def make_tf_dataset(Xseq, yseq, last_close, extended_trends, batch_size, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((
            Xseq, yseq, last_close.reshape(-1,1), extended_trends
        ))
        if shuffle:
            # Explicit seed: the shuffle order no longer depends on how many random ops were
            # created before it (the implicit op seed does).
            ds = ds.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_tf_dataset(X_train, y_train, last_close_train, extended_trends_train,
                               config.BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(X_test, y_test, last_close_test, extended_trends_test,
                             config.BATCH_SIZE, shuffle=False)
    return train_ds, val_ds
