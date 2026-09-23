"""Purged chronological splits (moved from model.py in Phase B9)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class FoldIndices:
    """Index blocks of one purged chronological fold (sequence indices, not bar indices)."""
    fold: int
    gap: int
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def make_purged_splits(n_seq, *, lookback, horizon_steps, window_step=1, n_folds=5,
                       val_fraction=0.066, cal_fraction=0.066, gap=None):
    """Chronological train | gap | val | gap | cal | gap | test blocks per TimeSeriesSplit fold.

    A training sequence anchored at bar i carries labels up to bar i + max(H) - 1; a later
    sequence anchored at bar j reads inputs from bar j - LOOKBACK. They share a bar iff
    j - i <= LOOKBACK + max(H) - 1, so a gap of LOOKBACK + max(H) sequences (79 + 1 for the
    default config) guarantees no bar is both a training label and an evaluation input.
    The gap is applied between every adjacent pair of blocks. With WINDOW_STEP > 1 the gap
    is expressed in sequences (ceil of the bar gap / step).

    Returns folds in chronological order; folds whose train block would be empty are omitted.
    """
    horizon_steps = [int(h) for h in horizon_steps]
    bar_gap = int(gap) if gap is not None else int(lookback) + int(max(horizon_steps))
    seq_gap = int(math.ceil(bar_gap / max(1, int(window_step))))
    val_len = max(1, int(round(val_fraction * n_seq)))
    cal_len = max(1, int(round(cal_fraction * n_seq)))

    folds = []
    tscv = TimeSeriesSplit(n_splits=int(n_folds))
    for k, (_, test_idx) in enumerate(tscv.split(np.arange(n_seq)), start=1):
        t0, t1 = int(test_idx[0]), int(test_idx[-1]) + 1
        cal1 = t0 - seq_gap
        cal0 = cal1 - cal_len
        val1 = cal0 - seq_gap
        val0 = val1 - val_len
        train1 = val0 - seq_gap
        if train1 <= 0:
            continue
        folds.append(FoldIndices(
            fold=k, gap=seq_gap,
            train=np.arange(0, train1), val=np.arange(val0, val1),
            cal=np.arange(cal0, cal1), test=np.arange(t0, t1),
        ))
    if not folds:
        raise ValueError(
            f"Not enough sequences ({n_seq}) for a purged split with gap={seq_gap}, "
            f"val_len={val_len}, cal_len={cal_len}, n_folds={n_folds}: add data, raise "
            f"MAX_SEQUENCE_COUNT, or lower VAL_FRACTION / CAL_FRACTION."
        )
    return folds
