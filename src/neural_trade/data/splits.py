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


def timed_fold_starts(ts, *, bar_minutes, gap_bars, train_minutes, val_minutes, cal_minutes, test_minutes,
                      n_folds, spacing_days, starts=None):
    """The training-block start of every timed fold, oldest first (NT-041).

    ``starts`` (timestamps, naive = UTC) are used as given. Without them the newest fold ends at the last
    anchor and the others follow ``spacing_days`` apart going back, ``n_folds`` of them, each one's span being
    its four blocks plus the three purge gaps; a fold that would start before the first anchor is not made."""
    ts = np.asarray(ts, dtype="datetime64[ns]")
    if starts is not None:
        out = []
        for s in starts:
            t = np.datetime64(_naive_utc(s), "ns")
            out.append(t)
        return out
    bar = np.timedelta64(int(round(float(bar_minutes) * 60e9)), "ns")
    span = (_minutes_td(train_minutes) + _minutes_td(val_minutes) + _minutes_td(cal_minutes)
            + _minutes_td(test_minutes) + 3 * (int(gap_bars) - 1) * bar)
    end = ts[-1] + bar
    step = np.timedelta64(int(round(float(spacing_days) * 86400e9)), "ns")
    found = []
    for k in range(int(n_folds)):
        start = end - k * step - span
        if start < ts[0]:
            break
        found.append(start)
    return found[::-1]


def _minutes_td(minutes):
    return np.timedelta64(int(round(float(minutes) * 60e9)), "ns")


def _naive_utc(value):
    import pandas as pd

    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.to_datetime64()


def make_timed_splits(ts, *, bar_minutes, gap_bars, train_minutes, val_minutes, cal_minutes, test_minutes,
                      starts, strict=False):
    """Chronological train | gap | val | gap | cal | gap | test blocks per fold, with block lengths in wall-clock
    time (NT-041), over sequences whose decision-bar timestamps are ``ts`` (sorted; a hole shortens the
    sequence count of a block, never its time span).

    A fold starting at ``S`` takes the sequences in ``[S, S + TRAIN)``; the next block starts ``gap_bars`` bars
    after the last sequence of the one before (the same purge as :func:`make_purged_splits`: a training label
    never lies in a later block's input), and takes the sequences in its own span of minutes; and so on. Returns
    ``(folds, spans)``: the :class:`FoldIndices` (fold numbers 1.. in chronological order, indices into ``ts``)
    and per fold the planned start and the time span actually read (first and last sequence). A fold with an
    empty block is left out (``strict``: an error, naming the block, for explicitly placed folds)."""
    ts = np.asarray(ts, dtype="datetime64[ns]")
    bar = _minutes_td(bar_minutes)
    gap = int(gap_bars) * bar
    lengths = (("train", train_minutes), ("val", val_minutes), ("cal", cal_minutes), ("test", test_minutes))
    folds, spans = [], []
    for start in starts:
        start = np.datetime64(start, "ns")
        cursor = start
        blocks = {}
        ok = True
        for name, minutes in lengths:
            lo = int(np.searchsorted(ts, cursor, side="left"))
            hi = int(np.searchsorted(ts, cursor + _minutes_td(minutes), side="left"))
            if hi <= lo or cursor + _minutes_td(minutes) > ts[-1] + bar:        # empty, or runs past the data
                if strict:
                    raise ValueError(f"the fold starting {np.datetime_as_string(start, unit='m')} has no complete "
                                     f"{name} block ({minutes:g} minutes from "
                                     f"{np.datetime_as_string(cursor, unit='m')}): the data ends at "
                                     f"{np.datetime_as_string(ts[-1], unit='m')} or has a hole there")
                ok = False
                break
            blocks[name] = np.arange(lo, hi)
            cursor = ts[hi - 1] + gap
        if not ok:
            continue
        folds.append(FoldIndices(fold=len(folds) + 1, gap=int(gap_bars), train=blocks["train"], val=blocks["val"],
                                 cal=blocks["cal"], test=blocks["test"]))
        spans.append({"start": np.datetime_as_string(start, unit="s"),
                      "first": np.datetime_as_string(ts[blocks["train"][0]], unit="s"),
                      "last": np.datetime_as_string(ts[blocks["test"][-1]], unit="s")})
    if not folds:
        raise ValueError(f"no timed fold fits the data ({len(ts)} sequences from {np.datetime_as_string(ts[0], unit='m')} "
                         f"to {np.datetime_as_string(ts[-1], unit='m')}): shorten TRAIN_MINUTES / VAL_MINUTES / "
                         "CAL_MINUTES / TEST_MINUTES, lower N_FOLDS, or use more data")
    return folds, spans
