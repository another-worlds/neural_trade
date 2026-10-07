"""The data plan: which windows exist and how the folds cut them (NT-041).

One place decides, for a prepared bar frame and a Config, the anchors of every sequence (after the hole policy
and the MAX_SEQUENCE_COUNT cap), the folds over them (``tscv`` = TimeSeriesSplit blocks as before, ``timed`` =
blocks of configured wall-clock length placed at configured dates or spacing) and the record of what the holes
cost. The trainer (``data.processor``), the re-scoring path (``split_arrays``) and the experiment engine's layout
(``experiments.dataset.data_layout``) all read it, so they cannot disagree about a block.

    plan = make_plan(config, df)
    plan.anchors              # np.ndarray of anchor bars (the first bar after each window), all folds
    plan.fold(config.FOLD_INDEX)   # FoldIndices into plan.anchors, and plan.read(fold) the span it reads
    plan.gaps                 # the hole record (data.gaps.gap_report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from neural_trade.data.gaps import gap_flags, gap_report, timestamps_ns
from neural_trade.data.splits import (FoldIndices, make_purged_splits, make_timed_splits, timed_fold_starts)
from neural_trade.data.windowing import anchor_positions


@dataclass
class DataPlan:
    anchors: np.ndarray                     # anchor bar of every kept sequence, chronological
    n_total: int                            # valid sequences before the MAX_SEQUENCE_COUNT cap
    folds: List[FoldIndices]                # indices into ``anchors``, chronological
    gaps: Dict[str, Any]
    layout: str = "tscv"
    spans: List[Dict[str, str]] = field(default_factory=list)   # timed layout: per fold, planned start and read span

    @property
    def n_sequences(self) -> int:
        return int(len(self.anchors))

    def fold(self, index: int) -> FoldIndices:
        """The fold at FOLD_INDEX ``index`` (Python indexing over the usable folds; -1 = the latest)."""
        try:
            return self.folds[int(index)]
        except IndexError:
            raise ValueError(f"FOLD_INDEX={index} but only {len(self.folds)} usable folds exist") from None

    def read(self, fold: FoldIndices) -> Tuple[int, int]:
        """``(lo, hi)``: the sequences ``anchors[lo:hi]`` a fold reads (the whole set for tscv)."""
        if self.layout == "tscv":
            return 0, self.n_sequences
        return int(fold.train[0]), int(fold.test[-1]) + 1


def _anchor_times(df, anchors) -> np.ndarray:
    """Decision-bar timestamps (the last input bar) of the anchors, UTC datetime64[ns]."""
    return timestamps_ns(df)[np.asarray(anchors, dtype=np.int64) - 1]


def make_plan(config, df) -> DataPlan:
    """The plan for ``df`` (a prepared OHLCV frame with ``timestamp``) under ``config``."""
    n_bars = len(df)
    policy = str(getattr(config, "GAP_POLICY", "drop"))
    flags = None if policy == "ignore" else gap_flags(df, float(config.RESAMPLE_MINUTES))
    if policy == "refuse" and flags is not None and flags.any():
        rep = gap_report(config, df, flags=flags)
        raise ValueError(f"the data has {rep['n_gaps']} hole(s) in its timestamps ({rep['n_missing_bars']} missing "
                         f"bars, the longest {rep['longest_gap_bars']}) and GAP_POLICY is 'refuse'")
    every = anchor_positions(config, n_bars)
    anchors = every if flags is None else anchor_positions(config, n_bars, flags)
    gaps = gap_report(config, df, len(every), len(anchors), flags=flags if flags is not None else
                      np.zeros(n_bars, dtype=bool))
    layout = str(getattr(config, "FOLD_LAYOUT", "tscv"))
    gap_bars = int(config.LOOKBACK) + int(max(config.HORIZON_STEPS))
    n_total = int(len(anchors))
    if layout == "tscv":
        cap = int(getattr(config, "MAX_SEQUENCE_COUNT", 0) or 0)
        if cap and n_total > cap:
            anchors = anchors[n_total - cap:]
        if not len(anchors):
            raise ValueError(f"{n_bars} bars give no sequence for LOOKBACK {config.LOOKBACK} and HORIZON_STEPS "
                             f"{list(config.HORIZON_STEPS)}")
        folds = make_purged_splits(len(anchors), lookback=config.LOOKBACK, horizon_steps=config.HORIZON_STEPS,
                                   window_step=int(max(1, getattr(config, "WINDOW_STEP", 1))),
                                   n_folds=int(getattr(config, "N_FOLDS", 5)),
                                   val_fraction=float(getattr(config, "VAL_FRACTION", 0.066)),
                                   cal_fraction=float(getattr(config, "CAL_FRACTION", 0.066)))
        return DataPlan(anchors, n_total, folds, gaps, "tscv")
    if not len(anchors):
        raise ValueError(f"{n_bars} bars give no sequence for LOOKBACK {config.LOOKBACK} and HORIZON_STEPS "
                         f"{list(config.HORIZON_STEPS)}")
    ts = _anchor_times(df, anchors)
    kw = dict(bar_minutes=float(config.RESAMPLE_MINUTES), gap_bars=gap_bars, train_minutes=config.TRAIN_MINUTES,
              val_minutes=config.VAL_MINUTES, cal_minutes=config.CAL_MINUTES, test_minutes=config.TEST_MINUTES)
    explicit = getattr(config, "FOLD_STARTS", None)
    starts = timed_fold_starts(ts, n_folds=int(config.N_FOLDS), spacing_days=float(config.FOLD_SPACING_DAYS),
                               starts=explicit, **kw)
    folds, spans = make_timed_splits(ts, starts=starts, strict=explicit is not None, **kw)
    return DataPlan(anchors, n_total, folds, gaps, "timed", spans)


__all__ = ["DataPlan", "make_plan"]
