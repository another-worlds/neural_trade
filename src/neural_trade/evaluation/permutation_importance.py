"""Grouped permutation importance of indicator channels (NT-048). Nothing under ``training/`` reads this.

One group is one family instance: every channel of that instance, and every timestep of those
channels, is shuffled by the same permutation of the windows. The raw close appended after the
indicator channels is not a group. Importance is the mean per-window change. Loss rising is
positive (``shuffled loss_i - baseline loss_i``). Direction skill falling is positive: ``auc`` is
the per-horizon drop of the ROC AUC (``baseline AUC - permuted AUC``), its band the 2.5 and 97.5
percentiles of the same drop recomputed on each moving-block bootstrap resample (``metrics.statistics``,
D-012; the block has to cover the horizon in bars, so a resample keeps the overlap of the labels).
``hit_drop`` is the mean drop of the per-window direction hit, kept beside it under its own name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from neural_trade.metrics.statistics import BLOCK, BOOT_N, block_bootstrap_counts


@dataclass(frozen=True)
class GroupImportance:
    """One family instance: loss importance, direction AUC drop and hit-rate drop, with their bootstrap bands."""

    name: str
    loss: float
    loss_lo: float
    loss_hi: float
    auc: Mapping[str, float]
    auc_lo: Mapping[str, float]
    auc_hi: Mapping[str, float]
    hit_drop: Mapping[str, float]
    hit_lo: Mapping[str, float]
    hit_hi: Mapping[str, float]
    auc_base: Mapping[str, float] = field(default_factory=dict)   # the unshuffled AUC per horizon


def indicator_channel_groups(config) -> List[Tuple[str, slice]]:
    """``(name, channel slice)`` for every configured family instance, in layer-column order.

    The slice indexes the last axis of the indicator layer output. The trailing raw-close
    channel that the layer appends is not a group and is not covered by any slice.
    """
    from neural_trade.indicators import Indicators, indicator_instances

    groups: List[Tuple[str, slice]] = []
    start = 0
    for name, raw in indicator_instances(config).items():
        width = len(Indicators.get(name).channels)
        for i in range(len(raw)):
            groups.append((f"{name} #{i}", slice(start, start + width)))
            start += width
    return groups


def _vector(value, n: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{what} must have one value per window ({n}), got {arr.shape}")
    return arr


def _score_parts(score: Mapping, n: int):
    missing = [key for key in ("loss", "loss_i", "hit_i", "labels", "scores") if key not in score]
    if missing:
        raise ValueError(f"score_fn must return {', '.join(missing)}")
    for key in ("hit_i", "labels", "scores"):
        if not isinstance(score[key], Mapping):
            raise ValueError(f"{key} must be a mapping over horizons")
    if not (set(score["hit_i"]) == set(score["labels"]) == set(score["scores"])):
        raise ValueError("hit_i, labels and scores must name the same horizons")
    loss_i = _vector(score["loss_i"], n, "loss_i").copy()
    hits = {str(h): _vector(score["hit_i"][h], n, f"hit_i[{h}]").copy() for h in score["hit_i"]}
    labels = {str(h): _vector(score["labels"][h], n, f"labels[{h}]").copy() for h in score["labels"]}
    scores = {str(h): _vector(score["scores"][h], n, f"scores[{h}]").copy() for h in score["scores"]}
    return loss_i, hits, labels, scores


def _weighted_auc(labels, scores, weights) -> np.ndarray:
    """ROC AUC of each weight row (``[B, n]`` multiplicities); NaN label = unlabelled, ties count half.

    Pairs are weighted by the product of the two windows' weights, which is the AUC of the resample.
    """
    keep = np.isfinite(labels) & np.isfinite(scores)
    pos = keep & (labels >= 0.5)
    neg = keep & (labels < 0.5)
    idx = np.flatnonzero(keep)
    out = np.full(weights.shape[0], np.nan)
    if not pos.any() or not neg.any():
        return out
    order = idx[np.argsort(scores[idx], kind="mergesort")]
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    w = weights[:, order]
    wp = np.add.reduceat(w * pos[order], starts, axis=1)
    wn = np.add.reduceat(w * neg[order], starts, axis=1)
    below = np.cumsum(wn, axis=1) - wn
    denom = wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (wp * (below + 0.5 * wn)).sum(axis=1) / denom
    return np.where(denom > 0, out, np.nan)


def _percentiles(values: np.ndarray) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _band(counts: np.ndarray, weight: np.ndarray, diff: np.ndarray) -> Tuple[float, float]:
    means = counts @ diff / weight
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _channel_slice(sl, n_channels: int) -> slice:
    if not isinstance(sl, slice):
        raise ValueError(f"a group channel selector must be a slice, got {sl!r}")
    start, stop, step = sl.indices(n_channels)
    if step != 1 or stop <= start:
        raise ValueError(f"a group channel slice must be a non-empty contiguous range, got {sl}")
    return slice(start, stop)


def grouped_importance(features, baseline_scores, score_fn, groups, *, block: int = BLOCK,
                       n_boot: int = BOOT_N, seed: int = 0, horizon_bars: int = 1) -> List[GroupImportance]:
    """Importance of each group on ``features`` of shape ``[N, LOOKBACK, C]``.

    ``baseline_scores`` is ``score_fn(features)`` on the unshuffled windows. ``score_fn`` is
    called once per group on a copy whose group channels have been reordered by one permutation
    along the window axis (the same order at every timestep and every channel of that group).
    ``block`` and ``n_boot`` are the moving-block bootstrap (default 80 and 500); ``block`` must be at
    least ``horizon_bars`` (the longest horizon in bars). ``score_fn`` returns ``loss``, ``loss_i``,
    ``hit_i``, and per horizon ``labels`` (NaN where unlabelled) and ``scores`` (the P(up)), from which
    the AUC is computed.
    """
    if int(block) < int(horizon_bars):
        raise ValueError(f"block ({block}) has to cover the horizon ({horizon_bars} bars)")
    base = np.asarray(features, dtype=float)
    if base.ndim != 3:
        raise ValueError(f"features must be [N, LOOKBACK, C], got {base.shape}")
    n, _lookback, n_channels = base.shape
    if n < 1:
        raise ValueError("features has no windows")
    base_loss, base_hits, base_labels, base_scores = _score_parts(baseline_scores, n)
    horizons = list(base_hits)
    counts = block_bootstrap_counts(n, block=block, n_boot=n_boot, seed=seed)
    weight = counts.sum(axis=1)
    base_auc_boot = {h: _weighted_auc(base_labels[h], base_scores[h], counts) for h in horizons}
    base_auc = {h: float(_weighted_auc(base_labels[h], base_scores[h], np.ones((1, n)))[0]) for h in horizons}
    rng = np.random.default_rng(seed)
    work = np.array(base, copy=True)
    out: List[GroupImportance] = []
    for name, sl in groups:
        sl = _channel_slice(sl, n_channels)
        perm = rng.permutation(n)
        saved = work[:, :, sl].copy()
        work[:, :, sl] = base[perm][:, :, sl]
        try:
            loss_i, hits, labels, scores = _score_parts(score_fn(work), n)
            if set(hits) != set(base_hits):
                raise ValueError("score_fn changed the horizons it returns")
        finally:
            work[:, :, sl] = saved
        diff = loss_i - base_loss
        loss_lo, loss_hi = _band(counts, weight, diff)
        auc, auc_lo, auc_hi = {}, {}, {}
        hit_drop, hit_lo, hit_hi = {}, {}, {}
        for h in horizons:
            # the hit rate is over the labelled windows only (an unlabelled window has no hit to lose)
            lab = np.isfinite(base_labels[h])
            gap = (base_hits[h] - hits[h])[lab]
            if lab.any():
                hit_drop[h] = float(gap.mean())
                sub = counts[:, lab]
                hit_lo[h], hit_hi[h] = _band(sub, sub.sum(axis=1), gap)
            else:
                hit_drop[h] = hit_lo[h] = hit_hi[h] = float("nan")
            perm_auc = _weighted_auc(labels[h], scores[h], np.ones((1, n)))[0]
            auc[h] = float(base_auc[h] - perm_auc)
            auc_lo[h], auc_hi[h] = _percentiles(base_auc_boot[h] - _weighted_auc(labels[h], scores[h], counts))
        out.append(GroupImportance(str(name), float(diff.mean()), loss_lo, loss_hi, auc, auc_lo, auc_hi,
                                   hit_drop, hit_lo, hit_hi, base_auc))
    return out


def _as_numpy(value):
    if isinstance(value, (list, tuple)):
        return [_as_numpy(part) for part in value]
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _concat(parts):
    if isinstance(parts[0], list):
        return [_concat([part[i] for part in parts]) for i in range(len(parts[0]))]
    return np.concatenate(parts, axis=0)


def _run_batched(model, values, batch: int):
    import tensorflow as tf

    arrays = list(values) if isinstance(values, (list, tuple)) else [values]
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    n = int(arrays[0].shape[0])
    step = max(1, int(batch))
    parts = []
    for start in range(0, n, step):
        feeds = [tf.constant(arr[start:start + step]) for arr in arrays]
        out = model(feeds[0] if len(feeds) == 1 else feeds, training=False)
        parts.append(_as_numpy(out))
    return _concat(parts)


def _probe_and_tail(model, layer):
    """The indicator features, and the rest of the model from those features.

    Returns ``(probe, tail, tail_takes_windows)``. ``probe`` reads the features from the model
    inputs. ``tail`` maps those features to the model outputs. When a later layer still reads the
    raw window (the energy gate, the direction skip), the features alone cannot rebuild the
    outputs, and ``tail`` takes the original inputs plus the features. The windows stay in their
    own order: a shuffle replaces indicator channels, not the price path those skips read.
    """
    import tensorflow as tf

    probe = tf.keras.Model(model.inputs, layer.output)
    try:
        tail = tf.keras.Model(layer.output, model.output)
        return probe, tail, False
    except ValueError:
        tail = tf.keras.Model([*model.inputs, layer.output], model.output)
        return probe, tail, True


def importance_from_model(model, windows, score_fn, *, groups: Sequence[Tuple[str, slice]] | None = None,
                          block: int = BLOCK, n_boot: int = BOOT_N, seed: int = 0, batch: int = 256,
                          horizon_bars: int = 1):
    """Permutation importance through the graph after the ``learnable_indicators`` layer.

    ``windows`` is the model input ``[N, LOOKBACK]`` or ``[N, LOOKBACK, C]``. The indicator
    features are read from that layer, and ``score_fn`` is called with the remainder of the
    model, in batches. Predictions are numpy arrays, or a list of them when the model has
    several outputs. A later layer that still reads the raw window keeps those windows in
    their original order; only the indicator features are shuffled.
    """
    layer = model.get_layer("learnable_indicators")
    if groups is None:
        groups = indicator_channel_groups(layer.config)
    try:
        probe, tail, takes_windows = _probe_and_tail(model, layer)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot cut the graph after learnable_indicators; the model has to be functional"
        ) from exc
    windows_np = np.asarray(windows, dtype=np.float32)
    features = _run_batched(probe, windows_np, batch)

    def score_features(feats):
        feed = [windows_np, np.asarray(feats, dtype=np.float32)] if takes_windows else feats
        return score_fn(_run_batched(tail, feed, batch))

    return grouped_importance(features, score_features(features), score_features, groups,
                              block=block, n_boot=n_boot, seed=seed, horizon_bars=horizon_bars)
