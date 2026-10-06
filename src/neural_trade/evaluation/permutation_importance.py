"""Grouped permutation importance of indicator channels (NT-048). Nothing under ``training/`` reads this.

One group is one family instance: every channel of that instance, and every timestep of those
channels, is shuffled by the same permutation of the windows. The raw close appended after the
indicator channels is not a group. Importance is the mean per-window change. Loss rising is
positive (``shuffled loss_i - baseline loss_i``). Direction hits falling are positive
(``baseline hit_i - shuffled hit_i``). The band is the 2.5 and 97.5 percentiles of the
moving-block bootstrap means of that per-window difference (``metrics.statistics``, D-012).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from neural_trade.metrics.statistics import BLOCK, BOOT_N, block_bootstrap_counts


@dataclass(frozen=True)
class GroupImportance:
    """One family instance: loss importance, direction-hit importance, and the bootstrap band of each."""

    name: str
    loss: float
    loss_lo: float
    loss_hi: float
    auc: Mapping[str, float]
    auc_lo: Mapping[str, float]
    auc_hi: Mapping[str, float]


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


def _score_parts(score: Mapping, n: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    missing = [key for key in ("loss", "auc", "loss_i", "hit_i") if key not in score]
    if missing:
        raise ValueError(f"score_fn must return {', '.join(missing)}")
    if not isinstance(score["auc"], Mapping) or not isinstance(score["hit_i"], Mapping):
        raise ValueError("auc and hit_i must be mappings over horizons")
    if set(score["hit_i"]) != set(score["auc"]):
        raise ValueError("hit_i and auc must name the same horizons")
    loss_i = _vector(score["loss_i"], n, "loss_i").copy()
    hits = {str(h): _vector(score["hit_i"][h], n, f"hit_i[{h}]").copy() for h in score["hit_i"]}
    return loss_i, hits


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
                       n_boot: int = BOOT_N, seed: int = 0) -> List[GroupImportance]:
    """Importance of each group on ``features`` of shape ``[N, LOOKBACK, C]``.

    ``baseline_scores`` is ``score_fn(features)`` on the unshuffled windows. ``score_fn`` is
    called once per group on a copy whose group channels have been reordered by one permutation
    along the window axis (the same order at every timestep and every channel of that group).
    ``block`` and ``n_boot`` are the moving-block bootstrap (default 80 and 500).
    """
    base = np.asarray(features, dtype=float)
    if base.ndim != 3:
        raise ValueError(f"features must be [N, LOOKBACK, C], got {base.shape}")
    n, _lookback, n_channels = base.shape
    if n < 1:
        raise ValueError("features has no windows")
    base_loss, base_hits = _score_parts(baseline_scores, n)
    counts = block_bootstrap_counts(n, block=block, n_boot=n_boot, seed=seed)
    weight = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    work = np.array(base, copy=True)
    out: List[GroupImportance] = []
    for name, sl in groups:
        sl = _channel_slice(sl, n_channels)
        perm = rng.permutation(n)
        saved = work[:, :, sl].copy()
        work[:, :, sl] = base[perm][:, :, sl]
        try:
            loss_i, hits = _score_parts(score_fn(work), n)
            if set(hits) != set(base_hits):
                raise ValueError("score_fn changed the horizons it returns")
        finally:
            work[:, :, sl] = saved
        diff = loss_i - base_loss
        loss_lo, loss_hi = _band(counts, weight, diff)
        auc: Dict[str, float] = {}
        auc_lo: Dict[str, float] = {}
        auc_hi: Dict[str, float] = {}
        for h, base_h in base_hits.items():
            gap = base_h - hits[h]
            lo, hi = _band(counts, weight, gap)
            auc[h] = float(gap.mean())
            auc_lo[h] = lo
            auc_hi[h] = hi
        out.append(GroupImportance(str(name), float(diff.mean()), loss_lo, loss_hi, auc, auc_lo, auc_hi))
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
                          block: int = BLOCK, n_boot: int = BOOT_N, seed: int = 0, batch: int = 256):
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
                              block=block, n_boot=n_boot, seed=seed)
