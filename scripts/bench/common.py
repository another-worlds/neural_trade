"""Shared helpers for the window-free benchmark kit (NT-059).

Op census and interleaved timing, and the two op sets that raise (or fall back to the host) under TF
op determinism (``TF_DETERMINISTIC_OPS=1``, switched on by ``import neural_trade``) on this TF 2.10
Windows build. Both sets, and their citations, are copied from
``docs/research/2026-09-29-window-free-plan/A/common.py`` (the NT-053 research prototype); evidence:
strings in ``_pywrap_tensorflow_internal.pyd`` (that folder's ``det_strings_all.txt``) and the TF
2.10.0 sources:

- ``GatherV2``'s gradient goes dense through ``UnsortedSegmentSum`` (``indexed_slices.py:447``);
- ``GatherNd``'s gradient uses ``ScatterNd``;
- ``ExtractImagePatches``'s gradient uses ``SparseTensorDenseMatMul``;
- ``segment_reduction_ops_impl.h`` sets ``use_deterministic_kernels = false`` under
  ``PLATFORM_WINDOWS``, so the *sorted* segment ops raise too on this build;
- ``scatter_nd_op.cc`` runs ``DoScatterNdOnCpu`` plus ``BlockHostUntilDone`` when determinism is on.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

RAISE_ON_GPU = {
    "UnsortedSegmentSum", "UnsortedSegmentProd",                 # float sums: non-associative atomics
    "SegmentSum", "SegmentProd", "SegmentMean",                   # sorted: PLATFORM_WINDOWS -> no det. kernel
    "SparseSegmentSum", "SparseSegmentMean", "SparseSegmentSqrtN",
    "SparseSegmentSumWithNumSegments", "SparseSegmentMeanWithNumSegments",
    "SparseSegmentSqrtNWithNumSegments", "SparseSegmentSumGrad", "SparseSegmentMeanGrad",
    "SparseSegmentSqrtNGrad",
    "SparseTensorDenseMatMul",                                   # ExtractImagePatches' gradient uses it
    "CropAndResizeGradImage", "CropAndResizeGradBoxes", "Bincount", "DenseBincount",
    "Dilation2DBackpropFilter", "Dilation2DBackpropInput", "AdjustContrastv2",
    "ResizeNearestNeighborGrad", "FakeQuantWithMinMaxVarsGradient",
    "FakeQuantWithMinMaxVarsPerChannelGradient", "MaxPoolGradWithArgmax", "Svd",
    "SoftmaxCrossEntropyWithLogits", "SparseSoftmaxCrossEntropyWithLogits", "Timestamp",
    "ScatterAdd", "ScatterSub", "ScatterMul", "ScatterDiv", "ScatterMin", "ScatterMax",
    "ScatterUpdate",                                              # ref-input scatter ops
}
HOST_ROUND_TRIP_ON_GPU = {
    "ScatterNd", "ScatterNdAdd", "ScatterNdSub", "ScatterNdUpdate", "ScatterNdMin", "ScatterNdMax",
    "TensorScatterAdd", "TensorScatterSub", "TensorScatterUpdate", "TensorScatterMin",
    "TensorScatterMax", "ResourceScatterNdAdd", "ResourceScatterNdSub", "ResourceScatterNdUpdate",
}
# Contraction ops the kernel and the assembly must never use: TF 2.10 runs every one of these in
# TensorFloat-32 on an Ampere/Ada GPU by default (window-free plan README "Answer"; A/FINDINGS.md Q5).
MATMUL_LIKE = {"MatMul", "BatchMatMul", "BatchMatMulV2", "Einsum"}
# Ops that do not launch a GPU kernel (metadata / host-side / folded); excluded from "compute ops".
NO_LAUNCH = {
    "Const", "Identity", "IdentityN", "NoOp", "Placeholder", "Shape", "ShapeN", "Size", "Rank",
    "StopGradient", "ReadVariableOp", "VarHandleOp", "Reshape", "ExpandDims", "Squeeze",
    "BroadcastGradientArgs", "PreventGradient", "_Arg", "_Retval", "PartitionedCall",
    "StatefulPartitionedCall", "VarIsInitializedOp", "AssignVariableOp",
}


def graph_census(fn, *args):
    """Op census of a tf.function's concrete graph (top-level nodes plus library-function bodies).

    ``fn`` may be a plain callable (wrapped here) or an already-``tf.function``-wrapped one, so a
    caller that also times the same callable can trace it once and reuse the concrete function."""
    tf_fn = fn if hasattr(fn, "get_concrete_function") else tf.function(fn)
    cf = tf_fn.get_concrete_function(*args)
    g = cf.graph
    types = [op.type for op in g.get_operations()]
    for f in g.as_graph_def().library.function:
        types += [n.op for n in f.node_def]
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    out_bytes = 0
    for op in g.get_operations():
        for o in op.outputs:
            try:
                n = int(np.prod(o.shape.as_list()))
                out_bytes += n * max(1, o.dtype.size)
            except (TypeError, ValueError):
                pass
    largest = [int(np.prod(o.shape.as_list())) * max(1, o.dtype.size)
               for op in g.get_operations() for o in op.outputs if o.shape.is_fully_defined()]
    return {
        "total_ops": len(types),
        "compute_ops": sum(v for k, v in counts.items() if k not in NO_LAUNCH),
        "types": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "raise_on_gpu": sorted(t for t in counts if t in RAISE_ON_GPU),
        "host_round_trip_on_gpu": sorted(t for t in counts if t in HOST_ROUND_TRIP_ON_GPU),
        "matmul_like_ops": sorted(t for t in counts if t in MATMUL_LIKE),
        "static_output_bytes": int(out_bytes),
        "largest_output_bytes": int(max(largest or [0])),
    }


def interleaved_timing(variants, reps=5, warm=1):
    """variants: {name: zero-arg callable}. Interleaved repeats (A B C A B C ...) so background load on
    the shared CPU hits every variant alike. Returns {name: {median_s, min_s, max_s, iqr_s, reps}}."""
    for f in variants.values():
        for _ in range(warm):
            f()
    times = {k: [] for k in variants}
    for _ in range(reps):
        for k, f in variants.items():
            t0 = time.perf_counter()
            f()
            times[k].append(time.perf_counter() - t0)
    res = {}
    for k, v in times.items():
        v = np.asarray(v)
        res[k] = {"median_s": float(np.median(v)), "min_s": float(v.min()), "max_s": float(v.max()),
                  "iqr_s": float(np.percentile(v, 75) - np.percentile(v, 25)), "reps": int(len(v))}
    return res


GATE_STATISTIC = ("median of >= 20 interleaved A/B repeats per arm (warm-up repeats excluded), ratio = "
                  "median(A) / median(B); spread = IQR (p75 - p25) of each arm")


def stable_ratio_gate(sample_a, sample_b, reps=20, warm=2, threshold=1.10):
    """NT-120: the G-A2 gate (A's time at most ``threshold`` x B's) on a stable statistic.

    ``sample_a`` / ``sample_b``: zero-arg callables that run one repeat and return its duration in
    seconds (a stub in the tests, a ``perf_counter`` wrapper on real work). ``warm`` repeats of each are
    run first and dropped; then ``reps`` (>= 20) repeats are taken interleaved A, B, A, B, ... so
    background load hits both arms alike. The statistic is the median of each arm and the ratio of the
    medians, so a few outlier repeats (the recorded 3.4-30.8 ms denominator) cannot flip the call; the
    IQR of each arm is reported beside it."""
    if reps < 20:
        raise ValueError(f"the G-A2 gate needs at least 20 repeats, got {reps}")
    for _ in range(warm):
        sample_a()
        sample_b()
    a, b = [], []
    for _ in range(reps):
        a.append(float(sample_a()))
        b.append(float(sample_b()))

    def stats(v):
        v = np.asarray(v)
        return {"median_s": float(np.median(v)), "iqr_s": float(np.percentile(v, 75) - np.percentile(v, 25)),
                "min_s": float(v.min()), "max_s": float(v.max()), "reps": int(len(v))}

    sa, sb = stats(a), stats(b)
    ratio = sa["median_s"] / sb["median_s"]
    return {"ratio": float(ratio), "threshold": float(threshold), "PASS": bool(ratio <= threshold),
            "numerator": sa, "denominator": sb, "warm_excluded": int(warm),
            "statistic": GATE_STATISTIC}


def timed(fn):
    """A ``sample`` callable for ``stable_ratio_gate``: one call of ``fn``, its wall time in seconds."""
    def sample():
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0
    return sample


def load_close(csv_path):
    return pd.read_csv(csv_path, usecols=["close"])["close"].to_numpy(np.float64)


def target_scale(csv_path):
    """The model's input/target scale: StandardScaler.scale_ of fold -1's training deltas on the given
    CSV (what WindowNormalizer divides by), via the default Config's purged split."""
    import logging

    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.data.scaling import fit_target_scaler

    # Quiet the split's INFO logging without leaking a changed level: this runs inside a shared
    # pytest process (the fast suite), and a bare setLevel() here left "neural_trade" at WARNING for
    # every later test, silencing another test's capsys-captured logger.info() output.
    logger = logging.getLogger("neural_trade")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        cfg = Config(CSV_PATH=csv_path)
        s = split_arrays(cfg)
    finally:
        logger.setLevel(previous_level)
    return float(fit_target_scaler(s["train"]["y"]).scale_[0])


def determinism_enabled():
    try:
        from tensorflow.python.util import _pywrap_determinism
        return bool(_pywrap_determinism.is_enabled())
    except Exception:
        return None


def env_info(device="cpu"):
    return {
        "device": device,
        "tf_version": tf.__version__,
        "op_determinism_enabled": determinism_enabled(),
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "cpu_count": os.cpu_count(),
    }
