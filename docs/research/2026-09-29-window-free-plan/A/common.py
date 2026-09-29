"""Shared helpers for NT-053 part A prototypes (CPU only; nothing in D:/neural_trade is modified).

Every script: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python <script>
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import neural_trade  # noqa: E402,F401  (sets TF_DETERMINISTIC_OPS=1 and the DLL path before TF loads)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

REPO = "D:/neural_trade"
CSV = os.path.join(REPO, "binance_btcusdt_1min_ccxt.csv")
LONG_CSV = os.path.join(REPO, "Bitcoin_BTCUSDT.csv")

# ----------------------------------------------------------------------------------------------
# Ops that have no deterministic GPU kernel in this TF 2.10 Windows build (they raise when op
# determinism is on, which `import neural_trade` switches on via TF_DETERMINISTIC_OPS=1), and ops
# whose deterministic GPU path is a host round trip. Evidence: strings in
# _pywrap_tensorflow_internal.pyd (det_strings_all.txt) and the v2.10.0 sources
# (segment_reduction_ops_impl.h: sorted segment ops have no deterministic kernel on Windows;
# scatter_nd_op.cc: ScatterNd-family ops run DoScatterNdOnCpu + BlockHostUntilDone).
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
# Ops that do not launch a GPU kernel (metadata / host-side / folded); excluded from "compute ops".
NO_LAUNCH = {
    "Const", "Identity", "IdentityN", "NoOp", "Placeholder", "Shape", "ShapeN", "Size", "Rank",
    "StopGradient", "ReadVariableOp", "VarHandleOp", "Reshape", "ExpandDims", "Squeeze",
    "BroadcastGradientArgs", "PreventGradient", "_Arg", "_Retval", "PartitionedCall",
    "StatefulPartitionedCall", "VarIsInitializedOp", "AssignVariableOp",
}


def graph_census(fn, *args):
    """Op census of a tf.function's concrete graph (top-level nodes plus library-function bodies)."""
    cf = tf.function(fn).get_concrete_function(*args)
    g = cf.graph
    types = [op.type for op in g.get_operations()]
    for f in g.as_graph_def().library.function:
        types += [n.op for n in f.node_def]
    counts = {}
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
    return {
        "total_ops": len(types),
        "compute_ops": sum(v for k, v in counts.items() if k not in NO_LAUNCH),
        "types": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "raise_on_gpu": sorted(t for t in counts if t in RAISE_ON_GPU),
        "host_round_trip_on_gpu": sorted(t for t in counts if t in HOST_ROUND_TRIP_ON_GPU),
        "static_output_bytes": int(out_bytes),
        "largest_output_bytes": int(max(
            [int(np.prod(o.shape.as_list())) * max(1, o.dtype.size)
             for op in g.get_operations() for o in op.outputs if o.shape.is_fully_defined()] or [0])),
    }


def interleaved_timing(variants, reps=7, warm=1):
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


def load_close(path=CSV):
    return pd.read_csv(path, usecols=["close"])["close"].to_numpy(np.float64)


_SCALE = None


def target_scale():
    """The model's input/target scale: StandardScaler.scale_ of the pooled training deltas of the default
    config's fold -1 on the bundled file (what WindowNormalizer divides by)."""
    global _SCALE
    if _SCALE is None:
        import logging
        logging.getLogger("neural_trade").setLevel(logging.WARNING)
        from neural_trade.core.config import Config
        from neural_trade.data.processor import split_arrays
        from neural_trade.data.scaling import fit_target_scaler
        cfg = Config(CSV_PATH=CSV)
        s = split_arrays(cfg)
        _SCALE = float(fit_target_scaler(s["train"]["y"]).scale_[0])
    return _SCALE


def dump(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path)
    return path


def env_info():
    return {"tf": tf.__version__, "device": "CPU (CUDA_VISIBLE_DEVICES=-1)",
            "determinism": bool(__import__("tensorflow.python.util._pywrap_determinism",
                                           fromlist=["x"]).is_enabled()),
            "tf32_default": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "cpu_count": os.cpu_count()}
