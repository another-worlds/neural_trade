"""One slice of the long file through split_arrays + prepare_datasets; prints JSON of sha256 per array,
the seconds and the peak working set (MB). Usage: trim_check.py <csv> <trim 0|1> [data_end] [cap]."""
import hashlib
import json
import sys
import time

import numpy as np
import psutil

from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor, split_arrays


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    csv, trim = sys.argv[1], bool(int(sys.argv[2]))
    data_end = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 126000
    cfg = Config(CSV_PATH=csv, DATA_END=data_end, MAX_SEQUENCE_COUNT=cap, N_FOLDS=2, VAL_FRACTION=0.1,
                 CAL_FRACTION=0.1, FOLD_INDEX=-2, DATA_TAIL_TRIM=trim)
    t0 = time.perf_counter()
    blocks = split_arrays(cfg)
    out = {}
    ts = blocks["df"]["timestamp"].to_numpy()
    for name in ("train", "val", "cal", "test"):
        b = blocks[name]
        for k in ("X", "X_model", "y", "last_close", "extended_trends", "index"):
            out[f"{name}.{k}"] = h(b[k])
        out[f"{name}.ts"] = h(ts[b["anchor_bar"]].astype("int64"))
    dp = DataProcessor(cfg)
    prep = dp.prepare_datasets(blocks["df"], blocks["close"])
    for i, a in enumerate(prep):
        if isinstance(a, np.ndarray):
            out[f"prep{i}"] = h(a)
    out["n_bars"] = int(len(blocks["df"]))
    out["sec"] = round(time.perf_counter() - t0, 1)
    out["peak_mb"] = round(psutil.Process().memory_info().peak_wset / 2**20)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
