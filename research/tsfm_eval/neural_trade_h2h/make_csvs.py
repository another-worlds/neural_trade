"""Write header-renamed (datetime -> timestamp) copies of the benchmark CSVs: full file + training slice
rows [0, first_origin] (inclusive). Values are copied as strings, unchanged."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
SCR_WORK = _os.environ.get("TSFM_EVAL_WORK", _os.path.join(_HERE, "..", "work"))
REPO_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

import sys
import pandas as pd
import numpy as np
sys.path.insert(0, _os.path.join(_HERE, "..", "bench"))
import importlib, os
OUT = _os.path.join(SCR_WORK, "nt_h2h", "data")
for data in ["repo", "post"]:
    os.environ["BENCH_DATA"] = data
    import common; importlib.reload(common)
    df_c, info = common.load()
    org = common.origins(len(df_c))
    raw = pd.read_csv(common.CSV, dtype=str)
    ts = pd.to_datetime(raw["datetime"])
    assert ts.is_monotonic_increasing and ts.is_unique, data
    assert len(raw) == len(df_c)
    assert np.array_equal(raw["close"].astype(float).to_numpy(), df_c["close"].to_numpy()), data
    raw = raw.rename(columns={"datetime": "timestamp"})
    raw.to_csv(f"{OUT}/{data}_full.csv", index=False)
    t0 = int(org[0])
    raw.iloc[: t0 + 1].to_csv(f"{OUT}/{data}_train_upto_{t0}.csv", index=False)
    print(data, len(raw), "t0", t0, raw["timestamp"].iloc[t0], "n_origins", len(org))
