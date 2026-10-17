"""M1 Chronos-2 minimal loader + smoke. Run: PYTHONPATH=<T14 dir> D:/nt/nt_mm_env/python.exe load.py"""
import os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch
from common_data import bars_5m
from chronos import Chronos2Pipeline

def load(name="amazon/chronos-2"):
    return Chronos2Pipeline.from_pretrained(name, device_map="cuda")

if __name__ == "__main__":
    t0 = time.time(); pipe = load(); r = bars_5m(512)
    df = pd.DataFrame({"item_id": "btc", "timestamp": r.index, "target": r.logrv.values})
    torch.cuda.reset_peak_memory_stats()
    out = pipe.predict_df(df, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
    print(out.head(3)); print("rows", len(out), "secs", round(time.time() - t0, 1), "peak_gb", torch.cuda.max_memory_allocated() / 1e9)
