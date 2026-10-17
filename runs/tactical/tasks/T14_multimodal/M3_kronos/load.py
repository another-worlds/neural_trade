"""M3 Kronos-small minimal loader + smoke. Run with D:/nt/nt_mm_legacy/python.exe (py3.10, torch 2.5.1+cu124).
Needs the clone D:/nt/nt_mm_cache/src_kronos (commit 67b630e) on sys.path. 1-min OHLCV, ctx 400, pred_len 60 (labels of future bars are synthetic)."""
import os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
sys.path.insert(0, "D:/nt/nt_mm_cache/src_kronos"); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch
from common_data import tail_1m
from model import Kronos, KronosTokenizer, KronosPredictor

def load(name="NeoQuasar/Kronos-small", ctx=512):
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    return KronosPredictor(Kronos.from_pretrained(name), tok, device="cuda:0", max_context=ctx)

if __name__ == "__main__":
    t0 = time.time(); p = load(); d = tail_1m(400)
    x = d[["open", "high", "low", "close", "volume"]]
    xt = d["timestamp"]; yt = pd.Series(pd.date_range(xt.iloc[-1] + pd.Timedelta(minutes=1), periods=60, freq="1min"))
    torch.cuda.reset_peak_memory_stats()
    out = p.predict(df=x, x_timestamp=xt, y_timestamp=yt, pred_len=60, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    print(out.head(3)); print("shape", out.shape, "last_close_in", x.close.iloc[-1], "secs", round(time.time() - t0, 1), "peak_gb", torch.cuda.max_memory_allocated() / 1e9)
