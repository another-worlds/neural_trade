"""M2 TTM r2 minimal loader + smoke (zero-shot, context 512, branch for 12-step... see PRED_LEN). Run with D:/nt/nt_mm_env/python.exe"""
import os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from common_data import bars_5m
from tsfm_public.toolkit.get_model import get_model

def load(ctx=512, pred=96):
    # branch chosen by (context, prediction length); returns a TinyTimeMixerForPrediction
    return get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=ctx, prediction_length=pred).cuda().eval()

if __name__ == "__main__":
    t0 = time.time(); m = load(); r = bars_5m(512)
    x = r[["logrv"]].values.astype("float32"); mu, sd = x.mean(0), x.std(0) + 1e-6
    inp = torch.tensor((x - mu) / sd)[None].cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): y = m(past_values=inp).prediction_outputs
    y = y.cpu().numpy()[0, :12, 0] * sd[0] + mu[0]
    print("pred first 12 (log RV)", np.round(y, 3), "secs", round(time.time() - t0, 1), "peak_gb", torch.cuda.max_memory_allocated() / 1e9, "n_params", sum(p.numel() for p in m.parameters()))
