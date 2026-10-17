"""M5 VisionTS minimal loader + smoke: 5-min log RV, 7-day context would be 2016 bars; smoke uses 1152 context (4 days), period 288, pred 12.
Run with D:/nt/nt_mm_env/python.exe."""
import os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from common_data import bars_5m
from visionts import VisionTS
CKPT = "D:/nt/nt_mm_cache/visionts_ckpt/"

def load(ctx=2016, pred=12, period=288):
    m = VisionTS(arch="mae_base", finetune_type="ln", ckpt_dir=CKPT, load_ckpt=True)
    m.update_config(context_len=ctx, pred_len=pred, periodicity=period)
    return m.cuda().eval()

if __name__ == "__main__":
    t0 = time.time(); ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    m = load(ctx=ctx); r = bars_5m(ctx)
    x = torch.tensor(r.logrv.values.astype("float32"))[None, :, None].cuda()   # (B, L, nvars)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): y = m(x, fp64=False)
    print("out", tuple(y.shape), y[0, :, 0].cpu().numpy().round(3), "secs", round(time.time() - t0, 1), "peak_gb", torch.cuda.max_memory_allocated() / 1e9)
