"""M4 regime card: render one 96x96 candlestick+volume image with numpy/PIL (no matplotlib), embed with frozen CLIP ViT-B/32 (open_clip, OpenAI weights).
Run with D:/nt/nt_mm_env/python.exe. Window = last 120 one-minute bars of the cut file (aggregated to 24 five-minute candles)."""
import os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from PIL import Image
from common_data import tail_1m

def render(o, h, l, c, v, size=96):
    """Causal per-window scaled candle chart: top 72 rows price, bottom 24 rows volume; green up, red down. Returns uint8 (size,size,3)."""
    n = len(o); img = np.zeros((size, size, 3), np.uint8); lo, hi = l.min(), h.max(); vmax = v.max() + 1e-12
    py = lambda p: int(round((72 - 1) * (hi - p) / (hi - lo + 1e-12)))
    w = size // n
    for i in range(n):
        x = i * w + w // 2; col = (60, 200, 90) if c[i] >= o[i] else (220, 60, 60)
        img[py(h[i]):py(l[i]) + 1, x] = col
        a, b = sorted((py(o[i]), py(c[i]))); img[a:b + 1, max(0, x - 1):x + 2] = col
        vh = int(round(23 * v[i] / vmax)); img[size - 1 - vh:size, max(0, x - 1):x + 2] = (120, 120, 120)
    return img

def load(name="ViT-B-32-quickgelu", pretrained="openai"):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained, cache_dir="D:/nt/nt_mm_cache/hf")
    return m.cuda().eval(), pre

if __name__ == "__main__":
    t0 = time.time(); d = tail_1m(120).set_index("timestamp")
    r = d.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    img = render(*[r[k].values for k in ("open", "high", "low", "close", "volume")])
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(os.path.dirname(__file__), "sample_card.png"))
    m, pre = load(); x = pre(Image.fromarray(img)).unsqueeze(0).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): e = m.encode_image(x)
    print("embedding", tuple(e.shape), "norm", float(e.norm()), "secs", round(time.time() - t0, 1), "peak_gb", torch.cuda.max_memory_allocated() / 1e9)
