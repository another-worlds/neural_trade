import os, sys, io
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import neural_trade  # noqa: F401  (CUDA DLL path)
import tensorflow as tf
from neural_trade.core.config import Config
from neural_trade.models.gru_attention import build_gru_attention

RUN = "D:/neural_trade/runs/scenarios/long_360d_stab/20260930T094257Z-dce15ed-e3669618-default__f-2__s0/config.yaml"
lookback = int(sys.argv[1]) if len(sys.argv) > 1 else None
cfg = Config.from_yaml(RUN)
if lookback:
    cfg.LOOKBACK = lookback
    cfg.MOMENTUM_CLIP_MAX = lookback
m = build_gru_attention(cfg)
buf = io.StringIO()
m.summary(line_length=160, print_fn=lambda s: buf.write(s + "\n"))
print(buf.getvalue())
print("TOTAL", m.count_params())
tot = 0
for l in m.layers:
    n = l.count_params()
    if n:
        shp = l.output_shape
        print(f"{l.name:40s} {type(l).__name__:28s} {n:8d} {shp}")
        tot += n
print("sum", tot)
