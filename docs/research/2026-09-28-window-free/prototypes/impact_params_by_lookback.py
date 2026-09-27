"""Which layers' parameter shapes depend on LOOKBACK? Build gru_attention at two lookbacks (CPU)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import neural_trade  # noqa: F401  (CUDA DLL path before TF)
import numpy as np
import tensorflow as tf
from neural_trade.core.config import Config
from neural_trade.registries.models import Models


def per_layer(lookback):
    m = Models.build(None, Config(LOOKBACK=lookback))
    return m, {l.name: int(sum(np.prod(w.shape) for w in l.weights)) for l in m.layers}


m60, a = per_layer(60)
m120, b = per_layer(120)
tot60, tot120 = m60.count_params(), m120.count_params()
print(f"total params LOOKBACK=60: {tot60:,}   LOOKBACK=120: {tot120:,}")
print("layers whose parameter count changes with LOOKBACK:")
# names differ by suffix between builds; match by order
for (n1, c1), (n2, c2) in zip(a.items(), b.items()):
    if c1 != c2:
        l = m60.get_layer(n1)
        shapes = [tuple(w.shape) for w in l.weights]
        print(f"  {n1:40s} {type(l).__name__:22s} {c1:>8,} -> {c2:>8,}  shapes@60={shapes}")
