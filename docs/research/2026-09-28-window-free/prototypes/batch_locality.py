"""How local are today's training batches? tf.data shuffle(buffer_size=2048) over 30,213 sequential
windows, batch 256 (data/datasets.py:21-22). Per batch: span of anchor indices and a crude effective
sample count = number of distinct 20-bar blocks (the longest horizon) among the 256 anchors.
Compared with a seq2seq batch of Bc random chunks of L consecutive bars (distinct 20-bar blocks)."""
import os
import json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import neural_trade  # noqa: F401
import tensorflow as tf

N, B, H = 30213, 256, 20
ds = tf.data.Dataset.from_tensor_slices(np.arange(N)).shuffle(2048, seed=42, reshuffle_each_iteration=True).batch(B)
spans, neff = [], []
for b in ds:
    b = b.numpy()
    spans.append(int(b.max() - b.min()))
    neff.append(len(np.unique(b // H)))
res = {"today_batches": len(spans), "span_median": float(np.median(spans)), "span_p10_p90": [float(np.percentile(spans, 10)), float(np.percentile(spans, 90))],
       "distinct_20bar_blocks_median": float(np.median(neff)), "distinct_p10_p90": [float(np.percentile(neff, 10)), float(np.percentile(neff, 90))]}
rng = np.random.default_rng(0)
for Bc, L in ((16, 256), (8, 512), (4, 1024)):
    ne = []
    for _ in range(200):
        st = rng.integers(0, N - L, size=Bc)
        idx = np.concatenate([np.arange(s, s + L) for s in st])
        ne.append(len(np.unique(idx // H)))
    res[f"seq2seq Bc{Bc} L{L} distinct_20bar_blocks_median"] = float(np.median(ne))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_locality.json"), "w"), indent=1)
