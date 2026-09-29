"""Q3 (batch composition): what restricting each step's series pass to a contiguous span of anchors does to
the effective samples per 256-batch, with the first round's method (stats_autocorr.py): proxy per-position
gradient series on the bundled file (same anchors and targets as data/windowing.py), design effect
deff = Var(batch mean under the layout) / (Var(x) / n), n_eff = n / deff.

Layouts (every anchor once per epoch in all but 'random'):
  random 256          i.i.d. anchors (the ideal)
  today               tf.data shuffle(buffer 2048) over chronological anchors, batch 256 (the real pipeline)
  span S              tile the block into contiguous spans of S anchors at a random phase, shuffle anchors
                      inside each span, cut 256-batches, shuffle the batch order (a batch = one span)
Also today's per-batch anchor span (max - min) from the real tf.data pipeline: the pass length a
span-restricted step would need WITHOUT changing the batch composition.

Blocks: the first 10,080 anchors (a 7-day block) and the first 30,213 anchors (run 20260924T182915Z).
Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_batches.py
Output:  q3_batches.json
"""
from common import CSV, dump, np, tf
import pandas as pd

rng = np.random.default_rng(0)
close = pd.read_csv(CSV)["close"].to_numpy(float)
L, H, DB_BPS = 60, (10, 15, 20), 5.0
start, end = L, len(close) - (max(H) - 1)
anchors = np.arange(start, end)
lc = close[anchors - 1]
d1 = np.diff(close)
csum = np.concatenate([[0], np.cumsum(d1)])
csum2 = np.concatenate([[0], np.cumsum(d1 ** 2)])
a_, b_ = anchors - L, anchors - 1
m_ = (csum[b_] - csum[a_]) / (b_ - a_)
tv = (csum2[b_] - csum2[a_]) / (b_ - a_) - m_ * m_
series = {}
for h in H:
    y = close[anchors + h - 1] - lc
    move = np.abs(1e4 * y / lc) > DB_BPS
    up = (y > 0).astype(float)
    series[f"price_h{h}"] = y
    series[f"dir_h{h}"] = np.where(move, up - up[move].mean(), 0.0)
    series[f"var_h{h}"] = 1.0 - y ** 2 / (h * np.maximum(tv, 1e-12))


def tf_shuffle_order(n, buffer, rng):
    buf = list(range(min(buffer, n)))
    nxt = len(buf)
    out = []
    while buf:
        j = rng.integers(len(buf))
        out.append(buf[j])
        if nxt < n:
            buf[j] = nxt
            nxt += 1
        else:
            buf[j] = buf[-1]
            buf.pop()
    return np.array(out)


def batches_today(n, epochs):
    out = []
    for _ in range(epochs):
        o = tf_shuffle_order(n, 2048, rng)
        m = len(o) // 256
        out += list(o[: m * 256].reshape(m, 256))
    return out


def batches_random(n, nb):
    return [rng.choice(n, 256, replace=False) for _ in range(nb)]


def batches_span(n, S, epochs):
    out = []
    for _ in range(epochs):
        phase = int(rng.integers(S))
        starts = np.arange(phase - S, n, S)
        for s0 in starts:
            idx = np.arange(max(0, s0), min(n, s0 + S))
            if len(idx) < 256:
                continue
            rng.shuffle(idx)
            m = len(idx) // 256
            out += list(idx[: m * 256].reshape(m, 256))
    return out


def deff(x, batches):
    means = np.array([x[b].mean() for b in batches])
    return means.var(ddof=1) / (x.var(ddof=1) / 256)


res = {}
for n in (10080, 30213):
    lay = {"random 256": batches_random(n, 3000), "today tf.shuffle(2048)": batches_today(n, 25 if n > 20000 else 60)}
    for S in (256, 512, 1024, 2048, 4096):
        lay[f"span {S}"] = batches_span(n, S, max(10, int(3000 * 256 / n)))
    block = {}
    for name, b in lay.items():
        row = {"n_batches": len(b)}
        for k, x in series.items():
            d = deff(x[:n], b)
            row[k] = {"deff": float(d), "n_eff": float(256 / d)}
        row["n_eff_range_all_proxies"] = [float(min(v["n_eff"] for k, v in row.items() if isinstance(v, dict))),
                                          float(max(v["n_eff"] for k, v in row.items() if isinstance(v, dict)))]
        block[name] = row
        print(n, name, "n_eff range", [round(v) for v in row["n_eff_range_all_proxies"]],
              "h15:", {k: round(row[k]['n_eff']) for k in ("price_h15", "dir_h15", "var_h15")})
    # today's per-batch anchor span in the REAL tf.data pipeline (datasets.py: shuffle(2048, seed), batch 256)
    ds = tf.data.Dataset.from_tensor_slices(np.arange(n)).shuffle(2048, seed=42, reshuffle_each_iteration=True).batch(256)
    spans = []
    for _ in range(3):
        spans += [int(x.numpy().max() - x.numpy().min()) for x in ds]
    block["today_real_pipeline_batch_span"] = {"median": float(np.median(spans)), "p10": float(np.percentile(spans, 10)),
                                               "p90": float(np.percentile(spans, 90)), "max": int(max(spans))}
    print(n, "today's batch span", block["today_real_pipeline_batch_span"])
    res[f"block_{n}"] = block
dump("q3_batches.json", res)
