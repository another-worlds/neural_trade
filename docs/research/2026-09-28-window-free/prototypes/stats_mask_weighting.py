"""Direction loss = sum(bce * mask) / sum(mask) per batch (losses/functions.py:627-629). Each scored position
then weighs 1 / (scored count of its batch). How uneven is that weight across positions, per batch layout?"""
import numpy as np, pandas as pd
exec(open("stats_autocorr.py").read().split("# ---------------------------------------------------------------- batch layouts")[1].split("layouts = {}")[0])
df = pd.read_csv("C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv")
c = df["close"].to_numpy(float)
anchors = np.arange(60, len(c) - 19)[:30213]
lc = c[anchors - 1]
rng = np.random.default_rng(1)
n = len(anchors)
for h in (10, 15, 20):
    y = c[anchors + h - 1] - lc
    move = (np.abs(1e4 * y / lc) > 5.0).astype(float)
    lay = {"tf.shuffle(2048) 256 (today)": batches_shuffle(n, 256, 2048, 5, rng),
           "chunks 16 x 150": batches_chunks(n, 16, 150, 30, rng),
           "chunks 16 x 256": batches_chunks(n, 16, 256, 30, rng),
           "chunks 4 x 1024": batches_chunks(n, 4, 1024, 30, rng)}
    print(f"h={h}: scored share {move.mean():.3f}")
    for k, bl in lay.items():
        frac = np.array([move[b].mean() for b in bl])
        w = np.concatenate([np.full(int(move[b].sum()), 1.0 / move[b].mean()) for b in bl])
        print(f"   {k:30s} per-batch scored share: mean {frac.mean():.3f} sd {frac.std():.3f}  "
              f"-> CV of position weights {w.std() / w.mean():.3f}")
