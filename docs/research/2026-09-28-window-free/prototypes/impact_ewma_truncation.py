"""How much of each in-window EWMA is the window's first bar (the truncation / cold-start weight)?

LearnableIndicators starts every EWMA at x[0] of the window (utils/math.py ewma_sequence_matrix:
M[t,0] = (1-a)^t). At the last bar (t = L-1) the first bar keeps weight (1-a)^(L-1). A true EWMA over
the whole history would put that weight on all bars before the window instead.
"""
import numpy as np

L = 60
periods = {"MA 5": 5, "MA 10": 10, "MA 30": 30,
           "MACD slow 26": 26, "MACD slow 35": 35, "MACD slow 17": 17,
           "RSI 21": 21, "BB 25": 25,
           "clip ceiling 60 (= LOOKBACK)": 60, "period 120 (clipped today)": 120,
           "period 240 (4 h)": 240, "period 1440 (1 day)": 1440}
print(f"window L={L}: weight of the window's first bar in the EWMA at the last bar, (1-a)^(L-1), a=2/(P+1)")
for name, p in periods.items():
    a = 2.0 / (p + 1.0)
    w = (1 - a) ** (L - 1)
    # bars needed so the init weight falls below 1% (burn-in for a window-free chunk)
    burn = int(np.ceil(np.log(0.01) / np.log(1 - a)))
    print(f"  {name:32s} a={a:.4f}  init weight={w:6.3f}   bars for <1% init weight={burn}")

# the second-stage EWMAs (MACD signal, BB variance) start from a first-stage series that is itself
# cold-started at x[0]; MACD line at t=0 is exactly 0 (fast and slow both = x[0]).
print("\nnote: MACD line = EMA_fast - EMA_slow is exactly 0 at the window's first bar (both start at x[0]);"
      " BB variance starts at 0 (x[0]-mean[0] = 0).")

# redundancy of the sliding window: every bar is recomputed in L windows
n_train = 30213  # train sequences, run 20260924T182915Z-1aeff1c-dirty-af67ee43 artifacts/meta.json
print(f"\nsliding windows: {n_train:,} train windows x {L} bars = {n_train*L:,} bar-positions per epoch "
      f"for ~{n_train + L - 1:,} distinct bars (x{n_train*L/(n_train+L-1):.1f} redundancy)")

# O(T^2) weight tensor of the batched EWMA (utils/math.py:236): [B, K, T, T] float32
B, K1, K2 = 256, 18, 6
for T in (60, 120, 240, 480, 1440):
    mb = B * (K1 + K2) * T * T * 4 / 2**20
    print(f"  T={T:5d}: EWMA weight tensors [B,K,T,T] stage1+stage2 = {mb:,.0f} MiB per forward (B={B})")
