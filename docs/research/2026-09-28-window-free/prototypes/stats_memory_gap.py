"""(1) Memory of learnable EWMAs and their cascades: lag L beyond which the tail weight < eps.
(2) Purge-gap options and the share of a fold they cost (7-day training block, D-022).
(3) Gradient-noise scale B_simple = tr(Sigma) / |G|^2 of linear proxies at w = 0 (in positions), on the
    bundled 30-day file's first 30,213 anchors (the train block size of run 20260924T182915Z-1aeff1c).
"""
import numpy as np
import pandas as pd

def ema_ir(p, n):
    a = 2.0 / (p + 1.0)
    return a * (1 - a) ** np.arange(n)

def tail_lag(h, eps):
    w = np.abs(h) / np.abs(h).sum()
    tail = 1.0 - np.cumsum(w)
    return int(np.argmax(tail < eps)) + 1

N = 40000
print("# (1) lag L (bars) with tail weight < eps")
print(f"{'filter':38s} {'eps=1e-2':>9s} {'eps=1e-3':>9s} {'eps=1e-4':>9s}")
cases = {f"EMA p={p}": ema_ir(p, N) for p in (5, 20, 26, 35, 60, 240, 1440)}
for f, s, g in ((12, 26, 9), (5, 35, 5), (60, 240, 60)):
    line = ema_ir(f, N) - ema_ir(s, N)
    cases[f"MACD signal ({f},{s},{g})"] = np.convolve(line, ema_ir(g, N))[:N]
for p in (25, 60, 240):
    cases[f"Bollinger var proxy EMA*EMA p={p}"] = np.convolve(ema_ir(p, N), ema_ir(p, N))[:N]
for k, h in cases.items():
    print(f"{k:38s} " + "".join(f"{tail_lag(h, e):9d}" for e in (1e-2, 1e-3, 1e-4)))

print("\n# (2) purge gap options (bars between adjacent blocks; 3 gaps per fold)")
H = 20
train, other = 7 * 1440, 1440          # 7-day training block; val/cal/test 1 day each (ASSUMPTION, NT-041 decides)
fold = train + 3 * other
mem = {p: tail_lag(cases[f"MACD signal (60,240,60)"] if p == 240 else ema_ir(p, N), 1e-3) for p in (60, 240)}
opts = {"label overlap only: max(H)": H,
        "today: LOOKBACK + max(H) (splits.py:36)": 60 + H,
        "memory eps 1e-3, p_max 60 + max(H)": tail_lag(ema_ir(60, N), 1e-3) + H,
        "memory eps 1e-3, MACD(60,240,60) + max(H)": tail_lag(cases["MACD signal (60,240,60)"], 1e-3) + H,
        "memory eps 1e-3, p_max 1440 + max(H)": tail_lag(ema_ir(1440, N), 1e-3) + H}
for k, g in opts.items():
    print(f"{k:46s} gap={g:5d} bars   3 gaps = {3 * g:6d} bars = {100 * 3 * g / (fold + 3 * g):5.1f}% of a "
          f"7d+3x1d fold")

print("\n# (3) gradient-noise scale at w = 0 (iid positions), and x tau for contiguous chunks")
df = pd.read_csv("C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv")
c = df["close"].to_numpy(float)
anchors = np.arange(60, len(c) - 19)[:30213]
lc = c[anchors - 1]
for h, tau in ((10, 8), (15, 11), (20, 15)):
    y = c[anchors + h - 1] - lc
    Bp = y.var() / y.mean() ** 2
    # logistic P(up) on trailing-return features (the DIRECTION_SKIP inputs, gru_attention.py:22-29), w = 0
    lags = (1, 5, 10, 15, 20, 30)
    X = np.stack([lc - c[anchors - 1 - k] for k in lags] + [lc - c[anchors - 60]], 1)
    X = (X - X.mean(0)) / X.std(0)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    move = np.abs(1e4 * y / lc) > 5.0
    up = (y > 0).astype(float)
    g = ((0.5 - up)[:, None] * X)[move]
    G = g.mean(0)
    Bd = np.trace(np.cov(g.T)) / float(G @ G)
    print(f"h={h:2d}: price-bias B_simple = {Bp:12,.0f} positions   logistic-direction B_simple = {Bd:10,.0f} "
          f"positions   (chunk positions: x tau ~{tau} -> {Bd * tau:12,.0f})")
