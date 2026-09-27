"""B_simple of the logistic direction proxy AFTER the class prior is learned (intercept at logit(up rate),
feature weights 0): the regime in which the weak trailing-return signal is learned."""
import numpy as np, pandas as pd
df = pd.read_csv("C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv")
c = df["close"].to_numpy(float)
anchors = np.arange(60, len(c) - 19)[:30213]
lc = c[anchors - 1]
for h, tau in ((10, 8), (15, 11), (20, 15)):
    y = c[anchors + h - 1] - lc
    lags = (1, 5, 10, 15, 20, 30)
    X = np.stack([lc - c[anchors - 1 - k] for k in lags] + [lc - c[anchors - 60]], 1)
    X = (X - X.mean(0)) / X.std(0)
    move = np.abs(1e4 * y / lc) > 5.0
    up = (y > 0).astype(float)
    p = up[move].mean()
    g = ((p - up)[:, None] * X)[move]
    G = g.mean(0)
    B = np.trace(np.cov(g.T)) / float(G @ G)
    print(f"h={h}: up-rate {p:.3f}; feature-gradient B_simple = {B:,.0f} positions (x tau {tau}: {B*tau:,.0f})")
