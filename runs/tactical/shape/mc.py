"""Brownian-with-a-clock simulator (SPEC section 5). Null A: a pure volatility clock (empirical minute-of-day profile x GARCH-like
persistence, symmetric jumps, bid-ask bounce on O and C); B adds a lag-2 autoregression phi in {0, +0.12, -0.12} on a Markov regime
as a positive control. usage: python mc.py profile | python mc.py build A|B"""
import os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shapelib as S

PHI = 0.12; N_YEARS_BARS = int(3.5 * 525600); START = "2020-01-01"


def minute_profile():
    """S[m] for m in 0..1439: mean |r| by minute of day over the real bars 2019-01..2024-06 (normalised to unit RMS), and the base sd."""
    df = S.load_bars().loc["2019-01-01":"2024-06-30 23:59"]
    r = np.diff(np.log(df["close"].to_numpy())); m = (df.index.hour * 60 + df.index.minute).to_numpy()[1:]
    prof = np.array([np.abs(r[m == k]).mean() for k in range(1440)]); prof /= np.sqrt((prof ** 2).mean())
    base = np.sqrt(np.mean(r ** 2))
    return prof, float(base)


def simulate(seed, planted, prof, base, n=N_YEARS_BARS):
    rng = np.random.default_rng(seed); tod = np.arange(n) % 1440
    alpha, beta = 0.05, 0.94; omega = (1 - alpha - beta) * base ** 2
    eps = rng.standard_normal(n); jump = rng.random(n) < 1 / 2000.0; jz = rng.standard_normal(n) * 6.0
    state = np.zeros(n, np.int8)
    if planted:                                           # Markov regime, mean duration one day
        sw = rng.random(n) < 1 / 1440.0; pick = rng.integers(0, 3, n); cur = 0
        for j in range(n):
            if sw[j]:
                cur = pick[j]
            state[j] = cur
    phis = np.array([0.0, PHI, -PHI])[state]
    r = np.zeros(n); sig = np.zeros(n); s2 = base ** 2; z_prev = 0.0; Sp = prof
    for j in range(n):                                    # GARCH on the deseasonalised return z, then the seasonal profile
        s2 = omega + alpha * z_prev * z_prev + beta * s2; sg = np.sqrt(s2); sig[j] = sg
        z = sg * eps[j]
        rj = Sp[tod[j]] * z + (Sp[tod[j]] * sg * 6.0 * jz[j] if jump[j] else 0.0)
        if planted and j >= 2:
            rj += phis[j] * r[j - 2]
        r[j] = rj; z_prev = z
    lc_end = np.log(10000.0) + np.cumsum(r); lc_start = np.concatenate([[np.log(10000.0)], lc_end[:-1]])
    # intra-minute path: a bridge of 4 sub-steps
    sd_in = np.maximum(Sp[tod] * sig, 1e-9) * 0.5
    inc = rng.standard_normal((n, 4)) * sd_in[:, None]; inc -= inc.mean(1, keepdims=True); inc += (r / 4.0)[:, None]
    path = lc_start[:, None] + np.concatenate([np.zeros((n, 1)), np.cumsum(inc, 1)], 1)
    ref = float(np.median(Sp[tod] * sig)); b = 0.3 * ref
    o = path[:, 0] + b * rng.choice([-1.0, 1.0], n); c = path[:, 4] + b * rng.choice([-1.0, 1.0], n)
    h = np.maximum(path.max(1), np.maximum(o, c)); l = np.minimum(path.min(1), np.minimum(o, c))
    vol = np.exp(1.0 + 0.8 * np.log(np.maximum(Sp[tod] * sig, 1e-9) / base) + 0.7 * rng.standard_normal(n))
    idx = pd.date_range(START, periods=n, freq="1min")
    return pd.DataFrame({"open": np.exp(o), "high": np.exp(h), "low": np.exp(l), "close": np.exp(c), "volume": vol}, index=idx)


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "profile":
        p, b = minute_profile(); np.savez(f"{S.CACHE}/mc_profile.npz", prof=p, base=b); print("profile saved", b, f"{time.time() - t0:.0f}s")
    else:
        z = np.load(f"{S.CACHE}/mc_profile.npz"); planted = sys.argv[2] == "B"
        df = simulate(11 if not planted else 12, planted, z["prof"], float(z["base"]))
        print("simulated", len(df), df.index.min(), df.index.max(), f"{time.time() - t0:.0f}s", flush=True)
        ds = S.build_dataset(df, workers=6); S.save_dataset(ds, "mc" + sys.argv[2]); print("saved", len(ds["t"]), f"{time.time() - t0:.0f}s")
