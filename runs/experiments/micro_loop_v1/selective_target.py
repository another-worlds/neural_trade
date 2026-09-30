"""The owner's target read as SELECTIVE trading: can a rule that trades only its most confident setups reach
>= 60% winning trades, with >= 300 trades, net PnL > 0 after the 26 bps round trip, and max drawdown < 5%,
out of sample?

    python runs/experiments/micro_loop_v1/selective_target.py    # uses the spot klines cached by the first probe

Data: spot BTCUSDT 1m 2024-01 .. 2025-07-20 (D:/nt_data_probe, public). Features: the first probe's flow model
(price lags, log sigma, spot taker imbalance 1/5/15/60, volume and trade-count ratios) plus the post-shock z
(60-bar return / sigma). Splits: TRAIN 2024-01..09 (fit the logistic model), VALID 2024-10..12 (choose the
confidence threshold: the smallest |p - 0.5| whose VALID winning share >= 0.60 with >= 60 VALID trades),
TEST 2025-01-01..07-20 (judged once). Horizons H in {60, 240} bars (moves large enough to matter against costs).
Trading: at a signal, enter at the next bar's close proxy, hold H bars, one position at a time, full size,
cost 26 bps per round trip; a "win" is a trade with net return > 0.
Pass (pre-registered): TEST net winning share >= 0.60, >= 300 trades, net PnL > 0, max drawdown < 5%.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SPOT = Path("D:/nt_data_probe")
OUT = Path(__file__).with_name("selective_target.json")
COST = 0.0026
MONTHS = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13) if (y, m) <= (2025, 7)]


def load() -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    frames = []
    for m in MONTHS:
        with zipfile.ZipFile(SPOT / f"BTCUSDT-1m-{m}.zip") as z:
            frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=cols))
    df = pd.concat(frames, ignore_index=True)
    t = df.open_time.astype("int64")
    t = np.where(t > 1e15, t // 1000, t)
    df["ts"] = pd.to_datetime(t, unit="ms")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def simulate(sig_idx, side, c, H):
    """Non-overlapping trades: returns per-trade net returns and the equity curve."""
    out, busy_until = [], -1
    for i, s in zip(sig_idx, side):
        if i <= busy_until:
            continue
        r = s * np.log(c[i + 1 + H] / c[i + 1]) - COST
        out.append(r)
        busy_until = i + 1 + H
    r = np.array(out)
    eq = np.exp(np.cumsum(r)) if len(r) else np.array([1.0])
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(r) else 0.0
    return r, dd


def main() -> None:
    df = load()
    c, v, tb, tr = (df[k].to_numpy(float) for k in ("close", "volume", "taker_buy_base", "trades"))
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    imb, cv = np.cumsum(2 * tb - v), np.cumsum(v)
    vr = np.log((pd.Series(v).rolling(60).mean() + 1e-9) / (pd.Series(v).rolling(1440).mean() + 1e-9)).to_numpy()
    trr = np.log((pd.Series(tr).rolling(60).mean() + 1e-9) / (pd.Series(tr).rolling(1440).mean() + 1e-9)).to_numpy()
    ts = df.ts.to_numpy()
    idx = np.arange(1500, len(c) - 250, 1)
    X = np.column_stack([np.log(c[idx] / c[idx - L]) for L in (1, 5, 15, 60, 240)] + [np.log(sig[idx] + 1e-12)]
                        + [(imb[idx] - imb[idx - L]) / (cv[idx] - cv[idx - L] + 1e-9) for L in (1, 5, 15, 60)]
                        + [np.nan_to_num(vr[idx]), np.nan_to_num(trr[idx]),
                           np.log(c[idx] / c[idx - 60]) / (sig[idx] * np.sqrt(60) + 1e-12)])
    t = ts[idx]
    train = t < np.datetime64("2024-10-01")
    valid = (t >= np.datetime64("2024-10-01")) & (t < np.datetime64("2025-01-01"))
    test = (t >= np.datetime64("2025-01-01")) & (t < np.datetime64("2025-07-21"))
    res = {"cost_round_trip": COST, "rows": []}
    for H in (60, 240):
        y = (c[idx + 1 + H] > c[idx + 1]).astype(int)
        m = LogisticRegression(max_iter=3000).fit(X[train][::5], y[train][::5])
        p = m.predict_proba(X)[:, 1]
        conf = np.abs(p - 0.5)
        chosen = None
        for q in np.linspace(0.5, 0.999, 60):
            thr = float(np.quantile(conf[valid], q))
            sel = valid & (conf >= thr)
            r, _ = simulate(idx[sel], np.where(p[sel] > 0.5, 1, -1), c, H)
            if len(r) >= 60 and np.mean(r > 0) >= 0.60:
                chosen = (q, thr, len(r), float(np.mean(r > 0)))
                break
        row = {"H": H, "valid_threshold_found": chosen is not None}
        if chosen is None:
            q, thr = 0.99, float(np.quantile(conf[valid], 0.99))  # report the most selective tried, for context
            row["note"] = "no threshold reached 60% net wins with >= 60 VALID trades; TEST shown at the 99th pct"
        else:
            q, thr = chosen[0], chosen[1]
            row.update({"valid_trades": chosen[2], "valid_win": round(chosen[3], 4)})
        sel = test & (conf >= thr)
        r, dd = simulate(idx[sel], np.where(p[sel] > 0.5, 1, -1), c, H)
        row.update({"threshold_quantile": round(q, 4), "test_trades": int(len(r)),
                    "test_win_net": round(float(np.mean(r > 0)) if len(r) else float("nan"), 4),
                    "test_win_gross": round(float(np.mean(r + COST > 0)) if len(r) else float("nan"), 4),
                    "test_net_pnl_pct": round(float(100 * (np.exp(r.sum()) - 1)) if len(r) else 0.0, 2),
                    "test_max_drawdown_pct": round(100 * dd, 2)})
        row["passes"] = bool(len(r) >= 300 and row["test_win_net"] >= 0.60 and row["test_net_pnl_pct"] > 0
                             and row["test_max_drawdown_pct"] < 5)
        res["rows"].append(row)
        print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
