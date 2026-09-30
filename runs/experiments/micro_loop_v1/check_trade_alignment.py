"""Audit (owner, 2026-09-30): the presentation's trade markers do not always sit on the price line. Is the backtest,
the data alignment, or only the drawing wrong?

    CUDA_VISIBLE_DEVICES=-1 python runs/experiments/micro_loop_v1/check_trade_alignment.py   # writes check_trade_alignment.json

Checks on the leader cell (long_360d_stab f-2 s0, calibrated_quantile 0.9, zero cost):
  A. the stored bars match the raw CSV: for every anchor, the CSV row at anchor_timestamp has the stored O/H/L/C;
  B. the target is forward-looking and nothing else is: y[t, h] == close[t + H] - close[t];
  C. every trade fills at the open of its entry bar, and that bar comes one bar after the decision bar;
  D. the decision at bar t uses only data up to t: the window of anchor t ends at bar t (sequence_index / anchor_bar);
  E. what the chart draws: the marker (entry bar time, fill price) against the price line as drawn (every 15th close)
     and against the full-resolution close and open.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import neural_trade  # noqa: F401
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block

ROOT = Path(__file__).resolve().parents[3]
CELL = ROOT / "runs/scenarios/long_360d_stab/20260930T094257Z-dce15ed-e3669618-default__f-2__s0"
OUT = Path(__file__).with_name("check_trade_alignment.json")
THIN = 15


def main() -> None:
    cal, _, _ = load_block(CELL / "predictions_cal.npz")
    oos, bars, extra = load_block(CELL / "predictions_oos.npz")
    ts = pd.to_datetime(np.asarray(extra["anchor_timestamp"]).astype(str))
    O, H, Lo, C = (np.asarray(x, dtype=float) for x in (bars.open, bars.high, bars.low, bars.close))
    res = {}

    # A. raw CSV
    raw = pd.read_csv(ROOT / "Bitcoin_BTCUSDT.csv")
    tcol = next(c for c in raw.columns if c.lower() in ("timestamp", "datetime", "date", "time", "open_time"))
    raw[tcol] = pd.to_datetime(raw[tcol])
    raw = raw.set_index(tcol).sort_index()
    cols = {k: next(c for c in raw.columns if c.lower() == k) for k in ("open", "high", "low", "close")}
    sub = raw.reindex(ts)
    found = sub[cols["close"]].notna().to_numpy()
    res["A_rows_found"] = f"{int(found.sum())}/{len(ts)}"
    for k, arr in (("open", O), ("high", H), ("low", Lo), ("close", C)):
        d = np.abs(sub[cols[k]].to_numpy(float)[found] - arr[found])
        res[f"A_max_abs_diff_{k}"] = float(d.max())
    # the next CSV row after each anchor: is the stored open of anchor t+1 the CSV open of the minute after t?
    nxt = raw.index.searchsorted(ts[:-1], side="right")
    res["A_next_row_minutes_after_anchor"] = sorted(set(((raw.index[nxt] - ts[:-1]).total_seconds() / 60).astype(int).tolist()))[:5]
    res["A_open_t1_equals_csv_next_open"] = bool(np.allclose(raw[cols["open"]].to_numpy(float)[nxt], O[1:]))

    # B. the target
    hs = [int(x) for x in oos.horizon_steps]
    y = np.asarray(oos.y, dtype=float)
    for i, hb in enumerate(hs):
        n = len(C) - hb
        res[f"B_y_h{i}_equals_close_t+{hb}_minus_close_t"] = bool(np.allclose(y[:n, i], C[hb:] - C[:n]))

    # C. fills
    sig = BlockSignals.build(cal, oos)
    bt, _ = fit_and_backtest(sig, bars, strategy="calibrated_quantile", strategy_params={"entry_quantile": 0.9},
                             backtest_params={"fee_bps": 0, "half_spread_bps": 0, "slippage_bps": 0, "random_seeds": 0})
    tf = bt.trades_frame()
    eb, ep = tf["entry_bar"].to_numpy(int), tf["entry_price"].to_numpy(float)
    res["C_trades"] = int(len(tf))
    res["C_entry_price_equals_open_of_entry_bar"] = bool(np.allclose(ep, O[eb]))
    dec = np.array([d["bar"] for d in bt.decisions])
    res["C_entry_bar_minus_decision_bar"] = sorted(set((eb - dec[:len(eb)]).tolist())) if len(dec) >= len(eb) else "n/a"
    xb, xp, why = tf["exit_bar"].to_numpy(int), tf["exit_price"].to_numpy(float), tf["exit_reason"].astype(str).to_numpy()
    m = np.isin(why, ["REV", "TIME"])
    res["C_rev_time_exit_price_equals_open_of_exit_bar"] = bool(np.allclose(xp[m], O[xb[m]]))

    # D. causality of the input window
    ab = np.asarray(extra["anchor_bar"], dtype=int)
    res["D_anchor_bars_consecutive"] = bool(np.all(np.diff(ab) == 1))
    res["D_last_close_equals_bar_close"] = bool(np.allclose(np.asarray(oos.last_close, dtype=float), C))

    # E. the drawing
    gap_bps = 1e4 * np.abs(O[1:] - C[:-1]) / C[:-1]
    res["E_open_vs_previous_close_bps"] = {"median": round(float(np.median(gap_bps)), 3), "p95": round(float(np.quantile(gap_bps, 0.95)), 3),
                                           "max": round(float(gap_bps.max()), 2), "share_exactly_equal": round(float(np.mean(O[1:] == C[:-1])), 4)}
    pick = np.arange(0, len(C), THIN)
    line = np.interp(eb, pick, C[pick])                     # the drawn line at the marker's x (linear between points)
    dev_drawn = 1e4 * np.abs(ep - line) / line
    dev_close = 1e4 * np.abs(ep - C[eb]) / C[eb]            # marker vs the full-resolution close of the same bar
    dev_prev = 1e4 * np.abs(ep - C[eb - 1]) / C[eb - 1]     # vs the close the decision was made on
    q = lambda a: {"median": round(float(np.median(a)), 2), "p90": round(float(np.quantile(a, 0.9)), 2), "max": round(float(a.max()), 1)}  # noqa: E731
    res["E_marker_vs_drawn_line_bps"] = q(dev_drawn)
    res["E_marker_vs_close_same_bar_bps"] = q(dev_close)
    res["E_marker_vs_decision_close_bps"] = q(dev_prev)
    worst = np.argsort(-dev_drawn)[:5]
    res["E_worst_by_drawn_line"] = [{"entry_time": str(ts[eb[i]]), "fill_open": ep[i], "drawn_line": round(float(line[i]), 1),
                                     "close_same_bar": C[eb[i]], "decision_close": C[eb[i] - 1], "bar_low": Lo[eb[i]], "bar_high": H[eb[i]]}
                                    for i in worst]
    res["E_fill_inside_bar_range"] = bool(np.all((ep >= Lo[eb] - 1e-9) & (ep <= H[eb] + 1e-9)))
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
