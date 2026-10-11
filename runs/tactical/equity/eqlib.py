"""Equity-curve measurement of the tactical strategies (descriptive; nothing is chosen here).
Pure functions: a non-overlapping trade sequence, per-trade P&L at round-trip costs, hit rate / drawdown / monthly hits,
a random-side null, and the matplotlib equity plot.

Conventions (fixed before any number was computed):
- One trade = (entry time, exit time, side, P&L of the long side, P&L of the short side) in return units (fraction of the entry price).
  Entry time is when the position opens (close of the anchor's last bar); exit time is when it is certainly closed (the end of the
  bar in which a barrier was touched, or of the label's last bar). A barrier touched inside a bar is conservatively counted as closed
  at the end of that bar, so the next position never opens earlier than it could.
- Non-overlapping sequence: signals are scanned in entry order; a signal is skipped while a position is open (entry < previous exit);
  a signal whose entry equals the previous exit is taken.
- Equal fixed notional: every trade risks the initial capital 1.0 (additive equity: equity = 1 + sum of per-trade returns), 1x
  notional, no compounding, no leverage. Max drawdown = largest (peak - equity) / peak, in percent of the running peak equity.
- Costs: a round-trip cost c bps is subtracted from every trade's P&L.
"""
import numpy as np
import pandas as pd

COSTS = (0, 1, 2, 5)
NS_H = 3600 * 10 ** 9
NS_D = 24 * NS_H


def non_overlapping(entry_ns, exit_ns):
    """Boolean mask over signals (any order) of the trades kept by the skip-while-open rule, and the sorted order used.
    Returns (keep_mask_in_input_order, n_skipped)."""
    entry_ns = np.asarray(entry_ns, np.int64); exit_ns = np.asarray(exit_ns, np.int64)
    order = np.argsort(entry_ns, kind="stable"); keep = np.zeros(len(entry_ns), bool); free_at = np.iinfo(np.int64).min
    for i in order:
        if entry_ns[i] >= free_at:
            keep[i] = True; free_at = exit_ns[i]
    return keep, int((~keep).sum())


def trade_pnl(side, entry_price, later_price):
    """Return of one trade: side (+1 long / -1 short) times the price change from the ENTRY price to a price at or after entry."""
    return side * (later_price / entry_price - 1.0)


def equity_stats(pnl_frac, t_entry_ns, t_exit_ns, start_ns, cost_bps=0.0):
    """Statistics of a trade sequence (already non-overlapping and sorted by entry)."""
    p = np.asarray(pnl_frac, float) - cost_bps * 1e-4
    n = len(p)
    if n == 0:
        return {"n_trades": 0}
    eq = 1.0 + np.cumsum(p); peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = (peak - eq) / peak; peak0 = np.maximum(peak, 1.0)
    # longest drawdown: from a peak's time to the time equity next exceeds that peak (or the end of the sequence)
    t_exit = np.asarray(t_exit_ns, np.int64); times = np.concatenate([[start_ns], t_exit]); eqf = np.concatenate([[1.0], eq])
    longest = 0; i = 0; m = len(eqf)
    run_peak = eqf[0]; peak_t = times[0]
    for j in range(1, m):
        if eqf[j] > run_peak:
            longest = max(longest, times[j] - peak_t); run_peak = eqf[j]; peak_t = times[j]
    longest = max(longest, times[-1] - peak_t)
    span_days = max((int(t_exit[-1]) - int(start_ns)) / NS_D, 1e-9)
    months = pd.Series(p > 0, index=pd.DatetimeIndex(np.asarray(t_entry_ns, np.int64).astype("datetime64[ns]"))).groupby(pd.Grouper(freq="ME")).agg(["mean", "size"])
    months = months[months["size"] > 0]
    return {"n_trades": int(n), "hit": float((p > 0).mean()), "mean_bps": float(p.mean() * 1e4),
            "trades_per_day": float(n / span_days), "cum_return_pct": float((eq[-1] - 1.0) * 100),
            "max_drawdown_pct": float(dd.max() * 100), "longest_drawdown_days": float(longest / NS_D),
            "n_months": int(len(months)), "monthly_hit_mean": float(months["mean"].mean()),
            "monthly_hit_min": float(months["mean"].min()), "monthly_hit_max": float(months["mean"].max()),
            "months_hit_above_60": int((months["mean"] > 0.6).sum()), "share_months_hit_above_60": float((months["mean"] > 0.6).mean())}


def null_paths(pl_long, pl_short, cost_bps, draws=1000, seed=0):
    """Random-side null on the SAME trades: each draw flips a fair coin per trade for the side (the sequence and timing are kept)."""
    rng = np.random.default_rng(seed); n = len(pl_long)
    side = rng.random((draws, n)) < 0.5
    p = np.where(side, pl_long[None, :], pl_short[None, :]) - cost_bps * 1e-4
    eq = 1.0 + np.cumsum(p, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((draws, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = ((peak - eq) / peak).max(1) * 100
    return eq, dd, (p > 0).mean(1)


def null_summary(pl_long, pl_short, cost_bps, draws=1000, seed=0):
    eq, dd, hit = null_paths(pl_long, pl_short, cost_bps, draws, seed)
    cum = (eq[:, -1] - 1.0) * 100
    q = lambda v, a: float(np.quantile(v, a))
    return {"cum_return_pct_median": q(cum, .5), "cum_return_pct_p95": q(cum, .95), "max_drawdown_pct_median": q(dd, .5),
            "max_drawdown_pct_p95": q(dd, .95), "hit_median": q(hit, .5), "hit_p95": q(hit, .95), "draws": draws}


def measure(name, t_entry_ns, t_exit_ns, pl_long, pl_short, side, start_ns, n_signals, costs=COSTS, draws=1000):
    """All levels for one strategy. Inputs are the NON-OVERLAPPING trades, sorted by entry."""
    pl = np.where(np.asarray(side) > 0, pl_long, pl_short)
    out = {"strategy": name, "n_signals": int(n_signals), "n_trades": int(len(pl)), "n_skipped": int(n_signals - len(pl)), "levels": {}}
    for c in costs:
        out["levels"][str(c)] = {"stats": equity_stats(pl, t_entry_ns, t_exit_ns, start_ns, c),
                                 "null": null_summary(pl_long, pl_short, c, draws)}
    return out, pl


# ---------------------------------------------------------------- plot (colours of visualization/theme.py, copied: importing the
# package pulls in TensorFlow)
PAPER, INK, MUTED, GRID = "#121211", "#f2f1ec", "#898781", "#2c2c2a"
C0, C2, NULLC = "#3987e5", "#d95926", "#9a9890"      # SERIES[0], SERIES[1], NEUTRAL


def plot_equity(path, title, t_exit_ns, pl, pl_long, pl_short, start_ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.to_datetime(np.concatenate([[start_ns], np.asarray(t_exit_ns, np.int64)]).astype("datetime64[ns]"))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}, facecolor=PAPER)
    eq0, _, _ = null_paths(pl_long, pl_short, 0.0)
    band = lambda a: np.concatenate([np.ones((eq0.shape[0], 1)), a], axis=1)
    lo, med, hi = (np.quantile(band(eq0), q, axis=0) for q in (0.05, 0.5, 0.95))
    ax[0].fill_between(t, (lo - 1) * 100, (hi - 1) * 100, color=NULLC, alpha=0.25, lw=0, label="random-side null, 5-95% (0 bps)")
    ax[0].plot(t, (med - 1) * 100, color=NULLC, lw=1, ls=":", label="null median")
    for c, col in ((0, C0), (2, C2)):
        p = np.asarray(pl, float) - c * 1e-4; eq = np.concatenate([[1.0], 1.0 + np.cumsum(p)])
        ax[0].plot(t, (eq - 1) * 100, color=col, lw=1.6, label=f"{c} bps round trip")
        dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq) * 100
        ax[1].plot(t, -dd, color=col, lw=1.2, label=f"{c} bps (max {dd.max():.1f}%)")
    ax[1].axhline(-5, color="#d03b3b", lw=0.8, ls="--"); ax[1].text(t[0], -5, " 5% drawdown limit", color="#d03b3b", va="bottom", fontsize=8)
    ax[0].axhline(0, color=MUTED, lw=0.6)
    ax[0].set_ylabel("cumulative return, % of initial capital (1x notional)"); ax[1].set_ylabel("drawdown, % of peak")
    ax[0].set_title(title, color=INK, fontsize=11)
    for a in ax:
        a.set_facecolor(PAPER); a.grid(color=GRID, lw=0.6); a.tick_params(colors=MUTED)
        a.yaxis.label.set_color(MUTED)
        for s in a.spines.values():
            s.set_color(GRID)
        leg = a.legend(facecolor=PAPER, edgecolor=GRID, fontsize=8, loc="upper left" if a is ax[0] else "lower left")
        for tx in leg.get_texts():
            tx.set_color(INK)
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor=PAPER); plt.close(fig)


def finish(tag, name, t_entry, t_exit, pl_long, pl_short, side, start_ns, outdir, extra=None):
    """Signals (any order) -> non-overlapping sequence -> measure -> JSON and PNG in outdir."""
    import json
    t_entry, t_exit = np.asarray(t_entry, np.int64), np.asarray(t_exit, np.int64)
    keep, skipped = non_overlapping(t_entry, t_exit)
    o = np.where(keep)[0]; o = o[np.argsort(t_entry[o], kind="stable")]
    res, pl = measure(name, t_entry[o], t_exit[o], np.asarray(pl_long)[o], np.asarray(pl_short)[o], np.asarray(side)[o], start_ns, len(t_entry))
    res["extra"] = extra or {}
    json.dump(res, open(f"{outdir}/{tag}.json", "w"), indent=1)
    plot_equity(f"{outdir}/{tag}.png", f"{name}: {len(o)} non-overlapping trades of {len(t_entry)} signals", t_exit[o], pl, np.asarray(pl_long)[o], np.asarray(pl_short)[o], start_ns)
    return res
