"""Q4 (measured): time gaps (> 1 minute between consecutive bars) in both data files, their length
distribution, and what each gap/data-start policy costs on 7-day training blocks of the long history.

Policies (anchors masked = excluded from the loss, early stopping and evaluation, identically in every arm):
  (i)   reset the state and mask a burn-in of M bars after EVERY gap and at the data start
  (ii)  elapsed-time alpha everywhere (log decay x dt): no reset, no mask except the data start
  (iii) hybrid: elapsed-time alpha for gaps <= G_reset minutes, reset + M-bar mask for longer gaps
M = burn_in(P) at eps 1e-3 after the -0.5 logit shift: P = 60 -> 339, 240 -> 1,362, 1440 -> 8,191 bars.
Independent of the policy (NT-041 (8)): anchors whose 60-bar window or 20-bar label spans a gap are dropped
(reported for reference; the same in both arms).

Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4_gaps.py
Output:  q4_gaps.json
"""
import time

import numpy as np
import pandas as pd

from common import CSV, LONG_CSV, dump
from kernel import burn_in

BINS = [(2, 2), (3, 5), (6, 15), (16, 60), (61, 240), (241, 1440), (1441, 10 ** 9)]
MS = {f"p{p}": burn_in(p) for p in (60, 240, 1440)}
L, HMAX = 60, 20


def census(ts):
    t = ts.to_numpy().astype("datetime64[m]").astype(np.int64)
    d = np.diff(t)
    gaps = d[d > 1]
    out = {"bars": int(len(t)), "first": str(ts.iloc[0]), "last": str(ts.iloc[-1]),
           "span_minutes": int(t[-1] - t[0] + 1), "missing_minutes": int((gaps - 1).sum()),
           "n_gaps": int(len(gaps)), "n_duplicate_or_backward": int((d <= 0).sum()),
           "longest_gap_minutes": int(gaps.max()) if len(gaps) else 0,
           "gap_length_minutes_quantiles": {q: float(np.percentile(gaps, q)) for q in (50, 90, 99)} if len(gaps) else {},
           "gap_histogram_by_length_minutes": {f"{a}-{b if b < 10 ** 9 else 'inf'}": int(((gaps >= a) & (gaps <= b)).sum())
                                               for a, b in BINS}}
    return out, t, d


def policy_costs(t, d, block_minutes=7 * 1440):
    """Per non-overlapping 7-day wall-clock block: masked-anchor fraction under each policy."""
    n = len(t)
    gap_start_bar = np.nonzero(d > 1)[0] + 1              # first bar after each gap
    gap_len = d[d > 1]
    blocks = np.arange(t[0], t[-1] - block_minutes, block_minutes)
    b_lo = np.searchsorted(t, blocks)
    b_hi = np.searchsorted(t, blocks + block_minutes)
    bars_per_block = b_hi - b_lo
    res = {"n_blocks": int(len(blocks)), "bars_per_block_median": float(np.median(bars_per_block)),
           "blocks_with_any_gap": int(sum(np.any((gap_start_bar >= lo) & (gap_start_bar < hi)) for lo, hi in zip(b_lo, b_hi)))}

    def masked_fraction(starts, M):
        ev = np.zeros(n + 1, np.int64)
        np.add.at(ev, np.clip(starts, 0, n), 1)
        np.add.at(ev, np.clip(starts + M, 0, n), -1)
        mask = np.cumsum(ev)[:n] > 0
        cm = np.concatenate([[0], np.cumsum(mask)])
        frac = (cm[b_hi] - cm[b_lo]) / np.maximum(bars_per_block, 1)
        return {"mean": float(frac.mean()), "p90": float(np.percentile(frac, 90)), "max": float(frac.max()),
                "blocks_with_masked_anchors": int((frac > 0).sum())}

    # NT-041 window/label rule: drop anchors whose window (L bars back) or label (HMAX bars ahead) spans a gap
    ev = np.zeros(n + 1, np.int64)
    np.add.at(ev, np.clip(gap_start_bar - (HMAX - 1), 0, n), 1)
    np.add.at(ev, np.clip(gap_start_bar + (L - 1), 0, n), -1)
    mask = np.cumsum(ev)[:n] > 0
    cm = np.concatenate([[0], np.cumsum(mask)])
    frac = (cm[b_hi] - cm[b_lo]) / np.maximum(bars_per_block, 1)
    res["nt041_window_label_rule_dropped"] = {"mean": float(frac.mean()), "p90": float(np.percentile(frac, 90)),
                                              "max": float(frac.max())}
    for mname, M in MS.items():
        res[f"(i)_reset_every_gap_{mname}_M{M}"] = masked_fraction(gap_start_bar, M)
        for G in (5, 60, 240, 1440):
            res[f"(iii)_reset_gaps_over_{G}min_{mname}_M{M}"] = masked_fraction(gap_start_bar[gap_len > G], M)
        res[f"(ii)_elapsed_time_only_{mname}"] = {"mean": 0.0, "note": "no mask except the data start"}
    return res


def zero_volume_flat(df):
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    zv = df["volume"] == 0
    same_as_prev = df["close"].eq(df["close"].shift())
    return {"zero_volume_bars": int(zv.sum()), "zero_volume_flat_bars_equal_prev_close": int((zv & flat & same_as_prev).sum()),
            "fraction_zero_volume": float(zv.mean())}


def main():
    t0 = time.perf_counter()
    out = {"burn_in_M": MS}
    b = pd.read_csv(CSV)
    ts = pd.to_datetime(b["datetime"], utc=True)
    c, tb, db = census(ts)
    c.update(zero_volume_flat(b))
    out["bundled_30day"] = c
    # the data start on the bundled file: fold -1's training block starts at the first sequence
    out["bundled_data_start"] = {f"{k}_M{M}": {"share_of_30213_train_anchors": M / 30213, "share_of_7day_block": M / 10080}
                                 for k, M in MS.items()}
    print("bundled:", c)
    lg = pd.read_csv(LONG_CSV)
    tsl = pd.to_datetime(lg["timestamp"], format="%Y-%m-%d %H:%M:%S")
    cl, tl, dl = census(tsl)
    cl.update(zero_volume_flat(lg))
    years = tsl.dt.year.to_numpy()[1:]
    cl["gaps_per_year"] = {int(y): int(((dl > 1) & (years == y)).sum()) for y in np.unique(years)}
    cl["missing_minutes_per_year"] = {int(y): int((np.where(dl > 1, dl - 1, 0) * (years == y)).sum()) for y in np.unique(years)}
    top = np.argsort(dl)[::-1][:10]
    cl["ten_longest_gaps"] = [{"after": str(tsl.iloc[i]), "minutes": int(dl[i])} for i in top]
    out["long_2017_2025"] = cl
    print("long:", {k: v for k, v in cl.items() if k not in ("ten_longest_gaps",)})
    out["long_policy_costs_7day_blocks"] = policy_costs(tl, dl)
    for k, v in out["long_policy_costs_7day_blocks"].items():
        print(" ", k, v)
    out["seconds"] = time.perf_counter() - t0
    dump("q4_gaps.json", out)


if __name__ == "__main__":
    main()
