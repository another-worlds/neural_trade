"""Q1: burn-in lengths M(eps) and the data cost of three purge rules for unbounded indicator memory.

CPU only, numpy only. Reads nothing from the repo except constants copied here with their source:
  - adaptive shift: meta_adjust = tanh(Dense(.)) in (-1, 1)  (src/neural_trade/models/gru_attention.py:56)
                    times meta_scale = 0.5                    (src/neural_trade/models/layers/learnable_indicators.py:33, :115)
    -> the logit moves by at most +-0.5; the longest applied period uses logit - 0.5.
  - alpha = sigmoid(logit), period = 2/alpha - 1           (src/neural_trade/utils/math.py:34-72)
  - gap today = LOOKBACK + max(H) = 60 + 20 = 80 sequences (src/neural_trade/data/splits.py:36)
  - label of anchor i: close[i+h-1] - close[i-1]            (src/neural_trade/data/windowing.py:3-8, :88)

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_purge_costs.py
Writes q1_purge_costs.json next to this file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("q1_purge_costs.json")
H_MAX = 20          # longest horizon (bars), reference setup
L_NET = 60          # network context in bars (window model and A2 series mode)
META_SHIFT = 0.5    # max |logit shift| = tanh bound 1 x meta_scale 0.5
EPS_LIST = [1e-2, 1e-3, 1e-4]
PERIODS = [60, 240, 1440, 10080]
DAY = 1440


def alpha_of(p):
    return 2.0 / (p + 1.0)


def logit(a):
    return math.log(a) - math.log(1.0 - a)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def shifted_alpha(p, shift=META_SHIFT):
    """Smallest alpha (longest memory) the per-window/per-bar shift can apply to learned period p."""
    return sigmoid(logit(alpha_of(p)) - shift)


def m_single(a, eps):
    """Bars until the tail weight (1-a)^M of one EWMA falls to eps."""
    return int(math.ceil(math.log(eps) / math.log(1.0 - a)))


def m_cascade(a1, a2, eps, max_bars=2_000_000):
    """Bars until the tail weight of two EWMAs in series (impulse response a1 (1-a1)^k * a2 (1-a2)^k)
    falls to eps: MACD signal (EWMA of fast-slow) or Bollinger variance (EWMA of (x-EMA)^2) shape."""
    # tail(M) = sum_{k>=M} (h1 * h2)[k]; closed form for a1 != a2:
    #   (h1*h2)[k] = a1 a2 (b1^{k+1} - b2^{k+1}) / (b1 - b2),  b = 1 - a
    b1, b2 = 1.0 - a1, 1.0 - a2
    if abs(a1 - a2) <= 1e-9 * a1:
        # equal alphas: (h*h)[k] = a^2 (k+1) b^k; tail(M) = b^M (1 + M a)  (exact)
        lo, hi = 0, max_bars
        f = lambda m: (b1 ** m) * (1.0 + m * a1)
    else:
        # tail = sum_{k>=m} a1 a2 (b1^{k+1} - b2^{k+1})/(b1-b2) = a1 a2/(b1-b2) * (b1^{m+1}/a1 - b2^{m+1}/a2)
        f = lambda m: a1 * a2 / (b1 - b2) * (b1 ** (m + 1) / a1 - b2 ** (m + 1) / a2)
        lo, hi = 0, max_bars
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) <= eps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def check_cascade_numerically(a1, a2, eps):
    """Brute-force impulse response to verify m_cascade on a short case."""
    n = m_cascade(a1, a2, eps) + 50
    k = np.arange(n)
    h1 = a1 * (1 - a1) ** k
    h2 = a2 * (1 - a2) ** k
    conv = np.convolve(h1, h2)[:n]
    tail = 1.0 - np.cumsum(conv)      # weight on lags >= m+1
    m_num = int(np.argmax(np.concatenate([[1.0], tail]) <= eps))
    return m_num


def block_layouts():
    """Assumed fold layouts (val/cal/test lengths are not yet set: NT-041 (6))."""
    return {
        "7d_train+1d_val+1d_cal+1d_test": dict(train=7 * DAY, val=DAY, cal=DAY, test=DAY),
        "7d_train+2d_val+2d_cal+5d_test(today's val/cal/test sizes)": dict(train=7 * DAY, val=2866, cal=2866, test=7236),
    }


def gaps_for_rule(rule, M):
    """Gap in bars between two adjacent blocks.

    (a)  label overlap + embargo: labels (i-1, i+H-1] of the earlier block and (j-1, j+H-1] of the later
         block are disjoint with an embargo of max(H): gap >= 2 max(H).
    (a+) = (a) with D-005's finite-window rule kept: gap = max(2 max(H), L_net + max(H)) (80 today).
    (b)  D-005 literally: reset at every evaluation-block start, after the last training label bar,
         then burn-in so that every bar of the first evaluation window has M bars of history since the
         reset: gap = max(H) + M + L_net - 1 (the -1 matches today's 80 = 60 + 20 at M = 0... today's
         formula keeps one spare bar, so we use max(H) + L_net + M to stay comparable).
    (c1) hybrid: (a+) at train|val and val|cal, (b) only at cal|test.
    """
    a_plus = max(2 * H_MAX, L_NET + H_MAX)
    b = H_MAX + L_NET + M
    if rule == "a":
        return [2 * H_MAX] * 3
    if rule == "a+":
        return [a_plus] * 3
    if rule == "b":
        return [b] * 3
    if rule == "c1":
        return [a_plus, a_plus, b]
    raise ValueError(rule)


def main():
    out = {"constants": dict(H_MAX=H_MAX, L_NET=L_NET, META_SHIFT=META_SHIFT, EPS_LIST=EPS_LIST,
                             today_gap=L_NET + H_MAX),
           "periods": {}, "rules": {}, "checks": {}}
    for p in PERIODS:
        a0 = alpha_of(p)
        a_s = shifted_alpha(p)
        row = {"alpha": a0, "alpha_after_max_shift": a_s, "period_after_max_shift": 2.0 / a_s - 1.0}
        for eps in EPS_LIST:
            row[f"M_single_noshift_eps{eps:g}"] = m_single(a0, eps)
            row[f"M_single_shift_eps{eps:g}"] = m_single(a_s, eps)
            # cascade: both stages at the (shifted) period -- the worst case for MACD signal / BB variance
            row[f"M_cascade_equal_shift_eps{eps:g}"] = m_cascade(a_s, a_s, eps)
            # cascade with a short second stage (MACD signal default 9, shifted)
            row[f"M_cascade_p_then_9_shift_eps{eps:g}"] = m_cascade(a_s, shifted_alpha(9), eps)
        out["periods"][str(p)] = row

    # numerical check of the cascade closed form on short cases
    for (p1, p2) in [(60, 60), (60, 9), (26, 9)]:
        a1, a2 = shifted_alpha(p1), shifted_alpha(p2)
        out["checks"][f"cascade_{p1}_{p2}_eps1e-3"] = dict(closed_form=m_cascade(a1, a2, 1e-3),
                                                            brute_force=check_cascade_numerically(a1, a2, 1e-3))

    # rule costs per fold, eps = 1e-3, single-EWMA M after the maximal shift and the cascade worst case
    for layout_name, L in block_layouts().items():
        fold_bars = L["train"] + L["val"] + L["cal"] + L["test"]
        res = {}
        for p in PERIODS:
            for kind in ("single", "cascade_equal"):
                key = "M_single_shift_eps0.001" if kind == "single" else "M_cascade_equal_shift_eps0.001"
                M = out["periods"][str(p)][key]
                entry = {"M": M}
                for rule in ("a", "a+", "b", "c1"):
                    g = gaps_for_rule(rule, M)
                    extra_vs_today = sum(g) - 3 * (L_NET + H_MAX)
                    entry[rule] = dict(gaps=g, gap_bars_total=sum(g),
                                       extra_vs_today_bars=extra_vs_today,
                                       gap_pct_of_7d_train=100.0 * sum(g) / L["train"],
                                       gap_pct_of_fold_span=100.0 * sum(g) / (fold_bars + sum(g)),
                                       gap_days=sum(g) / DAY)
                # data-start / data-gap burn-in (all rules): anchors whose state has seen < M bars are masked
                entry["data_start_mask_bars"] = M + L_NET
                entry["data_start_mask_pct_of_7d_train"] = 100.0 * (M + L_NET) / L["train"]
                # per-step series pass length under (a): [min anchor - M, max anchor]; today's shuffle(2048)
                # batches span a median of 11,130 anchors (window research, batch_locality.json)
                entry["series_pass_bars_per_step_a"] = 11130 + M
                res[f"p{p}_{kind}"] = entry
        out["rules"][layout_name] = dict(blocks=L, fold_labelled_bars=fold_bars, per_period=res)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # printed summary
    print("M(eps) after the maximal +-0.5 logit shift (single EWMA | cascade of two equal EWMAs):")
    for p in PERIODS:
        r = out["periods"][str(p)]
        print(f"  p={p:>6}: shifted period {r['period_after_max_shift']:9.1f}; "
              + "  ".join(f"eps={e:g}: {r[f'M_single_shift_eps{e:g}']:>7} | {r[f'M_cascade_equal_shift_eps{e:g}']:>7}"
                          for e in EPS_LIST)
              + f"   (no shift, eps=1e-3: {r['M_single_noshift_eps0.001']})")
    print("checks:", out["checks"])
    for layout_name, R in out["rules"].items():
        print(f"\nLayout {layout_name}: labelled bars per fold {R['fold_labelled_bars']}")
        for key, e in R["per_period"].items():
            print(f"  {key:>20} M={e['M']:>6}: "
                  + "; ".join(f"{rule}: gaps {e[rule]['gap_bars_total']:>6} bars "
                              f"({e[rule]['gap_pct_of_7d_train']:6.1f}% of 7d train, "
                              f"{e[rule]['gap_pct_of_fold_span']:5.1f}% of fold span)"
                              for rule in ("a", "a+", "b", "c1"))
                  + f"; data-start mask {e['data_start_mask_bars']} bars ({e['data_start_mask_pct_of_7d_train']:.1f}%)")


if __name__ == "__main__":
    main()
