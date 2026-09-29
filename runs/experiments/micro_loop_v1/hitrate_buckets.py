"""Micro loop (D-041): conditional hit rate by calibrated-confidence bucket for one stored cell.

    python runs/experiments/micro_loop_v1/hitrate_buckets.py <cell dir> [<cell dir> ...]

For each horizon: the calibrated P(up) distribution, and the hit rate of sign(p - 0.5) against the
k-bar-forward move on the top 100 / 10 / 2 / 0.5 % most confident bars, with a +-1.96 * sqrt(0.25 / n_eff)
band (n_eff = n / k, D-012). Descriptive; no choice is made here.
"""
from __future__ import annotations

import sys

import numpy as np

import neural_trade  # noqa: F401
from neural_trade.experiments.scorer import load_block

HORIZONS = ("h0", "h1", "h2")


def run(cell: str) -> None:
    oos, bars, _ = load_block(f"{cell}/predictions_oos.npz")
    c = bars.close
    print(f"== {cell}")
    for hi, h in enumerate(HORIZONS):
        k = int(oos.horizon_steps[hi])
        p = oos.direction_prob_calibrated[h] if oos.direction_prob_calibrated else oos.direction_prob[h]
        p = np.asarray(p, float)[:-k]
        up = (c[k:] - c[:-k]) > 0
        move_bps = 1e4 * np.abs(c[k:] - c[:-k]) / c[:-k]
        print(f"-- {h} ({k} bars)  p q01/med/q99: {np.quantile(p, .01):.3f}/{np.median(p):.3f}/"
              f"{np.quantile(p, .99):.3f}   |move| median {np.median(move_bps):.1f} bps")
        conf = np.abs(p - 0.5)
        for q in (0.0, 0.9, 0.98, 0.995):
            thr = np.quantile(conf, q)
            m = conf >= thr
            hit = float(np.mean((p[m] > 0.5) == up[m]))
            n = int(m.sum())
            neff = n / k
            se = float(np.sqrt(0.25 / max(neff, 1.0)))
            print(f"   top {100 * (1 - q):5.1f}%: n={n:6d}  hit={100 * hit:5.2f}%  +-{100 * 1.96 * se:4.1f}pp"
                  f"  (n_eff {neff:.0f})  |move| med {np.median(move_bps[m]):.1f} bps")


if __name__ == "__main__":
    for cell in sys.argv[1:]:
        run(cell)
