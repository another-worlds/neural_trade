"""Review check: does A/B-1 (series vs window indicator memory) change only the memory?

At fold -1's VAL anchors (not test), newest run's served logits and meta Dense, the d = EMA - close
channel (scaled units) for several instances, float64:
  Wc: today (60-bar cold window, ema[0] = x[0], constant per-window alpha from the anchor context)
  Wp: 60-bar cold window, per-bar alphas from the rolling context (adaptation granularity changed only)
  Sc: full history, constant alpha = the anchor's per-window alpha (memory changed only; hypothetical)
  Sp: full history, per-bar alphas (the plan's arm B)
Reports RMS differences relative to RMS(Wc).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python r1_ab1_one_change.py
"""
import json
import os
import sys

sys.path.insert(0, "D:/nt_research/wfp/B")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

import common as C

NEWEST = f"{C.RUNS}/20260924T182915Z-1aeff1c-dirty-af67ee43/"
L = 60


def main():
    info = C.run_info(NEWEST)
    B = C.blocks(-1)
    close = B["close"]
    s = info["scale"]
    anchors = B["val"]
    lg = info["logits"]
    ctx_r = C.rolling_context(close, s)
    shift_bar = C.meta_shift(ctx_r, info["W"], info["b"])            # [N, 18]
    shift_anchor = C.meta_shift(C.today_context(close, anchors, s), info["W"], info["b"])  # [A, 18]
    out = {"run": info["run"], "block": "val (fold -1)", "n_anchors": int(len(anchors)), "instances": {}}
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
    for n in ("ma_period_0", "ma_period_2", "macd_0_slow", "macd_1_slow", "bb_period_2", "rsi_period_1"):
        j = C.NAMES.index(n)
        a_anchor = C.sigmoid(lg[j] + shift_anchor[:, j])              # [A]
        a_bar = C.sigmoid(lg[j] + shift_bar[:, j])                    # [N]
        w = C.windows(close, anchors, L)                              # [A, L]
        # Wc: today
        Wc = C.ewma_windowed(w, a_anchor)[:, -1] - w[:, -1]
        # Wp: cold window, per-bar alphas
        idx = anchors[:, None] - (L - 1) + np.arange(L)[None, :]
        ab = a_bar[idx]
        e = w[:, 0].copy()
        for t in range(1, L):
            e = ab[:, t] * w[:, t] + (1 - ab[:, t]) * e
        Wp = e - w[:, -1]
        # Sp: full history per-bar (ema[0] = x[0])
        full = np.empty_like(close)
        full[0] = close[0]
        for t in range(1, len(close)):
            full[t] = a_bar[t] * close[t] + (1 - a_bar[t]) * full[t - 1]
        Sp = full[anchors] - close[anchors]
        # Sc: full history, constant alpha = the anchor's alpha (renormalised truncated kernel, 1e-12)
        Sc = np.empty(len(anchors))
        for i, (t, a) in enumerate(zip(anchors, a_anchor)):
            nl = min(t + 1, int(np.ceil(np.log(1e-12) / np.log1p(-a))))
            k = a * (1 - a) ** np.arange(nl)
            k[-1] += (1 - a) ** nl  # mass of the (older) initial state on the oldest bar
            Sc[i] = (k * close[t - np.arange(nl)]).sum() - close[t]
        ref = rms(Wc)
        out["instances"][n] = {
            "base_period": float(C.period_of_logit(lg[j])),
            "rel_Wp_vs_Wc_adaptation_granularity_only": rms(Wp - Wc) / ref,
            "rel_Sc_vs_Wc_memory_only": rms(Sc - Wc) / ref,
            "rel_Sp_vs_Wc_armB_vs_armA": rms(Sp - Wc) / ref,
            "rel_Sp_vs_Sc_adaptation_granularity_given_memory": rms(Sp - Sc) / ref,
            "corr_Sp_Wc": float(np.corrcoef(Sp, Wc)[0, 1]),
        }
        print(n, {k: round(v, 4) for k, v in out["instances"][n].items()})
    json.dump(out, open("D:/nt_research/wfp/review/r1_ab1_one_change.json", "w"), indent=1)


if __name__ == "__main__":
    main()
