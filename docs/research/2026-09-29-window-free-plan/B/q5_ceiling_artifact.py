"""Q5: is the drift of the slow MACD periods to the 60-bar ceiling an exploitation of the cold-start
artefact? CPU, float64, read-only (served logits and meta Dense of each run; the bundled 30-day file).

In today's layer every EWMA restarts at the window's first bar (utils/math.py:187-192, ema[0] = x[0]).
As the slow period grows, the windowed slow EMA tends to x[0] (weight (1-a)^59 on it), so the windowed
MACD line tends to "fast EMA minus the close 59 bars ago": a trailing-return feature. We measure, at the
learned (capped) periods and at the textbook start:
  * corr(windowed MACD line at the anchor, R59) against corr(warm = full-history MACD line, R59),
    R59 = (close_t - close_{t-59}) / scale (the raw window channel already holds it: -X[0]);
  * the same over the 60 window positions against the path relative to the window's first bar;
  * the direction of d line / d slow-logit (windowed vs warm) against R59;
  * a sweep of slow periods (mechanism curve);
  * predictive correlation with the targets y_h (h = 10, 15, 20) with n_eff = N // h bands (D-012).
Noise: moving-block bootstrap (block 240 bars, 300 reps) for correlations and windowed-minus-warm gaps.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q5_ceiling_artifact.py
Writes q5_ceiling_artifact.json.
"""
import os

import numpy as np

import common as C

L = 60
LAGS = 2400          # history required before an anchor (warm-up of the longest swept period 300: weight 1.1e-7)
BLOCK, REPS = 240, 300
RNG = np.random.default_rng(11)


def win_ema(close, anchors, alpha):
    """Windowed cold-start EMA at the anchor bar; alpha scalar or [N] per window."""
    w = C.windows(close, anchors, L)
    a = np.broadcast_to(np.asarray(alpha, np.float64), (len(anchors),))[:, None]
    k = np.arange(L)[None, :]
    wt = a * (1 - a) ** (L - 1 - k)                    # weight of position k at the last bar
    wt[:, 0] = (1 - a[:, 0]) ** (L - 1)               # ema[0] = x[0]
    return (wt * w).sum(1)


def win_ema_path(close, anchors, alpha):
    """Windowed cold-start EMA at every window position [N, L] (scalar alpha)."""
    return C.ewma_windowed(C.windows(close, anchors, L), alpha)


def warm_ema(close, anchors, alpha):
    """Full-history EMA at the anchor. Scalar alpha: exact recursion over the whole series (lfilter).
    Per-window alpha [N]: dot product with the renormalised kernel over the lags whose weight > 1e-10."""
    if np.ndim(alpha) == 0:
        return C.ewma_full(close, float(alpha))[anchors]
    a = np.asarray(alpha, np.float64)
    nl = int(np.ceil(np.log(1e-10) / np.log1p(-a.min())))
    lags = np.arange(nl)
    out = np.empty(len(anchors))
    for s in range(0, len(anchors), 1024):
        aa = a[s:s + 1024, None]
        wt = aa * np.exp(lags[None, :] * np.log1p(-aa))
        wt /= wt.sum(1, keepdims=True)
        idx = anchors[s:s + 1024, None] - lags[None, :]
        out[s:s + 1024] = (wt * close[idx]).sum(1)
    return out


def warm_path(close, anchors, alpha):
    full = C.ewma_full(close, alpha)                   # scalar alpha only
    idx = anchors[:, None] - (L - 1) + np.arange(L)[None, :]
    return full[idx]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def block_boot(fn, n):
    """Moving-block bootstrap over anchor positions (contiguous anchors are neighbouring bars)."""
    nb = int(np.ceil(n / BLOCK))
    vals = []
    for _ in range(REPS):
        starts = RNG.integers(0, n - BLOCK, nb)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        vals.append(fn(idx))
    v = np.asarray(vals)
    return [float(np.percentile(v, 2.5, axis=0)), float(np.percentile(v, 97.5, axis=0))] if v.ndim == 1 else \
        np.percentile(v, [2.5, 97.5], axis=0).T.tolist()


def macd_features(close, anchors, scale, a_fast, a_slow):
    wf, ws = win_ema(close, anchors, a_fast), win_ema(close, anchors, a_slow)
    mf, ms = warm_ema(close, anchors, a_fast), warm_ema(close, anchors, a_slow)
    return (wf - ws) / scale, (mf - ms) / scale


def analyse_run(close, B, info, block="train", label=None):
    anchors = B[block]
    anchors = anchors[anchors >= LAGS + L]                                   # warm history available
    anchors = anchors[anchors + 20 < len(close)]
    s = info["scale"]
    r59 = (close[anchors] - close[anchors - (L - 1)]) / s
    lg = dict(zip(C.NAMES, info["logits"]))
    out = {"block": block, "n_anchors": int(len(anchors)), "scale": s}
    ctx = C.today_context(close, anchors, s)
    delta = C.meta_shift(ctx, info["W"], info["b"])
    for i in range(3):
        fast, slow = f"macd_{i}_fast", f"macd_{i}_slow"
        jf, js = C.NAMES.index(fast), C.NAMES.index(slow)
        res = {"learned_fast": float(C.period_of_logit(lg[fast])), "learned_slow": float(C.period_of_logit(lg[slow]))}
        variants = {
            "learned_base": (C.sigmoid(lg[fast]), C.sigmoid(lg[slow])),
            "learned_applied": (C.sigmoid(lg[fast] + delta[:, jf]), C.sigmoid(lg[slow] + delta[:, js])),
            "textbook": (2 / (C.TEXTBOOK[fast] + 1), 2 / (C.TEXTBOOK[slow] + 1)),
            "learned_fast_textbook_slow": (C.sigmoid(lg[fast]), 2 / (C.TEXTBOOK[slow] + 1)),
        }
        for name, (af, as_) in variants.items():
            w_line, m_line = macd_features(close, anchors, s, af, as_)
            cw, cm = corr(w_line, r59), corr(m_line, r59)
            ci = block_boot(lambda idx: np.array([corr(w_line[idx], r59[idx]), corr(m_line[idx], r59[idx]),
                                                  corr(w_line[idx], r59[idx]) - corr(m_line[idx], r59[idx])]), len(anchors))
            res[name] = {"corr_windowed_R59": cw, "corr_warm_R59": cm, "gap": cw - cm,
                         "ci95_windowed": ci[0], "ci95_warm": ci[1], "ci95_gap": ci[2],
                         "corr_windowed_warm": corr(w_line, m_line)}
        # derivative direction of the line wrt the slow logit (finite difference in logit, base alphas)
        h = 1e-4
        af = C.sigmoid(lg[fast])
        wp, mp = macd_features(close, anchors, s, af, C.sigmoid(lg[slow] - h))   # longer period
        wm, mm = macd_features(close, anchors, s, af, C.sigmoid(lg[slow] + h))
        dw, dm = (wp - wm) / (2 * h), (mp - mm) / (2 * h)                          # d line / d(-logit): longer
        res["d_line_d_longer"] = {"corr_windowed_R59": corr(dw, r59), "corr_warm_R59": corr(dm, r59),
                                  "rms_windowed": float(np.sqrt(np.mean(dw ** 2))), "rms_warm": float(np.sqrt(np.mean(dm ** 2)))}
        # predictive correlation with the targets (train block: where the drift was learned)
        pred = {}
        wl, ml = macd_features(close, anchors, s, C.sigmoid(lg[fast]), C.sigmoid(lg[slow]))
        wt, mt = macd_features(close, anchors, s, 2 / (C.TEXTBOOK[fast] + 1), 2 / (C.TEXTBOOK[slow] + 1))
        for hz in (10, 15, 20):
            y = (close[anchors + hz] - close[anchors]) / s          # windowing.py:88 target, anchor bar t = i - 1
            band = 1.96 / np.sqrt(len(anchors) // hz)
            pred[f"h{hz}"] = {"n_eff": int(len(anchors) // hz), "band95": float(band),
                              "R59": corr(r59, y), "windowed_learned": corr(wl, y), "warm_learned": corr(ml, y),
                              "windowed_textbook": corr(wt, y), "warm_textbook": corr(mt, y)}
        res["predictive_corr_with_target"] = pred
        out[f"macd_{i}"] = res
    # window-position view for the capped MACD-1 (base alphas): line at every position vs the path from x[0]
    i = 1
    af, as_ = C.sigmoid(lg["macd_1_fast"]), C.sigmoid(lg["macd_1_slow"])
    sub = anchors[::7]
    wpath = (win_ema_path(close, sub, af) - win_ema_path(close, sub, as_)) / s
    mpath = (warm_path(close, sub, af) - warm_path(close, sub, as_)) / s
    wins = C.windows(close, sub, L)
    rel0 = (wins - wins[:, :1]) / s
    out["macd_1_path_view"] = {"n_windows": int(len(sub)),
                               "corr_windowed_vs_path_from_x0_all_positions": corr(wpath.ravel(), rel0.ravel()),
                               "corr_warm_vs_path_from_x0_all_positions": corr(mpath.ravel(), rel0.ravel()),
                               "corr_windowed_vs_path_from_x0_last_10_positions": corr(wpath[:, -10:].ravel(), rel0[:, -10:].ravel()),
                               "corr_warm_vs_path_from_x0_last_10_positions": corr(mpath[:, -10:].ravel(), rel0[:, -10:].ravel())}
    # mechanism sweep over the slow period (MACD-1 fast at its learned value)
    sweep = {}
    for ps in (17, 26, 35, 45, 60, 80, 98, 150, 300):
        w_line, m_line = macd_features(close, anchors, s, af, 2 / (ps + 1))
        sweep[str(ps)] = {"corr_windowed_R59": corr(w_line, r59), "corr_warm_R59": corr(m_line, r59),
                          "weight_on_x0_at_last_bar": float((1 - 2 / (ps + 1)) ** (L - 1))}
    out["macd_1_slow_sweep"] = sweep
    return out


def main():
    res = {"method": __doc__.split("\n")[0], "block_bootstrap": {"block": BLOCK, "reps": REPS}, "local": {}, "ablation_capped": {}}
    B = C.blocks(-1)
    close = B["close"]
    for r in C.run_dirs():
        info = C.run_info(r)
        res["local"][info["run"]] = {blk: analyse_run(close, B, info, blk) for blk in ("train", "test")}
        m1 = res["local"][info["run"]]["train"]["macd_1"]
        print(info["run"][:24], "slow %.1f" % m1["learned_slow"],
              "applied: win %.3f warm %.3f gap %.3f %s" % (m1["learned_applied"]["corr_windowed_R59"], m1["learned_applied"]["corr_warm_R59"],
                                                          m1["learned_applied"]["gap"], np.round(m1["learned_applied"]["ci95_gap"], 3)),
              "textbook: win %.3f warm %.3f" % (m1["textbook"]["corr_windowed_R59"], m1["textbook"]["corr_warm_R59"]))
    # ablation runs whose slow period reached the cap (fold -2 = P1, fold -1 = P2)
    import pandas as pd
    blocks_by_fold = {-1: B, -2: C.blocks(-2)}
    for r in C.ablation_dirs():
        h = pd.read_csv(r + "indicator_params_history.csv")
        if not (h[["macd_0_slow", "macd_1_slow"]].to_numpy() >= 59.95).any():
            continue
        fold = -2 if r.rstrip("/\\").endswith("P1") else -1
        info = C.run_info(r)
        a = analyse_run(blocks_by_fold[fold]["close"], blocks_by_fold[fold], info, "train")
        res["ablation_capped"][info["run"]] = {"fold": fold, **{k: a[k] for k in ("n_anchors", "macd_0", "macd_1")}}
    print("wrote", C.dump(res, "q5_ceiling_artifact.json"))


if __name__ == "__main__":
    main()
