"""Q3: how a per-bar adaptive period would be reported, measured on the newest run's served weights
(runs/20260924T182915Z-1aeff1c-dirty-af67ee43) as if its indicator layer ran in series mode (a hypothetical:
the network was trained in window mode). CPU, float64, read-only.

Per instance, over the fold -1 test block: the global learned (base) period, the textbook default, the
per-bar instantaneous period p_t = 2 / alpha_t - 1 (5 / 50 / 95 %), the per-bar EFFECTIVE period
p_eff,t = 2 m_t + 1 with m_t = (1 - alpha_t)(m_{t-1} + 1) the mean lag of the EWMA's weights at bar t
(exact for a time-varying alpha; equals p for a constant one), and today's per-window applied period.
Also: the semantics of the candidates on single channels (d = EMA - close, scaled) against the fixed-
period EWMA at the bar's own period; the textbook comparison; the off switch and the frozen twin.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_reporting.py
Writes q3_reporting.json.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np

import common as C

NEWEST = f"{C.RUNS}/20260924T182915Z-1aeff1c-dirty-af67ee43/"


def pct(v, qs=(5, 50, 95)):
    return [float(x) for x in np.percentile(v, qs)]


def d_rec(dx, alpha):
    """d_t = (1 - a_t)(d_{t-1} - dx_t), a scalar or per bar, float64 (d = EMA - close in scaled units)."""
    a = np.broadcast_to(np.asarray(alpha, np.float64), dx.shape)
    return C.linrec64((1 - a)[None], (-(1 - a) * dx)[None])[0]


def mean_lag(alpha):
    a = np.asarray(alpha, np.float64)
    m = np.empty_like(a)
    prev = 0.0
    for t in range(len(a)):
        prev = (1 - a[t]) * (prev + 1)
        m[t] = prev
    return m


def d_fixed_at(close, scale, bars, alpha):
    """Fixed-alpha EWMA over the full history at each bar's own alpha (renormalised kernel), minus close."""
    out = np.empty(len(bars))
    for i, (t, a) in enumerate(zip(bars, alpha)):
        nl = min(t + 1, int(np.ceil(np.log(1e-10) / np.log1p(-a))))
        w = a * (1 - a) ** np.arange(nl)
        w /= w.sum()
        out[i] = ((w * close[t - np.arange(nl)]).sum() - close[t]) / scale
    return out


def main():
    info = C.run_info(NEWEST)
    B = C.blocks(-1)
    close = B["close"]
    s = info["scale"]
    dx = np.diff(close, prepend=close[0]) / s                    # dx[t] = (close_t - close_{t-1}) / s, dx[0] = 0
    ctx_r = C.rolling_context(close, s)
    ctx_e = C.ew_context(close, s)
    lg = info["logits"]
    delta_r = C.meta_shift(ctx_r, info["W"], info["b"]).T        # [18, N]
    delta_e = C.meta_shift(ctx_e, info["W"], info["b"]).T
    al_r = np.clip(C.sigmoid(lg[:, None] + delta_r), 1e-6, 1 - 1e-6)
    al_e = np.clip(C.sigmoid(lg[:, None] + delta_e), 1e-6, 1 - 1e-6)
    test = B["test"]
    today = C.period_of_logit(lg[None, :] + C.meta_shift(C.today_context(close, test, s), info["W"], info["b"]))
    res = {"run": info["run"], "block": "test", "n_anchors": int(len(test)), "instances": {},
           "ew_context_vs_today": {"corr_mean_offset": float(np.corrcoef(ctx_e[test, 0], ctx_r[test, 0])[0, 1]),
                                   "corr_max_offset": float(np.corrcoef(ctx_e[test, 1], ctx_r[test, 1])[0, 1])}}
    inst = C.period_of_logit(lg[:, None] + delta_r)              # instantaneous per-bar period [18, N]
    res["rolling_mirror_equals_today_at_anchors_max_rel_diff"] = float(np.max(np.abs(inst[:, test].T - today) / today))
    for j, n in enumerate(C.NAMES):
        m = mean_lag(al_r[j]); me = mean_lag(al_e[j])
        peff, peff_e = 2 * m + 1, 2 * me + 1
        res["instances"][n] = {
            "textbook": C.TEXTBOOK[n], "base_global": float(C.period_of_logit(lg[j])),
            "today_per_window_p5_p50_p95": pct(today[:, j]),
            "instantaneous_p5_p50_p95": pct(inst[j, test]), "instantaneous_min_max": [float(inst[j, test].min()), float(inst[j, test].max())],
            "effective_p5_p50_p95": pct(peff[test]), "effective_min_max": [float(peff[test].min()), float(peff[test].max())],
            "ew_ctx_instantaneous_p5_p50_p95": pct(C.period_of_logit(lg[j] + delta_e[j, test])),
            "ew_ctx_effective_p5_p50_p95": pct(peff_e[test])}
    # semantics on single d-type channels
    sem = {}
    rng = np.random.default_rng(4)
    sample = np.sort(rng.choice(test, 400, replace=False))
    for n in ("ma_period_0", "ma_period_2", "macd_0_slow", "macd_1_slow", "bb_period_2"):
        j = C.NAMES.index(n)
        da = d_rec(dx, al_r[j])
        dc = d_rec(dx, C.sigmoid(lg[j]))
        dt = d_rec(dx, 2 / (C.TEXTBOOK[n] + 1))
        banks = {}
        for M in (3, 5):
            off = np.linspace(-0.5, 0.5, M)
            w = np.maximum(0, 1 - np.abs(delta_r[j][None, :] - off[:, None]) / (off[1] - off[0]))     # [M, N]
            banks[M] = sum(w[k] * d_rec(dx, C.sigmoid(lg[j] + off[k])) for k in range(M))
        exact = d_fixed_at(close, s, sample, al_r[j, sample])
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
        ref = rms(exact)
        sem[n] = {"rms_channel_test": rms(dc[test]),
                  "a_minus_fixed_at_bar_period_rel": rms(da[sample] - exact) / ref,
                  "bank3_minus_fixed_at_bar_period_rel": rms(banks[3][sample] - exact) / ref,
                  "bank5_minus_fixed_at_bar_period_rel": rms(banks[5][sample] - exact) / ref,
                  "a_minus_c_rel": rms(da[test] - dc[test]) / rms(dc[test]),
                  "bank3_minus_c_rel": rms(banks[3][test] - dc[test]) / rms(dc[test]),
                  "roughness_rel_to_c": {"a": rms(np.diff(da[test])) / rms(np.diff(dc[test])),
                                         "bank3": rms(np.diff(banks[3][test])) / rms(np.diff(dc[test])),
                                         "bank5": rms(np.diff(banks[5][test])) / rms(np.diff(dc[test]))},
                  "learned_adaptive_minus_textbook_rel": rms(da[test] - dt[test]) / rms(dt[test]),
                  "learned_base_minus_textbook_rel": rms(dc[test] - dt[test]) / rms(dt[test]),
                  "corr_learned_adaptive_textbook": float(np.corrcoef(da[test], dt[test])[0, 1])}
    res["semantics_single_channels"] = sem
    # off switch (delta = 0) and frozen twin (textbook logits, delta = 0) through the TF layer
    import tensorflow as tf
    from candidates import SeriesLayer, reference64
    n = 30720
    dxs = (np.diff(close[-(n + 1):]) / s).astype(np.float32)
    cx = ctx_r[-n:].astype(np.float32)
    off_a = SeriesLayer("a", lg, info["W"] * 0, info["b"] * 0)(dxs, cx).numpy()     # switch = zero shift
    off_c = SeriesLayer("c", lg, info["W"], info["b"])(dxs, cx).numpy()
    tb_lg = C.logit_of_period([C.TEXTBOOK[k] for k in C.NAMES])
    twin = SeriesLayer("c", tb_lg, info["W"], info["b"])(dxs, cx).numpy()
    twin_ref = reference64("c", dxs.astype(np.float64), tb_lg.astype(np.float32).astype(np.float64), None)
    res["off_switch"] = {"a_with_zero_shift_vs_c_max_abs": float(np.abs(off_a - off_c).max()),
                         "frozen_twin_vs_float64_textbook_max_abs": float(np.abs(twin - twin_ref).max()),
                         "note": "a with zero shift runs the per-bar kernel, c the Toeplitz kernel: equal to float32 round-off; "
                                 "a switch that also selects the Toeplitz kernel makes them identical"}
    path = C.dump(res, "q3_reporting.json")
    for k in ("ma_period_0", "macd_0_slow", "macd_1_slow", "bb_period_2"):
        v = res["instances"][k]
        print(k, "textbook", v["textbook"], "base %.1f" % v["base_global"], "today", np.round(v["today_per_window_p5_p50_p95"], 1),
              "inst", np.round(v["instantaneous_p5_p50_p95"], 1), "eff", np.round(v["effective_p5_p50_p95"], 1))
    print("mirror==today", res["rolling_mirror_equals_today_at_anchors_max_rel_diff"], "ew corr", res["ew_context_vs_today"])
    for k, v in sem.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else {a: round(b, 3) for a, b in vv.items()}) for kk, vv in v.items()})
    print("off", res["off_switch"])
    print("wrote", path)


if __name__ == "__main__":
    main()
