"""Q1: series kernel checks against PRE-SET tolerances (fixed before running, from the task / Challenge):

  forward:   max|h32 - h64| <= 1e-5 * max|h64| per channel  OR  <= 1e-4 absolute in the model's input
             scale (1 unit = the target scale S, the StandardScaler.scale_ of the training deltas that
             WindowNormalizer divides by; S is computed below from the bundled file, fold -1)
  split:     two halves with the carried state vs one pass, same tolerance
  causality: perturbing bars after t leaves outputs up to t BITWISE unchanged (CPU);
             d out[t] / d input[t' > t] == 0 exactly
  gradient:  d loss / d (period logit) within 1e-3 relative of float64 finite differences;
             finite at the alpha clamps (and beyond: no clamp is needed in the log-decay form)
  non-finite input: refused with an error

Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_kernel_checks.py
Output:  q1_results.json
"""
import time

from common import dump, env_info, load_close, np, target_scale, tf
from kernel import NEG, linrec, logit_period, ref_linrec, sigmoid64, softplus64

TOL_REL, TOL_ABS, TOL_GRAD = 1e-5, 1e-4, 1e-3
S = target_scale()
CLOSE = load_close()
PERIODS = [2, 5, 14, 30, 60, 240, 1440, 10080, 40000, 1e6]
INF_LOGIT = -40.0                        # "no ceiling" limit: alpha = 4e-18, exp(la) == 1.0f
NAMES = [f"p{int(p)}" for p in PERIODS] + ["p_inf(logit-40)"]
BASE = np.concatenate([logit_period(PERIODS), [INF_LOGIT]])

VARIANTS = {
    "V1_segsum_hier_mulsum": dict(seg="segsum", G=32, contract="mulsum", form="logdecay"),
    "V1e_segsum_hier_einsum": dict(seg="segsum", G=32, contract="einsum", form="logdecay"),
    "V2_segsub_hier_einsum": dict(seg="segsub", G=32, contract="einsum", form="logdecay"),
    "V3_mat2_global_decayinput": dict(seg="segsub", G=10 ** 9, contract="einsum", form="decay"),
    "V3b_mat2_global_logdecay": dict(seg="segsub", G=10 ** 9, contract="einsum", form="logdecay"),
    "V1_decayinput": dict(seg="segsum", G=32, contract="mulsum", form="decay"),
    "V1e_tf32_rn_emulated": dict(seg="segsum", G=32, contract="einsum", form="logdecay", tf32="rn"),
    "V1e_tf32_rz_emulated": dict(seg="segsum", G=32, contract="einsum", form="logdecay", tf32="rz"),
}


def series(T, perbar, base=BASE):
    c = CLOSE[-(T + 1):]
    dx64 = np.diff(c) / S
    dx32 = dx64.astype(np.float32)
    z = (dx64 - dx64.mean()) / dx64.std()
    shift = 0.5 * np.tanh(z) if perbar else np.zeros(T)
    lam32 = (np.asarray(base)[:, None] + shift[None, :]).astype(np.float32)
    return dx32, lam32, shift.astype(np.float32)


def tf_decay(lam, form):
    """(la, decay, alpha) in float32 from logits. 'logdecay': la = -softplus(logit) (exact);
    'decay': decay = 1 - sigmoid(logit) rounded in float32, la = log(decay) (the first round's input)."""
    if form == "logdecay":
        la = -tf.nn.softplus(lam)
        return la, tf.exp(la), tf.sigmoid(lam)
    dec = 1.0 - tf.sigmoid(lam)
    return tf.math.log(dec), dec, tf.sigmoid(lam)


def ref_decay(lam32):
    lam = np.asarray(lam32, np.float64)
    la = -softplus64(lam)
    return la, np.exp(la), sigmoid64(lam)


def stats(h, ref):
    err = np.abs(np.asarray(h, np.float64) - ref).max(axis=-1)
    mx = np.abs(ref).max(axis=-1)
    rel = err / np.maximum(mx, 1e-300)
    return err, mx, rel


def verdicts(err, mx, rel):
    return [{"max_abs_err": float(e), "max_abs_state": float(m), "rel_err": float(r),
             "pass_rel_1e-5": bool(r <= TOL_REL), "pass_abs_1e-4": bool(e <= TOL_ABS),
             "PASS": bool(r <= TOL_REL or e <= TOL_ABS)} for e, m, r in zip(err, mx, rel)]


def run_pipeline(dx32, lam32, v, C):
    """Stage 1 increment channels d = EMA - close, RSI gain/loss EWMAs, stage 2 EWMA(d^2), all periods."""
    kw = dict(C=C, G=v["G"], seg=v["seg"], contract=v["contract"], tf32=v.get("tf32"))
    lam = tf.constant(lam32)
    dx = tf.constant(dx32)[None, :]
    la, dec, al = tf_decay(lam, v["form"])
    d, _ = linrec(la, -dec * dx, **kw)
    g, _ = linrec(la, al * tf.nn.relu(dx), **kw)
    ls, _ = linrec(la, al * tf.nn.relu(-dx), **kw)
    var, _ = linrec(la, al * tf.square(d), **kw)
    return d.numpy(), g.numpy(), ls.numpy(), var.numpy()


def ref_pipeline(dx32, lam32):
    la, dec, al = ref_decay(lam32)
    dx = np.asarray(dx32, np.float64)[None, :]
    d = ref_linrec(la, -dec * dx)
    g = ref_linrec(la, al * np.maximum(dx, 0))
    ls = ref_linrec(la, al * np.maximum(-dx, 0))
    var = ref_linrec(la, al * d * d)
    return d, g, ls, var


def rsi(g, ls):
    return 100.0 - 100.0 / (1.0 + g / (ls + 1e-8))


def macd_check(T, perbar, v, C):
    """MACD(fast, slow, signal) as d_fast - d_slow, signal = EWMA(macd) (stage 2)."""
    triples = [(12, 26, 9), (60, 240, 60), (240, 1440, 240), (1440, 10080, 1440), (10080, 40000, 10080)]
    f, s_, g = (np.array([t[i] for t in triples], float) for i in range(3))
    dx32, lf, _ = series(T, perbar, logit_period(f))
    _, ls_, _ = series(T, perbar, logit_period(s_))
    _, lg, _ = series(T, perbar, logit_period(g))
    kw = dict(C=C, G=v["G"], seg=v["seg"], contract=v["contract"], tf32=v.get("tf32"))
    dx = tf.constant(dx32)[None, :]
    out = {}
    la_f, dec_f, _ = tf_decay(tf.constant(lf), v["form"])
    la_s, dec_s, _ = tf_decay(tf.constant(ls_), v["form"])
    la_g, _, al_g = tf_decay(tf.constant(lg), v["form"])
    df, _ = linrec(la_f, -dec_f * dx, **kw)
    ds, _ = linrec(la_s, -dec_s * dx, **kw)
    m = df - ds
    sig, _ = linrec(la_g, al_g * m, **kw)
    rf, rs, rg = ref_decay(lf), ref_decay(ls_), ref_decay(lg)
    dxx = np.asarray(dx32, np.float64)[None, :]
    m64 = ref_linrec(rf[0], -rf[1] * dxx) - ref_linrec(rs[0], -rs[1] * dxx)
    sig64 = ref_linrec(rg[0], rg[2] * m64)
    for name, h, r in (("macd_line", m.numpy(), m64), ("macd_signal", sig.numpy(), sig64)):
        e, mx, rel = stats(h, r)
        out[name] = dict(zip([f"{int(a)}-{int(b)}-{int(c)}" for a, b, c in triples], verdicts(e, mx, rel)))
    return out


def precision():
    res = {}
    for T in (30720, 43008):
        for perbar in (False, True):
            dx32, lam32, _ = series(T, perbar)
            t0 = time.perf_counter()
            ref = ref_pipeline(dx32, lam32)
            rsi64 = rsi(ref[1], ref[2])
            key_a = "per_bar_alpha" if perbar else "constant_alpha"
            print(f"T={T} {key_a}: reference {time.perf_counter() - t0:.1f}s")
            for C in (64, 128):
                for vn, v in VARIANTS.items():
                    out = run_pipeline(dx32, lam32, v, C)
                    entry = {}
                    for name, h, r in (("d_ema_minus_close", out[0], ref[0]), ("rsi_gain_ewma", out[1], ref[1]),
                                       ("rsi_loss_ewma", out[2], ref[2]), ("bb_var_ewma_d2", out[3], ref[3])):
                        e, mx, rel = stats(h, r)
                        entry[name] = dict(zip(NAMES, verdicts(e, mx, rel)))
                    rs = rsi(out[1], out[2])
                    e = np.abs(rs - rsi64)[:, 1000:].max(axis=-1)    # after the first 1,000 bars (RSI 0/0 start)
                    entry["rsi_value_points_max_abs_err_after_1000"] = dict(zip(NAMES, map(float, e)))
                    if vn in ("V1_segsum_hier_mulsum", "V1e_segsum_hier_einsum", "V1e_tf32_rn_emulated",
                              "V3_mat2_global_decayinput"):
                        entry["macd"] = macd_check(T, perbar, v, C)
                    entry["ALL_PASS"] = bool(all(c["PASS"] for k, ch in entry.items() if isinstance(ch, dict)
                                                 and k != "macd" for c in ch.values() if isinstance(c, dict)))
                    res.setdefault(f"T{T}", {}).setdefault(f"C{C}", {}).setdefault(key_a, {})[vn] = entry
                    fails = [f"{k}:{n}" for k, ch in entry.items() if isinstance(ch, dict) and k != "macd"
                             for n, c in ch.items() if isinstance(c, dict) and not c["PASS"]]
                    print(f"  C={C} {vn}: ALL_PASS={entry['ALL_PASS']} fails={fails[:8]}")
    return res


def split_invariance():
    res = {}
    T = 30720
    dx32, lam32, _ = series(T, True)
    ref = ref_pipeline(dx32, lam32)
    for vn in ("V1_segsum_hier_mulsum", "V1e_segsum_hier_einsum"):
        v = VARIANTS[vn]
        for C in (64, 128):
            kw = dict(C=C, G=v["G"], seg=v["seg"], contract=v["contract"])
            lam = tf.constant(lam32)
            dx = tf.constant(dx32)[None, :]
            la, dec, al = tf_decay(lam, "logdecay")
            b1 = -dec * dx
            full_d, _ = linrec(la, b1, **kw)
            full_v, _ = linrec(la, al * tf.square(full_d), **kw)
            for sp in (15360, 15001):
                d1, s1 = linrec(la[:, :sp], b1[:, :sp], **kw)
                d2, _ = linrec(la[:, sp:], b1[:, sp:], h0=s1, **kw)
                dd = tf.concat([d1, d2], -1)
                v1, sv = linrec(la[:, :sp], (al * tf.square(dd))[:, :sp], **kw)
                v2, _ = linrec(la[:, sp:], (al * tf.square(dd))[:, sp:], h0=sv, **kw)
                vv = tf.concat([v1, v2], -1)
                out = {}
                for name, two, one, r in (("d_ema_minus_close", dd.numpy(), full_d.numpy(), ref[0]),
                                          ("bb_var_ewma_d2", vv.numpy(), full_v.numpy(), ref[3])):
                    e = np.abs(two.astype(np.float64) - one).max(-1)
                    mx = np.abs(r).max(-1)
                    diffbits = int(np.sum(two.view(np.uint32) != one.view(np.uint32)))
                    out[name] = {"per_channel": dict(zip(NAMES, verdicts(e, mx, e / mx))),
                                 "n_elements_bitwise_different": diffbits,
                                 "bitwise_equal_first_half": bool(np.array_equal(two[:, :sp].view(np.uint32),
                                                                                 one[:, :sp].view(np.uint32)))}
                out["ALL_PASS"] = bool(all(c["PASS"] for k in ("d_ema_minus_close", "bb_var_ewma_d2")
                                           for c in out[k]["per_channel"].values()))
                res[f"{vn}_C{C}_split{sp}"] = out
                print(f"split {vn} C={C} at {sp}: ALL_PASS={out['ALL_PASS']} "
                      f"max rel d={max(c['rel_err'] for c in out['d_ema_minus_close']['per_channel'].values()):.2e} "
                      f"bitdiff={out['d_ema_minus_close']['n_elements_bitwise_different']}")
    return res


def causality():
    res = {}
    T = 30720
    dx32, lam32, _ = series(T, True)
    rng = np.random.default_rng(7)
    probes = [0, 63, 64, 127, 128, 2047, 2048, 4095, 4096, 15001, 30719]
    for vn in ("V1_segsum_hier_mulsum", "V1e_segsum_hier_einsum"):
        v = VARIANTS[vn]
        for C in (64, 128):
            kw = dict(C=C, G=v["G"], seg=v["seg"], contract=v["contract"])

            def fwd(dx_np, lam_np):
                lam = tf.constant(lam_np)
                dx = tf.constant(dx_np)[None, :]
                la, dec, al = tf_decay(lam, "logdecay")
                d, _ = linrec(la, -dec * dx, **kw)
                var, _ = linrec(la, al * tf.square(d), **kw)
                return d.numpy(), var.numpy()

            base_d, base_v = fwd(dx32, lam32)
            bit_ok = True
            for t in probes:
                if t == T - 1:
                    continue
                dxp, lp = dx32.copy(), lam32.copy()
                dxp[t + 1:] = rng.normal(0, 5, size=T - t - 1).astype(np.float32)
                lp[:, t + 1:] = (lp[:, t + 1:] + rng.normal(0, 1, size=lp[:, t + 1:].shape)).astype(np.float32)
                pd_, pv = fwd(dxp, lp)
                ok = (np.array_equal(pd_[:, :t + 1].view(np.uint32), base_d[:, :t + 1].view(np.uint32))
                      and np.array_equal(pv[:, :t + 1].view(np.uint32), base_v[:, :t + 1].view(np.uint32)))
                bit_ok &= bool(ok)
            # Jacobian rows: d out[:, t] / d dx[t'] and / d logit[:, t'] for t' > t must be exactly 0
            jac_ok, nz_past = True, []
            dxv = tf.Variable(dx32)
            lamv = tf.Variable(lam32)
            for t in probes:
                with tf.GradientTape() as tape:
                    la, dec, al = tf_decay(lamv, "logdecay")
                    d, _ = linrec(la, -dec * dxv[None, :], **kw)
                    var, _ = linrec(la, al * tf.square(d), **kw)
                    y = tf.reduce_sum(d[:, t]) + tf.reduce_sum(var[:, t])
                gdx, glam = tape.gradient(y, [dxv, lamv])
                gdx, glam = gdx.numpy(), glam.numpy()
                fut = np.concatenate([gdx[t + 1:], glam[:, t + 1:].ravel()])
                jac_ok &= bool(np.all(fut == 0.0))
                nz_past.append(int(np.count_nonzero(gdx[:t + 1])))
            res[f"{vn}_C{C}"] = {"perturb_after_t_bitwise_unchanged": bit_ok, "future_jacobian_exactly_zero": jac_ok,
                                 "nonzero_past_dx_grads_per_probe": dict(zip(map(str, probes), nz_past)),
                                 "probes": probes}
            print(f"causality {vn} C={C}: bitwise {bit_ok}, future jacobian zero {jac_ok}")
    return res


def gradients():
    """d L / d base logit, L = sum_t w_t h_t over the second half (anchors far from the start)."""
    res = {}
    T = 30720
    per = [2, 5, 14, 30, 60, 240, 1440, 10080, 40000]
    names = [f"p{p}" for p in per]
    base = logit_period(per)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(len(per), T))
    w[:, : T // 2] = 0.0
    for perbar in (False, True):
        dx32, _, shift32 = series(T, perbar, base)
        dxx = np.asarray(dx32, np.float64)[None, :]

        def L64(beta, kind):
            lam = np.asarray(beta, np.float64)[:, None] + np.asarray(shift32, np.float64)[None, :]
            la, dec, al = ref_decay(lam)
            d = ref_linrec(la, -dec * dxx)
            if kind == "incr":
                return (w * d).sum(1)
            var = ref_linrec(la, al * d * d)
            g = ref_linrec(la, al * np.maximum(dxx, 0))
            ls = ref_linrec(la, al * np.maximum(-dxx, 0))
            return (w * (d + var + rsi(g, ls) / 100.0)).sum(1)

        for kind in ("incr", "bb_rsi_pipeline"):
            fd = {}
            for eps in (1e-4, 1e-5):
                fd[eps] = (L64(base + eps, kind) - L64(base - eps, kind)) / (2 * eps)
            g_fd = fd[1e-5]
            fd_conv = np.abs(fd[1e-4] - fd[1e-5]) / np.abs(fd[1e-5])

            def tf_grad(dtype, v, C):
                beta = tf.Variable(base.astype(dtype))
                sh = tf.constant(np.asarray(shift32).astype(dtype))[None, :]
                dx = tf.constant(np.asarray(dx32).astype(dtype))[None, :]
                ww = tf.constant(w.astype(dtype))
                kw = dict(C=C, G=v["G"], seg=v["seg"], contract=v["contract"])
                with tf.GradientTape() as tape:
                    la, dec, al = tf_decay(beta[:, None] + sh, "logdecay")
                    d, _ = linrec(la, -dec * dx, **kw)
                    if kind == "incr":
                        Lv = tf.reduce_sum(ww * d)
                    else:
                        var, _ = linrec(la, al * tf.square(d), **kw)
                        g, _ = linrec(la, al * tf.nn.relu(dx), **kw)
                        ls, _ = linrec(la, al * tf.nn.relu(-dx), **kw)
                        r = 100.0 - 100.0 / (1.0 + g / (ls + 1e-8))
                        Lv = tf.reduce_sum(ww * (d + var + r / 100.0))
                gr = tape.gradient(Lv, beta).numpy().astype(np.float64)
                return gr

            g_ad64 = tf_grad(np.float64, VARIANTS["V1_segsum_hier_mulsum"], 64)
            entry = {"fd_eps_1e-4_vs_1e-5_rel": dict(zip(names, map(float, fd_conv))),
                     "tf_float64_autodiff_vs_fd_rel": dict(zip(names, map(float, np.abs(g_ad64 - g_fd) / np.abs(g_fd))))}
            for vn in ("V1_segsum_hier_mulsum", "V1e_segsum_hier_einsum"):
                for C in (64, 128):
                    g32 = tf_grad(np.float32, VARIANTS[vn], C)
                    rel = np.abs(g32 - g_fd) / np.abs(g_fd)
                    entry[f"{vn}_C{C}"] = {"rel_err_vs_fd": dict(zip(names, map(float, rel))),
                                           "finite": bool(np.isfinite(g32).all()),
                                           "PASS_1e-3": bool(np.all(rel <= TOL_GRAD))}
                    print(f"grad {kind} {'perbar' if perbar else 'const'} {vn} C={C}: max rel {rel.max():.2e} "
                          f"PASS={bool(np.all(rel <= TOL_GRAD))}")
            res[f"{kind}_{'per_bar_alpha' if perbar else 'constant_alpha'}"] = entry
    return res


def clamps():
    """Forward and gradient at and beyond the repo's alpha clamps (1e-6, 1-1e-6), per-bar shift on top."""
    res = {}
    T = 30720
    cases = {"alpha=1e-6 (logit -13.8)": np.log(1e-6) - np.log1p(-1e-6), "alpha=1-1e-6 (logit +13.8)": 13.815509557963773,
             "logit -30": -30.0, "logit +30": 30.0, "logit -40": -40.0}
    base = np.array(list(cases.values()))
    dx32, _, shift32 = series(T, True, base)
    rng = np.random.default_rng(3)
    w = rng.normal(size=(len(base), T))
    w[:, : T // 2] = 0

    def grad(dtype):
        beta = tf.Variable(base.astype(dtype))
        sh = tf.constant(np.asarray(shift32).astype(dtype))[None, :]
        dx = tf.constant(np.asarray(dx32).astype(dtype))[None, :]
        with tf.GradientTape() as tape:
            la, dec, al = tf_decay(beta[:, None] + sh, "logdecay")
            d, _ = linrec(la, -dec * dx, C=64)
            var, _ = linrec(la, al * tf.square(d), C=64)
            Lv = tf.reduce_sum(tf.constant(w.astype(dtype)) * (d + var))
        g = tape.gradient(Lv, beta).numpy()
        return g, bool(np.isfinite(d.numpy()).all() and np.isfinite(var.numpy()).all())

    g32, fin32 = grad(np.float32)
    g64, _ = grad(np.float64)
    for (name, _), a, b in zip(cases.items(), g32, g64):
        res[name] = {"grad_f32": float(a), "grad_f64": float(b), "finite": bool(np.isfinite(a)),
                     "rel_err_vs_f64": float(abs(a - b) / abs(b)) if b != 0 else None}
    res["forward_finite"] = fin32
    print("clamps:", res)
    return res


def nonfinite():
    res = {}
    T = 30720
    dx32, lam32, _ = series(T, True)
    for bad in (np.nan, np.inf):
        dxb = dx32.copy()
        dxb[20000] = bad
        la, dec, al = tf_decay(tf.constant(lam32), "logdecay")
        b = -dec * tf.constant(dxb)[None, :]
        try:
            linrec(la, b, check_finite=True)
            raised = None
        except tf.errors.InvalidArgumentError as e:
            raised = type(e).__name__ + ": " + str(e).splitlines()[0][:120]
        f = tf.function(lambda la_, b_: linrec(la_, b_, check_finite=True)[0])
        try:
            f(la, b)
            raised_fn = None
        except tf.errors.InvalidArgumentError as e:
            raised_fn = type(e).__name__
        h, _ = linrec(la, b)
        h = h.numpy()
        res[str(bad)] = {"eager_raises": raised, "tf_function_raises": raised_fn,
                         "without_check_nonfinite_outputs_before_bar_20000": int((~np.isfinite(h[:, :20000])).sum()),
                         "of": int(h[:, :20000].size)}
    print("nonfinite:", res)
    return res


def window_repro():
    """Cold start on one 60-bar window reproduces today's ewma_sequence_matrix_multi (round-1 gate: 1e-6)."""
    import neural_trade.utils.math as mh
    rng = np.random.default_rng(1)
    per = np.array([5, 10, 30, 12, 26, 9, 5, 35, 5, 8, 17, 9, 10, 20, 25, 9, 14, 21], float)
    errs = []
    for _ in range(20):
        i = int(rng.integers(100, len(CLOSE) - 1))
        x = ((CLOSE[i - 60:i] - CLOSE[i - 1]) / S).astype(np.float32)
        a = (2.0 / (per + 1.0)).astype(np.float32)
        today = mh.ewma_sequence_matrix_multi(tf.constant(np.broadcast_to(x, (1, len(per), 60)).copy()),
                                              tf.constant(a[None, :])).numpy()[0]
        lam = tf.constant(np.log(a) - np.log1p(-a))[:, None] * tf.ones([1, 60])
        la = -tf.nn.softplus(lam)
        b = tf.sigmoid(lam) * tf.constant(x)[None, :]
        h, _ = linrec(la, b, h0=tf.fill([len(per)], x[0]), C=64)     # h_{-1} = x_0  <=>  ema[0] = x[0]
        errs.append(float(np.abs(h.numpy() - today).max()))
    return {"max_abs_err_20_windows": max(errs), "PASS_1e-6": bool(max(errs) <= 1e-6)}


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = {"env": env_info(), "target_scale_S_dollars": S,
           "tolerances": {"forward_rel_of_max_state": TOL_REL, "forward_abs_input_units": TOL_ABS,
                          "gradient_rel_vs_fd64": TOL_GRAD, "causality": "bitwise (CPU)"},
           "periods": NAMES}
    out["window_repro"] = window_repro()
    print("window repro:", out["window_repro"])
    out["nonfinite"] = nonfinite()
    out["clamps"] = clamps()
    out["causality"] = causality()
    out["split_invariance"] = split_invariance()
    out["gradients"] = gradients()
    out["precision"] = precision()
    out["seconds"] = time.perf_counter() - t0
    dump("q1_results.json", out)
