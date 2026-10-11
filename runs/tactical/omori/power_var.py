"""Power simulation for hypothesis #6, BEFORE the real run (SPEC_var.md section 4). usage: python power_var.py [--workers 4] [--reps 20]
World: 1-minute returns r_t = s_t u_t, u_t iid unit-variance Student-t(5), s_t = exp(gamma x_t), x_t a unit-variance AR(1) with a 240-bar
time constant ('tail state'); sigma_hat is perfect (1), so FHS is the unconditional standardised quantile. The challenger sees the
state through a noisy proxy (x + N(0, 0.7^2)) plus 7 pure-noise features (the real challenger has 8 features): the fit cost is included.
12 training months, 66 scored months (monthly means, the DM-type t test over months, one-sided, Bonferroni over the 8 primary cells),
stride 5, T in {15, 60}, alpha in {2.5%, 5%, 1%}, lower tail. Output results/power_var.json: gain (%) of FZ0 and the rejection rate."""
import os, sys, json, math, time
import numpy as np
from scipy import stats
from scipy.signal import lfilter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import varlib as V

GAMMAS = (0.0, 0.10, 0.15, 0.20, 0.30); MONTH = 43200; NTR, NTE = 12, 66; STRIDE = 5; NOISE_SD = 0.7


def one(args):
    gamma, seed = args
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    rng = np.random.default_rng(seed); n = (NTR + NTE) * MONTH
    phi = 1 - 1 / 240.0; x = lfilter([math.sqrt(1 - phi ** 2)], [1, -phi], rng.normal(size=n))
    u = rng.standard_t(5, n) / math.sqrt(5 / 3.0); r = np.exp(gamma * x) * u; cs = np.concatenate([[0.0], np.cumsum(r)])
    res = {}
    for T in (15, 60):
        anch = np.arange(1500, n - T - 1, STRIDE); z = (cs[anch + T] - cs[anch]) / math.sqrt(T)
        xo = x[anch - 1] + rng.normal(0, NOISE_SD, len(anch)); F = np.concatenate([xo[:, None], rng.normal(size=(len(anch), 7))], 1)
        mon = anch // MONTH; tr = mon < NTR; te = ~tr
        for al in (0.025, 0.05, 0.01):
            q, es = V.fhs(z[tr], al); Xtr = (F[tr] - F[tr].mean(0)) / F[tr].std(0); Xte = (F[te] - F[tr].mean(0)) / F[tr].std(0)
            th = V.fit_fz0(Xtr, z[tr], al, q, es); v, e = V.predict(th, Xte)
            Lf = V.fz0(z[te], q, es, al); Lc = V.fz0(z[te], v, e, al)
            mm = mon[te]; d = np.array([(Lf[mm == m] - Lc[mm == m]).mean() for m in range(NTR, NTR + NTE)])
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            res[f"{T}_{al}"] = dict(gain=float(d.mean() / Lf.mean()), t=float(t), p1=float(1 - stats.t.cdf(t, len(d) - 1)))
    return gamma, seed, res


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
    jobs = [(g, 1000 + s) for g in GAMMAS for s in range(reps)]; t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(w) as pool:
        out = pool.map(one, jobs)
    summ = {}
    for g in GAMMAS:
        rs = [r for gg, s, r in out if gg == g]
        for key in rs[0]:
            gains = np.array([r[key]["gain"] for r in rs]); p1 = np.array([r[key]["p1"] for r in rs])
            primary = not key.endswith("_0.01"); thr = 0.05 / 8
            summ[f"gamma{g}_{key}"] = dict(gain_mean=float(gains.mean()), gain_sd=float(gains.std(ddof=1)),
                                           reject_bonf8=float(np.mean(p1 < thr)), reject_bonf8_and_gain1pct=float(np.mean((p1 < thr) & (gains >= 0.01))),
                                           reject_05=float(np.mean(p1 < 0.05)), primary=primary)
    json.dump(dict(reps=reps, seconds=time.time() - t0, summary=summ), open(os.path.join(HERE, "results", "power_var.json"), "w"), indent=1)
    lines = [f"power simulation: {reps} reps x {len(GAMMAS)} gammas, {time.time()-t0:.0f}s", "gamma  cell        gain%   sd%   reject(Bonf/8)  reject(Bonf/8 & gain>=1%)  reject(0.05)"]
    for k, v in summ.items():
        g, c = k.split("_", 1); lines.append(f"{g[5:]:>5}  {c:10s} {100*v['gain_mean']:6.2f} {100*v['gain_sd']:5.2f}  {v['reject_bonf8']:6.2f}  {v['reject_bonf8_and_gain1pct']:6.2f}  {v['reject_05']:6.2f}")
    open(os.path.join(HERE, "results", "power_var_table.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
