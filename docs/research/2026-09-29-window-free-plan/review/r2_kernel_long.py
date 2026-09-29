"""Review check: V1 kernel (segsum, G=32, mulsum, log-decay, C=16) precision beyond 43,008 bars, and the
tolerance's normaliser (max|state| vs RMS of the state).

Passes of 70,000 bars (C/ Q1: the per-step pass at a learned period of 10,080) and 262,144 bars
(about 6 months) of the 2017-2025 file's last closes, per-bar alpha (+-0.5 tanh shift, as A/ Q1),
channels d, RSI gain/loss EWMAs, EWMA(d^2). For each channel: max err / max|ref| (the plan's tolerance),
max err / RMS(ref), and p99 of |err_t| / max(|ref_t|, 1e-3 RMS(ref)).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python r2_kernel_long.py
"""
import json
import sys
import time

sys.path.insert(0, "D:/nt_research/wfp/A")
from common import LONG_CSV, np, pd, target_scale  # noqa: E402
import q1_kernel_checks as q1  # noqa: E402

S = target_scale()


def load_long_tail(n):
    c = pd.read_csv(LONG_CSV, usecols=["close"])["close"].to_numpy(np.float64)
    return c[-(n + 1):]


def run(T, closes):
    dx64 = np.diff(closes[-(T + 1):]) / S
    dx32 = dx64.astype(np.float32)
    z = (dx64 - dx64.mean()) / dx64.std()
    shift = 0.5 * np.tanh(z)
    lam32 = (q1.BASE[:, None] + shift[None, :]).astype(np.float32)
    v = dict(q1.VARIANTS["V1_segsum_hier_mulsum"])
    t0 = time.perf_counter()
    out = q1.run_pipeline(dx32, lam32, v, 16)
    t1 = time.perf_counter()
    ref = q1.ref_pipeline(dx32, lam32)
    res = {}
    for name, h, r in zip(("d", "gain", "loss", "var"), out, ref):
        err = np.abs(np.asarray(h, np.float64) - r)
        mx = np.abs(r).max(-1)
        rms = np.sqrt((r ** 2).mean(-1))
        loc = np.maximum(np.abs(r), 1e-3 * rms[:, None])
        rows = {}
        for k, nm in enumerate(q1.NAMES):
            rows[nm] = {"rel_max": float(err[k].max() / mx[k]), "rel_rms": float(err[k].max() / rms[k]),
                        "p99_local": float(np.percentile(err[k] / loc[k], 99)),
                        "max_over_rms": float(mx[k] / rms[k]),
                        "PASS_plan_tol": bool(err[k].max() <= 1e-5 * mx[k] or err[k].max() <= 1e-4)}
        res[name] = rows
    return {"T": T, "kernel_s": t1 - t0, "channels": res}


def main():
    closes = load_long_tail(262_144 + 1)
    out = {"scale": S}
    for T in (70_000, 262_144):
        r = run(T, closes)
        out[f"T{T}"] = r
        worst = max((v["rel_max"], ch, nm) for ch, rows in r["channels"].items() for nm, v in rows.items())
        worst_rms = max((v["rel_rms"], ch, nm) for ch, rows in r["channels"].items() for nm, v in rows.items())
        fails = [(ch, nm) for ch, rows in r["channels"].items() for nm, v in rows.items() if not v["PASS_plan_tol"]]
        print(T, "kernel s %.1f" % r["kernel_s"], "worst rel_max", worst, "worst rel_rms", worst_rms, "fails", fails)
        for nm in ("p1440", "p10080", "p1000000", "p_inf(logit-40)"):
            print("   d", nm, {k: round(v, 8) if isinstance(v, float) else v for k, v in r["channels"]["d"][nm].items()})
            print("   var", nm, {k: round(v, 8) if isinstance(v, float) else v for k, v in r["channels"]["var"][nm].items()})
    json.dump(out, open("D:/nt_research/wfp/review/r2_kernel_long.json", "w"), indent=1)


if __name__ == "__main__":
    main()
