"""usage: python run_climb.py real | null [--draws 40] | exhaustive      (SPEC.md; dev months only, no lockbox month is ever touched)
real       : the 10 climbs on the real dev months -> climb_real.json, trials_real.jsonl, winner.json (committed BEFORE the look)
null       : the same climbs on the sign-flipped worlds (40 draws) + the exhaustive best of 1,944 per draw -> null_climbs.json
exhaustive : all 1,944 configurations on the real dev months (descriptive) -> exhaustive_real.jsonl"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hc2lib as H  # noqa: E402

BUDGET_CPU_H = 6.0


def slim(r):
    return {k: (float(v) if not isinstance(v, (bool, int)) else v) for k, v in r.items() if k in ("n", "calmar", "calmar_raw", "cum", "dd", "hit", "mean_bps", "valid")}


def trials_dump(ev, path):
    with open(path, "w", encoding="utf-8") as f:
        for k, r in ev.memo.items():
            f.write(json.dumps({"cfg": k, **{a: (None if (isinstance(b, float) and not np.isfinite(b)) else b) for a, b in slim(r).items()}}) + "\n")


def real():
    t0 = time.process_time(); inp = H.Inputs(); st = H.Store(inp); ev = H.Evaluator(st)
    best, runs = H.run_climbs(ev)
    json.dump({"months": list(st.months), "years": st.years, "n_distinct_evaluated": ev.n_eval, "best": {k: v for k, v in best.items()},
               "starts": runs, "cpu_s": time.process_time() - t0}, open(f"{HERE}/climb_real.json", "w"), indent=1, default=str)
    trials_dump(ev, f"{HERE}/trials_real.jsonl")
    base = ev.eval(H.BASELINE)
    json.dump({"winner": best["cfg"], "dev": slim(ev.eval(best["cfg"])), "baseline": H.BASELINE, "baseline_dev": slim(base),
               "chosen_by": "best final dev Calmar over 10 climbs (baseline start + 9 random)", "n_distinct_evaluated": ev.n_eval}, open(f"{HERE}/winner.json", "w"), indent=1)
    print("winner", H.ckey(best["cfg"]), slim(ev.eval(best["cfg"])), "\nbaseline", slim(base), "\nevaluated", ev.n_eval)
    for j, d in enumerate(runs):
        print(j, d["start"], "->", d["final"], f"calmar {d['calmar']:.2f}", "steps", len(d["traj"]) - 1)


def exhaustive(ev=None, path=None):
    ev = ev or H.Evaluator(H.Store(H.Inputs()))
    for c in H.all_configs():
        ev.eval(c)
    if path:
        trials_dump(ev, path)
    return ev


def null_task(d):
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    t0 = time.process_time(); inp = H.Inputs(); st = H.Store(inp)
    flips = np.where(np.random.default_rng(d).random(len(st.months)) < 0.5, 1, -1)
    ev = H.Evaluator(st, flips=flips)
    best, runs = H.run_climbs(ev, seed=H.MO.SEED + 1000 * d)
    base = ev.eval(H.BASELINE)
    exhaustive(ev)
    ok = [r for r in ev.memo.values() if r["valid"]]
    top = max(ok, key=lambda r: r["calmar"]) if ok else None
    return {"draw": d, "n_flipped_months": int((flips < 0).sum()), "climb_final": H.ckey(best["cfg"]), "climb_calmar": best["calmar"], "climb_cum": best["cum"],
            "climb_dd": best["dd"], "baseline_calmar": base["calmar"], "baseline_cum": base["cum"], "exh_best_calmar": top["calmar"] if top else None,
            "exh_best_cum": top["cum"] if top else None, "n_valid_of_1944": len(ok), "cpu_s": time.process_time() - t0,
            "start_calmars": [r["calmar"] for r in runs]}


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        real()
    elif mode == "exhaustive":
        exhaustive(path=f"{HERE}/exhaustive_real.jsonl")
    elif mode == "null":
        import multiprocessing as mp
        n = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 40
        t0 = time.time()
        with mp.Pool(6) as pool:
            res = pool.map(null_task, range(1, n + 1))
        json.dump(res, open(f"{HERE}/null_climbs.json", "w"), indent=1, default=lambda x: None if x is None else float(x))
        cpu = sum(r["cpu_s"] for r in res)
        print(f"null: {n} draws, {time.time() - t0:.0f}s wall, {cpu / 3600:.2f} CPU-h")
        c = np.array([r["climb_calmar"] for r in res if np.isfinite(r["climb_calmar"])])
        print("climb calmar: median %.2f p90 %.2f p95 %.2f max %.2f" % (np.median(c), *np.quantile(c, [.9, .95]), c.max()))
