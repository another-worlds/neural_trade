"""T15 Day 2: falsification and controls (NOTE_day2.md). Resumable (raw/<job>.json). Run: D:/nt/nt_mm_env/python.exe day2.py [analyse-only]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t15_lib as T  # noqa: E402
import build_m7 as B7  # noqa: E402
import evaluate as E  # noqa: E402
import anchors as AN  # noqa: E402
import m7_lib as L7  # noqa: E402
from m6_lib import KEYS  # noqa: E402

DAY = "day2"
OUT = os.path.join(HERE, DAY); RAW = os.path.join(OUT, "raw")
H_SEEDS = list(range(3000, 3010)); W_SEEDS = list(range(4000, 4010))
SHIFTS = (6, 12, 18)
IX = {k: i for i, k in enumerate(KEYS)}
MODULES = ["M9h6", "M7c", "M7_gbm"]
_CTX = {}
_BASES = {}


def ctx():
    if not _CTX:
        D = B7.load_all(resid=True); K = D["keys"]
        okK = D["ok"] & B7.finite_rows(K); ok = okK & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"])
        okr = ok & B7.finite_rows(D["resA"], D["resB"])
        h, w = T.hour_wk(D["ta"])
        _CTX.update(D=D, ta=D["ta"], K=K, ok=ok, okr=okr, hour=h, wk=w, src=D["src"], n_cut=D["n_cut"])
    return _CTX


def base(source):
    if source not in _BASES:
        _BASES[source] = AN.load_base(source)
    return _BASES[source]


def frames_of(cols):
    c = ctx(); df = pd.DataFrame(cols, index=c["ta"]); n = c["n_cut"]; out = {}
    for s, sl in (("cut", slice(0, n)), ("fwd", slice(n, None))):
        part = df.iloc[sl]; out[s] = part[part.notna().all(axis=1)]
    return out


# ------------------------------------------------------------------ module views
def view_m9h6(hour=None, wk=None):
    c = ctx(); Z, names = T.harm_prof_str(6, c["hour"] if hour is None else hour, c["wk"] if wk is None else wk, c["K"]); Z = Z.copy(); Z[~c["ok"]] = np.nan
    return frames_of({n: Z[:, j] for j, n in enumerate(names)})


def month_start_fn(starts):
    """starts: None or {'look1': ts, 'look2': ts}; the start of the look a month belongs to (months before 2025-09 are Look 1)."""
    if not starts:
        return lambda mo: None
    return lambda mo: starts["look1"] if mo.start_time < pd.Timestamp("2025-09-01") else starts["look2"]


def mview(ta, X, y, ok, source, binary, pool_src=None, start_fn=None):
    """m7_lib.monthly_view (GBM) with an optional pool restricted to one source and a per-month pool start. With both None it is the same computation."""
    ta = pd.DatetimeIndex(ta); N = len(ta); pred = np.full(N, np.nan); rnk = np.full(N, np.nan); per = np.asarray(ta.to_period("M")); end = ta + pd.Timedelta(minutes=L7.LABEL_MIN)
    served = []
    for mo in sorted(set(per[ok])):
        m0 = mo.start_time; te = np.where(ok & (per == mo))[0]; lo = m0 - pd.Timedelta(days=L7.POOL_DAYS)
        st = start_fn(mo) if start_fn else None
        if st is not None:
            lo = max(lo, pd.Timestamp(st))
        tr_mask = ok & np.asarray(ta >= lo) & np.asarray(end <= m0)
        if pool_src:
            tr_mask &= (source == pool_src)
        tr = np.where(tr_mask)[0]
        if len(tr) < L7.MIN_POOL:
            continue
        pte, ptr, _ = L7.gbm_fit_predict(X[tr], y[tr], X[te], binary)
        pred[te] = pte; rnk[te] = L7.rank_copy(pte, ptr); served.append(str(mo))
    return pred, rnk, served


def view_gbm(kind, Kmat=None, pool_src=None, starts=None):
    c = ctx(); D = c["D"]; Kmat = c["K"] if Kmat is None else Kmat; sf = month_start_fn(starts)
    if kind == "M7c":
        ok = c["okr"]
        ra, _, _ = mview(c["ta"], Kmat, D["resA"], ok, c["src"], False, pool_src, sf); rb, _, _ = mview(c["ta"], Kmat, D["resB"], ok, c["src"], False, pool_src, sf)
        return frames_of({"gbm_resA_hat": ra, "gbm_resB_hat": rb})
    ok = c["ok"]
    pa, ra, _ = mview(c["ta"], Kmat, D["dA"], ok, c["src"], False, pool_src, sf); ph, rh, _ = mview(c["ta"], Kmat, D["hit"], ok, c["src"], True, pool_src, sf)
    return frames_of({"gbm_dA_hat": pa, "gbm_hit_hat": ph, "gbm_dA_rank": ra, "gbm_hit_rank": rh})


def shuffled_keys(p, which):
    c = ctx(); Kx = c["K"].copy(); cols = [IX["hsin"], IX["hcos"]] if which == "hour" else [IX["wkend"]]
    Kx[:, cols] = c["K"][p][:, cols]; return Kx


def view_for(module, **kw):
    """kw: control, seed, pool_src, starts"""
    c = ctx(); ctrl = kw.get("control")
    if ctrl in ("H", "W"):
        p = T.shuffle_within_month(c["ta"], c["okr"] if module == "M7c" else c["ok"], kw["seed"])
        if module == "M9h6":
            return view_m9h6(hour=c["hour"][p]) if ctrl == "H" else view_m9h6(wk=c["wk"][p])
        return view_gbm("M7c" if module == "M7c" else "M7", Kmat=shuffled_keys(p, "hour" if ctrl == "H" else "wk"))
    if module == "M9h6":
        return view_m9h6()
    return view_gbm("M7c" if module == "M7c" else "M7", pool_src=kw.get("pool_src"), starts=kw.get("starts"))


# ------------------------------------------------------------------ harness calls
def compact(r):
    a, b = r["A"], r["B"]; okm = [m for m in a["per_month"] if not m["skipped"]]
    return {"A": {"months": a["months"], "inc": a["increment_series"]["view"], "c1": a["increment_series"]["c1"], "mean": a["increment"]["view"]["mean"], "ci": a["increment"]["view"]["ci"],
                  "n_tr": [m["n_tr"] for m in okm], "flags_pass": a["flags"]["pass"]},
            "B": {"months": b["months"], "rel": b["monthly_rel"]["view"], "c1": b["monthly_rel"]["c1"], "overall": b["rel_improvement"]["view"], "flags_pass": b["flags"]["pass"]},
            "counts": {k: r["counts"][k] for k in ("view_rows_finite", "valid_both")}}


def run_look(frames, lk, shift=0, pool=None):
    """One look through evaluate.evaluate_look. `pool`: None (harness default) or 'cut' / 'fwd' / 'both' (concatenated base, train rows restricted)."""
    cfg = AN.LOOKS[lk]; src = cfg["source"]; months = AN.look_months(lk); start = cfg["train_from"]
    if shift:
        start = start + pd.DateOffset(months=shift)
    try:
        if pool is None:
            r = E.evaluate_look(base(src), frames[src], months, start, None, 5, 10000, 0)
        else:
            assert lk == "look2"
            bc, bf = base("cut"), base("fwd"); arr = {k: np.concatenate([bc.arr[k], bf.arr[k]]) for k in bc.arr}; ta = bc.ta.append(bf.ta); B = AN.Base(ta, arr)
            srcarr = np.array(["cut"] * len(bc.ta) + ["fwd"] * len(bf.ta)); mask = (srcarr == pool) if pool != "both" else np.ones(len(ta), bool)
            fr = pd.concat([frames["cut"], frames["fwd"]]); orig = E.train_mask
            E.train_mask = lambda ta_, valid, m0, start=None, H=60, days=E.TRAIN_DAYS: orig(ta_, valid & mask, m0, start, H, days)
            try:
                st = cfg["train_from"] if pool == "fwd" else pd.Timestamp("2023-07-01")
                r = E.evaluate_look(B, fr, months, st, None, 5, 10000, 0)
            finally:
                E.train_mask = orig
        r.pop("_B_raw", None)
        return compact(r)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def run_view(frames, looks=("look1", "look2"), **kw):
    return {lk: run_look(frames, lk, **kw) for lk in looks}


def save(name, obj):
    os.makedirs(RAW, exist_ok=True); T.dump(obj, os.path.join(RAW, name + ".json"))


def have(name):
    return os.path.exists(os.path.join(RAW, name + ".json"))


# ------------------------------------------------------------------ jobs
STORED = {"M9h6": "M9h6_prof_str", "M7c": "M7c_gbm_resid", "M7_gbm": "M7_gbm"}


def job_repro():
    out = {}
    for m in MODULES:
        res = run_view(view_for(m)); st = T.stored(STORED[m]); d = {}
        for lk in ("look1", "look2"):
            sa = np.array(st["looks"][lk]["A"]["increment_series"]["view"]); sb = np.array(st["looks"][lk]["B"]["monthly_rel"]["view"])
            ra = np.array(res[lk]["A"]["inc"]); rb = np.array(res[lk]["B"]["rel"])
            d[lk] = {"A_max_abs_diff": float(np.max(np.abs(sa - ra))) if len(sa) == len(ra) else "months differ", "B_max_abs_diff": float(np.max(np.abs(sb - rb))) if len(sb) == len(rb) else "months differ"}
        out[m] = {"diff_vs_stored": d, "run": res}
    # the Binance-only pool on the concatenated base must equal the stored Look 2 result of M9h6
    r = run_look(view_for("M9h6"), "look2", pool="fwd"); st = T.stored("M9h6_prof_str")["looks"]["look2"]
    out["pool_fwd_concat_vs_stored_M9h6_look2"] = {"A_max_abs_diff": float(np.max(np.abs(np.array(r["A"]["inc"]) - np.array(st["A"]["increment_series"]["view"])))),
                                                   "B_max_abs_diff": float(np.max(np.abs(np.array(r["B"]["rel"]) - np.array(st["B"]["monthly_rel"]["view"]))))}
    save("repro", out)


def job_shuffle(module, ctrl):
    seeds = H_SEEDS if ctrl == "H" else W_SEEDS; runs = {}
    for s in seeds:
        runs[str(s)] = run_view(view_for(module, control=ctrl, seed=s))
    save(f"{module}__C{ctrl}", {"seeds": seeds, "runs": runs})


def job_pool(module):
    out = {}
    for pool in ("cut", "fwd", "both"):
        if module == "M9h6":
            out[pool] = {"look2": run_look(view_for(module), "look2", pool=pool)}
        else:
            fr = view_for(module, pool_src=None if pool == "both" else pool)
            out[pool] = {"look2": run_look(fr, "look2")}
    save(f"{module}__CP", out)


def job_start(module):
    out = {}
    for s in SHIFTS:
        starts = {"look1": AN.LOOKS["look1"]["train_from"] + pd.DateOffset(months=s), "look2": AN.LOOKS["look2"]["train_from"] + pd.DateOffset(months=s)}
        fr = view_for(module, starts=starts)
        out[str(s)] = {lk: run_look(fr, lk, shift=s) for lk in ("look1", "look2")}
    save(f"{module}__CS", out)


# ------------------------------------------------------------------ analysis
def ser(run, tg):
    """{month: value} of a compact run for one look dict."""
    if run is None or "error" in run:
        return {}
    a = run[tg]; return dict(zip(a["months"], a["inc"] if tg == "A" else a["rel"]))


def pooled_series(runs, tg):
    d = {}
    for lk in ("look1", "look2"):
        if lk in runs:
            d.update(ser(runs[lk], tg))
    return d


def tci_dict(d):
    return T.tci(list(d.values()))


def analyse():
    R = {}
    for m in MODULES:
        o = {}
        real = T.load_json(os.path.join(RAW, "repro.json"))[m]["run"]
        o["real"] = {tg: {"pooled": tci_dict(pooled_series(real, tg)), "look1": tci_dict(ser(real["look1"], tg)), "look2": tci_dict(ser(real["look2"], tg))} for tg in "AB"}
        for ctrl in ("H", "W"):
            sh = T.load_json(os.path.join(RAW, f"{m}__C{ctrl}.json")); oc = {}
            for tg in "AB":
                per_seed = [pooled_series(r, tg) for r in sh["runs"].values()]
                months = sorted(set(pooled_series(real, tg)) & set.intersection(*[set(p) for p in per_seed]))
                rv = np.array([pooled_series(real, tg)[mo] for mo in months]); cv = np.mean([[p[mo] for mo in months] for p in per_seed], axis=0)
                diff = rv - cv; pooled_means = [float(np.mean([p[mo] for mo in months])) for p in per_seed]
                oc[tg] = {"months": len(months), "real_mean": float(rv.mean()), "control_mean_series": T.tci(cv), "paired_real_minus_control": T.tci(diff),
                          "survives": bool(T.tci(diff)["lo"] is not None and T.tci(diff)["lo"] > 0), "share_kept": float(cv.mean() / rv.mean()) if rv.mean() != 0 else None,
                          "per_seed_pooled_mean": pooled_means, "per_look": {lk: {"real": T.tci(np.array([ser(real[lk], tg)[mo] for mo in months if mo in ser(real[lk], tg)])),
                                                                                   "control": T.tci(np.array([np.mean([ser(r[lk], tg)[mo] for r in sh["runs"].values()]) for mo in months if mo in ser(real[lk], tg)]))} for lk in ("look1", "look2")}}
            o[ctrl] = oc
        cp = T.load_json(os.path.join(RAW, f"{m}__CP.json")); o["P"] = {}
        default = ser(cp["both"]["look2"], "A"), ser(cp["both"]["look2"], "B")
        for pool, rr in cp.items():
            oc = {}
            for tg in "AB":
                s = ser(rr["look2"], tg); t = T.tci(list(s.values())) if s else {"n": 0, "mean": None, "lo": None, "hi": None}
                dflt = default[0 if tg == "A" else 1]; common = [mo for mo in s if mo in dflt]
                pd_ = T.tci([s[mo] - dflt[mo] for mo in common]) if len(common) >= 2 else None
                oc[tg] = {"increment": t, "months": list(s), "testable": bool(t["n"] >= 5), "survives": bool(t["n"] >= 5 and t["lo"] is not None and t["lo"] > 0),
                          "paired_variant_minus_both": pd_}
            o["P"][pool] = oc
        cs = T.load_json(os.path.join(RAW, f"{m}__CS.json")); o["S"] = {}
        for s_, rr in cs.items():
            oc = {}
            for tg in "AB":
                s = pooled_series(rr, tg); t = T.tci(list(s.values())) if s else {"n": 0, "mean": None, "lo": None, "hi": None}
                oc[tg] = {"increment": t, "months": list(s), "testable": bool(t["n"] >= 5), "survives": bool(t["n"] >= 5 and t["lo"] is not None and t["lo"] > 0),
                          "look1": tci_dict(ser(rr["look1"], tg)), "look2": tci_dict(ser(rr["look2"], tg))}
            o["S"][s_] = oc
        o["passes"] = {}
        for tg in "AB":
            checks = {"C-H": o["H"][tg]["survives"], "C-W": o["W"][tg]["survives"]}
            for pool, oc in o["P"].items():
                if oc[tg]["testable"]:
                    checks[f"C-P {pool}"] = oc[tg]["survives"]
            for s_, oc in o["S"].items():
                if oc[tg]["testable"]:
                    checks[f"C-S +{s_}"] = oc[tg]["survives"]
            o["passes"][tg] = {"checks": checks, "pass": bool(all(checks.values()))}
        o["passes"]["module"] = bool(o["passes"]["A"]["pass"] or o["passes"]["B"]["pass"])
        R[m] = o
    T.dump(R, os.path.join(OUT, "results_day2.json"))
    f = {"A": lambda x: "n/a" if x is None else f"{x:+.4f}", "B": lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"}
    ci = lambda tg, d: "n/a" if d is None or d["lo"] is None else f"{f[tg](d['mean'])} [{f[tg](d['lo'])}, {f[tg](d['hi'])}] n={d['n']}"
    L = ["T15 Day 2: falsification and controls (rules: NOTE_day2.md)", ""]
    for m, o in R.items():
        L += [f"##### {m}", f"real pooled A {ci('A', o['real']['A']['pooled'])};  B {ci('B', o['real']['B']['pooled'])}"]
        for ctrl, nm in (("H", "hour shuffle (10 seeds)"), ("W", "weekend shuffle (10 seeds)")):
            for tg in "AB":
                c = o[ctrl][tg]
                L.append(f"  {nm} target {tg}: real {f[tg](c['real_mean'])}; shuffled (seed-mean series) {ci(tg, c['control_mean_series'])}; keeps {c['share_kept']:.2f} of the gain; "
                         f"paired real-shuffled {ci(tg, c['paired_real_minus_control'])}; survives {c['survives']}")
        for pool, oc in o["P"].items():
            for tg in "AB":
                c = oc[tg]
                L.append(f"  pool {pool:4s} (Look 2) target {tg}: {ci(tg, c['increment'])}; testable {c['testable']}; survives {c['survives']}; variant-minus-both {ci(tg, c['paired_variant_minus_both'])}")
        for s_, oc in o["S"].items():
            for tg in "AB":
                c = oc[tg]
                L.append(f"  start +{s_:>2s} months target {tg}: pooled {ci(tg, c['increment'])}; Look1 {ci(tg, c['look1'])}; Look2 {ci(tg, c['look2'])}; testable {c['testable']}; survives {c['survives']}")
        for tg in "AB":
            L.append(f"  PASS target {tg}: {o['passes'][tg]['pass']}  {o['passes'][tg]['checks']}")
        L += [f"  MODULE passes Day 2: {o['passes']['module']}", ""]
    open(os.path.join(OUT, "tables_day2.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    T.below_normal(); os.makedirs(RAW, exist_ok=True); T.wall_hours()
    if len(sys.argv) > 1 and sys.argv[1] == "analyse":
        analyse(); return
    jobs = [("reproduction checks (3 modules + concatenated fwd pool)", "repro", job_repro)]
    for m in MODULES:
        jobs += [(f"{m} hour shuffle x10", f"{m}__CH", lambda m=m: job_shuffle(m, "H")), (f"{m} weekend shuffle x10", f"{m}__CW", lambda m=m: job_shuffle(m, "W")),
                 (f"{m} pool sensitivity", f"{m}__CP", lambda m=m: job_pool(m)), (f"{m} start-date sensitivity", f"{m}__CS", lambda m=m: job_start(m))]
    for desc, name, fn in jobs:
        fname = name.replace("__CH", "__CH").replace("__CW", "__CW")
        if have(fname):
            continue
        T.run_job(DAY, desc, fn, f"day2/raw/{fname}.json")
    T.run_job(DAY, "analyse: survival rules", analyse, "day2/results_day2.json, day2/tables_day2.txt")


if __name__ == "__main__":
    main()
