"""Resumable monthly runner. usage: python run_omori.py simA|simB|sel|look [--workers 4]
sel: real 2019-01..2024-06 (T = 15, 30, 60); look: real 2024-07..2025-07, T = 30 only, refused unless results/selected.json says so;
simA / simB: simulated paths, scored months 2020-01..2021-12 (the SPEC section 5 control)."""
import os, sys, json, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import omorilib as O

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
PHASES = {"simA": ("simA", "2020-01", "2021-12", (30,)), "simB": ("simB", "2020-01", "2021-12", (30,)),
          "sel": ("real", "2019-01", "2024-06", O.TS), "look": ("real", "2024-07", "2025-07", (30,))}
_G = {}
AMP, THR = 4.0, 2.0


def simulate(kind, seed=7, months=36, start="2019-01-01"):
    """SPEC section 5: GARCH-like clock; kind B adds sigma_t = sigma_clock (1 + 6 sum_j 1{|eps_j|>3.5} (t-j+1)^-0.9), j within 4,320 bars."""
    n = int(pd.date_range(start, periods=months + 1, freq="MS")[-1].value - pd.Timestamp(start).value) // 60_000_000_000
    rng = np.random.default_rng(seed + (0 if kind == "A" else 1000)); eps = rng.normal(size=n)
    al, be = 0.04, 0.95; g = 1.0; H = np.zeros(n + O.D_MAX + 2); w = AMP * (np.arange(1, O.D_MAX + 1) + 1.0) ** (-0.9)
    sig0 = 1e-3; r = np.empty(n); sg = np.empty(n); lcs = np.zeros(n + 1); sclock = np.empty(n); last_trig = -10 ** 9; TT = 30
    for t in range(n):
        sclock[t] = sig0 * math_sqrt(g); s = sclock[t] * (1.0 + H[t]); sg[t] = s; r[t] = s * eps[t]; lcs[t + 1] = lcs[t] + r[t]
        g = (1 - al - be) + al * eps[t] ** 2 * g + be * g
        # amended trigger (SPEC amendment 1): onset of a T = 30 endpoint excursion beyond 3.85 sigma_clock(t-30) sqrt(30), 60-bar refractory
        if kind == "B" and t >= TT and t - last_trig > 60 and abs(lcs[t + 1] - lcs[t + 1 - TT]) > THR * sclock[t - TT] * math_sqrt(TT):
            H[t + 1:t + 1 + O.D_MAX] += w; last_trig = t
    lc = np.log(30000.0) + np.cumsum(r); c = np.exp(lc); o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 0.4, n)) * sg
    h = np.maximum(o, c) * (1 + spread); l = np.minimum(o, c) * (1 - spread)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(1, 10, n)},
                        index=pd.date_range(start, periods=n, freq="1min"))


def math_sqrt(x):
    return x ** 0.5


def _init(tag):
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if tag.startswith("sim"):
        _G["B"] = O.Bars(simulate(tag[-1])); _G["K"] = {T: O.calibrate_k(_G["B"], "2020-01", T) for T in O.TS}
    else:
        _G["B"] = O.Bars(); _G["K"] = O.K_T
    assert _G["B"].df.index.max() <= O.CUTOFF


def _job(arg):
    m, T = arg; t0 = time.time(); r = O.score_month(_G["B"], m, T, _G["K"][T]); return m, T, r, time.time() - t0


if __name__ == "__main__":
    phase = sys.argv[1]; w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    tag, first, last, Ts = PHASES[phase]
    if phase == "look":
        sel = json.load(open(f"{RES}/selected.json"))
        if not sel.get("selected"):
            sys.exit("look refused: the selection prediction did not hold")
        if os.path.exists(f"{RES}/look.jsonl"):
            sys.exit("look refused: the one look already ran")
    path = f"{RES}/{phase}.jsonl"; done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            j = json.loads(l); done.add((j["month"], j["T"]))
    jobs = [(str(m), T) for m in pd.period_range(first, last, freq="M") for T in Ts if (str(m), T) not in done]
    print(f"{phase}: {len(jobs)} jobs, {len(done)} done", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(w, initializer=_init, initargs=(tag,)) as pool, open(path, "a", encoding="utf-8") as fh:
        for m, T, r, sec in pool.imap_unordered(_job, jobs):
            if r is None:
                print(f"  {m} T={T}: skipped", flush=True); continue
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(f"  {m} T={T} {sec:.0f}s", flush=True)
    print(f"{phase} done in {time.time() - t0:.0f}s", flush=True)
