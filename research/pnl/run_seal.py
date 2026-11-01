"""Sealed test (SPEC 'selection_and_preregistration' ENFORCEMENT and SEAL VERDICTS). Runs ONCE.

Refuses unless: the working tree is clean; HEAD contains research/pnl/prereg.json and PNL_UNSEAL equals its SHA-256;
$PNL_WORK/seal_lock.json does not exist (it is written first); every pre-registered file hash matches.
Then: loads the full data (seal=True), recomputes features on it and asserts every dev row is bit-identical to the
cached dev features; scores every candidate, baseline and neural_trade in all six scenarios on the full seal and the
clean seal with the frozen fold-S models; computes the verdicts of the deployed candidates; writes
results/seal.json and prints only at the end.

  PNL_UNSEAL=<sha256 of prereg.json> python run_seal.py
  python run_seal.py --dry-run     (pipeline test on dev 2025-01-01..2025-10-31 with the fold-S models, which are
                                    in-sample there: the numbers mean nothing; never touches the seal)
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import data as D
import features as FE
import folds as FO
import net as NET
import prep
import rules
import sim
import stats as ST
from policy import BandPolicy, HysteresisPolicy, C_RT

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
PREREG = os.path.join(HERE, "prereg.json")
SCEN = [("C7", "B"), ("C7", "S"), ("C4", "B"), ("C4", "S"), ("C10", "B"), ("C10", "S")]
CLEAN_EXCLUDE = [(FO.SEAL_START, FO.ts(2025, 11, 10) + int(7.5 * 3600)), (FO.ts(2026, 9, 20), FO.SEAL_END + 86400)]
TF_PY = os.environ.get("NT_PYTHON", "")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def guard(prereg):
    if git("status", "--porcelain").stdout.strip():
        sys.exit("refused: the working tree is not clean")
    blob = subprocess.run(["git", "-C", REPO, "show", "HEAD:research/pnl/prereg.json"], capture_output=True)
    if blob.returncode != 0:
        sys.exit("refused: HEAD does not contain research/pnl/prereg.json")
    h = hashlib.sha256(blob.stdout).hexdigest()
    if os.environ.get("PNL_UNSEAL", "") != h or sha(PREREG) != h:
        sys.exit("refused: PNL_UNSEAL must equal the SHA-256 of the committed prereg.json")
    lock = os.path.join(prep.WORK, "seal_lock.json")
    if os.path.exists(lock):
        sys.exit("refused: seal_lock.json exists (the seal was already run)")
    bad = [p for p, hh in prereg["hashes"].items() if not os.path.exists(p) or sha(p) != hh]
    if bad:
        sys.exit(f"refused: hash mismatch for {bad}")
    json.dump({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "head": git("rev-parse", "HEAD").stdout.strip(),
               "prereg_sha256": h}, open(lock, "w"))


class FrozenScaler(prep.Scaler):
    def __init__(self, med, iqr, cols):
        self.med, self.iqr, self.cols = np.asarray(med), np.asarray(iqr), np.asarray(cols)


def ridge_policy(st, track, s, e):
    z = np.load(os.path.join(prep.WORK, f"ridge_T{track}_S.npz"))
    sc = FrozenScaler(z["med"], z["iqr"], z["cols"])
    H = int(z["H"])
    rows = np.arange(s, e)
    if track == 2:
        rows = rows[prep.decision_mask(st.b.t[s:e], 5)]
    mu = np.full(st.n, np.nan)
    if not bool(z["cash"]):
        for i in range(0, len(rows), 250_000):
            r = rows[i:i + 250_000]
            Z = np.c_[sc(st.X, r).astype(np.float64), np.ones(len(r))]
            mu[r] = (Z @ z["w"]) * st.sigma_H(r, H) * 1e4
    else:
        mu[rows] = 0.0
    dec = ~np.isnan(mu)
    K, ch = (1, False) if track == 1 else (15, True)
    return lambda: HysteresisPolicy(mu, dec, H, float(z["theta"]) * C_RT, K=K, chase=ch)


def net_policy(st, track, sel, s, e):
    n = st.n
    tau, band, dec = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    h = int(sel["h"])
    if sel["cash"]:
        dec[np.arange(s, e)[prep.decision_mask(st.b.t[s:e], 60)]] = True
    else:
        tr = 1 if h <= 15 else 2
        bundle = torch.load(os.path.join(prep.WORK, "nets", f"T{tr}_S_h{h}_{sel['cell']}.pt"), weights_only=False)
        sc = FrozenScaler(bundle["scaler"]["med"], bundle["scaler"]["iqr"], bundle["scaler"]["cols"])
        rows = np.arange(s, e)[prep.decision_mask(st.b.t[s:e], h)]
        Xr = np.concatenate([sc(st.X, rows[i:i + 250_000]) for i in range(0, len(rows), 250_000)])
        taus, bs = [], []
        for blob in bundle["states"]:
            import io
            m = NET.PnLNet(len(sc.cols), bundle["arch"])
            m.load_state_dict(torch.load(io.BytesIO(blob), weights_only=False))
            t_, b_, _ = NET.infer(m, torch.from_numpy(Xr))
            taus.append(t_)
            bs.append(b_)
        tau[rows], band[rows], dec[rows] = np.mean(taus, 0), np.mean(bs, 0), True
    K = (1 if h <= 5 else 3) if track == 1 else 15
    return lambda: BandPolicy(tau, band, dec, K=K, chase=track == 2)


def nt_policy(st, prereg, s, e, dry):
    """T1-NT: frozen fold-S neural_trade bundle, mu at every seal minute (TF venv subprocess), hysteresis theta = 1."""
    info = prereg.get("t1_nt")
    if not info or not TF_PY:
        return None
    out = os.path.join(prep.WORK, "nt", "seal_mu.npz" if not dry else "dryrun_mu.npz")
    env = dict(os.environ)
    cmd = [TF_PY, os.path.join(HERE, "nt_adapter.py"), "predict", info["bundle"], str(int(st.b.t[s])),
           str(int(st.b.t[e - 1]) + 60), out] + (["--dev"] if dry else ["--seal"])
    subprocess.run(cmd, check=True, env=env)
    z = np.load(out)
    mu = np.full(st.n, np.nan)
    idx = np.searchsorted(st.b.t, z["t"])
    j = {1: 0, 5: 1, 15: 2}[int(info["h"])]
    mu[idx] = 0.0 if info.get("cash") else z["mu"][:, j]
    dec = ~np.isnan(mu)
    H = int(info["h"])
    return lambda: HysteresisPolicy(mu, dec, H, C_RT, K=1 if H <= 5 else 3, chase=False)


def summarise(d, mask=None):
    x = d if mask is None else d[mask]
    sd = np.std(x)
    return {"days": int(len(x)), "mean_daily_bps": float(np.mean(x) * 1e4), "total_log_bps": float(np.sum(x) * 1e4),
            "sharpe": float(np.mean(x) / sd * np.sqrt(365)) if sd > 0 else 0.0}


def verdict_t1(d, clean, halves):
    dist = ST.stationary_bootstrap(d, B=10_000, mean_block=10, seed=21)
    p = ST.boot_pvalue(dist)
    m_full, m_clean = float(np.mean(d)), float(np.mean(d[clean]))
    h = [float(np.mean(d[m])) for m in halves]
    if m_full <= 0 or m_clean <= 0:
        v = "KILL"
    elif p < 0.05 and m_clean > 0 and all(x > 0 for x in h):
        v = "PASS"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "boot_p": p, "mean_full_bps": m_full * 1e4, "mean_clean_bps": m_clean * 1e4,
            "halves_bps": [x * 1e4 for x in h]}


def verdict_t2(d, clean, sr_dev, t_dev_years, dev_daily):
    def sr(x):
        return float(np.mean(x) / np.std(x) * np.sqrt(365)) if np.std(x) > 0 else 0.0
    out = {}
    fals = False
    for name, x in (("full", d), ("clean", d[clean])):
        T = len(x) / 365.0
        bound = sr_dev - 1.645 * np.sqrt(1 / T + 1 / t_dev_years)
        out[name] = {"sharpe": sr(x), "bound": float(bound), "years": T}
        fals |= sr(x) < bound
    pnl = float(np.sum(d))
    v = "FALSIFIED" if fals else ("CONSISTENT" if pnl > 0 else "WEAK")
    pooled = np.r_[dev_daily, d]
    lb = ST.lower_bound(ST.stationary_bootstrap(pooled, B=10_000, mean_block=20, seed=22))
    out.update({"verdict": v, "seal_net_log_bps": pnl * 1e4, "confirmed": bool(v == "CONSISTENT" and lb > 0),
                "pooled_lb95_daily_bps": lb * 1e4})
    return out


def main(dry=False):
    prereg = json.load(open(PREREG)) if os.path.exists(PREREG) else None
    if not dry:
        if prereg is None:
            sys.exit("refused: no prereg.json")
        guard(prereg)
        b = D.load(1, seal=True)
        s0, e0 = FO.SEAL_START, b.t[-1] + 60
    else:
        prereg = prereg or json.load(open(os.path.join(prep.WORK, "prereg_draft.json")))
        b = D.load(1)
        s0, e0 = FO.ts(2025, 1, 1), FO.SEAL_START
    X, names, sig = FE.build(b)
    cached = np.load(os.path.join(prep.WORK, "features_v1.npy"), mmap_mode="r")
    n_dev = len(cached)
    ident = bool(np.array_equal(X[:n_dev], cached))
    if not ident:
        sys.exit("K0: recomputed dev features are not bit-identical to the cached ones")
    st = prep.Store(bars=b, X=X, names=names, sigma=sig.astype(np.float32).astype(np.float64))
    s, e = st.idx(s0), st.idx(e0)
    S = rules.build_signals(b, st.sigma)
    facs = {}
    for track in (1, 2):
        for k, f in rules.policies(S, track).items():
            facs.setdefault(k, f)
        facs[f"T{track}-LIN"] = ridge_policy(st, track, s, e)
        facs[f"T{track}-NET"] = net_policy(st, track, prereg["net_selection_S"][str(track)], s, e)
    nt = nt_policy(st, prereg, s, e, dry)
    if nt is not None:
        facs["T1-NT"] = nt
    for c, mx in prereg.get("static_mean_x", {}).items():
        facs[f"static[{c}]"] = (lambda mx=mx: rules.StaticPolicy(mx, S["dec"]))
    tsd = None
    res = {"window": [int(s0), int(e0)], "dry_run": dry, "features_bit_identical": ident, "strategies": {}}
    daily = {}
    for name, fac in facs.items():
        res["strategies"][name] = {}
        for fee, fill in SCEN:
            r = sim.simulate(b, fac(), s, e, fee=fee, fill=fill)
            daily[f"{name}|{fee}{fill}"] = r["daily"]
            tsd = r["days"] * 86400
            clean = np.ones(len(tsd), bool)
            for a, z in CLEAN_EXCLUDE:
                clean &= ~((tsd + 86400 > a) & (tsd < z))
            res["strategies"][name][fee + fill] = {"full": summarise(r["daily"]), "clean": summarise(r["daily"], clean),
                                                   "trades": len(r["trades"]), "fees": r["fees"], "funding": r["funding"]}
    mid = tsd[0] + (tsd[-1] - tsd[0]) // 2
    halves = [tsd <= mid, tsd > mid]
    clean = np.ones(len(tsd), bool)
    for a, z in CLEAN_EXCLUDE:
        clean &= ~((tsd + 86400 > a) & (tsd < z))
    dev = dict(np.load(os.path.join(prep.WORK, "dev_daily.npz")))
    res["verdicts"] = {}
    for track in (1, 2):
        tp = prereg["tracks"][str(track)]
        for role in ("deployed", "information_only"):
            c = tp.get(role)
            if not c or c == "CASH" or f"{c}|C7B" not in daily:
                continue
            d = daily[f"{c}|C7B"]
            if track == 1:
                v = verdict_t1(d, clean, halves)
            else:
                dd = dev[f"{c}|C7B"]
                sr_dev = float(np.mean(dd) / np.std(dd) * np.sqrt(365))
                v = verdict_t2(d, clean, sr_dev, len(dd) / 365.0, dd)
            v["role"] = role
            res["verdicts"][f"T{track}:{c}"] = v
    path = os.path.join(prep.WORK, "seal_dryrun.json") if dry else os.path.join(RESULTS, "seal.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    np.savez(path.replace(".json", "_daily.npz"), **daily)
    print(json.dumps(res["verdicts"], indent=1, default=float))
    for name, v in res["strategies"].items():
        print(f"{name:22s} C7B full {v['C7B']['full']['mean_daily_bps']:8.2f} bps/d  clean "
              f"{v['C7B']['clean']['mean_daily_bps']:8.2f}  C7S {v['C7S']['full']['mean_daily_bps']:8.2f}")
    return res


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
