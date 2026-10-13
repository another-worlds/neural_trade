"""Hill-climb v2 library (SPEC.md): monthly walk-forward predictions, minute-path tables, the trade-sequence evaluator, the climb.

Phases: 'dev' scores the DEV months only; 'look' scores the LOCKBOX months and exists only for run_lockbox.py (flag `look=True`).
A lockbox month is never predicted, pathed or scored in the 'dev' phase (months.py, tests in test_hc2.py).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
for d in ("hc2", "common", "recent", "equity", "sizing", "stops", "fwd", "newnet"):
    sys.path.insert(0, os.path.join(TAC, d))
import months as MO          # noqa: E402
import cutoff as C           # noqa: E402  (no bar after C.CUTOFF is read; the loaders below cut again at DATA_END)
import wf                    # noqa: E402
import fwdlib as F           # noqa: E402
import sizelib as S          # noqa: E402
import stoplib as L          # noqa: E402

FEAT_CACHE = "D:/nt/nt_recut_cache/fwd"          # tb7, rich, t, open, close of the hourly bars < 2024-07-01 (fwd study)
CACHE = "D:/nt/nt_hc2_cache"
DATA_END = "2024-07-01"
HOUR = 3600 * 10 ** 9
COST = 2.0
MIN_TRADES = 500
BRAKE_DD, BRAKE_F = S.BRAKE_DD, S.BRAKE_F
EXITS = {"time": "X0_time", "tp1.5": "X2_sym15", "tp2": "X3_sym2"}
EXIT_ORDER = ("X0_time", "X2_sym15", "X3_sym2")
HS = (1, 2, 4)
LEVERS = (("feat", ("tb7", "rich", "rich+ctx")), ("model", ("logreg", "l1", "hgb")), ("T", (365, 730)), ("H", HS),
          ("q", (0.05, 0.10, 0.20)), ("side", ("follow", "agree")), ("exit", ("time", "tp1.5", "tp2")), ("size", ("fixed", "brake")))
BASELINE = {"feat": "rich", "model": "logreg", "T": 730, "H": 4, "q": 0.10, "side": "follow", "exit": "time", "size": "fixed"}
N_CONFIGS = int(np.prod([len(v) for _, v in LEVERS]))


def ckey(cfg):
    return "|".join(str(cfg[k]) for k, _ in LEVERS)


def all_configs():
    import itertools
    names = [k for k, _ in LEVERS]
    return [dict(zip(names, v)) for v in itertools.product(*[vals for _, vals in LEVERS])]


def pkey(cfg, H=None):
    return (cfg["feat"], cfg["model"], cfg["T"], cfg["H"] if H is None else H)


# ------------------------------------------------------------------ inputs
class Inputs:
    """Hourly anchors of the fwd cache (signal bar open time t[i]; entry = the next hour's open) plus the per-bar context block."""
    def __init__(self):
        c = FEAT_CACHE
        self.t = np.load(f"{c}/t.npy"); self.anch = np.load(f"{c}/anch.npy").astype(np.int64)
        self.op = np.load(f"{c}/open.npy"); self.cl = np.load(f"{c}/close.npy")
        assert pd.Timestamp(self.t.max()) < pd.Timestamp(DATA_END) and len(self.op) < 70000
        self.N = len(self.t)
        self.X = {"tb7": np.load(f"{c}/tb7.npy"), "rich": np.load(f"{c}/rich.npy")}
        self.X["rich+ctx"] = np.concatenate([self.X["rich"], self.ctx()], 1)
        self.r = {H: (self.cl[self.anch + H - 1] / self.op[self.anch] - 1.0) for H in HS}
        ti = pd.DatetimeIndex(self.t.astype("datetime64[ns]"))
        self.mon = np.asarray(ti.to_period("M").astype(str))
        self.mon_idx = {m: np.flatnonzero(self.mon == m) for m in pd.unique(self.mon)}

    def ctx(self):
        path = f"{CACHE}/ctx.npy"
        if os.path.exists(path):
            return np.load(path)
        import data as ND
        s = F.load_bars()                                   # hourly OHLCV bars < 2024-07-01
        A = s.values.astype(np.float64); assert len(A) == len(self.op)
        tod = (s.index.hour * 60 + s.index.minute).to_numpy(np.float64); dow = s.index.dayofweek.to_numpy(np.float64)
        ctx = np.nan_to_num(ND.context_features(A, tod, dow)[self.anch - 1]).astype(np.float32)
        os.makedirs(CACHE, exist_ok=True); np.save(path, ctx)
        return ctx


# ------------------------------------------------------------------ predictions
def predict_set(inp, feat, model, T, H, months, look=False):
    """float32 array over the anchors: P(up) from the monthly walk-forward (wf._fit as frozen) in `months`, NaN elsewhere.
    Refuses a lockbox month unless look=True and a non-lockbox month when look=True."""
    for m in months:
        if look != (m in MO.LOCKBOX) and m != MO.WARMUP:
            raise ValueError(f"month {m} is {'a lockbox' if m in MO.LOCKBOX else 'not a lockbox'} month; look={look}")
    from sklearn.preprocessing import StandardScaler
    X, t, r = inp.X[feat], inp.t, inp.r[H]
    ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); mask = np.abs(r) > F.deadband(H); y = (r > 0).astype(int)
    out = np.full(inp.N, np.nan, np.float32)
    for m in months:
        m0, m1 = MO.month_bounds(m)
        tr, te = wf.split_masks(ti, pd.Timestamp(m0), pd.Timestamp(m1), H, F.BAR_MIN, T); trm = tr & mask; tem = te & mask
        if trm.sum() < wf.MIN_TRAIN or tem.sum() < wf.MIN_TEST or len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        Xtr = np.asarray(X[trm], np.float32); sc = StandardScaler().fit(Xtr)
        g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
        mdl, _ = wf._fit(model, g(Xtr), y[trm], gap_rows=H + 1)
        idx = np.flatnonzero(te); out[idx] = mdl.predict_proba(g(X[idx]))[:, 1]
    return out


def pred_path(phase, key):
    return f"{CACHE}/pred_{phase}_{key[0]}_{key[1]}_{key[2]}_{key[3]}.npy"


# ------------------------------------------------------------------ minute-path tables
def build_paths(inp, mins, st_sigma, H, months):
    """{rule: gross/exit/kind arrays [N,2]} for the anchors of `months` (NaN gross elsewhere); the reach test uses THIS horizon."""
    mset = set(months); sel = np.zeros(inp.N, bool)
    for m in months:
        idx = inp.mon_idx.get(m)
        if idx is None:
            continue
        _, m1 = MO.month_bounds(m); nxt = str(pd.Period(m, "M") + 1)
        sel[idx] = (inp.t[idx] + (H + 1) * HOUR <= m1) | (nxt in mset)
    sel &= (inp.t + (H + 1) * HOUR <= mins["t"][-1] + 60 * 10 ** 9)               # the path must lie inside the data
    ii = np.flatnonzero(sel)
    entry = inp.t[ii] + HOUR; exit_ = inp.t[ii] + (H + 1) * HOUR
    bar = inp.anch[ii] - 1
    sig_trade = st_sigma[bar] * np.sqrt(H)
    ok = np.isfinite(sig_trade); ii, entry, exit_, sig_trade = ii[ok], entry[ok], exit_[ok], sig_trade[ok]
    pre = L.precompute(mins, entry, exit_, sig_trade, rules=EXIT_ORDER)
    out = {r: {"gross": np.full((inp.N, 2), np.nan), "exit": np.zeros((inp.N, 2), np.int64), "kind": np.zeros((inp.N, 2), np.int8)} for r in EXIT_ORDER}
    for r in EXIT_ORDER:
        for k in ("gross", "exit", "kind"):
            out[r][k][ii] = pre[r][k]
    return out


def paths_file(phase, H):
    return f"{CACHE}/paths_{phase}_H{H}.npz"


def save_paths(phase, H, p):
    np.savez(paths_file(phase, H), **{f"{r}_{k}": p[r][k] for r in EXIT_ORDER for k in ("gross", "exit", "kind")})


def load_paths(phase, H):
    z = np.load(paths_file(phase, H))
    return {r: {k: z[f"{r}_{k}"] for k in ("gross", "exit", "kind")} for r in EXIT_ORDER}


# ------------------------------------------------------------------ evaluation
def scan(entry, exit_):
    """Skip-while-open over signals sorted by entry; entry == previous exit is taken. Returns the kept indices."""
    keep = []; free = -(1 << 62)
    for i, (e, x) in enumerate(zip(entry.tolist(), exit_.tolist())):
        if e >= free:
            keep.append(i); free = x
    return np.asarray(keep, np.int64)


def weights(net, brake):
    """Per-trade weights: 1, or 0.5 while the running additive-equity drawdown is >= BRAKE_DD (closed trades only; until a new peak)."""
    if not brake:
        return np.ones(len(net))
    w = np.empty(len(net)); eq = 1.0; peak = 1.0; on = False
    for i, x in enumerate(net.tolist()):
        w[i] = BRAKE_F if on else 1.0
        eq += w[i] * x
        if eq > peak:
            peak = eq; on = False
        elif (peak - eq) / peak >= BRAKE_DD:
            on = True
    return w


def max_dd_pct(pnl):
    if len(pnl) == 0:
        return 0.0
    eq = 1.0 + np.cumsum(pnl); peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(((peak - eq) / peak).max() * 100)


class Store:
    """The data one phase may touch: anchors, predictions, paths, the scored months. phase 'dev' (default) or 'look' (lockbox months)."""
    def __init__(self, inp, phase="dev", look=False):
        assert phase in ("dev", "look") and (phase == "look") == look
        self.inp, self.phase, self.look = inp, phase, look
        self.months = MO.LOCKBOX if look else MO.DEV
        self.excluded = set() if look else set(MO.LOCKBOX)         # months whose predictions the thresholds may not use
        self.mpos = np.full(inp.N, -1, np.int64)
        for k, m in enumerate(self.months):
            if m in inp.mon_idx:
                self.mpos[inp.mon_idx[m]] = k
        self.days = float(sum(pd.Period(m, "M").days_in_month for m in self.months))
        self.years = self.days / 365.25
        self._pred, self._paths, self._sig = {}, {}, {}

    def preds(self, key):
        if key not in self._pred:
            a = np.load(pred_path("dev", key))
            if self.look:
                a = np.where(np.isnan(a), np.load(pred_path("look", key)), a)
            else:
                lb = np.concatenate([self.inp.mon_idx[m] for m in MO.LOCKBOX if m in self.inp.mon_idx])
                assert np.isnan(a[lb]).all(), "a dev prediction store holds a lockbox month"
            self._pred[key] = a
        return self._pred[key]

    def paths(self, H):
        if H not in self._paths:
            self._paths[H] = load_paths(self.phase, H)
        return self._paths[H]

    def signals(self, key, q):
        """(idx, p) of the signals of the scored months: |P - 0.5| >= the (1 - q) quantile of the reference month's |P - 0.5|."""
        k = (key, q)
        if k not in self._sig:
            p = self.preds(key); conf = np.abs(p - 0.5); idxs = []
            for m in self.months:
                mi = self.inp.mon_idx.get(m)
                if mi is None:
                    continue
                ref = MO.ref_month(m, self.excluded); ri = self.inp.mon_idx.get(ref)
                if ri is None or np.isnan(conf[ri]).all():
                    continue
                thr = np.quantile(conf[ri][~np.isnan(conf[ri])], 1 - q)
                idxs.append(mi[conf[mi] >= thr])               # NaN compares False
            idx = np.concatenate(idxs) if idxs else np.zeros(0, np.int64)
            self._sig[k] = (idx, p[idx])
        return self._sig[k]


class Evaluator:
    """Evaluates configurations on a Store. flips: None or an array of +-1 per scored month (the null world: side * sign of the signal's month)."""
    def __init__(self, store, flips=None):
        self.s, self.flips, self.memo, self.seqmemo, self.n_eval = store, flips, {}, {}, 0

    def sequence(self, cfg, sides_override=None):
        """The taken trades of cfg without the size rule: dict idx, side, gross, exit, kind, entry (arrays), month position."""
        sk = tuple(cfg[k] for k in ("feat", "model", "T", "H", "q", "side", "exit"))
        if sides_override is None and sk in self.seqmemo:
            return self.seqmemo[sk]
        s = self.s; inp = s.inp; H = cfg["H"]
        idx, p = s.signals(pkey(cfg), cfg["q"])
        side = np.where(p > 0.5, 1, -1)
        if cfg["side"] == "agree":
            ok = np.ones(len(idx), bool)
            for h in HS:
                if h == H:
                    continue
                ph = s.preds(pkey(cfg, h))[idx]
                ok &= (np.where(ph > 0.5, 1, -1) == side) & ~np.isnan(ph)
            idx, side = idx[ok], side[ok]
        if sides_override is not None:
            side = sides_override(idx, side)
        elif self.flips is not None:
            side = side * self.flips[s.mpos[idx]]
        pt = s.paths(H)[EXITS[cfg["exit"]]]; si = (side < 0).astype(int)
        g = pt["gross"][idx, si]; valid = np.isfinite(g)
        idx, side, si, g = idx[valid], side[valid], si[valid], g[valid]
        ex = pt["exit"][idx, si]; kd = pt["kind"][idx, si]; entry = inp.t[idx] + HOUR
        k = scan(entry, ex)
        out = {"idx": idx[k], "side": side[k], "gross": g[k], "exit": ex[k], "kind": kd[k], "entry": entry[k], "mpos": s.mpos[idx[k]]}
        if sides_override is None:
            self.seqmemo[sk] = out
        return out

    def metrics(self, seq, size):
        net = seq["gross"] - COST * 1e-4
        w = weights(net, size == "brake"); pnl = w * net
        nm = len(self.s.months); n = len(pnl)
        mbps = np.bincount(seq["mpos"], weights=pnl * 1e4, minlength=nm) if n else np.zeros(nm)
        cum = float(pnl.sum() * 100); dd = max_dd_pct(pnl)
        years = self.s.years
        calmar = cum / years / max(dd, 1e-9) if n else float("-inf")
        return {"n": int(n), "cum": cum, "dd": dd, "calmar": calmar if n >= MIN_TRADES else float("-inf"), "calmar_raw": calmar, "hit": float((net > 0).mean()) if n else 0.0,
                "mean_bps": float(net.mean() * 1e4) if n else 0.0, "mbps": mbps, "pnl": pnl, "w": w, "valid": n >= MIN_TRADES}

    def eval(self, cfg):
        k = ckey(cfg)
        if k not in self.memo:
            self.n_eval += 1
            self.memo[k] = self.metrics(self.sequence(cfg), cfg["size"])
        return self.memo[k]


# ------------------------------------------------------------------ the climb
def paired_ok(cand, inc):
    return float(np.mean(cand["mbps"] - inc["mbps"])) > 0.0


def climb(ev, start, rng, log=None):
    """Coordinate ascent (SPEC): visit the levers in a random order; per lever take, among the other values that raise the Calmar AND have a positive
    paired monthly net-bps difference, the best Calmar; stop after a pass without an accepted step. Returns (incumbent, result, trajectory)."""
    inc = dict(start); r = ev.eval(inc)
    traj = [{"step": 0, "cfg": ckey(inc), "calmar": r["calmar"], "cum": r["cum"], "dd": r["dd"], "n": r["n"], "lever": None}]
    step = 0
    while True:
        moved = False
        for li in rng.permutation(len(LEVERS)):
            name, vals = LEVERS[li]
            best = None
            for v in vals:
                if v == inc[name]:
                    continue
                c = dict(inc); c[name] = v; rc = ev.eval(c)
                if rc["calmar"] > r["calmar"] and paired_ok(rc, r) and (best is None or rc["calmar"] > best[1]["calmar"]):
                    best = (c, rc)
            if best:
                step += 1; moved = True
                traj.append({"step": step, "cfg": ckey(best[0]), "calmar": best[1]["calmar"], "cum": best[1]["cum"], "dd": best[1]["dd"],
                             "n": best[1]["n"], "lever": name, "from": inc[name], "to": best[0][name]})
                inc, r = best
        if not moved:
            return inc, r, traj


def random_start(i, seed=MO.SEED):
    rng = np.random.default_rng(seed + i)
    return {k: vals[int(rng.integers(len(vals)))] for k, vals in LEVERS}


def run_climbs(ev, n_random=9, seed=MO.SEED, tag=""):
    """The baseline start plus n_random random starts. Returns (best (cfg, result), list of per-start dicts)."""
    starts = [dict(BASELINE)] + [random_start(i + 1, seed) for i in range(n_random)]
    out = []
    for j, s0 in enumerate(starts):
        rng = np.random.default_rng(seed * 7 + j)
        inc, r, traj = climb(ev, s0, rng)
        out.append({"start": ckey(s0), "final": ckey(inc), "calmar": r["calmar"], "cum": r["cum"], "dd": r["dd"], "n": r["n"], "traj": traj, "cfg": inc})
    best = max(out, key=lambda d: (d["calmar"], d["cum"]))
    return best, out


# ------------------------------------------------------------------ reporting helpers
def full_stats(ev, cfg, seq=None):
    """Descriptive numbers of one configuration on the evaluator's months."""
    seq = seq or ev.sequence(cfg); m = ev.metrics(seq, cfg["size"]); pnl = m["pnl"]; net = seq["gross"] - COST * 1e-4
    mp = seq["mpos"]; nm = len(ev.s.months)
    hit_m, dd_m, ntr = [], [], []
    for k in range(nm):
        sel = mp == k
        if sel.any():
            hit_m.append(float((net[sel] > 0).mean())); dd_m.append(max_dd_pct(pnl[sel])); ntr.append(int(sel.sum()))
    kd = seq["kind"]
    unit = net                                                                # weight-1 pnl for the leverage frontier
    fr = L.frontier(unit) if len(unit) else None
    return {"n_trades": m["n"], "hit": m["hit"], "mean_net_bps": m["mean_bps"], "cum_pct": m["cum"], "max_dd_pct": m["dd"], "calmar": m["calmar_raw"],
            "years": ev.s.years, "months_with_trades": len(hit_m), "months_hit_gt_60": int(sum(h > 0.6 for h in hit_m)),
            "months_dd_lt_5": int(sum(d < 5.0 for d in dd_m)), "monthly_net_bps": [float(x) for x in m["mbps"]],
            "share_stop": float((kd == L.KIND_STOP).mean()) if len(kd) else 0.0, "share_tp": float((kd == L.KIND_TP).mean()) if len(kd) else 0.0,
            "share_time": float((kd == L.KIND_TIME).mean()) if len(kd) else 0.0, "avg_weight": float(m["w"].mean()) if m["n"] else 0.0,
            "frontier": fr}


def random_side_null(ev, cfg, draws=500, seed=0):
    """Fair coin per signal for the side, the whole sequence re-simulated (non-overlap, exits, brake). Returns cum% / hit / dd arrays."""
    rng = np.random.default_rng(seed)
    idx0, p = ev.s.signals(pkey(cfg), cfg["q"])
    cums, dds, hits = [], [], []
    for _ in range(draws):
        coin = rng.random(len(idx0)) < 0.5

        def so(idx, side, coin=coin, idx0=idx0):
            pos = np.searchsorted(idx0, idx)
            return np.where(coin[pos], 1, -1)
        seq = ev.sequence(cfg, sides_override=so); m = ev.metrics(seq, cfg["size"])
        cums.append(m["cum"]); dds.append(m["dd"]); hits.append(m["hit"])
    return np.array(cums), np.array(dds), np.array(hits)
