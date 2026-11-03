"""New-data backtest, step 2: engine H (hourly) and scoring of the frozen daily trend rules on data the study never scored.

  python nd_run.py validate   BTC only: engine H against the scored engine-E results (dev 2021-01..2025-10 from
                              results/dev_runs.json, sealed year from results/seal_daily.npz). Reads no new market.
                              -> results/newdata_validation.json
  python nd_run.py score      The pre-registered test (research/pnl/prereg_newdata.md). Refuses to run unless that
                              file is committed at HEAD unchanged; runs once (work/nd/score_lock.json).
                              -> results/newdata.json, results/newdata_daily.npz

ENGINE H (one market, hourly bars; the daily rules decide only at the 23:00 bar close = the 00:00 UTC daily close)
- State: signed contracts q, equity E, exposure x = q c / E. Equity is renormalised to 1 at each UTC day start (exact:
  every PnL term is linear in size), so a day's net return is E_end - 1 in units of start-of-day equity.
- Bar j: funding first (a print stamped in (t_{j-1}, t_j] is paid at bar j's open at c_{j-1}; longs pay positive
  rates), then the price move q (c_j - c_{j-1}). A print at bar j-1's close is paid by the position held BEFORE a trade
  decided at that close, as in engine E, where a decision fills from the next bar on (the old position pays the 00:00
  print); prints inside a data hole are paid by the position held across the hole.
- Trades at bar closes at c_j. The INCREASE leg of |x| costs c_in, the REDUCE leg c_out (fractions of traded notional),
  split exactly as engine F (a flip = full reduce + new increase).
- Rules (rules.py, SPEC 'track2'): T2-P band 0.2; tsmom_unscaled trades when tau changes; voltarget_long band 0.2;
  B&H x = 1 from its first bar; cash.
- 1x cap as engine E: trim to |x| = 1 (reduce, c_out) when |x| > 1.10 at any bar close, or > 1.02 at a decision bar
  where the rule does not trade; as in E, the trim acts first and the rule decides on the trimmed exposure. Engine E
  checks every minute, engine H every hour: H misses wicks inside the hour. That favours a short on a spike that
  reverts inside the hour and costs it more on a squeeze that persists; the net direction is not known.
- A day whose loss exceeds the day's equity (possible only if one hour moves > ~100% against a 1x position, or a fee
  exceeds what is left) is recorded as -100% and the position is closed; every rule re-enters at its next decision
  (tsmom_unscaled at its next decision, not only at the next change of tau; B&H at the next bar).
- Days without bars inside a segment (archive holes): no decision, no return; the position carries and the move is
  booked on the next bar.
- Positions are closed (c_out) at the last bar of the run or of a data segment. Bars with zero quote volume are
  dropped (the archive keeps flat zero-volume bars after a delisting), so a market ends at its last traded hour; a
  gap > 7 days splits a segment, and a new segment needs 56 days of warm-up again.
- Signals are causal: tau and sigma_ann for day d use daily closes up to and including day d (rules.tsmom_daily on the
  forward-filled calendar of the segment); the first decision is on day 56 of a segment.
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from numba import njit

import metrics
import rules
import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ND = os.environ.get("ND_DATA", os.path.join(HERE, "work", "nd"))
RES = os.path.join(HERE, "results")
DAY, HOUR = 86400, 3600
SEG_GAP = 7 * DAY
STRATS = ("T2-P", "tsmom_unscaled", "voltarget_long", "B&H", "cash")
MODE = {"T2-P": 0, "tsmom_unscaled": 1, "voltarget_long": 0, "B&H": 2, "cash": 3}


def ts(s):
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def day_str(d):
    return dt.datetime.fromtimestamp(int(d) * DAY, dt.timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------------------------------- engine H

@njit(cache=True)
def engine_h(c, fund_old, fund_new, dayid, dec, tgt, mode, band, c_in, c_out, s, e, nday):
    """Bars s..e-1 of one segment. dayid[j] in 0..nday-1. Returns per-day arrays (units of start-of-day equity):
    ret, long price PnL, short price PnL, funding paid, fees, turnover (|dx| x equity), end-of-day x, ruin flag.
    fund_old[j]: prints at bar j-1's close (t_{j-1} + 1h), paid by the position held at the end of bar j-1 BEFORE a
    trade decided at that close (engine E: the order fills from the next bar on). fund_new[j]: later prints that fall
    in a data hole before bar j, paid by the position held across the hole (after that trade)."""
    ret = np.zeros(nday)
    lp = np.zeros(nday)
    sp = np.zeros(nday)
    fu = np.zeros(nday)
    fe = np.zeros(nday)
    tu = np.zeros(nday)
    xe = np.zeros(nday)
    ru = np.zeros(nday)
    q, E = 0.0, 1.0
    pend = 0.0                                   # fund_old due at the next bar's open, fixed before this bar's trade
    last = np.nan
    started = False
    cur = dayid[s]
    for j in range(s, e):
        d = dayid[j]
        if d != cur:
            ret[cur] = E - 1.0 if E > 0.0 else -1.0
            for dd in range(cur + 1, d):         # days without bars: the position carries, no return is booked
                xe[dd] = xe[cur]
            if E > 0.0:
                q = q / E
                pend = pend / E
            else:
                q = 0.0
                pend = 0.0
            E = 1.0
            cur = d
        if j > s and E > 0.0:
            pay = pend + fund_new[j] * q * c[j - 1]
            if pay != 0.0:
                E -= pay
                fu[d] += pay
            pnl = q * (c[j] - c[j - 1])
            E += pnl
            if q > 0.0:
                lp[d] += pnl
            elif q < 0.0:
                sp[d] += pnl
            if E <= 0.0:                         # ruin inside the day: the position is gone
                E = 0.0
                q = 0.0
                ru[d] = 1.0
                last = np.nan
                started = False
        x = q * c[j] / E if E > 0.0 else 0.0
        pend = fund_old[j + 1] * q * c[j] if (j + 1 < e and E > 0.0) else 0.0
        # engine E order: the drift trim (|x| > 1.10) acts first, then the rule decides on the trimmed exposure
        xd = (1.0 if x > 0 else -1.0) if abs(x) > 1.10 else x
        target = np.nan
        if E > 0.0:
            if j == e - 1:
                target = 0.0                     # close at the end of the run / segment
            elif mode == 2 and not started:
                target = 1.0
                started = True
            elif (mode == 0 or mode == 1) and dec[j]:
                tg = tgt[j]
                if tg == tg:
                    if mode == 0:
                        if abs(tg - xd) >= band - 1e-12:
                            target = tg
                    else:
                        if not (last == last and tg == last):
                            last = tg
                            if abs(tg - xd) >= 1e-9:
                                target = tg
            if target != target:
                if abs(x) > 1.10 or (dec[j] and (mode == 0 or mode == 1) and abs(xd) > 1.02):
                    target = 1.0 if x > 0 else -1.0
        if target == target:
            t = min(1.0, max(-1.0, target))
            dx = abs(t - x)
            inc = (abs(t) + dx - abs(x)) * 0.5
            red = (abs(x) + dx - abs(t)) * 0.5
            cost = (c_in * inc + c_out * red) * E
            tu[d] += dx * E
            E -= cost
            fe[d] += cost
            if E <= 0.0:                         # the trade's fee used up what was left of the equity
                E = 0.0
                q = 0.0
                x = 0.0
                ru[d] = 1.0
                last = np.nan
                started = False
            else:
                q = t * E / c[j]
                x = t
        xe[d] = x
    ret[cur] = E - 1.0 if E > 0.0 else -1.0
    return ret, lp, sp, fu, fe, tu, xe, ru


# --------------------------------------------------------------------------------------------------- one market

class Market:
    """Hourly bars of one symbol, split into segments at gaps > 7 days, with funding per bar and daily signals."""

    def __init__(self, path):
        z = np.load(path)
        t, c = z["t"].astype(np.int64), z["close"].astype(np.float64)
        # zero-volume bars are dropped: the archive keeps flat zero-volume bars after a delisting (and during
        # suspensions), so a market ends at its last traded hour, and a stale stretch > 7 days splits a segment
        ok = np.isfinite(c) & (c > 0) & (z["quote_volume"] > 0)
        self.t, self.c = t[ok], c[ok]
        self.qv = z["quote_volume"][ok]
        ft, fr = z["funding_t"].astype(np.int64), z["funding_rate"].astype(np.float64)
        # a print maps to the first bar that opens at or after it; it is 'old' (paid before the previous bar's trade)
        # when it falls exactly at the previous bar's close, and 'new' when it falls inside a data hole
        self.fund_old = np.zeros(len(self.t))
        self.fund_new = np.zeros(len(self.t))
        idx = np.searchsorted(self.t, ft, side="left")
        m = idx < len(self.t)
        idx, ft, fr = idx[m], ft[m], fr[m]
        prev_close = np.where(idx > 0, self.t[np.maximum(idx - 1, 0)] + HOUR, -1)
        old = (idx == 0) | (ft <= prev_close)
        np.add.at(self.fund_old, idx[old], fr[old])
        np.add.at(self.fund_new, idx[~old], fr[~old])
        br = np.flatnonzero(np.diff(self.t) > SEG_GAP) + 1
        self.segments = list(zip(np.r_[0, br], np.r_[br, len(self.t)]))

    def segment_signals(self, a, b):
        """Per-bar day id (from the segment's first day), decision flags and per-bar targets of each rule."""
        t = self.t[a:b]
        day = t // DAY
        d0 = day[0]
        nday = int(day[-1] - d0 + 1)
        last = np.full(nday, -1, np.int64)
        last[day - d0] = np.arange(len(t))
        C = np.full(nday, np.nan)
        okd = last >= 0
        C[okd] = np.log(self.c[a:b][last[okd]])
        C = _ffill(C)
        tau, s_ann, valid = rules.tsmom_daily(C)
        scale = np.minimum(1.0, 0.5 / s_ann)
        by_day = {"T2-P": np.where(valid, np.round(tau * scale, 1), np.nan),
                  "tsmom_unscaled": np.where(valid, tau, np.nan),
                  "voltarget_long": np.where(valid, np.round(scale, 1), np.nan)}
        dec = (t % DAY) == 23 * HOUR
        di = (day - d0).astype(np.int64)
        tg = {k: np.where(dec, v[di], np.nan) for k, v in by_day.items()}
        return d0, nday, di, dec, tg, by_day

    def run(self, w0, w1, c_in, c_out, strats=STRATS):
        """Simulate every rule over UTC days [w0, w1] (day numbers). Returns {strategy: {day: (ret, lp, sp, fu, fe,
        tu, xe, ru)}} merged over segments, the per-day long/short exposure, and the active day range per segment."""
        out = {k: {} for k in strats}
        spans = []
        for a, b in self.segments:
            if b - a < 2 or self.t[b - 1] // DAY - self.t[a] // DAY + 1 <= 56:
                continue                             # too short for a first decision (segment day 56)
            d0, nday, di, dec, tg, _ = self.segment_signals(a, b)
            first = max(w0, d0 + 56)                 # first decision day (tau valid from segment day 56)
            lastd = min(w1, d0 + nday - 1)
            if first > lastd:
                continue
            day_abs = di + d0
            s = int(np.searchsorted(day_abs, first, side="left"))
            e = int(np.searchsorted(day_abs, lastd, side="right"))
            if e - s < 2:
                continue
            spans.append((int(first), int(lastd)))
            for k in strats:
                arr = tg.get(k, np.full(b - a, np.nan))
                r = engine_h(self.c[a:b], self.fund_old[a:b], self.fund_new[a:b], di, dec, arr, MODE[k], 0.2,
                             c_in, c_out, s, e, nday)
                for dd in range(first - d0, lastd - d0 + 1):
                    out[k][int(dd + d0)] = tuple(float(v[dd]) for v in r)
        return out, spans


def _ffill(C):
    out = C.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def series(res, days, field=0):
    """Dense per-day array over `days` (np.nan where the market is not active)."""
    return np.array([res[d][field] if d in res else np.nan for d in days])


def summary(r):
    """Daily log-return series -> headline numbers (bps / fractions)."""
    r = np.asarray(r, np.float64)
    if len(r) < 2:
        return {"days": int(len(r)), "total_log": float(r.sum())}
    out = {"days": int(len(r)), "total_log": float(r.sum()), "ann_additive_pct": float(365 * r.mean() * 100),
           "ann_compound_pct": float(np.expm1(365 * r.mean()) * 100), "vol_ann": float(r.std(ddof=1) * np.sqrt(365)),
           "sharpe": float(stats.sharpe_ann(r)), "max_dd": float(stats.max_drawdown(r)),
           "hac_t": float(stats.newey_west_t(r, lag=20))}
    return out


def by_year(r, days):
    yr = np.array([int(day_str(d)[:4]) for d in days])
    return {int(y): float(r[yr == y].sum()) for y in np.unique(yr)}


# --------------------------------------------------------------------------------------------------- validation

def validate():
    """Engine H on BTCUSDT perp against the scored engine-E numbers (C7-B: maker 2 bps in, taker 5 bps out)."""
    m = Market(os.path.join(ND, "hourly", "BTCUSDT.npz"))
    dev = json.load(open(os.path.join(RES, "dev_runs.json")))
    sd = np.load(os.path.join(RES, "seal_daily.npz"))
    out = {}
    for label, w0, w1 in (("dev", "2021-01-01", "2025-10-31"), ("seal", "2025-11-01", "2026-10-30")):
        res, _ = m.run(ts(w0) // DAY, ts(w1) // DAY, 2e-4, 5e-4, strats=("T2-P", "tsmom_unscaled", "voltarget_long",
                                                                            "B&H"))
        days = list(range(ts(w0) // DAY, ts(w1) // DAY + 1))
        out[label] = {}
        for k in ("T2-P", "tsmom_unscaled", "voltarget_long", "B&H"):
            h = np.log1p(np.nan_to_num(series(res[k], days)))
            row = {"H_total_log": float(h.sum()), "H_ann_additive_pct": float(365 * h.mean() * 100),
                   "H_sharpe": float(stats.sharpe_ann(h))}
            if label == "dev":
                sc = dev[k]["scenarios"]["C7B"]
                row.update({"E_ann_additive_pct": sc["ann_additive_bps"] / 100, "E_days": sc["days"],
                            "E_sharpe": sc.get("sharpe")})
                row["gap_pct_per_year"] = row["H_ann_additive_pct"] - row["E_ann_additive_pct"]
            else:
                ev = sd[f"{k}|C7B"]
                row.update({"E_total_log": float(ev.sum()), "E_days": int(len(ev)), "E_sharpe": float(stats.sharpe_ann(ev))})
                row["gap_total_log"] = row["H_total_log"] - row["E_total_log"]
            out[label][k] = row
    json.dump(out, open(os.path.join(RES, "newdata_validation.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------------------------------- scoring

PREREG = os.path.join(HERE, "prereg_newdata.md")
LOCK = os.path.join(HERE, "work", "nd", "score_lock.json")      # fixed path: ND_DATA cannot move it
LOCK_COPY = os.path.join(RES, "newdata_lock.json")              # committed with the results
MANIFEST = os.path.join(RES, "newdata_manifest.json")            # data hashes committed with the pre-registration
W0, W1 = "2021-01-01", "2026-10-31"
DEV_END, SEAL0 = "2025-10-31", "2025-11-01"
COSTS = {"primary": (5e-4, 10e-4), "btc_level": (2e-4, 5e-4), "stress": (25e-4, 25e-4)}
INDEX_CONTRACTS = ("DEFIUSDT",)
LIQ_FROM, LIQ_TO, LIQ_N = "2020-07-01", "2020-12-31", 20
LIQ20_PREREG = ["ETHUSDT", "LINKUSDT", "YFIUSDT", "LTCUSDT", "SUSHIUSDT", "XRPUSDT", "UNIUSDT", "BCHUSDT", "BNBUSDT",
                "YFIIUSDT", "ADAUSDT", "DOTUSDT", "EOSUSDT", "SXPUSDT", "TRXUSDT", "CRVUSDT", "XTZUSDT", "WAVESUSDT",
                "BANDUSDT", "XLMUSDT"]


def _git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def _guard():
    """Score only (1) from a clean research/pnl tree at a committed HEAD (code and pre-registration as committed),
    (2) on data whose hashes equal the manifest committed with the pre-registration, (3) once: a re-run needs
    ND_RESCORE=1 AND a 'newdata rescore' line in CHANGES.log that names the reason."""
    raw = open(PREREG, "rb").read()
    head = subprocess.run(["git", "-C", HERE, "show", "HEAD:research/pnl/prereg_newdata.md"], capture_output=True)
    if head.returncode != 0 or head.stdout != raw:
        sys.exit("prereg_newdata.md must be committed at HEAD and unchanged before scoring")
    dirty = _git("status", "--porcelain", "--", ".").stdout.strip()
    if dirty:
        sys.exit("research/pnl has uncommitted or untracked files; commit first:\n" + dirty)
    if not os.path.exists(MANIFEST):
        sys.exit("results/newdata_manifest.json missing")
    want = json.load(open(MANIFEST))
    for rel, h in want.items():
        got = hashlib.sha256(open(os.path.join(ND, rel), "rb").read()).hexdigest()
        if got != h:
            sys.exit(f"data file {rel} differs from the committed manifest")
    # the committed copy counts as well as the ignored lock, so a fresh clone or a deleted lock is not a first run
    hist = [json.load(open(p)) for p in (LOCK, LOCK_COPY) if os.path.exists(p)]
    prev = max(hist, key=lambda h: len(h["runs"])) if hist else {"runs": []}
    rescore = len(prev["runs"]) > 0
    if rescore:
        n_entries = sum("newdata rescore" in line for line in open(os.path.join(HERE, "CHANGES.log")))
        n_rescores = sum(r.get("rescore", False) for r in prev["runs"])
        if os.environ.get("ND_RESCORE") != "1" or n_entries <= n_rescores:
            sys.exit(f"already scored ({len(prev['runs'])} run(s)); a re-run needs ND_RESCORE=1 and a NEW "
                     "'newdata rescore' entry in CHANGES.log giving its reason (one entry per re-run)")
    sha = hashlib.sha256(raw).hexdigest()
    commit = _git("rev-parse", "HEAD").stdout.strip()
    prev["runs"].append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "prereg_sha256": sha,
                         "head": commit, "manifest_files": len(want), "rescore": rescore})
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    json.dump(prev, open(LOCK, "w"), indent=1)
    json.dump(prev, open(LOCK_COPY, "w"), indent=1)
    return sha, commit


def verdict(r, seed=0):
    r = np.asarray(r, np.float64)
    dist = stats.stationary_bootstrap(r, B=10000, mean_block=20, seed=seed)
    lb = stats.lower_bound(dist)
    m = float(r.mean())
    v = "REPLICATED" if lb > 0 else ("CONSISTENT" if m > 0 else "NOT REPLICATED")
    return {"verdict": v, "mean_daily_bps": m * 1e4, "lb95_daily_bps": lb * 1e4,
            "ci95_daily_bps": [x * 1e4 for x in stats.percentile_ci(dist)], "boot_p": stats.boot_pvalue(dist)}


def portfolio(mat, sel=None):
    """mat: coins x days of daily simple net returns (nan = inactive). Equal capital per active market, daily."""
    m = mat if sel is None else mat[sel]
    act = np.isfinite(m)
    n = act.sum(0)
    simple = np.where(n > 0, np.nansum(m, 0) / np.maximum(n, 1), 0.0)
    return np.log1p(simple), n


def liquid_set(panel):
    a, b = ts(LIQ_FROM) // DAY, ts(LIQ_TO) // DAY
    med = {}
    for s in panel:
        z = np.load(os.path.join(ND, "hourly", f"{s}.npz"))
        d = z["t"] // DAY
        k = (d >= a) & (d <= b)
        if k.sum() < 90 * 24:                      # pre-registered: at least 90 days of bars in 2020-07..12
            continue
        u, inv = np.unique(d[k], return_inverse=True)
        med[s] = float(np.median(np.bincount(inv, weights=z["quote_volume"][k])))
    return sorted(med, key=lambda s: -med[s])[:LIQ_N], med


def score():
    sha, commit = _guard()
    uni = json.load(open(os.path.join(ND, "universe.json")))
    panel = uni["panel"]
    d0, d1 = ts(W0) // DAY, ts(W1) // DAY
    days = np.arange(d0, d1 + 1)
    dev_k = days <= ts(DEV_END) // DAY
    out = {"prereg_sha256": sha, "head": commit, "window": [W0, W1], "panel_size": len(panel), "costs": COSTS}
    mats = {(k, c): np.full((len(panel), len(days)), np.nan) for k in STRATS for c in COSTS}
    comp = {f: np.full((len(panel), len(days)), np.nan) for f in ("lp", "sp", "fu", "fe", "tu", "ru", "xe", "wo")}
    per = {}
    alive_end = {}
    fund_missing_days = 0
    for i, s in enumerate(panel):
        m = Market(os.path.join(ND, "hourly", f"{s}.npz"))
        z = np.load(os.path.join(ND, "hourly", f"{s}.npz"))
        miss = set(str(v) for v in z["funding_missing"])
        alive_end[s] = bool(m.t[-1] >= ts("2026-10-30"))   # last TRADED bar (zero-volume bars dropped)
        for c, (ci, co) in COSTS.items():
            res, spans = m.run(d0, d1, ci, co)
            for k in STRATS:
                for dd, v in res[k].items():
                    mats[(k, c)][i, dd - d0] = v[0]
                if k == "T2-P" and c == "primary":
                    for dd, v in res[k].items():
                        for f, idx in (("lp", 1), ("sp", 2), ("fu", 3), ("fe", 4), ("tu", 5), ("xe", 6), ("ru", 7)):
                            comp[f][i, dd - d0] = v[idx]
                        # a ruin day floors ret at -1: the rest of the loss is a write-off, so the parts still add up
                        comp["wo"][i, dd - d0] = v[0] - (v[1] + v[2] - v[3] - v[4])
                        if day_str(dd)[:7] in miss:
                            fund_missing_days += 1
        # per-market statistics from SIMPLE daily returns, so a -100% (ruin) day is counted, not dropped as -inf
        sr = mats[("T2-P", "primary")][i]
        act_i = np.isfinite(sr)
        a = sr[act_i]
        bh = mats[("B&H", "primary")][i]
        bh = bh[np.isfinite(bh)]
        per[s] = {"days": int(act_i.sum()), "first": day_str(days[act_i][0]) if act_i.any() else None,
                  "last": day_str(days[act_i][-1]) if act_i.any() else None, "alive_at_end": alive_end[s],
                  "T2-P_total_compound": float(np.prod(1.0 + a) - 1.0),
                  "T2-P_total_log": float(np.log1p(a[a > -1]).sum()),    # over non-ruin days (see ruin_days)
                  "T2-P_sharpe_simple": float(stats.sharpe_ann(a)) if len(a) > 2 else None,
                  "B&H_total_compound": float(np.prod(1.0 + bh) - 1.0),
                  "long_pnl": float(np.nansum(comp["lp"][i])), "short_pnl": float(np.nansum(comp["sp"][i])),
                  "funding_paid": float(np.nansum(comp["fu"][i])), "fees": float(np.nansum(comp["fe"][i])),
                  "ruin_days": int(np.nansum(comp["ru"][i])), "ruin_writeoff": float(np.nansum(comp["wo"][i]))}
        print(s, per[s]["days"], round(per[s]["T2-P_total_compound"], 3), flush=True)
    out["funding_missing_coin_days_T2P"] = fund_missing_days
    out["markets_ever_active"] = int(sum(per[s]["days"] > 0 for s in panel))
    # primary
    prim, nact = portfolio(mats[("T2-P", "primary")])
    out["primary"] = {"strategy": "T2-P", "costs": "primary", **verdict(prim), **summary(prim),
                      "by_year": by_year(prim, days), "active_markets_min_median_max":
                          [int(nact.min()), int(np.median(nact)), int(nact.max())]}
    # descriptive portfolios
    desc = {}
    for k in STRATS:
        for c in COSTS:
            r, _ = portfolio(mats[(k, c)])
            desc[f"{k}|{c}"] = {**summary(r), "by_year": by_year(r, days)}
    out["portfolios"] = desc
    # T2-P decomposition (primary costs), in units of start-of-day equity, portfolio-weighted
    act = np.isfinite(comp["lp"])
    n = np.maximum(act.sum(0), 1)
    out["T2-P_decomposition_sum_simple"] = {f: float((np.nansum(comp[f], 0) / n).sum())
                                            for f in ("lp", "sp", "fu", "fe", "wo")}
    out["T2-P_mean_abs_exposure"] = float(np.nanmean(np.abs(comp["xe"])))
    out["T2-P_turnover_per_market_year"] = float(np.nansum(comp["tu"]) / max(act.sum(), 1) * 365)
    out["T2-P_ruin_days_total"] = int(np.nansum(comp["ru"]))
    # secondary verdicts on subsets (reported, not gates)
    sec = {}
    sec["dev_window"] = {**verdict(prim[dev_k]), **summary(prim[dev_k])}
    sec["sealed_window"] = {**verdict(prim[~dev_k]), **summary(prim[~dev_k])}
    liq, med = liquid_set(panel)
    sec["liquid20_matches_prereg"] = liq == LIQ20_PREREG
    sel = np.array([s in liq for s in panel])
    r, n = portfolio(mats[("T2-P", "primary")], sel)
    sec["liquid20"] = {"markets": liq, **verdict(r[n > 0]), **summary(r[n > 0])}
    sel = np.array([s not in INDEX_CONTRACTS for s in panel])
    r, n = portfolio(mats[("T2-P", "primary")], sel)
    sec["no_index_contracts"] = {**verdict(r[n > 0]), **summary(r[n > 0])}
    for lab, want in (("survivors", True), ("delisted", False)):
        sel = np.array([alive_end[s] == want and per[s]["days"] > 0 for s in panel])
        if sel.any():
            r, n = portfolio(mats[("T2-P", "primary")], sel)
            sec[lab] = {"markets": int(sel.sum()), "active_days": int((n > 0).sum()), **summary(r[n > 0])}
    tot = np.array([per[s]["T2-P_total_compound"] for s in panel if per[s]["days"] > 0])
    shp = np.array([per[s]["T2-P_sharpe_simple"] for s in panel if per[s]["T2-P_sharpe_simple"] is not None])
    sec["markets_positive_share"] = float((tot > 0).mean())
    sec["median_market_sharpe"] = float(np.median(shp))
    out["secondary"] = sec
    out["per_market"] = per
    # BTC: early spot, 2020 perp, after the seal; and BTC T2-P over the panel window for correlation
    btc = {}
    sp = Market(os.path.join(ND, "hourly", "BTCUSDT_spot.npz"))
    pp = Market(os.path.join(ND, "hourly", "BTCUSDT.npz"))
    for lab, mk, a, b in (("early_spot_2017_2019", sp, "2017-08-01", "2019-12-31"), ("perp_2020", pp, "2020-01-01", "2020-12-31")):
        res, spans = mk.run(ts(a) // DAY, ts(b) // DAY, 2e-4, 5e-4)
        dd = np.arange(spans[0][0], spans[-1][1] + 1)
        row = {"span": [day_str(spans[0][0]), day_str(spans[-1][1])]}
        for k in STRATS:
            r = np.log1p(np.nan_to_num(series(res[k], dd)))
            row[k] = {**summary(r), "by_year": by_year(r, dd)}
            if k == "T2-P":
                row["T2-P"].update(verdict(r))
        btc[lab] = row
    res, _ = pp.run(ts("2026-01-01") // DAY, ts("2026-11-02") // DAY, 2e-4, 5e-4)
    post = [d for d in sorted(res["T2-P"]) if d >= ts("2026-10-31") // DAY]
    seg = pp.segments[-1]
    sd0, _, _, _, _, by_day = pp.segment_signals(*seg)
    btc["after_seal"] = {"days": [day_str(d) for d in post],
                         **{k: {"daily_net_pct": [res[k][d][0] * 100 for d in post],
                                "exposure_held_over_day": [res[k][d - 1][6] for d in post],
                                "rule_target_at_close": [float(by_day[k][d - sd0]) if k in by_day else 1.0
                                                         for d in post]}
                            for k in ("T2-P", "tsmom_unscaled", "B&H")},
                         "note": "the run ends at the 2026-11-02 close, so that day's net includes the exit fee "
                                 "(taker 5 bps x |x|); the 11-02 target is the position the rule holds into 11-03. "
                                 "No funding archive for 2026-11 yet: funding 0 on 11-01 and 11-02."}
    res, _ = pp.run(d0, d1, 2e-4, 5e-4, strats=("T2-P",))
    b = np.log1p(np.nan_to_num(series(res["T2-P"], days)))
    btc["corr_panel_T2P_vs_BTC_T2P"] = float(np.corrcoef(prim, b)[0, 1])
    out["btc"] = btc
    json.dump(out, open(os.path.join(RES, "newdata.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(RES, "newdata_daily.npz"), days=days, btc_T2P=b, panel=np.array(panel),
                        **{f"portfolio|{k}|{c}": portfolio(mats[(k, c)])[0] for k in STRATS for c in COSTS})
    print(json.dumps({k: out[k] for k in ("primary",)}, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "validate":
        validate()
    elif cmd == "score":
        score()
    else:
        print(__doc__)
