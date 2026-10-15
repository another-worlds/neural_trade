"""O11 part d (../SPEC.md): direction accuracy of the entry that carries O9/O10. p_up = logistic regression (L2, C 0.1, standardised) for the sign of the
next-open 60-minute return at the 15-minute decision times, trained with a 5 bps dead band, monthly walk-forward. Variants: base (tb7, 365 days),
d1 (tb7 + G15, 365 days), d2 (tb7, 90 days), d3 (tb7 + G15, 90 days). Reuses O9's lib unchanged (decision frame, tb7_at, train_extras, p_up fit rules).
Everything at a decision uses bars < t; every fit uses rows whose labels end at or before the month start. CPU only."""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(OUTLOOK, "O9_ensemble_strategy"))
import o9_lib as L9  # noqa: E402
import o9_p2 as P2  # noqa: E402
L8 = L9.L8; P5 = L9.P5; O3 = L9.O3; T = L9.T
H = L9.H
SCRATCH = "D:/nt/nt_scratch/o11d"
NG = 164                                           # geometry features per scale: G = [G1 (164) | G15 (164)]
G15 = slice(NG, 2 * NG)
VARIANTS = {"base": (False, 365), "d1": (True, 365), "d2": (False, 90), "d3": (True, 90)}     # name -> (uses G15, training days)
CHOOSABLE = ("d1", "d2", "d3")
GATE = 0.02                                        # the strategy's gated subset: |p - 0.5| >= 0.02
MIN_GATED = 30                                     # a month's gated AUC needs at least this many gated rows
S_MONTHS = list(L8.S_SEL)                          # 2021-01 .. 2024-06
B_MONTHS = list(L8.SPANS["B"])                     # 2024-07 .. 2025-07 (to 07-20)
C_MONTHS = [str(m) for m in pd.period_range("2025-10", "2026-09", freq="M")]
C_END = pd.Timestamp("2026-10-02 00:00")           # one day past the last scored month, for the 60-minute labels (T01's FWD_LOAD_END)


# ------------------------------------------------------------------ training side
def train_data(TF, X):
    """Training dict: ta, tb7, G15 (stride-7 frame), ret60, pok (O9's mask: finite label outside the dead band, finite tb7, ok60), gok (G15 finite)."""
    g = np.asarray(TF["G"][:, G15], np.float32); assert len(g) == len(X["ta"])
    return {"ta": X["ta"], "tb7": X["tb7"], "G15": g, "ret60": X["ret60"], "pok": X["pok"], "gok": np.isfinite(g).all(1)}


def fit_variant(TX, var, m0):
    """Logistic fit of one variant for a test starting at m0: rows of the previous `days` days (365 or 90) whose label (the open 60 minutes later) ends at or
    before m0. Base is exactly O9's p_up."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    use_g, days = VARIANTS[var]
    ok = TX["pok"] & TX["gok"] if use_g else TX["pok"]
    tr = O3.train_rows(TX["ta"], ok, H, m0, days=days); assert len(tr) > 0
    Xf = np.concatenate([TX["tb7"][tr], TX["G15"][tr]], 1) if use_g else TX["tb7"][tr]
    y = (TX["ret60"][tr] > 0).astype(int); sc = StandardScaler().fit(Xf)
    lr = LogisticRegression(C=L9.C_LR, max_iter=500).fit(sc.transform(Xf), y)
    return {"sc": sc, "lr": lr, "use_g": use_g, "days": days, "n_tr": int(len(tr)), "first_ta": pd.Timestamp(TX["ta"][tr].min()),
            "last_label_end": pd.Timestamp(TX["ta"][tr].max()) + pd.Timedelta(minutes=H), "base": float(y.mean())}


def predict_variant(st, tb7, g15):
    Xf = np.concatenate([tb7, g15], 1) if st["use_g"] else tb7
    return st["lr"].predict_proba(st["sc"].transform(Xf))[:, 1]


# ------------------------------------------------------------------ decision side
def dec_ret60(s, a):
    """open[a+60] / open[a] - 1 at bar indices a (nan across a data gap, as train_extras)."""
    A = s.values.astype(np.float64); a = np.asarray(a); n = len(A)
    tm = s.index.values.astype("datetime64[m]").astype(np.int64)
    ok = a + H < n; a_ = np.where(ok, a, 0); b = np.minimum(a_ + H, n - 1)
    cont = ok & ((tm[b] - tm[a_]) == H)
    return np.where(cont, A[b, 0] / A[a_, 0] - 1, np.nan)


def decision_data(s, DF, DX):
    """Decision dict: ta, tb7, G15, ret60, and the scoring row mask `valid` = strategy-valid decision (DF ok), finite tb7 and G15, a finite non-zero label.
    No dead band in scoring. The same rows score every variant."""
    g = np.asarray(DF["G"][:, G15], np.float32); ret = dec_ret60(s, DF["a"])
    valid = DF["ok"].astype(bool) & np.isfinite(DX["tb7"]).all(1) & np.isfinite(g).all(1) & np.isfinite(ret) & (ret != 0)
    return {"ta": DF["ta"], "tb7": DX["tb7"], "G15": g, "ret60": ret, "valid": valid}


def month_idx(ta, mo):
    m0, m1 = P5.month_bounds(mo); return L8.month_rows(ta, m0, m1)


def predict_months(TX, DD, months, variants=tuple(VARIANTS), frozen_m0=None, log=None):
    """{variant: p over all decision rows (nan outside the months' valid rows)}. Monthly walk-forward, or (frozen_m0) one fit at that time for all months."""
    n = len(DD["ta"]); out = {v: np.full(n, np.nan) for v in variants}
    groups = [(frozen_m0, list(months))] if frozen_m0 is not None else [(P5.month_bounds(mo)[0], [mo]) for mo in months]
    for m0, ms in groups:
        rows = np.concatenate([month_idx(DD["ta"], mo) for mo in ms]); rows = rows[DD["valid"][rows]]
        if len(rows) == 0:
            continue
        for v in variants:
            st = fit_variant(TX, v, m0); assert st["last_label_end"] <= m0
            out[v][rows] = predict_variant(st, DD["tb7"][rows], DD["G15"][rows])
        if log:
            log("fit", ms[0] if len(ms) == 1 else f"{ms[0]}..{ms[-1]}", "rows", len(rows))
    return out


# ------------------------------------------------------------------ metrics
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def month_metrics(p, ret, rows):
    """AUC (all rows), Brier, and the gated subset |p - 0.5| >= GATE: n, AUC (nan below MIN_GATED rows or one class), hit rate. y = ret > 0."""
    pp = p[rows]; y = (ret[rows] > 0).astype(int); g = np.abs(pp - 0.5) >= GATE
    ng = int(g.sum())
    return {"n": int(len(rows)), "base_rate": float(y.mean()), "auc": auc(y, pp), "brier": float(np.mean((pp - y) ** 2)),
            "hit_all": float(((pp > 0.5) == (y == 1)).mean()), "n_gated": ng,
            "auc_gated": auc(y[g], pp[g]) if ng >= MIN_GATED else float("nan"),
            "hit_gated": float(((pp[g] > 0.5) == (y[g] == 1)).mean()) if ng else float("nan")}


def table(P, DD, months):
    """{variant: {month: metrics}} for the valid rows of each month."""
    out = {v: {} for v in P}
    for mo in months:
        rows = month_idx(DD["ta"], mo); rows = rows[DD["valid"][rows] & np.isfinite(next(iter(P.values()))[rows])]
        if len(rows) == 0:
            continue
        for v in P:
            out[v][mo] = month_metrics(P[v], DD["ret60"], rows)
    return out


def mean_t(x):
    """mean, 95% t-interval, t, number of months, months above zero (of the series as given) and the raw two-sided p of mean = 0."""
    from scipy import stats
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan")}
    mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    t = float(mu / se) if se > 0 else 0.0
    return {"n": int(n), "mean": float(mu), "lo": float(mu - h), "hi": float(mu + h), "t": t, "pos": int((x > 0).sum()), "p": float(2 * stats.t.sf(abs(t), n - 1))}


def choose(summary):
    """The pre-registered rule: among d1, d2, d3 the variant with the highest mean monthly AUC on S. `summary`: {variant: mean AUC}."""
    return max(CHOOSABLE, key=lambda v: summary[v])


def paired(tab, a, b, key, months):
    """Per-month difference a - b of metric `key` over months where both are finite."""
    d = [tab[a][m][key] - tab[b][m][key] for m in months if m in tab[a] and m in tab[b] and np.isfinite(tab[a][m][key]) and np.isfinite(tab[b][m][key])]
    return np.array(d)


# ------------------------------------------------------------------ the choice gate (as O8/O9)
CHOICE = os.path.join(HERE, "choice_d.json")


def _git(*args):
    import subprocess
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True)


def choice_gate():
    """Refuse anything on J unless choice_d.json is tracked at HEAD and unchanged in the working tree. Returns its content."""
    nm = os.path.basename(CHOICE)
    if _git("ls-files", "--error-unmatch", "--", CHOICE).returncode != 0:
        raise SystemExit(f"REFUSED: {nm} is not tracked in git. Commit it before any J number is computed.")
    if _git("diff", "--quiet", "HEAD", "--", CHOICE).returncode != 0:
        raise SystemExit(f"REFUSED: {nm} differs from HEAD (changed after its commit).")
    if _git("cat-file", "-e", f"HEAD:./{nm}").returncode != 0:
        raise SystemExit(f"REFUSED: {nm} is not in HEAD.")
    return json.load(open(CHOICE, encoding="utf-8"))


# ------------------------------------------------------------------ the frozen artefact (P)
def st_to_json(st):
    return {"use_g": st["use_g"], "days": st["days"], "mean": st["sc"].mean_.tolist(), "scale": st["sc"].scale_.tolist(), "coef": st["lr"].coef_[0].tolist(),
            "icpt": float(st["lr"].intercept_[0]), "n_tr": st["n_tr"], "first_ta": str(st["first_ta"]), "last_label_end": str(st["last_label_end"]), "base": st["base"]}


def predict_j(j, tb7, g15):
    Xf = np.concatenate([tb7, g15], 1) if j["use_g"] else tb7
    z = (np.asarray(Xf, np.float64) - np.array(j["mean"])) / np.array(j["scale"]); return 1 / (1 + np.exp(-(z @ np.array(j["coef"]) + j["icpt"])))


def freeze_fit(bars, m0, chosen, first=O3.FIRST):
    """Fit base and the chosen variant ONCE on every anchor of `bars` whose label ends at or before m0 (the Binance file's 2025-10-01 .. 2026-09-30 for the
    real freeze). Returns the artefact dict (with sha256)."""
    F = P5.build_frame(bars, first); ex = L9.train_extras(bars, F["ta"]); X = P2.make_X(F, ex); TX = train_data(F, X)
    art = {"spec": "O11 part d SPEC.md freeze", "m0": str(m0), "last_bar": str(bars.index.max()), "first_bar": str(bars.index.min()), "n_bars": int(len(bars)),
           "stride": T.STRIDE, "warm_bars": int(first), "horizon": H, "dead_band": L9.DEAD, "C": L9.C_LR, "gate": GATE, "chosen": chosen, "models": {}}
    for v in ("base", chosen):
        st = fit_variant(TX, v, m0); assert st["last_label_end"] <= m0; art["models"][v] = st_to_json(st)
    art["sha256"] = P5.sha_of(art); return art
