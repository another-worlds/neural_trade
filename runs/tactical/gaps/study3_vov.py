"""Study 3 (SPEC.md): predictability of the next-hour change in log realised volatility (target A) and of the realised vol-of-vol over the next 4
hours (target B) from HAR-style inputs vs a random walk and an AR baseline. Monthly walk-forward, stride 7, 365-day training.
usage: python study3_vov.py run --stage sel|look | select | report"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaps_lib as G  # noqa: E402
import volnet_lib as V  # noqa: E402

STRIDE = 7; STAGES = {"sel": ("2019-01", "2024-06"), "look": ("2024-07", "2025-07")}; SETS = ["S1", "S2", "S3"]; TARGETS = {"A": 60, "B": 240}


def build(s):
    D = V.Data(s.values.astype(np.float64), s.index, stride=STRIDE, with_tb7=False); a = D.anch; cs = D.cs2
    rvw = lambda lo, hi: cs[a + hi] - cs[a + lo]                              # RV over bars a+lo .. a+hi-1 (lo, hi relative to the anchor)
    prev = [rvw(-60 * (k + 1), -60 * k) for k in range(4)]; nxt = [rvw(60 * j, 60 * (j + 1)) for j in range(4)]
    ln = lambda x: np.log(x + 1e-12)
    L = np.column_stack([ln(p) for p in prev]); N = np.column_stack([ln(x) for x in nxt])
    vp = np.log(L.std(1, ddof=1) + 0.01); vB = np.log(N.std(1, ddof=1) + 0.01); dA = N[:, 0] - L[:, 0]
    okx = np.all([p > 0 for p in prev], 0) & np.all([x > 0 for x in nxt], 0) & np.all(D.har[:, :4] > -20, 1)
    S1 = D.har; S2 = np.column_stack([S1, D.extra]); S3 = np.column_stack([S2, L[:, 1:4], vp])
    global LV0; LV0 = L[:, 0]
    return D, {"S1": S1, "S2": S2, "S3": S3}, {"A": dA, "B": vB}, {"A": L[:, 0] - L[:, 1], "B": vp}, okx


def ols1(x, y):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return b


def run(stage):
    G.nice(); t0 = time.time(); s = G.load_bars(); D, FS, TG, AR, okx = build(s); lo, hi = STAGES[stage]
    rp = f"{HERE}/results3_{stage}.jsonl"; open(rp, "w").close()
    print("anchors", len(D.anch), "valid", int(okx.sum()), "dropped", int((~okx).sum()), f"{time.time() - t0:.0f}s", flush=True)
    for m in pd.period_range(lo, hi, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time; rows = []
        for tg, Hl in TARGETS.items():
            tr, te = V.month_masks(D.ta, m0, m1, Hl); tr_idx = np.where(tr & okx)[0]; te_idx = np.where(te & okx)[0]
            if len(tr_idx) < 5000 or len(te_idx) < 500:
                rows = []; break
            y = TG[tg]; yt = y[te_idx]; preds = {}
            preds["rw"] = np.zeros(len(te_idx)) if tg == "A" else AR[tg][te_idx]
            b = ols1(AR[tg][tr_idx], y[tr_idx]); preds["ar"] = b[0] + b[1] * AR[tg][te_idx]
            if tg == "A":                                                 # post-hoc descriptive baseline (added after the selection numbers, before the look): AR on the LEVEL L_0
                b2 = ols1(LV0[tr_idx], y[tr_idx]); preds["arlvl"] = b2[0] + b2[1] * LV0[te_idx]
            for nm in SETS:
                X = FS[nm]; mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-12
                preds[nm] = Ridge(alpha=1.0).fit((X[tr_idx] - mu) / sd, y[tr_idx]).predict((X[te_idx] - mu) / sd)
            mse = {k: float(np.mean((yt - p) ** 2)) for k, p in preds.items()}
            for k, p in preds.items():
                sp = float(spearmanr(p, yt)[0]) if np.std(p) > 0 else float("nan")
                rows.append({"month": str(m), "target": tg, "model": k, "mse": mse[k], "spearman": sp, "r2_rw": 1 - mse[k] / mse["rw"],
                             "skill_ar": 1 - mse[k] / mse["ar"], "n_te": len(te_idx), "n_tr": len(tr_idx)})
        if not rows:
            print(m, "skipped", flush=True); continue
        with open(rp, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(m, {(r["target"], r["model"]): round(r["skill_ar"], 4) for r in rows if r["model"] == "S1"}, f"{time.time() - t0:.0f}s", flush=True)
    assert D.ta[-1] <= G.CUTOFF + pd.Timedelta(minutes=1)


def load(stage):
    return pd.DataFrame([json.loads(l) for l in open(f"{HERE}/results3_{stage}.jsonl")])


def ci(x):
    x = np.asarray(x, float); n = len(x); mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return float(mu), float(mu - h), float(mu + h), float(mu / se if se > 0 else 0.0)


def levels(df, tg, label):
    lines = [f"-- target {tg} ({label}): mean over months of MSE / Spearman / R2 vs rw / skill vs ar"]
    for k in ["rw", "ar", "arlvl"] + SETS:
        d = df[(df.target == tg) & (df.model == k)]
        if len(d) == 0:
            continue
        lines.append(f"   {k:3s} mse {d.mse.mean():.4f}  spearman {d.spearman.mean():+.4f}  R2_rw {d.r2_rw.mean():+.4f}  skill_ar {d.skill_ar.mean():+.4f}  n={len(d)}")
    return lines


def select():
    df = load("sel"); sc = {}; lines = ["== selection months 2019-01..2024-06 (stride 7): paired skill vs ar =="]
    for tg in TARGETS:
        lines += levels(df, tg, "selection")
        sc[tg] = {}
        for nm in SETS:
            mu, lo, hi, t = ci(df[(df.target == tg) & (df.model == nm)].skill_ar); sc[tg][nm] = {"mean": mu, "lo": lo, "hi": hi, "t": t}
            lines.append(f"   skill_ar {nm}: {mu:+.4f} [{lo:+.4f},{hi:+.4f}] t={t:+.2f}")
    sel = {tg: max(SETS, key=lambda n: sc[tg][n]["t"]) for tg in TARGETS}
    json.dump({"selected": sel, "detail": sc}, open(f"{HERE}/selection3.json", "w"), indent=1)
    lines.append(f"SELECTED: {sel}"); open(f"{HERE}/selection3_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines))


def report():
    sel = json.load(open(f"{HERE}/selection3.json"))["selected"]; df = load("look"); res = {"selected": sel}; lines = ["== THE ONE LOOK 2024-07..2025-07 (partial July) =="]
    for lab, ms in (("all months", None), ("ex-July", [m for m in df.month.unique() if m <= "2025-06"])):
        d = df if ms is None else df[df.month.isin(ms)]
        for tg in TARGETS:
            lines += levels(d, tg, lab); nm = sel[tg]; x = d[(d.target == tg) & (d.model == nm)]
            mu, lo, hi, t = ci(x.skill_ar); r2 = x.r2_rw.mean(); r2ci = ci(x.r2_rw)
            ok = bool(mu >= 0.02 and lo > 0 and r2 > 0)
            lines.append(f"   SELECTED {nm}: skill vs ar {mu:+.4f} [{lo:+.4f},{hi:+.4f}], R2 vs rw {r2:+.4f} [{r2ci[1]:+.4f},{r2ci[2]:+.4f}] -> threshold met: {ok}")
            for o in SETS:
                if o != nm:
                    z = ci(d[(d.target == tg) & (d.model == o)].skill_ar); lines.append(f"   (descriptive) {o}: skill vs ar {z[0]:+.4f} [{z[1]:+.4f},{z[2]:+.4f}]")
            if ms is None:
                res[tg] = {"set": nm, "skill_ar": [mu, lo, hi], "r2_rw": r2, "success": ok, "n_months": len(x)}
    json.dump(res, open(f"{HERE}/look3.json", "w"), indent=1); open(f"{HERE}/look3_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[3])
    elif sys.argv[1] == "select":
        select()
    else:
        report()
