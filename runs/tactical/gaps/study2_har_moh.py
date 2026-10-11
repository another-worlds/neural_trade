"""Study 2 (SPEC.md): HAR+tod vs HAR+tod+intra-hour periodic multiplier, monthly walk-forward (volnet protocol).
usage: python study2_har_moh.py run --stage sel|look --stride 7|10   -> results2_<stage>_s<stride>.jsonl
       python study2_har_moh.py select                               -> selection2.json, selection2_tables.txt (stride 7, selection months)
       python study2_har_moh.py report                               -> look2.json, look2_tables.txt (the ONE look; stride 7 primary, stride 10 descriptive + H57 comparison)"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaps_lib as G  # noqa: E402
import volnet_lib as V  # noqa: E402

STAGES = {"sel": ("2021-01", "2024-06"), "look": ("2024-07", "2025-07")}
VARIANTS = ["har", "m15", "moh", "cal"]; CAND = ["m15", "moh", "cal"]


def dummies(v, n):
    """one-hot of integer array v in [0,n) without the 0 column"""
    out = np.zeros((len(v), n - 1));
    for k in range(1, n):
        out[:, k - 1] = (v == k)
    return out


def design(D):
    mi = D.ta.minute.values; hr = D.ta.hour.values; har = D.har
    return {"har": har, "m15": np.column_stack([har, dummies(mi % 15, 15)]), "moh": np.column_stack([har, dummies(mi, 60)]),
            "cal": np.column_stack([har, dummies(mi, 60), dummies(hr, 24)])}


def run(stage, stride):
    G.nice(); t0 = time.time(); V.STRIDE = stride
    s = G.load_bars(); D = V.Data(s.values.astype(np.float64), s.index, stride=stride, with_tb7=False)
    X = design(D); Hs = D.horizons; Hmax = max(Hs)
    lo, hi = STAGES[stage]; rp = f"{HERE}/results2_{stage}_s{stride}.jsonl"; open(rp, "w").close()
    print("anchors", len(D.anch), "last ta", D.ta[-1], f"{time.time() - t0:.0f}s", flush=True)
    for m in pd.period_range(lo, hi, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = V.month_masks(D.ta, m0, m1, Hmax); tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        if len(tr_idx) < 5000 or len(te_idx) < 500:
            print(m, "skipped"); continue
        fit, ho = V.fit_hold(tr_idx, Hmax); rows = []
        for name in VARIANTS:
            mean, std, coef, ic = V.ridge_fit(X[name][fit], D.y[fit])
            pred = lambda idx: ((X[name][idx] - mean) / std) @ coef.T + ic
            mt, mh = pred(te_idx), pred(ho)
            for j, H in enumerate(Hs):
                rows.append({"month": str(m), "H": H, "model": name, **V.metrics(mt[:, j], mh[:, j], D.absr[te_idx, j], D.v2[te_idx, j], D.v2[ho, j]), "n_te": len(te_idx)})
        with open(rp, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(m, {r["model"]: round(r["spearman"], 4) for r in rows if r["H"] == 15}, f"{time.time() - t0:.0f}s", flush=True)
    assert D.ta[-1] <= G.CUTOFF + pd.Timedelta(minutes=1)


def load(stage, stride):
    return pd.DataFrame([json.loads(l) for l in open(f"{HERE}/results2_{stage}_s{stride}.jsonl")])


def paired(df, model, H, base="har"):
    b = df[(df.model == base) & (df.H == H)].set_index("month"); m = df[(df.model == model) & (df.H == H)].set_index("month"); j = b.index.intersection(m.index)
    return pd.DataFrame({"sp": m.loc[j, "spearman"] - b.loc[j, "spearman"], "ql": b.loc[j, "qlike"] - m.loc[j, "qlike"]}).sort_index()


def ci(x):
    x = np.asarray(x, float); n = len(x); mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return float(mu), float(mu - h), float(mu + h), float(mu / se if se > 0 else 0.0)


def table(df, models, label, months=None):
    out, lines = {}, [f"== {label} =="]
    if months is not None:
        df = df[df.month.isin(months)]
    for H in sorted(df.H.unique()):
        d = df[(df.model == "har") & (df.H == H)]
        lines.append(f"-- H={H} min: har level spearman {d.spearman.mean():.4f}, qlike {d.qlike.mean():.4f}, n months {len(d)}")
        for mo in models:
            p = paired(df, mo, H)
            for col, nm in (("sp", "dSpear"), ("ql", "dQLIKE")):
                mu, lo, hi, t = ci(p[col]); out.setdefault(f"{mo}_{H}", {})[col] = {"mean": mu, "lo": lo, "hi": hi, "t": t, "n": len(p)}
                lines.append(f"   {mo:4s} {nm:7s} {mu:+.5f} [{lo:+.5f},{hi:+.5f}] t={t:+.2f} n={len(p)}")
    return out, "\n".join(lines)


def select():
    df = load("sel", 7); out, txt = table(df, CAND, "selection months 2021-01..2024-06, stride 7 (paired vs har; dQLIKE = har - model)")
    score = {mo: float(np.mean([out[f"{mo}_{H}"][c]["t"] for H in sorted(df.H.unique()) for c in ("sp", "ql")])) for mo in CAND}
    best = max(score, key=score.get)
    json.dump({"selected": best, "scores_mean_t": score, "detail": out}, open(f"{HERE}/selection2.json", "w"), indent=1)
    open(f"{HERE}/selection2_tables.txt", "w").write(txt + f"\n\nscores (mean of six t): {score}\nSELECTED: {best}\n"); print(txt); print(score, "SELECTED", best)


def report():
    sel = json.load(open(f"{HERE}/selection2.json"))["selected"]; lines = []; res = {"selected": sel}
    d7 = load("look", 7); allm = sorted(d7.month.unique()); exj = [m for m in allm if m <= "2025-06"]
    for lab, ms in (("all look months incl. partial 2025-07", None), ("ex-July (<= 2025-06)", exj)):
        out, txt = table(d7, CAND if ms is None else [sel], f"THE ONE LOOK, stride 7, {lab}", ms); lines.append(txt)
        if ms is None:
            res["success_by_horizon"] = {str(H): bool(out[f"{sel}_{H}"]["sp"]["lo"] > 0 and out[f"{sel}_{H}"]["ql"]["lo"] > 0) for H in sorted(d7.H.unique())}
            res["detail_stride7"] = out
    lines.append(f"\nSUCCESS (selected={sel}, both lower bounds > 0): {res['success_by_horizon']}")
    # stride 10 (H57's stride) on H57's months 2024-07..2025-06 and the H57 ensemble
    d10 = load("look", 10); out10, txt10 = table(d10, CAND, "stride 10 (H57 stride), all look months (descriptive)", None); lines.append("\n" + txt10)
    h57 = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{HERE}/../volnet/results_look_[0-9].jsonl")) for l in open(f)])
    h57 = h57[h57.month.isin(exj)]
    cmp = {}; lines.append("\n== H57 ensemble vs the periodic multiplier, same months (2024-07..2025-06), Spearman diff vs har ==")
    for H in sorted(d7.H.unique()):
        e = paired(h57, "ens", H)["sp"]; row = {"ens_h57": float(e.mean()), "n_months": len(e)}
        for nm, dd in (("stride10", d10), ("stride7", d7)):
            for mo in CAND:
                p = paired(dd[dd.month.isin(exj)], mo, H)["sp"]; row[f"{mo}_{nm}"] = float(p.mean()); row[f"{mo}_{nm}_ci"] = [ci(p)[1], ci(p)[2]]
        cmp[str(H)] = row
        lines.append(f"H={H}: H57 ens {row['ens_h57']:+.5f} | " + " | ".join(f"{mo}/s{k} {row[f'{mo}_stride{k}']:+.5f}" for k in (10, 7) for mo in CAND))
    res["h57_comparison"] = cmp
    ok = {}
    for H in ("15", "60"):
        e = cmp[H]["ens_h57"]; v = cmp[H][f"{sel}_stride10"]; lo = cmp[H][f"{sel}_stride10_ci"][0]; ok[H] = (v / e if e else float("nan"), lo)
    frac = {H: ok[H][0] for H in ok}
    if all(ok[H][0] >= 2 / 3 and ok[H][1] > 0 for H in ok):
        verdict = "explains most"
    elif all(ok[H][0] >= 1 / 3 for H in ok):
        verdict = "explains part"
    else:
        verdict = "does not explain"
    res["h57_explained_fraction_stride10"] = frac; res["h57_verdict"] = verdict
    lines.append(f"\nfraction of H57's ensemble gain reproduced by {sel} at stride 10: {frac}; pre-registered reading: {verdict}")
    json.dump(res, open(f"{HERE}/look2.json", "w"), indent=1); open(f"{HERE}/look2_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = dict(zip(sys.argv[2::2], sys.argv[3::2])); run(a["--stage"], int(a["--stride"]))
    elif sys.argv[1] == "select":
        select()
    else:
        report()
