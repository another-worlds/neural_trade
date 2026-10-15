"""O11 part c runner. usage: run_c.py run [S|BC|all] | tables. Quantile regression with geometry for the 5% / 95% tails of the next-60-minute log return r60.
base (O7 method): L1 ridge of ln RV60 fitted at the start of the previous month f; the empirical 5% / 95% quantiles of r60 / sigma_hat on month f (labels ending by
  the start of the scored month) times sigma_hat of the scored rows.
cand: for the scored month M, the L1 ridge fitted on the 365 days before M gives ln sigma_hat; QuantileRegressor(alpha 1e-4, highs) at 5% and 95% on a stride-70
  subsample of those rows, features [ln sigma_hat, G15 (164), G1 top-15 by |standardised L1 coefficient| of that fit, hour sin/cos], standardised on the subsample.
Spans: S (descriptive, 2021-01..2024-06), B (2024-07..2025-07), C (2025-11..2026-09: one frozen fit on the 365 days to 2025-07-20 for both models, the
  base calibrating on the previous month with the frozen sigma; 2025-10 has no previous month, so C starts in 2025-11)."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_vol_lib as L  # noqa: E402
import common_run as R  # noqa: E402
log, done_keys, write, rows_df, RES = R.make(HERE)
O3 = L.O3; C_SCORED = L.C_MONTHS[1:]


def prev(m):
    return str(pd.Period(m, freq="M") - 1)


def score_pair(Fte, te, qlo_b, qhi_b, qlo_c, qhi_c, span, mo, extra):
    r = Fte[f"r{L.H_C}"][te]; out = []
    for nm, (lo, hi) in (("base", (qlo_b, qhi_b)), ("cand", (qlo_c, qhi_c))):
        out.append({"span": span, "month": mo, "model": nm, **L.tail_row(r, lo, hi), **extra})
    return out


def run(which="all"):
    R.nice(); t_all = time.time(); assert R.free_gb() >= 8
    Fc = L.load_cut(); done = done_keys(("span", "month", "model"))
    spans = []
    if which in ("S", "all"):
        spans.append(("S", L.S_MONTHS))
    if which in ("BC", "all"):
        spans.append(("B", L.B_MONTHS))
    for span, months in spans:
        for mo in months:
            if (span, mo, "base") in done and (span, mo, "cand") in done:
                continue
            t0 = time.time(); f = prev(mo); m0f = L.month_bounds(f)[0]; m0 = L.month_bounds(mo)[0]
            cal, te = L.month_rows_c(Fc, f); assert len(cal) and len(te)
            trf = O3.train_rows(Fc["ta"], Fc["ok60"], 60, m0f); L.check_rows_clean(Fc, trf, 60, m0f)
            assert (Fc["ta"][cal] + np.timedelta64(60, "m") <= np.datetime64(m0, "ns")).all()
            mf = L.l1_fit(Fc, trf); bl, bh = L.base_quantiles(Fc, mf, cal, te)
            tr = O3.train_rows(Fc["ta"], Fc["ok60"], 60, m0); L.check_rows_clean(Fc, tr, 60, m0)
            mM = L.l1_fit(Fc, tr); qm = L.fit_qr(Fc, tr, mM); cl, ch = L.qr_predict(qm, Fc, te, mM)
            write(score_pair(Fc, te, bl, bh, cl, ch, span, mo, {"n_cal": int(len(cal)), "n_tr": int(len(tr)), "nz05": qm["nz"][L.Q_LO], "nz95": qm["nz"][L.Q_HI]}))
            log(span, mo, f"{time.time() - t0:.0f}s", f"free {R.free_gb():.0f} GB")
    if which in ("BC", "all"):
        Ff = L.load_fwd_frame(); F = L.combine(Fc, Ff)
        tr = O3.train_rows(F["ta"], F["ok60"], 60, L.TRAIN_END); L.check_rows_clean(F, tr, 60, L.TRAIN_END)
        assert pd.Timestamp(F["ta"][tr].max()) < L.TRAIN_END
        m = L.l1_fit(F, tr); qm = None
        for mo in C_SCORED:
            if ("C", mo, "base") in done and ("C", mo, "cand") in done:
                continue
            if qm is None:
                qm = L.fit_qr(F, tr, m)
            f = prev(mo); cal, te = L.month_rows_c(F, f); assert len(cal) and len(te)
            bl, bh = L.base_quantiles(F, m, cal, te); cl, ch = L.qr_predict(qm, F, te, m)
            write(score_pair(F, te, bl, bh, cl, ch, "C", mo, {"n_cal": int(len(cal)), "n_tr": int(len(tr)), "nz05": qm["nz"][L.Q_LO], "nz95": qm["nz"][L.Q_HI]}))
            log("C", mo, f"free {R.free_gb():.0f} GB")
    log("total", which, f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s_" + which: time.time() - t_all}, open(os.path.join(HERE, f"runtime_{which}.json"), "w"))


def diff_series(df, months, col="pin_bps"):
    d = df[df.month.isin(months)]
    c = d[d.model == "cand"].set_index("month")[col].astype(float); b = d[d.model == "base"].set_index("month")[col].astype(float)
    return (c - b).reindex(months).dropna()


def pooled(df, model, months):
    d = df[(df.model == model) & df.month.isin(months)]; n = d.n.sum()
    return {"hit05": float(d.n_lo.sum() / n), "hit95": float(d.n_hi.sum() / n), "pin_bps": float(d.pin_bps.mean()), "n": int(n), "months": int(d.month.nunique())}


def tables():
    df = rows_df(); out = []; Q = out.append; J = L.B_MONTHS + C_SCORED; summary = {}
    Q(f"O11 part c tables. rows {len(df)}. pinball in bps (lower is better), mean of the 5% and 95% losses; difference = candidate - base (negative = candidate better)")
    Q("S = 2021-01..2024-06 (descriptive only), B = 2024-07..2025-07-20, C = 2025-11..2026-09 (frozen), J = B + C (24 months). Hit rates are pooled over rows: P(r < q05), P(r < q95)")
    for name, ms in (("S", L.S_MONTHS), ("B", L.B_MONTHS), ("C", C_SCORED), ("J=B+C", J)):
        s = L.series_stats(diff_series(df, ms)); s05 = L.series_stats(diff_series(df, ms, "pin05_bps")); s95 = L.series_stats(diff_series(df, ms, "pin95_bps"))
        pb, pc = pooled(df, "base", ms), pooled(df, "cand", ms)
        Q(f"\n  {name:6s} months {pb['months']:2d}  mean pinball base {pb['pin_bps']:.3f} cand {pc['pin_bps']:.3f}")
        Q(f"         diff (avg of two) {L.fmt(s, 3)}  [candidate better in {s['neg']} months]")
        Q(f"         diff 5% {L.fmt(s05, 3)}\n         diff 95% {L.fmt(s95, 3)}")
        Q(f"         pooled hit rates  base {pb['hit05']:.4f} / {pb['hit95']:.4f}   cand {pc['hit05']:.4f} / {pc['hit95']:.4f}")
        summary[name] = {"diff": s, "diff05": s05, "diff95": s95, "base": pb, "cand": pc}
    g = summary["J=B+C"]["diff"]; pc = summary["J=B+C"]["cand"]
    c1 = g["mean"] < 0 and g["hi"] < 0; c2 = 0.04 <= pc["hit05"] <= 0.06 and 0.94 <= pc["hit95"] <= 0.96
    Q("\n===== PRIMARY TEST (J = B + C, 24 months, candidate - base pinball) =====")
    Q(f"  mean {g['mean']:+.4f} bps [CI {g['lo']:+.4f},{g['hi']:+.4f}] candidate better in {g['neg']}/{g['n']} months, raw p {g['p']:.5f}")
    Q(f"  mean < 0 and upper bound < 0: {c1}; pooled candidate hit rates {pc['hit05']:.4f} (in [0.04,0.06]) and {pc['hit95']:.4f} (in [0.94,0.96]): {c2} -> {'PASS' if c1 and c2 else 'FAIL'} before Holm")
    Q("\n===== monthly series (B, C): pinball base | cand | diff ; hit05 base/cand ; hit95 base/cand ; nonzero QR coefficients 05/95 =====")
    for sp_, ms in (("B", L.B_MONTHS), ("C", C_SCORED)):
        for m in ms:
            x = df[df.month == m].set_index("model")
            if len(x) == 2:
                Q(f"  {sp_} {m} n {int(x.loc['cand', 'n'])}  {x.loc['base', 'pin_bps']:.3f} | {x.loc['cand', 'pin_bps']:.3f} | {x.loc['cand', 'pin_bps'] - x.loc['base', 'pin_bps']:+.3f}   "
                  f"hit05 {x.loc['base', 'n_lo'] / x.loc['base', 'n']:.3f}/{x.loc['cand', 'n_lo'] / x.loc['cand', 'n']:.3f}  hit95 {x.loc['base', 'n_hi'] / x.loc['base', 'n']:.3f}/{x.loc['cand', 'n_hi'] / x.loc['cand', 'n']:.3f}  "
                  f"nz {int(x.loc['cand', 'nz05'])}/{int(x.loc['cand', 'nz95'])}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"primary": g, "cand_pooled": pc, "pass_before_holm": bool(c1 and c2), "summary": summary}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "run"
    if a == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else "all")
    else:
        tables()
