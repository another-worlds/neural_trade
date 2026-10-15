"""O11 part b runner. usage: run_b.py [run|tables]. Monthly refit instead of a frozen fit (C only, scored months 2025-11 .. 2026-09).
Setting: the volatility level, O5's L1 (and S3), ln RV at H = 15 and 60, ridge alpha 1.
  frozen        : fitted once on the 365 days to 2025-07-20 (as O1).
  refit_cal     : (PRIMARY) refitted for each month M on the 365 calendar days before M, from the cut file (<= 2025-07-20) and Binance bars from 2025-10-01
                  before M; rows of the empty gap 2025-07-21..2025-09-30 do not exist, so the early windows are shorter than 365 days of data.
  refit_avail   : (sensitivity, stated before the run) the most recent 365 days of AVAILABLE data before M: the window start is moved back by the 72-day gap
                  while it lies before the Binance file; the early windows reach into the last weeks of the cut file.
All features come from their own file (the cut file's frame and the Binance file's frame, the latter with O3's 6144-bar warm-up, so its first anchor is
2025-10-05 06:24). The refit never reads a bar of 2025-07-21..2025-09-30 (T.load_bars asserts it; the rows are checked too)."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_vol_lib as L  # noqa: E402
import common_run as R  # noqa: E402
log, done_keys, write, rows_df, RES = R.make(HERE)
O3 = L.O3; MODELS = ("S3", "L1"); HS = (15, 60); SCORED = L.C_MONTHS[1:]
ARMS = ("frozen", "refit_cal", "refit_avail")


def train_rows(F, arm, H, m0):
    if arm == "frozen":
        return O3.train_rows(F["ta"], F[f"ok{H}"], H, L.TRAIN_END)
    lo = m0 - pd.Timedelta(days=365) if arm == "refit_cal" else L.avail_lo(m0)
    return L.train_rows_between(F["ta"], F[f"ok{H}"], H, lo, m0)


def run():
    R.nice(); t_all = time.time(); assert R.free_gb() >= 8
    F = L.combine(L.load_cut(), L.load_fwd_frame()); done = done_keys(("arm", "month", "H"))
    log("combined frame", len(F["ta"]), pd.Timestamp(F["ta"][0]), pd.Timestamp(F["ta"][-1]), f"free {R.free_gb():.0f} GB")
    frozen = {}
    for H in HS:
        tr = train_rows(F, "frozen", H, None); L.check_rows_clean(F, tr, H, L.TRAIN_END)
        assert pd.Timestamp(F["ta"][tr].max()) < L.TRAIN_END
        frozen[H] = L.fit_ridge_h(F, H, tr, MODELS)
    for mo in SCORED:
        m0, m1 = L.month_bounds(mo); t0 = time.time()
        te_by_h = {H: O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1) for H in HS}
        for arm in ARMS:
            for H in HS:
                if (arm, mo, H) in done:
                    continue
                te = te_by_h[H]; assert len(te) > 0
                if arm == "frozen":
                    st = frozen[H]
                else:
                    tr = train_rows(F, arm, H, m0); L.check_rows_clean(F, tr, H, m0); st = L.fit_ridge_h(F, H, tr, MODELS)
                    assert pd.Timestamp(st["last_label_end"]) <= m0
                write(L.score_b(st, F, te, arm, mo))
        log(mo, f"{time.time() - t0:.0f}s", f"free {R.free_gb():.0f} GB")
    log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime.json"), "w"))


def diff_series(df, a, b, model, H, col, months=SCORED):
    d = df[(df.H == H) & (df.model == model)]
    x = d[d.arm == a].set_index("month")[col].astype(float); y = d[d.arm == b].set_index("month")[col].astype(float)
    return (x - y).reindex(months).dropna()


def tables():
    df = rows_df(); out = []; Q = out.append; summary = {}
    Q(f"O11 part b tables. rows {len(df)}. scored months {SCORED[0]}..{SCORED[-1]} ({len(SCORED)}); difference = refit - frozen per month; one-sample t over months")
    Q("QLIKE difference: negative = refit better. Spearman difference: positive = refit better.")
    for model in MODELS:
        for H in HS:
            Q(f"\n===== model {model}, H={H} =====")
            d = df[(df.H == H) & (df.model == model)]
            for arm in ARMS:
                x = d[d.arm == arm]; Q(f"  {arm:12s} mean Spearman {x.spearman.mean():.4f}  mean QLIKE {x.qlike.mean():.4f}  (months {x.month.nunique()})")
            for arm in ("refit_cal", "refit_avail"):
                s = L.series_stats(diff_series(df, arm, "frozen", model, H, "spearman")); q = L.series_stats(diff_series(df, arm, "frozen", model, H, "qlike"))
                Q(f"  {arm}-frozen Spearman {L.fmt(s)}\n  {arm}-frozen QLIKE    {L.fmt(q)}")
                summary[f"{model}|H{H}|{arm}"] = {"spearman": s, "qlike": q}
    g = summary["L1|H15|refit_cal"]["spearman"]; c1 = g["mean"] >= 0.003; c2 = g["lo"] > 0
    Q("\n===== PRIMARY TEST (L1, H=15, refit_cal - frozen, Spearman, 11 months) =====")
    Q(f"  mean {g['mean']:+.4f} [CI {g['lo']:+.4f},{g['hi']:+.4f}] positive {g['pos']}/{g['n']} raw p {g['p']:.5f}")
    Q(f"  mean >= +0.003: {c1}; lower bound > 0: {c2} -> {'PASS' if c1 and c2 else 'FAIL'} before Holm (the lead applies Holm)")
    Q("\n===== monthly series, L1 H=15: Spearman frozen | refit_cal | refit_avail | train days (cal / avail) =====")
    d = df[(df.H == 15) & (df.model == "L1")].set_index(["arm", "month"])
    for m in SCORED:
        a = d.loc[("frozen", m)]; b = d.loc[("refit_cal", m)]; c = d.loc[("refit_avail", m)]
        td = lambda r: (pd.Timestamp(m + "-01") - pd.Timestamp(r.first_ta)).days
        Q(f"  {m} {a.spearman:.4f} | {b.spearman:.4f} | {c.spearman:.4f}   train days {td(b)} / {td(c)}   n_tr {int(b.n_tr)} / {int(c.n_tr)}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"primary": g, "pass_before_holm": bool(c1 and c2), "summary": summary}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    {"run": run, "tables": tables}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
