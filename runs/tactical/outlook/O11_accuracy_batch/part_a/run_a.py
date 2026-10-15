"""O11 part a runner. usage: run_a.py [run|tables]. Geometry for the volatility-CHANGE forecast (T01's dA at 60 minutes; secondary: the 4-hour vol-of-vol B).
base = ridge on S3, candidate = ridge on L1's features (S3 + bank + G1 + G15); monthly walk-forward on 365 days (S, B), frozen on the 365 days to 2025-07-20 (C)."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_vol_lib as L  # noqa: E402
import common_run as R  # noqa: E402
log, done_keys, write, rows_df, RES = R.make(HERE)
O3 = L.O3


def run():
    R.nice(); t_all = time.time(); assert R.free_gb() >= 8
    Fc = L.load_cut(); Ff = L.load_fwd_frame(); done = done_keys(("span", "month", "tg"))
    log("frames", len(Fc["ta"]), len(Ff["ta"]), f"free {R.free_gb():.0f} GB")
    for span, months in (("S", L.S_MONTHS), ("B", L.B_MONTHS)):
        for mo in months:
            m0, m1 = L.month_bounds(mo); t0 = time.time()
            for tg in L.TG:
                if (span, mo, tg) in done:
                    continue
                st = L.fit_a(Fc, tg, m0); te = np.where((Fc["ta"] >= np.datetime64(m0, "ns")) & (Fc["ta"] < np.datetime64(m1, "ns")) & Fc["oka"])[0]
                assert len(te) > 0
                write(L.score_a(st, Fc, te, span, mo))
            log(span, mo, f"{time.time() - t0:.0f}s", f"free {R.free_gb():.0f} GB")
    for tg in L.TG:
        if all(("C", m, tg) in done for m in L.C_MONTHS):
            continue
        st = L.fit_a(Fc, tg, L.TRAIN_END)
        for mo in L.C_MONTHS:
            m0, m1 = L.month_bounds(mo)
            te = np.where((Ff["ta"] >= np.datetime64(m0, "ns")) & (Ff["ta"] < np.datetime64(m1, "ns")) & Ff["oka"])[0]
            assert len(te) > 0
            write(L.score_a(st, Ff, te, "C", mo))
        log("C frozen", tg, "n_tr", st["n_tr"])
    log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime.json"), "w"))


def gain_series(df, tg, months, col, a="cand", b="base"):
    d = df[(df.tg == tg) & df.month.isin(months)]
    c = d[d.model == a].set_index("month")[col].astype(float); b = d[d.model == b].set_index("month")[col].astype(float)
    return (c - b).reindex(months).dropna()


def tables():
    df = rows_df(); out = []; Q = out.append; J = L.B_MONTHS + L.C_MONTHS
    summary = {}
    Q(f"O11 part a tables. rows {len(df)}. skill = 1 - MSE/MSE_AR (T01); gain = candidate - base per month; one-sample t over months")
    Q("S = 2021-01..2024-06 (descriptive only), B = 2024-07..2025-07-20, C = 2025-10..2026-09 (frozen at 2025-07-20); J = B + C")
    for tg, lab in (("A", "PRIMARY target A: dA (next-hour minus last-hour ln RV)"), ("B", "SECONDARY target B: 4-hour vol-of-vol")):
        Q(f"\n===== {lab} =====")
        for name, ms in (("S", L.S_MONTHS), ("B", L.B_MONTHS), ("C", L.C_MONTHS), ("J=B+C", J)):
            d = df[(df.tg == tg) & df.month.isin(ms)]
            mb = d[d.model == "base"].skill.mean(); mc = d[d.model == "cand"].skill.mean()
            g = L.series_stats(gain_series(df, tg, ms, "skill")); sp = L.series_stats(gain_series(df, tg, ms, "spearman"))
            Q(f"  {name:6s} months {d.month.nunique():2d}  mean skill base {mb:+.4f} cand {mc:+.4f}\n         gain skill   {L.fmt(g)}\n         gain Spearman {L.fmt(sp)}")
            summary[f"{tg}|{name}"] = {"skill_gain": g, "spearman_gain": sp, "skill_base": float(mb), "skill_cand": float(mc)}
    g = summary["A|J=B+C"]["skill_gain"]
    c1 = g["mean"] >= 0.010; c2 = g["lo"] > 0
    Q("\n===== PRIMARY TEST (target A, J = B + C, skill gain candidate - base) =====")
    Q(f"  mean {g['mean']:+.4f} [CI {g['lo']:+.4f},{g['hi']:+.4f}] positive {g['pos']}/{g['n']} raw p {g['p']:.5f}")
    Q(f"  mean >= +0.010: {c1}; lower bound > 0: {c2} -> {'PASS' if c1 and c2 else 'FAIL'} before Holm (the lead applies Holm)")
    Q("\n===== DECOMPOSITION of the candidate gain (target A, skill; descriptive, not part of the test): bank = S3 + EWMA bank, no geometry =====")
    for name, ms in (("S", L.S_MONTHS), ("B", L.B_MONTHS), ("C", L.C_MONTHS), ("J=B+C", J)):
        for a_, b_ in (("bank", "base"), ("cand", "bank")):
            st_ = L.series_stats(gain_series(df, "A", ms, "skill", a_, b_)); summary[f"A|{name}|{a_}-{b_}"] = st_
            Q(f"  {name:6s} {a_}-{b_}: {L.fmt(st_)}")
    Q("\n===== monthly series, target A: skill base | cand | gain | Spearman gain =====")
    dA = df[df.tg == "A"]
    for sp_, ms in (("B", L.B_MONTHS), ("C", L.C_MONTHS)):
        for m in ms:
            x = dA[dA.month == m].set_index("model")
            if {'base', 'cand'} <= set(x.index):
                Q(f"  {sp_} {m} n_te {int(x.loc['cand', 'n_te'])}  {x.loc['base', 'skill']:+.4f} | {x.loc['cand', 'skill']:+.4f} | {x.loc['cand', 'skill'] - x.loc['base', 'skill']:+.4f} | "
                  f"{x.loc['cand', 'spearman'] - x.loc['base', 'spearman']:+.4f}")
    Q("\n===== by year (target A, skill gain) =====")
    for y in ("2021", "2022", "2023", "2024", "2025", "2026"):
        ms = [m for m in L.S_MONTHS + J if m.startswith(y)]
        Q(f"  {y}: {L.fmt(L.series_stats(gain_series(df, 'A', ms, 'skill')))}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"primary": g, "pass_before_holm": bool(c1 and c2), "summary": summary}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    {"run": run, "tables": tables}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
