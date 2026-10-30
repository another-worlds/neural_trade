"""Build the markdown result tables for the evaluation report from the metrics JSON files.

Usage: python make_tables.py > tables.md
Reads results/ (repo holdout) and results_post/ (post-release) metrics2.json, the *_vol metrics,
incremental2.json, samay_equivalence.json and results_latency.jsonl. No numbers are typed by hand.
"""
import json
import os

BASE = os.environ.get("TSFM_EVAL_WORK",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
SETS = {"repo": f"{BASE}/results", "post": f"{BASE}/results_post"}
H = [1, 5, 15]

NAMES = {
    "timestral-5M_ctx512_price": "Timestral-5M, ctx 512",
    "timestral-28M_ctx512_price": "Timestral-28M, ctx 512",
    "timestral-28M_ctx2048_price": "Timestral-28M, ctx 2048",
    "timestral-209M_ctx512_price": "Timestral-209M, ctx 512",
    "timestral-209M_ctx2048_price": "Timestral-209M, ctx 2048",
    "official-chronos2_ctx512_price": "Chronos-2 (Samay family), ctx 512",
    "official-chronos2_ctx2048_price": "Chronos-2 (Samay family), ctx 2048",
    "official-bolt_small_ctx512_price": "Chronos-Bolt-S (Samay family), ctx 512 †",
    "samay-timesfm25_ctx512_price": "TimesFM-2.5 via Samay, ctx 512 †",
    "samay-timesfm25_ctx2048_price": "TimesFM-2.5 via Samay, ctx 2048 †",
    "neural_trade-asis-seed0_ctx60_price": "neural_trade as-is, seed 0",
    "neural_trade-asis-seed1_ctx60_price": "neural_trade as-is, seed 1",
    "neural_trade-asis-seed2_ctx60_price": "neural_trade as-is, seed 2",
    "neural_trade-rolling1440m-seed0_ctx60_price": "neural_trade retrained daily, seed 0",
    "neural_trade-rolling1440m-seed1_ctx60_price": "neural_trade retrained daily, seed 1",
    "neural_trade-rolling1440m-seed2_ctx60_price": "neural_trade retrained daily, seed 2",
    "official-chronos2_ctx512_logret": "Chronos-2, ctx 512, log-return input ‡",
    "official-chronos2_ctx2048_logret": "Chronos-2, ctx 2048, log-return input ‡",
    "samay-timesfm25_ctx512_logret": "TimesFM-2.5 via Samay, ctx 512, log-return input † ‡",
    "timestral-28M_ctx512_aggret": "Timestral-28M, h-min return input, 512 bars §",
    "timestral-28M_ctx2048_aggret": "Timestral-28M, h-min return input, 2048 bars §",
    "official-chronos2_ctx512_aggret": "Chronos-2, h-min return input, 512 bars §",
    "ewma_t_scaled": "EWMA-t, variance scale refit for QLIKE (pre-test)",
    "fhs_ewma": "FHS: EWMA σ × empirical residual quantiles (pre-test)",
    "ewma_gauss_0.94": "EWMA-Gaussian (λ 0.94)",
    "hist_quantiles_1d": "Trailing 1-day empirical quantiles",
    "logreg_pastret_prefit": "Logistic regression on past returns",
    "reversal_sign": "15/5/1-min reversal sign",
}
MAIN = list(NAMES)


def load(ds):
    p = f"{SETS[ds]}/metrics2.json"
    return json.load(open(p)) if os.path.exists(p) else None


def rows_by(d, setname="full"):
    out = {}
    for r in d["rows"]:
        if r["set"] == setname:
            name = r["model"] if not r["model"].startswith("REF_") else "REF"
            out[(name, r["h"])] = r
    return out


def ci(v):
    return "" if not v else f"[{v[0]:+.2f}, {v[1]:+.2f}]"


def sig(pct, c):
    if pct is None or not c:
        return f"{pct:+.2f}" if pct is not None else "–"
    star = "**" if (c[0] > 0 or c[1] < 0) else ""
    return f"{star}{pct:+.2f}{star} {ci(c)}"


def fmt(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table_point_direction(ds):
    d = load(ds)
    if not d:
        return f"_{ds}: no results_\n"
    R = rows_by(d)
    lines = ["| Model | MAE skill vs RW (1 / 5 / 15 min) | MCC of sign (1 / 5 / 15) | AUC of P(up) (1 / 5 / 15) | Brier skill ×1000 (1 / 5 / 15) |",
             "|---|---|---|---|---|"]
    for m in MAIN:
        if (m, 1) not in R or m in ("ewma_t_scaled", "fhs_ewma"):
            continue
        mae = " / ".join(f"{R[(m, h)].get('mae_skill_vs_rw', 0) * 100:+.2f}%" for h in H)
        mcc = " / ".join(fmt(R[(m, h)].get("dir_mcc")) for h in H)
        aucs = []
        for h in H:
            r = R[(m, h)]
            a = r.get("auc_up", r.get("auc_dirhead"))
            c = r.get("auc_up_ci95", r.get("auc_dirhead_ci95"))
            star = "**" if (a is not None and c and (c[0] > 0.5 or c[1] < 0.5)) else ""
            cell = "–" if a is None else f"{star}{a:.3f}{star}"
            if "auc_up" in r and "auc_dirhead" in r:
                cell += f" (head {r['auc_dirhead']:.3f})"
            aucs.append(cell)
        bs = " / ".join("–" if R[(m, h)].get("brier_skill") is None else f"{1000 * R[(m, h)]['brier_skill']:+.1f}" for h in H)
        lines.append(f"| {NAMES[m]} | {mae} | {mcc} | {' / '.join(aucs)} | {bs} |")
    return "\n".join(lines) + "\n"


def table_distribution(ds):
    d = load(ds)
    if not d:
        return f"_{ds}: no results_\n"
    R = rows_by(d)
    lines = ["| Model | h | Pinball (9 deciles) vs REF, % [95% CI] | QLIKE vs REF, % [95% CI] | Cover 80% | Cover 98% | Spearman(width, \\|y\\|) |",
             "|---|---|---|---|---|---|---|"]
    for m in ["REF"] + MAIN:
        for h in H:
            r = R.get((m, h))
            if r is None or "pinball9_bps" not in r:
                continue
            name = f"REF: EWMA-Student-t (λ {d['baseline_fit'][str(h)]['lambda']}, ν {d['baseline_fit'][str(h)]['nu']:g})" if m == "REF" else NAMES[m]
            pb = "0 (reference)" if m == "REF" else sig(r.get("pinball9_vs_ref_pct"), r.get("pinball9_vs_ref_ci95_pct"))
            ql = "0 (reference)" if m == "REF" else sig(r.get("qlike_vs_ref_pct"), r.get("qlike_vs_ref_ci95_pct"))
            lines.append(f"| {name} | {h} | {pb} | {ql} | {fmt(r.get('cov80'))} | {fmt(r.get('cov98'))} | {fmt(r.get('vol_spearman'))} |")
    return "\n".join(lines) + "\n"


def table_ablation():
    lines = ["| Set | Config | h=1 pinball / QLIKE vs REF, % | h=5 pinball / QLIKE | h=15 pinball / QLIKE | ms/series (batched) |",
             "|---|---|---|---|---|---|"]
    for ds in SETS:
        d = load(ds)
        if not d:
            continue
        groups = {}
        for r in d["rows"]:
            if r["model"].startswith("timestral") or r["model"].startswith("official-chronos2"):
                groups.setdefault((r["set"], r["model"]), {})[r["h"]] = r
        for (s, m), hr in sorted(groups.items()):
            cells = []
            for h in H:
                r = hr.get(h, {})
                if "pinball9_vs_ref_pct" in r:
                    cells.append(f"{sig(r['pinball9_vs_ref_pct'], r['pinball9_vs_ref_ci95_pct'])} / "
                                 f"{sig(r['qlike_vs_ref_pct'], r['qlike_vs_ref_ci95_pct'])}")
                else:
                    cells.append(f"MAE skill {r.get('mae_skill_vs_rw', 0) * 100:+.2f}%" if r else "–")
            ms = hr.get(1, {}).get("ms_per_series_batched", "")
            lines.append(f"| {ds}/{s} | {m} | {cells[0]} | {cells[1]} | {cells[2]} | {ms} |")
    return "\n".join(lines) + "\n"


def table_tails(ds):
    d = load(ds)
    if not d:
        return ""
    R = rows_by(d)
    lines = ["| Model | h | P(y < q0.01) | P(y < q0.05) | P(y > q0.95) | P(y > q0.99) | Kupiec p (q0.01 / q0.99) | Newey-West p (q0.01 / q0.99) |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ["REF", "ewma_t_scaled", "fhs_ewma", "timestral-28M_ctx2048_price", "timestral-209M_ctx2048_price",
              "timestral-28M_ctx2048_aggret", "official-chronos2_ctx512_price", "official-chronos2_ctx512_aggret",
              "neural_trade-asis-seed0_ctx60_price", "neural_trade-rolling1440m-seed0_ctx60_price"]:
        for h in H:
            r = R.get((m, h))
            if not r or "hit_q01" not in r:
                continue
            lines.append(f"| {NAMES.get(m, 'REF: EWMA-Student-t')} | {h} | {r['hit_q01']:.4f} | {r['hit_q05']:.4f} | "
                         f"{1 - r['hit_q95']:.4f} | {1 - r['hit_q99']:.4f} | {fmt(r['kupiec_p_q01'])} / {fmt(r['kupiec_p_q99'])} | "
                         f"{fmt(r.get('nw_p_q01'))} / {fmt(r.get('nw_p_q99'))} |")
    return "\n".join(lines) + "\n"


def table_trades(ds):
    d = load(ds)
    if not d:
        return ""
    R = rows_by(d)
    lines = ["| Model | h | 4 bps: n / net bps (t) | 10 bps: n / net (t) | 20 bps: n / net (t) |", "|---|---|---|---|---|"]
    for m in MAIN:
        for h in H:
            r = R.get((m, h))
            if not r or "trades_nonoverlap" not in r or m in ("ewma_gauss_0.94", "ewma_t_scaled", "fhs_ewma",
                                                               "logreg_pastret_prefit", "reversal_sign", "hist_quantiles_1d"):
                continue
            cells = []
            for k in ("cost4", "cost10", "cost20"):
                t = r["trades_nonoverlap"][k]
                cells.append(f"{t['n']}" + (f" / {t['net_bps']:+.1f} ({t['t_stat']:+.1f})" if "net_bps" in t else ""))
            lines.append(f"| {NAMES[m]} | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_strong_baselines(ds):
    d = load(ds)
    if not d:
        return ""
    R = rows_by(d)
    lines = ["| Model | h | Pinball (9) vs EWMA-t scale-refit, % | QLIKE vs EWMA-t scale-refit, % | Pinball (9) vs FHS, % | QLIKE vs FHS, % |",
             "|---|---|---|---|---|---|"]
    for m in ["fhs_ewma", "timestral-28M_ctx512_price", "timestral-28M_ctx2048_price", "timestral-209M_ctx2048_price",
              "timestral-28M_ctx512_aggret", "timestral-28M_ctx2048_aggret", "official-chronos2_ctx512_price",
              "official-chronos2_ctx2048_logret", "official-chronos2_ctx512_aggret", "neural_trade-asis-seed0_ctx60_price",
              "neural_trade-rolling1440m-seed0_ctx60_price"]:
        for h in H:
            r = R.get((m, h))
            if not r or "qlike_vs_scl_pct" not in r:
                continue
            cells = [sig(r.get(f"{k}_vs_{b}_pct"), r.get(f"{k}_vs_{b}_ci95_pct")) if f"{k}_vs_{b}_pct" in r else "–"
                     for b in ("scl", "fhs") for k in ("pinball9", "qlike")]
            lines.append(f"| {NAMES[m]} | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_vol(ds):
    p = f"{SETS[ds]}_vol/metrics_vol.json"
    if not os.path.exists(p):
        return f"_{ds}: no volatility results_\n"
    rows = json.load(open(p))
    full = max({r["set"] for r in rows}, key=lambda s: int(s[1:]))
    lines = ["| Model | h | Mean abs log error | Spearman | QLIKE | vs EWMA: QLIKE %, [CI] | vs HAR (smeared): QLIKE %, [CI] | "
             "vs HAR (plain, log-OLS): log-error %, [CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["h"], r["mean_abs_log_err"])):
        if r["set"] != full or "medianpath" in r["model"] or r["model"].startswith(("rollmean", "prev_day")):
            continue
        lines.append(f"| {r['model']} | {r['h']} | {r['mean_abs_log_err']:.4f} | {r['spearman']:.3f} | {r['qlike']:.4f} | "
                     f"{sig(r.get('qlike_vs_ewma_pct'), r.get('qlike_vs_ewma_ci95_pct')) if 'qlike_vs_ewma_pct' in r else '–'} | "
                     f"{sig(r.get('qlike_vs_harS_pct'), r.get('qlike_vs_harS_ci95_pct')) if 'qlike_vs_harS_pct' in r else '–'} | "
                     f"{sig(r.get('male_vs_har_pct'), r.get('male_vs_har_ci95_pct')) if 'male_vs_har_pct' in r else '–'} |")
    return "\n".join(lines) + "\n"


def table_incremental(ds):
    p = f"{SETS[ds]}/incremental2.json"
    if not os.path.exists(p):
        return ""
    lines = ["| Forecast features from | h | AUC base | AUC model-only | AUC base + model | ΔAUC [95% CI] |", "|---|---|---|---|---|---|"]
    for r in json.load(open(p)):
        c = r["delta_auc_ci95"]
        star = "**" if (c[0] > 0 or c[1] < 0) else ""
        lines.append(f"| {NAMES.get(r['tag'], r['tag'])} | {r['h']} | {r['auc_base']:.3f} | {r['auc_tsfm_only']:.3f} | "
                     f"{r['auc_base+tsfm']:.3f} | {star}{r['delta_auc_base+tsfm_minus_base']:+.4f}{star} [{c[0]:+.4f}, {c[1]:+.4f}] |")
    return "\n".join(lines) + "\n"


def table_latency():
    p = f"{BASE}/results_latency.jsonl"
    if not os.path.exists(p):
        return "_no latency results_\n"
    rows = [json.loads(l) for l in open(p)]
    lines = ["| Model | Context | Symmetric avg. | p50 ms | p95 ms | p99 ms | Cold start ms | Peak RSS MB |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['ctx']} | {r.get('symmetric', '–')} | {r['p50_ms']} | {r.get('p95_ms', '–')} | "
                     f"{r.get('p99_ms', '–')} | {r.get('cold_start_ms_ctx2048', '–')} | {r.get('peak_rss_mb', '–')} |")
    pn = f"{BASE}/nt_latency.jsonl"
    if os.path.exists(pn):
        for r in (json.loads(l) for l in open(pn) if l.strip()):
            for path, label in [("predict_path", "Model.predict, as in model.py"), ("compiled_call", "tf.function call")]:
                s = r[path]
                lines.append(f"| neural_trade as-is ({label}, {r['threads']} thr) | {r['ctx']} | – | {s['p50_ms']} | "
                             f"{s['p95_ms']} | {s['p99_ms']} | – | – |")
    return "\n".join(lines) + "\n"


def table_matched(ds):
    p = f"{SETS[ds]}/matched.json"
    if not os.path.exists(p):
        return ""

    def cell(v):
        if v is None:
            return "–"
        m, (lo, hi) = v
        s = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        return f"**{s}**" if lo > 0 or hi < 0 else s

    lines = ["| Variant | Base | n | Pinball (9 deciles) h=1 / 5 / 15 | QLIKE h=1 / 5 / 15 |", "|---|---|---|---|---|"]
    for r in json.load(open(p))["rows"]:
        pb = " / ".join(cell(r.get(f"pinball9_h{h}")) for h in H)
        ql = " / ".join(cell(r.get(f"qlike_h{h}")) for h in H)
        lines.append(f"| {r['variant']} | {r['base']} | {r['n']} | {pb} | {ql} |")
    return "\n".join(lines) + "\n"


LEGEND = ("Notes for all tables. **Bold** = the 95% CI excludes 0 (0.5 for AUC). Bold does not mean better: in loss tables "
          "a positive value is worse and a negative value is better. QLIKE is the unit-free form y²/v − ln(y²/v) − 1 on origins "
          "with y ≠ 0; v is the variance implied by the 21 quantiles. Chronos-2 and Chronos-Bolt rows come from the "
          "official `chronos-forecasting` package; for these two models Samay gives the same numbers to 0.0008 bp (T-I). "
          "‡ Log-return input: only h = 1 has a forecast distribution; h = 5 and 15 use the summed per-step medians as a "
          "point forecast. § h-min return input: for each h, the input is the series of non-overlapping h-minute returns "
          "ending at the origin (512 or 2048 bars), and the model forecasts one step. "
          "† Decile-only model (q0.1 … q0.9): its q0.01, q0.05, q0.95 and q0.99 are clamped to q0.1 / q0.9, so its 98% coverage, "
          "tail rates and QLIKE are biased against it. Compare it on 9-decile pinball only. "
          "neural_trade quantiles come from its Student-t head (ν = 3) centred on its price head; the 21-level grid keeps "
          "about 79% of a t(3) variance (97% for the t(6) EWMA-t), so its QLIKE here is slightly flattering to it. "
          "Trades fill at the forecast bar's close with no slippage and no delay.\n")


if __name__ == "__main__":
    out = [LEGEND]
    for ds, title in [("repo", "Repo holdout (2,895 origins, 2025-11-04 → 11-10)"),
                      ("post", "Post-release clean set (6,715 origins, 2026-10-16 → 10-29)")]:
        d = load(ds)
        out.append(f"\n#### T-A ({ds}). Point and direction — {title}\n")
        out.append(table_point_direction(ds))
        out.append(f"\n#### T-B ({ds}). Distribution and variance vs EWMA-Student-t — {title}\n")
        out.append(table_distribution(ds))
        out.append(f"\n#### T-B2 ({ds}). Against two stronger cheap baselines (% change of loss; negative = model better)\n")
        out.append(table_strong_baselines(ds))
        out.append(f"\n#### T-C ({ds}). Tail calibration (target 1% / 5% / 5% / 1%; p < 0.05 = fail)\n")
        out.append(table_tails(ds))
        out.append(f"\n#### T-D ({ds}). Non-overlapping trades when |forecast| > cost\n")
        out.append(table_trades(ds))
        out.append(f"\n#### T-E ({ds}). Volatility role: mean 1-min high-low range over next h min\n")
        out.append(table_vol(ds))
        out.append(f"\n#### T-F ({ds}). Feature role: walk-forward logistic regression\n")
        out.append(table_incremental(ds))
        if d:
            out.append(f"\nWindow facts ({ds}): `{json.dumps(d['sets'])}`; baseline fit: `{json.dumps(d['baseline_fit'])}`\n")
    out.append("\n#### T-G. Context, size and framing ablations (all sets)\n")
    out.append(table_ablation())
    for ds in ("repo", "post"):
        out.append(f"\n#### T-J ({ds}). Paired ablations on identical origins (variant vs base, % change of loss; negative = variant better)\n")
        out.append(table_matched(ds))
    out.append("\n#### T-H. Single-series CPU latency (4 threads unless stated, fp32, package defaults)\n")
    out.append(table_latency())
    p = f"{SETS['repo']}/samay_equivalence.json"
    if os.path.exists(p):
        out.append("\n#### T-I. Samay (locked env) vs official packages, identical origins\n")
        out.append("```json\n" + open(p).read() + "\n```\n")
    print("\n".join(out))
