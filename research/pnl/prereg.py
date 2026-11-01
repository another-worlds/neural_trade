"""Pre-registration (SPEC 'selection_and_preregistration' PRE-REGISTRATION): writes research/pnl/prereg.json and
prereg.md from the dev gate results and the frozen fold-S models. Commit and push both BEFORE the seal is opened;
run_seal.py then requires PNL_UNSEAL = SHA-256(prereg.json).

  python prereg.py [--draft]     (--draft writes $PNL_WORK/prereg_draft.json only, for run_seal.py --dry-run)
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import data as D
import prep

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(draft=False):
    gate = json.load(open(os.path.join(RESULTS, "gate.json")))
    sel = {t: json.load(open(os.path.join(RESULTS, f"net_selection_T{t}.json"))) for t in (1, 2)}
    ridge = {t: json.load(open(os.path.join(RESULTS, f"ridge_T{t}.json"))) for t in (1, 2)}
    files = [os.path.join(D.DATA, "btcusdt_perp_1m.npz"), os.path.join(D.DATA, "funding_full.npz"),
             os.path.join(prep.WORK, "features_v1.json"), os.path.join(prep.WORK, "features_v1.npy"),
             os.path.join(prep.WORK, "sigma_v1.npy"), os.path.join(prep.WORK, "dev_daily.npz"),
             os.path.join(prep.WORK, "ridge_T1_S.npz"), os.path.join(prep.WORK, "ridge_T2_S.npz")]
    files += sorted(glob.glob(os.path.join(prep.WORK, "nets", "T*_S_h*_*.pt")))
    nt_path = os.path.join(prep.WORK, "nt", "T1-NT_S.json")
    t1_nt = json.load(open(nt_path)) if os.path.exists(nt_path) else None
    if t1_nt:
        files += [p for p in sorted(glob.glob(os.path.join(t1_nt["bundle"], "**", "*"), recursive=True))
                  if os.path.isfile(p)]
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "research/pnl", "model.py"],
                           capture_output=True, text=True).stdout.strip()
    if dirty and not draft:
        sys.exit(f"refused: commit research/pnl and model.py first:\n{dirty}")
    import run_seal
    by_role = {k: sha(p) for k, p in run_seal.role_files().items()}
    tracks = {}
    for t in ("1", "2"):
        g = gate["tracks"][t]
        tracks[t] = {"deployed": g["deployed"], "information_only": g["information_only"], "p_sel": g["p_sel"],
                     "candidates": {c: {k: v[k] for k in ("gates", "pass", "mean_daily_bps", "ann_bps", "sharpe",
                                                         "boot_ci95_daily_bps", "boot_lb95_daily_bps", "hac_t",
                                                         "deflated_sharpe", "n_trials", "by_year_bps", "halves_bps",
                                                         "scenarios_ann_bps", "mean_x")}
                                    for c, v in g["candidates"].items()},
                     "comparators": g["comparators"], "missing": g.get("missing", [])}
        if t == "1":
            tracks[t]["K1_track1_closed"] = g.get("K1_track1_closed")
    out = {
        "title": "PnL optimisation, BTCUSDT perp: pre-registration of the sealed test",
        "code_commit": head, "uncommitted_changes_at_write": bool(dirty),
        "seal": {"start": D.utc(D.SEAL_START).isoformat(), "end": "2026-10-30 23:59 UTC (last bar in the store)",
                 "clean_seal_excludes": ["2025-11-01 00:00 .. 2025-11-10 07:30", "2026-09-20 .. 2026-10-30"]},
        "tracks": tracks,
        "net_selection_S": {t: sel[int(t)]["S"] for t in ("1", "2")},
        "ridge_S": {t: {k: ridge[int(t)]["folds"]["S"][k] for k in ("H", "theta", "lambda", "cash")} for t in ("1", "2")},
        "t1_nt": t1_nt,
        "static_mean_x": {c: v["mean_x"] for t in ("1", "2") for c, v in gate["tracks"][t]["candidates"].items()},
        "procedure": {
            "features": "features_v1 (F-CORE 48 + F-DAILY 7; Track 1 uses 42 intraday columns)",
            "folds": "annual expanding; fold S trained on all dev data to 2025-10-31, frozen for the seal",
            "fees": "C7 primary (maker 2 bps entry, taker 5 bps exit); C4 and C10 reported",
            "fills": "B base (1-tick trade-through, next-open taker); S stress (1 bps trade-through, +1 bar latency, "
                     "+1 bps taker slippage)",
            "track1_execution": "maker entries K = 1 (h <= 5) or 3 (h = 15) bars then cancel; taker exits",
            "track2_execution": "maker at the decision close, chase to taker after 15 bars",
            "nets": "LIN/MLP x kappa {1, 2} x h; 5 seeds; ensemble tau/b; hard band; weights rounded to 0.05",
            "ridge": "H x theta {0.5, 1, 2} x lambda {1, 10, 1e2, 1e3, 1e4}; purged blocked 5-fold CV on engine F",
            "positions_start_flat_at_seal_start": True,
        },
        "verdict_rules": {
            "track1": "PASS: one-sided stationary-bootstrap p < 0.05 for mean daily net > 0 at C7-B on the full seal, "
                      "AND mean > 0 on the clean seal, AND mean > 0 in both seal halves. KILL: mean <= 0 on the full "
                      "or the clean seal. Otherwise INCONCLUSIVE.",
            "track2": "FALSIFIED: seal Sharpe < SR_dev - 1.645 sqrt(1/T_seal + 1/T_dev) on the full or the clean seal. "
                      "CONSISTENT: not falsified and seal net PnL at C7-B > 0. WEAK: not falsified, PnL <= 0. "
                      "CONFIRMED (flag): one-sided 95% lower bound of the pooled dev-OOS-plus-seal mean daily net > 0.",
            "meaning": "PASS or CONSISTENT means only 'go to at least 12 months of paper trading', never capital. "
                       "A deployed candidate that is KILLED or FALSIFIED is dropped for this venue and fee tier.",
            "cash_track": "If a track deploys CASH, its information-only candidate is scored with the same rules, "
                          "labelled information only.",
        },
        "secondary_information_only": [
            {"id": "S1", "candidate": "T1-LIN", "scenario": "C10S",
             "hypothesis": "With taker entries, T1-LIN's extreme-volatility reversal edge survives one bar of latency "
                           "and 1 bps slippage: mean daily net at C10-S > 0 on the full seal (one-sided stationary "
                           "bootstrap p < 0.05, mean block 10) and mean > 0 on the clean seal.",
             "why": "post-hoc dev diagnostics (results/diag_t1lin.json): C7-S fails through maker entries under "
                    "latency; C10-S was +0.75 bps/day on dev. Information only: it can never deploy anything.",
             "power_note": "T1-LIN traded about 34 times a year on dev, 80% of them in 2021 crash days; the seal may "
                           "hold few or no trades, in which case S1 is untestable, not passed."}],
        "power": {"track1": "dev OOS detects about 1 bps per trade; the seal about 2 bps (2,000-4,000 fade trades)",
                  "track2": "minimum detectable Sharpe 2.49 on the seal, 2.68 on the clean seal; the seal can "
                            "falsify Track 2, it cannot confirm it"},
        "disclosures": [
            "Seen windows inside the seal: 2025-10-11..2025-11-10 and 2026-09-20..2026-10-30 were studied in earlier "
            "work in this repository (clean seal excludes them).",
            "Trend contamination: earlier checks computed daily TSMOM and vol-targeted long BTC on 2022-01..2026-11, "
            "covering most of dev OOS and the whole seal; trend-family results are not clean on either period.",
            "Earlier strat_probe and money-blocker critique looked at dev data.",
            "The designers (language models) may know the 2025-26 BTC price path from training data.",
            "2026-10 funding prints are estimated from the premium index (MAE 0.02 bps per print).",
            "The 2026-10-28 kline file was missing (a 1-day gap in the seal).",
            "MLP band nets stayed at cash on Track 1 (also with the declared extended warm-up); LIN cells carry "
            "T1-NET. Reported as found, procedure unchanged.",
            "T1-NT's fold-S selection abstained (no horizon had positive validation net), so T1-NT is flat on the "
            "seal. On dev it is retired from trading use by K2 (T1-LIN beats it on C7-B net), although its OOS IC "
            "is small but positive (h = 1/5/15: 0.022/0.042/0.047).",
            "Two code reviews and one documented full dev rerun (C4 rows only) happened before this file; see "
            "CHANGES.log. The C7 gate scenarios did not change in the rerun.",
        ],
        "hashes": {p: sha(p) for p in files},
        "hashes_by_role": by_role,
    }
    path = os.path.join(prep.WORK, "prereg_draft.json") if draft else os.path.join(HERE, "prereg.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    if not draft:
        md = ["# Pre-registration: sealed test of the PnL study", "",
              f"Code commit: `{head}`. Seal: {out['seal']['start']} .. {out['seal']['end']}.", ""]
        for t in ("1", "2"):
            tr = tracks[t]
            md += [f"## Track {t}", "", f"- Deployed: **{tr['deployed']}**; information only: {tr['information_only']}",
                   f"- p_sel (dev OOS): {tr['p_sel']:.3f}", ""]
            md += ["| candidate | ann. net C7-B (bps) | Sharpe | 95% CI daily (bps) | gates passed |",
                   "|---|---:|---:|---|---|"]
            for c, v in tr["candidates"].items():
                ci = v["boot_ci95_daily_bps"]
                md.append(f"| {c} | {v['ann_bps']:.0f} | {v['sharpe']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                          f"{', '.join(k for k, ok in v['gates'].items() if ok) or 'none'} |")
            md.append("")
        md += ["## Verdict rules", ""] + [f"- {k}: {v}" for k, v in out["verdict_rules"].items()]
        md += ["", "## Disclosures", ""] + [f"- {d}" for d in out["disclosures"]]
        md += ["", f"Hashes of {len(out['hashes'])} data, feature and model files are in prereg.json."]
        open(os.path.join(HERE, "prereg.md"), "w").write("\n".join(md) + "\n")
    print(path)
    return out


if __name__ == "__main__":
    main(draft="--draft" in sys.argv)
