"""Strategy study v1 (NT-005): apply the SPEC's guard-rails and winner rule to a rescore's cells.csv.

    python runs/experiments/strategy_study_v1/analyze.py <rescore dir>   # writes <rescore dir>/study_analysis.{json,md}

Everything follows runs/experiments/strategy_study_v1/SPEC.md, fixed before scoring. Only dev cells rank and
choose; test cells are summarised in their own columns (D-020).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CANDIDATES_EXCLUDE_SUFFIX = "_ewma"          # twins: model-free baselines, never the winner
LONG_BIASED = ("vol_target", "vol_regime_long")
MINUTES_PER_YEAR = 525_600
EULER = 0.5772156649


def dsr(sr_annual: float, trial_srs_annual: np.ndarray, k: int, t_bars: int) -> float:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014), Gaussian per-bar returns assumed (skew 0,
    kurtosis 3): P(true SR > SR0), SR0 the expected maximum of k zero-skill trials whose SR estimates have
    the observed cross-trial variance. Per-bar units inside."""
    s = sr_annual / math.sqrt(MINUTES_PER_YEAR)
    v = float(np.var(np.asarray(trial_srs_annual) / math.sqrt(MINUTES_PER_YEAR), ddof=1))
    sr0 = math.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / k) + EULER * norm.ppf(1 - 1 / (k * math.e)))
    return float(norm.cdf((s - sr0) * math.sqrt(t_bars - 1) / math.sqrt(1 + 0.5 * s * s)))


def dev_bars(cells: pd.DataFrame, root: Path) -> int:
    """Pooled out-of-sample bars of the dev folds (one seed per fold: seeds share the market path)."""
    total = 0
    for _, g in cells[cells.role == "dev"].groupby("fold"):
        rel = Path(g.iloc[0]["run_dir"])
        path = next(p for p in (root / rel, Path("runs") / rel, rel) if (p / "predictions_oos.npz").exists())
        with np.load(path / "predictions_oos.npz", allow_pickle=False) as z:
            total += len(z["last_close"])
    return total


def main(rescore_dir: str) -> None:
    out = Path(rescore_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    root = Path(meta.get("store_root") or meta.get("store") or "runs")
    cells = pd.read_csv(out / "cells.csv")
    dev, test = cells[cells.role == "dev"], cells[cells.role == "test"]
    t_bars = dev_bars(cells, root)
    rows = []
    for cid, g in dev.groupby("config_id", sort=False):
        strat = g.iloc[0]["strategy"]
        excess = g["total_return"] - g["buy_and_hold/total_return"]
        null_by_fold = g.groupby("fold")["random_same_freq/percentile_sharpe_net"].mean()
        dd, bh_dd = g["max_drawdown"].mean(), g["buy_and_hold/max_drawdown"].mean()
        gt = test[test.config_id == cid]
        r = {
            "config_id": cid, "strategy": strat, "twin": CANDIDATES_EXCLUDE_SUFFIX in cid,
            "dev_sharpe_mean": g["sharpe_net"].mean(), "dev_sharpe_sd": g["sharpe_net"].std(ddof=1),
            "dev_sharpes": [round(x, 2) for x in g["sharpe_net"]],
            "dev_return_mean": g["total_return"].mean(), "dev_excess_vs_bh_mean": excess.mean(),
            "dev_null_pct_min_fold": float(null_by_fold.min()) if null_by_fold.notna().any() else float("nan"),
            "dev_maxdd_mean": dd, "dev_bh_maxdd_mean": bh_dd, "dev_trades_mean": g["n_trades"].mean(),
            "dev_breakeven_bps_mean": g["breakeven_cost_bps"].mean() if "breakeven_cost_bps" in g else float("nan"),
            "dev_gross_edge_bps_mean": (g["gross_edge_per_trade_bps"].mean()
                                        if "gross_edge_per_trade_bps" in g else float("nan")),
            "test_sharpe_mean": gt["sharpe_net"].mean(), "test_return_mean": gt["total_return"].mean(),
            "test_bh_return_mean": gt["buy_and_hold/total_return"].mean(),
        }
        r["g1_beats_flat"] = bool(r["dev_return_mean"] > 0)
        r["g2_beats_bh_paired"] = bool(r["dev_excess_vs_bh_mean"] > 0)
        r["g3_beats_null"] = bool(np.isfinite(r["dev_null_pct_min_fold"]) and r["dev_null_pct_min_fold"] >= 95)
        r["g4_drawdown"] = bool(dd <= bh_dd) if strat in LONG_BIASED else bool(dd <= 0.10)
        r["g5_activity"] = bool(r["dev_trades_mean"] >= 20)
        r["passes"] = all(r[k] for k in ("g1_beats_flat", "g2_beats_bh_paired", "g3_beats_null", "g4_drawdown",
                                         "g5_activity"))
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("dev_sharpe_mean", ascending=False, kind="stable").reset_index(drop=True)
    cand = tab[~tab.twin]
    k_cand, k_all = len(cand), len(tab)
    passing = cand[cand.passes]
    winner = None
    if len(passing):
        w = passing.iloc[0]
        winner = {"config_id": w.config_id, "strategy": w.strategy, "dev_sharpe_mean": w.dev_sharpe_mean,
                  "dsr_k_candidates": dsr(w.dev_sharpe_mean, cand.dev_sharpe_mean.values, k_cand, t_bars),
                  "dsr_k_all": dsr(w.dev_sharpe_mean, tab.dev_sharpe_mean.values, k_all, t_bars)}
        winner["evidenced"] = bool(winner["dsr_k_candidates"] >= 0.95)
    top = cand.iloc[0]
    result = {"rescore_dir": str(out), "dev_bars_pooled": t_bars, "k_candidates": k_cand, "k_with_twins": k_all,
              "winner": winner or {"config_id": "always_flat", "reason": "no candidate passes every guard-rail"},
              "top_candidate_by_dev_sharpe": {
                  "config_id": top.config_id, "dev_sharpe_mean": top.dev_sharpe_mean,
                  "dsr_k_candidates": dsr(top.dev_sharpe_mean, cand.dev_sharpe_mean.values, k_cand, t_bars)},
              "table": json.loads(tab.to_json(orient="records"))}
    (out / "study_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cols = ["config_id", "dev_sharpe_mean", "dev_sharpe_sd", "dev_return_mean", "dev_excess_vs_bh_mean",
            "dev_null_pct_min_fold", "dev_maxdd_mean", "dev_trades_mean", "dev_breakeven_bps_mean",
            "g1_beats_flat", "g2_beats_bh_paired", "g3_beats_null", "g4_drawdown", "g5_activity", "passes",
            "test_sharpe_mean", "test_return_mean", "test_bh_return_mean"]
    md = ["# Strategy study v1: guard-rails and winner (SPEC rules)", "",
          f"Dev bars pooled: {t_bars}; K = {k_cand} candidates ({k_all} with twins). Ranked by mean dev net Sharpe; "
          "test columns shown, never used.", "", "```", json.dumps(result["winner"], indent=2), "```", "",
          tab[cols].to_markdown(index=False, floatfmt=".4g")]
    (out / "study_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("winner", "top_candidate_by_dev_sharpe", "dev_bars_pooled")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
