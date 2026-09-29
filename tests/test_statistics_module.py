"""NT-027: one metrics and statistics module (AUC with its DeLong variance, the effective-sample
helpers, the block bootstrap, the long-run variance) - evaluation/report.py and the figure modules
call it instead of computing their own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

SRC = Path(__file__).resolve().parents[1] / "src" / "neural_trade"
STATS_MODULE = SRC / "metrics" / "statistics.py"
# D-023's frozen set: history, exempt from this and every other NT-027 layering rule.
FROZEN = {SRC / "experiments" / "ablation.py"}


def _py_files():
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts or path == STATS_MODULE or path in FROZEN:
            continue
        yield path


def test_no_other_auc_implementation_in_src():
    """``roc_auc_score`` (the sklearn AUC) and a rank-based AUC (sort/rank + cumsum or trapz, the
    ROC-curve pattern this codebase used independently in three places before NT-027) appear nowhere
    under src/neural_trade except the shared statistics module."""
    offenders = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        if "roc_auc_score" in text:
            offenders.append(f"{path.relative_to(SRC)}: imports/uses roc_auc_score directly")
            continue
        has_rank = ("argsort" in text and "-scores" in text) or "rankdata" in text
        has_curve_math = "trapz" in text or "trapezoid" in text
        if has_rank and has_curve_math:
            offenders.append(f"{path.relative_to(SRC)}: looks like an independent rank-based AUC/ROC calc")
    assert not offenders, "AUC computed outside metrics/statistics.py:\n" + "\n".join(offenders)


def test_statistics_module_holds_auc_delong_bootstrap_and_long_run_variance():
    from neural_trade.metrics import statistics as S

    for name in ("auc_score", "roc_curve", "roc_points", "delong_placements", "auc_difference",
                "long_run_variance", "dm_z", "block_bootstrap_counts", "n_eff", "wilson", "mean_ci",
                "corr_null", "corr_null_r", "auc_ci", "thin"):
        assert callable(getattr(S, name)), f"metrics.statistics is missing {name}"


def test_auc_score_matches_sklearn_and_report_and_figures_agree_with_it():
    from neural_trade.evaluation.report import direction_block
    from neural_trade.metrics.statistics import auc_score, roc_curve, roc_points

    rng = np.random.default_rng(0)
    labels = (rng.random(300) > 0.4).astype(float)
    scores = rng.normal(size=300) + labels * 0.6

    want = roc_auc_score(labels, scores)
    assert auc_score(labels, scores) == pytest.approx(want)
    assert roc_curve(labels, scores, max_points=100000)[2] == pytest.approx(want)
    assert roc_points(labels, scores, max_points=100000)[3] == pytest.approx(want)

    mask = np.ones(len(labels), dtype=bool)
    row = direction_block(labels, mask, scores)
    assert row["auc"] == pytest.approx(want)


def test_evaluation_report_and_figures_import_the_shared_module():
    import ast

    for path, needle in ((SRC / "evaluation" / "report.py", "neural_trade.metrics.statistics"),
                         (SRC / "visualization" / "analytics_common.py", "neural_trade.metrics.statistics"),
                         (SRC / "visualization" / "analytics_direction.py", "neural_trade.metrics.statistics")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {n.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for n in [node]}
        assert needle in modules, f"{path.relative_to(SRC)} does not import {needle}"
