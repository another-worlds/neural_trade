"""The paired comparator for "A beats B" verdicts (NT-032, D-025; D-037 amendment).

Run directories are hand-written (meta.json + result.json only): the comparator reads only those
two files through the run store (NT-026), so a real training run is unnecessary here and the fast
suite stays fast. ``tests/test_experiment_engine.py`` covers the store's own reading of real runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from neural_trade.experiments.comparator import (
    CompareError, CompareSpec, compare, intersection_union_verdict, non_inferiority_verdict,
    pair_runs, per_fold_retention, simulate_error_rates,
)
from neural_trade.experiments.store import RunStore
from neural_trade.metrics.statistics import hodges_lehmann, pocock_alpha, wilcoxon_hl_ci


def _write_run(root: Path, scenario: str, *, seed: int, fold: int, scores: dict, run_id=None,
               dataset_sha256="ds-1", bar_minutes=1.0, horizon_steps=(10, 15, 20), created_utc="20260101T000000Z",
               status="done"):
    run_id = run_id or f"{created_utc}-{scenario}-s{seed}-f{fold}"
    d = root / "scenarios" / scenario / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id, "seed": seed, "tags": [], "created_utc": created_utc,
        "engine": {"scenario": scenario, "cell_key": run_id, "configuration": "default", "variant": None,
                  "params": {}, "fold": fold, "fold_id": fold, "role": "dev", "seed": seed, "commit": "deadbeef",
                  "config_hash": "cfg1", "settings_hash": "set1", "spec_hash": "spec1"},
        "dataset": {"sha256": dataset_sha256, "path": "bars.csv", "first_timestamp": "2020-01-01",
                   "last_timestamp": "2020-01-08", "n_bars": 10000},
        "setup": {"bar_minutes": bar_minutes, "LOOKBACK": 60, "HORIZON_STEPS": list(horizon_steps)},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    result = {"status": status, "scores": scores, "wall_s": 1.0, "sec_per_step": 0.1,
             "finished_utc": created_utc}
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return d


def _base_spec(root, **kw):
    d = dict(name="t", scenario_a="A", scenario_b="B", metric="h1/direction/auc", min_effect=0.01,
            judgment_folds=[-1], root=str(root), registered_utc="20260101T000000Z")
    d.update(kw)
    return CompareSpec.from_dict(d)


AUC_KEY = "h1/direction/auc"


def _populate(root, n_pairs=5, a_auc=0.55, b_auc=0.50, noise=0.0, seed_offset=0, fold=-1,
             dataset_sha256=("ds-1", "ds-1")):
    rng = np.random.default_rng(0)
    for i in range(n_pairs):
        s = seed_offset + i
        av = a_auc + (rng.normal(0, noise) if noise else 0.0)
        bv = b_auc + (rng.normal(0, noise) if noise else 0.0)
        _write_run(root, "A", seed=s, fold=fold, scores={AUC_KEY: av}, dataset_sha256=dataset_sha256[0],
                  created_utc=f"2026010{1 + i}T000000Z")
        _write_run(root, "B", seed=s, fold=fold, scores={AUC_KEY: bv}, dataset_sha256=dataset_sha256[1],
                  created_utc=f"2026010{1 + i}T000000Z")


# ------------------------------------------------------------------------------- (1) pairing, verdict
def test_pairs_by_seed_and_fold_and_verdicts_a_beats_b_when_the_ci_clears_the_minimum_effect(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.60, b_auc=0.50)
    spec = _base_spec(tmp_path, min_effect=0.02)
    result = compare(spec)
    assert result.verdict == "A beats B"
    assert len(result.pairs) == 6
    assert result.refusal_reason is None


def test_inconclusive_when_the_ci_straddles_the_minimum_effect(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.501, b_auc=0.500, noise=0.02)
    spec = _base_spec(tmp_path, min_effect=0.05)
    result = compare(spec)
    assert result.verdict == "inconclusive"


def test_b_beats_a_when_b_is_the_better_scenario(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.50, b_auc=0.60)
    spec = _base_spec(tmp_path, min_effect=0.02)
    result = compare(spec)
    assert result.verdict == "B beats A"


def test_fewer_than_min_pairs_is_refused(tmp_path):
    _populate(tmp_path, n_pairs=4)
    spec = _base_spec(tmp_path)
    result = compare(spec)
    assert result.verdict == "refused"
    assert "pair(s) survived" in result.refusal_reason


def test_min_pairs_below_five_is_refused_by_the_spec_itself(tmp_path):
    with pytest.raises(CompareError):
        _base_spec(tmp_path, min_pairs=3)


def test_a_fold_the_spec_does_not_name_as_a_judgement_fold_is_excluded_not_paired(tmp_path):
    _populate(tmp_path, n_pairs=6, fold=-2)
    spec = _base_spec(tmp_path, judgment_folds=[-1])          # runs are all on fold -2
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("judgement fold" in e.reason for e in excluded)
    result = compare(spec)
    assert result.verdict == "refused"


def test_a_dataset_fingerprint_mismatch_between_a_and_b_excludes_the_pair(tmp_path):
    _populate(tmp_path, n_pairs=6, dataset_sha256=("ds-1", "ds-2"))
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("fingerprint mismatch" in e.reason for e in excluded)


# --------------------------------------------------------------------- (2) pre-registration, spec hash
def test_spec_hash_is_recorded_in_the_output(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02)
    out = compare(spec).to_dict()
    assert out["spec_hash"] == spec.spec_hash
    assert len(out["spec_hash"]) == 12


def test_a_spec_registered_after_a_compared_run_started_is_refused(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.6, b_auc=0.5)     # runs created_utc 20260101.. 20260106
    spec = _base_spec(tmp_path, min_effect=0.02, registered_utc="20260201T000000Z")   # registered later
    result = compare(spec)
    assert result.verdict == "refused"
    assert "before the spec's registration" in result.refusal_reason


def test_a_pre_registered_pair_count_is_enforced_no_peeking(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02, pairs_planned=5)
    result = compare(spec)
    assert result.verdict == "refused"
    assert "no peeking" in result.refusal_reason
    spec_ok = _base_spec(tmp_path, min_effect=0.02, pairs_planned=6)
    assert compare(spec_ok).verdict == "A beats B"


# ------------------------------------------------------------------------------------ (3) guard-rails
def test_a_guard_rail_breach_is_judged_by_the_same_paired_test(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(6):
        s = i
        _write_run(tmp_path, "A", seed=s, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={AUC_KEY: 0.60, "h1/variance/crpss": 0.01 + rng.normal(0, 0.001)})
        _write_run(tmp_path, "B", seed=s, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={AUC_KEY: 0.50, "h1/variance/crpss": 0.05 + rng.normal(0, 0.001)})
    spec = _base_spec(tmp_path, min_effect=0.02, guard_rails=[
        {"metric": "h1/variance/crpss", "direction": "higher_better", "max_degradation": 0.01}])
    result = compare(spec)
    assert result.verdict == "A beats B"
    assert result.guard_rail_results[0]["verdict"] == "breach"


def test_a_guard_rail_within_tolerance_passes(tmp_path):
    for i in range(6):
        s = i
        _write_run(tmp_path, "A", seed=s, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={AUC_KEY: 0.60, "h1/variance/crpss": 0.050})
        _write_run(tmp_path, "B", seed=s, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={AUC_KEY: 0.50, "h1/variance/crpss": 0.051})
    spec = _base_spec(tmp_path, min_effect=0.02, guard_rails=[
        {"metric": "h1/variance/crpss", "direction": "higher_better", "max_degradation": 0.01}])
    result = compare(spec)
    assert result.guard_rail_results[0]["verdict"] == "pass"


# --------------------------------------------------------------------------- (4) simulated error rates
def test_the_null_false_beats_rate_is_at_most_five_percent_plus_monte_carlo_error(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02, noise_sd=0.02, pairs_planned=10)
    sim = simulate_error_rates(spec, n_pairs=10, n_sim=1000, seed=0)
    assert sim["n_sim"] >= 1000
    assert sim["false_beats_rate"] <= 0.05 + 3 * sim["false_beats_mc_error"]
    assert 0.0 <= sim["power_at_2x_min_effect"] <= 1.0


def test_the_simulation_is_seeded_and_reproducible(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02, noise_sd=0.02, pairs_planned=8)
    a = simulate_error_rates(spec, n_pairs=8, n_sim=1000, seed=7)
    b = simulate_error_rates(spec, n_pairs=8, n_sim=1000, seed=7)
    assert a == b


def test_simulation_needs_a_variance_component_or_it_is_refused(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02)
    with pytest.raises(CompareError):
        simulate_error_rates(spec, n_pairs=8)


def test_seed_and_block_components_combine_the_same_as_an_equal_noise_sd(tmp_path):
    spec_direct = _base_spec(tmp_path, min_effect=0.02, noise_sd=0.025)
    spec_split = _base_spec(tmp_path, min_effect=0.02, seed_sd=0.015, block_sd=0.02)
    sim_direct = simulate_error_rates(spec_direct, n_pairs=10, seed=3)
    sim_split = simulate_error_rates(spec_split, n_pairs=10, seed=3)
    assert sim_split["noise_sd"] == pytest.approx(np.sqrt(0.015 ** 2 + 0.02 ** 2))
    assert sim_direct["noise_sd"] == pytest.approx(0.025)


# ------------------------------------------------------------------------------ (5) JSON + markdown
def test_output_is_json_serialisable_and_the_markdown_names_pairs_metric_effect_and_verdict(tmp_path):
    _populate(tmp_path, n_pairs=6, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02)
    result = compare(spec)
    doc = json.dumps(result.to_dict())            # raises if not serialisable
    assert AUC_KEY in doc
    md = result.to_markdown()
    assert "A beats B" in md
    assert "0.02" in md
    assert AUC_KEY in md
    assert str(spec.judgment_folds[0]) in md
    assert spec.spec_hash in md


def test_a_refused_comparison_still_renders_a_short_markdown(tmp_path):
    _populate(tmp_path, n_pairs=2)
    spec = _base_spec(tmp_path)
    result = compare(spec)
    md = result.to_markdown()
    assert "Refused" in md


# --------------------------------------------------------------------- spec parsing / CLI-facing bits
def test_from_yaml_round_trips_and_rejects_unknown_keys(tmp_path):
    d = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01, judgment_folds=[-1])
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    spec = CompareSpec.from_yaml(path)
    assert spec.name == "t" and spec.judgment_folds == (-1,)

    bad = dict(d, not_a_field=1)
    path2 = tmp_path / "bad.yaml"
    path2.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(CompareError):
        CompareSpec.from_yaml(path2)


def test_an_invalid_direction_or_metric_kind_is_refused(tmp_path):
    base = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01, judgment_folds=[-1])
    with pytest.raises(CompareError):
        CompareSpec.from_dict({**base, "direction": "sideways"})
    with pytest.raises(CompareError):
        CompareSpec.from_dict({**base, "metric_kind": "ratio"})


# ------------------------------------------------------------------------------ D-037 amendment points
def test_log_ratio_metric_kind_uses_the_log_of_the_ratio(tmp_path):
    for i in range(6):
        _write_run(tmp_path, "A", seed=i, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={"m": 1.10})
        _write_run(tmp_path, "B", seed=i, fold=-1, created_utc=f"2026010{1 + i}T000000Z",
                  scores={"m": 1.00})
    spec = _base_spec(tmp_path, metric="m", min_effect=0.05, metric_kind="log_ratio")
    result = compare(spec)
    assert result.pairs[0].diff == pytest.approx(np.log(1.10 / 1.00), abs=1e-9)
    assert result.verdict == "A beats B"


def test_hodges_lehmann_estimator_is_robust_to_one_outlier_pair(tmp_path):
    vals_a = [0.55] * 5 + [0.90]        # one wild outlier pair
    for i, av in enumerate(vals_a):
        _write_run(tmp_path, "A", seed=i, fold=-1, created_utc=f"2026010{1 + i}T000000Z", scores={AUC_KEY: av})
        _write_run(tmp_path, "B", seed=i, fold=-1, created_utc=f"2026010{1 + i}T000000Z", scores={AUC_KEY: 0.50})
    spec_mean = _base_spec(tmp_path, min_effect=0.02, estimator="mean")
    spec_hl = _base_spec(tmp_path, min_effect=0.02, estimator="hodges_lehmann")
    r_mean = compare(spec_mean)
    r_hl = compare(spec_hl)
    # the outlier pulls the mean up more than the HL estimate (HL ~ median of Walsh averages)
    assert r_hl.estimate["estimate"] < r_mean.estimate["estimate"]


def test_hodges_lehmann_and_wilcoxon_ci_on_a_symmetric_sample_are_consistent():
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    est = hodges_lehmann(d)
    lo, hi = wilcoxon_hl_ci(d, alpha=0.05)
    assert lo <= est <= hi


def test_pocock_alpha_two_looks_is_about_0_03_one_sided():
    assert pocock_alpha(0.05, 2) == pytest.approx(0.0294 / 2, abs=1e-4)
    assert pocock_alpha(0.05, 1) == 0.025


def test_non_inferiority_pass_breach_undecided():
    assert non_inferiority_verdict(-0.01, 0.02, margin=0.05) == "pass"
    assert non_inferiority_verdict(-0.10, -0.06, margin=0.05) == "breach"
    assert non_inferiority_verdict(-0.10, 0.02, margin=0.05) == "undecided"


def test_per_fold_retention_generic_helper():
    diff_by_fold = {-1: [0.01, 0.012, 0.011], -2: [0.02, 0.021]}
    baseline_by_fold = {-1: 0.03, -2: 0.06}
    out = per_fold_retention(diff_by_fold, baseline_by_fold, denom=3)
    assert out["folds"] == [-2, -1]
    assert len(out["r_f"]) == 2
    assert out["verdict"] in {"non_inferior", "breach", "undecided"}


def test_intersection_union_verdict_adopts_only_when_every_component_passes():
    assert intersection_union_verdict({"g1": "pass", "g2": "non_inferior"})["overall"] == "adopt"
    out = intersection_union_verdict({"g1": "pass", "speed": "breach"}, owner_route=["speed"])
    assert out["overall"] == "needs_owner"
    out2 = intersection_union_verdict({"g1": "breach"})
    assert out2["overall"] == "reject"


# ---------------------------------------------------------------------------------- run-store reading
def test_pair_runs_reads_the_engine_run_store_scores_table(tmp_path):
    _populate(tmp_path, n_pairs=5, a_auc=0.55, b_auc=0.50)
    store = RunStore(tmp_path)
    store.sync("A")
    rows = store.index.rows(scenario="A")
    assert len(rows) == 5
    assert all(r["status"] == "done" for r in rows)
