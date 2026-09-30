"""The paired comparator for "A beats B" verdicts (NT-032, D-025; D-046's fold-level repair).

Run directories are hand-written (meta.json + result.json only): the comparator reads only those
two files directly from the run store's directories (never through the sqlite index: F below), so a
real training run is unnecessary here and the fast suite stays fast.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from neural_trade.experiments.comparator import (
    CompareError, CompareSpec, _estimate, _verdict_from_ci, compare, intersection_union_verdict,
    non_inferiority_verdict, pair_runs, per_fold_retention, simulate_error_rates,
)
from neural_trade.experiments.store import RunStore
from neural_trade.metrics.statistics import hodges_lehmann, pocock_alpha, wilcoxon_hl_ci

AUC_KEY = "h1/direction/auc"
FIVE_FOLDS = (-1, -2, -3, -4, -5)


def _write_run(root: Path, scenario: str, *, seed: int, fold: int, scores: dict, run_id=None,
               dataset_sha256="ds-1", bar_minutes=1.0, horizon_steps=(10, 15, 20), lookback=60,
               created_utc="20260101T000000Z", status="done", configuration="default",
               test_block=(0, 43200)):
    run_id = run_id or f"{created_utc}-{scenario}-s{seed}-f{fold}-{configuration}"
    d = root / "scenarios" / scenario / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id, "seed": seed, "tags": [], "created_utc": created_utc,
        "engine": {"scenario": scenario, "cell_key": run_id, "configuration": configuration, "variant": None,
                  "params": {}, "fold": fold, "fold_id": fold, "role": "dev", "seed": seed, "commit": "deadbeef",
                  "config_hash": "cfg1", "settings_hash": "set1", "spec_hash": "spec1"},
        "dataset": {"sha256": dataset_sha256, "path": "bars.csv", "first_timestamp": "2020-01-01",
                   "last_timestamp": "2020-01-08", "n_bars": 10000},
        "setup": {"bar_minutes": bar_minutes, "LOOKBACK": lookback, "HORIZON_STEPS": list(horizon_steps)},
        "blocks": {"test": {"start": test_block[0], "stop": test_block[1]}},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    result = {"status": status, "scores": scores, "wall_s": 1.0, "sec_per_step": 0.1, "finished_utc": created_utc}
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return d


def _base_spec(root, **kw):
    d = dict(name="t", scenario_a="A", scenario_b="B", metric=AUC_KEY, min_effect=0.01,
            judgment_folds=list(FIVE_FOLDS), root=str(root), registered_utc="20260101T000000Z")
    d.update(kw)
    return CompareSpec.from_dict(d)


def _populate(root, *, folds=FIVE_FOLDS, seeds_per_fold=1, a_auc=0.55, b_auc=0.50, noise=0.0,
             dataset_sha256=("ds-1", "ds-1"), scenario_a="A", scenario_b="B", configuration="default"):
    rng = np.random.default_rng(0)
    day = 1
    for f in folds:
        for s in range(seeds_per_fold):
            created = f"202601{day:02d}T000000Z"
            day += 1
            av = a_auc + (rng.normal(0, noise) if noise else 0.0)
            bv = b_auc + (rng.normal(0, noise) if noise else 0.0)
            _write_run(root, scenario_a, seed=s, fold=f, scores={AUC_KEY: av}, dataset_sha256=dataset_sha256[0],
                      created_utc=created, configuration=configuration)
            _write_run(root, scenario_b, seed=s, fold=f, scores={AUC_KEY: bv}, dataset_sha256=dataset_sha256[1],
                      created_utc=created, configuration=configuration)


# --------------------------------------------------------------------------------- (1) pairing, verdict, D-046 A
def test_pairs_by_seed_and_fold_and_verdicts_a_beats_b_when_the_ci_clears_the_minimum_effect(tmp_path):
    _populate(tmp_path, a_auc=0.60, b_auc=0.50)
    spec = _base_spec(tmp_path, min_effect=0.02)
    result = compare(spec)
    assert result.verdict == "A beats B"
    assert len(result.pairs) == 5
    assert len(result.fold_rows) == 5
    assert result.refusal_reason is None


def test_inconclusive_when_the_ci_straddles_the_minimum_effect(tmp_path):
    _populate(tmp_path, a_auc=0.501, b_auc=0.500, noise=0.02)
    spec = _base_spec(tmp_path, min_effect=0.05)
    assert compare(spec).verdict == "inconclusive"


def test_b_beats_a_when_b_is_the_better_scenario(tmp_path):
    _populate(tmp_path, a_auc=0.50, b_auc=0.60)
    spec = _base_spec(tmp_path, min_effect=0.02)
    assert compare(spec).verdict == "B beats A"


def test_fewer_than_min_folds_is_refused(tmp_path):
    _populate(tmp_path, folds=(-1, -2, -3))          # only 3 judgement folds
    spec = _base_spec(tmp_path, judgment_folds=[-1, -2, -3, -4, -5])
    result = compare(spec)
    assert result.verdict == "refused"
    assert "judgement fold" in result.refusal_reason


def test_min_folds_below_five_is_refused_by_the_spec_itself(tmp_path):
    with pytest.raises(CompareError):
        _base_spec(tmp_path, min_folds=3)


def test_a_fold_the_spec_does_not_name_as_a_judgement_fold_is_excluded_not_paired(tmp_path):
    _populate(tmp_path, folds=(-11, -12, -13, -14, -15))
    spec = _base_spec(tmp_path, judgment_folds=list(FIVE_FOLDS))       # runs are on different folds
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("judgement fold" in e.reason for e in excluded)


def test_a_dataset_fingerprint_mismatch_between_a_and_b_excludes_the_pair(tmp_path):
    _populate(tmp_path, dataset_sha256=("ds-1", "ds-2"))
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("fingerprint mismatch" in e.reason for e in excluded)


def test_fold_aggregation_averages_seeds_within_a_fold_before_the_paired_test_runs_over_folds(tmp_path):
    """D-046 point A: 5 folds x 4 seeds, each fold's seeds sharing a fold-specific bias. If inference
    ran over the 20 raw pairs instead of the 5 fold means, the interval would be far narrower (df=19
    vs df=4) because pooling ignores the shared per-fold bias."""
    rng = np.random.default_rng(3)
    fold_bias = {-1: 0.00, -2: 0.01, -3: -0.01, -4: 0.02, -5: -0.02}
    day = 1
    for f, bias in fold_bias.items():
        for s in range(4):
            av = 0.55 + bias + rng.normal(0, 0.001)
            created = f"202601{day:02d}T000000Z"
            day += 1
            _write_run(tmp_path, "A", seed=s, fold=f, scores={AUC_KEY: av}, created_utc=created)
            _write_run(tmp_path, "B", seed=s, fold=f, scores={AUC_KEY: 0.50}, created_utc=created)
    spec = _base_spec(tmp_path, min_effect=0.001)
    result = compare(spec)
    assert len(result.pairs) == 20
    assert len(result.fold_rows) == 5
    from neural_trade.metrics.statistics import paired_t_ci
    fold_means = [0.05 + b for b in fold_bias.values()]
    m, lo, hi, _t = paired_t_ci(fold_means)
    assert result.estimate["ci_lo"] == pytest.approx(lo, abs=2e-3)
    assert result.estimate["ci_hi"] == pytest.approx(hi, abs=2e-3)
    raw_diffs = [p.diff for p in result.pairs]
    _, wrong_lo, wrong_hi, _ = paired_t_ci(raw_diffs)
    assert (hi - lo) > (wrong_hi - wrong_lo)


# ------------------------------------------------------------------------ (2) pre-registration, spec hash, C
def test_spec_hash_is_recorded_in_the_output(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02)
    out = compare(spec).to_dict()
    assert out["spec_hash"] == spec.spec_hash
    assert len(out["spec_hash"]) == 12


def test_registered_utc_is_required_not_defaulted_to_now(tmp_path):
    base = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01,
               judgment_folds=list(FIVE_FOLDS), root=str(tmp_path))
    with pytest.raises(CompareError):
        CompareSpec.from_dict(base)


@pytest.mark.parametrize("fmt", ["20260201T000000Z", "2026-02-01T00:00:00Z"])
def test_a_spec_registered_after_a_compared_run_started_is_refused_in_either_timestamp_format(tmp_path, fmt):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)          # runs created in January 2026
    spec = _base_spec(tmp_path, min_effect=0.02, registered_utc=fmt)
    result = compare(spec)
    assert result.verdict == "refused"
    assert "before the spec's effective registration" in result.refusal_reason


@pytest.mark.parametrize("fmt", ["20251201T000000Z", "2025-12-01T00:00:00Z"])
def test_a_spec_registered_before_the_runs_compares_normally_in_either_format(tmp_path, fmt):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02, registered_utc=fmt)
    assert compare(spec).verdict == "A beats B"


def test_an_unparsable_timestamp_is_refused_at_construction(tmp_path):
    with pytest.raises(CompareError):
        _base_spec(tmp_path, registered_utc="not-a-timestamp")


def test_a_pre_registered_pair_count_is_enforced_no_peeking(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, name="peek1", min_effect=0.02, pairs_planned=4)
    result = compare(spec)
    assert result.verdict == "refused"
    assert "no peeking" in result.refusal_reason
    spec_ok = _base_spec(tmp_path, name="peek2", min_effect=0.02, pairs_planned=5)
    assert compare(spec_ok).verdict == "A beats B"


def test_pairs_planned_is_enforced_even_when_looks_is_greater_than_one(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02, looks=2, look_index=2, pairs_planned=[3, 4])
    result = compare(spec)
    assert result.verdict == "refused"
    assert "no peeking" in result.refusal_reason


def test_the_spec_hash_is_recorded_at_first_use_and_a_later_edit_is_refused(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec1 = _base_spec(tmp_path, name="locked", min_effect=0.02)
    r1 = compare(spec1)
    assert r1.verdict == "A beats B"
    reg_path = tmp_path / "compares" / "locked" / "registration.json"
    assert reg_path.exists()
    spec2 = _base_spec(tmp_path, name="locked", min_effect=0.0)          # edited, same registered_utc
    r2 = compare(spec2)
    assert r2.verdict == "refused"
    assert "changed since it was registered" in r2.refusal_reason


def test_git_commit_time_of_the_spec_file_is_preferred_over_the_declared_string(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    d = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01,
            judgment_folds=list(FIVE_FOLDS), registered_utc="19990101T000000Z", root=str(tmp_path))
    path = repo / "spec.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    subprocess.run(["git", "add", "spec.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "spec"], cwd=repo, check=True)
    spec = CompareSpec.from_yaml(path)
    assert spec.registered_utc_source == "git_commit_time"
    assert spec.registered_utc == "19990101T000000Z"          # the declared string is untouched
    assert spec.effective_registered_utc != "19990101T000000Z"


# --------------------------------------------------------------------------------------- (3) guard-rails
def test_a_guard_rail_breach_is_judged_by_the_same_paired_test(tmp_path):
    rng = np.random.default_rng(1)
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.60, "h1/variance/crpss": 0.01 + rng.normal(0, 0.0005)})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.50, "h1/variance/crpss": 0.05 + rng.normal(0, 0.0005)})
    spec = _base_spec(tmp_path, min_effect=0.02, guard_rails=[
        {"metric": "h1/variance/crpss", "direction": "higher_better", "max_degradation": 0.01}])
    result = compare(spec)
    assert result.verdict == "A beats B"
    assert result.guard_rail_results[0]["verdict"] == "breach"


def test_a_guard_rail_within_tolerance_passes(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.60, "h1/variance/crpss": 0.050})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.50, "h1/variance/crpss": 0.051})
    spec = _base_spec(tmp_path, min_effect=0.02, guard_rails=[
        {"metric": "h1/variance/crpss", "direction": "higher_better", "max_degradation": 0.01}])
    assert compare(spec).guard_rail_results[0]["verdict"] == "pass"


# ----------------------------------------------------------------- (4) simulated error rates, D-046 B
def test_the_null_false_beats_rate_is_at_most_five_percent_plus_monte_carlo_error(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02, block_sd=0.02, seed_sd=0.0, pairs_planned=5)
    sim = simulate_error_rates(spec, n_folds=5, seeds_per_fold=1, n_sim=1000, seed=0)
    assert sim["n_sim"] >= 1000
    assert sim["false_beats_rate"] <= 0.05 + 3 * sim["false_beats_mc_error"]
    assert 0.0 <= sim["power_at_2x_min_effect"] <= 1.0


def test_the_simulation_is_seeded_and_reproducible(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02, noise_sd=0.02)
    a = simulate_error_rates(spec, n_folds=8, n_sim=1000, seed=7)
    b = simulate_error_rates(spec, n_folds=8, n_sim=1000, seed=7)
    assert a == b


def test_simulation_needs_a_variance_component_or_it_is_refused(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02)
    with pytest.raises(CompareError):
        simulate_error_rates(spec, n_folds=8)


def test_simulation_refuses_fewer_folds_than_min_folds(tmp_path):
    spec = _base_spec(tmp_path, min_effect=0.02, noise_sd=0.02, min_folds=5)
    with pytest.raises(CompareError):
        simulate_error_rates(spec, n_folds=3)


def test_one_fold_many_seeds_inflates_the_false_beats_rate_at_the_pair_level_but_the_design_needs_five_folds(tmp_path):
    """D-046 point B: the regression test. 1 judgement fold x 5 seeds sharing 'block' noise: computing
    the verdict over the 5 raw (pair-level) diffs, ignoring that they share one fold's noise, exceeds
    the nominal 5% false-'beats' rate -- exactly the mechanism the fold-level design (min_folds >= 5)
    rules out by construction (a single fold can never reach a verdict, whatever its seed count)."""
    rng = np.random.default_rng(11)
    min_effect = 0.01
    n_sim = 3000
    beats = 0
    for _ in range(n_sim):
        block = rng.normal(0, 0.01)                       # shared by every seed on this one fold
        pair_diffs = block + rng.normal(0, 0.01, size=5)   # 5 seeds, WRONGLY treated as independent pairs
        est = _estimate(pair_diffs, estimator="mean", alpha=0.05)
        v = _verdict_from_ci(est["ci_lo"], est["ci_hi"], min_effect, "A", "B")
        beats += v != "inconclusive"
    pair_level_false_rate = beats / n_sim
    assert pair_level_false_rate > 0.05          # demonstrates the bug the fold-level design avoids

    # the fold-level design simply cannot produce a verdict from 1 fold: min_folds >= 5 refuses it
    spec = _base_spec(tmp_path, min_effect=min_effect, min_folds=5, block_sd=0.01, seed_sd=0.01)
    with pytest.raises(CompareError):
        simulate_error_rates(spec, n_folds=1, seeds_per_fold=5)

    # with 5 independent folds (1 seed each, seed_sd irrelevant), the false rate is controlled
    sim5 = simulate_error_rates(spec, n_folds=5, seeds_per_fold=1, n_sim=n_sim, seed=1)
    assert sim5["false_beats_rate"] <= 0.05 + 4 * sim5["false_beats_mc_error"]


def test_min_effect_zero_changes_the_false_beats_rate_with_alpha_catching_a_wrong_level_interval(tmp_path):
    """A test whose outcome depends on alpha: at min_effect 0, 'beats' triggers whenever the CI excludes
    0, so the false rate should track alpha itself (about alpha, not some other fixed number) -- a
    comparator that used the wrong confidence level (e.g. quartered by double-halving, D-046 point G)
    would show a false rate far from the alpha it claims."""
    spec_05 = _base_spec(tmp_path, min_effect=0.0, noise_sd=0.02, alpha=0.05)
    spec_50 = _base_spec(tmp_path, min_effect=0.0, noise_sd=0.02, alpha=0.50)
    sim_05 = simulate_error_rates(spec_05, n_folds=10, n_sim=4000, seed=2)
    sim_50 = simulate_error_rates(spec_50, n_folds=10, n_sim=4000, seed=2)
    assert sim_05["false_beats_rate"] == pytest.approx(0.05, abs=0.02)
    assert sim_50["false_beats_rate"] == pytest.approx(0.50, abs=0.05)
    assert sim_50["false_beats_rate"] > sim_05["false_beats_rate"]


def test_seed_and_block_components_combine_as_advertised(tmp_path):
    spec_split = _base_spec(tmp_path, min_effect=0.02, seed_sd=0.015, block_sd=0.02)
    sim_split = simulate_error_rates(spec_split, n_folds=10, seeds_per_fold=3, seed=3)
    assert sim_split["noise_sd"] == pytest.approx(np.sqrt(0.02 ** 2 + 0.015 ** 2 / 3))


def test_pocock_alpha_is_passed_two_sided_once_not_halved_twice(tmp_path):
    """D-046 point G: alpha_used fed into the (two-sided) t/HL interval must be the Pocock per-look
    value at its OWN two-sided level, not further halved (which would quarter it overall)."""
    spec_1look = _base_spec(tmp_path, min_effect=0.0, noise_sd=0.02, alpha=0.05, looks=1)
    spec_2look = _base_spec(tmp_path, min_effect=0.0, noise_sd=0.02, alpha=0.05, looks=2, look_index=2)
    sim1 = simulate_error_rates(spec_1look, n_folds=10, n_sim=4000, seed=4)
    sim2 = simulate_error_rates(spec_2look, n_folds=10, n_sim=4000, seed=4)
    assert sim1["alpha_used"] == pytest.approx(0.05)
    assert sim2["alpha_used"] == pytest.approx(pocock_alpha(0.05, 2, one_sided=False))
    assert sim2["alpha_used"] < sim1["alpha_used"]              # a two-look design is stricter per look
    # at min_effect 0, the false-'beats' rate tracks alpha directly: a double-halved (quartered)
    # alpha_used (~0.0147) would show up as a false rate far below the correct ~0.0294
    assert sim1["false_beats_rate"] == pytest.approx(0.05, abs=0.02)
    assert sim2["false_beats_rate"] == pytest.approx(pocock_alpha(0.05, 2, one_sided=False), abs=0.015)


# ----------------------------------------------------------------------------------- (5) JSON + markdown
def test_output_is_json_serialisable_and_the_markdown_names_pairs_metric_effect_and_verdict(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
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
    _populate(tmp_path, folds=(-1, -2))
    spec = _base_spec(tmp_path)
    md = compare(spec).to_markdown()
    assert "Refused" in md


# ------------------------------------------------------------------------ spec parsing / CLI-facing bits
def test_from_yaml_round_trips_and_rejects_unknown_keys(tmp_path):
    d = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01,
            judgment_folds=list(FIVE_FOLDS), registered_utc="20260101T000000Z", root=str(tmp_path))
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    spec = CompareSpec.from_yaml(path)
    assert spec.name == "t" and spec.judgment_folds == tuple(FIVE_FOLDS)

    bad = dict(d, not_a_field=1)
    path2 = tmp_path / "bad.yaml"
    path2.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(CompareError):
        CompareSpec.from_yaml(path2)


def test_an_invalid_direction_or_metric_kind_is_refused(tmp_path):
    base = dict(name="t", scenario_a="A", scenario_b="B", metric="m", min_effect=0.01,
               judgment_folds=list(FIVE_FOLDS), registered_utc="20260101T000000Z")
    with pytest.raises(CompareError):
        CompareSpec.from_dict({**base, "direction": "sideways"})
    with pytest.raises(CompareError):
        CompareSpec.from_dict({**base, "metric_kind": "ratio"})


# -------------------------------------------------------------------------------- D-037 building blocks
def test_log_ratio_metric_kind_uses_the_log_of_the_ratio(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={"m": 1.10})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={"m": 1.00})
    spec = _base_spec(tmp_path, metric="m", min_effect=0.05, metric_kind="log_ratio")
    result = compare(spec)
    assert result.pairs[0].diff == pytest.approx(np.log(1.10 / 1.00), abs=1e-9)
    assert result.verdict == "A beats B"


def test_a_non_finite_log_ratio_pair_is_excluded_not_silently_zeroed(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={"m": 1.10})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={"m": 0.0})
    spec = _base_spec(tmp_path, metric="m", min_effect=0.05, metric_kind="log_ratio")
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("non-finite" in e.reason for e in excluded)


def test_hodges_lehmann_estimator_is_robust_to_one_outlier_pair(tmp_path):
    vals_a = [0.55] * 4 + [0.90]        # one wild outlier fold
    for i, (f, av) in enumerate(zip(FIVE_FOLDS, vals_a)):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: av})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec_mean = _base_spec(tmp_path, name="outlier_mean", min_effect=0.02, estimator="mean")
    spec_hl = _base_spec(tmp_path, name="outlier_hl", min_effect=0.02, estimator="hodges_lehmann")
    r_mean = compare(spec_mean)
    r_hl = compare(spec_hl)
    assert "note" in r_hl.estimate            # n=5 folds: no exact 95% Wilcoxon CI, falls back to t (D-046 G)
    assert r_hl.estimate["estimate"] < r_mean.estimate["estimate"]


def test_hodges_lehmann_falls_back_to_the_t_interval_when_no_exact_wilcoxon_ci_exists_at_n5(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.55 + 0.001 * i})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path, min_effect=0.02, estimator="hodges_lehmann")
    result = compare(spec)
    assert len(result.fold_rows) == 5
    assert "note" in result.estimate
    assert "fell back to the t interval" in result.estimate["note"]


def test_hodges_lehmann_and_wilcoxon_ci_on_a_symmetric_sample_at_n6_are_consistent():
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    est = hodges_lehmann(d)
    lo, hi = wilcoxon_hl_ci(d, alpha=0.05)
    assert lo <= est <= hi


def test_pocock_alpha_two_looks_is_about_0_03_one_sided():
    assert pocock_alpha(0.05, 2) == pytest.approx(0.0294 / 2, abs=1e-4)
    assert pocock_alpha(0.05, 1) == 0.025
    assert pocock_alpha(0.05, 2, one_sided=False) == pytest.approx(0.0294, abs=1e-4)


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


def test_a_coverage_style_metric_compares_like_any_other_paired_metric(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={"h1/variance/coverage": 0.905})
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={"h1/variance/coverage": 0.870})
    spec = _base_spec(tmp_path, metric="h1/variance/coverage", min_effect=0.01)
    assert compare(spec).verdict == "A beats B"


# ------------------------------------------------------------------------------------ D-046 point D: fingerprint
def test_a_moved_test_block_between_a_and_b_excludes_the_pair(tmp_path):
    _populate(tmp_path)
    for d in (tmp_path / "scenarios" / "B").iterdir():
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["engine"]["fold"] == -1:
            meta["blocks"]["test"] = {"start": 999999, "stop": 1043199}
            (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    moved = [e for e in excluded if e.fold == -1]
    assert moved and all("fingerprint mismatch" in e.reason and "judged_block" in e.reason for e in moved)
    assert len(pairs) == 4                    # the other 4 folds still pair


def test_a_lookback_mismatch_excludes_the_pair(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.6}, lookback=60)
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.5}, lookback=240)
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("lookback" in e.reason for e in excluded)


# --------------------------------------------------------------------------------- D-046 point E: configuration
def test_ambiguous_mixed_configurations_are_excluded_not_silently_resolved(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.55}, configuration="default")
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202602{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.75}, configuration="other", run_id=f"other-{i}")
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("multiple configurations" in e.reason for e in excluded)


def test_naming_the_configuration_resolves_the_ambiguity(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.55}, configuration="default")
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202602{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.75}, configuration="other", run_id=f"other-{i}")
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path, configuration_a="default", min_effect=0.02)
    result = compare(spec)
    assert result.verdict == "A beats B"
    assert all(p.a_value == pytest.approx(0.55) for p in result.pairs)


def test_a_duplicate_done_run_for_the_same_configuration_is_refused_by_default(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.55}, run_id=f"a1-{i}")
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202602{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.56}, run_id=f"a2-{i}")
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path)
    pairs, excluded = pair_runs(spec)
    assert pairs == []
    assert all("done runs for the same" in e.reason for e in excluded)


def test_duplicate_policy_latest_picks_the_most_recent_done_run(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.55}, run_id=f"a1-{i}")
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202602{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.90}, run_id=f"a2-{i}")
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path, duplicate_policy="latest", min_effect=0.02)
    result = compare(spec)
    assert all(p.a_value == pytest.approx(0.90) for p in result.pairs)


def test_duplicate_policy_average_averages_the_metric(tmp_path):
    for i, f in enumerate(FIVE_FOLDS):
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.50}, run_id=f"a1-{i}")
        _write_run(tmp_path, "A", seed=0, fold=f, created_utc=f"202602{i + 1:02d}T000000Z",
                  scores={AUC_KEY: 0.60}, run_id=f"a2-{i}")
        _write_run(tmp_path, "B", seed=0, fold=f, created_utc=f"202601{i + 1:02d}T000000Z", scores={AUC_KEY: 0.50})
    spec = _base_spec(tmp_path, duplicate_policy="average", min_effect=0.02)
    result = compare(spec)
    assert all(p.a_value == pytest.approx(0.55) for p in result.pairs)


# ---------------------------------------------------------------------------- D-046 point F: read-only index
def test_compare_never_writes_the_sqlite_index(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    index_path = tmp_path / "index.sqlite"
    assert not index_path.exists()
    spec = _base_spec(tmp_path, min_effect=0.02)
    compare(spec)
    assert not index_path.exists()             # compare() must never create or touch it


def test_pair_runs_never_calls_the_run_store_index(tmp_path):
    _populate(tmp_path, a_auc=0.6, b_auc=0.5)
    spec = _base_spec(tmp_path, min_effect=0.02)
    pair_runs(spec)
    assert not (tmp_path / "index.sqlite").exists()


def test_scenario_run_dirs_are_still_readable_by_the_ordinary_run_store(tmp_path):
    _populate(tmp_path, a_auc=0.55, b_auc=0.50)
    store = RunStore(tmp_path)
    assert len(store.run_dirs("A")) == 5
