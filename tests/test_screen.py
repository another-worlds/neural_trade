"""Screen mode (NT-088): the spec, trial generation, sharding, resumability, data caching, the
pre-registered rules, and DATA_END slicing / the protected-span refusal.

Fast tests avoid real training (``_load_cached`` is spied on directly, ``run_trial`` is given a
stub trainer); one real CPU training test (marked ``slow``) checks that a known-bad config (LR=1.0)
fails the pre-registered rules on real training, per NT-088's acceptance criteria.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor, apply_data_end
from neural_trade.experiments.screen import (ScreenError, ScreenSpec, _load_cached, apply_rules, build_trials,
                                             parse_shard, run_screen, run_trial, shard_of)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("screen_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def spec_dict(csv, **changes):
    s = {"schema_version": 1, "name": "tiny_screen", "description": "test",
         "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2,
                       "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1},
         "grid": {"axes": {"BATCH_SIZE": [32, 64]}}, "slices": [None], "seeds": [0, 1],
         "run": {"calibrate": False, "epochs": 1}, "rules": {"finite": True}}
    s.update(changes)
    return s


def _stub_trainer(cfg, cache, *, calibrate):
    """A fast stand-in for ``_run_trial_light``: deterministic health/AUC/timings, no TensorFlow."""
    rng = np.random.default_rng(int(cfg.SEED))
    health = {"finite": True, "nonfinite_grad_steps": 0, "grad_global_norm_max": 1.0,
             "grad_global_norm_mean": 0.5, "clipped_share": 0.0, "n_logged_steps": 10,
             "train_loss_drop": float(rng.uniform(0.0, 0.2)), "final_train_loss": 1.0,
             "final_val_loss": 1.1, "loss_term_shares": {"point_loss": 0.5}, "max_term_share": 0.5,
             "epochs_run": 1}
    auc = {"h0": {"auc": 0.5, "n": 10, "n_eff": 2}}
    timings = {"load_s": 0.001, "build_s": 0.001, "train_s": 0.001, "score_s": 0.001}
    return health, auc, timings


# ------------------------------------------------------------------ (1) the spec
def test_spec_refuses_unknown_top_level_key(bars_csv):
    with pytest.raises(ScreenError, match="unknown key"):
        ScreenSpec.from_dict({**spec_dict(bars_csv), "bogus": 1})


def test_spec_refuses_unknown_key_in_grid_sample_run_rules(bars_csv):
    for bad in ({"grid": {"bogus": {}}}, {"sample": {"n": 1, "method": "random", "seed": 0, "space": {},
                                                     "bogus": 1}},
                {"run": {"bogus": True}}, {"rules": {"bogus": 1}}):
        with pytest.raises(ScreenError, match="unknown key"):
            ScreenSpec.from_dict({**spec_dict(bars_csv), **bad})


def test_spec_refuses_unknown_config_field_in_overrides_axes_and_sample_space(bars_csv):
    with pytest.raises(ScreenError, match="unknown Config field"):
        build_trials(ScreenSpec.from_dict({**spec_dict(bars_csv), "overrides":
                                           {**spec_dict(bars_csv)["overrides"], "NOT_A_FIELD": 1}}))
    with pytest.raises(ScreenError, match="unknown Config field"):
        build_trials(ScreenSpec.from_dict({**spec_dict(bars_csv), "grid": {"axes": {"NOT_A_FIELD": [1, 2]}}}))
    s = spec_dict(bars_csv)
    s["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"NOT_A_FIELD": {"low": 0.0, "high": 1.0}}}
    with pytest.raises(ScreenError, match="unknown Config field"):
        build_trials(ScreenSpec.from_dict(s))


def test_sample_space_needs_explicit_bounds(bars_csv):
    s = spec_dict(bars_csv)
    s["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"LR": {"low": 0.0001}}}
    with pytest.raises(ScreenError, match="low.*high"):
        ScreenSpec.from_dict(s)


# ------------------------------------------------------------------ (2) trial generation
def test_grid_x_slices_x_seeds_gives_the_expected_count_with_unique_keys(bars_csv):
    s = spec_dict(bars_csv, slices=["2020-01-01T00:00:00", "2020-02-01T00:00:00"])
    trials = build_trials(ScreenSpec.from_dict(s))
    assert len(trials) == 2 * 2 * 2   # 2 BATCH_SIZE values x 2 slices x 2 seeds
    assert len({t.key for t in trials}) == len(trials)
    assert {t.config.BATCH_SIZE for t in trials} == {32, 64}
    assert {t.config.SEED for t in trials} == {0, 1}
    assert {t.config.DATA_END for t in trials} == {"2020-01-01T00:00:00", "2020-02-01T00:00:00"}


def test_grid_and_sample_points_are_concatenated_not_crossed(bars_csv):
    s = spec_dict(bars_csv, slices=[None], seeds=[0])
    s["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"LR": {"low": 0.0001, "high": 0.01}}}
    trials = build_trials(ScreenSpec.from_dict(s))
    # 2 grid points (BATCH_SIZE axis) + 3 sample points, x 1 slice x 1 seed
    assert len(trials) == 2 + 3
    assert sum(t.source == "grid" for t in trials) == 2
    assert sum(t.source == "sample" for t in trials) == 3


def test_lhs_sampling_is_deterministic_and_stays_in_bounds(bars_csv):
    s = spec_dict(bars_csv, slices=[None], seeds=[0])
    s["sample"] = {"n": 8, "method": "lhs", "seed": 7, "space": {"LR": {"low": 1e-4, "high": 1e-2, "log": True}}}
    t1 = build_trials(ScreenSpec.from_dict(s))
    t2 = build_trials(ScreenSpec.from_dict(s))
    lr1 = sorted(t.config.LR for t in t1 if t.source == "sample")
    lr2 = sorted(t.config.LR for t in t2 if t.source == "sample")
    assert lr1 == lr2                                   # same seed -> same draw
    assert all(1e-4 <= v <= 1e-2 for v in lr1)


# ------------------------------------------------------------------ (3) sharding
def test_shard_partition_is_disjoint_and_its_union_is_every_trial(bars_csv):
    s = spec_dict(bars_csv, slices=["2020-01-01T00:00:00", "2020-02-01T00:00:00", "2020-03-01T00:00:00"],
                 seeds=[0, 1, 2])
    trials = build_trials(ScreenSpec.from_dict(s))
    n = 3
    shards = [{t.index for t in trials if shard_of(t.index, parse_shard(f"{i}/{n}"))} for i in range(n)]
    union = set().union(*shards)
    assert union == {t.index for t in trials}
    for a in range(n):
        for b in range(a + 1, n):
            assert not (shards[a] & shards[b])


def test_parse_shard_rejects_bad_syntax():
    with pytest.raises(ScreenError):
        parse_shard("not-a-shard")
    with pytest.raises(ScreenError):
        parse_shard("3/2")   # i must be < N


# ------------------------------------------------------------------ (4) data caching
def test_load_cached_builds_windows_once_per_data_key_per_process(monkeypatch, bars_csv):
    s = spec_dict(bars_csv, slices=[None], seeds=[0, 1, 2])
    s["grid"] = {"axes": {"LR": [1e-4, 1e-3, 1e-2], "BATCH_SIZE": [32, 64]}}   # varies fields NOT in data_key
    trials = build_trials(ScreenSpec.from_dict(s))
    assert len(trials) == 3 * 2 * 3   # 6 configurations x 1 slice x 3 seeds, one data key

    calls = {"n": 0}
    original = DataProcessor.load_and_prepare_data

    def counting(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(DataProcessor, "load_and_prepare_data", counting)
    cache = {}
    for t in trials:
        _load_cached(t.config, cache)
    assert calls["n"] == 1                # one data key: LR/BATCH_SIZE/SEED never change it
    assert len(cache) == 1


def test_load_cached_builds_windows_once_per_distinct_data_key(monkeypatch, bars_csv):
    s = spec_dict(bars_csv, slices=["2025-10-11 05:00:00+00:00", "2025-10-11 10:00:00+00:00"], seeds=[0])
    s["overrides"]["DATA_END_PROTECTED_DAYS"] = 0.0
    trials = build_trials(ScreenSpec.from_dict(s))

    calls = {"n": 0}
    original = DataProcessor.load_and_prepare_data

    def counting(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(DataProcessor, "load_and_prepare_data", counting)
    cache = {}
    for t in trials:
        _load_cached(t.config, cache)
    assert calls["n"] == 2                # two distinct DATA_END values -> two distinct data keys
    assert len(cache) == 2


# ------------------------------------------------------------------ (5) DATA_END slicing / protected span
def test_data_end_slices_before_max_sequence_count_trims(synthetic_bars, bars_csv):
    cfg = Config(CSV_PATH=str(bars_csv), DATA_END_PROTECTED_DAYS=0.5)
    ts = synthetic_bars["datetime"]
    cutoff = ts.iloc[1500]
    cfg.override(DATA_END=str(cutoff))
    df, _ = DataProcessor(cfg).load_and_prepare_data()
    assert df["timestamp"].max() <= cutoff
    assert len(df) <= 1501


def test_data_end_within_the_protected_span_is_refused(synthetic_bars, bars_csv):
    ts = synthetic_bars["datetime"]
    last = ts.iloc[-1]
    cfg = Config(CSV_PATH=str(bars_csv), DATA_END_PROTECTED_DAYS=1.0)
    near_the_end = last - (last - ts.iloc[0]) * 0.1   # well inside the last 1 day of a ~2-day file
    cfg.override(DATA_END=str(near_the_end))
    with pytest.raises(ValueError, match="protected dev/test span"):
        DataProcessor(cfg).load_and_prepare_data()


def test_data_end_before_the_protected_span_is_allowed(synthetic_bars, bars_csv):
    ts = synthetic_bars["datetime"]
    cfg = Config(CSV_PATH=str(bars_csv), DATA_END_PROTECTED_DAYS=1.0)
    early = ts.iloc[500]   # well before the last 1 day of this ~2-day file
    cfg.override(DATA_END=str(early))
    df, _ = DataProcessor(cfg).load_and_prepare_data()
    assert len(df) >= 62


def test_apply_data_end_is_a_no_op_when_unset(synthetic_bars):
    cfg = Config()
    out = apply_data_end(synthetic_bars, cfg)
    assert out is synthetic_bars


# ------------------------------------------------------------------ (6) rules
def test_apply_rules_pass_and_fail_with_reasons():
    healthy = {"finite": True, "nonfinite_grad_steps": 0, "clipped_share": 0.1, "train_loss_drop": 0.2,
              "max_term_share": 0.4}
    passed, reasons = apply_rules(healthy, {"finite": True, "max_nonfinite_grad_steps": 0,
                                            "max_clipped_share": 0.5, "min_train_loss_drop": 0.01,
                                            "max_term_share": 0.9})
    assert passed and not reasons

    broken = {"finite": False, "nonfinite_grad_steps": 5, "clipped_share": 0.9, "train_loss_drop": -0.1,
             "max_term_share": 0.99}
    passed, reasons = apply_rules(broken, {"finite": True, "max_nonfinite_grad_steps": 0,
                                           "max_clipped_share": 0.5, "min_train_loss_drop": 0.01,
                                           "max_term_share": 0.9})
    assert not passed
    assert len(reasons) == 5  # non-finite value, nonfinite_grad_steps, clipped_share, loss drop, term share


# ------------------------------------------------------------------ (7) resumability
def test_resuming_a_screen_skips_trials_already_in_results_jsonl(tmp_path, bars_csv):
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[None], seeds=[0, 1]))
    store = tmp_path / "runs"
    r1 = run_screen(spec, store=store, trainer=_stub_trainer)
    n = r1.n_trials
    assert r1.ran == n and r1.skipped == 0

    r2 = run_screen(spec, store=store, trainer=_stub_trainer)
    assert r2.ran == 0 and r2.skipped == n

    lines = (store / "screens" / spec.name / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n   # no duplicate rows on the resumed run
    keys = [json.loads(line)["trial_key"] for line in lines]
    assert len(set(keys)) == n


def test_max_trials_stops_early_and_the_same_command_resumes(tmp_path, bars_csv):
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[None], seeds=[0, 1]))
    store = tmp_path / "runs"
    r1 = run_screen(spec, store=store, max_trials=1, trainer=_stub_trainer)
    assert r1.ran == 1
    r2 = run_screen(spec, store=store, trainer=_stub_trainer)
    assert r2.skipped == 1
    assert r2.ran == r1.n_trials - 1


def test_run_trial_writes_a_pass_fail_row_with_reasons(bars_csv):
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[None], seeds=[0]))
    trial = build_trials(spec)[0]
    spec.rules.update(max_nonfinite_grad_steps=0)
    row = run_trial(trial, spec, {}, trainer=_stub_trainer)
    assert row["trial_key"] == trial.key
    assert row["passed"] is True and row["reasons"] == []
    assert set(row["timings"]) == {"load_s", "build_s", "train_s", "score_s"}


# ------------------------------------------------------------------ (8) the one real CPU training test
@pytest.mark.slow
def test_a_known_bad_learning_rate_fails_the_rules_on_real_cpu_training(bars_csv):
    """LR=1.0 (the top of its valid range) on real training: a known-bad config must fail the
    pre-registered rules (real training, not the fake trainer, per NT-088's acceptance criteria)."""
    s = spec_dict(bars_csv, slices=[None], seeds=[0])
    s["overrides"].update(LR=1.0, BATCH_SIZE=32)
    s["grid"] = {"axes": {}}
    s["rules"] = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 0.3,
                 "min_train_loss_drop": 0.0, "max_term_share": 0.9}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {})   # the real light path: no trainer override
    assert row["passed"] is False, f"a known-bad LR should fail the rules: {row['reasons']} / {row['health']}"
    assert row["reasons"]
