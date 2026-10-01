"""Screen mode (NT-088): the spec, trial generation, sharding, resumability, data (and windowing)
caching, the pre-registered rules (with lambda-weighted term shares), non-finite handling, DATA_END
slicing / the protected-span preflight, and the timing breakdown.

Fast tests avoid real training (``_load_cached``/``_windowed_cached`` are spied on directly,
``run_trial`` is given a stub trainer); a handful of real CPU training tests (marked ``slow``) check
that known-bad configs (LR=1.0, LAMBDA_DIR=1e6, an extreme LAMBDA_POINT) fail the pre-registered
rules, or are recorded and not re-attempted, on real training.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
import tensorflow as tf
from neural_trade.data.processor import DataProcessor, apply_data_end
from neural_trade.experiments.dataset import data_layout
from neural_trade.experiments.screen import (ScreenError, ScreenSpec, _load_cached, _sanitize_nonfinite,
                                             _windowed_cached, apply_rules, build_trials, merge_results,
                                             parse_shard, run_screen, run_trial, shard_of)

# The synthetic_bars fixture (tests/conftest.py) is a fixed 3,000-minute series starting
# 2025-10-11T02:30:00Z: a safe, always-in-range, always-far-from-its-own-tail DATA_END and a matching
# tiny protected span, for tests that do not care about DATA_END/protected-span semantics themselves.
# NT-088 round 2's preflight (D-020) now refuses the implicit "use the newest data" case exactly as it
# would an explicit one, since the newest data trivially sits inside the file's own protected span for
# any DATA_END_PROTECTED_DAYS >= 0 - so every screen spec here needs an explicit, safely-old DATA_END.
SAFE_DATA_END = "2025-10-13T04:19:00+00:00"      # 10 minutes before the fixture's last bar
SAFE_PROTECTED_DAYS = 0.001                       # ~86 seconds: SAFE_DATA_END sits safely before it


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("screen_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def spec_dict(csv, **changes):
    s = {"schema_version": 1, "name": "tiny_screen", "description": "test",
         "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2,
                       "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1, "DATA_END_PROTECTED_DAYS": SAFE_PROTECTED_DAYS},
         "grid": {"axes": {"BATCH_SIZE": [32, 64]}}, "slices": [SAFE_DATA_END], "seeds": [0, 1],
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
    timings = {"load_s": 0.001, "prep_s": 0.001, "build_s": 0.001, "train_s": 0.001, "score_s": 0.001,
              "epoch_s": [0.001]}
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


# ------------------------------------------------------------------ (1b) spec-load-time bounds (P3)
def test_sample_space_log_with_nonpositive_low_is_refused_before_any_trial_runs(bars_csv):
    s = spec_dict(bars_csv)
    s["sample"] = {"n": 3, "method": "random", "seed": 0,
                   "space": {"LAMBDA_HD": {"low": 0.0, "high": 1.0, "log": True}}}
    with pytest.raises(ScreenError, match="log sampling needs low > 0"):
        build_trials(ScreenSpec.from_dict(s))


def test_sample_space_bounds_outside_the_field_range_are_refused_before_any_trial_runs(bars_csv):
    s = spec_dict(bars_csv)
    s["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"LR": {"low": 0.5, "high": 2.0}}}  # LR <= 1.0
    with pytest.raises(ScreenError, match="outside"):
        build_trials(ScreenSpec.from_dict(s))
    s2 = spec_dict(bars_csv)
    s2["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"LAMBDA_HD": {"low": -1.0, "high": 1.0}}}
    with pytest.raises(ScreenError, match="outside"):
        build_trials(ScreenSpec.from_dict(s2))


# ------------------------------------------------------------------ (2) trial generation
def test_grid_x_slices_x_seeds_gives_the_expected_count_with_unique_keys(bars_csv):
    s = spec_dict(bars_csv, slices=["2025-10-11T06:00:00+00:00", "2025-10-12T00:00:00+00:00"])
    trials = build_trials(ScreenSpec.from_dict(s))
    assert len(trials) == 2 * 2 * 2   # 2 BATCH_SIZE values x 2 slices x 2 seeds
    assert len({t.key for t in trials}) == len(trials)
    assert {t.config.BATCH_SIZE for t in trials} == {32, 64}
    assert {t.config.SEED for t in trials} == {0, 1}
    assert {t.config.DATA_END for t in trials} == {"2025-10-11T06:00:00+00:00", "2025-10-12T00:00:00+00:00"}


def test_grid_and_sample_points_are_concatenated_not_crossed(bars_csv):
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["sample"] = {"n": 3, "method": "random", "seed": 0, "space": {"LR": {"low": 0.0001, "high": 0.01}}}
    trials = build_trials(ScreenSpec.from_dict(s))
    # 2 grid points (BATCH_SIZE axis) + 3 sample points, x 1 slice x 1 seed
    assert len(trials) == 2 + 3
    assert sum(t.source == "grid" for t in trials) == 2
    assert sum(t.source == "sample" for t in trials) == 3


def test_lhs_sampling_is_deterministic_and_stays_in_bounds(bars_csv):
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["sample"] = {"n": 8, "method": "lhs", "seed": 7, "space": {"LR": {"low": 1e-4, "high": 1e-2, "log": True}}}
    t1 = build_trials(ScreenSpec.from_dict(s))
    t2 = build_trials(ScreenSpec.from_dict(s))
    lr1 = sorted(t.config.LR for t in t1 if t.source == "sample")
    lr2 = sorted(t.config.LR for t in t2 if t.source == "sample")
    assert lr1 == lr2                                   # same seed -> same draw
    assert all(1e-4 <= v <= 1e-2 for v in lr1)


# ------------------------------------------------------------------ (3) sharding
def test_shard_partition_is_disjoint_and_its_union_is_every_trial(bars_csv):
    s = spec_dict(bars_csv, slices=["2025-10-11T06:00:00+00:00", "2025-10-11T12:00:00+00:00",
                                    "2025-10-12T00:00:00+00:00"], seeds=[0, 1, 2])
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


def test_shard_runs_write_separate_files_and_merge_reconstructs_the_union(tmp_path, bars_csv):
    """NT-088 round 2, finding 6: concurrent --shard processes must never append to the same file."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END, "2025-10-11T06:00:00+00:00"], seeds=[0, 1])
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "runs"
    all_trials = build_trials(spec)
    n = 3
    for i in range(n):
        r = run_screen(spec, store=store, shard=parse_shard(f"{i}/{n}"), trainer=_stub_trainer)
        assert r.results_path.endswith(f"results.shard-{i}-of-{n}.jsonl")
    out_dir = store / "screens" / spec.name
    shard_files = sorted(out_dir.glob("results.shard-*-of-*.jsonl"))
    assert len(shard_files) == n
    assert not (out_dir / "results.jsonl").exists()   # never the plain, shared file
    merged = merge_results(store=store, name=spec.name, n=n)
    assert {row["trial_key"] for row in merged} == {t.key for t in all_trials}


def test_run_screen_with_shard_also_checks_other_shard_files_for_the_same_trial(tmp_path, bars_csv):
    """Resuming one shard must skip a trial ANY shard already finished, not only its own file."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1])
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "runs"
    n = 2
    trials = build_trials(spec)
    out_dir = store / "screens" / spec.name
    out_dir.mkdir(parents=True)
    shard0_trial = next(t for t in trials if shard_of(t.index, (0, n)))
    fake_row = run_trial(shard0_trial, spec, {}, trainer=_stub_trainer)
    (out_dir / f"results.shard-1-of-{n}.jsonl").write_text(json.dumps(fake_row) + "\n", encoding="utf-8")
    r = run_screen(spec, store=store, shard=parse_shard(f"0/{n}"), trainer=_stub_trainer)
    assert r.skipped == 1   # found via the OTHER shard's file


def test_shard_progress_log_position_never_exceeds_the_shards_own_trial_count(tmp_path, bars_csv):
    """The previous log line showed the trial's GLOBAL index against the SHARD's local trial count
    (a "trial 25/22"-style bug, NT-088 round 2 finding 6); position must never exceed the total.

    A handler is attached directly to the module logger (not ``caplog``): TensorFlow/absl resets the
    root logger's handlers on import, which silently empties ``caplog.records`` for any logger that
    propagates to root."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1, 2, 3, 4, 5])
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "runs"
    n = 3
    records: list = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("neural_trade.experiments.screen")
    handler = _Collect()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        run_screen(spec, store=store, shard=parse_shard(f"2/{n}"), trainer=_stub_trainer)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    lines = [m for m in records if m.startswith(f"[screen {spec.name}] trial ")]
    assert lines
    for line in lines:
        pos_str, total_str = line.split("trial ", 1)[1].split(" ", 1)[0].split("/")
        assert 1 <= int(pos_str) <= int(total_str)


# ------------------------------------------------------------------ (4) data / windowing caching
def test_load_cached_loads_the_raw_data_once_per_data_key_per_process(monkeypatch, bars_csv):
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1, 2])
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


def test_load_cached_loads_the_raw_data_once_per_distinct_data_key(monkeypatch, bars_csv):
    s = spec_dict(bars_csv, slices=["2025-10-11T05:00:00+00:00", "2025-10-11T10:00:00+00:00"], seeds=[0])
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


def test_windowed_cached_builds_windows_exactly_once_per_data_key_per_process(monkeypatch, bars_csv):
    """NT-088 round 2, finding 1 (P1 perf): every trial used to rebuild windows over the whole sliced
    history (``make_sequences_with_extended_trends``, a per-bar loop) even when only a field OUTSIDE
    ``data_key`` (LR, BATCH_SIZE, SEED) varied. ``_windowed_cached`` must build them once per data key
    per process, like ``_load_cached`` already does for the raw load."""
    import neural_trade.data.processor as proc

    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1, 2])
    s["grid"] = {"axes": {"LR": [1e-4, 1e-3, 1e-2], "BATCH_SIZE": [32, 64]}}
    trials = build_trials(ScreenSpec.from_dict(s))
    assert len(trials) == 3 * 2 * 3

    calls = {"n": 0}
    original = proc.make_sequences_with_extended_trends

    def counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(proc, "make_sequences_with_extended_trends", counting)
    cache = {}
    for t in trials:
        _windowed_cached(t.config, cache)
    assert calls["n"] == 1   # exactly one data key -> windows built exactly once


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


def test_implicit_none_data_end_is_checked_against_the_protected_span(bars_csv):
    """NT-088 round 2, finding 3a: an unset DATA_END ("use the newest data") used to skip the
    protected-span check entirely in screen mode. The newest bar is trivially inside the file's own
    protected span for any DATA_END_PROTECTED_DAYS >= 0, so this must now always be refused."""
    s = spec_dict(bars_csv, slices=[None], seeds=[0])
    with pytest.raises(ScreenError, match="protected dev/test span"):
        build_trials(ScreenSpec.from_dict(s))


def test_screen_spec_cannot_lower_protected_days_against_a_file_at_least_that_long(monkeypatch, bars_csv):
    """NT-088 round 2, finding 3b: a screen spec must not lower DATA_END_PROTECTED_DAYS below the
    Config default against a file that actually spans at least that default (the real long
    2017-2025 file). Only a file SHORTER than the default (the bundled/synthetic CSV, RUNBOOK
    "Screen mode") may lower it - forcing the floor unconditionally would make it impossible to
    screen against the bundled 30-day CSV at all, which RUNBOOK and example_6h.yaml already document
    as the sanctioned exception."""
    long_ts = pd.date_range("2020-01-01", periods=200, freq="1D", tz="UTC")
    long_df = pd.DataFrame({"timestamp": long_ts})
    monkeypatch.setattr(DataProcessor, "load_raw", lambda self, *a, **k: None)
    monkeypatch.setattr(DataProcessor, "preprocess", lambda self, df: long_df)
    s = spec_dict(bars_csv, slices=["2020-02-01T00:00:00+00:00"], seeds=[0])
    s["overrides"]["DATA_END_PROTECTED_DAYS"] = 1.0   # far below the 64-day default
    with pytest.raises(ScreenError, match="DATA_END_PROTECTED_DAYS"):
        build_trials(ScreenSpec.from_dict(s))


def test_preflight_refuses_the_whole_spec_before_any_trial_trains(tmp_path, bars_csv):
    """NT-088 round 2, finding 3c: one violating slice must refuse the WHOLE spec before any trial
    trains, not only its own - not even the trials that would have been fine may run."""
    good = SAFE_DATA_END
    bad = "2025-10-13T04:28:59+00:00"   # 1 second before the fixture's last bar: inside the protected span
    s = spec_dict(bars_csv, slices=[good, bad], seeds=[0])
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "runs"
    with pytest.raises(ScreenError, match="protected dev/test span"):
        run_screen(spec, store=store, trainer=_stub_trainer)
    results_path = store / "screens" / spec.name / "results.jsonl"
    assert not results_path.exists() or results_path.read_text(encoding="utf-8").strip() == ""


def test_example_6h_spec_trains_on_360_windows_6_hours():
    """NT-088 round 2, finding 4: FOLD_INDEX -1 (with N_FOLDS: 2) gave a 1,860-window (31-hour)
    training block; the plan's reference size is 360 windows = 6 hours (FOLD_INDEX -2)."""
    spec = ScreenSpec.from_yaml("configs/screens/example_6h.yaml")
    trial = build_trials(spec)[0]
    layout = data_layout(trial.config)
    fold = layout.fold(int(trial.config.FOLD_INDEX))
    assert fold["blocks"]["train"]["n"] == 360


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


def test_loss_term_shares_apply_the_lambda_multiplier():
    """NT-088 round 2, finding 5b: history logs ``dir_loss`` RAW (unweighted); a huge LAMBDA_DIR must
    still show up as a large share once the multiplier is applied, or ``max_term_share`` can never see
    it dominating."""
    from neural_trade.experiments.screen import _GradNormSampler, _health_from_history

    class FakeHistory:
        def __init__(self, history):
            self.history = history

    cfg = Config(LAMBDA_DIR=1000.0, LAMBDA_DIR_OUTER=1.0)
    hist = {"loss": [10.0, 5.0], "point_loss": [1.0, 1.0], "dir_loss": [0.01, 0.004], "nll_loss": [0.0, 0.0]}
    health = _health_from_history(FakeHistory(hist), _GradNormSampler(0.0), cfg)
    # true contribution: LAMBDA_DIR_OUTER * LAMBDA_DIR * raw = 1000 * 0.004 = 4.0, of a final total of
    # 5.0 -> a 0.8 share (the previous, unweighted computation gave 0.004 / 5.0 = 0.0008).
    assert health["loss_term_shares"]["dir_loss"] == pytest.approx(4.0 / 5.0)
    assert health["max_term_share"] == pytest.approx(4.0 / 5.0)


def test_sanitize_nonfinite_replaces_nan_and_inf_with_null():
    obj = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")}, "e": 1.0}
    out, bad = _sanitize_nonfinite(obj)
    assert out == {"a": None, "b": [1.0, None], "c": {"d": None}, "e": 1.0}
    assert set(bad) == {"a", "b[1]", "c.d"}
    json.dumps(out, allow_nan=False)   # must not raise


# ------------------------------------------------------------------ (7) resumability
def test_resuming_a_screen_skips_trials_already_in_results_jsonl(tmp_path, bars_csv):
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1]))
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
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1]))
    store = tmp_path / "runs"
    r1 = run_screen(spec, store=store, max_trials=1, trainer=_stub_trainer)
    assert r1.ran == 1
    r2 = run_screen(spec, store=store, trainer=_stub_trainer)
    assert r2.skipped == 1
    assert r2.ran == r1.n_trials - 1


def test_run_trial_writes_a_pass_fail_row_with_reasons(bars_csv):
    spec = ScreenSpec.from_dict(spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0]))
    trial = build_trials(spec)[0]
    spec.rules.update(max_nonfinite_grad_steps=0)
    row = run_trial(trial, spec, {}, trainer=_stub_trainer)
    assert row["trial_key"] == trial.key
    assert row["passed"] is True and row["reasons"] == []
    assert set(row["timings"]) == {"load_s", "prep_s", "build_s", "train_s", "score_s", "epoch_s"}


# ------------------------------------------------------------------ (8) real CPU training
@pytest.mark.slow
def test_a_known_bad_learning_rate_fails_the_rules_on_real_cpu_training(bars_csv):
    """LR=1.0 (the top of its valid range) on real training: a known-bad config must fail the
    pre-registered rules (real training, not the fake trainer, per NT-088's acceptance criteria)."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"].update(LR=1.0, BATCH_SIZE=32)
    s["grid"] = {"axes": {}}
    # clip_skip_epochs: 0 (NT-092's default is 1, which excludes epoch 0's steps from clipped_share -
    # exactly the epoch where this LR=1.0 config's damage shows up in a 2-epoch trial; the whole point
    # of this test is to prove real bad training IS caught, so it turns the skip back off).
    s["rules"] = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 0.3,
                 "min_train_loss_drop": 0.0, "max_term_share": 0.9, "clip_skip_epochs": 0}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {})   # the real light path: no trainer override
    assert row["passed"] is False, f"a known-bad LR should fail the rules: {row['reasons']} / {row['health']}"
    assert row["reasons"]


@pytest.mark.slow
def test_a_lambda_of_1e6_fails_the_rules_on_real_cpu_training(bars_csv):
    """NT-088 round 2, finding 5a: a LAMBDA of 1e6 (LAMBDA_DIR) must also fail the pre-registered
    rules on real training, now that its weighted share can actually dominate max_term_share
    (finding 5b) - or, failing that, the training itself goes non-finite."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"].update(LAMBDA_DIR=1e6, BATCH_SIZE=32)
    s["run"] = {"calibrate": False, "epochs": 1}
    s["grid"] = {"axes": {}}
    s["rules"] = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0,
                 "min_train_loss_drop": -1.0, "max_term_share": 0.9}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {})
    assert row["passed"] is False, f"LAMBDA_DIR=1e6 should fail the rules: {row['reasons']} / {row['health']}"
    assert row["reasons"]


@pytest.mark.slow
def test_a_nonfinite_trial_is_recorded_failed_and_not_retried_on_resume(tmp_path, bars_csv):
    """NT-088 round 2, finding 2: an extreme LAMBDA_POINT (QA repro) drives training non-finite; the
    JSONL row must still be written (never crash ``json.dumps(allow_nan=False)``), recorded failed,
    and a resumed run must not re-attempt it."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"].update(LAMBDA_POINT=1e38, BATCH_SIZE=32)
    s["run"] = {"calibrate": False, "epochs": 2}
    s["grid"] = {"axes": {}}
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "runs"
    r1 = run_screen(spec, store=store)   # the real light path
    assert r1.ran == 1 and r1.failed == 1
    lines = (store / "screens" / spec.name / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])           # a plain json.loads: the row is valid JSON, no NaN/Infinity tokens
    assert row["passed"] is False
    assert any("non-finite" in r for r in row["reasons"])

    r2 = run_screen(spec, store=store)   # resumed: must not re-attempt the failed trial
    assert r2.ran == 0 and r2.skipped == 1


@pytest.mark.slow
def test_run_trial_reports_prep_build_epoch_timings_on_real_training(bars_csv):
    """NT-088 round 2, finding 7: prep_s (windowing/dataset construction) and epoch_s (per-epoch
    wall-clock times) are separate from build_s, for the phase-2 tracing decision."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["run"] = {"calibrate": False, "epochs": 2}
    s["grid"] = {"axes": {}}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {})
    t = row["timings"]
    assert set(t) == {"load_s", "prep_s", "build_s", "train_s", "score_s", "epoch_s"}
    assert isinstance(t["epoch_s"], list) and len(t["epoch_s"]) == 2
    assert all(isinstance(x, float) and x >= 0 for x in t["epoch_s"])


@pytest.mark.slow
def test_loss_term_shares_use_the_trained_models_own_lambdas_on_real_calibrated_training(bars_csv):
    """NT-088 round 3, QA finding 5 (re-check): with ``run.calibrate: true`` AND a non-default
    LAMBDA_T_PERP (100), the shares must (a) sum to about 1 and (b) NOT double-count t_perp.

    Before this fix, on the same kind of trial (QA's repro against the reference-setup CSV):
    reading ``cfg.LAMBDA_*`` instead of the model's actually-applied (post-calibration) lambdas made
    calibrated shares sum to 1.232 instead of ~1, and treating ``t_perp_loss`` as RAW (applying
    LAMBDA_T_PERP a second time on top of the ALREADY lambda_t_perp-weighted ``c.t_perp_total``)
    reported a share around 78 for a LAMBDA_T_PERP of 100 instead of its true, order-of-magnitude-
    smaller contribution. Recomputing both scenarios by hand from the captured history/model
    confirms the exact old numbers (1.2322 and 78.5654 respectively; see the repair-round report)."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"].update(LAMBDA_T_PERP=100.0, BATCH_SIZE=32)
    s["run"] = {"calibrate": True, "epochs": 2}
    s["grid"] = {"axes": {}}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]
    row = run_trial(trial, spec, {})
    shares = row["health"]["loss_term_shares"]
    assert shares, f"expected non-empty loss_term_shares: {row['health']}"

    total_share = sum(shares.values())
    assert total_share == pytest.approx(1.0, abs=0.2), (
        f"loss_term_shares should sum to about 1 (the only unaccounted terms are the unlogged "
        f"dir_align_loss/coherence_penalty); got {total_share} from {shares}")

    # t_perp_loss must NOT be inflated by double-applying LAMBDA_T_PERP: its share stays a small
    # fraction of the total, nowhere near the ~78 the double-counting bug produced.
    t_perp_share = shares.get("t_perp_loss")
    if t_perp_share is not None:
        assert t_perp_share < 2.0, (
            f"t_perp_loss share {t_perp_share} looks double-counted (LAMBDA_T_PERP applied twice); "
            f"expected an order-of-magnitude-smaller share, well under 2.0")


def test_grad_norm_sampler_reports_clipped_share_as_none_when_every_sampled_norm_is_nonfinite():
    """NT-088 round 3, fix 4: an all-non-finite trial (e.g. an extreme LAMBDA_* that drives the
    gradient norm itself to NaN on every logged step) must report ``clipped_share`` as ``None``
    (JSON ``null``), not ``0.0`` -- ``0.0`` would misread as "nothing was clipped" when in fact there
    was no valid clip/no-clip signal at all."""
    from neural_trade.experiments.screen import _GradNormSampler

    class _FakeMean:
        def __init__(self):
            self.total = tf.Variable(0.0)
            self.count = tf.Variable(0.0)

    sampler = _GradNormSampler(clip_norm=1.0)
    sampler.model = type("M", (), {"_step_means": {"grad_global_norm": _FakeMean()}})()
    sampler.on_epoch_begin(0)
    # Three logged steps, each pushing the running total to NaN (a non-finite grad_global_norm
    # sampled every time): the per-step delta is NaN throughout.
    for i in range(1, 4):
        sampler.model._step_means["grad_global_norm"].total.assign(float("nan"))
        sampler.model._step_means["grad_global_norm"].count.assign(float(i))
        sampler.on_train_batch_end(i - 1)

    assert sampler.n_steps == 3
    assert sampler.n_valid_steps == 0
    assert sampler.clipped_share is None
    assert sampler.mean_norm is None


def test_merged_existing_keys_docstring_names_same_shard_count_only():
    """NT-088 round 3, fix 4: :func:`_merged_existing_keys` only ever merges/considers shard files
    for the SAME total shard count N; a stale-docstring regression would be easy to reintroduce
    silently, so pin the documented behaviour by exercising it directly."""
    import shutil

    from neural_trade.experiments.screen import _append_jsonl, _merged_existing_keys, _shard_result_path

    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        out_dir = __import__("pathlib").Path(tmp)
        # A shard file from a DIFFERENT total shard count (N=3) must be ignored when resuming N=2.
        _append_jsonl(_shard_result_path(out_dir, (0, 3)), {"trial_key": "stale-n3"})
        _append_jsonl(_shard_result_path(out_dir, (0, 2)), {"trial_key": "this-n2"})
        keys = _merged_existing_keys(out_dir, (1, 2))
        assert keys == {"this-n2"}, f"expected only the N=2 shard's keys, got {keys}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pnl_utility_term_counts_in_the_loss_term_shares_at_weight_one():
    """pnl_utility (NT-087) logs pnl_val already multiplied by LAMBDA_PNL, so it joins the shares at 1.0."""
    from neural_trade.core.config import Config
    from neural_trade.experiments.screen import LOSS_TERM_KEYS, _term_multiplier

    assert "pnl_val" in LOSS_TERM_KEYS
    assert _term_multiplier("pnl_val", Config()) == 1.0


# ------------------------------------------------------------------ (9) phase 2: reused-graph trials (NT-092)
def test_structural_key_ignores_continuous_fields_seed_data_end_and_epochs(bars_csv):
    """Two configs differing only in a CONTINUOUS_FIELDS entry, SEED, DATA_END or EPOCHS share one
    structural_key; a BATCH_SIZE (structural) difference gets a different one."""
    from neural_trade.experiments.screen import structural_key

    base_overrides = {"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2,
                      "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1, "DATA_END_PROTECTED_DAYS": SAFE_PROTECTED_DAYS}
    base = Config().override(**base_overrides).copy()  # .copy() twice: normalises numeric field types
    same = base.copy(LR=0.01, LAMBDA_HD=0.5, GRAD_CLIP_NORM=5.0, ADAM_BETA1=0.8, INDICATOR_LR_MULT=2.0,
                     SEED=7, DATA_END=SAFE_DATA_END, EPOCHS=3)
    assert structural_key(base) == structural_key(same), "continuous-only + SEED/DATA_END/EPOCHS changes must not retrace"
    different = base.copy(BATCH_SIZE=64)
    assert structural_key(base) != structural_key(different), "BATCH_SIZE is structural"


def test_group_order_keeps_each_structural_groups_trials_contiguous(bars_csv):
    from neural_trade.experiments.screen import _group_order, structural_key

    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["grid"] = {"axes": {"BATCH_SIZE": [32, 64], "LR": [0.001, 0.002]}}
    spec = ScreenSpec.from_dict(s)
    trials = build_trials(spec)
    ordered = _group_order(trials)
    assert {t.index for t in ordered} == {t.index for t in trials}          # every trial, once
    keys = [structural_key(t.config) for t in ordered]
    seen = []
    for k in keys:
        if k not in seen:
            seen.append(k)
        else:
            assert k == seen[-1], f"group {k!r} is not contiguous in {keys}"


@pytest.mark.slow
def test_reuse_graph_compiles_once_per_structural_group_not_once_per_trial(bars_csv, monkeypatch, tmp_path):
    """Acceptance 1/3/4: continuous-only trials of one screen share one compiled model (one
    ``CustomTrainModel.compile`` call, real CPU training); a structural axis alongside them starts a
    second group (a second compile)."""
    from neural_trade.training.custom_model import CustomTrainModel

    calls = {"n": 0}
    orig_compile = CustomTrainModel.compile

    def counting_compile(self, *a, **kw):
        calls["n"] += 1
        return orig_compile(self, *a, **kw)

    monkeypatch.setattr(CustomTrainModel, "compile", counting_compile)

    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0, 1])
    s["overrides"]["BATCH_SIZE"] = 32
    s["grid"] = {"axes": {"LR": [0.001, 0.002, 0.003]}}   # continuous only: one structural group
    s["run"] = {"calibrate": False, "epochs": 1}
    spec = ScreenSpec.from_dict(s)
    report = run_screen(spec, store=str(tmp_path / "continuous_only"))
    assert report.ran == 6      # 3 LR x 1 slice x 2 seeds
    assert report.failed == 0, "every trial should train and be scored"
    assert calls["n"] == 1, f"6 continuous-only trials should share one compiled graph, compiled {calls['n']} times"

    calls["n"] = 0
    s2 = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s2["grid"] = {"axes": {"BATCH_SIZE": [32, 64]}}       # structural: two groups
    s2["run"] = {"calibrate": False, "epochs": 1}
    spec2 = ScreenSpec.from_dict(s2)
    report2 = run_screen(spec2, store=str(tmp_path / "structural"))
    assert report2.ran == 2
    assert calls["n"] == 2, f"a structural (BATCH_SIZE) change must start a new group, compiled {calls['n']} times"


@pytest.mark.slow
def test_reused_first_trial_matches_a_fresh_trial_of_the_same_config_and_seed(bars_csv):
    """Acceptance 2: a reused-graph trial (here, the first trial of a fresh _TrialGroup - the case
    every later trial in the group also resets weights/optimizer/RNGs to, per training/reset.py's
    module doc) equals a standalone fresh-graph trial of the exact same config and seed. Compared on
    the final train/val loss (the scalar every other health number derives from) and the direction
    AUCs; both are exact floats from the same deterministic CPU op set, so this is bit-for-bit, not a
    tolerance."""
    from neural_trade.experiments.screen import _TrialGroup, _run_trial_light

    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[3])
    s["grid"] = {"axes": {}}
    s["overrides"]["LR"] = 0.002
    s["run"] = {"calibrate": False, "epochs": 1}
    spec = ScreenSpec.from_dict(s)
    trial = build_trials(spec)[0]

    health_fresh, auc_fresh, _ = _run_trial_light(trial.config, {}, calibrate=False)
    group = _TrialGroup(trial.config)
    health_reused, auc_reused, _ = group.run_one(trial.config, {}, calibrate=False)

    assert health_reused["final_train_loss"] == pytest.approx(health_fresh["final_train_loss"], rel=0, abs=0), (
        health_reused["final_train_loss"], health_fresh["final_train_loss"])
    assert health_reused["final_val_loss"] == pytest.approx(health_fresh["final_val_loss"], rel=0, abs=0)
    for h in auc_fresh:
        assert auc_reused[h]["auc"] == auc_fresh[h]["auc"]


@pytest.mark.slow
def test_reused_later_trial_matches_an_independent_fresh_run_with_default_dropout_and_noise(bars_csv):
    """Acceptance 2, QA repair round 1: trial 3 of a 4-trial group (LR, a LAMBDA, GRAD_CLIP_NORM and
    DATA_END all changed from trial 0; calibrate on; default dropout rate 0.1 and the
    VacuumSaturationNoise layer both ACTIVE, i.e. LAMBDA_T_PERP left at its nonzero default) matches
    an INDEPENDENT fresh run of that exact config and seed, bit-for-bit. Before the round-1 fix,
    Dropout/MultiHeadAttention's legacy stateful RNG and the noise layer's unseeded tf.random.normal
    both kept advancing across the group's earlier trials instead of resetting to trial 3's own seed,
    so this failed (QA's evidence: trial 3 val 6.5250 reused vs 6.0214 fresh, real BTC 1-minute data).
    Config.SEEDED_STOCHASTIC_LAYERS (forced True by screen.py itself, not this test) is what fixes it."""
    from neural_trade.experiments.screen import _TrialGroup, _run_trial_light

    base_overrides = {"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2,
                      "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1, "DATA_END_PROTECTED_DAYS": SAFE_PROTECTED_DAYS,
                      "BATCH_SIZE": 32, "EPOCHS": 2, "DATA_END": SAFE_DATA_END}
    base = Config().override(**base_overrides)
    trial_overrides = [
        dict(SEED=0, LR=1e-3),
        dict(SEED=1, LR=2e-3, LAMBDA_HD=0.3),
        dict(SEED=2, LR=1e-3, GRAD_CLIP_NORM=0.5, ADAM_BETA1=0.8),
        dict(SEED=3, LR=5e-4, LAMBDA_DIR_OUTER=0.3, GRAD_CLIP_NORM=0.0,
            DATA_END="2025-10-12T04:19:00+00:00"),
    ]
    cfgs = [base.copy(**o) for o in trial_overrides]

    group = _TrialGroup(cfgs[0])
    for cfg in cfgs:
        health_reused, auc_reused, _ = group.run_one(cfg, {}, calibrate=True)
    # health_reused/auc_reused now hold trial 3's (the last one run) result.
    health_fresh, auc_fresh, _ = _run_trial_light(cfgs[3], {}, calibrate=True)

    assert health_reused["final_train_loss"] == pytest.approx(health_fresh["final_train_loss"], rel=0, abs=0), (
        health_reused["final_train_loss"], health_fresh["final_train_loss"])
    assert health_reused["final_val_loss"] == pytest.approx(health_fresh["final_val_loss"], rel=0, abs=0)
    for h in auc_fresh:
        assert auc_reused[h]["auc"] == auc_fresh[h]["auc"]


@pytest.mark.slow
def test_reuse_graph_median_trial_wall_after_the_first_is_well_under_the_first(bars_csv, tmp_path):
    """Acceptance 4: 8 continuous-only trials in one group, CPU: build_s is 0 for every trial but the
    first (nothing is rebuilt), and the group's later trials' median wall is well under the first
    trial's (which pays the one-off trace cost inside its first `fit()` call)."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"]["BATCH_SIZE"] = 32
    s["grid"] = {"axes": {"LR": [0.001 * i for i in range(1, 9)]}}   # 8 continuous-only trials
    s["run"] = {"calibrate": False, "epochs": 1}
    spec = ScreenSpec.from_dict(s)
    store = tmp_path / "timing"
    run_screen(spec, store=str(store))
    rows = sorted(merge_results(store=str(store), name="tiny_screen"), key=lambda r: r["trial_index"])
    assert len(rows) == 8
    wall = [r["wall_s"] for r in rows]
    train_s = [r["timings"]["train_s"] for r in rows]
    # build_s (Models.build for fresh initial weights + optimizer/lambda reset) is a real, roughly
    # constant per-trial cost in this design (~0.7s on this machine) - it is NOT the ~12s trace cost
    # the plan measured, which lives inside train_s's first `fit()` call. The trace saving shows up as
    # a much smaller train_s (and total wall) from trial 2 on, not as build_s == 0.
    median_train_rest = float(np.median(train_s[1:]))
    median_rest = float(np.median(wall[1:]))
    logging.getLogger(__name__).info(
        "NT-092 CPU timing (8-trial group, 1 epoch each): trial0 wall=%.3fs train_s=%.3fs | "
        "trials 2..8 median wall=%.3fs median train_s=%.3fs | full breakdown=%s",
        wall[0], train_s[0], median_rest, median_train_rest, rows[0]["timings"])
    assert median_train_rest < train_s[0], (
        f"trials after the first should train faster (no retrace of train_step): train_s={train_s}")
    assert median_rest < wall[0], f"trials after the first should have a lower total wall: {wall}"


def test_min_epochs_guard_refuses_an_explicit_clip_skip_epochs_at_or_above_epochs(bars_csv):
    """Acceptance 5: rules.clip_skip_epochs explicitly >= EPOCHS is refused before any trial trains."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["run"] = {"calibrate": False, "epochs": 2}
    s["rules"] = {"finite": True, "clip_skip_epochs": 2}
    with pytest.raises(ScreenError, match="clip_skip_epochs"):
        ScreenSpec.from_dict(s).base()


def test_min_epochs_guard_is_checked_per_trial_when_epochs_varies_by_axis(bars_csv):
    """QA repair round 1, P2: EPOCHS is in _STRUCTURAL_IGNORE, so a grid axis may override it per
    trial. rules.clip_skip_epochs=1 is fine against the spec-wide base EPOCHS=2, but a trial whose
    OWN EPOCHS axis value is 1 must still be refused (not silently scored on zero steps)."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["overrides"]["EPOCHS"] = 2
    s["grid"] = {"axes": {"EPOCHS": [1, 2]}}
    s["run"] = {"calibrate": False}
    s["rules"] = {"finite": True, "clip_skip_epochs": 1}
    spec = ScreenSpec.from_dict(s)
    spec.base()  # the spec-wide base (EPOCHS=2) alone must not raise
    with pytest.raises(ScreenError, match="clip_skip_epochs"):
        build_trials(spec)  # the EPOCHS=1 trial must be refused


def test_default_clip_skip_epochs_is_clamped_for_a_one_epoch_screen_not_refused(bars_csv):
    """The unset default (1) must not trip the min_epochs guard for an ordinary 1-epoch smoke screen
    (effective_clip_skip_epochs clamps it to 0 instead)."""
    s = spec_dict(bars_csv, slices=[SAFE_DATA_END], seeds=[0])
    s["run"] = {"calibrate": False, "epochs": 1}
    spec = ScreenSpec.from_dict(s)
    spec.base()  # must not raise
    assert spec.effective_clip_skip_epochs(1) == 0


def test_grad_norm_sampler_excludes_the_first_clip_skip_epochs_from_clipped_share():
    """clip_skip_epochs (default 1): steps logged during the skipped epoch(s) count toward n_steps
    but never toward clipped_share / the norm max/mean."""
    from neural_trade.experiments.screen import _GradNormSampler

    class _FakeMean:
        def __init__(self):
            self.total = tf.Variable(0.0)
            self.count = tf.Variable(0.0)

    sampler = _GradNormSampler(clip_norm=1.0, clip_skip_epochs=1)
    sampler.model = type("M", (), {"_step_means": {"grad_global_norm": _FakeMean()}})()
    m = sampler.model._step_means["grad_global_norm"]
    # Epoch 0 (skipped): two steps, both loudly over the clip norm. Keras resets a Mean metric at
    # every epoch boundary, so this fake resets total/count too (on_epoch_begin only resets the
    # sampler's OWN _prev_total/_prev_count, matching that reset).
    sampler.on_epoch_begin(0)
    total, count = 0.0, 0.0
    for value in (5.0, 5.0):
        total += value
        count += 1.0
        m.total.assign(total)
        m.count.assign(count)
        sampler.on_train_batch_end(int(count) - 1)
    # Epoch 1 (counted): one step, under the clip norm.
    sampler.on_epoch_begin(1)
    total, count = 0.0, 0.0
    total += 0.2
    count += 1.0
    m.total.assign(total)
    m.count.assign(count)
    sampler.on_train_batch_end(0)

    assert sampler.n_steps == 3
    assert sampler.n_valid_steps == 1
    assert sampler.n_clipped == 0
    assert sampler.clipped_share == 0.0
    assert sampler.max_norm == pytest.approx(0.2)


def test_continuous_fields_are_exactly_the_ones_this_module_documents_as_variable_backed():
    """CONTINUOUS_FIELDS matches training/lambdas.py's variable-backed lambda keys plus the
    Keras-hyper/tf.Variable fields this item added (LR, betas, GRAD_CLIP_NORM, INDICATOR_LR_MULT)."""
    from neural_trade.experiments.screen import CONTINUOUS_FIELDS
    from neural_trade.training.lambdas import CONFIG_NAME_OF_KEY

    for name in CONFIG_NAME_OF_KEY.values():
        assert name in CONTINUOUS_FIELDS, f"{name} is a tf.Variable-backed lambda but missing from CONTINUOUS_FIELDS"
    for extra in ("LR", "ADAM_BETA1", "ADAM_BETA2", "GRAD_CLIP_NORM", "INDICATOR_LR_MULT"):
        assert extra in CONTINUOUS_FIELDS
    # Known NOT continuous (documented exclusions): baked as plain Config reads outside training/, or
    # gating a Python `if` inside the traced loss.
    for excluded in ("LAMBDA_VAC", "LAMBDA_DIR_ALIGN", "LAMBDA_INTER", "LAMBDA_DIR_ALIGN_OUTER",
                     "INDICATOR_GRAD_MULT", "ABLATE_LAMBDAS"):
        assert excluded not in CONTINUOUS_FIELDS


def test_screen_mode_numbers_moved_once_when_the_seed_derivation_became_name_based():
    """NT-108 acceptance 2: a "golden screen record" of an actual trained loss cannot be pinned
    bit-for-bit here - CPU training of this model is already known to differ run-to-run at a fixed
    seed for reasons unrelated to this item (NT-074, P1: "op determinism is free but same-seed runs
    differ at epoch 0", docs/STATUS.md; observed while writing this test: a real-CPU
    ``_run_trial_light`` on this exact config gave ``final_train_loss`` 10.050235748291016 on one run
    and 10.259147644042969 on another, both BEFORE this item's fix). So the record this test pins is
    the derived SEED itself: an exact integer from sha256 + arithmetic, with no CPU thread-scheduling
    sensitivity at all.

    :func:`training.reset.reset_stateful_rngs` used to derive each stochastic layer's seed from
    ``i = enumerate(model.submodules)`` (this exact formula, at commit f7d4a41, reproduced below only
    to document the bug): ``seed * 1_000_003 + i``. NT-037 found that adding 18
    ``tf.keras.metrics.Mean`` objects to ``CustomTrainModel`` shifted a stochastic layer from
    position 10 (Keras ``lambda_t_perp`` 0.89 screen result) to position 28 (1.29) - the layer never
    moved, only its neighbours' names sorted differently in ``model.submodules``' attribute-name
    ordering. The new record (this commit) derives the same layer's seed from its own name instead,
    which the same change leaves alone."""
    from neural_trade.training.reset import _identity_offset

    def old_formula(seed: int, position: int) -> int:
        return int(seed) * 1_000_003 + position

    seed = 5
    # The SAME Dropout layer, only its position in model.submodules changed (NT-037's 18 new Mean
    # metrics, tracked earlier in the model's attribute traversal, pushed it from 10 to 28).
    old_seed_before = old_formula(seed, position=10)
    old_seed_after = old_formula(seed, position=28)
    assert old_seed_before != old_seed_after, "documents the bug: an unrelated attribute changed the seed"

    class _Named:
        def __init__(self, name):
            self.name = name

    layer = _Named("dropout")   # same layer, same name, regardless of what else is on the model
    new_seed_before = (seed * 1_000_003 + _identity_offset(layer, 0)) % (2**31 - 1)
    new_seed_after = (seed * 1_000_003 + _identity_offset(layer, 0)) % (2**31 - 1)
    assert new_seed_before == new_seed_after, "the fix: the same layer keeps the same seed"
