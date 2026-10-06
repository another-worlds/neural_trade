"""The stability harness and the config guard (NT-038, D-026).

Fast tests run the real engine (scenario, runner, scorer, run store, index, report) with a fake trainer that writes
the telemetry a real run writes; two tests (marked ``stability``) train for real on CPU at the tiny size. The
guard's attribution is tested on the real loss model by fault injection (as tests/test_stability.py does)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.core.guard import (REGIONS_ENV, SCORE_ELEMENTS_WARN, Region, check_config, find_region, load_regions,
                                     memory_warning, regions_disabled, write_regions)
from neural_trade.experiments import stability as st
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.store import RunStore
from neural_trade.experiments.sweep import FailingRegionHit, SearchSpace, SweepError, dev_net_sharpe, run_health
from neural_trade.training.stability_guard import StabilityGuard, UnstableTrainingError, blame, describe

from .test_sweep import FakeTrainer as SweepFakeTrainer
from .test_sweep import SEARCH, fake_result, make_sweep, scenario_dict

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("stability_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ helpers
def write_run_telemetry(run_dir, *, epochs=1, n_steps=20.0, nonfinite=0.0, clip_main=2.0, clip_ind=2.0, masked=None,
                        probe=None, var_floor=0.0, periods=None, losses=None, horizons=("h0",)):
    """metrics.jsonl and status.json as the trainer writes them (the keys the harness reads)."""
    rows = []
    for e in range(len(losses) if losses else epochs):
        r = {"epoch": e, "loss": losses[e] if losses else 1.0, "val_loss": 0.9, "n_steps": n_steps,
             "nonfinite_grad_steps": nonfinite,
             "grad_clip_steps_main": clip_main, "grad_clip_steps_indicator": clip_ind, "grad_norm_max_main": 5.0,
             "grad_norm_max_indicator": 5.0}
        for h in horizons:
            r.update({f"dir_n_{h}": 100.0, f"var_at_floor_{h}": var_floor})
        r.update({f"masked_{k}": v for k, v in (masked or {}).items()})
        r.update({f"probe_grad_share_{k}": v for k, v in (probe or {}).items()})
        r.update(periods or {})
        rows.append(r)
    Path(run_dir, "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    Path(run_dir, "status.json").write_text(json.dumps({"val_loss": 0.9, "weights_val_loss": 0.9}), encoding="utf-8")


class HarnessFake:
    """The engine trainer stand-in: healthy telemetry and fake predictions, except per case ``behaviour``:
    'raise' (a fault stops the run, naming crps_loss), 'quiet' (a fault is NOT detected), 'nonfinite' (a clean run
    with non-finite steps and a masked term)."""

    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []

    def __call__(self, ctx, *, calibrate, save_artifacts):
        eng = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]
        case = eng["configuration"]
        self.calls.append(eng["cell_key"])
        how = self.behaviour.get(case, "raise" if case.startswith("fault_") else "ok")
        if how == "raise":
            write_run_telemetry(ctx.run_dir)
            msg, terms = describe({"crps_loss": 1.0, "total_loss": 1.0}, 1.0, 1)
            raise UnstableTrainingError(msg, terms, 1, 1.0)
        if how == "nonfinite":
            write_run_telemetry(ctx.run_dir, nonfinite=3.0, masked={"crps_loss": 3.0})
        else:
            write_run_telemetry(ctx.run_dir)
        return fake_result(ctx.config)


def fake_harness(tmp_path, csv, behaviour=None, cases=None, seeds=None, **kw):
    return st.run_harness(profile="tiny", csv=csv, store=tmp_path / "runs", case_ids=cases, seeds=seeds,
                          trainer=HarnessFake(behaviour), **kw)


# ------------------------------------------------------------------ (1) the thresholds file
def test_the_thresholds_hash_is_the_files_sha256_and_any_edit_changes_it(tmp_path):
    import hashlib

    t = st.load_thresholds()
    assert t.path == REPO / "configs" / "stability_thresholds.yaml"
    assert t.sha256 == hashlib.sha256(t.path.read_bytes()).hexdigest() and len(t.sha256) == 64
    copy = tmp_path / "thr.yaml"
    shutil.copy(t.path, copy)
    assert st.load_thresholds(copy).sha256 == t.sha256
    copy.write_text(copy.read_text(encoding="utf-8") + "# an edit\n", encoding="utf-8")
    assert st.load_thresholds(copy).sha256 != t.sha256


def test_every_threshold_the_harness_reads_is_in_the_file_with_a_rationale_comment():
    text = (REPO / "configs" / "stability_thresholds.yaml").read_text(encoding="utf-8")
    t = st.load_thresholds()
    assert t.seeds == 3 and t.fault_detection["error_type"] == "UnstableTrainingError"
    for key in t.checks:
        i = text.index(f"  {key}:")
        assert "#" in text[max(0, i - 700):i], f"{key} has no rationale comment above it"


def test_the_report_carries_the_thresholds_hash_and_a_different_file_gives_a_different_hash(tmp_path, bars_csv):
    thr = tmp_path / "thr.yaml"
    shutil.copy(st.THRESHOLDS_FILE, thr)
    res = fake_harness(tmp_path, bars_csv, cases=["control"], seeds=[0], thresholds_path=thr)
    assert res.thresholds_sha256 == st.file_sha256(thr)
    assert res.thresholds_sha256 in res.report.read_text(encoding="utf-8")
    assert res.thresholds_sha256 in (res.out_dir / "verdicts.json").read_text(encoding="utf-8")
    assert all(v.thresholds_sha256 == res.thresholds_sha256 for v in res.verdicts)
    thr.write_text(thr.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    res2 = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs2", case_ids=["control"], seeds=[0],
                          thresholds_path=thr, trainer=HarnessFake())
    assert res2.thresholds_sha256 != res.thresholds_sha256 and res2.thresholds_sha256 in res2.report.read_text(encoding="utf-8")


# ------------------------------------------------------------------ (2) the cases
def test_the_default_cases_cover_scale_volatility_fuzzing_faults_and_the_named_cases():
    cs = {c.id: c for c in st.default_cases()}
    assert {"scale_x0.1", "scale_x10", "vol_x0.1", "vol_x10"} <= set(cs)
    assert {"fuzz_constant", "fuzz_jumps", "fuzz_large_price", "fuzz_small_price"} <= set(cs)
    faults = {c.fault["kind"] for c in cs.values() if c.group == "fault"}
    assert faults == {"nan_input", "nan_term", "nan_gradient"} and all(cs[i].expect == "detect"
                                                                       for i in cs if cs[i].group == "fault")
    wide = cs["horizons_5_60_240"]
    assert wide.overrides["HORIZON_STEPS"] == [5, 60, 240] and "0.054/0.586/2.359" in wide.description
    for n in (1440, 10080):                           # the long-memory case: defined, marked GPU, never run here
        for lr in (5, 1):
            c = cs[f"long_memory_{n}_lr{lr}"]
            assert not c.runnable and "GPU, NT-051" in c.note and c.overrides["INDICATOR_LR_MULT"] == float(lr)
    assert not cs["long_memory_scale_norm"].runnable and "GPU, NT-051" in cs["long_memory_scale_norm"].note
    assert cs["slow_periods_proxy_lr5"].runnable and cs["slow_periods_proxy_lr1"].runnable


def test_the_scenario_is_strict_with_one_variant_per_runnable_case_and_three_seeds_by_default(tmp_path, bars_csv):
    cases = st.default_cases()
    spec = st.build_scenario([c for c in cases if c.runnable], name="stab-x", profile="tiny", seeds=[0, 1, 2],
                             case_csv={"scale_x10": "x.csv"})
    sc = Scenario.from_dict(spec)
    assert sc.seeds == [0, 1, 2] and spec["overrides"]["STRICT_LOSS_MASKS"] is True
    assert set(spec["variants"]) == {c.id for c in cases if c.runnable}
    assert spec["variants"]["scale_x10"]["CSV_PATH"] == "x.csv"
    assert spec["variants"]["horizons_5_60_240"]["N_FOLDS"] == 4          # the tiny layout the 480-bar gap needs
    assert st.PROFILES["reference"]["PROBE_GRADIENTS"] is True
    res = fake_harness(tmp_path, bars_csv, cases=["control"])               # default seeds: 3
    assert sorted(v.seed for v in res.verdicts) == [0, 1, 2]


def test_transform_bars_scales_prices_and_volatility_and_builds_flat_blocks_and_jumps(synthetic_bars):
    df = synthetic_bars
    c0 = df["close"].to_numpy()
    s = st.transform_bars(df, {"kind": "scale", "k": 10.0})
    assert np.allclose(s["close"], 10 * c0) and np.allclose(s["high"], 10 * df["high"]) and (s["volume"] == df["volume"]).all()
    r0 = np.diff(np.log(c0))
    for k in (0.1, 10.0):
        v = st.transform_bars(df, {"kind": "vol", "k": k})
        assert v["close"].iloc[0] == pytest.approx(c0[0])
        assert np.std(np.diff(np.log(v["close"]))) == pytest.approx(k * np.std(r0), rel=1e-6)
    f = st.transform_bars(df, {"kind": "constant", "bars": 100, "start": 700})
    blk = f.iloc[710:810]
    assert blk[["open", "high", "low", "close"]].nunique().max() == 1
    assert (f.iloc[:710]["close"] == df.iloc[:710]["close"]).all()                 # nothing before the used bars
    j = st.transform_bars(df, {"kind": "jumps", "start": 700})
    assert np.abs(np.diff(np.log(j["close"]))).max() > 1.0 and len(j) == len(df)
    assert np.allclose(j["close"].iloc[:700], df["close"].iloc[:700])


def test_the_cases_run_as_engine_scenarios_with_runs_and_verdicts_in_the_store_and_its_index(tmp_path, bars_csv):
    res = fake_harness(tmp_path, bars_csv, cases=["control", "scale_x10", "fault_nan_term", "long_memory_1440_lr5"],
                        seeds=[0, 1, 2])
    store = RunStore(tmp_path / "runs")
    rows = store.index.rows(res.scenario)
    assert len(rows) == 9 and {r["configuration"] for r in rows} == {"control", "scale_x10", "fault_nan_term"}
    by = {r["configuration"]: r for r in rows}
    assert by["fault_nan_term"]["status"] == "failed" and "UnstableTrainingError" in by["fault_nan_term"]["error"]
    assert "crps_loss" in by["fault_nan_term"]["error"]
    for r in rows:
        sc = store.index.scores(r["run_id"])
        assert sc["stability/passed"] == 1.0, (r["cell_key"], sc)         # the verdict is in the index
        assert (store.root / r["run_dir"] / st.VERDICT_FILE).is_file()
    assert (store.root / by["scale_x10"]["run_dir"]).parent.name == res.scenario
    # the scaled case trained on its own transformed file, which the dataset fingerprint records
    assert by["scale_x10"]["dataset_sha256"] != by["control"]["dataset_sha256"]
    # a rebuilt index equals the kept one: the verdicts live in the run directories
    kept = store.index.dump()
    store.rebuild_index()
    assert store.index.dump() == kept
    text = res.report.read_text(encoding="utf-8")
    assert res.report == tmp_path / "runs" / "stability" / res.harness_id / "REPORT.md"
    for case in ("control", "scale_x10", "fault_nan_term"):
        assert f"| {case} |" in text
    assert res.passed and "crps_loss" in text and "Defined, not run here" in text and "GPU, NT-051" in text
    assert [c.id for c in res.not_run] == ["long_memory_1440_lr5"] and len(rows) == 9     # defined, never run


def test_a_failing_clean_case_fails_the_report_names_the_term_and_writes_a_failing_region(tmp_path, bars_csv):
    res = fake_harness(tmp_path, bars_csv, behaviour={"horizons_5_60_240": "nonfinite"},
                       cases=["control", "horizons_5_60_240"], seeds=[0, 1, 2])
    assert not res.passed and res.case_passed == {"control": True, "horizons_5_60_240": False}
    text = res.report.read_text(encoding="utf-8")
    row = next(ln for ln in text.splitlines() if ln.startswith("| horizons_5_60_240 |"))
    assert "FAIL" in row and "crps_loss" in row and "nonfinite_step_rate" in row and "0/3" in row
    doc = json.loads((res.out_dir / "failing_regions.json").read_text(encoding="utf-8"))
    (region,) = doc["regions"]
    assert region["case"] == "horizons_5_60_240" and region["report"].endswith("REPORT.md")
    assert region["conditions"]["HORIZON_STEPS"] == {"values": [[5, 60, 240]]}
    assert "N_FOLDS" not in region["conditions"]                         # layout is not part of the region
    # the regions file is what the guard reads: a config at that point is refused, naming region and report
    cfg = Config(HORIZON_STEPS=[5, 60, 240], EXTENDED_TREND_PERIODS=[5, 60, 240])
    with pytest.raises(InvalidConfigurationError, match=r"stability:horizons_5_60_240.*REPORT\.md"):
        check_config(cfg, load_regions(res.out_dir / "failing_regions.json"))


def test_an_undetected_fault_fails_its_case(tmp_path, bars_csv):
    res = fake_harness(tmp_path, bars_csv, behaviour={"fault_nan_term": "quiet"}, cases=["fault_nan_term"], seeds=[0])
    assert not res.passed
    (v,) = res.verdicts
    assert [c.name for c in v.failed_checks] == ["fault_stopped_run", "fault_names_term"] or \
        "fault_stopped_run" in [c.name for c in v.failed_checks]


def _verdict(tmp_path, case_id="control", **kw):
    """evaluate_run on a hand-made run directory."""
    d = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    Config().to_yaml(d / "config.yaml")
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "seed": 0, "engine": {"cell_key": d.name}}))
    (d / "result.json").write_text(json.dumps({"status": "done", "scores": kw.pop("scores", {})}))
    write_run_telemetry(d, **kw)
    case = {c.id: c for c in st.default_cases()}[case_id]
    return st.evaluate_run(d, case, st.load_thresholds())


@pytest.mark.parametrize("kw, check", [
    ({"nonfinite": 1.0}, "nonfinite_step_rate"),
    ({"clip_main": 20.0}, "clipped_share_main"),
    ({"clip_ind": 20.0}, "clipped_share_indicator"),
    ({"var_floor": 100.0}, "var_at_floor_share"),
    ({"var_floor": 100.0, "horizons": ("h0", "h3")}, "var_at_floor_share"),
    ({"masked": {"nll_loss": 1.0}}, "masked_term_steps"),
    ({"scores": {"h1/variance/coverage90": 0.2, "h1/n_eff": 100}}, "coverage90"),
    ({"losses": [1.0, 3.0]}, "loss_over_first"),
    ({"losses": [5000.0, 4000.0]}, "max_abs_loss"),
    ({"scores": {"h1/variance/nll": 25.0, "baseline/const_var/h1/variance/nll": 25.0}}, "variance_nll"),
    ({"scores": {"h1/variance/nll": 9.0, "baseline/const_var/h1/variance/nll": 6.0}}, "variance_nll_over_const"),
    ({"scores": {"h1/variance/crps": 16.0, "baseline/const_var/h1/variance/crps": 10.0}}, "variance_crps_over_const"),
    ({"scores": {"h1/variance/nll": float("inf")}}, "variance_nll"),
])
def test_each_pre_registered_threshold_fails_its_check(tmp_path, kw, check):
    v = _verdict(tmp_path, **kw)
    assert not v.passed and check in [c.name for c in v.failed_checks], [c.to_dict() for c in v.checks]
    assert _verdict(tmp_path).passed                                       # the clean run passes every check


def test_gradient_shares_and_periods_at_the_bound_are_reported_never_failing_and_the_probe_still_blames(tmp_path):
    """Report-only (thresholds file): NT-098 has not measured shares; D-037: a period at the data bound is reported."""
    v = _verdict(tmp_path, probe={"crps_trunk": 0.97, "nll_trunk": 0.03}, periods={"period/ma_0": 2.0})
    by = {c.name: c for c in v.checks}
    assert v.passed and not v.failed_checks
    assert by["term_gradient_share"].report_only and not by["term_gradient_share"].passed
    assert by["periods_at_bound"].report_only and not by["periods_at_bound"].passed
    assert v.blamed == ["crps"]
    assert _verdict(tmp_path, probe={"crps_trunk": float("nan")}).passed


def test_a_run_without_n_steps_is_not_evaluated_for_the_step_rate_and_a_short_run_for_divergence(tmp_path):
    v = _verdict(tmp_path, n_steps=0.0, nonfinite=0.0)
    by = {c.name: c for c in v.checks}
    assert v.passed and not by["nonfinite_step_rate"].evaluated and not by["loss_over_first"].evaluated


def test_coverage_is_judged_only_on_horizons_with_enough_effective_samples(tmp_path):
    low = {"h2/variance/coverage90": 0.037, "h2/n_eff": 2}
    v = _verdict(tmp_path, scores=low)
    cov = {c.name: c for c in v.checks}["coverage90"]
    assert v.passed and not cov.evaluated and "h2" in cov.detail                  # n_eff 2: sampling noise, not a failure
    ok = {"h2/variance/coverage90": 0.037, "h2/n_eff": 200, "h1/variance/coverage90": 0.9, "h1/n_eff": 5}
    v = _verdict(tmp_path, scores=ok)
    assert not v.passed and "coverage90" in [c.name for c in v.failed_checks]


HEALTHY = REPO / "tests" / "fixtures" / "nt038_healthy"


def _copy_fixture(name, tmp_path):
    d = tmp_path / name
    shutil.copytree(HEALTHY / name, d)
    return d


def test_the_final_thresholds_pass_every_stored_healthy_run_including_ones_with_a_period_at_the_bound(tmp_path):
    """Seven real finished runs (capacity_v1 x4, reference_default, micro_horizons, micro_ohlcv_duel; light files copied
    from their run directories): every one passes the pre-registered file. Two have periods at the data bound, three
    predate n_steps."""
    names = sorted(p.name for p in HEALTHY.iterdir())
    assert len(names) == 7
    case = {c.id: c for c in st.default_cases()}["control"]
    T = st.load_thresholds()
    bound = 0
    for n in names:
        v = st.evaluate_run(_copy_fixture(n, tmp_path), case, T)
        assert v.passed, (n, [c.to_dict() for c in v.failed_checks])
        by = {c.name: c for c in v.checks}
        assert by["variance_nll"].evaluated and by["coverage90"].evaluated
        bound += int(by["periods_at_bound"].value > 0)
    assert bound >= 2


def _broken(tmp_path, name, scale_loss=1.0, nll=None, crps=None, n_eff=None, nll_const=None):
    d = _copy_fixture("capacity_control_f95", tmp_path / name)
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    rows = [json.loads(ln) for ln in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()][:1]
    for r in rows:
        r["loss"] *= scale_loss
        r["val_loss"] *= scale_loss
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    for h in ("h0", "h1", "h2"):
        if nll is not None:
            res["scores"][f"{h}/variance/nll"] = nll
            res["scores"][f"baseline/const_var/{h}/variance/nll"] = nll if nll_const is None else nll_const
        if crps is not None:
            res["scores"][f"{h}/variance/crps"] = crps
            res["scores"][f"baseline/const_var/{h}/variance/crps"] = crps
        if n_eff is not None:
            res["scores"][f"{h}/n_eff"] = n_eff
    (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"val_loss": rows[0]["val_loss"], "weights_val_loss": rows[0]["val_loss"]}))
    return st.evaluate_run(d, {c.id: c for c in st.default_cases()}["control"], st.load_thresholds())


def test_qas_two_synthetic_broken_runs_fail_while_the_losses_stay_finite(tmp_path):
    """(1) losses x1e30 (finite) with variance NLL and CRPS at 1e12; (2) a variance head pinned at the cap (NLL 1e3,
    the constant baseline equally bad, nothing at the floor)."""
    b1 = _broken(tmp_path, "b1", scale_loss=1e30, nll=1e12, crps=1e12, n_eff=10)
    assert not b1.passed
    assert {"max_abs_loss", "variance_nll"} <= {c.name for c in b1.failed_checks}
    assert {c.name: c for c in b1.checks}["loss_finite"].passed                       # still finite
    b2 = _broken(tmp_path, "b2", nll=1000.0)
    assert not b2.passed and [c.name for c in b2.failed_checks] == ["variance_nll"]
    assert {c.name: c for c in b2.checks}["var_at_floor_share"].value == 0.0


# ------------------------------------------------------------------ repair round 1: the fuzz cases, the profiles
def _variant_config(profile, case_id, csv):
    spec = st.build_scenario([c for c in st.default_cases() if c.id == case_id], name="stab-x", profile=profile,
                             seeds=[0], case_csv={case_id: str(csv)})
    return Config(**{**spec["overrides"], **spec["variants"][case_id]}, FOLD_INDEX=-2)


@pytest.mark.parametrize("profile", ["tiny", "reference"])
def test_the_fuzz_transforms_are_inside_the_bars_the_run_sees_and_change_its_training_windows(tmp_path, profile,
                                                                                            bars_csv):
    from neural_trade.data.processor import split_arrays

    if profile == "reference":
        csv = REPO / "binance_btcusdt_1min_ccxt.csv"
        if not csv.exists():
            pytest.skip("the bundled CSV is not present")
    else:
        csv = bars_csv
    cases = [c for c in st.default_cases() if c.id in ("control", "fuzz_constant", "fuzz_jumps")]
    data = st.write_case_data(cases, csv, tmp_path, profile)
    base = split_arrays(_variant_config(profile, "control", csv))["train"]
    flat = split_arrays(_variant_config(profile, "fuzz_constant", data["fuzz_constant"]))["train"]
    jump = split_arrays(_variant_config(profile, "fuzz_jumps", data["fuzz_jumps"]))["train"]
    rng = lambda X: np.ptp(X, axis=1)                                                    # noqa: E731
    assert (rng(flat["X"]) == 0).sum() >= 10 and (rng(base["X"]) == 0).sum() == 0     # flat training windows exist
    assert np.abs(np.diff(np.log(jump["X"]), axis=1)).max() > 1.0                       # a spike or jump in a window
    assert np.abs(np.diff(np.log(base["X"]), axis=1)).max() < 0.2
    assert not np.allclose(flat["y"], base["y"]) and not np.allclose(jump["y"], base["y"])   # the targets feel it


def test_every_runnable_case_plans_under_both_profiles_without_training(tmp_path, bars_csv):
    cases = [c for c in st.default_cases() if c.runnable]
    planned = st.plan_cases(profile="tiny", csv=bars_csv, seeds=[0])
    assert {p.cell.configuration.name for p in planned} == {c.id for c in cases}
    csv = REPO / "binance_btcusdt_1min_ccxt.csv"
    if not csv.exists():
        pytest.skip("the bundled CSV is not present")
    planned = st.plan_cases(profile="reference", csv=csv, seeds=[0])
    assert {p.cell.configuration.name for p in planned} == {c.id for c in cases}
    assert all(p.role == "dev" for p in planned)


def test_the_harness_scenario_runs_the_shipped_path_with_calibration():
    spec = st.build_scenario([c for c in st.default_cases() if c.runnable], name="stab-x", profile="tiny", seeds=[0],
                             case_csv={})
    assert spec["run"]["calibrate"] is True


def test_a_configuration_inside_a_known_failing_region_can_be_retested_and_judged(tmp_path, bars_csv, regions_file):
    regions_file(Region("known", {"HORIZON_STEPS": {"values": [[5, 60, 240]]}}, "horizons_5_60_240", RPT, "r"))
    res = fake_harness(tmp_path, bars_csv, cases=["horizons_5_60_240"], seeds=[0])
    assert res.passed and [v.passed for v in res.verdicts] == [True]
    with pytest.raises(InvalidConfigurationError, match="known"):
        Config(HORIZON_STEPS=[5, 60, 240], EXTENDED_TREND_PERIODS=[5, 60, 240])           # normal use stays refused


def test_slow_periods_at_the_bound_write_no_failing_region(tmp_path, bars_csv):
    class Bound(HarnessFake):
        def __call__(self, ctx, *, calibrate, save_artifacts):
            out = super().__call__(ctx, calibrate=calibrate, save_artifacts=save_artifacts)
            write_run_telemetry(ctx.run_dir, periods={"period/ma_0": 2.0})
            return out
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs",
                         case_ids=["slow_periods_proxy_lr5", "slow_periods_proxy_lr1"], seeds=[0], trainer=Bound())
    assert res.passed and res.regions == []


def test_an_int_range_with_a_step_keeps_its_lattice_when_a_region_trims_it(regions_file):
    regions_file(Region("small", {"BATCH_SIZE": {"max": 128}}, "c", RPT, "r"),
                 Region("tiny-dim", {"T_PERP_DIM": {"max": 4}}, "c", RPT, "r"))
    space = _space(BATCH_SIZE={"low": 64, "high": 512, "step": 64}, T_PERP_DIM={"low": 4, "high": 20, "step": 4})
    bs, td = space.params
    assert (bs.low, bs.high) == (192.0, 512.0) and (td.low, td.high) == (8.0, 20.0)
    rng = np.random.default_rng(0)
    pts = [space.sample(rng) for _ in range(400)]
    assert {p["BATCH_SIZE"] for p in pts} == {192, 256, 320, 384, 448, 512}            # the lattice 64k survives
    assert {p["T_PERP_DIM"] for p in pts} == {8, 12, 16, 20}
    regions_file(Region("big", {"BATCH_SIZE": {"min": 400}}, "c", RPT, "r"))
    assert _space(BATCH_SIZE={"low": 64, "high": 512, "step": 64}).params[0].high == 384.0


def test_non_finite_inputs_are_mentioned_when_a_head_or_most_terms_are_non_finite_at_once():
    msg, terms = describe({"head_price_h0": 3.0, "point_loss": 3.0, "total_loss": 3.0}, 3.0, 1)
    assert "inputs may be non-finite" in msg and "head_price_h0" in terms
    many = {t: 1.0 for t in ("point_loss", "dir_loss", "nll_loss", "crps_loss", "vol_loss", "total_loss")}
    assert "inputs may be non-finite" in describe(many, 1.0, 1)[0]
    assert "inputs may be non-finite" not in describe({"crps_loss": 1.0, "total_loss": 1.0}, 1.0, 1)[0]


def test_the_regions_file_is_found_from_the_working_directory_when_the_source_tree_has_none(tmp_path, monkeypatch):
    import neural_trade.core.guard as guard

    monkeypatch.delenv(REGIONS_ENV, raising=False)
    monkeypatch.setattr(guard, "DEFAULT_REGIONS_FILE", tmp_path / "nowhere" / "stability_failing_regions.json")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    write_regions(tmp_path / "configs" / "stability_failing_regions.json", [Region("cwd", {"LR": {"min": 0.0}})])
    assert [r.id for r in guard.load_regions()] == ["cwd"]


def test_the_memory_warning_says_where_its_evidence_comes_from_and_that_it_is_unvalidated_for_ohlcv():
    msg = memory_warning(Config(LOOKBACK=240, BATCH_SIZE=512))
    assert "close-only" in msg and "unvalidated for the OHLCV default" in msg
    assert "close-only" in (REPO / "src" / "neural_trade" / "core" / "guard.py").read_text(encoding="utf-8")
    assert "close-only" in (REPO / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")


@pytest.mark.stability
@pytest.mark.slow
def test_the_fuzz_cases_change_the_epoch_metrics_against_the_control_in_a_real_tiny_run(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs",
                         case_ids=["control", "fuzz_constant", "fuzz_jumps"], seeds=[0])
    # the control passes; a fuzz case may legitimately FAIL (a flat block breaks the tiny model: a valid outcome),
    # what it must do is act on the run, so its epoch metrics differ from the control's
    assert res.case_passed["control"], res.report.read_text(encoding="utf-8")
    store = RunStore(tmp_path / "runs")
    loss = {}
    for r in store.index.rows(res.scenario):
        m = json.loads((store.root / r["run_dir"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        loss[r["configuration"]] = (m["loss"], m["val_loss"])
    assert loss["fuzz_constant"] != loss["control"] and loss["fuzz_jumps"] != loss["control"]
    assert abs(loss["fuzz_constant"][0] - loss["control"][0]) > 1e-6 and abs(loss["fuzz_jumps"][0] - loss["control"][0]) > 1e-6



def test_a_nan_loss_fails_loss_finite(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    Config().to_yaml(d / "config.yaml")
    (d / "meta.json").write_text(json.dumps({"run_id": "r", "seed": 0, "engine": {"cell_key": "r"}}))
    (d / "result.json").write_text(json.dumps({"status": "done", "scores": {}}))
    write_run_telemetry(d)
    (d / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "loss": None, "val_loss": 0.9, "n_steps": 5}))
    v = st.evaluate_run(d, st.default_cases()[0], st.load_thresholds())
    assert "loss_finite" in [c.name for c in v.failed_checks]


def test_the_stability_subcommand_refuses_an_unknown_case_and_describes_itself(capsys):
    from neural_trade.cli import main

    assert main(["stability", "--cases", "no_such_case"]) == 2
    with pytest.raises(SystemExit):
        main(["stability", "--help"])
    out = capsys.readouterr().out
    assert "--profile" in out and "stability_thresholds.yaml" in out


# ------------------------------------------------------------------ (4) an unstable run stops, naming the term
def test_blame_names_the_terms_that_fired_and_total_loss_only_when_alone():
    assert blame({"crps_loss": 2.0, "total_loss": 2.0, "nll_loss": 0.0}) == ["crps_loss"]
    assert blame({"total_loss": 1.0}) == ["total_loss"] and blame({"crps_loss": 0.0}) == []
    msg, terms = describe({"crps_loss": 1.0, "total_loss": 1.0}, 1.0, 3)
    assert terms == ["crps_loss"] and "epoch 3" in msg and "crps_loss (1 step(s))" in msg
    msg, terms = describe({}, 2.0, 1)
    assert terms == [] and "every loss term finite" in msg


@pytest.mark.stability
def test_a_nan_loss_term_stops_the_run_with_an_error_naming_the_term(make_loss_model, monkeypatch):
    import tensorflow as tf

    import neural_trade.losses.functions as lf

    m = make_loss_model(config=Config(LAMBDA_CRPS=1.0, STRICT_LOSS_MASKS=True))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(1)
    B = 32
    batch = (tf.constant(rng.normal(size=(B, 60)).astype(np.float32)), tf.constant(rng.normal(size=(B, 3)).astype(np.float32)),
             tf.constant((110000 + rng.normal(0, 500, (B, 1))).astype(np.float32)),
             tf.constant(rng.normal(0, 200, (B, 3)).astype(np.float32)))
    guard = StabilityGuard()
    guard.set_model(m)
    m.train_step(batch)                                                      # a clean step: nothing to report
    guard.on_epoch_end(0, {**m.train_epoch_logs(), "loss": 1.0})
    monkeypatch.setattr(lf, "crps_gaussian_loss", lambda *a, **k: tf.constant(float("nan"), dtype=tf.float32))
    m2 = make_loss_model(config=Config(LAMBDA_CRPS=1.0, STRICT_LOSS_MASKS=True))
    m2.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    guard.set_model(m2)
    m2.train_step(batch)
    with pytest.raises(UnstableTrainingError, match="crps_loss") as exc:
        guard.on_epoch_end(0, {**m2.train_epoch_logs(), "loss": 1.0})
    assert exc.value.terms == ("crps_loss",) and exc.value.epoch == 1


@pytest.mark.stability
def test_a_clean_epoch_does_not_raise_and_a_nan_gradient_with_finite_losses_is_reported_unattributed(make_loss_model, monkeypatch):
    import tensorflow as tf

    import neural_trade.losses.functions as lf

    rng = np.random.default_rng(2)
    B = 32
    batch = (tf.constant(rng.normal(size=(B, 60)).astype(np.float32)), tf.constant(rng.normal(size=(B, 3)).astype(np.float32)),
             tf.constant((110000 + rng.normal(0, 500, (B, 1))).astype(np.float32)),
             tf.constant(rng.normal(0, 200, (B, 3)).astype(np.float32)))
    m = make_loss_model(config=Config(STRICT_LOSS_MASKS=True))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    g = StabilityGuard()
    g.set_model(m)
    m.train_step(batch)
    g.on_epoch_end(0, {**m.train_epoch_logs(), "loss": 1.0})                  # nothing fired: returns

    @tf.custom_gradient
    def nan_grad(x):
        return x, lambda dy: dy * float("nan")

    orig = lf.point_huber
    monkeypatch.setattr(lf, "point_huber", lambda model, yt, yp, last_close_scaled=None, delta=None:
                        nan_grad(orig(model, yt, yp, last_close_scaled=last_close_scaled, delta=delta)))
    m3 = make_loss_model(config=Config(STRICT_LOSS_MASKS=True))
    m3.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    g.set_model(m3)
    m3.train_step(batch)
    with pytest.raises(UnstableTrainingError, match="every loss term finite") as exc:
        g.on_epoch_end(0, {**m3.train_epoch_logs(), "loss": 1.0})
    assert exc.value.terms == ()


def test_train_cell_adds_the_guard_in_strict_mode_only(monkeypatch, tmp_path):
    import neural_trade.training.trainer as trainer
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.experiments.runner import train_cell

    seen = {}
    monkeypatch.setattr(trainer, "train_and_evaluate", lambda **kw: seen.update(kw) or "result")
    for strict in (False, True):
        ctx = RunContext.create(Config(STRICT_LOSS_MASKS=strict), root=tmp_path / str(strict), write_env=False)
        assert train_cell(ctx, calibrate=False, save_artifacts=False) == "result"
        cbs = seen["extra_callbacks"]
        assert (cbs is not None and isinstance(cbs[0], StabilityGuard)) == strict


def test_the_sweep_records_an_unstable_run_as_failed_naming_the_term(tmp_path, bars_csv):
    pytest.importorskip("optuna")

    class Raising(SweepFakeTrainer):
        def __call__(self, ctx, *, calibrate, save_artifacts):
            eng = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]
            if eng["variant"] == "t0001":
                msg, terms = describe({"crps_loss": 2.0, "total_loss": 2.0}, 2.0, 1)
                raise UnstableTrainingError(msg, terms, 1, 2.0)
            return super().__call__(ctx, calibrate=calibrate, save_artifacts=save_artifacts)

    sw = make_sweep(tmp_path, bars_csv, Raising(), n_trials=3, top_k=5)
    res = sw.run()
    by = {t["number"]: t for t in res.trials}
    assert by[1]["state"] == "FAIL" and by[1]["value"] is None
    assert "UnstableTrainingError" in by[1]["reason"] and "crps_loss" in by[1]["reason"]
    assert by[0]["state"] == by[2]["state"] == "COMPLETE"
    # and the leaderboard-side aggregation reads the same failed cell the same way
    rows = [{"fold": -2, "role": "dev", "status": "failed", "sharpe_net": None, "seed": 0,
             "error": "UnstableTrainingError: unstable training in epoch 1 ... blamed: crps_loss (1 step(s))"}]
    s = dev_net_sharpe(rows, [-2])
    assert s.value is None and "crps_loss" in s.reason


def test_run_health_names_the_masked_loss_term_beside_the_non_finite_steps(tmp_path):
    write_run_telemetry(tmp_path, nonfinite=2.0, masked={"nll_loss": 2.0, "total_loss": 2.0})
    why = run_health(tmp_path)
    assert "nonfinite_grad_steps" in why and "non-finite loss term(s): nll_loss" in why
    write_run_telemetry(tmp_path, masked={"nll_loss": 2.0})                    # a masked term alone fails nothing (unchanged)
    assert run_health(tmp_path) is None


# ------------------------------------------------------------------ (5) failing regions: the guard and the search spaces
@pytest.fixture
def regions_file(tmp_path, monkeypatch):
    def make(*regions):
        p = write_regions(tmp_path / "regions.json", list(regions))
        monkeypatch.setenv(REGIONS_ENV, str(p))
        return p
    return make


RPT = "runs/stability/20261006T000000Z-tiny/REPORT.md"


def test_validate_refuses_a_config_inside_a_failing_region_naming_the_region_and_the_report(regions_file):
    regions_file(Region("lr-high", {"LR": {"min": 0.05}, "BATCH_SIZE": {"values": [64, 128]}}, "scale_x10", RPT,
                        "the loss went non-finite"))
    Config(LR=0.05, BATCH_SIZE=256).validate()                  # batch outside the region
    Config(LR=0.01, BATCH_SIZE=64).validate()                   # LR outside it
    with pytest.raises(InvalidConfigurationError) as exc:
        Config(LR=0.06, BATCH_SIZE=64).validate()
    msg = str(exc.value)
    assert "lr-high" in msg and RPT in msg and "scale_x10" in msg and "non-finite" in msg


def test_the_shipped_regions_file_is_empty_and_the_guard_can_be_switched_off_or_suspended(regions_file, monkeypatch):
    assert load_regions(REPO / "configs" / "stability_failing_regions.json") == []
    p = regions_file(Region("all", {"LR": {"min": 0.0}}, "c", RPT, "r"))
    with pytest.raises(InvalidConfigurationError, match="all"):
        Config().validate()
    with regions_disabled():
        Config().validate()                                      # the harness re-tests failing configurations
    monkeypatch.setenv(REGIONS_ENV, "off")
    Config().validate()
    monkeypatch.setenv(REGIONS_ENV, str(p))
    with regions_disabled():
        cfg = Config()
    assert find_region(cfg).id == "all"


def test_region_conditions_cover_bounds_values_and_lists_and_a_bad_region_is_refused():
    r = Region.from_dict({"id": "x", "conditions": {"HORIZON_STEPS": {"values": [[5, 60, 240]]}, "LR": {"max": 0.001}}})
    assert r.contains({"HORIZON_STEPS": [5, 60, 240], "LR": 0.0005})
    assert not r.contains({"HORIZON_STEPS": [5, 60, 241], "LR": 0.0005}) and not r.contains({"HORIZON_STEPS": [5, 60, 240]})
    with pytest.raises(InvalidConfigurationError, match="no conditions"):
        Region.from_dict({"id": "x", "conditions": {}})
    with pytest.raises(InvalidConfigurationError, match="min / max / values"):
        Region.from_dict({"id": "x", "conditions": {"LR": {"above": 1}}})


def test_batch_size_times_lookback_squared_beyond_the_measured_good_level_warns_and_never_refuses(monkeypatch):
    import neural_trade.core.guard as guard

    seen = []
    monkeypatch.setattr(guard.logger, "warning", lambda msg, *a: seen.append(msg % a if a else msg))
    assert memory_warning(Config()) is None and memory_warning(Config(LOOKBACK=240, BATCH_SIZE=256)) is None
    assert memory_warning(Config(LOOKBACK=60, BATCH_SIZE=2048)) is None                     # the micro layout of D-041
    assert 256 * 240 ** 2 <= SCORE_ELEMENTS_WARN < 512 * 240 ** 2                           # between the measured cases
    for lb, bs in ((240, 512), (240, 2048)):
        assert "ran out of memory" in memory_warning(Config(LOOKBACK=lb, BATCH_SIZE=bs))
    Config(LOOKBACK=240, BATCH_SIZE=512).validate()                                          # a warning, not a refusal
    assert any("BATCH_SIZE 512 x LOOKBACK 240^2" in m for m in seen)


def _space(**search):
    ov = {"CSV_PATH": "x.csv", "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 2, "BATCH_SIZE": 256}
    return SearchSpace.from_scenario(Scenario.from_dict(scenario_dict("x.csv", search=search, overrides=ov)))


def test_a_failing_region_at_the_end_of_a_range_trims_the_search_space_exactly(regions_file):
    regions_file(Region("lr-high", {"LR": {"min": 0.005}}, "c", RPT, "r"),
                 Region("small-batch", {"BATCH_SIZE": {"max": 63}}, "c", RPT, "r"))
    space = _space(LR={"low": 1e-4, "high": 1e-2, "log": True}, BATCH_SIZE={"low": 32, "high": 512})
    lr, bs = space.params
    assert lr.high < 0.005 and lr.high == pytest.approx(0.005, rel=1e-9) and lr.low == 1e-4
    assert bs.low == 64 and bs.high == 512 and space.regions == ()
    rng = np.random.default_rng(0)
    pts = [space.sample(rng) for _ in range(300)]
    assert all(p["LR"] < 0.005 and p["BATCH_SIZE"] >= 64 for p in pts)


def test_an_interior_or_multi_field_region_is_rejected_by_sampling_and_by_suggest(regions_file):
    regions_file(Region("corner", {"LR": {"min": 0.005}, "LAMBDA_DIR": {"min": 1.5}}, "c", RPT, "r"),
                 Region("fixed-elsewhere", {"LR": {"min": 0.0}, "EPOCHS": {"min": 1000}}, "c", RPT, "r"))
    space = _space(**SEARCH)
    assert [r.id for r in space.regions] == ["corner"]            # the region on a field the search does not touch cannot be reached
    rng = np.random.default_rng(1)
    pts = [space.sample(rng) for _ in range(300)]
    assert not any(p["LR"] >= 0.005 and p["LAMBDA_DIR"] >= 1.5 for p in pts)
    assert any(p["LR"] >= 0.005 for p in pts) and any(p["LAMBDA_DIR"] >= 1.5 for p in pts)

    class Trial:                                                   # an Optuna-like trial proposing inside the region
        def suggest_float(self, name, low, high, log=False):
            return {"LR": 0.007, "LAMBDA_DIR": 1.8}[name]
    with pytest.raises(FailingRegionHit, match="corner"):
        space.suggest(Trial())


def test_a_region_covering_the_whole_range_refuses_the_search(regions_file):
    regions_file(Region("everything", {"LR": {"min": 0.0}}, "c", RPT, "r"))
    with pytest.raises((SweepError, InvalidConfigurationError)):
        _space(**SEARCH)


def test_a_sweep_never_runs_a_configuration_inside_a_failing_region(tmp_path, bars_csv, regions_file):
    pytest.importorskip("optuna")
    regions_file(Region("corner", {"LR": {"min": 0.005}, "LAMBDA_DIR": {"min": 1.5}}, "c", RPT, "r"))
    trainer = SweepFakeTrainer()
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=6, top_k=2)
    res = sw.run()
    assert res.state in ("complete", "stopped")
    for row in RunStore(tmp_path / "runs").index.rows(sw.sweep_id):
        cfg = Config.from_yaml(Path(tmp_path / "runs") / row["run_dir"] / "config.yaml")
        assert not (cfg.LR >= 0.005 and cfg.LAMBDA_DIR >= 1.5), row["cell_key"]


# ------------------------------------------------------------------ (3) a tiny real CPU run (stability marker)
@pytest.mark.stability
@pytest.mark.slow
def test_a_tiny_cpu_harness_run_passes_a_clean_case_and_stops_a_faulted_one_naming_the_loss_term(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", case_ids=["control", "fault_nan_term"],
                         seeds=[0])
    assert res.case_passed == {"control": True, "fault_nan_term": True}, res.report.read_text(encoding="utf-8")
    store = RunStore(tmp_path / "runs")
    rows = {r["configuration"]: r for r in store.index.rows(res.scenario)}
    assert rows["control"]["status"] == "done" and rows["fault_nan_term"]["status"] == "failed"
    assert "crps_loss" in rows["fault_nan_term"]["error"] and "UnstableTrainingError" in rows["fault_nan_term"]["error"]
    meta = json.loads((store.root / rows["control"]["run_dir"] / "config.yaml").read_text(encoding="utf-8")) \
        if False else yaml.safe_load((store.root / rows["control"]["run_dir"] / "config.yaml").read_text(encoding="utf-8"))
    assert meta["STRICT_LOSS_MASKS"] is True                                  # the harness ran in strict mode
    assert store.index.scores(rows["control"]["run_id"])["stability/passed"] == 1.0


@pytest.mark.stability
@pytest.mark.slow
def test_a_tiny_cpu_run_with_a_nan_in_the_input_and_one_with_a_nan_gradient_are_stopped(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs",
                         case_ids=["fault_nan_input", "fault_nan_gradient"], seeds=[0])
    assert res.case_passed == {"fault_nan_input": True, "fault_nan_gradient": True}, res.report.read_text(encoding="utf-8")
