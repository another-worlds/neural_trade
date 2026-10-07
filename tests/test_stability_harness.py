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


# one probe sample as the trainer logs it: per variable group the term shares sum to 1
PROBE_SAMPLE = {"crps_trunk": 0.7, "point_trunk": 0.3, "crps_head": 0.5, "point_head": 0.5,
                "nll_indicator": 0.5, "point_indicator": 0.5}


class HarnessFake:
    """The engine trainer stand-in: healthy telemetry and fake predictions, except per case ``behaviour``:
    'raise' (a fault stops the run, naming crps_loss), 'quiet' (a fault is NOT detected), 'nonfinite' (a clean run
    with non-finite steps and a masked term)."""

    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []
        self.probes = []         # (case, PROBE_GRADIENTS, PROBE_EVERY) of every call

    def __call__(self, ctx, *, calibrate, save_artifacts):
        eng = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]
        case = eng["configuration"]
        self.calls.append(eng["cell_key"])
        how = self.behaviour.get(case, "raise" if case.startswith("fault_") else "ok")
        self.probes.append((case, bool(ctx.config.PROBE_GRADIENTS), int(ctx.config.PROBE_EVERY)))
        probe = PROBE_SAMPLE if ctx.config.PROBE_GRADIENTS else None
        if how == "oom":                 # a crash, not a verdict (NT-191)
            write_run_telemetry(ctx.run_dir)
            raise MemoryError("out of memory")
        if how == "oserror":             # disk full: the machine, not the setup
            raise OSError(28, "No space left on device")
        if how == "filenotfound":        # a deterministic setup error: a verdict-side failure
            raise FileNotFoundError("the bars file went away")
        if how in ("tf_oom", "tf_other"):   # TensorFlow's InternalError, named as the runner records it
            InternalError = type("InternalError", (RuntimeError,), {})
            raise InternalError("Failed copying input tensor: out of memory" if how == "tf_oom"
                                else "Graph execution error: shape mismatch")
        if how == "probe_crash" and ctx.config.PROBE_GRADIENTS:
            raise MemoryError("out of memory in the probe re-run")
        if how == "probe_crash":
            write_run_telemetry(ctx.run_dir, losses=[1.0, 1e12])
            return fake_result(ctx.config)
        if how == "bigloss":             # fails max_abs_loss and loss_over_first; no term is named by the run
            write_run_telemetry(ctx.run_dir, losses=[1.0, 1e12], probe=probe)
            return fake_result(ctx.config)
        if how == "raise":
            write_run_telemetry(ctx.run_dir)
            msg, terms = describe({"crps_loss": 1.0, "total_loss": 1.0}, 1.0, 1)
            raise UnstableTrainingError(msg, terms, 1, 1.0)
        if how == "nonfinite":
            write_run_telemetry(ctx.run_dir, nonfinite=3.0, masked={"crps_loss": 3.0}, probe=probe)
        else:
            write_run_telemetry(ctx.run_dir, probe=probe)
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
    # NT-187: a minority of them, counted both ways (flat inputs; flat inputs AND every label bar flat)
    n_train = len(flat["X"])
    flat_inputs = rng(flat["X"]) == 0
    flat_all = flat_inputs & np.all(np.asarray(flat["y"]).reshape(n_train, -1) == 0, axis=1)
    assert 0 < flat_all.sum() <= flat_inputs.sum() < n_train / 2, (flat_all.sum(), flat_inputs.sum(), n_train)
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

    assert main(["stability", "--cases", "no_such_case"]) == 64
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


# ------------------------------------------------------------------ NT-187: thresholds v2 and the harness fixes
V1_SHA256 = "0b706aa2c9415a2e7c34ee4183b174c055b6ec965b9aaa9646a6f7abd9e7f36e"
V2_SHA256 = "34a122b28861c13622165aed81fdb1e9405eea91fe4823d0a754d070cbade2cb"
V2_FILE = REPO / "configs" / "stability_thresholds_v2.yaml"


def _v2_with(tmp_path, **changes):
    """A copy of the v2 file with some check values replaced (a gate 'removed' by setting it to 0 or to a huge bound)."""
    import re

    text = V2_FILE.read_text(encoding="utf-8")
    for key, value in changes.items():
        text, n = re.subn(rf"(?m)^  {key}: .*$", f"  {key}: {value}", text)
        assert n == 1, key
    out = tmp_path / f"v2_{len(list(tmp_path.iterdir()))}.yaml"
    out.write_text(text, encoding="utf-8")
    return st.load_thresholds(out)


def _scored(tmp_path, scores, thresholds, base="capacity_control_f95"):
    """evaluate_run on a copy of a stored healthy run whose `scores` are updated with ``scores``."""
    d = _copy_fixture(base, tmp_path / f"s{len(list(tmp_path.iterdir()))}")
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    res["scores"].update(scores)
    (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return st.evaluate_run(d, {c.id: c for c in st.default_cases()}["control"], thresholds)


def _check(v, name):
    return {c.name: c for c in v.checks}[name]


def test_thresholds_v1_is_frozen_still_loads_the_default_and_gives_todays_verdicts(tmp_path):
    t = st.load_thresholds()
    assert t.sha256 == V1_SHA256 == st.file_sha256(REPO / "configs" / "stability_thresholds.yaml")
    assert t.schema_version == 1 and t.path == st.THRESHOLDS_FILE and t.name == "stability_thresholds_v1"
    assert st.resolve_thresholds_path(None, "reference") == st.THRESHOLDS_FILE            # the default stays v1
    # v1 judges as before: no n_eff gate (an n_eff 5 NLL excess of 3.8 fails), the absolute NLL in quote units
    low = {f"h{i}/n_eff": 5.0 for i in range(3)}
    v = _scored(tmp_path, {**low, "h2/variance/nll": 9.0, "baseline/const_var/h2/variance/nll": 5.2}, t)
    assert "variance_nll_over_const" in [c.name for c in v.failed_checks]
    assert not _scored(tmp_path, {"h2/variance/nll": 25.0}, t).passed


def test_thresholds_v2_is_pre_registered_named_by_hash_and_every_check_has_a_rationale():
    t = st.load_thresholds(V2_FILE)
    assert t.sha256 == V2_SHA256 and t.schema_version == 2 and t.name == "stability_thresholds_v2"
    assert "max_variance_nll" not in t.checks and t.check("max_variance_nll_scaled") == 8.0
    text = V2_FILE.read_text(encoding="utf-8")
    for key in t.checks:
        i = text.index(f"  {key}:")
        assert "#" in text[max(0, i - 900):i], f"{key} has no rationale comment above it"
    assert st.load_thresholds().sha256 == V1_SHA256                    # v1 untouched


def test_the_thresholds_file_is_chosen_by_name_by_path_or_by_the_profile_default(tmp_path, bars_csv, monkeypatch):
    assert st.resolve_thresholds_path("v2") == V2_FILE == st.resolve_thresholds_path("stability_thresholds_v2.yaml")
    assert st.resolve_thresholds_path(str(V2_FILE)) == V2_FILE
    with pytest.raises(FileNotFoundError):
        st.resolve_thresholds_path("v9")
    monkeypatch.setitem(st.PROFILE_THRESHOLDS, "tiny", V2_FILE)
    res = fake_harness(tmp_path, bars_csv, cases=["control"], seeds=[0])               # the profile default (v2 here)
    assert res.thresholds_sha256 == V2_SHA256 and V2_SHA256 in res.report.read_text(encoding="utf-8")
    monkeypatch.setitem(st.PROFILE_THRESHOLDS, "tiny", None)
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "r2", case_ids=["control"], seeds=[0],
                         thresholds_path="v2", trainer=HarnessFake())                   # by name
    assert res.thresholds_sha256 == V2_SHA256
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "r3", case_ids=["control"], seeds=[0],
                         trainer=HarnessFake())                                         # no name: v1
    assert res.thresholds_sha256 == V1_SHA256
    text = res.report.read_text(encoding="utf-8")
    assert "n_eff of the scored block per case" in text and "| control | h0 " in text


def test_v2_passes_every_stored_healthy_run_and_still_fails_qas_broken_runs(tmp_path, monkeypatch):
    T = st.load_thresholds(V2_FILE)
    case = {c.id: c for c in st.default_cases()}["control"]
    for n in sorted(p.name for p in HEALTHY.iterdir()):
        v = st.evaluate_run(_copy_fixture(n, tmp_path), case, T)
        assert v.passed, (n, [c.to_dict() for c in v.failed_checks])
        by = {c.name: c for c in v.checks}
        assert by["variance_nll"].evaluated and by["variance_nll_over_const"].evaluated and by["coverage90"].evaluated
        assert 1.0 < by["variance_nll"].value < 3.0                       # scaled units: observed 1.14-2.88
    # QA's two synthetic broken runs and the variance-head cap run (_broken judges with st.load_thresholds())
    monkeypatch.setattr(st, "load_thresholds", lambda *a, **k: T)
    b1 = _broken(tmp_path, "b1", scale_loss=1e30, nll=1e12, crps=1e12, n_eff=200)
    b2 = _broken(tmp_path, "b2", nll=1000.0)
    b3 = _broken(tmp_path, "b3", scale_loss=1e30, nll=1e12, crps=1e12, n_eff=10)
    assert not b1.passed and {"max_abs_loss", "variance_nll"} <= {c.name for c in b1.failed_checks}
    assert not b2.passed and [c.name for c in b2.failed_checks] == ["variance_nll"]
    assert not b3.passed and "max_abs_loss" in [c.name for c in b3.failed_checks]      # caught without the head


def test_the_variance_checks_are_not_evaluated_below_their_n_eff_gates_and_each_gate_matters(tmp_path):
    T = st.load_thresholds(V2_FILE)
    bad = {"h2/variance/nll": 9.0, "baseline/const_var/h2/variance/nll": 5.2,        # +3.8 over the baseline
           "h2/variance/crps": 40.0, "baseline/const_var/h2/variance/crps": 20.0}    # ratio 2.0
    low = _scored(tmp_path, {**bad, "h2/n_eff": 5.0}, T)
    assert low.passed
    for name in ("variance_nll_over_const", "variance_crps_over_const"):
        c = _check(low, name)
        assert "h2 (n_eff 5 <" in c.detail                                  # h0 and h1 are still judged
    # n_eff between the two gates: the excess is still not evaluated, the CRPS ratio is
    mid = _scored(tmp_path, {**bad, "h2/n_eff": 50.0}, T)
    assert "h2 (n_eff 50 <" in _check(mid, "variance_nll_over_const").detail
    assert [c.name for c in mid.failed_checks] == ["variance_crps_over_const"]
    high = _scored(tmp_path, {**bad, "h2/n_eff": 200.0}, T)
    assert {"variance_nll_over_const", "variance_crps_over_const"} == {c.name for c in high.failed_checks}
    # removing a gate (threshold 0) makes the same n_eff 5 run fail: the gate is what keeps it quiet
    no_excess = _scored(tmp_path, {**bad, "h2/n_eff": 5.0}, _v2_with(tmp_path, min_n_eff_variance_excess=0))
    assert [c.name for c in no_excess.failed_checks] == ["variance_nll_over_const"]
    no_scale = _scored(tmp_path, {**bad, "h2/n_eff": 5.0}, _v2_with(tmp_path, min_n_eff_variance=0))
    assert "variance_crps_over_const" in [c.name for c in no_scale.failed_checks]
    # nothing evaluated at all (every horizon below the gates): not evaluated, reported, no failure
    tiny = _scored(tmp_path, {f"h{i}/n_eff": 5.0 for i in range(3)} | bad, T)
    c = _check(tiny, "variance_nll_over_const")
    assert tiny.passed and not c.evaluated and c.value is None and "h0 (n_eff 5 <" in c.detail
    # a missing n_eff is not evaluated either
    missing = _scored(tmp_path, {**bad, "h2/n_eff": None}, T)
    assert missing.passed and "h2 (n_eff missing <" in _check(missing, "variance_nll_over_const").detail


def test_a_degenerate_constant_baseline_is_not_evaluated_and_reported(tmp_path):
    T = st.load_thresholds(V2_FILE)
    absurd = {"baseline/const_var/h1/variance/nll": 1e28, "baseline/const_var/h1/variance/crps": 1e28}
    v = _scored(tmp_path, absurd, T)
    assert v.passed
    for name in ("variance_nll_over_const", "variance_crps_over_const"):
        c = _check(v, name)
        assert "h1: the constant baseline NLL is absurd" in c.detail and c.value is not None   # h0 and h2 still judged
    # the guard is what makes it so: with the bound lifted the absurd baseline is judged (as an excess of -1e28)
    lifted = _scored(tmp_path, absurd, _v2_with(tmp_path, max_const_baseline_nll_scaled='1.0e+300'))
    assert "absurd" not in _check(lifted, "variance_nll_over_const").detail
    # non-finite or non-positive baselines
    for k, bad_value in (("baseline/const_var/h0/variance/nll", float("nan")),
                         ("baseline/const_var/h0/variance/crps", 0.0),
                         ("baseline/const_var/h0/variance/crps", float("inf"))):
        w = _scored(tmp_path, {k: bad_value}, T)
        assert w.passed and "h0: the constant baseline is not finite" in _check(w, "variance_nll_over_const").detail
    # every horizon degenerate: not evaluated, no failure, the scaled absolute NLL still judged
    allbad = {f"baseline/const_var/h{i}/variance/nll": float("nan") for i in range(3)}
    w = _scored(tmp_path, allbad, T)
    c = _check(w, "variance_nll_over_const")
    assert w.passed and not c.evaluated and c.value is None and _check(w, "variance_nll").evaluated


def test_the_absolute_nll_is_in_scaled_units_and_does_not_move_with_the_price_level(tmp_path):
    T1, T2 = st.load_thresholds(), st.load_thresholds(V2_FILE)
    k = 1e6                                          # prices x1e6: every quote-unit number moves by ln k = 13.8
    shift = float(np.log(k))
    base = "micro_horizons_h4h"
    res = json.loads((HEALTHY / base / "result.json").read_text(encoding="utf-8"))["scores"]
    moved = {}
    for h in ("h0", "h1", "h2"):
        moved[f"{h}/variance/nll"] = res[f"{h}/variance/nll"] + shift
        moved[f"baseline/const_var/{h}/variance/nll"] = res[f"baseline/const_var/{h}/variance/nll"] + shift
        moved[f"{h}/delta/rmse_zero"] = res[f"{h}/delta/rmse_zero"] * k
        moved[f"{h}/variance/crps"] = res[f"{h}/variance/crps"] * k
        moved[f"baseline/const_var/{h}/variance/crps"] = res[f"baseline/const_var/{h}/variance/crps"] * k
    v2_orig = _scored(tmp_path, {}, T2, base)
    v2_moved = _scored(tmp_path, moved, T2, base)
    assert v2_moved.passed
    assert _check(v2_moved, "variance_nll").value == pytest.approx(_check(v2_orig, "variance_nll").value)
    v1_moved = _scored(tmp_path, moved, T1, base)
    assert not v1_moved.passed and "variance_nll" in [c.name for c in v1_moved.failed_checks]    # the v1 artefact
    # the cap run: NLL 1e3 with an equally bad baseline fails the scaled limit; without it, nothing catches it
    cap = {f"h{i}/variance/nll": 1000.0 for i in range(3)}
    cap.update({f"baseline/const_var/h{i}/variance/nll": 1000.0 for i in range(3)})
    assert [c.name for c in _scored(tmp_path, cap, T2, base).failed_checks] == ["variance_nll"]
    assert _scored(tmp_path, cap, _v2_with(tmp_path, max_variance_nll_scaled='1.0e+9'), base).passed
    # a horizon without a price-change scale cannot be scaled: not evaluated
    noscale = _scored(tmp_path, {f"h{i}/delta/rmse_zero": 0.0 for i in range(3)}, T2, base)
    assert not _check(noscale, "variance_nll").evaluated and noscale.passed


def test_fuzz_constant_is_a_block_of_60_to_120_bars_and_the_case_design_is_pre_registered():
    case = {c.id: c for c in st.default_cases()}["fuzz_constant"]
    t = st.load_thresholds(V2_FILE)
    assert case.data["bars"] == st.FUZZ_CONSTANT_BARS == t.case_design["fuzz_constant_bars"] == 100
    assert 60 <= st.FUZZ_CONSTANT_BARS <= 120 and "100-bar" in case.description


def test_the_expected_n_eff_table_of_thresholds_v2_matches_the_engine_planner():
    csv = REPO / "binance_btcusdt_1min_ccxt.csv"
    if not csv.exists():
        pytest.skip("the bundled CSV is not present")
    t = st.load_thresholds(V2_FILE)
    for profile in ("tiny", "reference"):
        exp = t.expected_n_eff[profile]
        planned = {p.cell.configuration.name: p for p in st.plan_cases(
            profile=profile, csv=csv, case_ids=["control", "fuzz_constant", "horizons_5_60_240"], seeds=[0])}
        for case in ("control", "fuzz_constant"):
            p = planned[case]
            n = p.fold["blocks"]["test"]["n"]
            assert n == exp["scored_windows_default"] and p.fold["blocks"]["train"]["n"] == exp["train_windows_default"]
            assert [n // h for h in p.config.HORIZON_STEPS] == exp["default_cases"]
        p = planned["horizons_5_60_240"]
        wide = exp["horizons_5_60_240"]
        assert p.fold["blocks"]["train"]["n"] == wide["train_windows"]
        assert p.fold["blocks"]["test"]["n"] == wide["scored_windows"]
        assert [wide["scored_windows"] // h for h in p.config.HORIZON_STEPS] == wide["n_eff"]


@pytest.mark.stability
@pytest.mark.slow
def test_fuzz_constant_in_a_real_tiny_run_has_no_constant_baseline_artefact_and_still_shows_its_metrics(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", case_ids=["control", "fuzz_constant"],
                         seeds=[0], thresholds_path="v2")
    assert res.thresholds_sha256 == V2_SHA256
    store = RunStore(tmp_path / "runs")
    first = {}
    for r in store.index.rows(res.scenario):
        first[r["configuration"]] = json.loads(
            (store.root / r["run_dir"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["fuzz_constant"]["loss"] != first["control"]["loss"]                    # the case acts on the run
    assert first["fuzz_constant"]["val_loss"] != first["control"]["val_loss"]
    assert first["fuzz_constant"].get("dir_loss", 1.0) > 0.0                             # not a constant training set
    (v,) = [v for v in res.verdicts if v.case == "fuzz_constant"]
    assert not {c.name for c in v.failed_checks} & {"variance_nll_over_const", "variance_crps_over_const"}, \
        [c.to_dict() for c in v.checks]
    assert not _check(v, "variance_nll_over_const").evaluated                            # tiny n_eff 11/7/5 < 100


# ------------------------------------------------------------------ NT-191: probe-off default, non-verdict class, dry run
def _rows(tmp_path, res):
    store = RunStore(tmp_path / "runs")
    return {r["cell_key"]: r for r in store.index.rows(res.scenario)}


def test_the_probe_mode_defaults_to_failed_on_the_reference_profile_and_off_on_tiny():
    assert st.PROFILE_PROBE == {"tiny": "off", "reference": "failed"}
    cases = [c for c in st.default_cases() if c.runnable]
    ov = {m: st.build_scenario(cases, name="x", profile="reference", seeds=[0], case_csv={}, probe=m)["overrides"]
          for m in (None, "off", "failed", "on")}
    assert ov[None]["PROBE_GRADIENTS"] is False and ov["failed"]["PROBE_GRADIENTS"] is False
    assert ov["off"]["PROBE_GRADIENTS"] is False
    assert ov["on"]["PROBE_GRADIENTS"] is True and ov["on"]["PROBE_EVERY"] == 5     # today's behaviour
    assert st.build_scenario(cases, name="x", profile="tiny", seeds=[0], case_csv={})["overrides"][
        "PROBE_GRADIENTS"] is False
    with pytest.raises(ValueError, match="probe"):
        st.build_scenario(cases, name="x", profile="tiny", seeds=[0], case_csv={}, probe="sometimes")


def test_failed_mode_runs_probe_off_then_reruns_only_the_failing_cell_once_with_the_probe_and_blames_from_it(
        tmp_path, bars_csv):
    fake = HarnessFake({"horizons_5_60_240": "bigloss"})
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0, 1],
                         case_ids=["control", "horizons_5_60_240"], trainer=fake, probe="failed")
    primary = [p for p in fake.probes if p[1] is False]
    probed = [p for p in fake.probes if p[1] is True]
    assert len(primary) == 4                                        # every cell first runs with the probe OFF
    assert probed == [("horizons_5_60_240", True, 1)] * 2           # the two failing cells, once each, PROBE_EVERY 1
    assert res.case_passed == {"control": True, "horizons_5_60_240": False}
    assert len(res.reruns) == 2 and all(v.kind == "probe_rerun" and v.probe for v in res.reruns)
    for v in (v for v in res.verdicts if v.case == "horizons_5_60_240"):
        assert not v.passed and v.blamed == ["crps"] and v.blame_source == st.PROBE_SOURCE
        (rr,) = [r for r in res.reruns if r.rerun_of == v.run_id]
        assert rr.run_id != v.run_id
    text = res.report.read_text(encoding="utf-8")
    assert st.PROBE_SOURCE in text and "Probe re-runs" in text
    for v in res.reruns:
        assert v.run_id in text and v.rerun_of in text                  # both runs of a failed cell are listed
    assert "no failing region" in text and "by design" in text          # data and fault cases: stated in the REPORT
    assert (tmp_path / "runs" / "stability" / res.harness_id / "verdicts.json").is_file()


def test_a_cell_that_passes_is_never_rerun_and_modes_off_and_on_do_not_rerun(tmp_path, bars_csv):
    fake = HarnessFake()
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "a", seeds=[0], case_ids=["control"],
                         trainer=fake, probe="failed")
    assert [p[1] for p in fake.probes] == [False] and res.reruns == []
    fake = HarnessFake({"control": "bigloss"})
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "b", seeds=[0], case_ids=["control"],
                         trainer=fake, probe="off")
    assert [p[1] for p in fake.probes] == [False] and res.reruns == []
    (v,) = res.verdicts
    assert v.blamed == [] and "probe off" in v.blame_reason                    # '-' with the reason
    assert "probe off" in res.report.read_text(encoding="utf-8")
    fake = HarnessFake({"control": "bigloss"})
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "c", seeds=[0], case_ids=["control"],
                         trainer=fake, probe="on")
    assert fake.probes == [("control", True, 5)] and res.reruns == []          # `on` = every cell probed, no re-run
    assert res.verdicts[0].blamed == ["crps"] and res.verdicts[0].blame_source == st.PROBE_SOURCE


def test_probe_blame_takes_the_largest_share_of_the_first_epoch_whose_shares_sum_to_one():
    rows = [{"epoch": 0, "probe_grad_share_crps_trunk": 0.0, "probe_grad_share_nll_trunk": 0.0},      # not probed
            {"epoch": 1, "probe_grad_share_crps_trunk": 0.2, "probe_grad_share_nll_trunk": 0.8,
             "probe_grad_share_crps_head": 0.9, "probe_grad_share_nll_head": 0.1},
            {"epoch": 2, "probe_grad_share_crps_trunk": 0.9, "probe_grad_share_nll_trunk": 0.1}]
    assert st.probe_blame(rows) == ("crps", "head", 0.9, 1)
    assert st.probe_blame(rows[:1]) is None
    assert st.probe_blame([{"epoch": 0, "probe_grad_share_crps_trunk": float("nan"),
                            "probe_grad_share_nll_trunk": 1.0}]) is None


def test_the_probe_never_changes_a_cells_verdict_fields_in_the_harness_path(tmp_path, bars_csv):
    off = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "off", seeds=[0], case_ids=["horizons_5_60_240"],
                         trainer=HarnessFake({"horizons_5_60_240": "bigloss"}), probe="off")
    on = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "on", seeds=[0], case_ids=["horizons_5_60_240"],
                        trainer=HarnessFake({"horizons_5_60_240": "bigloss"}), probe="on")
    a, b = off.verdicts[0], on.verdicts[0]
    assert a.passed == b.passed

    def fields(v):          # every check but the report-only probe share, which is the one thing a probe adds
        return [(c.name, c.value, c.limit, c.passed, c.evaluated) for c in v.checks if c.name != "term_gradient_share"]
    assert fields(a) == fields(b)


@pytest.mark.parametrize("error", [MemoryError, OSError, "ResourceExhaustedError", "BrokenProcessPool"])
def test_resource_errors_are_not_a_verdict_and_check_failures_and_unstable_runs_are(error):
    name = error if isinstance(error, str) else error.__name__
    assert st.is_non_verdict_error(name)
    for verdict_error in ("UnstableTrainingError", "ValueError", "InvalidConfigurationError", "", None):
        assert not st.is_non_verdict_error(verdict_error)


def test_deterministic_setup_errors_are_verdicts_and_tf_internal_errors_need_a_resource_message():
    for name in ("FileNotFoundError", "PermissionError", "NotADirectoryError", "IsADirectoryError", "FileExistsError"):
        assert not st.is_non_verdict_error(name, "x"), name
    assert st.is_non_verdict_error("ConnectionError") and st.is_non_verdict_error("TimeoutError")   # OSError subclasses
    for name in ("InternalError", "UnknownError"):
        for msg in ("Failed to allocate memory", "OOM when allocating tensor", "CUDNN_STATUS_EXECUTION_FAILED",
                    "cuDNN launch failure", "CUDA_ERROR_OUT_OF_MEMORY"):
            assert st.is_non_verdict_error(name, msg), (name, msg)
        for msg in ("", "Graph execution error: shape mismatch", "Incompatible shapes"):
            assert not st.is_non_verdict_error(name, msg), (name, msg)


def test_a_tf_internal_error_is_not_a_verdict_only_with_a_resource_message_and_a_missing_file_is(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0],
                         case_ids=["control", "scale_x10", "horizons_5_60_240"],
                         trainer=HarnessFake({"control": "tf_oom", "scale_x10": "tf_other", "horizons_5_60_240": "filenotfound"}))
    assert res.case_status == {"control": "NOT A VERDICT", "scale_x10": "FAIL", "horizons_5_60_240": "FAIL"}


def test_a_crashed_cell_is_not_a_verdict_is_excluded_from_the_counts_and_sets_exit_code_2(tmp_path, bars_csv):
    fake = HarnessFake({"scale_x10": "oom", "fault_nan_term": "raise"})
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0, 1],
                         case_ids=["control", "scale_x10", "fault_nan_term"], trainer=fake)
    assert res.case_status == {"control": "PASS", "scale_x10": "NOT A VERDICT", "fault_nan_term": "PASS"}
    assert res.case_passed["scale_x10"] is False and not res.verdict_failed
    nv = [v for v in res.verdicts if v.non_verdict]
    assert len(nv) == 2 and all(not v.passed and "MemoryError" in v.error for v in nv)
    assert [v for v in res.verdicts if v.case == "fault_nan_term" and v.non_verdict] == []   # Unstable: a verdict
    assert res.exit_code == 2
    text = res.report.read_text(encoding="utf-8")
    assert "NOT A VERDICT" in text and "2 of 3 cases passed" in text
    assert "1 case(s) not a verdict" in text and "0 failed" in text
    assert res.reruns == []                                                   # a crash is never probed


def test_a_verdict_failure_wins_the_exit_code_over_a_crash(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0],
                         case_ids=["control", "scale_x10"],
                         trainer=HarnessFake({"control": "bigloss", "scale_x10": "oserror"}))
    assert res.case_status == {"control": "FAIL", "scale_x10": "NOT A VERDICT"} and res.exit_code == 1
    assert res.regions == []                                                  # no region for a crash


def test_a_run_directory_without_a_result_is_a_worker_crash_and_not_a_verdict(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": "r1", "seed": 3, "engine": {"cell_key": "c"}}), encoding="utf-8")
    v = st.evaluate_run(d, next(c for c in st.default_cases() if c.id == "control"), st.load_thresholds())
    assert v.non_verdict and not v.passed and "worker crash" in v.error and v.checks == []
    assert st.Verdict.from_dict(v.to_dict()).non_verdict


def test_retry_non_verdict_reruns_only_those_cells_as_a_new_launch_and_the_case_uses_the_rerun(tmp_path, bars_csv):
    first = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0, 1],
                           case_ids=["control", "scale_x10"], trainer=HarnessFake({"scale_x10": "oom"}))
    assert first.exit_code == 2
    healthy = HarnessFake()
    second = st.run_harness(profile=None, store=tmp_path / "runs", trainer=healthy,
                            retry_non_verdict_of=first.harness_id)
    assert second.harness_id != first.harness_id and second.retry_of == first.harness_id
    assert sorted(healthy.calls) == sorted(v.cell_key for v in first.verdicts if v.non_verdict)    # only those cells
    assert second.case_status == {"control": "PASS", "scale_x10": "PASS"} and second.exit_code == 0
    assert len(second.verdicts) == 4 and not any(v.non_verdict for v in second.verdicts)
    assert len(second.superseded) == 2 and all(v.non_verdict for v in second.superseded)
    text = second.report.read_text(encoding="utf-8")
    for v in second.superseded:
        assert v.run_id in text                                               # both the crash and the re-run are listed
    for v in second.verdicts:
        assert v.run_id in text
    assert [v.kind for v in second.verdicts if v.case == "scale_x10"] == ["retry", "retry"]
    assert [v.kind for v in second.verdicts if v.case == "control"] == ["primary", "primary"]
    # a retry that crashes again leaves the exit code at 2, with both crashes listed
    third = st.run_harness(profile=None, store=tmp_path / "runs", trainer=HarnessFake({"scale_x10": "oom"}),
                           retry_non_verdict_of=first.harness_id)
    assert third.exit_code == 2 and third.case_status["scale_x10"] == "NOT A VERDICT"


def test_retry_refuses_other_thresholds_a_missing_launch_and_a_launch_with_nothing_to_retry(tmp_path, bars_csv):
    first = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0], case_ids=["control"],
                           trainer=HarnessFake({"control": "oom"}))
    with pytest.raises(ValueError, match="thresholds"):
        st.run_harness(profile=None, store=tmp_path / "runs", thresholds_path="v2", trainer=HarnessFake(),
                       retry_non_verdict_of=first.harness_id)
    with pytest.raises(ValueError, match="no such harness"):
        st.run_harness(profile=None, store=tmp_path / "runs", trainer=HarnessFake(), retry_non_verdict_of="nope")
    with pytest.raises(ValueError, match="profile"):
        st.run_harness(profile="reference", store=tmp_path / "runs", trainer=HarnessFake(),
                       retry_non_verdict_of=first.harness_id)
    ok = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs2", seeds=[0], case_ids=["control"],
                        trainer=HarnessFake())
    with pytest.raises(ValueError, match="nothing to retry"):
        st.run_harness(profile=None, store=tmp_path / "runs2", trainer=HarnessFake(), retry_non_verdict_of=ok.harness_id)


def test_the_stability_command_exit_codes_are_0_1_2_and_it_has_the_new_options(tmp_path, bars_csv, monkeypatch, capsys):
    from neural_trade.cli import main

    real = st.run_harness

    def run(behaviour, *extra):
        monkeypatch.setattr(st, "run_harness", lambda **kw: real(trainer=HarnessFake(behaviour), **kw))
        capsys.readouterr()
        code = main(["stability", "--csv", str(bars_csv), "--store", str(tmp_path / "s"), "--cases", "control",
                     "--seeds", "0", *extra])
        return code, json.loads(capsys.readouterr().out)

    code, out = run({})
    assert code == 0 and out["case_status"] == {"control": "PASS"}
    assert run({"control": "bigloss"})[0] == 1
    code, out = run({"control": "oom"})
    assert code == 2 and out["non_verdict_cells"] == 1
    capsys.readouterr()
    monkeypatch.setattr(st, "run_harness", lambda **kw: real(trainer=HarnessFake(), **kw))
    code = main(["stability", "--store", str(tmp_path / "s"), "--retry-non-verdict", out["harness_id"]])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["case_status"] == {"control": "PASS"}
    with pytest.raises(SystemExit):
        main(["stability", "--help"])
    help_text = capsys.readouterr().out
    assert "--probe" in help_text and "--retry-non-verdict" in help_text


def test_a_refused_command_exits_64_not_2_and_a_nothing_to_retry_refusal_creates_no_run_directory(tmp_path, bars_csv,
                                                                                               capsys, monkeypatch):
    from neural_trade.cli import main

    assert st.EXIT_REFUSED == 64
    store = tmp_path / "s"
    assert main(["stability", "--csv", str(bars_csv), "--store", str(store), "--cases", "no_such_case"]) == 64
    assert main(["stability", "--store", str(store), "--retry-non-verdict", "nope"]) == 64
    assert not (store / "stability").exists() or not any((store / "stability").iterdir())     # nothing was created
    real = st.run_harness
    monkeypatch.setattr(st, "run_harness", lambda **kw: real(trainer=HarnessFake({"control": "oom"}), **kw))
    assert main(["stability", "--csv", str(bars_csv), "--store", str(store), "--cases", "control", "--seeds", "0"]) == 2
    capsys.readouterr()
    # a clean launch has nothing to retry: refused (64), and no new launch directory appears
    monkeypatch.setattr(st, "run_harness", lambda **kw: real(trainer=HarnessFake(), **kw))
    assert main(["stability", "--csv", str(bars_csv), "--store", str(tmp_path / "ok"), "--cases", "control",
                 "--seeds", "0"]) == 0
    before = sorted(p.name for p in (tmp_path / "ok" / "stability").iterdir())
    assert main(["stability", "--store", str(tmp_path / "ok"), "--retry-non-verdict"]) == 64
    assert sorted(p.name for p in (tmp_path / "ok" / "stability").iterdir()) == before


def test_the_probe_reruns_of_a_launch_are_capped_and_the_rest_are_listed_as_not_rerun(tmp_path, bars_csv):
    fake = HarnessFake({"horizons_5_60_240": "bigloss"})
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0, 1, 2],
                         case_ids=["control", "horizons_5_60_240"], trainer=fake, probe="failed", max_probe_reruns=1)
    assert [p for p in fake.probes if p[1] is True] == [("horizons_5_60_240", True, 1)]       # one re-run only
    assert len(res.reruns) == 1 and len(res.not_rerun) == 2 and res.case_status["horizons_5_60_240"] == "FAIL"
    text = res.report.read_text(encoding="utf-8")
    assert st.NOT_RERUN_CAP in text and "Failed cells not re-run" in text
    for v in res.not_rerun:
        assert v.run_id in text and v.blamed == [] and st.NOT_RERUN_CAP in v.blame_reason
    doc = json.loads((res.out_dir / "verdicts.json").read_text(encoding="utf-8"))
    assert len(doc["not_rerun"]) == 2 and doc["max_probe_reruns"] == 1
    none = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "r0", seeds=[0], case_ids=["horizons_5_60_240"],
                          trainer=HarnessFake({"horizons_5_60_240": "bigloss"}), probe="failed", max_probe_reruns=0)
    assert none.reruns == [] and len(none.not_rerun) == 1
    with pytest.raises(ValueError, match="max_probe_reruns"):
        st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "rn", seeds=[0], case_ids=["control"],
                       trainer=HarnessFake(), max_probe_reruns=-1)
    from neural_trade.cli import build_parser

    assert build_parser().parse_args(["stability"]).max_probe_reruns == st.MAX_PROBE_RERUNS == 10


def test_a_failed_cells_rerun_never_replaces_its_verdict_and_a_crashed_rerun_says_so(tmp_path, bars_csv):
    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0], case_ids=["control"],
                         trainer=HarnessFake({"control": "bigloss"}), probe="failed")
    (v,) = res.verdicts
    (rr,) = res.reruns
    assert v.kind == "primary" and not v.probe and rr.run_id != v.run_id and rr.rerun_of == v.run_id
    assert rr.run_id not in [x.run_id for x in res.verdicts] and not v.passed        # the first run's verdict counts
    crashed = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs2", seeds=[0], case_ids=["control"],
                             trainer=HarnessFake({"control": "probe_crash"}), probe="failed")
    (v,), (rr,) = crashed.verdicts, crashed.reruns
    assert rr.non_verdict and not v.passed and v.blamed == [] and crashed.case_status == {"control": "FAIL"}
    assert "re-run crashed" in v.blame_reason and "out of memory" in v.blame_reason
    assert "re-run crashed" in crashed.report.read_text(encoding="utf-8")


def test_a_retry_keeps_the_probe_mode_of_its_launch_and_refuses_another(tmp_path, bars_csv):
    first = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0], case_ids=["control"],
                           trainer=HarnessFake({"control": "oom"}), probe="off")
    for other in ("on", "failed"):
        with pytest.raises(ValueError, match="retry keeps it"):
            st.run_harness(profile=None, store=tmp_path / "runs", trainer=HarnessFake(), probe=other,
                           retry_non_verdict_of=first.harness_id)
    assert sorted(p.name for p in (tmp_path / "runs" / "stability").iterdir()) == [first.harness_id]    # no new launch
    again = st.run_harness(profile=None, store=tmp_path / "runs", trainer=HarnessFake(), probe="off",
                           retry_non_verdict_of=first.harness_id)
    assert again.probe_mode == "off" and again.exit_code == 0


def test_dry_run_honours_seeds_and_thresholds_and_prints_n_eff_probe_mode_and_steps(tmp_path, bars_csv, capsys):
    from neural_trade.cli import main

    csv = REPO / "binance_btcusdt_1min_ccxt.csv"
    if not csv.exists():
        pytest.skip("the bundled CSV is not present")
    capsys.readouterr()
    code = main(["stability", "--dry-run", "--profile", "reference", "--csv", str(csv), "--seeds", "4,5",
                 "--thresholds", "v2", "--cases", "control,horizons_5_60_240"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["probe"] == "failed" and out["thresholds_sha256"] == st.load_thresholds(V2_FILE).sha256
    cells = out["cells"]
    assert [(c["case"], c["seed"]) for c in cells] == [("control", 4), ("control", 5), ("horizons_5_60_240", 4),
                                                       ("horizons_5_60_240", 5)]
    ctrl = cells[0]
    assert ctrl["n_eff"] == {"h0": 150, "h1": 100, "h2": 75} and ctrl["expected_n_eff"] == [150, 100, 75]
    assert ctrl["n_eff_matches_expected"] is True and ctrl["probe"] == "failed"
    assert ctrl["epochs"] == 3 and ctrl["steps_per_epoch"] == 2 and ctrl["steps"] == 6        # 360 windows / batch 256
    wide = cells[2]
    assert wide["n_eff"] == {"h0": 450, "h1": 37, "h2": 9} and wide["n_eff_matches_expected"] is True
    # v1 carries no table: the planner's n_eff is printed alone
    code = main(["stability", "--dry-run", "--profile", "tiny", "--csv", str(bars_csv), "--cases", "control",
                 "--probe", "on"])
    out = json.loads(capsys.readouterr().out)
    assert [c["seed"] for c in out["cells"]] == [0, 1, 2] and out["cells"][0]["expected_n_eff"] is None
    assert out["cells"][0]["probe"] == "on" and out["cells"][0]["epochs"] == 1
    with pytest.raises(FileNotFoundError):
        main(["stability", "--dry-run", "--thresholds", "no_such_file", "--cases", "control"])


def test_the_data_csvs_are_untracked_but_their_sha256_stays_in_each_cells_meta(tmp_path, bars_csv):
    import subprocess

    res = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "runs", seeds=[0],
                         case_ids=["control", "scale_x10"], trainer=HarnessFake())
    rows = _rows(tmp_path, res)
    meta = json.loads((tmp_path / "runs" / rows["scale_x10__f-2__s0"]["run_dir"] / "meta.json").read_text(encoding="utf-8"))
    ctl = json.loads((tmp_path / "runs" / rows["control__f-2__s0"]["run_dir"] / "meta.json").read_text(encoding="utf-8"))
    data = res.out_dir / "data" / "scale_x10.csv"
    assert data.is_file() and meta["dataset"]["sha256"] == st.file_sha256(data)
    assert ctl["dataset"]["sha256"] == st.file_sha256(bars_csv) != meta["dataset"]["sha256"]
    for path in ("runs/stability/20261007T000000Z-reference/data/scale_x10.csv", "runs/stability/x/data/y.csv"):
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
        assert out.returncode == 0, f"{path} is not ignored"
    for path in ("runs/stability/x/REPORT.md", "runs/experiments/x/results.csv"):     # everything else stays tracked
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode == 1, path


@pytest.mark.stability
@pytest.mark.slow
def test_the_probe_does_not_change_training_epoch_metrics_are_bitwise_equal_and_verdicts_equal(tmp_path, bars_csv,
                                                                                                 monkeypatch):
    """The acceptance of NT-191 (1): the same real tiny cell with the probe off and on. `loss` and every `val_*` key are
    bitwise equal; the per-term training sums agree within 2 float32 ULP (the probe graph adds nll_h0+nll_h1+nll_h2 in
    another order: nll_loss 5.028296947 against 5.028297424). Two runs of ONE setup differ
    in the 7th digit unless DETERMINISTIC_GRU is on (NT-114: 5.028296947 against 5.028297901 in nll_loss, probe off in
    both), so the comparison runs with it, which makes two probe-off runs bitwise equal."""
    monkeypatch.setitem(st.PROFILES["tiny"], "DETERMINISTIC_GRU", True)
    off = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "off", case_ids=["control"], seeds=[0],
                         probe="off")
    on = st.run_harness(profile="tiny", csv=bars_csv, store=tmp_path / "on", case_ids=["control"], seeds=[0],
                        probe="on")
    from neural_trade.telemetry.epoch_logger import read_metrics

    def metrics(res, root):
        (row,) = RunStore(root).index.rows(res.scenario)
        return read_metrics(root / row["run_dir"] / "metrics.jsonl")

    a, b = metrics(off, tmp_path / "off"), metrics(on, tmp_path / "on")
    assert len(a) == len(b) >= 1
    assert any(k.startswith("probe_") for k in b[0]) and not any(k.startswith("probe_") for k in a[0])

    def skip(k):
        # the probe's own keys, wall-clock keys and run identifiers (strings) are not training numbers
        return k.startswith("probe_") or "time" in k or "sec" in k or "seconds" in k or isinstance(a[0].get(k), str)

    ulp = np.float32(1.1920929e-07)         # one float32 ULP at 1.0: the probe graph sums nll_h0+nll_h1+nll_h2 in
    # another order (nll_loss differs by 1 ULP, 5.028296947 against 5.028297424; loss and every val_* key do not)
    for ra, rb in zip(a, b):
        ka = {k for k in ra if not skip(k)}
        assert ka == {k for k in rb if not skip(k)}
        for k in sorted(ka):
            if k == "loss" or k.startswith("val_"):
                assert ra[k] == rb[k], f"{k}: {ra[k]!r} != {rb[k]!r}"             # bitwise: the total and every validation key
            else:                                                                  # per-term training sums: within 2 ULP
                assert abs(ra[k] - rb[k]) <= 2 * float(ulp) * max(abs(ra[k]), abs(rb[k]), 1.0), f"{k}: {ra[k]!r} != {rb[k]!r}"
    va, vb = off.verdicts[0], on.verdicts[0]
    assert va.passed == vb.passed
    assert [(c.name, c.value, c.passed) for c in va.checks if c.name != "term_gradient_share"] == \
        [(c.name, c.value, c.passed) for c in vb.checks if c.name != "term_gradient_share"]
