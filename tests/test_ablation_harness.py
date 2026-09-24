"""Ablation harness (plan C4): grid construction, resume, verdict rules, and one real cell."""
from __future__ import annotations

import csv
import io
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from neural_trade.experiments.ablation import (SUMMARY_FIELDS, AblationSpec, Criteria, analyze, dry_run, load_rows,
                                               run_grid, summarize_dir, value_withdrawn)

REPO = Path(__file__).resolve().parent.parent
TERMS = {"LAMBDA_T_PERP": 0.1, "LAMBDA_CASIMIR": 0.1, "LAMBDA_HD": 0.1, "LAMBDA_IFE": 0.1,
         "LAMBDA_VAC_OVERFLOW": 0.1, "LAMBDA_VAC": 0.5}
CRIT = Criteria.from_dict({
    "min_agree_frac": 5 / 6,
    "primary": {"default": [{"metric": "h1/variance/crpss", "higher_is_better": True, "mde": 0.005}],
                "family": [{"metric": "h1/variance/crpss", "higher_is_better": True, "mde": 0.005}]},
    "guardrails": [{"metric": "h1/delta/rmse", "higher_is_better": False, "tolerance": 2.0}],
})


def _spec(**kw):
    return AblationSpec(name="t", terms=dict(TERMS), scales={"smoke": {"EPOCHS": 1}}, calibrate="none", **kw)


def _fake_runner(effects, rmse_effects=None, calls=None):
    """crpss = 0.02 + sum(effect of every term that is on) + a (seed, period) shift shared by all
    conditions + tiny idiosyncratic noise."""
    def run(spec, cell, scale, out_dir, frozen):
        if calls is not None:
            calls.append(cell.key)
        on = [t for t, v in cell.condition.overrides.items() if v > 0]
        rng = np.random.default_rng(zlib.crc32(cell.key.encode()))
        shared = 0.01 * cell.seed + (0.003 if cell.period == "P2" else 0.0)
        crpss = 0.02 + sum(effects.get(t, 0.0) for t in on) + shared + rng.normal(0, 1e-4)
        rmse = 230.0 + sum((rmse_effects or {}).get(t, 0.0) for t in on) + rng.normal(0, 0.01)
        return {"key": cell.key, "condition": cell.condition.name, "mode": cell.condition.mode,
                "term": cell.condition.term, "seed": cell.seed, "period": cell.period, "run_id": f"run-{cell.key}",
                "wall_s": 60.0, "epochs_run": 1, "h1/variance/crpss": crpss, "h1/delta/rmse": rmse}
    return run


def test_grid_is_14_conditions_by_seeds_by_periods():
    spec = _spec()
    conds = spec.conditions()
    assert len(conds) == 14 and len(spec.cells()) == 84 and len({c.key for c in spec.cells()}) == 84
    only = next(c for c in conds if c.name == "only:LAMBDA_HD")
    assert only.overrides["LAMBDA_HD"] == 0.1 and sum(v > 0 for v in only.overrides.values()) == 1
    without = next(c for c in conds if c.name == "without:LAMBDA_VAC")
    assert without.overrides["LAMBDA_VAC"] == 0.0 and without.overrides["LAMBDA_CASIMIR"] == 0.1
    ov = spec.overrides_for(spec.cells()[0], "smoke", {"LAMBDA_DIR": 2.5})
    assert ov["EPOCHS"] == 1 and ov["LAMBDA_DIR"] == 2.5 and "SEED" in ov and "FOLD_INDEX" in ov
    with pytest.raises(ValueError, match="unknown scale"):
        spec.overrides_for(spec.cells()[0], "huge")


def test_spec_files_load_and_reject_unknown_keys(tmp_path):
    spec = AblationSpec.from_yaml(REPO / "configs" / "ablation_physics.yaml")
    crit = Criteria.from_yaml(REPO / "configs" / "ablation_criteria.yaml")
    assert len(spec.cells()) == 84 and set(spec.scales) == {"smoke", "full"}
    assert crit.for_term("LAMBDA_IFE")[0].metric == "h1/direction/mcc" and crit.for_term(None)
    from neural_trade.core.config import Config

    Config().override(**{k: v for k, v in spec.overrides_for(spec.cells()[0], "full").items() if k != "EPOCHS"})
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nterms: {LAMBDA_HD: 0.1}\nseedz: [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seedz"):
        AblationSpec.from_yaml(bad)


def test_verdicts_resume_limit_and_reports(tmp_path):
    spec = _spec()
    calls = []
    runner = _fake_runner({"LAMBDA_T_PERP": 0.02, "LAMBDA_CASIMIR": -0.02}, calls=calls)
    run_grid(spec, "smoke", tmp_path, runner=runner, limit=10, log=lambda *_: None)
    assert len(calls) == 10
    info = dry_run(spec, tmp_path)
    assert info["done"] == 10 and info["pending"] == 74 and info["projected_hours"] == pytest.approx(74 / 60)
    run_grid(spec, "smoke", tmp_path, runner=runner, log=lambda *_: None)
    assert len(calls) == 84 and len(set(calls)) == 84  # resumed: nothing ran twice
    run_grid(spec, "smoke", tmp_path, runner=runner, log=lambda *_: None)
    assert len(calls) == 84

    analysis = summarize_dir(spec, CRIT, tmp_path, "smoke")
    v = {t: a["verdict"] for t, a in analysis["terms"].items()}
    assert v["LAMBDA_T_PERP"] == "VALUE" and v["LAMBDA_CASIMIR"] == "HARMFUL"
    assert v["LAMBDA_HD"] == v["LAMBDA_IFE"] == v["LAMBDA_VAC"] == "NEUTRAL"
    assert analysis["family"]["verdict"] == "NEUTRAL"   # +0.02 - 0.02
    for name in ("results.csv", "summary.csv", "analysis.json", "report.md"):
        assert (tmp_path / name).exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "`LAMBDA_T_PERP` | VALUE | VALUE | **VALUE**" in report and "run-all_on__s0__P1" in report


def test_guardrail_breach_withdraws_value(tmp_path):
    spec = _spec()
    run_grid(spec, "smoke", tmp_path, runner=_fake_runner({"LAMBDA_HD": 0.02}, rmse_effects={"LAMBDA_HD": 5.0}),
             log=lambda *_: None)
    a = analyze(load_rows(tmp_path), spec, CRIT)
    hd = a["terms"]["LAMBDA_HD"]
    assert hd["verdict"] == "INCONCLUSIVE"
    assert hd["modes"]["leave_one_in"]["guardrail_breaches"][0]["metric"] == "h1/delta/rmse"


def test_report_lists_the_family_breach_and_says_why_value_was_withdrawn(tmp_path):
    # T_PERP lifts crpss (so the family's crpss is VALUE); HD worsens the rmse guard-rail by 5 > 2 in
    # every condition that has it on, all_on included: the family's VALUE is withdrawn by that breach
    spec = _spec()
    run_grid(spec, "smoke", tmp_path, runner=_fake_runner({"LAMBDA_T_PERP": 0.02}, rmse_effects={"LAMBDA_HD": 5.0}),
             log=lambda *_: None)
    a = summarize_dir(spec, CRIT, tmp_path, "smoke")
    fam = a["family"]
    assert fam["verdict"] == "INCONCLUSIVE" and [c["verdict"] for c in fam["metrics"]] == ["VALUE"]
    assert value_withdrawn(fam) and fam["guardrail_breaches"][0]["metric"] == "h1/delta/rmse"
    raw = (tmp_path / "report.md").read_bytes()
    assert b"\r\n" not in raw                                          # LF, also when written on Windows
    report = raw.decode("utf-8")
    # the family's breach gets its own BREACH row, as the terms' breaches do
    assert re.search(r"^\| family \| all_on vs all_off \| guard-rail `h1/delta/rmse` \| \| -[45]\.\d{4} \| \| \| "
                     r"tol 2 \| \| BREACH \|$", report, re.M)
    assert re.search(r"^\| `LAMBDA_HD` \| leave_one_in \| guard-rail `h1/delta/rmse` \|.*\| BREACH \|$", report, re.M)
    # the verdict table says why: VALUE withdrawn (family), a breach that changed nothing (HD)
    assert "| family (all_on vs all_off) | | | **INCONCLUSIVE** (VALUE withdrawn: guard-rail `h1/delta/rmse` " \
           "breached) |" in report
    assert "| `LAMBDA_HD` | NEUTRAL (guard-rail `h1/delta/rmse` breached) | " \
           "NEUTRAL (guard-rail `h1/delta/rmse` breached) | **NEUTRAL** |" in report
    assert "`LAMBDA_T_PERP` | VALUE | VALUE | **VALUE**" in report
    assert "(5 of 6)" in report and "withdraws a VALUE" in report


def test_summary_csv_lists_every_guardrail_breach_as_the_report_does(tmp_path):
    # same grid as above: HD breaches the rmse guard-rail in both of its modes and in the family
    spec = _spec()
    run_grid(spec, "smoke", tmp_path, runner=_fake_runner({"LAMBDA_T_PERP": 0.02}, rmse_effects={"LAMBDA_HD": 5.0}),
             log=lambda *_: None)
    a = summarize_dir(spec, CRIT, tmp_path, "smoke")
    raw = (tmp_path / "summary.csv").read_bytes()
    assert b"\r\n" not in raw and b"\r\n" not in (tmp_path / "results.csv").read_bytes()   # LF (tracked files)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    assert list(rows[0]) == SUMMARY_FIELDS and SUMMARY_FIELDS[-1] == "tolerance"   # the old columns keep their order
    breaches = [r for r in rows if r["verdict"] == "BREACH"]
    got = {(r["term"], r["mode"], r["metric"], r["treatment"], r["control"]) for r in breaches}
    assert got == {("LAMBDA_HD", "leave_one_in", "h1/delta/rmse", "only:LAMBDA_HD", "all_off"),
                   ("LAMBDA_HD", "leave_one_out", "h1/delta/rmse", "all_on", "without:LAMBDA_HD"),
                   ("family", "all_on_vs_all_off", "h1/delta/rmse", "all_on", "all_off")}
    fam = next(r for r in breaches if r["term"] == "family")
    assert float(fam["tolerance"]) == 2.0 and np.isclose(float(fam["mean_delta"]),
                                                         a["family"]["guardrail_breaches"][0]["mean_delta"])
    assert fam["mode_verdict"] == fam["term_verdict"] == "INCONCLUSIVE" and fam["n_pairs"] == fam["mde"] == ""
    assert all(r["tolerance"] == "" for r in rows if r["verdict"] != "BREACH")
    n_metrics = sum(len(m["metrics"]) for t in a["terms"].values() for m in t["modes"].values()) \
        + len(a["family"]["metrics"])
    assert len(rows) - len(breaches) == n_metrics
    # analysis.json records the pre-registered share the verdicts used (the figure's subtitle reads it)
    saved = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert saved["min_agree_frac"] == a["min_agree_frac"] == CRIT.min_agree_frac


def test_tracked_ablation_summary_and_analysis_carry_breaches_and_the_agree_share():
    d = REPO / "runs" / "ablations" / "ablate_physics_v1-full"
    if not (d / "analysis.json").exists() or not (d / "summary.csv").exists():
        pytest.skip("no tracked ablation outputs")
    a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    assert a["min_agree_frac"] == Criteria.from_yaml(REPO / "configs" / "ablation_criteria.yaml").min_agree_frac
    rows = list(csv.DictReader(io.StringIO((d / "summary.csv").read_text(encoding="utf-8"))))
    want = {(term, mode, b["metric"]) for term, t in a["terms"].items() for mode, m in t["modes"].items()
            for b in m["guardrail_breaches"]}
    want |= {("family", "all_on_vs_all_off", b["metric"]) for b in (a.get("family") or {}).get("guardrail_breaches", [])}
    assert want and {(r["term"], r["mode"], r["metric"]) for r in rows if r["verdict"] == "BREACH"} == want
    assert sum(r["verdict"] == "BREACH" for r in rows) == len(want)


def test_value_withdrawn_only_when_the_breach_changed_the_verdict():
    breach = [{"metric": "h1/direction/auc", "mean_delta": -0.0119, "tolerance": 0.01}]
    vs = lambda *v: [{"verdict": x} for x in v]                        # noqa: E731
    assert value_withdrawn({"verdict": "INCONCLUSIVE", "metrics": vs("VALUE", "INCONCLUSIVE"),
                            "guardrail_breaches": breach})
    assert not value_withdrawn({"verdict": "INCONCLUSIVE", "metrics": vs("NEUTRAL", "INCONCLUSIVE"),
                                "guardrail_breaches": breach})           # not VALUE before the breach
    assert not value_withdrawn({"verdict": "NEUTRAL", "metrics": vs("NEUTRAL"), "guardrail_breaches": breach})
    assert not value_withdrawn({"verdict": "INCONCLUSIVE", "metrics": vs("VALUE", "INCONCLUSIVE"),
                                "guardrail_breaches": []})
    assert not value_withdrawn({"verdict": "HARMFUL", "metrics": vs("VALUE", "HARMFUL"), "guardrail_breaches": breach})


def test_tracked_ablation_report_lists_every_breach_of_its_analysis():
    d = REPO / "runs" / "ablations" / "ablate_physics_v1-full"
    if not (d / "analysis.json").exists() or not (d / "report.md").exists():
        pytest.skip("no tracked ablation report")
    a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    report = (d / "report.md").read_text(encoding="utf-8")
    blocks = [(f"`{term}`", mode, m) for term, t in a["terms"].items() for mode, m in t["modes"].items()]
    if a.get("family"):
        blocks.append(("family", "all_on vs all_off", a["family"]))
    n = 0
    for label, mode, m in blocks:
        for b in m["guardrail_breaches"]:
            n += 1
            assert (f"| {label} | {mode} | guard-rail `{b['metric']}` | | {b['mean_delta']:+.4f} | | | "
                    f"tol {b['tolerance']:g} | | BREACH |") in report, (label, mode, b["metric"])
    assert n == report.count("| BREACH |")
    if a.get("family") and value_withdrawn(a["family"]):
        assert "| family (all_on vs all_off) | | | **INCONCLUSIVE** (VALUE withdrawn: guard-rail" in report


def test_too_few_pairs_is_inconclusive(tmp_path):
    spec = _spec(seeds=[0], periods={"P1": -1})
    run_grid(spec, "smoke", tmp_path, runner=_fake_runner({"LAMBDA_T_PERP": 0.05}), log=lambda *_: None)
    a = summarize_dir(spec, CRIT, tmp_path, "smoke")
    assert a["terms"]["LAMBDA_T_PERP"]["verdict"] == "INCONCLUSIVE"


@pytest.mark.slow
def test_one_real_cell_and_calibrate_once(tmp_path, synthetic_bars, monkeypatch):
    from neural_trade.experiments.ablation import calibrate_once, execute_cell

    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    spec = AblationSpec(name="real", terms={"LAMBDA_HD": 0.1, "LAMBDA_T_PERP": 0.1}, modes=["all_on", "all_off"],
                        seeds=[0], periods={"P": -1}, calibrate="once", strategy="liberal",
                        base_overrides={"BATCH_SIZE": 32, "CALLBACKS": ["early_stopping"]},
                        scales={"smoke": {"MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1}})
    frozen = calibrate_once(spec, "smoke", tmp_path, str(csv))
    assert frozen and "LAMBDA_DIR" in frozen and "LAMBDA_HD" not in frozen
    assert calibrate_once(spec, "smoke", tmp_path, str(csv)) == frozen  # cached
    row = execute_cell(spec, spec.cells()[0], "smoke", tmp_path, frozen, str(csv))
    for key in ("h1/variance/crpss", "h1/direction/auc", "backtest/n_trades", "coherence/coherence_primary"):
        assert key in row, key
    assert row["condition"] == "all_on" and (tmp_path / "runs" / row["run_id"] / "eval_report_test.json").exists()
