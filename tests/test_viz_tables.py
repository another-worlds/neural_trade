"""The analytics tables: every number checked against an independent computation, and readable output."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.visualization import analytics_tables as AT
from neural_trade.visualization import stats as S

BETAS = {"h0": 0.2, "h1": 0.02, "h2": 0.25}


def _served(frame, betas=BETAS):
    served = copy.deepcopy(frame)
    raw = {h: np.asarray(frame.delta[h], float).copy() for h in HORIZONS}
    served.delta = {h: betas[h] * raw[h] for h in HORIZONS}
    return served, raw


def _windows_frame(n=3000, seed=0, contrarian=True):
    """Closes on a random walk; the realised move over each horizon is minus the trailing move (or plus)."""
    rng = np.random.default_rng(seed)
    close = 100_000 + np.cumsum(rng.normal(0, 30, n + 60))
    X = np.stack([close[i:i + 60] for i in range(n)])
    trail = np.stack([X[:, -1] - X[:, -1 - p] for p in (10, 15, 20)], 1)
    y = (-1 if contrarian else 1) * trail + rng.normal(0, 1, (n, 3))
    delta = {h: 0.5 * trail[:, i] for i, h in enumerate(HORIZONS)}
    prob = {h: np.full(n, 0.5) for h in HORIZONS}
    var = {h: np.ones(n) for h in HORIZONS}
    return PredictionFrame(y, X[:, -1], delta, prob, var, 100.0, X_raw=X), trail


# ------------------------------------------------------------------ classification
def test_classification_table_matches_sklearn_and_counts_add_up(viz_frame, viz_config):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)

    df = AT.classification_table(viz_frame, viz_config, digits=None)
    assert list(df.columns) == list(HORIZONS)
    for i, h in enumerate(HORIZONS):
        ret = viz_frame.y[:, i] / viz_frame.last_close
        mask = np.abs(ret) > viz_config.DIR_DEADBAND_BPS / 1e4
        t = (ret[mask] > 0).astype(int)
        p = viz_frame.prob(h)[mask]
        pb = (p > 0.5).astype(int)
        col = df[h]
        assert col["n scored (moves beyond 5 bps)"] == mask.sum()
        assert col["TP (called up, went up)"] + col["FP (called up, went down)"] + \
            col["TN (called down, went down)"] + col["FN (called down, went up)"] == mask.sum()
        for label, want in (("accuracy", accuracy_score(t, pb)), ("balanced accuracy", balanced_accuracy_score(t, pb)),
                            ("precision (up)", precision_score(t, pb)), ("recall / sensitivity (up)",
                                                                         recall_score(t, pb)),
                            ("F1 (up)", f1_score(t, pb)), ("MCC", matthews_corrcoef(t, pb)),
                            ("AUC", roc_auc_score(t, p)), ("calls up (predicted up-rate)", pb.mean()),
                            ("true up-rate", t.mean())):
            assert math.isclose(col[label], want, abs_tol=1e-9), (h, label)
        assert math.isclose(col["specificity (down)"], np.mean(pb[t == 0] == 0))
        # the effective size of what is scored: moves beyond the deadband, not every sample
        assert col["n_eff of the scored moves (n scored // bars ahead)"] == mask.sum() // S.horizon_steps(viz_frame, h)


def test_classification_intervals_use_the_effective_sample_size(viz_frame, viz_config):
    df = AT.classification_table(viz_frame, viz_config, digits=None)
    h = "h2"
    n_scored = df[h]["n scored (moves beyond 5 bps)"]
    k = df[h]["TP (called up, went up)"] + df[h]["TN (called down, went down)"]
    lo, hi = map(float, df[h]["accuracy 95% CI (n_eff)"].strip("[]").split(","))
    _, nlo, nhi = S.wilson(k, n_scored, steps=1)          # the naive (independent-samples) interval
    _, elo, ehi = S.wilson(k, n_scored, steps=20)
    assert math.isclose(lo, elo, abs_tol=1e-4) and math.isclose(hi, ehi, abs_tol=1e-4)
    assert hi - lo > 3 * (nhi - nlo)                        # ~sqrt(20) wider than the naive one


def test_classification_table_references_and_readouts(viz_frame, viz_config):
    df = AT.classification_table(viz_frame, viz_config, digits=None)
    for h in HORIZONS:
        u = df[h]["true up-rate"]
        assert math.isclose(df[h]["majority-class accuracy (hindsight)"], max(u, 1 - u))
        assert math.isclose(df[h]["ECE of a constant 0.5"], abs(u - 0.5))
        assert math.isclose(df[h]["Brier of a constant = the up-rate (hindsight)"], u * (1 - u))
    raw = AT.classification_table(viz_frame, viz_config, calibrated=False, digits=None)
    # temperature-like calibration keeps P(up) on the same side of 0.5: counts equal, Brier differs
    assert (raw.loc["TP (called up, went up)"] == df.loc["TP (called up, went up)"]).all()
    assert not np.allclose(raw.loc["Brier"].astype(float), df.loc["Brier"].astype(float))
    gauss = AT.classification_table(viz_frame, viz_config, readout="gaussian")
    assert set(gauss.columns) == set(HORIZONS)
    with pytest.raises(ValueError):
        AT.classification_table(viz_frame, viz_config, readout="nope")


# ------------------------------------------------------------------ price heads
def test_delta_quality_table_raw_served_and_zero(viz_frame, viz_config):
    served, raw = _served(viz_frame)
    df = AT.delta_quality_table(served, viz_config, raw_delta=raw, digits=None, usd_digits=None)
    for i, h in enumerate(HORIZONS):
        y = viz_frame.y[:, i]
        col = df[h]
        assert math.isclose(col["RMSE ($), raw heads"], math.sqrt(np.mean((y - raw[h]) ** 2)))
        assert math.isclose(col["RMSE ($), served"], math.sqrt(np.mean((y - BETAS[h] * raw[h]) ** 2)))
        assert math.isclose(col["RMSE ($), zero prediction"], math.sqrt(np.mean(y ** 2)))
        assert math.isclose(col["MAE ($), zero prediction"], np.mean(np.abs(y)))
        assert math.isclose(col["shrink beta (served = beta x raw)"], BETAS[h])   # least squares served / raw
        assert math.isclose(col["corr, Pearson"], np.corrcoef(y, raw[h])[0, 1])
        assert math.isclose(col["corr noise band +/- (95%, n_eff)"],
                            S.corr_null(len(y), steps=S.horizon_steps(viz_frame, h)))
        assert math.isclose(col["mean predicted ($), raw heads"], raw[h].mean())
        assert math.isclose(col["mean predicted ($), served"], BETAS[h] * raw[h].mean())
        assert math.isclose(col["share predicted up (delta > 0)"], np.mean(raw[h] > 0))
        assert math.isclose(col["max predicted ($), raw heads"], raw[h].max())
        slope = float(np.dot(y, raw[h]) / np.dot(raw[h], raw[h]))
        assert math.isclose(col["LS slope of realised on the raw head (unclipped)"], slope)
        assert col["beta this block would fit: clip(LS slope, 0, 1)"] == min(max(slope, 0.0), 1.0)
    given = AT.delta_quality_table(served, viz_config, raw_delta=raw, delta_scale={"h0": 0.3, "h1": 0.3, "h2": 0.3})
    assert given.loc["shrink beta (served = beta x raw)", "h1"] == 0.3
    # without raw heads or betas the frame's deltas are "model" (never "predicted, predicted") and a caption says why
    plain = AT.delta_quality_table(viz_frame, viz_config)
    assert not any("raw heads" in str(i) or str(i).endswith(", predicted") for i in plain.index)
    assert {"RMSE ($), model", "mean predicted ($), model"} <= set(plain.index)
    assert "raw_delta" in plain.attrs["caption"] and "caption" in AT.styled(plain).to_html()
    assert "caption" not in AT.delta_quality_table(served, viz_config, raw_delta=raw).attrs


def test_the_clipped_beta_row_shows_what_the_calibration_would_fit(viz_frame, viz_config):
    anti = copy.deepcopy(viz_frame)
    anti.delta = {h: -viz_frame.delta[h] for h in HORIZONS}          # the heads point the wrong way
    df = AT.delta_quality_table(anti, viz_config, raw_delta=anti.delta, digits=None)
    assert (df.loc["LS slope of realised on the raw head (unclipped)"].astype(float) < 0).all()
    assert (df.loc["beta this block would fit: clip(LS slope, 0, 1)"].astype(float) == 0.0).all()


def test_delta_quality_table_rounds_dollars_to_cents_and_ratios_to_four_places(viz_frame, viz_config):
    served, raw = _served(viz_frame)
    df = AT.delta_quality_table(served, viz_config, raw_delta=raw)
    assert df.loc["RMSE ($), served", "h0"] == round(df.loc["RMSE ($), served", "h0"], 2)
    assert df.loc["corr, Pearson", "h0"] == round(df.loc["corr, Pearson", "h0"], 4)
    assert isinstance(df.loc["n samples", "h0"], int)


# ------------------------------------------------------------------ across horizons
def test_magnitude_ordering_separates_the_trained_heads_from_the_shrink():
    rng = np.random.default_rng(1)
    n = 2000
    mags = np.sort(np.abs(rng.normal(0, 40, (n, 3))), axis=1)
    frame = PredictionFrame(rng.normal(0, 100, (n, 3)), np.full(n, 1e5), {h: mags[:, i] for i, h in enumerate(HORIZONS)},
                            {h: np.full(n, 0.5) for h in HORIZONS}, {h: np.ones(n) for h in HORIZONS}, 100.0)
    served, raw = _served(frame)
    df = AT.magnitude_ordering_table(served, raw_delta=raw, digits=None)
    assert df["raw heads (trained ordering)"].tolist() == [1.0, 1.0, 1.0]
    assert df["served deltas"].iloc[0] < 0.2 and df["served deltas"].iloc[2] < 0.2
    assert df["magnitudes in random order"].tolist() == pytest.approx([0.5, 0.5, 1 / 6])
    a = np.abs(frame.y)
    share = np.mean((a[:, 0] <= a[:, 1]) & (a[:, 1] <= a[:, 2]))
    assert df["realised moves"].iloc[2] == pytest.approx(share)
    assert "raw heads (trained ordering)" not in AT.magnitude_ordering_table(served).columns
    # Wilson intervals on n / bars ahead of the longer horizon in the check (15, 20, 20 bars)
    assert df["n_eff"].tolist() == [n // 15, n // 20, n // 20]
    lo, hi = map(float, df["realised moves 95% CI (n_eff)"].iloc[2].strip("[]").split(","))
    _, wlo, whi = S.wilson(round(share * n), n, steps=20)
    assert lo == pytest.approx(wlo, abs=1e-4) and hi == pytest.approx(whi, abs=1e-4) and lo < share < hi
    assert list(df.columns[:2]) == ["raw heads (trained ordering)", "raw heads 95% CI (n_eff)"]


def test_alignment_table_per_horizon_all_three_and_the_independence_reference(viz_frame, viz_config):
    noisy = copy.deepcopy(viz_frame)   # in the fixture the delta and P(up) signs agree everywhere; blur them
    rng = np.random.default_rng(5)
    noisy.delta = {h: noisy.delta[h] + rng.normal(0, 60, len(noisy)) for h in HORIZONS}
    served, raw = _served(noisy)
    df = AT.alignment_table(served, viz_config, raw_delta=raw, digits=None)
    assert list(df.index) == [*HORIZONS, "all 3"]
    D = np.stack([raw[h] > 0 for h in HORIZONS], 1)
    P = np.stack([viz_frame.prob(h) > 0.5 for h in HORIZONS], 1)
    for i, h in enumerate(HORIZONS):
        assert df.loc[h, "agree"] == pytest.approx(np.mean(D[:, i] == P[:, i]))
        sd, sp = D[:, i].mean(), P[:, i].mean()
        assert df.loc[h, "expected if independent"] == pytest.approx(sd * sp + (1 - sd) * (1 - sp))
        assert df.loc[h, "95% CI low"] < df.loc[h, "agree"] < df.loc[h, "95% CI high"]
    assert df.loc["all 3", "agree"] == pytest.approx(np.mean((D == P).all(1)))
    rng = np.random.default_rng(0)
    perm = np.mean([np.mean((D[rng.permutation(len(D))] == P).all(1)) for _ in range(300)])
    assert df.loc["all 3", "expected if independent"] == pytest.approx(perm, abs=0.01)
    # the served sign is the raw sign (beta > 0)
    assert AT.alignment_table(served, viz_config, digits=None)["agree"].tolist() == df["agree"].tolist()
    assert df["n"].dtype.kind == "i" and df.loc["all 3", "n_eff"] == len(viz_frame) // 20


# ------------------------------------------------------------------ baselines
def _report():
    from neural_trade.core.config import Config
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.report import evaluate

    frame, trail = _windows_frame(n=2500, seed=4)
    train, _ = _windows_frame(n=2500, seed=5)
    rng = np.random.default_rng(6)
    for f in (frame, train):                   # a noisy contrarian block: neither the model nor a baseline is perfect
        f.y = f.y + rng.normal(0, 60, f.y.shape)
    frame.direction_prob = {h: 1 / (1 + np.exp(trail[:, i] / 40 + rng.normal(0, 1, len(frame))))
                            for i, h in enumerate(HORIZONS)}
    frame.variance_scaled = {h: (1 + np.abs(trail[:, i]) / 50) ** 2 for i, h in enumerate(HORIZONS)}
    base = BaselineSet.fit(train.X_raw, train.y, train.last_close, 5.0)
    return evaluate(frame, Config(), baselines=base)


def test_baseline_table_shows_values_margins_and_noise():
    from neural_trade.evaluation.report import HIGHER_IS_BETTER

    rep = _report()
    df = AT.baseline_table(rep, digits=None)
    assert df.index.names == ["baseline", "metric"] and ("zero_delta", "delta/ev") not in df.index
    assert ("zero_delta", "delta/ev") in AT.baseline_table(rep, compact=False).index
    n_untested = 0
    for (name, key), row in df.iterrows():
        group, metric = key.split("/", 1)
        for h in HORIZONS:
            m = rep.model["horizons"][h][group][metric]
            b = rep.baselines[name]["horizons"][h][group][metric]
            if not (math.isfinite(m) and math.isfinite(b)):
                continue
            assert row[(h, "model")] == pytest.approx(m, abs=0.006)
            assert row[(h, "baseline")] == pytest.approx(b, abs=0.006)
            want = (m - b) if metric in HIGHER_IS_BETTER else (b - m)
            assert row[(h, AT.MARGIN)] == pytest.approx(want, abs=0.011)
            beats = rep.beats_baseline[name][key][h]
            assert row[(h, "verdict")].startswith("beats" if beats else "does not beat")
            z = row[(h, "z")]
            mg = rep.baseline_margins[name][key][h]
            if not math.isfinite(z):   # no test only when the difference is the same in every sample / resample
                assert mg.get("dm_z") is None and mg.get("boot_z") is None and "(" not in row[(h, "verdict")]
                assert mg.get("boot_se", 0.0) == 0.0
                n_untested += 1
                continue
            assert ("(noise" in row[(h, "verdict")]) == (abs(z) < 1.96)
            assert ("significantly worse" in row[(h, "verdict")]) == (z <= -1.96)
            assert row[(h, "verdict")].endswith("DM)" if mg.get("dm_z") is not None else "boot)")
    assert n_untested == 0                                     # every compared metric has a noise test here
    assert df.index[0][0] == "logreg_lags"                     # the informative comparison first
    assert "Bartlett" in df.attrs["caption"] and "bootstrap" in df.attrs["caption"]


def test_baseline_table_reads_older_jsons_and_says_their_z_is_not_current(tmp_path):
    rep = _report()
    d = json.loads(rep.to_json())
    d["meta"].pop("noise_tests")                               # written before the HAC / bootstrap tests
    for per_key in d["baseline_margins"].values():
        for per_h in per_key.values():
            for mg in per_h.values():
                mg.pop("boot_z", None), mg.pop("boot_se", None)
    path = tmp_path / "eval_report_test.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    df = AT.baseline_table(path)
    assert "predates" in df.attrs["caption"]
    assert math.isnan(df.loc[("logreg_lags", "direction/auc"), ("h1", "z")])      # no test for a ranked metric
    assert math.isfinite(df.loc[("logreg_lags", "direction/brier"), ("h1", "z")])  # the old DM z is still shown
    d.pop("baseline_margins")
    path.write_text(json.dumps(d), encoding="utf-8")
    df = AT.baseline_table(path)
    assert len(df) and df[("h1", "z")].isna().all() and "no noise test" in df.attrs["caption"]
    assert not df[("h1", "model")].isna().any()


# ------------------------------------------------------------------ extended trends
def test_trailing_move_table_finds_a_contrarian_block_and_a_momentum_block(viz_config):
    contra, trail = _windows_frame(seed=2, contrarian=True)
    momentum, _ = _windows_frame(seed=3, contrarian=False)
    df = AT.trailing_move_table({"cal": momentum, "test": contra}, viz_config)
    assert df.index.names == ["block", "horizon"]
    assert (df.loc["test", "corr(trailing move, realised)"] < -0.99).all()
    assert (df.loc["cal", "corr(trailing move, realised)"] > 0.99).all()
    assert (df.loc["test", "trailing sign = realised sign (beyond the deadband)"] < 0.05).all()
    head_col = [c for c in df.columns if c.startswith("corr(head, trailing move)")]
    assert head_col and "lambda" in head_col[0] and (df[head_col[0]] > 0.99).all()   # head = 0.5 x trailing
    assert df["trailing window (bars)"].tolist() == [10, 15, 20, 10, 15, 20]
    same = AT.trailing_move_table({"test": contra}, viz_config, trends={"test": trail})
    pd.testing.assert_frame_equal(same, df.loc[["test"]])
    no_windows = copy.deepcopy(contra)
    no_windows.X_raw = None
    with pytest.raises(ValueError, match="X_raw"):
        AT.trailing_move_table({"test": no_windows}, viz_config)


# ------------------------------------------------------------------ run settings
def test_run_settings_table_puts_the_effective_values_next_to_the_config(tmp_path):
    from neural_trade.core.config import Config

    cfg = Config(EPOCHS=4)
    cfg.to_yaml(tmp_path / "config.yaml")
    rows = [{"epoch": e, "lr": 1e-3 if e < 2 else 5e-4, "lr_indicator": 5e-3, "lambda_dir": 0.659,
             "lambda_vol": 3.324, "lambda_extended_trend": 0.1} for e in range(3)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "artifacts" / "calibration").mkdir(parents=True)
    (tmp_path / "artifacts" / "meta.json").write_text(json.dumps(
        {"fold": {"train": 100, "val": 10, "cal": 10, "test": 30, "gap": 80}, "epochs_run": 3, "var_scale": 1.13}))
    (tmp_path / "artifacts" / "calibration" / "pipeline_meta.json").write_text(json.dumps(
        {"conformal_scale": "realized_vol", "shrink_delta": True, "delta_scale": {"h0": 0.21, "h1": 0.023,
                                                                                  "h2": 0.25}}))
    (tmp_path / "artifacts" / "calibration" / "temperature_params.json").write_text(json.dumps(
        {"temperatures": {"h0": 1.5, "h1": 1.77, "h2": 1.34}}))
    (tmp_path / "meta.json").write_text(json.dumps({"run_id": "r1", "seed": 7, "tags": ["t"]}))
    df = AT.run_settings_table(tmp_path)
    assert df.index.names == ["group", "setting"] and list(df.columns) == ["config", "effective", "note"]
    # the indicator learning rate has its own row, in learning-rate units, and says it was not reduced
    ind = df.loc[("training", "indicator learning rate (LR x INDICATOR_LR_MULT)")]
    assert ind["config"] == "0.005" and ind["effective"] == "0.005" and "NOT reduced" in ind["note"]
    assert "0.001 -> 0.0005" in ind["note"] and "from 5 to 10" in ind["note"]
    assert df.loc[("training", "INDICATOR_LR_MULT"), "config"] == "5" and \
        df.loc[("training", "INDICATOR_LR_MULT"), "effective"] == ""
    # gated and threshold "weights" are not shown as live weights
    assert df.loc[("loss weights", "LAMBDA_DIR_ALIGN"), "effective"] == "off"
    assert "LAMBDA_DIR_ALIGN_OUTER" in df.loc[("loss weights", "LAMBDA_DIR_ALIGN"), "note"]
    assert df.loc[("loss weights", "LAMBDA_DIR_ALIGN_OUTER"), "effective"] == "off"
    assert df.loc[("loss weights", "LAMBDA_VAC"), "effective"] == "off"
    assert "threshold" in df.loc[("loss weights", "LAMBDA_VAC"), "note"]
    assert "fixed 0.1" in df.loc[("loss weights", "LAMBDA_VOL"), "note"] and \
        "auto-calibrated" in df.loc[("loss weights", "LAMBDA_VOL"), "note"]
    assert df.loc[("loss weights", "LAMBDA_DIR"), "effective"] == "0.659"
    assert "auto-calibrated" in df.loc[("loss weights", "LAMBDA_DIR"), "note"]
    assert df.loc[("loss weights", "LAMBDA_EXTENDED_TREND"), "note"] == ""        # unchanged by the pass
    assert "not logged" in df.loc[("loss weights", "LAMBDA_INTER"), "note"]
    assert ("loss weights", "LAMBDA_LOCAL_TREND") not in df.index                 # retired
    assert ("loss weights", "LAMBDA_QUANTILE") not in df.index                    # unused
    assert df.loc[("training", "LR"), "effective"] == "0.0005" and "1 reduction" in df.loc[("training", "LR"), "note"]
    assert df.loc[("run", "EPOCHS"), "effective"] == "3" and "stopped" in df.loc[("run", "EPOCHS"), "note"]
    assert df.loc[("serving calibration", "delta shrink beta h0 / h1 / h2"), "effective"] == "0.210 / 0.023 / 0.250"
    assert df.loc[("serving calibration", "temperature h0 / h1 / h2"), "effective"] == "1.500 / 1.770 / 1.340"
    assert "30" in df.loc[("data / split", "block sizes: train / val / cal / test (purge gap)"), "effective"]
    assert df.loc[("direction", "DIR_DEADBAND_BPS"), "config"] == "5"


def test_run_settings_table_needs_a_run_or_a_config():
    with pytest.raises(ValueError):
        AT.run_settings_table()


# ------------------------------------------------------------------ readability
def test_tables_have_no_empty_rows_and_unique_labels(viz_frame, viz_config):
    served, raw = _served(viz_frame)
    for df in (AT.classification_table(served, viz_config), AT.delta_quality_table(served, viz_config, raw_delta=raw)):
        assert df.index.is_unique and list(df.columns) == list(HORIZONS)
        empty = [i for i, row in df.iterrows() if all(isinstance(v, float) and math.isnan(v) for v in row)]
        assert empty == [], empty


def test_styled_prints_fixed_decimals_counts_and_missing_values(viz_frame, viz_config):
    served, raw = _served(viz_frame)
    html = AT.styled(AT.delta_quality_table(served, viz_config, raw_delta=raw)).to_html()
    rmse = AT.delta_quality_table(served, viz_config, raw_delta=raw).loc["RMSE ($), served", "h0"]
    assert f">{rmse:.2f}<" in html                                   # dollars in cents
    assert ">3000<" in html                                          # a count, no decimals
    al = AT.styled(AT.alignment_table(served, viz_config, raw_delta=raw)).to_html()
    assert ">n/a<" in al and ">3000<" in al
    ct = AT.classification_table(served, viz_config)
    ct_html = AT.styled(ct).to_html()
    assert f">{float(ct.loc['precision (up)', 'h1']):.4f}<" in ct_html


def test_run_settings_table_shows_live_gated_terms_and_thresholds_as_configured():
    from neural_trade.core.config import Config

    cfg = Config(LAMBDA_DIR_ALIGN_OUTER=0.5, LAMBDA_VAC=0.3)
    df = AT.run_settings_table(config=cfg, metrics=[])
    assert df.loc[("loss weights", "LAMBDA_DIR_ALIGN"), "effective"] != "off"
    assert df.loc[("loss weights", "LAMBDA_DIR_ALIGN"), "config"] == "0.7"
    assert df.loc[("loss weights", "LAMBDA_VAC"), "effective"] == ""
    assert "threshold" in df.loc[("loss weights", "LAMBDA_VAC"), "note"]
    ablated = AT.run_settings_table(config=Config(LAMBDA_DIR_ALIGN_OUTER=0.5,
                                                  ABLATE_LAMBDAS=["LAMBDA_DIR_ALIGN_OUTER"]), metrics=[])
    assert ablated.loc[("loss weights", "LAMBDA_DIR_ALIGN"), "effective"] == "off"
    assert "ablated" in ablated.loc[("loss weights", "LAMBDA_DIR_ALIGN_OUTER"), "note"]


# ------------------------------------------------------------------ captions for the figures
def test_figure_captions_carry_the_table_numbers(viz_frame, viz_config):
    from neural_trade.evaluation.report import coherence_block

    served, raw = _served(viz_frame)
    ct = AT.classification_table(served, viz_config, digits=None)
    text = AT.scored_counts_text(served, viz_config, block="test")
    n_sc = [int(ct[h]["n scored (moves beyond 5 bps)"]) for h in HORIZONS]
    assert text.startswith(f"test block {len(served):,} samples; scored beyond the 5 bps deadband: h0 {n_sc[0]:,}")
    assert f"n_eff {n_sc[0] // 10:,} / {n_sc[1] // 15:,} / {n_sc[2] // 20:,}" in text
    title = AT.confusion_title(served, viz_config, "h1")
    line1, line2 = title.split("<br>")
    assert f"n {n_sc[1]:,}" in line1 and f"{100 * ct['h1']['calls up (predicted up-rate)']:.0f}%" in line1
    for key, label in (("acc", "accuracy"), ("prec", "precision (up)"), ("rec", "recall / sensitivity (up)"),
                       ("spec", "specificity (down)"), ("F1", "F1 (up)"), ("MCC", "MCC")):
        assert f"{key} {ct['h1'][label]:.3f}" in line2, key
    stats = AT.direction_stats_line(served, viz_config, "h2", raw_delta=raw)
    coh = coherence_block(served, raw)
    assert f"acc {ct['h2']['accuracy']:.3f}" in stats and f"Brier {ct['h2']['Brier']:.4f}" in stats
    assert f"sign(delta) = call {coh['delta_dir_align_h2']:.3f} ({coh['delta_dir_align_indep_h2']:.3f} " in stats
    dq = AT.delta_quality_table(served, viz_config, raw_delta=raw, delta_scale=BETAS, digits=None, usd_digits=None)
    txt = AT.delta_summary_text(served, "h0", raw_delta=raw, delta_scale=BETAS)
    assert (f"RMSE $ raw {dq['h0']['RMSE ($), raw heads']:.2f} / served {dq['h0']['RMSE ($), served']:.2f} / zero "
            f"{dq['h0']['RMSE ($), zero prediction']:.2f}") in txt
    assert f"vs realised {dq['h0']['mean realised ($)']:.2f}" in txt and "beta 0.200" in txt
    assert "raw" not in AT.delta_summary_text(viz_frame, "h1") and "model" in AT.delta_summary_text(viz_frame, "h1")


def test_styled_shows_the_caption_and_two_decimal_z():
    rep = _report()
    df = AT.baseline_table(rep)
    html = AT.styled(df).to_html()
    assert "<caption>" in html and "Bartlett" in html
    z = df[("h1", "z")].dropna().iloc[0]
    assert f">{z:.2f}<" in html
