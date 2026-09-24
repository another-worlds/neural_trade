"""Training dashboard, health tiles, epoch table, companion figures and the live batch strip."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")
HORIZON_HEX = set(T.HORIZON_COLORS.values())


def _served_rows(n=8, best=6):
    """Epoch rows whose validation loss is lowest at 0-based epoch ``best`` (then rises), with a distinct
    val MCC per epoch so a tile can be traced back to the epoch it describes."""
    from tests.conftest import _viz_history

    rows = _full(_viz_history(n))
    for e, r in enumerate(rows):
        r["val_loss"] = 6.0 - 0.1 * e if e <= best else 6.0 - 0.1 * best + 0.05 * (e - best)
        for i, h in enumerate(H):
            r[f"val_dir_mcc_{h}"] = round(0.01 * e + 0.001 * i, 4)
            r[f"train_dir_mcc_{h}"] = r[f"val_dir_mcc_{h}"] + 0.05      # a steady train - val gap
    return rows


def _full(rows):
    """The shared fixture plus the Brier / up-rate keys the trainer also logs."""
    for r in rows:
        for h in H:
            for pre in ("val_", "train_"):
                r.setdefault(f"{pre}dir_brier_{h}", 0.25)
                r.setdefault(f"{pre}true_up_rate_{h}", 0.5)
                r.setdefault(f"{pre}pred_up_rate_{h}", 0.48)
    return rows


def _traces(fig, **match):
    return [t for t in fig.data if all(getattr(t, k, None) == v for k, v in match.items())]


def _panel_axes(fig, row, col):
    sp = fig.get_subplot(row, col)
    return sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")


def _panel_traces(fig, row, col):
    xa, ya = _panel_axes(fig, row, col)
    return [t for t in fig.data if (t.xaxis or "x") == xa and (t.yaxis or "y") == ya]


# ------------------------------------------------------------------ the dashboard
def test_training_dashboard_draws_every_logged_metric(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(_full(viz_history()), viz_config)
    assert T.empty_panels(fig) == []
    names = {t.name for t in fig.data}
    assert {"h0 (10 bars)", "h1 (15 bars)", "h2 (20 bars)", "network"} <= names
    assert _traces(fig, legendgroup="best") and _traces(fig, legendgroup="served")


def test_training_dashboard_marks_metrics_that_were_not_logged(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(viz_history(full=False), viz_config)
    notes = [a.text for a in fig.layout.annotations if a.text == "not logged in this run"]
    assert len(notes) == len(T.empty_panels(fig)) > 0


def test_default_training_visualization_is_the_dashboard(viz_config, viz_history):
    """plotly_interactive (Config.VISUALIZATION) used to draw an empty direction panel."""
    from neural_trade.registries.visualizations import Visualizations

    fig = Visualizations.build("plotly_interactive", _full(viz_history()), viz_config)
    assert T.empty_panels(fig) == []
    assert any(t.name == "h1 (15 bars)" for t in fig.data)


def test_the_served_epoch_is_ringed_and_the_horizon_tiles_describe_it(viz_config):
    """Finding 34: the tiles used the LAST epoch while the star marked the best one; nothing said which
    weights were evaluated and bundled."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    rows = _served_rows(8, best=6)
    fig = training_dashboard_figure(rows, viz_config, weights_epoch=8)
    (ring,) = _traces(fig, legendgroup="served")
    (star,) = _traces(fig, legendgroup="best")
    assert list(ring.x) == [8] and list(star.x) == [7] and "≠ best" in ring.name
    assert ring.marker.symbol == "circle-open"
    checks = training_health(rows, viz_config, weights_epoch=7)
    assert checks["served"]["value"].startswith("epoch 7 = best") and checks["served"]["status"] == "good"
    assert "MCC +0.060" in checks["direction h0"]["value"]      # epoch 7's MCC, not the last epoch's +0.070
    assert "last +0.070" in checks["direction h0"]["note"]
    off = training_health(rows, viz_config, weights_epoch=8)
    assert off["served"]["status"] == "warning" and "≠ best (7)" in off["served"]["value"]
    # the best-loss tile no longer claims the best epoch is what the checkpoint / early stopping kept
    assert "NOT the served weights" in off["best"]["detail"] and "epoch 8" in off["best"]["detail"]
    assert "the served weights are this epoch's" in checks["best"]["detail"]


def test_served_epoch_rules_for_bundles_old_and_new_and_live_runs(viz_config):
    from neural_trade.core.config import Config
    from neural_trade.visualization.training_dashboard import served_epoch

    rows = _served_rows(8, best=6)      # best epoch 7 of 8, one epoch since: EarlyStopping (EARLY 3) did not fire
    assert served_epoch(rows, viz_config, meta={"weights_epoch": 5})[0] == 5
    assert served_epoch(rows, viz_config, meta={"fold": {}})[0] == 8           # old bundle: Keras kept the last
    fired = _served_rows(8, best=2)     # 5 epochs since the best >= EARLY 3: it fired and restored epoch 3
    assert served_epoch(fired, viz_config, meta={"fold": {}})[0] == 3
    assert served_epoch(rows, viz_config)[0] == 7                                 # live: will be restored
    assert served_epoch(rows, Config(EPOCHS=8, EARLY=3, CALLBACKS=["csv_logger"]))[0] == 8
    # metrics.jsonl of an older trainer without a bundle: the same rule as an older bundle
    old = [dict(r, run_id="old", sec_per_step=0.2) for r in rows]
    assert served_epoch(old, viz_config)[0] == 8 and "older run" in served_epoch(old, viz_config)[1]
    assert served_epoch([dict(r, lr_used=1e-3) for r in old], viz_config)[0] == 7   # a current trainer's rows


def test_a_metrics_path_reads_the_bundle_next_to_it(tmp_path, viz_config):
    """Notebook 04 passes run_dir / metrics.jsonl: the served epoch, the validation size and the old
    one-epoch-late learning rates come from artifacts/meta.json without any extra argument."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    rows = _served_rows(8, best=6)
    for e, r in enumerate(rows):
        r["lr"] = 1e-3 if e < 5 else 5e-4        # logged after the scheduler (old runs): epoch 6 trained at 1e-3
        r["run_id"], r["sec_per_step"] = "old-run", 0.2      # written by the jsonl epoch logger
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "meta.json").write_text(json.dumps({"format_version": 1, "fold": {"val": 2866}}))
    fig = training_dashboard_figure(tmp_path / "metrics.jsonl", viz_config)
    assert list(_traces(fig, legendgroup="served")[0].x) == [8]
    (lr,) = _traces(fig, legendgroup="lr")
    assert lr.line.shape == "hv" and "re-aligned" in lr.name
    assert np.allclose(lr.y, [1e-3] * 6 + [5e-4] * 2)
    checks = training_health(tmp_path / "metrics.jsonl", viz_config)
    assert "±0.12" in checks["direction h0"]["value"]            # the band from fold.val = 2866, h0 = 10 bars


def test_learning_rates_are_realigned_only_when_they_were_logged_late(viz_config):
    """Round-2 review: the shift applied to any bundle without weights_epoch (even with no scheduler)
    and never to an older run without a bundle. Rows of a current trainer carry lr_used."""
    from neural_trade.core.config import Config
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    rows = _served_rows(8, best=6)
    for e, r in enumerate(rows):
        r["lr"] = 1e-3 if e < 5 else 5e-4                      # the NEXT epoch's rate (logger after the scheduler)
        r["run_id"], r["sec_per_step"] = "old-run", 0.2
    trained = [1e-3] * 6 + [5e-4] * 2

    def lr_trace(fig):
        (t,) = _traces(fig, legendgroup="lr")
        return t

    t = lr_trace(training_dashboard_figure(rows, viz_config))            # older run, no bundle
    assert "re-aligned" in t.name and np.allclose(t.y, trained)
    no_sched = Config(EPOCHS=8, EARLY=3, CALLBACKS=["early_stopping", "model_checkpoint"])
    t = lr_trace(training_dashboard_figure(rows, no_sched, meta={"format_version": 1}))
    assert "re-aligned" not in t.name and np.allclose(t.y, [r["lr"] for r in rows])
    current = [dict(r, lr_used=v) for r, v in zip(rows, trained)]       # read at each epoch's start
    t = lr_trace(training_dashboard_figure(current, viz_config))
    assert "re-aligned" not in t.name and np.allclose(t.y, trained)
    t = lr_trace(training_dashboard_figure([{k: v for k, v in r.items() if k not in ("run_id", "sec_per_step")}
                                            for r in rows], viz_config))   # a Keras History: 'lr' is in effect
    assert "re-aligned" not in t.name


def test_direction_tiles_are_graded_against_the_chance_band(viz_config):
    """Finding 36: +/-0.02 thresholds sat far inside the noise of a 2,866-sample validation block."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    rows = _served_rows(6, best=5)
    band = S.corr_null(2866 // 10)
    for m, status in ((0.05, "info"), (band + 0.01, "good"), (-band - 0.01, "critical")):
        rows[-1]["val_dir_mcc_h0"] = m
        tile = training_health(rows, viz_config, n_val=2866)["direction h0"]
        assert tile["status"] == status, (m, tile)
    assert "within chance" in training_health(rows[:-1] + [dict(rows[-1], val_dir_mcc_h0=0.05)], viz_config,
                                              n_val=2866)["direction h0"]["value"]
    fig = training_dashboard_figure(rows, viz_config, n_val=2866)
    xa, ya = _panel_axes(fig, 2, 1)
    rects = [s for s in fig.layout.shapes if s.type == "rect" and s.yref == ya]
    # n_eff = N // steps, as the evaluation report counts it
    assert sorted(round(s.y1, 4) for s in rects) == sorted(round(S.corr_null(2866 // k), 4) for k in (10, 15, 20))
    xa, ya = _panel_axes(fig, 3, 1)
    rects = [s for s in fig.layout.shapes if s.type == "rect" and s.yref == ya]
    assert min(round(s.y1 - 0.5, 4) for s in rects) == round(band / 2, 4)   # balanced accuracy: half the MCC band


def _horizon_edges(fig, row, col):
    """{horizon: sorted y} of the dashed horizon-coloured lines in a panel."""
    _, ya = _panel_axes(fig, row, col)
    by_hex = {c: h for h, c in T.HORIZON_COLORS.items()}
    out = {}
    for s in fig.layout.shapes:
        if s.type == "line" and s.yref == ya and s.line.color in HORIZON_HEX:
            assert s.line.dash == "dash" and s.y0 == s.y1, s
            out.setdefault(by_hex[s.line.color], []).append(round(s.y0, 4))
    return {h: sorted(v) for h, v in out.items()}


def test_two_sided_chance_bands_mark_each_horizons_own_edges(viz_config):
    """Final check: the MCC and balanced-accuracy panels shaded three neutral tiers with no horizon marking,
    so a price-head h0 MCC of -0.148 (critical: below h0's ±0.12) sat inside h2's ±0.17 tier and read as
    chance, while the subtitle promised 'dashed = each horizon's limit'."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    rows = _served_rows(6, best=5)
    rows[-1]["val_gauss_dir_mcc_h0"] = -0.148
    fig = training_dashboard_figure(rows, viz_config, n_val=2866)
    band = {h: S.corr_null(2866 // k) for h, k in zip(H, (10, 15, 20))}
    for (r, c), center, scale in (((2, 1), 0.0, 1.0), ((2, 2), 0.0, 1.0), ((3, 1), 0.5, 0.5)):
        edges = _horizon_edges(fig, r, c)
        assert sum(len(v) for v in edges.values()) == 6, ((r, c), edges)
        for h in H:   # both edges of each horizon's own band, in that horizon's colour
            assert edges[h] == [round(center - scale * band[h], 4), round(center + scale * band[h], 4)], (r, c, h)
    # the price-head h0 point is outside its own band, as its tile says, though inside h2's shaded tier
    lo_h0 = _horizon_edges(fig, 2, 2)["h0"][0]
    assert -0.148 < lo_h0 and -0.148 > -band["h2"]
    assert training_health(rows, viz_config, n_val=2866)["price head h0"]["status"] == "critical"
    assert "dashed = each horizon's limit, in its colour" in fig.layout.title.text.replace("<br>", " ")


def test_the_dashed_limit_phrase_is_never_broken_across_subtitle_lines(viz_config):
    """Round 2 of the final check: the subtitle wrapped inside the phrase ('... dashed = each' / 'horizon's
    limit, in its colour') because it was one long part with the shading note; it is now its own part, so
    _wrap breaks between the two phrases, with and without a figure-wide title and in a long run."""
    import re

    from neural_trade.visualization.training_dashboard import _SUB_CHARS, training_dashboard_figure

    phrase = "dashed = each horizon's limit, in its colour"
    for n, best, served in ((6, 5, None), (20, 17, 18)):
        fig = training_dashboard_figure(_served_rows(n, best=best), viz_config, n_val=2866, weights_epoch=served)
        head, sub = fig.layout.title.text.split("<br>", 1)
        lines = [re.sub("<[^>]+>", "", ln) for ln in sub.split("<br>")]
        assert any(phrase in ln for ln in lines), lines
        assert not any(ln.rstrip().endswith("dashed = each") for ln in lines), lines
        shading = "shaded = 95% of what chance gives on the validation block with no skill (ECE, PIT-KS: a calibrated head)"
        assert sum(shading in ln for ln in lines) == 1, lines
        assert all(len(ln) <= _SUB_CHARS for ln in lines), lines
    # no chance band (tiny validation block): neither phrase is printed
    fig = training_dashboard_figure(_served_rows(6, best=5), viz_config, n_val=90)
    assert "dashed = each horizon" not in fig.layout.title.text and "shaded = 95%" not in fig.layout.title.text


def test_convergence_and_stability_do_not_flip_on_epoch_noise(viz_config):
    """Finding 37: a 3-point slope turned a steadily improving run 'worsening' (critical) on noise."""
    from neural_trade.visualization.training_dashboard import training_health

    rng = np.random.default_rng(7)
    rows = [{"epoch": e, "loss": 7 - 0.07 * e, "val_loss": 6 * (1 - 0.009 * e) * (1 + 0.023 * rng.normal())}
            for e in range(20)]
    for n in range(5, 21):
        checks = training_health(rows[:n], viz_config)
        assert checks["convergence"]["status"] != "critical", (n, checks["convergence"])
        assert checks["stability"]["status"] == "info"
        assert "t =" in checks["convergence"]["note"]


def test_a_diverged_epoch_is_critical_and_nothing_green_says_nan(viz_config, viz_history):
    """Finding 103: a NaN val loss produced a green 'gap +nan (steady)'."""
    from neural_trade.visualization.training_dashboard import training_health

    rows = viz_history(6)
    rows[-1]["val_loss"] = None
    checks = training_health(rows, viz_config)
    assert checks["divergence"]["status"] == "critical" and "epoch 6" in checks["divergence"]["value"]
    for name, c in checks.items():
        assert not ("nan" in c["value"].lower() and c["status"] in ("good", "warning")), (name, c)
    rows = viz_history(6)
    rows[2]["val_loss"] = float("nan")                            # an earlier NaN: trends use finite epochs
    checks = training_health(rows, viz_config)
    assert "nan" not in checks["convergence"]["value"].lower()


def test_the_gap_tile_says_which_way_val_moves(viz_config):
    from neural_trade.visualization.training_dashboard import training_health

    rows = [{"epoch": e, "loss": 7 - 0.2 * e, "val_loss": 5.0 - 0.05 * e} for e in range(8)]   # gap -2 -> -0.95
    gap = training_health(rows, viz_config)["gap"]
    assert "val rising vs train" in gap["value"] and "widening" not in gap["value"]
    assert gap["status"] == "warning" and "not comparable" in gap["detail"]


def test_learning_rate_tile_counts_reductions_in_english(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_health

    assert "(1 reduction)" in training_health(viz_history(), viz_config)["learning rate"]["value"]


def test_training_health_flags_collapse_patience_and_nonfinite_gradients(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_health, training_health_html

    rows = viz_history()
    rows[1]["val_pred_up_rate_h1"] = 0.995                       # the served (best) epoch predicts one class
    rows[-1]["nonfinite_grad_steps"] = 3.0
    for r in rows[2:]:
        r["val_loss"] = 10.0                                     # best epoch stays at 1-2
    rows[1]["val_loss"] = 1.0
    checks = training_health(rows, viz_config)
    assert checks["direction h1"]["status"] == "critical" and "collapsed" in checks["direction h1"]["value"]
    assert checks["gradients"]["status"] == "critical"
    assert checks["patience"]["status"] == "critical" and checks["patience"]["value"].endswith("/ 3")
    html = training_health_html(rows, viz_config)
    assert "direction head · h1 (15 bars)" in html
    rows[1]["val_pred_up_rate_h1"], rows[-1]["val_pred_up_rate_h1"] = 0.48, 0.99    # collapsing now, not served
    assert "last epoch collapsed" in training_health(rows, viz_config)["direction h1"]["value"]


def test_health_tiles_cover_both_heads_the_base_rate_and_overfitting(viz_config):
    from neural_trade.visualization.training_dashboard import training_health

    rows = _served_rows(10, best=9)
    for e, r in enumerate(rows):
        for h in H:
            r[f"train_gauss_dir_mcc_{h}"] = 0.02 * e          # train rises ...
            r[f"val_gauss_dir_mcc_{h}"] = -0.01 * e           # ... while val falls
            r[f"val_dir_brier_{h}"], r[f"val_true_up_rate_{h}"] = 0.2535, 0.46
            r[f"val_dir_loss_{h}"] = 0.70
    checks = training_health(rows, viz_config, n_val=2866)
    assert "overfitting" in checks["price head h0"]["value"] and checks["price head h0"]["status"] != "good"
    skill = checks["skill"]
    assert skill["status"] == "warning" and "-2.1%" in skill["value"] and "ln 2" in skill["note"]
    assert all(c.get("detail") for c in checks.values())


def test_health_html_shows_the_meaning_and_the_numbers_without_hover(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_health_html

    html = training_health_html(viz_history(), viz_config, n_val=2866)
    assert "title=" not in html                                   # meanings are printed, not tooltips
    assert "epochs since the best val loss" in html and "Direction metrics at epoch" in html
    assert "Loss terms" not in training_health_html(viz_history(), viz_config, table=False)


def test_overfitting_needs_a_significant_widening_beyond_the_chance_band(viz_config):
    """Round-2 review: the flag used only slope signs and a fixed 0.05 widening (inside the ±0.12-0.17
    MCC band), its early window overlapped the late one, and noise flipped it epoch to epoch."""
    from neural_trade.visualization.training_dashboard import training_health

    rng = np.random.default_rng(3)
    rows = _served_rows(20, best=19)
    for e, r in enumerate(rows):
        for h in H:
            r[f"train_gauss_dir_mcc_{h}"] = 0.05 + 0.002 * e + 0.02 * rng.normal()   # train creeps up
            r[f"val_gauss_dir_mcc_{h}"] = 0.08 * rng.normal()                          # val: noise only
    for n in range(6, 21):
        for h in H:
            tile = training_health(rows[:n], viz_config, n_val=2866)[f"price head {h}"]
            assert "overfitting" not in tile["value"], (n, h, tile["value"])
    # the reviewer's case: train rising, val flat, the gap +0.05 wider after 6 epochs: inside the band
    six = _served_rows(6, best=5)
    for e, r in enumerate(six):
        r["train_gauss_dir_mcc_h1"], r["val_gauss_dir_mcc_h1"] = 0.01 * e, -0.002 * e
    assert "overfitting" not in training_health(six, viz_config, n_val=2866)["price head h1"]["value"]
    # a real widening is flagged; the first and last windows (2 epochs each out of 7) do not overlap
    for e, r in enumerate(rows):
        r["train_gauss_dir_mcc_h0"], r["val_gauss_dir_mcc_h0"] = 0.02 * e, -0.012 * e
    tile = training_health(rows[:7], viz_config, n_val=2866)["price head h0"]
    assert "overfitting: train − val gap +0.02 → +0.18 (first / last 2 epochs" in tile["value"]
    assert tile["status"] != "good"
    # without the validation size there is no band to exceed: no verdict
    assert "overfitting" not in training_health(rows, viz_config)["price head h0"]["value"]


def test_progress_is_graded_on_trends_not_on_endpoint_signs(viz_config):
    """Round-2 review: a +0.5% endpoint change in the forecast terms (noise) flipped the tile."""
    from neural_trade.visualization.training_dashboard import training_health

    rng = np.random.default_rng(5)
    rows = _weighted_rows(14)
    for e, r in enumerate(rows):
        point = 0.4 + 0.01 * (-1) ** e                        # forecast terms: no trend, the endpoint sign flips
        r["val_point_loss"], r["val_nll_loss"] = point, 2.0 - 0.05 * e
        r["val_loss"] = 5.0 + (point - 0.4) - 0.25 * 0.05 * e  # NLL enters x lambda_var 0.25
    for n in range(5, 15):
        tile = training_health(rows[:n], viz_config)["progress"]
        assert tile["status"] == "warning" and "rose" not in tile["value"], (n, tile)
        assert "(t = " in tile["note"]
        if n >= 8:                                            # a few epochs more and the total's fall is clear
            assert "calibration terms only" in tile["value"], (n, tile)
    for e, r in enumerate(rows):                              # the forecast terms fall too: good
        r["val_point_loss"] = 0.4 - 0.01 * e
        r["val_loss"] = 5.0 - 0.01 * e - 0.25 * 0.05 * e
    assert training_health(rows, viz_config)["progress"]["status"] == "good"
    for r in rows:                                            # no trend at all
        r["val_loss"], r["val_point_loss"], r["val_nll_loss"] = 5.0 + 0.01 * rng.normal(), 0.4, 2.0
    tile = training_health(rows, viz_config)["progress"]
    assert tile["status"] == "warning" and "no clear trend" in tile["value"]


def test_brier_skill_is_graded_against_the_no_information_ceiling(viz_config):
    """Round-2 review: the tile turned green whenever all skills were > 0, at any sample size."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    def rows_with(skill):
        rows = _served_rows(6, best=5)
        for r in rows:
            for h in H:
                r[f"val_true_up_rate_{h}"], r[f"val_dir_brier_{h}"] = 0.46, 0.2484 * (1 - skill)
        return rows

    ceiling = S.Z95 ** 2 / 286                                   # h0: n_eff = 2866 / 10
    tile = training_health(rows_with(0.01), viz_config, n_val=2866)["skill"]
    assert tile["status"] == "info" and f"chance ceiling +{100 * ceiling:.1f}%" in tile["note"]
    tile = training_health(rows_with(0.05), viz_config, n_val=2866)["skill"]
    assert tile["status"] == "good" and tile["value"].count("beyond chance") == 3
    tile = training_health(rows_with(0.17), viz_config, n_val=90)["skill"]       # n_eff 9 / 6 / 4
    assert tile["status"] != "good" and "too few validation samples" in tile["note"]
    assert training_health(rows_with(-0.02), viz_config, n_val=2866)["skill"]["status"] == "warning"
    fig = training_dashboard_figure(rows_with(0.01), viz_config, n_val=2866)
    _, ya = _panel_axes(fig, 3, 2)
    rects = sorted(round(s.y1, 5) for s in fig.layout.shapes if s.type == "rect" and s.yref == ya)
    assert rects == sorted(round(S.Z95 ** 2 / int(2866 / k), 5) for k in (10, 15, 20))


def test_chance_ranges_are_capped_or_withheld_on_a_tiny_validation_block(viz_config):
    """Round-2 review: with n_eff 9 / 6 / 4 the MCC band reached ±1.40 and the tiles said 'within chance (±1.40)'."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    rows = _served_rows(6, best=5)
    checks = training_health(rows, viz_config, n_val=90)
    for h in H:
        assert "too few validation samples for a chance band (n_eff" in checks[f"direction {h}"]["value"]
        assert checks[f"direction {h}"]["status"] == "info"
    fig = training_dashboard_figure(rows, viz_config, n_val=90)
    assert not [s for s in fig.layout.shapes if s.type == "rect"]
    for r, c in ((2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)):    # no band, so no horizon edges either
        assert not _horizon_edges(fig, r, c), (r, c)
    assert "fewer than 10 effective samples" in fig.layout.title.text
    fig = training_dashboard_figure(rows, viz_config, n_val=200)      # n_eff 20 / 13 / 10: all drawn, all inside
    for (r, c), (lo, hi) in (((2, 1), (-1, 1)), ((2, 2), (-1, 1)), ((3, 1), (0, 1))):
        _, ya = _panel_axes(fig, r, c)
        rects = [s for s in fig.layout.shapes if s.type == "rect" and s.yref == ya]
        assert len(rects) == 3 and all(lo <= s.y0 < s.y1 <= hi for s in rects)
        edges = _horizon_edges(fig, r, c)
        assert sorted(edges) == list(H) and all(lo <= y <= hi for v in edges.values() for y in v)


def test_calibration_panels_show_what_a_calibrated_head_reaches_by_chance(viz_config):
    """Round-2 review: PIT-KS and ECE had no finite-sample reference, so every value read as miscalibration."""
    from neural_trade.visualization.training_dashboard import _ECE95, _KS95, epoch_table_html, training_dashboard_figure

    rows = _served_rows(6, best=5)
    fig = training_dashboard_figure(rows, viz_config, n_val=2866)
    n_eff = [int(2866 / k) for k in (10, 15, 20)]
    for (r, c), expect in (((4, 2), [_KS95 / math.sqrt(n) for n in n_eff]),
                           ((4, 1), [_ECE95 * math.sqrt(0.25 / n) for n in n_eff])):     # up-rate 0.5
        _, ya = _panel_axes(fig, r, c)
        rects = sorted(round(s.y1, 5) for s in fig.layout.shapes if s.type == "rect" and s.yref == ya)
        assert rects == sorted(round(v, 5) for v in expect), (r, c)
        edges = [s for s in fig.layout.shapes if s.type == "line" and s.yref == ya and s.line.color in HORIZON_HEX]
        assert sorted(round(s.y0, 5) for s in edges) == rects          # each horizon's own limit, in its colour
    html = epoch_table_html(rows, viz_config, n_val=2866)
    assert f"calibrated &lt; {_KS95 / math.sqrt(286):.3f}" in html


def test_subtitles_fit_a_900px_cell(viz_config):
    """Round-2 review: the loss-terms subtitle was one line cut off at 1000 px."""
    import re

    from neural_trade.visualization.training_dashboard import (_SUB_CHARS, direction_detail_figure,
                                                               loss_terms_figure, training_dashboard_figure)

    rows = _served_rows(8, best=6)
    for fig in (training_dashboard_figure(rows, viz_config, n_val=2866, weights_epoch=8),
                direction_detail_figure(rows, viz_config, n_val=2866), loss_terms_figure(rows, viz_config)):
        head, sub = fig.layout.title.text.split("<br>", 1)
        lines = re.sub("<[^>]+>", "\n", sub.replace("<br>", "\n")).split("\n")
        lines = [ln for ln in lines if ln.strip()]
        assert len(lines) >= 2 and all(len(ln) <= _SUB_CHARS for ln in lines), lines
        assert fig.layout.margin.t >= 40 + 17 * len(lines) + 26


# ------------------------------------------------------------------ loss terms, weights and shares
def _weighted_rows(n=4):
    rows = []
    for e in range(n):
        r = {"epoch": e, "loss": 6.0, "val_loss": 5.0,
             "lambda_dir": 0.5, "lambda_var": 0.25, "lambda_crps": 0.8, "lambda_soft_ece": 2.0, "lambda_vol": 3.0,
             "lambda_t_perp": 0.1, "lambda_casimir": 0.1, "lambda_hd": 0.1, "lambda_ife": 0.1,
             "lambda_vac_overflow": 0.1}
        for pre in ("", "val_"):
            r.update({pre + "point_loss": 0.4, pre + "dir_loss": 2.0, pre + "nll_loss": 2.0, pre + "crps_loss": 1.0,
                      pre + "soft_ece_loss": 0.3, pre + "vol_loss": 1.0, pre + "inter_reg": 1e-4,
                      pre + "t_perp_loss": 0.2, pre + "hd_loss": 0.03, pre + "ife_loss": 0.0 if e < 2 else 1e-3,
                      pre + "casimir_loss": 1e-4, pre + "vac_loss": 0.0, pre + "vac_overflow_loss": 0.0 if pre else 0.09})
            for i in range(3):
                r[f"{pre}extended_h{i}"] = 0.01
        rows.append(r)
    return rows


def test_loss_contributions_follow_the_total_formula():
    """Findings 27 / 38: the logged terms are on mixed bases (point, trend, volatility and physics carry
    their lambda; direction, NLL, CRPS and soft ECE do not; volatility and the regulariser enter at 0.1)."""
    from neural_trade.core.config import Config
    from neural_trade.visualization.training_dashboard import loss_contributions

    c = loss_contributions(_weighted_rows(), Config()).iloc[-1]
    expect = dict(point=0.4, trend=0.03, direction=1.0, nll=0.5, crps=0.8, soft_ece=0.6, volatility=0.1,
                  physics=0.2 + 0.03 + 1e-3 + 1e-4, regulariser=1e-5)
    for k, v in expect.items():
        assert math.isclose(c[k], v, rel_tol=1e-6), (k, c[k], v)
    assert math.isclose(c["other"], 5.0 - sum(expect.values()), rel_tol=1e-9)


def test_val_loss_composition_stacks_to_the_val_loss_with_weights_in_the_legend(viz_config):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(_weighted_rows(), viz_config)
    stack = [t for t in fig.data if t.stackgroup == "val-loss"]
    assert np.allclose(np.sum([np.asarray(t.y, float) for t in stack], axis=0), 5.0, atol=1e-5)
    names = {t.name for t in stack}
    assert {"direction BCE ×0.5", "NLL ×0.25", "CRPS ×0.8", "soft ECE ×2", "volatility ×0.1"} <= names
    assert any(n.startswith("other: coherence") for n in names) and not any("regulariser" in n for n in names)
    assert "unweighted" not in " ".join(a.text or "" for a in fig.layout.annotations)


def test_physics_zeros_and_switched_off_terms_are_visible(viz_config):
    """Findings 87 / 105: exact zeros vanished on the log axis and an off term was silently missing."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    rows = _weighted_rows()
    rows[0]["ife_loss"] = 1e-3                     # training IFE is zero at epoch 2 only, validation at 1 and 2
    fig = training_dashboard_figure(rows, viz_config)
    names = [t.name for t in _panel_traces(fig, 5, 2)]
    assert "IFE (= 0 in 2/4 val, 1/4 train epochs)" in names      # both zero counts
    assert "vacuum: off (λ = 0)" in names
    assert "vac overflow (train; 0 at eval by design)" in names
    # validation zeros (filled ▼) and training zeros (open ▽) sit on their own rows, each labelled on the axis
    (vz,) = [t for t in _panel_traces(fig, 5, 2) if t.marker.symbol == "triangle-down"]
    (tz,) = [t for t in _panel_traces(fig, 5, 2) if t.marker.symbol == "triangle-down-open"]
    assert list(vz.x) == [1, 2] and list(tz.x) == [2]
    assert vz.y[0] > tz.y[0] and "val" in vz.hovertemplate and "train" in tz.hovertemplate
    _, ya = _panel_axes(fig, 5, 2)
    ax = fig.layout["yaxis" + ya[1:]]
    ticks = dict(zip(ax.ticktext, ax.tickvals))
    assert math.isclose(ticks["= 0 val"], vz.y[0], rel_tol=1e-5) and math.isclose(ticks["= 0 train"], tz.y[0], rel_tol=1e-5)
    heading = [a.text for a in fig.layout.annotations if (a.text or "").startswith("Physics terms")]
    assert heading and "λ = 0.1" in heading[0]


def test_non_horizon_series_never_wear_horizon_colours(viz_config):
    """Finding 101: point / trend / direction and the gradient norm were drawn in the h0/h1/h2 colours."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(_weighted_rows(), viz_config)
    for t in fig.data:
        colors = {getattr(t.line, "color", None), getattr(t.marker, "color", None), getattr(t, "fillcolor", None)}
        if colors & HORIZON_HEX:
            assert t.legendgroup in H, (t.name, colors)
    (best,) = _traces(fig, legendgroup="best")
    assert best.marker.color != T.WARNING
    # OTHER_SERIES[2] (#008300) sits next to the status green T.GOOD: no series wears it
    drawn = json.dumps([t.to_plotly_json() for t in fig.data] + [s.to_plotly_json() for s in fig.layout.shapes],
                       default=str)
    green = (T.OTHER_SERIES[2], T.rgba(T.OTHER_SERIES[2], 0.78), T.rgba(T.OTHER_SERIES[2], 0.35))
    assert not any(g in drawn for g in green)


def test_panel_legends_fit_their_columns_and_headings_stay(viz_config):
    """Finding 39: an 8-entry legend pushed the right margin and narrowed every panel by ~22%."""
    from neural_trade.visualization.training_dashboard import (_MIN_FIG_WIDTH, _legend_width,
                                                               training_dashboard_figure)

    fig = training_dashboard_figure(_weighted_rows(), viz_config)
    layout = fig.layout.to_plotly_json()
    ids = [k for k in layout if k.startswith("legend") and k[6:].isdigit()]
    assert len(ids) >= 4
    for lid in ids:
        assert not (layout[lid].get("title") or {}).get("text")
        groups = {}
        for t in fig.data:
            if t.legend == lid and t.showlegend is not False:
                groups.setdefault(t.legendgroup, t.name)
        col_px = 0.46 * (_MIN_FIG_WIDTH - 64 - 24)
        assert sum(_legend_width(n) for n in groups.values()) <= col_px or len(groups) == 1, (lid, groups)
    texts = {a.text for a in fig.layout.annotations}
    assert {"Total loss", "Direction head MCC (validation / training)", "Gradient norm: epoch mean, both groups, "
            "pre-clip"} <= texts
    assert fig.layout.margin.r == 24
    assert all(ax.showticklabels for ax in (fig.layout[k] for k in layout if k.startswith("xaxis")))


def test_gradient_panel_draws_the_clip_level(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(viz_history(), viz_config)
    xa, ya = _panel_axes(fig, 6, 2)
    assert any(s.type == "line" and s.y0 == viz_config.GRAD_CLIP_NORM and s.yref == ya for s in fig.layout.shapes)
    (gn,) = [t for t in fig.data if t.legendgroup == "gn"]
    assert gn.line.color == T.INK_2


def test_up_rate_panel_plots_the_bias_with_no_unexplained_style(viz_config, viz_history):
    """Finding 102: three dash-dot 'true' lines drew on top of each other with no key."""
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    rows = viz_history()
    fig = training_dashboard_figure(rows, viz_config)
    assert not any((t.name or "").endswith(" true") for t in fig.data)
    assert not any(getattr(t.line, "dash", None) == T.ALT_DASH for t in fig.data)
    h0 = [t for t in _panel_traces(fig, 5, 1) if t.name == "h0 (10 bars)"]
    assert h0 and np.allclose(h0[0].y, 0.48 - 0.5)


def test_isolated_points_keep_markers_in_long_runs_and_the_figure_stays_small(viz_config):
    from tests.conftest import _viz_history
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    rows = _viz_history(200)
    for e, r in enumerate(rows):
        r["val_ife_loss"] = 1e-3 if e == 100 else 0.0
    fig = training_dashboard_figure(rows, viz_config)
    assert len(fig.to_json()) < 600_000
    (ife,) = [t for t in fig.data if (t.name or "").startswith("IFE")]
    sizes = np.asarray(ife.marker.size)
    assert sizes.shape == (200,) and sizes[100] > 0 and sizes[50] == 0


# ------------------------------------------------------------------ epoch table and companions
def test_epoch_table_prints_the_served_epoch_numbers(viz_config):
    """Findings 80 / 104 / 27: exact values without hover, weights and shares, zeros as 0."""
    from neural_trade.visualization.training_dashboard import epoch_table_html

    rows = _weighted_rows()
    for r in rows:
        for h in H:
            r.update({f"val_dir_brier_{h}": 0.2535, f"val_true_up_rate_{h}": 0.46, f"val_dir_acc_{h}": 0.497,
                      f"val_dir_mcc_{h}": 0.01, f"val_dir_loss_{h}": 0.7})
    html = epoch_table_html(rows, viz_config, weights_epoch=2, n_val=2866)
    assert "epoch 2 (served)" in html and "epoch 4 (last)" in html
    assert "λ_dir 0.5" in html and "20.0%" in html                # direction 1.0 of val loss 5.0
    assert ">0<" in html                                          # the vacuum term, exactly zero
    assert "-2.1%" in html and "ref 0.2484" in html and "ref 0.540" in html
    assert "±0.12" in html


def test_direction_detail_figure_draws_every_direction_metric(viz_config, viz_history):
    """Finding 26: accuracy, F1, sensitivity / specificity, Brier and mean P(up) were logged, never drawn."""
    from neural_trade.visualization.training_dashboard import direction_detail_figure

    rows = viz_history()
    for r in rows:
        for h in H:
            for pre in ("val_", "train_"):
                r.update({f"{pre}dir_acc_{h}": 0.5, f"{pre}dir_f1_{h}": 0.45, f"{pre}dir_sensitivity_{h}": 0.6,
                          f"{pre}dir_specificity_{h}": 0.4, f"{pre}dir_brier_{h}": 0.25,
                          f"{pre}mean_dir_prob_{h}": 0.5, f"{pre}true_up_rate_{h}": 0.46})
    fig = direction_detail_figure(rows, viz_config, weights_epoch=3)
    assert T.empty_panels(fig) == []
    sens = [t for t in fig.data if "sensitivity" in (t.name or "") and t.legendgroup != "key-sensitivity ▲"]
    spec = [t for t in fig.data if "specificity" in (t.name or "") and t.legendgroup != "key-specificity ▼"]
    assert sens and all(t.line.color == T.UP_COLOR and t.marker.symbol == "triangle-up" for t in sens)
    assert spec and all(t.line.color == T.DOWN_COLOR and t.marker.symbol == "triangle-down" for t in spec)
    refs = [t for t in fig.data if t.line.dash == "dash" and t.showlegend is False]
    assert len(refs) == 9                                          # majority, always-up F1, base-rate Brier x 3
    assert any(s.type == "line" and s.x0 == 3 for s in fig.layout.shapes)
    assert not [s for s in fig.layout.shapes if s.type == "rect"]   # validation size unknown: no noise ranges
    # the training lines get references from the training block's own up-rate (dotted), not the validation one
    for r in rows:
        for h in H:
            r[f"train_true_up_rate_{h}"] = 0.52
    fig = direction_detail_figure(rows, viz_config, weights_epoch=3, n_val=2866)
    train_refs = [t for t in fig.data if t.line.dash == T.TRAIN_DASH and t.showlegend is False
                  and t.line.color == T.NEUTRAL]
    assert len(train_refs) == 9 and all("training block" in t.hovertemplate for t in train_refs)
    assert any(np.allclose(t.y, 0.52) for t in train_refs)          # majority accuracy of the training block
    # noise ranges around the validation references: accuracy, sens + spec, F1, Brier, bias per horizon
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 18
    _, ya = _panel_axes(fig, 1, 1)
    (acc,) = [s for s in rects if s.yref == ya]
    half = S.Z95 * math.sqrt(0.54 * 0.46 / 286)
    assert math.isclose(acc.y0, 0.54 - half, rel_tol=1e-6) and math.isclose(acc.y1, 0.54 + half, rel_tol=1e-6)


def test_loss_terms_figure_puts_each_term_on_its_own_axis(viz_config):
    from neural_trade.visualization.training_dashboard import LN2, loss_terms_figure

    rows = _weighted_rows()
    for r in rows:
        for pre in ("", "val_"):
            for i in range(3):
                r.update({f"{pre}point_h{i}": 0.1, f"{pre}dir_loss_h{i}": 0.7, f"{pre}nll_h{i}": 0.7,
                          f"{pre}crps_h{i}": 0.3, f"{pre}soft_ece_h{i}": 0.1})
    fig = loss_terms_figure(rows, viz_config)
    assert T.empty_panels(fig) == []
    assert any(s.type == "line" and math.isclose(s.y0, LN2) for s in fig.layout.shapes)
    assert any("× 0.5" in (a.text or "") for a in fig.layout.annotations)
    assert all(fig.layout[k].type != "log" for k in fig.layout.to_plotly_json() if k.startswith("yaxis"))


def test_batch_loss_figure():
    from neural_trade.visualization.training_dashboard import batch_loss_figure

    fig = batch_loss_figure([(0.1, 7.0), (0.5, 6.8)], [(1, 6.0)])
    assert [len(t.x) for t in fig.data] == [2, 1]


def test_batch_figure_shows_the_running_up_calls_per_horizon():
    """Finding 81: the old live strip had a direction row (class collapse within an epoch)."""
    from neural_trade.visualization.training_dashboard import batch_loss_figure

    pts = [(0.25, 7.0), (0.75, 6.9), (1.25, 6.8), (1.75, 6.7)]
    dirp = [(x, {h: (0.5, 0.48, 0.5) for h in H}) for x, _ in pts]
    fig = batch_loss_figure(pts, [(1, 6.0), (2, 5.9)], dir_points=dirp)
    ups = [t for t in fig.data if (t.name or "").endswith("up calls (train)")]
    assert [t.line.color for t in ups] == [T.HORIZON_COLORS[h] for h in H]
    assert all(t.line.dash == T.TRAIN_DASH for t in ups)
    assert np.isnan(np.asarray(ups[0].y, float)).sum() == 1        # the running mean restarts each epoch
    assert sum(1 for s in fig.layout.shapes if s.type == "rect") == 2   # collapse zones


def test_training_session_passes_the_served_epoch_and_validation_size(viz_config, viz_history):
    pytest.importorskip("ipywidgets")
    from types import SimpleNamespace

    from neural_trade.notebook import TrainingSession

    s = TrainingSession(viz_config)
    s.history = _served_rows(8, best=6)
    s.result = SimpleNamespace(weights_epoch=8, fold=SimpleNamespace(val=np.arange(2866)), artifacts_dir=None)
    fig = s.curves_figure()
    assert list(_traces(fig, legendgroup="served")[0].x) == [8]
    assert "±0.12" in s.health_html() and "epoch 8 (served)" in s.epoch_table_html()
    assert s.direction_figure() is not None and s.loss_terms_figure() is not None
    frame = s.history_frame()
    assert list(frame["epoch"])[:2] == [1, 2] and "val_dir_mcc_h2" in frame.columns
