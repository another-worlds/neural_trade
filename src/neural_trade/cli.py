"""``neural-trade`` command line.

    neural-trade train    --config configs/default.yaml [--set KEY=VALUE ...] [--epochs N]
    neural-trade predict  --artifacts runs/<id>/artifacts --csv bars.csv [--out preds.csv] [--last]
    neural-trade backtest --artifacts runs/<id>/artifacts --csv bars.csv [--strategy NAME] [--params YAML]
    neural-trade scenario run|plan configs/scenarios/<name>.yaml [--store runs] [--max-cells N] [--retry-failed]
    neural-trade scenario reindex [--store runs] [--index runs/index.sqlite]
    neural-trade scenario rescore configs/scenarios/<name>.yaml --study configs/strategy_studies/<study>.yaml
                                  [--store runs] [--random-seeds N]
    neural-trade screen configs/screens/<name>.yaml [--shard i/N] [--store runs] [--max-trials N]
    neural-trade compare configs/compares/<name>.yaml [--out DIR] [--simulate] [--n-sim N]
    neural-trade leaderboard [SCENARIO] [--store runs] [--index runs/index.sqlite]
                                  [--spec FILE] [--out DIR] [--max-drawdown F] [--min-trades N]
                                  [--random-null-percentile P] [--no-beat-buy-and-hold] [--no-beat-random-null]
    neural-trade registry list | info REGISTRY NAME | search QUERY
    neural-trade env

``train`` creates a run directory (runs/<UTC time>-<git sha>-<config hash>/) holding the
config, per-epoch metrics, the serving bundle (artifacts/) and the test evaluation report.
Values given to ``--set`` are parsed as YAML (``--set EPOCHS=5 --set HORIZON_STEPS=[5,10,20]``).

``screen`` (neural_trade.experiments.screen, NT-088) is a separate, lighter path for mass sub-30-second
trials (a grid and/or a random/LHS sample of Config fields, crossed with DATA_END slices and seeds):
one JSON line per trial in <store>/screens/<name>/results.jsonl (health numbers, per-horizon direction
AUC, pass/fail against the spec's rules), no baselines, backtest, random null, npz or serving bundle,
and no per-trial run directory. It finds broken math and unstable configurations; ranking quality is
``scenario run``'s job, on the survivors (see docs/RUNBOOK.md "Screen mode"). ``--shard i/N``: each
shard writes its OWN file (results.shard-i-of-N.jsonl, 0-indexed), never the shared results.jsonl;
``neural_trade.experiments.screen.merge_results`` reads every shard file back together.

``scenario run`` is the experiment engine (neural_trade.experiments.runner): every (variant,
fold, seed) cell of the spec trains into its own directory under runs/scenarios/<name>/ and is
scored on its fold's out-of-sample block (dev or test fold) with the scenario's strategy after
costs; the sqlite index (runs/index.sqlite) records every cell. Running it again resumes: finished
cells are skipped. ``scenario plan`` validates the spec and lists each cell's state without
training; ``scenario reindex`` rebuilds the index from the run directories. ``scenario rescore``
(CPU, no training) backtests every configuration of a strategy study on the scenario's stored cells
(neural_trade.experiments.rescore) into runs/scenarios/<name>/rescore/<study>-<UTC time>/: cells.csv,
a leaderboard ranked on the dev cells' mean net Sharpe, the normalised study and meta.json.

``compare`` (neural_trade.experiments.comparator, NT-032, D-025) is the paired "A beats B" verdict:
a pre-registered spec names two engine scenarios, a metric, a minimum effect and the judgement folds;
the comparator pairs their runs by (seed, fold), refuses a mismatched pair or too few pairs, and
prints the paired estimate, its interval and the verdict (JSON to stdout, plus <out>/result.json and
<out>/report.md when --out is given). ``--simulate`` adds the calibrated null/power check (spec
needs noise_sd, or seed_sd and block_sd).

``leaderboard`` (neural_trade.experiments.leaderboard, NT-031, D-020) prints, per scenario (one
named, or every scenario under --store when none is given), one Markdown table row per
configuration: the ranking column is the dev-fold net Sharpe after costs (mean over the dev folds
and their seeds, D-046, with the spread and the counts); guard-rails (maximum drawdown, trades,
beating buy-and-hold, beating the random null) sit beside it and can disqualify a row from the
winner; the test-fold columns are shown on every row, labelled "test, not used for ranking"
(D-020), and never affect the order. A configuration with every cell failed appears as a failed,
disqualified row.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_sets(items: Optional[List[str]]) -> Dict[str, object]:
    import yaml

    out = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        value = yaml.safe_load(raw) if raw.strip() else ""
        if isinstance(value, str):  # YAML 1.1 reads 1e-3 as a string
            try:
                value = float(value)
            except ValueError:
                pass
        out[key.strip()] = value
    return out


def _load_config(path: Optional[str], sets: Dict[str, object]):
    from neural_trade.core.config import Config

    cfg = Config.from_yaml(path) if path else Config()
    return cfg.override(**sets) if sets else cfg


# ------------------------------------------------------------------ commands
def cmd_train(args) -> int:
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import evaluate
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.registries import load_all
    from neural_trade.training.trainer import train_and_evaluate

    cfg = _load_config(args.config, _parse_sets(args.set))
    if args.csv:
        cfg.override(CSV_PATH=args.csv)
    load_all(cfg, plugins_dir=getattr(cfg, "PLUGINS_DIR", None), strict=False)
    ctx = RunContext.create(cfg, root=args.runs_dir, seed=args.seed, tags=args.tag or [], name=args.name)
    logger.info("run %s -> %s", ctx.run_id, ctx.run_dir)
    result = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=args.epochs, force=True,
                                calibrate=not args.no_calibrate, fit_calibration=True, save_artifacts=True)
    test = PredictionFrame.from_result(result, "test")
    cal = PredictionFrame.from_result(result, "cal") if result.predictions_cal is not None else None
    baselines = None
    if not args.no_baselines:
        from neural_trade.data.processor import split_arrays

        train = split_arrays(ctx.config)["train"]
        baselines = BaselineSet.fit(train["X"], train["y"], train["last_close"], float(ctx.config.DIR_DEADBAND_BPS))
    report = evaluate(test, ctx.config, baselines=baselines, cal_frame=cal, run_id=ctx.run_id)
    report.to_json(ctx.path("eval_report_test.json"))
    report.to_markdown(ctx.path("eval_report_test.md"))
    h1 = report.model["horizons"]["h1"]
    logger.info("h1: AUC %.4f  MCC %.4f  EV(delta) %.4f  CRPSS %s  coverage90 %s", h1["direction"]["auc"],
                h1["direction"]["mcc"], h1["delta"]["ev"], h1["variance"].get("crpss"),
                h1["variance"].get("coverage90"))
    print(ctx.run_dir)  # noqa: T201 - the command's result, for scripting
    try:
        from neural_trade.serving.indicator_report import write_indicator_report

        written = write_indicator_report(ctx.run_dir)
    except Exception:
        logger.exception("indicator report failed; weights are already saved in %s", ctx.run_dir)
        return 1
    logger.info("indicator report %s", written)
    return 0


def cmd_indicators(args) -> int:
    from neural_trade.serving.indicator_report import write_indicator_report

    print(write_indicator_report(args.run_dir))  # noqa: T201 - the command's result, for scripting
    return 0


def cmd_predict(args) -> int:
    import pandas as pd

    from neural_trade.serving.predictor import Predictor

    predictor = Predictor.from_artifacts(args.artifacts)
    df = pd.read_csv(args.csv)
    if args.last:
        if len(predictor.config.input_series()) > 1:  # OHLCV input (NT-047)
            print(json.dumps(predictor.predict_last(df, alpha=args.alpha), indent=2))  # noqa: T201
            return 0
        close = df[[c for c in df.columns if c.lower() == "close"][0]].to_numpy()
        print(json.dumps(predictor.predict_last(close, alpha=args.alpha), indent=2))  # noqa: T201
        return 0
    frame = predictor.predict_frame(df, alpha=args.alpha, batch_size=args.batch_size)
    if args.out:
        frame.to_csv(args.out)
        logger.info("%d rows -> %s", len(frame), args.out)
    else:
        print(frame.tail(args.tail).to_string())  # noqa: T201
    return 0


def cmd_backtest(args) -> int:
    import pandas as pd

    from neural_trade.serving.predictor import Predictor
    from neural_trade.strategy import (Bars, SignalFrame, Strategies, backtest, build_backtest_config, build_strategy,
                                       load_params, var_scale_from)

    predictor = Predictor.from_artifacts(args.artifacts)
    batch, df, anchors = predictor.predict_windows_frame(pd.read_csv(args.csv), batch_size=args.batch_size)
    frame = batch.to_prediction_frame(predictor.bundle.pred_scale, predictor.bundle.pred_mean)
    params = load_params(args.params) if args.params else {}
    strategy = build_strategy(args.strategy or params.get("strategy", Strategies.default), params.get("params"),
                              calibration=predictor.bundle.meta.get("weighted_direction_quantiles"))
    bcfg = build_backtest_config({**(params.get("backtest") or {}), "random_seeds": args.random_seeds,
                                  "bar_minutes": float(predictor.config.RESAMPLE_MINUTES)})
    var_scale = predictor.bundle.meta.get("var_scale")
    if var_scale is None:
        logger.warning("the artifacts carry no calibration-split var_scale; using this data's own (look-ahead)")
        var_scale = var_scale_from(frame)
    bars = Bars.from_frame(df, anchors)
    res = backtest(SignalFrame.build(frame, float(var_scale)), bars, strategy, bcfg)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "backtest.json").write_text(json.dumps(res.to_dict(), indent=2, default=float), encoding="utf-8")
        res.trades_frame().to_csv(out / "trades.csv", index=False)
        if args.plot:
            from neural_trade.registries.visualizations import Visualizations

            fig = Visualizations.build("plotly_trading", res, predictor.config, bars=bars)
            fig.write_html(str(out / "backtest.html"), include_plotlyjs="cdn")
    s = res.summary
    rnd = res.baselines.get("random_same_freq", {})
    print(json.dumps({"strategy": res.strategy, "n_trades": s["n_trades"], "total_return": s["total_return"],  # noqa: T201
                      "sharpe_net": s["sharpe_net"], "max_drawdown": s["max_drawdown"], "hit_rate": s["hit_rate"],
                      "buy_and_hold_return": res.baselines["buy_and_hold"]["total_return"],
                      "random_percentile_return": rnd.get("percentile_total_return")}, indent=2, default=float))
    return 0


def cmd_scenario(args) -> int:
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.runner import Runner, plan_table
    from neural_trade.experiments.store import RunStore

    store = RunStore(args.store, args.index)
    if args.action == "reindex":
        index = store.rebuild_index()
        rows = index.rows()
        print(json.dumps({"index": str(index.path), "runs": len(rows),  # noqa: T201 - the command's result
                          "scenarios": sorted({r["scenario"] for r in rows})}, indent=2))
        return 0
    if not args.spec:
        raise SystemExit(f"usage: neural-trade scenario {args.action} SPEC (a scenario YAML, see configs/scenarios/)")
    if args.action == "rescore":
        return _scenario_rescore(args, store)
    try:
        runner = Runner.from_spec(args.spec, store=store, claim_cells=bool(args.claim_cells))
        if args.action == "plan":
            cells = plan_table(runner.plan())
            counts = {s: sum(c["state"] == s for c in cells) for s in ("done", "failed", "pending")}
            print(json.dumps({"scenario": runner.scenario.name, "spec_hash": runner.scenario.spec_hash,  # noqa: T201
                              "index": str(store.index_path), "counts": counts, "cells": cells}, indent=2))
            return 0
        report = runner.run(max_cells=args.max_cells, retry_failed=args.retry_failed)
    except InvalidConfigurationError as exc:
        logger.error("scenario refused, nothing was run: %s", exc)
        return 2
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 - the command's result, for scripting
    return 1 if report.failed else 0


def cmd_sweep(args) -> int:
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.sweep import Sweep, SweepOptions
    from neural_trade.experiments.store import RunStore

    opts = SweepOptions(mode=args.mode, n_trials=args.n_trials, stop_after=args.stop_after, max_hours=args.max_hours,
                        parallel=args.parallel, parallel_record=args.parallel_record, sec_per_step=args.sec_per_step,
                        quick_minutes=args.quick_minutes, overhead_s=args.overhead_s, top_k=args.top_k,
                        rerun_seeds=args.rerun_seeds, sampler_seed=args.sampler_seed, resume=args.resume,
                        when_busy=args.when_busy, dry_run=args.dry_run)
    try:
        sweep = Sweep(Scenario.from_yaml(args.spec), RunStore(args.store, args.index), opts,
                      announce=lambda text: print(text, flush=True))  # noqa: T201 - the budget / estimate, before any trial
        result = sweep.run()
    except InvalidConfigurationError as exc:
        logger.error("sweep refused, nothing was started: %s", exc)
        return 2
    print(json.dumps({"sweep": result.sweep_id, "mode": result.mode, "label": result.label, "state": result.state,  # noqa: T201
                      "stop_reason": result.stop_reason, "directory": result.directory,
                      "winner": result.winner, "ranking": result.ranking[:10]}, indent=2, default=str))
    return 0 if result.state in ("complete", "quick_complete", "dry_run") else 1


def _scenario_rescore(args, store) -> int:
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.rescore import RescoreError, StrategyStudy, rescore
    from neural_trade.experiments.scenario import Scenario

    if not args.study:
        raise SystemExit("usage: neural-trade scenario rescore SPEC --study STUDY (a strategy study YAML, "
                         "see configs/strategy_studies/)")
    try:
        report = rescore(Scenario.from_yaml(args.spec), StrategyStudy.from_yaml(args.study), store,
                         random_seeds=args.random_seeds)
    except InvalidConfigurationError as exc:
        logger.error("rescore refused, nothing was run: %s", exc)
        return 2
    except RescoreError as exc:
        logger.error("rescore: nothing to score, nothing was written: %s", exc)
        return 1
    for s in report.skipped:
        logger.warning("rescore skipped %s (%s): %s", s["run_dir"], s["cell_key"], s["reason"])
    print(json.dumps(report.to_dict(), indent=2, default=float))  # noqa: T201 - the command's result
    return 0


def cmd_leaderboard(args) -> int:
    from neural_trade.experiments.leaderboard import (
        build_leaderboard, find_scenario_spec, leaderboard_markdown, scenario_cost_profile, scenario_guard_rails,
        spec_parts,
    )
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import ENGINE_SUBTREE, RunStore

    store = RunStore(args.store, args.index)
    if args.scenario:
        scenarios = [args.scenario]
    else:
        base = store.root / ENGINE_SUBTREE
        scenarios = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not scenarios:
        print(json.dumps({"scenarios": []}))  # noqa: T201 - the command's result
        return 0
    if args.spec and not Path(args.spec).is_file():
        raise SystemExit(f"leaderboard: scenario spec {args.spec} does not exist")
    for name in scenarios:
        # the spec by the scenario's `name:` key (configs/scenarios/*.yaml), else the one the store recorded
        scenario, where = ((Scenario.from_yaml(args.spec), Path(args.spec).name) if args.spec else
                           find_scenario_spec(name, args.specs_dir, store.scenario_dir(name) / "specs"))
        spec, source = scenario_guard_rails(
            scenario, max_drawdown=args.max_drawdown, min_trades=args.min_trades,
            random_null_percentile=args.random_null_percentile,
            beat_buy_and_hold=False if args.no_beat_buy_and_hold else None,
            beat_random_null=False if args.no_beat_random_null else None)
        if scenario is not None:
            source = f"{where}: {source}"
        _, backtest, folds = spec_parts(scenario)
        board = build_leaderboard(store.sync(name), guard_rails=spec, store_root=store.root,
                                  board_cost=scenario_cost_profile(backtest), spec_folds=folds)
        text = leaderboard_markdown(board, guard_rails=spec, guard_rail_source=source)
        print(text)  # noqa: T201 - the command's result
        if args.out:
            from neural_trade.visualization.leaderboard_fig import leaderboard_figure, write_png

            out = Path(args.out) / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "leaderboard.md").write_text(text, encoding="utf-8")
            fig = leaderboard_figure(board)
            fig.write_html(str(out / "leaderboard.html"), include_plotlyjs="cdn")
            if not write_png(fig, out / "leaderboard.png"):
                logger.warning("leaderboard: %s was not written", out / "leaderboard.png")
    return 0


def cmd_screen(args) -> int:
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.screen import ScreenSpec, parse_shard, run_screen

    try:
        shard = parse_shard(args.shard)
        spec = ScreenSpec.from_yaml(args.spec)
        report = run_screen(spec, store=args.store, shard=shard, max_trials=args.max_trials)
    except InvalidConfigurationError as exc:
        logger.error("screen refused, nothing was run: %s", exc)
        return 2
    print(json.dumps(report.to_dict(), indent=2, default=str))  # noqa: T201 - the command's result
    return 0


def cmd_compare(args) -> int:
    from neural_trade.experiments.comparator import (CompareError, CompareSpec, compare, observed_design,
                                                     simulate_error_rates)

    try:
        spec = CompareSpec.from_yaml(args.spec)
        result = compare(spec)
        out = result.to_dict()
        if args.simulate:
            try:
                # calibrated to the design actually paired (judgement folds x seeds per fold), not a
                # made-up default (QA repair round 2, point 3)
                n_folds, seeds_per_fold = observed_design(spec)
                out["simulation"] = simulate_error_rates(spec, n_folds=n_folds, seeds_per_fold=seeds_per_fold,
                                                         n_sim=args.n_sim)
            except CompareError as exc:
                out["simulation"] = {"error": str(exc)}
    except CompareError as exc:
        logger.error("compare refused: %s", exc)
        return 2
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        (out_dir / "report.md").write_text(result.to_markdown(), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))  # noqa: T201 - the command's result, for scripting
    return 1 if result.verdict == "refused" else 0


def cmd_registry(args) -> int:
    from neural_trade.registries import all_registries, load_all, registry_summary

    plugins = args.plugins or ("plugins" if Path("plugins").is_dir() else None)  # the repository's plugins/
    load_all(None, plugins_dir=plugins, strict=False)
    regs = all_registries()
    if args.action == "list":
        if args.registry:
            reg = regs[_registry_key(regs, args.registry)]
            for name in reg.list_names():
                meta = reg.get_metadata(name)
                print(f"{name:28s} {', '.join(meta.get('tags', []))}")  # noqa: T201
        else:
            print(registry_summary())  # noqa: T201
    elif args.action == "info":
        if not args.registry or not args.name:
            raise SystemExit("usage: neural-trade registry info REGISTRY NAME")
        reg = regs[_registry_key(regs, args.registry)]
        print(json.dumps(reg.get_metadata(args.name), indent=2, default=str))  # noqa: T201
    else:  # search
        query = args.registry or args.name
        if not query:
            raise SystemExit("usage: neural-trade registry search QUERY")
        for rname, reg in regs.items():
            for hit in reg.search(query):
                print(f"{rname}: {hit}")  # noqa: T201
    return 0


def _registry_key(regs, name: str) -> str:
    low = {k.lower(): k for k in regs}
    key = low.get(name.lower()) or low.get(name.lower().rstrip("s") + "s")
    if key is None:
        raise SystemExit(f"unknown registry {name!r}; known: {', '.join(regs)}")
    return key


def cmd_env(args) -> int:
    from neural_trade.utils.env import fingerprint

    print(json.dumps(fingerprint(include_devices=not args.no_devices), indent=2, default=str))  # noqa: T201
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neural-trade", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-level", default=None, help="DEBUG, INFO (default), WARNING")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train, evaluate and save a serving bundle")
    t.add_argument("--config", default=None, help="flat YAML of Config fields (default: built-in defaults)")
    t.add_argument("--set", action="append", metavar="KEY=VALUE")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--csv", default=None, help="OHLCV CSV (overrides CSV_PATH)")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--runs-dir", default="runs")
    t.add_argument("--name", default=None, help="suffix for the run id")
    t.add_argument("--tag", action="append")
    t.add_argument("--no-calibrate", action="store_true", help="skip the loss-weight calibration pass")
    t.add_argument("--no-baselines", action="store_true")
    t.set_defaults(func=cmd_train)

    ind = sub.add_parser("indicators", help="write indicator_report.html for an existing run directory")
    ind.add_argument("run_dir")
    ind.set_defaults(func=cmd_indicators)

    p = sub.add_parser("predict", help="forecast every window of a CSV with a saved bundle")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--last", action="store_true", help="only the newest window, as JSON")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tail", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None,
                   help="bulk scoring batch (default: the training batch, bit-identical to training)")
    p.set_defaults(func=cmd_predict)

    b = sub.add_parser("backtest", help="backtest a strategy on a CSV with a saved bundle")
    b.add_argument("--artifacts", required=True)
    b.add_argument("--csv", required=True)
    b.add_argument("--strategy", default=None)
    b.add_argument("--params", default=None, help="strategy/backtest YAML (strategy.params.from_file format)")
    b.add_argument("--out", default=None)
    b.add_argument("--plot", action="store_true", help="write backtest.html into --out")
    b.add_argument("--random-seeds", type=int, default=100)
    b.add_argument("--batch-size", type=int, default=None, help="prediction batch (default: the training batch)")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("scenario", help="run or resume a scenario (the experiment engine), plan it, rebuild "
                                        "the run index, or re-score its stored cells with a strategy study",
                       description="run: train and score every pending cell of a scenario spec (resumable: "
                                   "finished cells are skipped). plan: validate the spec and list each cell's state "
                                   "without training. reindex: rebuild the sqlite index from the run directories. "
                                   "rescore: backtest every configuration of a strategy study (--study) on the "
                                   "scenario's stored cells, on CPU without retraining.")
    s.add_argument("action", choices=["run", "plan", "reindex", "rescore"])
    s.add_argument("spec", nargs="?", help="scenario YAML (configs/scenarios/*.yaml); not used by reindex")
    s.add_argument("--store", default="runs", help="run store root; engine runs go to <store>/scenarios/<name>/")
    s.add_argument("--index", default=None, help="sqlite index (default <store>/index.sqlite)")
    s.add_argument("--max-cells", type=int, default=None,
                   help="train at most N pending cells, then stop (the same command resumes)")
    s.add_argument("--retry-failed", action="store_true", help="train failed cells again (into new directories)")
    s.add_argument("--claim-cells", action="store_true",
                   help="take a lock file per cell before training it, so two processes on one scenario never train "
                        "the same cell (a sweep's parallel trials use it)")
    s.add_argument("--study", default=None,
                   help="rescore: a strategy study YAML (configs/strategy_studies/*.yaml)")
    s.add_argument("--random-seeds", type=int, default=None,
                   help="rescore: random-null seeds per backtest (default: the scenario's backtest setting)")
    s.set_defaults(func=cmd_scenario)

    sw = sub.add_parser("sweep", help="search Config fields on the dev folds (NT-030): quick mode (about 5 minutes, "
                                      "sized from a measured sec_per_step) or optuna mode (a resumable study with a "
                                      "stated GPU budget)",
                        description="The search space is the scenario's `search:` block (FIELD: {low, high, log, step} "
                                    "or choices; only Config fields marked tunable; RESAMPLE_MINUTES is refused until "
                                    "NT-040). quick: trials, epochs and dev folds are sized so that the estimate "
                                    "(printed first) is at most --quick-minutes; results are labelled quick, one seed, "
                                    "no winner. optuna: a TPE study in <store>/sweeps/<id>/study.db; the GPU budget "
                                    "(trials x dev folds x steps x sec_per_step + the top-K x seeds re-run) is "
                                    "printed and recorded before the first trial and refused above --max-hours; "
                                    "after the search the top K are re-run with several seeds and the winner is the "
                                    "best dev-fold seed mean (test columns shown, never ranking). --parallel N "
                                    "launches N trials at once, only up to NT-035's recorded allowed N, after the "
                                    "GPU-free check.")
    sw.add_argument("spec", help="scenario YAML (configs/scenarios/*.yaml) with an optional `search:` block")
    sw.add_argument("--mode", choices=["quick", "optuna"], required=True)
    sw.add_argument("--store", default="runs", help="run store root; trials go to <store>/scenarios/<name>-<mode>/")
    sw.add_argument("--index", default=None, help="sqlite index (default <store>/index.sqlite)")
    sw.add_argument("--resume", action="store_true", help="continue an earlier sweep of this scenario and mode "
                                                          "(without it an existing sweep is refused)")
    sw.add_argument("--n-trials", type=int, default=30, help="optuna: the study's total number of trials")
    sw.add_argument("--stop-after", type=int, default=None,
                    help="run at most N new trials in this call, then stop without the re-run (--resume continues)")
    sw.add_argument("--max-hours", type=float, default=12.0,
                    help="optuna: refuse to start when the estimated GPU budget is above this (default 12: one night)")
    sw.add_argument("--parallel", type=int, default=1, help="trials launched at once (needs NT-035's record)")
    sw.add_argument("--parallel-record", default="runs/experiments/gpu_measurements_v1/parallel_n.json",
                    help="NT-035's result file (allowed_n, utilization); no file means --parallel 1")
    sw.add_argument("--sec-per-step", type=float, default=None,
                    help="measured seconds per training step (default: the latest run of the same setup in the index)")
    sw.add_argument("--quick-minutes", type=float, default=5.0, help="quick: the estimate's ceiling")
    sw.add_argument("--overhead-s", type=float, default=30.0, help="estimated fixed seconds per cell (data, calibration, scoring)")
    sw.add_argument("--top-k", type=int, default=5, help="optuna: trials re-run with several seeds")
    sw.add_argument("--rerun-seeds", type=int, default=3, help="optuna: seeds of the re-run")
    sw.add_argument("--sampler-seed", type=int, default=0)
    sw.add_argument("--when-busy", choices=["stop", "wait"], default="stop",
                    help="when the GPU-free check fails: stop (resume later) or wait and check again")
    sw.add_argument("--dry-run", action="store_true", help="print the estimate / GPU budget and stop")
    sw.set_defaults(func=cmd_sweep)

    sc = sub.add_parser("screen", help="mass, sub-30-second CPU/GPU trials over a grid/sample of Config fields "
                                       "(NT-088): finds broken math and unstable hyperparameter regions, ranks "
                                       "nothing (that is scenario run's job on the survivors)",
                       description="one line per trial in <store>/screens/<name>/results.jsonl: health numbers, "
                                   "per-horizon direction AUC and a pass/fail against the spec's pre-registered "
                                   "rules. Resumable: a trial already in results.jsonl is skipped.")
    sc.add_argument("spec", help="screen spec YAML (configs/screens/*.yaml)")
    sc.add_argument("--shard", default=None, help="i/N: this process runs trial j only when j %% N == i "
                                                  "(shards are disjoint; their union is every trial)")
    sc.add_argument("--store", default="runs", help="run store root; screens go to <store>/screens/<name>/")
    sc.add_argument("--max-trials", type=int, default=None,
                    help="run at most N pending trials, then stop (the same command resumes)")
    sc.set_defaults(func=cmd_screen)

    cp = sub.add_parser("compare", help="a pre-registered paired \"A beats B\" verdict over two scenarios "
                                        "(D-025, NT-032): pairs by (seed, fold), refuses fewer than 5 pairs, "
                                        "a fold the spec does not name, or a mismatched dataset/setup fingerprint")
    cp.add_argument("spec", help="compare spec YAML (configs/compares/*.yaml)")
    cp.add_argument("--out", default=None, help="write result.json and report.md here")
    cp.add_argument("--simulate", action="store_true", help="add the calibrated null/power simulation")
    cp.add_argument("--n-sim", type=int, default=1000)
    cp.set_defaults(func=cmd_compare)

    lb = sub.add_parser("leaderboard", help="print the leaderboard (NT-031, D-020): one row per "
                                            "configuration, ranked by the dev-fold net Sharpe after costs, "
                                            "guard-rails beside it, test-fold columns shown but never ranked")
    lb.add_argument("scenario", nargs="?", help="scenario name (default: every scenario under --store)")
    lb.add_argument("--store", default="runs", help="run store root")
    lb.add_argument("--index", default=None, help="sqlite index (default <store>/index.sqlite)")
    lb.add_argument("--out", default=None, help="also write <out>/<scenario>/leaderboard.md, .html and .png "
                                                "(the PNG through kaleido or headless Edge, when available)")
    lb.add_argument("--max-drawdown", type=float, default=None,
                    help="guard-rail: dev max drawdown must be <= this fraction (default: not checked)")
    lb.add_argument("--spec", default=None, help="scenario spec for the guard-rail thresholds, the board's cost "
                    "profile and the dev folds (default: the --specs-dir YAML whose name: is the scenario, else "
                    "the newest spec the store recorded under <store>/scenarios/<scenario>/specs/)")
    lb.add_argument("--specs-dir", default="configs/scenarios", help="where to look for the scenario's spec")
    lb.add_argument("--min-trades", type=float, default=None,
                    help="guard-rail: trades must be >= this on the dev mean and every dev fold (default 1: "
                         "a 0-trade row is disqualified); overrides the scenario's leaderboard block")
    lb.add_argument("--random-null-percentile", type=float, default=None,
                    help="guard-rail: dev random-null percentile must be >= this (default 50)")
    lb.add_argument("--no-beat-buy-and-hold", action="store_true", help="drop the buy-and-hold guard-rail")
    lb.add_argument("--no-beat-random-null", action="store_true", help="drop the random-null guard-rail")
    lb.set_defaults(func=cmd_leaderboard)

    r = sub.add_parser("registry", help="list / inspect / search the component registries")
    r.add_argument("action", choices=["list", "info", "search"])
    r.add_argument("registry", nargs="?")
    r.add_argument("name", nargs="?")
    r.add_argument("--plugins", default=None, help="plugin directory (default: ./plugins when it exists)")
    r.set_defaults(func=cmd_registry)

    e = sub.add_parser("env", help="versions, CUDA build, devices, git state")
    e.add_argument("--no-devices", action="store_true")
    e.set_defaults(func=cmd_env)
    return ap


def main(argv=None) -> int:
    from neural_trade.core.logging import configure_logging

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, stream="stderr")  # stdout carries each command's result
    try:
        return int(args.func(args) or 0)
    finally:
        configure_logging(stream="stdout")


if __name__ == "__main__":
    sys.exit(main())
