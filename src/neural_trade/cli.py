"""``neural-trade`` command line.

    neural-trade train    --config configs/default.yaml [--set KEY=VALUE ...] [--epochs N]
    neural-trade predict  --artifacts runs/<id>/artifacts --csv bars.csv [--out preds.csv] [--last]
    neural-trade backtest --artifacts runs/<id>/artifacts --csv bars.csv [--strategy NAME] [--params YAML]
    neural-trade registry list | info REGISTRY NAME | search QUERY
    neural-trade env

``train`` creates a run directory (runs/<UTC time>-<git sha>-<config hash>/) holding the
config, per-epoch metrics, the serving bundle (artifacts/) and the test evaluation report.
Values given to ``--set`` are parsed as YAML (``--set EPOCHS=5 --set HORIZON_STEPS=[5,10,20]``).
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
    return 0


def cmd_predict(args) -> int:
    import pandas as pd

    from neural_trade.serving.predictor import Predictor

    predictor = Predictor.from_artifacts(args.artifacts)
    df = pd.read_csv(args.csv)
    if args.last:
        close = df[[c for c in df.columns if c.lower() == "close"][0]].to_numpy()
        print(json.dumps(predictor.predict_last(close, alpha=args.alpha), indent=2))  # noqa: T201
        return 0
    frame = predictor.predict_frame(df, alpha=args.alpha)
    if args.out:
        frame.to_csv(args.out)
        logger.info("%d rows -> %s", len(frame), args.out)
    else:
        print(frame.tail(args.tail).to_string())  # noqa: T201
    return 0


def cmd_backtest(args) -> int:
    import pandas as pd

    from neural_trade.serving.predictor import Predictor
    from neural_trade.strategy import (Bars, SignalFrame, backtest, build_backtest_config, build_strategy, load_params,
                                       var_scale_from)

    predictor = Predictor.from_artifacts(args.artifacts)
    batch, df, anchors = predictor.predict_windows_frame(pd.read_csv(args.csv))
    frame = batch.to_prediction_frame(predictor.bundle.pred_scale, predictor.bundle.pred_mean)
    params = load_params(args.params) if args.params else {}
    strategy = build_strategy(args.strategy or params.get("strategy", "enhanced_multi_horizon"), params.get("params"))
    bcfg = build_backtest_config({**(params.get("backtest") or {}), "random_seeds": args.random_seeds})
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


def cmd_registry(args) -> int:
    from neural_trade.registries import all_registries, load_all, registry_summary

    load_all(None, plugins_dir=args.plugins, strict=False)
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

    p = sub.add_parser("predict", help="forecast every window of a CSV with a saved bundle")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--last", action="store_true", help="only the newest window, as JSON")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tail", type=int, default=10)
    p.set_defaults(func=cmd_predict)

    b = sub.add_parser("backtest", help="backtest a strategy on a CSV with a saved bundle")
    b.add_argument("--artifacts", required=True)
    b.add_argument("--csv", required=True)
    b.add_argument("--strategy", default=None)
    b.add_argument("--params", default=None, help="strategy/backtest YAML (strategy.params.from_file format)")
    b.add_argument("--out", default=None)
    b.add_argument("--plot", action="store_true", help="write backtest.html into --out")
    b.add_argument("--random-seeds", type=int, default=100)
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("registry", help="list / inspect / search the component registries")
    r.add_argument("action", choices=["list", "info", "search"])
    r.add_argument("registry", nargs="?")
    r.add_argument("name", nargs="?")
    r.add_argument("--plugins", default=None, help="also load plugins from this directory")
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
