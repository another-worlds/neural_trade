"""Run (or resume, or project) the physics-term ablation grid and write its report.

    python scripts/ablate.py --scale smoke                      # 84 short runs
    python scripts/ablate.py --scale full --out runs/ablations/ablate_physics_v1
    python scripts/ablate.py --scale full --dry-run             # pending cells and projected hours
    python scripts/ablate.py --scale full --report-only         # re-analyse completed cells

Each cell trains in its own process and writes cells/<key>.json; re-running skips finished
cells. Outputs: results.csv, summary.csv, analysis.json, report.md (see
neural_trade.experiments.ablation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(REPO / "configs" / "ablation_physics.yaml"))
    ap.add_argument("--criteria", default=str(REPO / "configs" / "ablation_criteria.yaml"))
    ap.add_argument("--scale", default="smoke")
    ap.add_argument("--out", default=None, help="default runs/ablations/<spec name>-<scale>")
    ap.add_argument("--csv", default=str(REPO / "binance_btcusdt_1min_ccxt.csv"))
    ap.add_argument("--limit", type=int, default=None, help="run at most N pending cells")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--sec-per-run", type=float, default=None, help="dry-run estimate before any cell finished")
    args = ap.parse_args(argv)

    from neural_trade.experiments.ablation import AblationSpec, Criteria, dry_run, run_grid, summarize_dir

    spec = AblationSpec.from_yaml(args.spec)
    criteria = Criteria.from_yaml(args.criteria)
    out = Path(args.out or REPO / "runs" / "ablations" / f"{spec.name}-{args.scale}")
    if args.dry_run:
        info = dry_run(spec, out, args.sec_per_run)
        info.pop("pending_keys")
        print(json.dumps(info, indent=2))
        return 0
    if not args.report_only:
        run_grid(spec, args.scale, out, resume=not args.no_resume, limit=args.limit, csv_path=args.csv)
    analysis = summarize_dir(spec, criteria, out, args.scale)
    for term, t in analysis["terms"].items():
        print(f"{term:22s} {t['verdict']}")
    if analysis.get("family"):
        print(f"{'family':22s} {analysis['family']['verdict']}")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
