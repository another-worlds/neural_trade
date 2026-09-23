"""Per-package coverage gates (plan C1).

    pytest --cov=neural_trade --cov-report=json:coverage.json -m "not gpu"
    python scripts/check_coverage.py coverage.json

Fails (exit 1) when a package's line coverage is below its gate. Gates are floors; raise them
as the suite grows, never lower them to make a run pass.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

GATES = {                      # package under src/neural_trade -> minimum line coverage (%)
    "losses": 85, "evaluation": 85, "strategy": 85, "calibration": 80, "data": 75, "core": 95,
    "training": 50, "models": 50,
}


def package_of(path: str):
    parts = Path(path).as_posix().split("/")
    if "neural_trade" not in parts:
        return None
    rest = parts[parts.index("neural_trade") + 1:]
    return rest[0] if len(rest) > 1 else None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    data = json.loads(Path(argv[0] if argv else "coverage.json").read_text(encoding="utf-8"))
    covered, total = defaultdict(int), defaultdict(int)
    for path, info in data["files"].items():
        pkg = package_of(path)
        if pkg is None:
            continue
        covered[pkg] += info["summary"]["covered_lines"]
        total[pkg] += info["summary"]["num_statements"]
    failed = False
    for pkg in sorted(total):
        pct = 100.0 * covered[pkg] / total[pkg] if total[pkg] else 100.0
        gate = GATES.get(pkg)
        status = "" if gate is None else ("ok  " if pct >= gate else "FAIL")
        failed |= status == "FAIL"
        print(f"{status:4s} {pkg:15s} {pct:5.1f}%" + (f"  (gate {gate}%)" if gate is not None else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
