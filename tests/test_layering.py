"""NT-027: no circular imports between ``neural_trade`` subpackages.

The graph is built from the source with the standard library ``ast`` module (so an import
written inside a function or method counts exactly like a top-of-file one) and does not import
any of the code: it cannot be fooled by a module that fails to import, and it does not need
TensorFlow. A dynamic import (``importlib.import_module("...")``, a plain string) is not an
``ast.Import``/``ast.ImportFrom`` node and is deliberately not part of this graph: it is the
project's documented plugin-discovery mechanism (``BaseRegistry.auto_discover``), not a static
layering dependency.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

SRC = Path(__file__).resolve().parents[1] / "src" / "neural_trade"

# scripts/gate_run.py, check_gates.py, backtest_gate.py, direction_experiments.py, ablate.py and
# experiments/ablation.py are the D-023 frozen set: history, not part of the layering being fixed.
FROZEN_ABLATION = SRC / "experiments" / "ablation.py"


def _subpackages() -> Dict[str, Path]:
    return {p.name: p for p in SRC.iterdir()
            if p.is_dir() and not p.name.startswith("__") and not p.name.startswith(".")}


def _pkg_of(modname: str, subpkgs: Set[str]):
    if not modname or not modname.startswith("neural_trade"):
        return None
    parts = modname.split(".")
    if len(parts) < 2:
        return None
    return parts[1] if parts[1] in subpkgs else None


def _iter_py_files(pkgdir: Path):
    for path in pkgdir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def build_import_graph() -> Tuple[Dict[Tuple[str, str], Set[Path]], Dict[str, Set[Path]]]:
    """``{(from_pkg, to_pkg): {files}}`` for every cross-subpackage import in the source tree."""
    subpkgs = set(_subpackages())
    edges: Dict[Tuple[str, str], Set[Path]] = defaultdict(set)
    files_by_pkg: Dict[str, Set[Path]] = defaultdict(set)

    for pkg, pkgdir in _subpackages().items():
        for fpath in _iter_py_files(pkgdir):
            files_by_pkg[pkg].add(fpath)
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        # relative import: only ever used inside a single subpackage (core/) today,
                        # so it can never itself point at a different subpackage.
                        continue
                    targets = [node.module or ""]
                for modname in targets:
                    tp = _pkg_of(modname, subpkgs)
                    if tp and tp != pkg:
                        edges[(pkg, tp)].add(fpath)
    return edges, files_by_pkg


def mutual_pairs(edges) -> Dict[Tuple[str, str], Tuple[Set[Path], Set[Path]]]:
    pairs = {}
    seen = set()
    for (a, b) in edges:
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        if (b, a) in edges:
            pairs[tuple(sorted((a, b)))] = (edges[(a, b)], edges[(b, a)])
    return pairs


def test_no_subpackage_imports_another_that_imports_it_back():
    """No two subpackages of neural_trade import each other (directly, anywhere in the file)."""
    edges, _ = build_import_graph()
    pairs = mutual_pairs(edges)
    if pairs:
        lines = []
        for (a, b), (ab_files, ba_files) in sorted(pairs.items()):
            lines.append(f"{a} <-> {b}:")
            lines.extend(f"    {a} -> {b}: {f.relative_to(SRC)}" for f in sorted(ab_files))
            lines.extend(f"    {b} -> {a}: {f.relative_to(SRC)}" for f in sorted(ba_files))
        raise AssertionError("mutual subpackage imports found:\n" + "\n".join(lines))


def test_evaluation_does_not_import_training_or_experiments():
    edges, _ = build_import_graph()
    for target in ("training", "experiments"):
        files = edges.get(("evaluation", target))
        assert not files, f"evaluation/ imports {target}/ from: {sorted(f.name for f in files)}"


def test_data_does_not_import_visualization():
    edges, _ = build_import_graph()
    files = edges.get(("data", "visualization"))
    assert not files, f"data/ imports visualization/ from: {sorted(f.name for f in files)}"
