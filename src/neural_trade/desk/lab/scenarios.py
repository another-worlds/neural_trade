"""Lab scenarios (DESK-5.2, PLAN A8): list the experiment-engine scenario specs, read one, validate an edit, show its
plan table and copy it, with edits, into a desk directory (the router passes ``configs/scenarios/desk/``).

The engine does the parsing and checking (``Scenario.from_yaml`` / ``from_dict`` / ``validate``, ``Runner.plan``); this
module only wraps it for a request handler: a bad spec never raises a ``ScenarioError`` at the caller (it is returned
as text), and :func:`copy_with_edits` writes one file, inside the target directory only, atomically, never over an
existing file, and only after the edited spec has loaded and validated.

Gaps in CONTRACTS (minimum implemented): C5 names ``GET /lab/scenarios``, ``GET /lab/scenarios/{name}`` and
``POST /lab/scenarios`` without fields. The row fields here and the ``edits`` format are this module's own:
``edits`` maps a top-level spec key (``folds``, ``seeds``, ``variants``, ...) to its new value, and a dotted key
(``strategy.name``, ``run.train``) sets one nested key. ``name`` is never an edit: the copy's name is ``new_name``.
Import-light: the engine is imported inside functions.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from neural_trade.desk.config import PROJECT_ROOT

__all__ = ["SCENARIO_DIRS", "DESK_SCENARIO_DIR", "ScenarioRefused", "discover", "discover_all", "read", "validate",
           "plan_table", "copy_with_edits"]

DESK_SCENARIO_DIR = "configs/scenarios/desk"
SCENARIO_DIRS = ("configs/scenarios", DESK_SCENARIO_DIR)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")     # the engine's NAME_RE: 1-48 characters
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"} | {f"{p}{d}" for p in ("COM", "LPT") for d in "123456789¹²³"})


class ScenarioRefused(ValueError):
    """A scenario name, source or write the Lab refuses (not a spec problem: those come back as text)."""


def _dir(specs_dir: Union[str, Path], root: Optional[Union[str, Path]]) -> Path:
    p = Path(specs_dir)
    return p if p.is_absolute() else Path(root if root is not None else PROJECT_ROOT) / p


def discover(specs_dir: Union[str, Path] = "configs/scenarios", *, root: Optional[Union[str, Path]] = None
             ) -> List[Dict[str, Any]]:
    """Every ``*.yaml`` of one directory, through the notebook panel's ``discover_scenarios`` (so the two lists
    agree); a file that does not load is listed with its ``problem``."""
    from neural_trade.notebook.panel_data import discover_scenarios

    rows: List[Dict[str, Any]] = []
    for c in discover_scenarios(_dir(specs_dir, root)):
        sc = c.scenario
        rows.append({"name": c.name, "path": str(c.path), "problem": c.problem, "trains": c.trains,
                     "description": sc.description if sc else "",
                     "folds": list(sc.folds) if sc else [], "seeds": list(sc.seeds) if sc else [],
                     "variants": list(sc.variants) if sc else [], "axes": sorted(sc.axes) if sc else []})
    return rows


def discover_all(dirs=SCENARIO_DIRS, *, root: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """:func:`discover` over the scenario directories, with a ``source`` of ``repo`` or ``desk`` on each row."""
    out: List[Dict[str, Any]] = []
    for d in dirs:
        for row in discover(d, root=root):
            row["source"] = "desk" if Path(d).as_posix().rstrip("/").endswith("/desk") else "repo"
            out.append(row)
    return out


def read(path: Union[str, Path]) -> Dict[str, Any]:
    """``{"name", "path", "text", "spec", "problem"}``: the YAML text, the parsed spec (``Scenario.to_dict``) and, when
    the file does not load as a scenario, ``spec: None`` with the engine's message."""
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.scenario import Scenario

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        sc = Scenario.from_yaml(path)
    except (InvalidConfigurationError, ValueError, TypeError, KeyError) as exc:
        return {"name": path.stem, "path": str(path), "text": text, "spec": None,
                "problem": f"{type(exc).__name__}: {exc}"}
    return {"name": sc.name, "path": str(path), "text": text, "spec": sc.to_dict(), "problem": None}


def validate(source: Union[str, Path, Mapping[str, Any]], *, base_dir: Optional[Union[str, Path]] = None
             ) -> Optional[str]:
    """None when the spec (a file path or a dict) loads and every cell's Config validates, else the problem text.
    Never raises to the caller for a bad spec."""
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.scenario import Scenario

    try:
        if isinstance(source, Mapping):
            sc = Scenario.from_dict(source, base_dir=base_dir)
        else:
            sc = Scenario.from_yaml(source)
        sc.validate()
    except (InvalidConfigurationError, ValueError, TypeError, KeyError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def plan_table(path: Union[str, Path], store: Union[str, Path], *, check_components: bool = True) -> Dict[str, Any]:
    """``Runner.from_spec(path, store).plan()`` as ``{"name", "n_cells", "n_pending", "n_done", "n_failed", "cells"}``;
    ``cells`` are the engine's ``plan_table`` rows. A spec that does not plan returns ``{"problem": text}``. The
    plan writes the run store's index only, so pass the desk's own store directory."""
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.runner import Runner, plan_table as _rows

    try:
        runner = Runner.from_spec(path, str(store), check_components=check_components)
        rows = _rows(runner.plan())
    except (InvalidConfigurationError, ValueError, TypeError, KeyError, OSError) as exc:
        return {"problem": f"{type(exc).__name__}: {exc}"}

    def count(state: str) -> int:
        return sum(1 for r in rows if r["state"] == state)

    return {"name": runner.scenario.name, "n_cells": len(rows), "n_pending": count("pending"),
            "n_done": count("done"), "n_failed": count("failed"), "cells": rows}


def _check_new_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ScenarioRefused("a scenario copy needs a name")
    if name != name.strip() or name.endswith(".") or "/" in name or "\\" in name or ":" in name or ".." in name \
            or os.path.isabs(name):
        raise ScenarioRefused(f"bad scenario name {name!r}: no path parts, drive letters, '..' or trailing dot/space")
    if name.split(".", 1)[0].rstrip(" ").upper() in _RESERVED:
        raise ScenarioRefused(f"bad scenario name {name!r}: a reserved device name")
    if not _NAME.match(name):
        raise ScenarioRefused(f"bad scenario name {name!r}: 1-48 characters of letters, digits, '.', '_' and '-'")
    return name


def _set_path(spec: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = spec
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def copy_with_edits(source: Union[str, Path], new_name: str, edits: Optional[Mapping[str, Any]],
                    target_dir: Union[str, Path]) -> Dict[str, Any]:
    """Write ``<target_dir>/<new_name>.yaml``: the source spec with ``edits`` applied and ``name`` set to ``new_name``.

    Returns ``{"ok": True, "name", "path"}`` or ``{"ok": False, "problem": text}`` where the text is the engine's
    ``ScenarioError`` for the edited spec; in the second case nothing is written. Raises :class:`ScenarioRefused` for a
    name that is unsafe, ``FileExistsError`` when the target exists (never overwritten), ``FileNotFoundError`` for a
    missing source. A relative ``base_config`` is rebased so it still points at the same file from the target."""
    import yaml

    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.utils.atomic import atomic_write_text

    new_name = _check_new_name(new_name)
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"scenario spec {source} does not exist")
    target_dir = Path(target_dir)
    target = target_dir / f"{new_name}.yaml"
    if target.resolve().parent != target_dir.resolve():
        raise ScenarioRefused(f"scenario name {new_name!r} leaves the target directory")
    if target.exists():
        raise FileExistsError(f"scenario {target.name} already exists: choose another name")
    try:
        spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {"ok": False, "problem": f"ScenarioError: scenario spec {source}: not valid YAML: {exc}"}
    if not isinstance(spec, dict):
        return {"ok": False, "problem": f"ScenarioError: {source}: expected a mapping of scenario keys"}
    spec = copy.deepcopy(spec)
    for key, value in (edits or {}).items():
        if not isinstance(key, str) or not key or key.split(".")[0] == "name":
            return {"ok": False, "problem": f"ScenarioError: {key!r} is not an editable key (the name is new_name)"}
        _set_path(spec, key, value)
    spec["name"] = new_name
    base = spec.get("base_config")
    if isinstance(base, str) and base and not os.path.isabs(base):
        resolved = (source.parent / base).resolve()
        try:
            spec["base_config"] = Path(os.path.relpath(resolved, target_dir.resolve())).as_posix()
        except ValueError:                       # another drive: no relative path exists
            spec["base_config"] = str(resolved)
    try:
        text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
        Scenario.from_dict(yaml.safe_load(text), source=target, base_dir=target_dir).validate()
    except (InvalidConfigurationError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        return {"ok": False, "problem": f"{type(exc).__name__}: {exc}"}
    target_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"scenario {target.name} already exists: choose another name")
    atomic_write_text(target, text)
    return {"ok": True, "name": new_name, "path": str(target)}
