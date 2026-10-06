"""The configuration guard (NT-038, D-026): refuse hyperparameter regions the stability harness showed to fail.

``Config.validate`` calls :func:`check_config` once. Two checks:

* **Failing regions.** ``configs/stability_failing_regions.json`` (written by the stability harness,
  ``neural_trade.experiments.stability``; empty until a harness run finds a failing configuration)
  lists regions as ``{id, case, report, reason, conditions}``. ``conditions`` maps a Config field to
  ``{"min": x, "max": y}`` (inclusive numeric bounds, either may be absent) or ``{"values": [...]}``
  (the field equals one of them). A config is INSIDE a region when every condition holds, and then
  :func:`check_config` raises :class:`InvalidConfigurationError` naming the region, its reason and the
  harness report. The file is found next to the repo's ``configs/``; ``NT_FAILING_REGIONS=<path>``
  points elsewhere and ``NT_FAILING_REGIONS=off`` switches the check off (the harness itself re-runs
  failing cases: :func:`regions_disabled`).
* **GPU memory.** :func:`memory_warning`: BATCH_SIZE x LOOKBACK^2 above :data:`SCORE_ELEMENTS_WARN` is
  logged as a warning (never refused: the bound is two measured points, not a model of the card).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from neural_trade.core.exceptions import InvalidConfigurationError

logger = logging.getLogger(__name__)

REGIONS_ENV = "NT_FAILING_REGIONS"
REGIONS_SCHEMA_VERSION = 1
DEFAULT_REGIONS_FILE = Path(__file__).resolve().parents[3] / "configs" / "stability_failing_regions.json"

#: BATCH_SIZE x LOOKBACK^2 (the element count of one [B, L, L] score tensor per head or indicator
#: channel) above which GPU memory is a risk on the 12 GB RTX 4070 Ti. Evidence, both from the same model
#: (14 indicator families, 4-8 attention heads): runs/scenarios/micro_lookback finished at LOOKBACK 240 with
#: BATCH_SIZE 256 (256 x 240^2 = 14.7M) and ran out of memory at BATCH_SIZE 512 (29.5M) and 2048 (118M);
#: BATCH_SIZE 2048 at LOOKBACK 60 (7.4M, the micro layout of D-041) is fine. The warning level is the
#: largest measured-good product plus 2%: any larger one is unmeasured. Two L^2 tensors count (NT-038
#: amendment 2026-09-30): the batched EWMA weights [B, K, L, L] and the attention scores [B, heads, L, L];
#: their sizes follow the model, so the level may move once the experimenter records a memory profile.
SCORE_ELEMENTS_WARN = 15_000_000
SCORE_ELEMENTS_OOM_MEASURED = 29_491_200       # 512 x 240^2: the smallest measured out-of-memory case


@dataclass(frozen=True)
class Region:
    id: str
    conditions: Mapping[str, Mapping[str, Any]]
    case: str = ""
    report: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "case": self.case, "report": self.report, "reason": self.reason,
                "conditions": {k: dict(v) for k, v in self.conditions.items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Region":
        conds = d.get("conditions")
        if not isinstance(conds, Mapping) or not conds:
            raise InvalidConfigurationError(f"failing region {d.get('id')!r} has no conditions")
        for name, c in conds.items():
            if not isinstance(c, Mapping) or not (set(c) & {"min", "max", "values"}) or set(c) - {"min", "max", "values"}:
                raise InvalidConfigurationError(f"failing region {d.get('id')!r}: condition on {name} must use "
                                                "min / max / values only")
        return cls(str(d.get("id")), {k: dict(v) for k, v in conds.items()}, str(d.get("case", "")),
                   str(d.get("report", "")), str(d.get("reason", "")))

    def condition_holds(self, name: str, value: Any) -> bool:
        c = self.conditions[name]
        if "values" in c and not any(_same(value, v) for v in c["values"]):
            return False
        try:
            if "min" in c and not value >= c["min"]:
                return False
            if "max" in c and not value <= c["max"]:
                return False
        except TypeError:
            return False
        return True

    def contains(self, values: Mapping[str, Any]) -> bool:
        """True when every condition holds for ``values`` (a field missing from it never holds)."""
        return all(n in values and self.condition_holds(n, values[n]) for n in self.conditions)

    def describe(self) -> str:
        parts = []
        for n, c in self.conditions.items():
            if "values" in c:
                parts.append(f"{n} in {list(c['values'])}")
            else:
                parts.append(f"{c.get('min', '-inf')} <= {n} <= {c.get('max', 'inf')}")
        return " and ".join(parts)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b) if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) else False
    return a == b


_DISABLED = 0


@contextlib.contextmanager
def regions_disabled() -> Iterator[None]:
    """The harness re-tests configurations inside known failing regions: the region check is off inside this."""
    global _DISABLED
    _DISABLED += 1
    try:
        yield
    finally:
        _DISABLED -= 1


def regions_path() -> Optional[Path]:
    env = os.environ.get(REGIONS_ENV)
    if env is not None:
        return None if env.strip().lower() in ("", "off") else Path(env)
    return DEFAULT_REGIONS_FILE


def load_regions(path=None) -> List[Region]:
    """The regions of a failing-regions file (default: the repo's); a missing file means none."""
    p = Path(path) if path is not None else regions_path()
    if p is None or not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise InvalidConfigurationError(f"failing-regions file {p} is not valid JSON: {exc}") from exc
    if doc.get("schema_version") != REGIONS_SCHEMA_VERSION:
        raise InvalidConfigurationError(f"failing-regions file {p}: schema_version must be {REGIONS_SCHEMA_VERSION}")
    return [Region.from_dict(r) for r in doc.get("regions", [])]


def write_regions(path, regions: Sequence[Region]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema_version": REGIONS_SCHEMA_VERSION, "regions": [r.to_dict() for r in regions]},
                            indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return p


def config_values(config: Any, names: Sequence[str]) -> Dict[str, Any]:
    return {n: getattr(config, n) for n in names if hasattr(config, n)}


def find_region(config: Any, regions: Optional[Sequence[Region]] = None) -> Optional[Region]:
    regions = load_regions() if regions is None else regions
    for r in regions:
        if r.contains(config_values(config, list(r.conditions))):
            return r
    return None


def memory_warning(config: Any) -> Optional[str]:
    """A message when BATCH_SIZE x LOOKBACK^2 is above the measured-good level, else None."""
    b, lb = int(getattr(config, "BATCH_SIZE", 0) or 0), int(getattr(config, "LOOKBACK", 0) or 0)
    elements = b * lb * lb
    if elements <= SCORE_ELEMENTS_WARN:
        return None
    return (f"BATCH_SIZE {b} x LOOKBACK {lb}^2 = {elements / 1e6:.1f}M exceeds {SCORE_ELEMENTS_WARN / 1e6:.0f}M, the "
            f"largest product measured to fit the 12 GB card (BATCH_SIZE 256, LOOKBACK 240); "
            f"{SCORE_ELEMENTS_OOM_MEASURED / 1e6:.1f}M (512 x 240^2) ran out of memory. The attention scores and the "
            "EWMA weights are quadratic in the window: lower BATCH_SIZE or LOOKBACK (core/guard.py)")


def check_config(config: Any, regions: Optional[Sequence[Region]] = None) -> None:
    """Refuse a config inside a failing region (:class:`InvalidConfigurationError` naming the region and the
    harness report); warn on a BATCH_SIZE x LOOKBACK^2 that may not fit in GPU memory."""
    if regions is not None or not _DISABLED:
        r = find_region(config, regions)
        if r is not None:
            why = r.reason or "the stability harness showed it to fail"
            raise InvalidConfigurationError(
                f"the configuration is inside failing region {r.id!r} ({r.describe()}): {why}; "
                f"case {r.case or '?'}, report {r.report or 'not recorded'}")
    msg = memory_warning(config)
    if msg:
        logger.warning(msg)


__all__ = ["DEFAULT_REGIONS_FILE", "REGIONS_ENV", "Region", "SCORE_ELEMENTS_WARN", "check_config", "config_values",
           "find_region", "load_regions", "memory_warning", "regions_disabled", "regions_path", "write_regions"]
