"""Parameter specs shared by strategies and indicators (C1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

KINDS = ("int", "float", "bool", "choice", "str")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None
    doc: str = ""


def params_schema(specs: tuple[ParamSpec, ...]) -> list[dict]:
    """JSON-ready description of the specs (dialogs, sweep spaces)."""
    return [asdict(s) for s in specs]


def _coerce(spec: ParamSpec, value: Any) -> Any:
    n = spec.name
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{n}: expected bool, got {value!r}")
        return value
    if spec.kind == "int":
        if isinstance(value, bool) or not (isinstance(value, int) or (isinstance(value, float) and value.is_integer())):
            raise ValueError(f"{n}: expected int, got {value!r}")
        return int(value)
    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError(f"{n}: expected float, got {value!r}")
        return float(value)
    if spec.kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{n}: expected str, got {value!r}")
        return value
    if spec.kind == "choice":
        if spec.choices is None or value not in spec.choices:
            raise ValueError(f"{n}: {value!r} is not one of {spec.choices!r}")
        return value
    raise ValueError(f"{n}: unknown kind {spec.kind!r}")


def validate_params(specs: tuple[ParamSpec, ...], params: dict) -> dict:
    """Return a new dict with defaults filled; raise ValueError naming the parameter and the violated bound.

    Unknown keys are refused. ``step`` is a UI/sweep hint and is not enforced.
    """
    known = {s.name: s for s in specs}
    extra = sorted(set(params) - set(known))
    if extra:
        raise ValueError(f"unknown parameter(s): {', '.join(extra)}")
    out: dict = {}
    for s in specs:
        v = _coerce(s, params[s.name] if s.name in params else s.default)
        if s.kind in ("int", "float"):
            if s.minimum is not None and v < s.minimum:
                raise ValueError(f"{s.name}: {v} is below the minimum {s.minimum}")
            if s.maximum is not None and v > s.maximum:
                raise ValueError(f"{s.name}: {v} is above the maximum {s.maximum}")
        out[s.name] = v
    return out
