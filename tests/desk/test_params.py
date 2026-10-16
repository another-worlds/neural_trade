"""DESK-0.4: ParamSpec, params_schema, validate_params."""
from __future__ import annotations

import pytest

from neural_trade.desk.core.params import ParamSpec, params_schema, validate_params

SPECS = (
    ParamSpec("length", "int", 14, 2, 500, 1),
    ParamSpec("k", "float", 2.0, 0.1, 5.0),
    ParamSpec("on", "bool", True),
    ParamSpec("mode", "choice", "a", choices=["a", "b"]),
)


def test_defaults_filled_and_input_untouched():
    given = {"length": 20}
    out = validate_params(SPECS, given)
    assert out == {"length": 20, "k": 2.0, "on": True, "mode": "a"} and given == {"length": 20}


def test_float_accepts_int_and_int_accepts_integral_float():
    out = validate_params(SPECS, {"k": 3, "length": 10.0})
    assert out["k"] == 3.0 and isinstance(out["k"], float) and isinstance(out["length"], int)


@pytest.mark.parametrize(("params", "needle"), [
    ({"length": 1}, "length: 1 is below the minimum 2"),
    ({"length": 501}, "length: 501 is above the maximum 500"),
    ({"k": 5.5}, "k: 5.5 is above the maximum 5.0"),
    ({"mode": "z"}, "mode"),
    ({"length": 3.5}, "length"),
    ({"on": 1}, "on"),
    ({"nope": 1}, "nope"),
])
def test_violations_name_the_parameter_and_bound(params, needle):
    with pytest.raises(ValueError, match=needle):
        validate_params(SPECS, params)


def test_schema_is_json_ready():
    import json

    schema = params_schema(SPECS)
    assert [s["name"] for s in schema] == ["length", "k", "on", "mode"]
    assert schema[0]["minimum"] == 2 and schema[3]["choices"] == ["a", "b"]
    json.dumps(schema)
