import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from nlpstack.common import generate_json_schema


def test_jsonschema() -> None:
    @dataclasses.dataclass
    class Foo:
        name: str
        label: Literal["xxx", "yyy"]
        value: Union[str, int]

    @dataclasses.dataclass
    class Bar:
        foos: Sequence[Foo]
        metadata: Optional[Mapping[str, Any]] = None

    schema = generate_json_schema(Bar)

    desired_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {
            "Foo": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"enum": ["xxx", "yyy"]},
                    "value": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                },
                "required": ["name", "label", "value"],
            },
        },
        "type": "object",
        "properties": {
            "foos": {
                "type": "array",
                "items": {"$ref": "#/definitions/Foo"},
            },
            "metadata": {
                "type": "object",
                "additionalProperties": {"type": "object"},
            },
        },
        "required": ["foos"],
    }

    assert schema == desired_schema
