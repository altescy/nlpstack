import dataclasses
import sys
import typing
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Dict, List, Literal, Optional, Type, Union

if sys.version_info >= (3, 10):
    from types import UnionType
else:

    class UnionType: ...


JsonType = Literal["string", "number", "integer", "boolean", "object", "array", "null"]


def generate_json_schema(
    cls: Type[Any],
    root: bool = True,
    definitions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if definitions is None:
        definitions = {}

    origin = typing.get_origin(cls)
    args = typing.get_args(cls)

    schema: Dict[str, Any] = {}

    if dataclasses.is_dataclass(cls):
        if not root and cls.__name__ in definitions:
            return {"$ref": f"#/definitions/{cls.__name__}"}

        properties = {}
        required = []
        for field in dataclasses.fields(cls):
            if not is_optional(field.type) or field.default is dataclasses.MISSING:
                required.append(field.name)
            properties[field.name] = generate_json_schema(field.type, root=False, definitions=definitions)

        schema = {"type": "object", "properties": properties, "required": required}
        if not root:
            definitions[cls.__name__] = schema
            schema = {"$ref": f"#/definitions/{cls.__name__}"}
    elif is_namedtuple(cls):
        if not root and cls.__name__ in definitions:
            return {"$ref": f"#/definitions/{cls.__name__}"}

        properties = {}
        required = []
        for field_name, field_type in typing.get_type_hints(cls).items():
            if not is_optional(field_type) or field_name not in cls._field_defaults:
                required.append(field_name)
            properties[field_name] = generate_json_schema(field_type, root=False, definitions=definitions)

        schema = {"type": "object", "properties": properties, "required": required}
        if not root:
            definitions[cls.__name__] = schema
            schema = {"$ref": f"#/definitions/{cls.__name__}"}
    elif origin and args:  # for handling Optional, List, Dict, Literal and other special types
        if origin in (Union, UnionType):  # for Optional and UnionType
            types = [
                generate_json_schema(t, root=False, definitions=definitions)
                for t in cls.__args__
                if t is not type(None)  # noqa
            ]  # exclude None
            schema = (
                {"anyOf": types} if len(types) > 1 else types[0]
            )  # if only one type excluding None, no need to use array
        elif origin in (list, List, Sequence):  # for List
            schema = {
                "type": "array",
                "items": generate_json_schema(args[0], root=False, definitions=definitions),
            }
        elif origin in (dict, Dict, Mapping, MutableMapping):  # for Dict
            schema = {
                "type": "object",
                "additionalProperties": generate_json_schema(args[1], root=False, definitions=definitions),
            }
        elif origin is Literal:  # for Literal
            schema = {"enum": list(cls.__args__)}
        else:
            schema = {"type": "object"}  # other special types will be treated as object
    else:
        schema = {"type": _get_json_type(cls)}

    if root:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": definitions,
            **schema,
        }
    return schema


def is_optional(python_type: Type) -> bool:
    origin = typing.get_origin(python_type)
    args = typing.get_args(python_type)
    if origin:
        return origin is Union and type(None) in args
    return False


def is_namedtuple(python_type: Type) -> bool:
    bases = getattr(python_type, "__bases__", [])
    if len(bases) != 1 or bases[0] != tuple:
        return False
    f = getattr(python_type, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _get_json_type(python_type: Union[Type, None]) -> JsonType:
    if python_type == int:
        return "integer"
    elif python_type == float:
        return "number"
    elif python_type == str:
        return "string"
    elif python_type == bool:
        return "boolean"
    elif python_type is None:
        return "null"
    elif python_type == list:
        return "array"
    return "object"
