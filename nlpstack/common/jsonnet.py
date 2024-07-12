import copy
import itertools
import json
import os
from os import PathLike
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Set, Type, TypeVar, Union, cast

import colt
from rjsonnet import evaluate_file, evaluate_snippet

T = TypeVar("T", Dict, List)


def _is_encodable(value: str) -> bool:
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def _parse_overrides(serialized_overrides: str, ext_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars = {**_environment_variables(), **(ext_vars or {})}
        output = json.loads(evaluate_snippet("", serialized_overrides, ext_vars=ext_vars))
        assert isinstance(output, dict), "Overrides must be a JSON object."
        return output
    else:
        return {}


def _with_overrides(original: T, overrides_dict: Dict[str, Any], prefix: str = "") -> T:
    merged: T
    keys: Union[Iterable[str], Iterable[int]]
    if isinstance(original, list):
        merged = [None] * len(original)
        keys = cast(Iterable[int], range(len(original)))
    elif isinstance(original, dict):
        merged = {}
        keys = cast(
            Iterable[str],
            itertools.chain(original.keys(), (k for k in overrides_dict if "." not in k and k not in original)),
        )
    else:
        if prefix:
            raise ValueError(
                f"overrides for '{prefix[:-1]}.*' expected list or dict in original, " f"found {type(original)} instead"
            )
        else:
            raise ValueError(f"expected list or dict, found {type(original)} instead")

    used_override_keys: Set[str] = set()
    for key in keys:
        if str(key) in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[str(key)])
            used_override_keys.add(str(key))
        else:
            overrides_subdict = {}
            for o_key in overrides_dict:
                if o_key.startswith(f"{key}."):
                    overrides_subdict[o_key[len(f"{key}.") :]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = _with_overrides(original[key], overrides_subdict, prefix=prefix + f"{key}.")
            else:
                merged[key] = copy.deepcopy(original[key])

    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f"overrides dict contains unused keys: {unused_override_keys}")

    return merged


def load_jsonnet(
    filename: Union[str, PathLike],
    ext_vars: Optional[Mapping[str, Any]] = None,
    overrides: Optional[str] = None,
) -> Any:
    ext_vars = {**_environment_variables(), **(ext_vars or {})}
    output = json.loads(evaluate_file(str(filename), ext_vars=ext_vars))
    if overrides:
        output = _with_overrides(output, _parse_overrides(overrides, ext_vars=ext_vars))
    return output


_T_FromJsonnet = TypeVar("_T_FromJsonnet", bound="FromJsonnet")


class FromJsonnet:
    __COLT_BUILDER__: ClassVar = colt.ColtBuilder(typekey="type")

    @classmethod
    def from_jsonnet(
        cls: Type[_T_FromJsonnet],
        filename: Union[str, PathLike],
        ext_vars: Optional[Mapping[str, Any]] = None,
        overrides: Optional[str] = None,
    ) -> _T_FromJsonnet:
        json_config = load_jsonnet(filename, ext_vars=ext_vars, overrides=overrides)
        obj: _T_FromJsonnet = cls.__COLT_BUILDER__(json_config, cls)
        setattr(obj, "__json_config__", json_config)
        return obj

    def to_json(self) -> Any:
        return getattr(self, "__json_config__")
