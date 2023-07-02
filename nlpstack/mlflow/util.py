from collections import abc
from typing import Any, Dict, Mapping


def flatten_dict_for_mlflow_log(data: Mapping[str, Any]) -> Mapping[str, Any]:
    output: Dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, (int, float, str, bool, type(None))):
            output[prefix] = obj
        elif isinstance(obj, abc.Mapping):
            for key, value in obj.items():
                _flatten(value, f"{prefix}.{key}")
        elif isinstance(obj, abc.Sequence):
            for index, value in enumerate(obj):
                _flatten(value, f"{prefix}.{index}")
        else:
            output[prefix] = obj

    _flatten(data)

    return {key.lstrip("."): value for key, value in output.items()}
