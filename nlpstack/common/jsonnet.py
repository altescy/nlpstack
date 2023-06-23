import json
import os
from os import PathLike
from typing import Any, Dict, Optional, Union

from rjsonnet import evaluate_file


def _is_encodable(value: str) -> bool:
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def load_jsonnet(filename: Union[str, PathLike], ext_vars: Optional[Dict[str, Any]] = None) -> Any:
    ext_vars = {**_environment_variables(), **(ext_vars or {})}
    return json.loads(evaluate_file(str(filename), ext_vars=_environment_variables()))
