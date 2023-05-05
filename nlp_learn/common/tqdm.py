import os
from typing import Any

from tqdm.auto import tqdm as _tqdm

TQDM_DISABLE = os.environ.get("NLP_LEARN_TQDM_DISABLE", "0").lower() in ("1", "true")


def tqdm(*args: Any, **kwargs: Any) -> Any:
    return _tqdm(*args, disable=TQDM_DISABLE, **kwargs)
