from logging import getLogger
from os import PathLike
from typing import Dict, NamedTuple, Union

try:
    import transformers
except ModuleNotFoundError:
    transformers = None


logger = getLogger(__name__)


class _ModelSpec(NamedTuple):
    model_name: str
    with_head: bool


class _TokenizerSpec(NamedTuple):
    model_name: str


_model_cache: Dict[_ModelSpec, "transformers.PretrainedModel"] = {}
_tokenizer_cache: Dict[_TokenizerSpec, "transformers.PreTrainedTokenizer"] = {}


def get_pretrained_model(
    pretrained_model_name_or_path: Union[str, PathLike],
    with_head: bool = False,
) -> "transformers.PretrainedModel":
    global _model_cache
    if transformers is None:
        raise ModuleNotFoundError("transformers is not installed.")
    spec = _ModelSpec(str(pretrained_model_name_or_path), with_head)
    if spec in _model_cache:
        logger.debug(f"Found cached model: {spec}")
    else:
        if with_head:
            _model_cache[spec] = transformers.AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)
        else:
            _model_cache[spec] = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path)
    return _model_cache[spec]


def get_pretrained_tokenizer(
    pretrained_model_name_or_path: Union[str, PathLike],
) -> "transformers.PreTrainedTokenizer":
    global _tokenizer_cache
    if transformers is None:
        raise ModuleNotFoundError("transformers is not installed.")
    spec = _TokenizerSpec(str(pretrained_model_name_or_path))
    if spec in _tokenizer_cache:
        logger.debug(f"Found cached tokenizer: {spec}")
    else:
        _tokenizer_cache[spec] = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    return _tokenizer_cache[spec]


def clear() -> None:
    global _model_cache
    global _tokenizer_cache
    _model_cache.clear()
    _tokenizer_cache.clear()
