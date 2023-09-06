from logging import getLogger
from os import PathLike
from typing import Dict, NamedTuple, Optional, Type, Union

try:
    import transformers
except ModuleNotFoundError:
    transformers = None


logger = getLogger(__name__)


class _ModelSpec(NamedTuple):
    model_name: str
    auto_cls: Type


class _TokenizerSpec(NamedTuple):
    model_name: str


_model_cache: Dict[_ModelSpec, "transformers.PretrainedModel"] = {}
_tokenizer_cache: Dict[_TokenizerSpec, "transformers.PreTrainedTokenizer"] = {}


def get_pretrained_model(
    pretrained_model_name_or_path: Union[str, PathLike],
    auto_cls: Optional[Type] = None,
) -> "transformers.PretrainedModel":
    global _model_cache
    if transformers is None:
        raise ModuleNotFoundError("transformers is not installed.")
    auto_cls = auto_cls or transformers.AutoModel
    spec = _ModelSpec(str(pretrained_model_name_or_path), auto_cls)
    if spec in _model_cache:
        logger.debug(f"Found cached model: {spec}")
    else:
        _model_cache[spec] = auto_cls.from_pretrained(pretrained_model_name_or_path)
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
