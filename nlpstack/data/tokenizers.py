from contextlib import suppress
from os import PathLike
from typing import List, NamedTuple, Optional, Union

import minato
import numpy

from nlpstack.common import cached_property
from nlpstack.transformers import cache as transformers_cache

try:
    import transformers
except ModuleNotFoundError:
    transformers = None


try:
    import spacy
except ModuleNotFoundError:
    spacy = None  # type: ignore[assignment]


class Token(NamedTuple):
    surface: str
    postag: Optional[str] = None
    lemma: Optional[str] = None
    vector: Optional[numpy.ndarray] = None


class Tokenizer:
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface) for surface in text.split()]


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface) for surface in text]


class SpacyTokenizer(Tokenizer):
    def __init__(self, lang: str) -> None:
        self._lang = lang

    @cached_property
    def nlp(self) -> "spacy.language.Language":
        if spacy is None:
            raise ModuleNotFoundError("spacy is not installed.")
        return spacy.load(self._lang)

    def tokenize(self, text: str) -> List[Token]:
        doc = self.nlp(text)
        return [
            Token(
                t.text,
                t.pos_,
                t.lemma_,
                vector=numpy.array(t.vector) if t.has_vector else None,
            )
            for t in doc
        ]


class PretrainedTransformerTokenizer(Tokenizer):
    def __init__(self, pretrained_model_name: Union[str, PathLike]) -> None:
        self._pretrained_model_name = pretrained_model_name

    @cached_property
    def tokenizer(self) -> "transformers.PreTrainedTokenizer":
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed.")
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.tokenizer.tokenize(text)  # type: ignore
        return [Token(t) for t in tokens]
