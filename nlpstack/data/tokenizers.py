from contextlib import suppress
from os import PathLike
from typing import List, NamedTuple, Optional, Union

import minato
import numpy


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
        import spacy

        self.nlp = spacy.load(lang)

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
        from transformers import AutoTokenizer

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.tokenizer.tokenize(text)
        return [Token(t) for t in tokens]
