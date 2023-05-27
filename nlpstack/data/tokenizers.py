from __future__ import annotations

from typing import NamedTuple

import numpy


class Token(NamedTuple):
    surface: str
    postag: str | None = None
    lemma: str | None = None
    vector: numpy.ndarray | None = None


class Tokenizer:
    def tokenize(self, text: str) -> list[Token]:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[Token]:
        return [Token(surface) for surface in text.split()]


class SpacyTokenizer(Tokenizer):
    def __init__(self, lang: str) -> None:
        import spacy

        self.nlp = spacy.load(lang)

    def tokenize(self, text: str) -> list[Token]:
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
    def __init__(self, pretrained_model_name: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def tokenize(self, text: str) -> list[Token]:
        tokens = self.tokenizer.tokenize(text)
        return [Token(t) for t in tokens]
