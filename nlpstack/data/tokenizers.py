from __future__ import annotations

from typing import NamedTuple


class Token(NamedTuple):
    surface: str
    postag: str | None = None
    lemma: str | None = None


class Tokenizer:
    def tokenize(self, text: str) -> list[Token]:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[Token]:
        return [Token(surface) for surface in text.split()]


class PretrainedTransformerTokenizer(Tokenizer):
    def __init__(self, pretrained_model_name: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def tokenize(self, text: str) -> list[Token]:
        tokens = self.tokenizer.tokenize(text)
        return [Token(t) for t in tokens]
