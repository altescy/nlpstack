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
