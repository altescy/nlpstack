from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import numpy

from nlp_learn.data.tokenizers import Token
from nlp_learn.data.vocabulary import Vocabulary


class TokenIndexer:
    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        pass

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> dict[str, Any]:
        raise NotImplementedError


class SingleIdTokenIndexer(TokenIndexer):
    def __init__(self, namespace: str = "tokens", feature_name: str = "surface") -> None:
        self._namespace = namespace
        self._feature_name = feature_name

    def _get_token_feature(self, token: Token) -> str:
        feature = str(getattr(token, self._feature_name))
        return feature

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        def document_iterator() -> Iterator[list[str]]:
            for tokens in documents:
                yield [self._get_token_feature(token) for token in tokens]

        vocab.build_vocab_from_documents(self._namespace, document_iterator())

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> dict[str, Any]:
        token_ids = [
            vocab.get_index_by_token(
                self._namespace,
                self._get_token_feature(token),
            )
            for token in tokens
        ]
        mask = [True] * len(token_ids)
        return {"token_ids": numpy.array(token_ids, dtype=int), "mask": numpy.array(mask, dtype=bool)}
