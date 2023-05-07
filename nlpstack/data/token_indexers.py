from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import numpy

from nlpstack.data.tokenizers import Token
from nlpstack.data.vocabulary import Vocabulary


class TokenIndexer:
    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        pass

    def get_pad_index(self, vocab: Vocabulary) -> int:
        raise NotImplementedError

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

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return vocab.get_pad_index(self._namespace)

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


class PretrainedTransformerIndexer(TokenIndexer):
    def __init__(
        self,
        pretrained_model_name: str,
        namespace: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._namespace = namespace
        self._pretrained_model_name = pretrained_model_name

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        if self._namespace is not None:
            raise ValueError("Currently, PretrainedTransformerIndexer does not support building vocabulary.")

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return int(self._tokenizer.pad_token_id)

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> dict[str, Any]:
        indices: list[int] = []
        type_ids: list[int] = []
        mask: list[bool] = []
        for token in tokens:
            indices.append(self._tokenizer.convert_tokens_to_ids(token.surface))
            type_ids.append(0)
            mask.append(True)
        return {
            "token_ids": numpy.array(indices, dtype=int),
            "mask": numpy.array(mask, dtype=bool),
            "type_ids": numpy.array(type_ids, dtype=int),
        }
