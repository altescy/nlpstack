from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy

from nlpstack.data.tokenizers import Token
from nlpstack.data.vocabulary import Vocabulary


class TokenIndexer:
    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        pass

    def get_vocab_namespace(self) -> Optional[str]:
        return None

    def get_pad_index(self, vocab: Vocabulary) -> int:
        raise NotImplementedError

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        raise NotImplementedError


class SingleIdTokenIndexer(TokenIndexer):
    def __init__(self, namespace: str = "tokens", feature_name: str = "surface") -> None:
        self._namespace = namespace
        self._feature_name = feature_name

    def _get_token_feature(self, token: Token) -> str:
        feature = str(getattr(token, self._feature_name))
        return feature

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        def document_iterator() -> Iterator[List[str]]:
            for tokens in documents:
                yield [self._get_token_feature(token) for token in tokens]

        vocab.build_vocab_from_documents(self._namespace, document_iterator())

    def get_vocab_namespace(self) -> Optional[str]:
        return self._namespace

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return vocab.get_pad_index(self._namespace)

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        token_ids = [
            vocab.get_index_by_token(
                self._namespace,
                self._get_token_feature(token),
            )
            for token in tokens
        ]
        mask = [True] * len(token_ids)
        return {"token_ids": numpy.array(token_ids, dtype=int), "mask": numpy.array(mask, dtype=bool)}


class TokenCharactersIndexer(TokenIndexer):
    def __init__(
        self,
        namespace: str = "token_characters",
        feature_name: str = "surface",
        min_padding_length: int = 0,
    ) -> None:
        self._namespace = namespace
        self._feature_name = feature_name
        self._min_padding_length = min_padding_length

    def _get_token_feature(self, token: Token) -> str:
        feature = str(getattr(token, self._feature_name))
        return feature

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        def document_iterator() -> Iterator[List[str]]:
            for tokens in documents:
                for token in tokens:
                    yield list(self._get_token_feature(token))

        vocab.build_vocab_from_documents(self._namespace, document_iterator())

    def get_vocab_namespace(self) -> Optional[str]:
        return self._namespace

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return vocab.get_pad_index(self._namespace)

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        token_ids = [
            [vocab.get_index_by_token(self._namespace, character) for character in self._get_token_feature(token)]
            for token in tokens
        ]
        mask = [True] * len(tokens)
        max_token_length = max(self._min_padding_length, max(len(token_id) for token_id in token_ids))
        token_ids = [
            token_ids_ + [self.get_pad_index(vocab)] * (max_token_length - len(token_ids_)) for token_ids_ in token_ids
        ]
        subword_mask = [
            [True] * len(token_ids_) + [False] * (max_token_length - len(token_ids_)) for token_ids_ in token_ids
        ]
        return {
            "token_ids": numpy.array(token_ids, dtype=int),
            "mask": numpy.array(mask, dtype=bool),
            "subword_mask": numpy.array(subword_mask, dtype=bool),
        }


class TokenVectorIndexer(TokenIndexer):
    def __init__(self, namespace: Optional[str] = None) -> None:
        self._namespace = namespace

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        if self._namespace is not None:
            raise ValueError("Currently, TokenVectorIndexer does not support building vocabulary.")

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return 0

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        if not all(token.vector is not None for token in tokens):
            raise ValueError("TokenVectorIndexer requires all tokens to have vector.")
        return {
            "embeddings": numpy.array([token.vector for token in tokens], dtype=float),
            "mask": numpy.array([True] * len(tokens), dtype=bool),
        }


class PretrainedTransformerIndexer(TokenIndexer):
    def __init__(
        self,
        pretrained_model_name: str,
        namespace: Optional[str] = None,
        tokenize_subwords: bool = False,
    ) -> None:
        from transformers import AutoTokenizer

        self._pretrained_model_name = pretrained_model_name
        self._namespace = namespace
        self._tokenize_subwords = tokenize_subwords
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        if self._namespace is not None:
            raise ValueError("Currently, PretrainedTransformerIndexer does not support building vocabulary.")

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return int(self._tokenizer.pad_token_id)

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        if self._tokenize_subwords:
            return self._index_with_subword_tokenization(tokens, vocab)
        return self._index_without_subword_tokenization(tokens, vocab)

    def _index_without_subword_tokenization(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        indices: List[int] = []
        type_ids: List[int] = []
        mask: List[bool] = []
        for token in tokens:
            indices.append(self._tokenizer.convert_tokens_to_ids(token.surface))
            type_ids.append(0)
            mask.append(True)
        return {
            "token_ids": numpy.array(indices, dtype=int),
            "mask": numpy.array(mask, dtype=bool),
            "type_ids": numpy.array(type_ids, dtype=int),
        }

    def _index_with_subword_tokenization(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        indices: List[int] = []
        type_ids: List[int] = []
        subword_mask: List[bool] = []
        offsets: List[Tuple[int, int]] = []
        mask: List[bool] = []

        for token in tokens:
            subwords = self._tokenizer.encode_plus(
                token.surface,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            subword_ids = subwords["input_ids"]

            if subword_ids:
                offsets.append((len(indices), len(indices) + len(subword_ids) - 1))
                indices.extend(subword_ids)
                type_ids.extend([0] * len(subword_ids))
                subword_mask.extend([True] * len(subword_ids))
                mask.append(True)
            else:
                offsets.append((-1, -1))
                mask.append(True)

        return {
            "token_ids": numpy.array(indices, dtype=int),
            "mask": numpy.array(mask, dtype=bool),
            "type_ids": numpy.array(type_ids, dtype=int),
            "subword_mask": numpy.array(subword_mask, dtype=bool),
            "offsets": numpy.array(offsets, dtype=int),
        }
