"""
Indexers for converting tokens into indices.
"""


from contextlib import suppress
from os import PathLike
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import minato
import numpy

from nlpstack.common import cached_property
from nlpstack.data.embeddings import WordEmbedding
from nlpstack.data.tokenizers import Token
from nlpstack.data.vocabulary import Vocabulary
from nlpstack.transformers import cache as transformers_cache

try:
    import fasttext
except ModuleNotFoundError:
    fasttext = None


try:
    import transformers
except ModuleNotFoundError:
    transformers = None


class TokenIndexer:
    """
    TokenIndexer is a class that converts tokens into indices.
    """

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        """
        Build vocabulary from documents.

        Args:
            vocab: Vocabulary to build.
            documents: Documents to build vocabulary from.
        """
        pass

    def get_vocab_namespace(self) -> Optional[str]:
        """
        Get vocabulary namespace.

        Returns:
            Vocabulary namespace.
        """
        return None

    def get_pad_index(self, vocab: Vocabulary) -> int:
        """
        Get padding index.

        Returns:
            Padding index.
        """
        raise NotImplementedError

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices.
        """
        raise NotImplementedError


class SingleIdTokenIndexer(TokenIndexer):
    """
    SingleIdTokenIndexer is a class that converts tokens into single indices.

    Args:
        namespace: Vocabulary namespace. Defaults to `"tokens"`.
        feature_name: The feature name of tokens to use. Defaults to `"surface"`.
        lowercase: Whether to lowercase tokens. Defaults to `False`.
    """

    def __init__(
        self,
        namespace: str = "tokens",
        feature_name: str = "surface",
        lowercase: bool = False,
    ) -> None:
        self._namespace = namespace
        self._feature_name = feature_name
        self._lowercase = lowercase

    def _get_token_feature(self, token: Token) -> str:
        feature = getattr(token, self._feature_name)
        if not isinstance(feature, str):
            raise ValueError(f"token.{self._feature_name} must be str, but got {type(feature)}")
        if self._lowercase:
            feature = feature.lower()
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
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices containing `"token_ids"` and `"mask"`.
        """

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
    """
    A TokenIndexer represents tokens as sequences of character indices.

    Args:
        namespace: Vocabulary namespace. Defaults to `"token_characters"`.
        feature_name: The feature name of tokens to use. Defaults to `"surface"`.
        lowercase: Whether to lowercase tokens. Defaults to `False`.
        min_padding_length: Minimum padding length. Defaults to `0`.
    """

    def __init__(
        self,
        namespace: str = "token_characters",
        feature_name: str = "surface",
        lowercase: bool = False,
        min_padding_length: int = 0,
    ) -> None:
        self._namespace = namespace
        self._feature_name = feature_name
        self._lowercase = lowercase
        self._min_padding_length = min_padding_length

    def _get_token_feature(self, token: Token) -> str:
        feature = getattr(token, self._feature_name)
        if not isinstance(feature, str):
            raise ValueError(f"token.{self._feature_name} must be str, but got {type(feature)}")
        if self._lowercase:
            feature = feature.lower()
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
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices containing `"token_ids"`, `"mask"`, and `"subword_mask"`.
        """

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
    """
    A TokenIndexer represents tokens as vectors. This indexer does not support building vectors and vocabulary.
    You need use tokenizers that support vectorization, such as `SpacyTokenizer`.

    Args:
        namespace: Vocabulary namespace. Defaults to `None`.
    """

    def __init__(self, namespace: Optional[str] = None) -> None:
        self._namespace = namespace

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        if self._namespace is not None:
            raise ValueError("Currently, TokenVectorIndexer does not support building vocabulary.")

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return 0

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices containing `"embeddings"` and `"mask"`.
        """

        if not all(token.vector is not None for token in tokens):
            raise ValueError("TokenVectorIndexer requires all tokens to have vector.")
        return {
            "embeddings": numpy.array([token.vector for token in tokens], dtype=float),
            "mask": numpy.array([True] * len(tokens), dtype=bool),
        }


class PretrainedEmbeddingIndexer(TokenIndexer):
    """
    A TokenIndexer represents tokens as pretrained embeddings.

    Args:
        embedding: Pretrained embedding to use.
        feature_name: The feature name of tokens to use. Defaults to `"surface"`.
        lowercase: Whether to lowercase tokens. Defaults to `False`.
        namespace: Vocabulary namespace. Defaults to `None`.
    """

    def __init__(
        self,
        embedding: WordEmbedding,
        feature_name: str = "surface",
        lowercase: bool = False,
        namespace: Optional[str] = None,
    ) -> None:
        self._embedding = embedding
        self._feature_name = feature_name
        self._lowercase = lowercase
        self._namespace = namespace

    def _get_token_feature(self, token: Token) -> str:
        feature = getattr(token, self._feature_name)
        if not isinstance(feature, str):
            raise ValueError(f"token.{self._feature_name} must be str, but got {type(feature)}")
        if self._lowercase:
            feature = feature.lower()
        return feature

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        if self._namespace is not None:
            raise ValueError("Currently, TokenVectorIndexer does not support building vocabulary.")

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return 0

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices containing `"embeddings"` and `"mask"`.
        """

        return {
            "embeddings": numpy.array([self._embedding[self._get_token_feature(token)] for token in tokens]),
            "mask": numpy.array([True] * len(tokens), dtype=bool),
        }


class PretrainedTransformerIndexer(TokenIndexer):
    """
    A TokenIndexer represents tokens as indices with pretrained transformer tokenizer.

    Args:
        pretrained_model_name: Pretrained model name or path.
        namespace: Vocabulary namespace. Defaults to `None`.
        tokenize_subwords: Whether to tokenize subwords. Defaults to `False`.
        add_special_tokens: Whether to add special tokens. Defaults to `False`.
    """

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        namespace: str = "tokens",
        tokenize_subwords: bool = False,
        add_special_tokens: bool = False,
    ) -> None:
        if tokenize_subwords and add_special_tokens:
            raise ValueError("Currently, tokenize_subwords and add_special_tokens cannot be True at the same time.")

        self._namespace = namespace
        self._tokenize_subwords = tokenize_subwords
        self._add_special_tokens = add_special_tokens
        self._pretrained_model_name = pretrained_model_name

    @cached_property
    def tokenizer(self) -> "transformers.PreTrainedTokenizer":
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed.")
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    def build_vocab(self, vocab: Vocabulary, documents: Iterable[Sequence[Token]]) -> None:
        try:
            token_to_index = self.tokenizer.get_vocab()
        except NotImplementedError:
            token_to_index = {self.tokenizer.convert_ids_to_tokens(i): i for i in range(self.tokenizer.vocab_size)}

        def document_iterator() -> Iterator[Sequence[str]]:
            for tokens in documents:
                if self._tokenize_subwords:
                    surfaces = [subword for token in tokens for subword in self.tokenizer.tokenize(token.surface)]
                else:
                    surfaces = [token.surface for token in tokens]
                yield surfaces

        vocab.build_vocab_from_documents(
            self._namespace,
            document_iterator(),
            token_to_index=token_to_index,
        )

    def get_pad_index(self, vocab: Vocabulary) -> int:
        return vocab.get_pad_index(self._namespace)

    def get_vocab_namespace(self) -> str:
        return self._namespace

    def __call__(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        """
        Convert tokens into indices.

        Args:
            tokens: Tokens to convert.
            vocab: Vocabulary to use.

        Returns:
            Dictionary of indices containing `"token_ids"`, `"type_ids"`, and `"mask"`. `"subword_mask"` and
            `"offsets"` are also included if `tokenize_subwords` is `True`.
        """
        if self._tokenize_subwords:
            return self._index_with_subword_tokenization(tokens, vocab)
        return self._index_without_subword_tokenization(tokens, vocab)

    def _index_without_subword_tokenization(self, tokens: Sequence[Token], vocab: Vocabulary) -> Dict[str, Any]:
        indices: List[int] = []
        type_ids: List[int] = []
        mask: List[bool] = []

        tokenized = self.tokenizer.prepare_for_model(
            self.tokenizer.convert_tokens_to_ids([token.surface for token in tokens]),
            add_special_tokens=self._add_special_tokens,
        )

        indices = tokenized["input_ids"]
        type_ids = tokenized["token_type_ids"] if "token_type_ids" in tokenized else [0] * len(indices)
        mask = tokenized["attention_mask"]

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
            subwords = self.tokenizer.encode_plus(
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
