from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


class Vocabulary:
    def __init__(
        self,
        min_df: Mapping[str, int | float] = {},
        max_df: Mapping[str, int | float] = {},
        pad_token: Mapping[str, str] = {},
        oov_token: Mapping[str, str] = {},
        bos_token: Mapping[str, str] = {},
        eos_token: Mapping[str, str] = {},
        special_tokens: Mapping[str, set[str]] = {},
        ignored_tokens: Mapping[str, set[str]] = {},
    ) -> None:
        for namespace, token in pad_token.items():
            if token not in special_tokens[namespace]:
                raise ValueError(f"Pad token {token} not in special tokens of namespace {namespace}.")
        for namespace, token in oov_token.items():
            if token not in special_tokens[namespace]:
                raise ValueError(f"OOV token {token} not in special tokens of namespace {namespace}.")
        for namespace, token in bos_token.items():
            if token not in special_tokens[namespace]:
                raise ValueError(f"BOS token {token} not in special tokens of namespace {namespace}.")
        for namespace, token in eos_token.items():
            if token not in special_tokens[namespace]:
                raise ValueError(f"EOS token {token} not in special tokens of namespace {namespace}.")

        self._min_df = min_df
        self._max_df = max_df
        self._pad_token = pad_token
        self._oov_token = oov_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._special_tokens = special_tokens
        self._ignored_tokens = ignored_tokens
        self._token_to_index: dict[str, dict[str, int]] = {}
        self._index_to_token: dict[str, dict[int, str]] = {}

    def get_token_by_index(self, namespace: str, index: int) -> str:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        token = self._index_to_token[namespace].get(index)
        if token is not None:
            return token
        oov_token = self._oov_token.get(namespace)
        if oov_token in self._token_to_index[namespace]:
            return oov_token
        raise KeyError(f"Index {index} not found in namespace {namespace}.")

    def get_index_by_token(self, namespace: str, token: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        index = self._token_to_index[namespace].get(token)
        if index is not None:
            return index
        oov_token = self._oov_token.get(namespace)
        if oov_token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][oov_token]
        raise KeyError(f"Token {token} not found in namespace {namespace}.")

    def get_pad_index(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        pad_token = self._pad_token.get(namespace)
        if pad_token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][pad_token]
        raise KeyError(f"Pad token not found in namespace {namespace}.")

    def get_oov_index(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        oov_token = self._oov_token.get(namespace)
        if oov_token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][oov_token]
        raise KeyError(f"OOV token not found in namespace {namespace}.")

    def get_bos_index(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        bos_token = self._bos_token.get(namespace)
        if bos_token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][bos_token]
        raise KeyError(f"BOS token not found in namespace {namespace}.")

    def get_eos_index(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        eos_token = self._eos_token.get(namespace)
        if eos_token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][eos_token]
        raise KeyError(f"EOS token not found in namespace {namespace}.")

    def get_pad_token(self, namespace: str) -> str:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        pad_token = self._pad_token.get(namespace)
        if pad_token in self._token_to_index[namespace]:
            return pad_token
        raise KeyError(f"Pad token not found in namespace {namespace}.")

    def get_oov_token(self, namespace: str) -> str:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        oov_token = self._oov_token.get(namespace)
        if oov_token in self._token_to_index[namespace]:
            return oov_token
        raise KeyError(f"OOV token not found in namespace {namespace}.")

    def get_bos_token(self, namespace: str) -> str:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        bos_token = self._bos_token.get(namespace)
        if bos_token in self._token_to_index[namespace]:
            return bos_token
        raise KeyError(f"BOS token not found in namespace {namespace}.")

    def get_eos_token(self, namespace: str) -> str:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        eos_token = self._eos_token.get(namespace)
        if eos_token in self._token_to_index[namespace]:
            return eos_token
        raise KeyError(f"EOS token not found in namespace {namespace}.")

    def get_token_to_index(self, namespace: str) -> Mapping[str, int]:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._token_to_index[namespace]

    def get_index_to_token(self, namespace: str) -> Mapping[int, str]:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._index_to_token[namespace]

    def get_special_tokens(self, namespace: str) -> set[str]:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return {token for token in self._special_tokens.get(namespace, set())}

    def get_vocab_size(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return len(self._index_to_token[namespace])

    def has_namespace(self, namespace: str) -> bool:
        return namespace in self._index_to_token

    def clear(self, namespace: str) -> None:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        self._token_to_index[namespace].clear()
        self._index_to_token[namespace].clear()

    def build_vocab_from_documents(
        self,
        namespace: str,
        documents: Iterable[Sequence[str]],
    ) -> None:
        if namespace in self._token_to_index:
            raise ValueError(f"Namespace {namespace} already exists.")

        self._token_to_index[namespace] = {}
        self._index_to_token[namespace] = {}

        for token in self._special_tokens.get(namespace, set()):
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token

        num_documents = 0
        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in documents:
            num_documents += 1
            for token in set(tokens):
                document_frequency[token] += 1

        min_df = self._min_df.get(namespace, 0)
        max_df = self._max_df.get(namespace, 1.0)
        abs_min_df = min_df if isinstance(min_df, int) else int(min_df * num_documents)
        abs_max_df = max_df if isinstance(max_df, int) else int(max_df * num_documents)

        for token, df in document_frequency.items():
            if token in self._ignored_tokens.get(namespace, set()):
                continue
            if abs_min_df <= df <= abs_max_df:
                index = len(self._token_to_index[namespace])
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
