from collections import Counter, defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Union


class Vocabulary:
    def __init__(
        self,
        min_df: Mapping[str, Union[int, float]] = {},
        max_df: Mapping[str, Union[int, float]] = {},
        min_count: Mapping[str, int] = {},
        max_count: Mapping[str, int] = {},
        pad_token: Mapping[str, str] = {},
        oov_token: Mapping[str, str] = {},
        bos_token: Mapping[str, str] = {},
        eos_token: Mapping[str, str] = {},
        special_tokens: Mapping[str, Set[str]] = {},
        ignored_tokens: Mapping[str, Set[str]] = {},
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
        self._min_count = min_count
        self._max_count = max_count
        self._pad_token = pad_token
        self._oov_token = oov_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._special_tokens = special_tokens
        self._ignored_tokens = ignored_tokens
        self._token_to_index: Dict[str, Dict[str, int]] = {}
        self._index_to_token: Dict[str, Dict[int, str]] = {}
        self._token_to_count: Dict[str, Dict[str, int]] = {}
        self._num_documents: Dict[str, int] = {}

    def __getitem__(self, namespace: str) -> Mapping[str, int]:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._token_to_index[namespace]

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

    def get_special_tokens(self, namespace: str) -> Set[str]:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return {token for token in self._special_tokens.get(namespace, set())}

    def get_token_count(self, namespace: str, token: str) -> int:
        if namespace not in self._token_to_count:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._token_to_count[namespace].get(token, 0)

    def get_total_tokens(self, namespace: str) -> int:
        if namespace not in self._token_to_count:
            raise KeyError(f"Namespace {namespace} not found.")
        return sum(self._token_to_count[namespace].values())

    def get_token_to_count(self, namespace: str) -> Mapping[str, int]:
        if namespace not in self._token_to_count:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._token_to_count[namespace]

    def get_vocab_size(self, namespace: str) -> int:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return len(self._index_to_token[namespace])

    def get_num_documents(self, namespace: str) -> int:
        if namespace not in self._num_documents:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._num_documents[namespace]

    def has_namespace(self, namespace: str) -> bool:
        return namespace in self._index_to_token

    def has_token(self, namespace: str, token: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return token in self._token_to_index[namespace]

    def has_pad_token(self, namespace: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._pad_token.get(namespace) in self._token_to_index[namespace]

    def has_oov_token(self, namespace: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._oov_token.get(namespace) in self._token_to_index[namespace]

    def has_bos_token(self, namespace: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._bos_token.get(namespace) in self._token_to_index[namespace]

    def has_eos_token(self, namespace: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return self._eos_token.get(namespace) in self._token_to_index[namespace]

    def extend_vocab(self, namespace: str, tokens: Iterable[str]) -> None:
        """
        Add extra tokens to the vocabulary namespace.
        This method does not update token counts, just adds new tokens to the vocabulary.
        """
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        new_tokens = (
            set(tokens) - set(self._token_to_index[namespace].keys()) - set(self._ignored_tokens.get(namespace, set()))
        )
        for token in sorted(new_tokens):
            index = len(self._index_to_token[namespace])
            self._index_to_token[namespace][index] = token
            self._token_to_index[namespace][token] = index

    def is_extended_token(self, namespace: str, token: str) -> bool:
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace {namespace} not found.")
        return (
            token in self._token_to_index[namespace]
            and token not in self._special_tokens
            and token not in self._token_to_count[namespace]
        )

    def is_extended_index(self, namespace: str, index: int) -> bool:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        return index in self._index_to_token[namespace] and self.is_extended_token(
            namespace, self._index_to_token[namespace][index]
        )

    def clear(self, namespace: str) -> None:
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace {namespace} not found.")
        self._token_to_index[namespace].clear()
        self._index_to_token[namespace].clear()

    def build_vocab_from_documents(
        self,
        namespace: str,
        documents: Iterable[Sequence[str]],
        token_to_index: Optional[Mapping[str, int]] = None,
    ) -> None:
        if namespace in self._token_to_index:
            raise ValueError(f"Namespace {namespace} already exists.")

        def get_token_index(token: str) -> int:
            if token_to_index is None:
                return len(self._index_to_token[namespace])
            return token_to_index[token]

        self._token_to_index[namespace] = {}
        self._index_to_token[namespace] = {}
        self._token_to_count[namespace] = {}

        for token in sorted(self._special_tokens.get(namespace, set())):
            index = get_token_index(token)
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token

        num_documents = 0
        token_count: Dict[str, int] = defaultdict(int)
        document_frequency: Dict[str, int] = defaultdict(int)
        for tokens in documents:
            num_documents += 1
            for token, count in Counter(tokens).items():
                token_count[token] += count
                document_frequency[token] += 1

        min_df = self._min_df.get(namespace, 0)
        max_df = self._max_df.get(namespace, 1.0)
        abs_min_df = min_df if isinstance(min_df, int) else int(min_df * num_documents)
        abs_max_df = max_df if isinstance(max_df, int) else int(max_df * num_documents)

        min_count = self._min_count.get(namespace, 0)
        max_count = self._max_count.get(namespace, float("inf"))

        ignored_tokens = self._ignored_tokens.get(namespace, set())

        self._num_documents[namespace] = num_documents

        for token in sorted(token_count):
            if token in ignored_tokens:
                continue
            count = token_count[token]
            df = document_frequency[token]
            if (abs_min_df <= df <= abs_max_df) and (min_count <= count <= max_count):
                index = get_token_index(token)
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
                self._token_to_count[namespace][token] = count

        if token_to_index is not None:
            extra_tokens = set(token_to_index) - set(self._token_to_index)
            for token in extra_tokens:
                index = get_token_index(token)
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
