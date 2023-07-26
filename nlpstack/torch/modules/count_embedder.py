from typing import Any, Mapping, Optional, cast

import torch

from nlpstack.data import Vocabulary
from nlpstack.torch.util import get_mask_from_text


class CountEmbedder(torch.nn.Module):
    def __init__(self, ignore_oov: bool = False, token_namespace: str = "tokens") -> None:
        super().__init__()
        self._ignore_oov = ignore_oov
        self._token_namespace = token_namespace
        self._vocab_size: Optional[int] = None
        self._oov_index: Optional[int] = None

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        self._vocab_size = vocab.get_vocab_size(self._token_namespace)
        if self._ignore_oov:
            self._oov_index = vocab.get_oov_index(self._token_namespace)

    def _get_token_ids(self, text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
        for text_field in text.values():
            if "token_ids" not in text_field:
                continue
            return text_field["token_ids"]
        raise ValueError("No token_ids found in text")

    def forward(self, text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
        if self._vocab_size is None:
            raise ValueError("vocab_size must be set before calling forward")

        mask = get_mask_from_text(text)
        token_ids = self._get_token_ids(text)

        bag_of_words_vectors = []
        if self._ignore_oov:
            if self._oov_index is None:
                raise ValueError("oov_idx must be set before calling forward")
            mask = cast(torch.BoolTensor, (mask & token_ids != self._oov_index))
        for document, doc_mask in zip(token_ids, mask):
            document = torch.masked_select(document, doc_mask)
            vec = torch.bincount(document, minlength=self._vocab_size).float()
            vec = vec.view(1, -1)
            bag_of_words_vectors.append(vec)
        bag_of_words_output = torch.cat(bag_of_words_vectors, 0)
        return bag_of_words_output

    def get_output_dim(self) -> int:
        if self._vocab_size is None:
            raise ValueError("vocab_size must be set before calling get_output_dim")
        return self._vocab_size
