from __future__ import annotations

from typing import Any, cast

import torch

from nlpstack.data import Vocabulary


class TokenEmbedder(torch.nn.Module):
    def get_output_dim(self) -> int:
        raise NotImplementedError


class PassThroughTokenEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    def forward(self, embeddings: torch.FloatTensor, **kwargs: Any) -> torch.FloatTensor:
        return embeddings

    def get_output_dim(self) -> int:
        return self._embedding_dim


class LazyEmbedding(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Embedding):
    cls_to_become = torch.nn.Embedding  # type: ignore[assignment]
    weight: torch.nn.UninitializedParameter

    def __init__(
        self,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings=0,
            embedding_dim=embedding_dim,
            padding_idx=None,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.weight = torch.nn.UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_embeddings != 0:
            super().reset_parameters()

    def initialize_parameters(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.has_uninitialized_params():
            num_embeddings = kwargs.get("num_embeddings", 0)
            padding_idx = kwargs.get("padding_idx", None)
            with torch.no_grad():
                self.num_embeddings = num_embeddings
                self.padding_idx = padding_idx
                self.weight.materialize((self.num_embeddings, self.embedding_dim))
                self.reset_parameters()


class Embedding(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self._embedding = LazyEmbedding(
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self._namespace = namespace

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        self._embedding.initialize_parameters(
            num_embeddings=vocab.get_vocab_size(self._namespace),
            padding_idx=vocab.get_pad_index(self._namespace),
        )

    def forward(self, token_ids: torch.LongTensor, **kwargs: Any) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._embedding(token_ids))

    def get_output_dim(self) -> int:
        return self._embedding.embedding_dim
