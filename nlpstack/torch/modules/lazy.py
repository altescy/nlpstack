from __future__ import annotations

from typing import Any

import torch


class LazyLinearOutput(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Linear):
    cls_to_become = torch.nn.Linear  # type: ignore[assignment]
    weight: torch.nn.UninitializedParameter
    bias: torch.nn.UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self,
        in_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=in_features, out_features=0, bias=bias)
        self.weight = torch.nn.UninitializedParameter()
        if bias:
            self.bias = torch.nn.UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.out_features != 0:
            super().reset_parameters()

    def initialize_parameters(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.has_uninitialized_params():
            num_embeddings = kwargs.get("out_features", 0)
            with torch.no_grad():
                self.out_features = num_embeddings
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


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
