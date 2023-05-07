from __future__ import annotations

from typing import Literal, cast

import torch


class Seq2VecEncoder(torch.nn.Module):
    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor | None = None) -> torch.FloatTensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class BagOfEmbeddings(Seq2VecEncoder):
    def __init__(
        self,
        input_dim: int,
        pooling: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._pooling = pooling

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor | None = None) -> torch.FloatTensor:
        """
        :param inputs: (batch_size, seq_len, embedding_dim)
        :param mask: (batch_size, seq_len)
        :return: (batch_size, embedding_dim)
        """
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones_like(inputs[..., 0], dtype=torch.bool))

        if self._pooling == "mean":
            return cast(
                torch.FloatTensor,
                torch.sum(inputs * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1).unsqueeze(-1),
            )
        elif self._pooling == "max":
            return cast(
                torch.FloatTensor,
                inputs.masked_fill_(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values,
            )
        raise ValueError(f"Unknown pooling: {self._pooling}")

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


class TokenPooler(Seq2VecEncoder):
    def __init__(self, input_dim: int, position: int = 0) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._position = position

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor | None = None) -> torch.FloatTensor:
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones_like(inputs[..., 0], dtype=torch.bool))

        lengths = mask.sum(dim=1).long()
        positions = torch.full_like(lengths, self._position) if self._position >= 0 else lengths + self._position
        return cast(torch.FloatTensor, inputs[torch.arange(inputs.size(0)), positions])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim
