from typing import Literal, cast

import torch


class Seq2VecEncoder(torch.nn.Module):
    def forward(self, embeddings: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class BagOfEmbeddings(Seq2VecEncoder):
    def __init__(
        self,
        embedding_dim: int,
        pooling: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._pooling = pooling

    def forward(self, embeddings: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        """
        :param embeddings: (batch_size, seq_len, embedding_dim)
        :param mask: (batch_size, seq_len)
        :return: (batch_size, embedding_dim)
        """
        if self._pooling == "mean":
            return cast(
                torch.FloatTensor,
                torch.sum(embeddings * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1).unsqueeze(-1),
            )
        elif self._pooling == "max":
            return cast(
                torch.FloatTensor,
                embeddings.masked_fill_(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values,
            )
        raise ValueError(f"Unknown pooling: {self._pooling}")

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim
