from typing import Mapping, cast

import torch

from nlpstack.torch.modules.token_embedders import TokenEmbedder


class TextEmbedder(torch.nn.Module):
    def __init__(self, token_embedders: Mapping[str, TokenEmbedder]) -> None:
        super().__init__()
        self._token_embedders = torch.nn.ModuleDict(token_embedders)

    def forward(self, text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.FloatTensor:
        """
        :param text: Mapping[str, torch.nn.LongTensor]
        :return: torch.FloatTensor
        """
        return cast(
            torch.FloatTensor,
            torch.cat(
                [self._token_embedders[name](**inputs) for name, inputs in text.items()],
                dim=-1,
            ),
        )

    def get_output_dim(self) -> int:
        return sum(embedder.get_output_dim() for embedder in self._token_embedders.values())
