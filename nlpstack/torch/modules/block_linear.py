from typing import Sequence, Tuple

import torch
from torch.nn.functional import linear


class BlockLinear(torch.nn.Module):
    def __init__(
        self,
        block_dims: Sequence[Tuple[int, int]],
        bias: bool = False,
    ):
        super(BlockLinear, self).__init__()
        self._input_dim = sum(m for m, _ in block_dims)
        self._output_dim = sum(n for _, n in block_dims)
        self._blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(size, requires_grad=True)) for size in block_dims]
        )
        self._bias = torch.nn.Parameter(torch.empty(self.output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for block in self._blocks:
            torch.nn.init.xavier_uniform_(block)
        if self._bias is not None:
            torch.nn.init.zeros_(self._bias)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.block_diag(*self._blocks)  # type: ignore[no-untyped-call]
        out = linear(x, weight.T, self._bias)
        return out
