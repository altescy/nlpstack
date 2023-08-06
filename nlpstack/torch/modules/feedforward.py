from typing import Callable, Sequence

import torch


class FeedForward(torch.nn.Module):
    """
    A simple feed forward neural network.

    Args:
        input_dim: The dimension of the input.
        hidden_dims: A sequence of integers specifying the dimensions of each layer.
        dropout: The dropout probability. Defaults to `0.0`.
        activation: The activation function. Defaults to `torch.nn.ReLU()`
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        activation: Callable[[torch.FloatTensor], torch.FloatTensor] = torch.nn.ReLU(),
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._dropout = dropout
        self._activation = activation

        layer_dims = [input_dim] + list(hidden_dims)
        self._layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(  # type: ignore[call-overload]
                    torch.nn.Linear(layer_dims[i], layer_dims[i + 1]),
                    torch.nn.Dropout(dropout),
                    activation,
                )
                for i in range(len(layer_dims) - 1)
            ]
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, ..., input_dim)`.

        Returns:
            A tensor of shape `(batch_size, ..., hidden_dims[-1])`.
        """
        output = inputs
        for layer in self._layers:
            output = layer(output)
        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dims[-1]
