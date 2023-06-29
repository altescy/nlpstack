from typing import Callable, Literal, Optional, Sequence, cast

import torch

from nlpstack.torch.util import min_value_of_dtype


class Seq2VecEncoder(torch.nn.Module):
    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
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

    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
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

    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones_like(inputs[..., 0], dtype=torch.bool))

        lengths = mask.sum(dim=1).long()
        positions = torch.full_like(lengths, self._position) if self._position >= 0 else lengths + self._position
        return cast(torch.FloatTensor, inputs[torch.arange(inputs.size(0)), positions])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


class CnnEncoder(Seq2VecEncoder):
    def __init__(
        self,
        input_dim: int,
        num_filters: int,
        ngram_filter_sizes: Sequence[int] = (2, 3, 4, 5),
        conv_layer_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or torch.nn.ReLU()

        self._convolution_layers = [
            torch.nn.Conv1d(
                in_channels=self._input_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size,
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        self.projection_layer: Optional[torch.nn.Linear]
        self._output_dim: int
        if output_dim:
            self.projection_layer = torch.nn.Linear(maxpool_output_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            mask = cast(torch.BoolTensor, torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool())

        tokens = torch.transpose(tokens, 1, 2)

        filter_outputs = []
        batch_size = tokens.shape[0]
        last_unmasked_tokens = mask.sum(dim=1).unsqueeze(dim=-1)  # Shape: (batch_size, 1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = tokens.shape[2] - convolution_layer.kernel_size[0] + 1

            activations = self._activation(convolution_layer(tokens))

            indices = (
                torch.arange(pool_length, device=activations.device).unsqueeze(0).expand(batch_size, pool_length)
            )  # Shape: (batch_size, pool_length)
            activations_mask = indices.ge(
                last_unmasked_tokens - convolution_layer.kernel_size[0] + 1
            )  # Shape: (batch_size, pool_length)
            activations_mask = activations_mask.unsqueeze(1).expand_as(
                activations
            )  # Shape: (batch_size, num_filters, pool_length)

            activations = activations + (
                activations_mask * min_value_of_dtype(activations.dtype)
            )  # Shape: (batch_size, pool_length)

            # Pick out the max filters
            filter_outputs.append(activations.max(dim=2)[0])

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        maxpool_output[maxpool_output == min_value_of_dtype(maxpool_output.dtype)] = 0.0

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return cast(torch.FloatTensor, result)
