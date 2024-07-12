from typing import Callable, List, Literal, Optional, Sequence, Union, cast

import torch

from nlpstack.integrations.torch.modules.feedforward import FeedForward
from nlpstack.integrations.torch.modules.time_distributed import TimeDistributed
from nlpstack.integrations.torch.util import (
    combine_tensors,
    masked_mean,
    masked_pool,
    masked_softmax,
    min_value_of_dtype,
)


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
        pooling: Literal["mean", "max", "sum", "hier"] = "mean",
        window_size: Optional[int] = None,
    ) -> None:
        if pooling not in ("mean", "max", "sum", "hier"):
            raise ValueError(f"pooling must be one of 'mean', 'max', 'sum', or 'hier', got {pooling}")
        if pooling == "hier" and window_size is None:
            raise ValueError("window_size must be specified when pooling is 'hier'")

        super().__init__()
        self._input_dim = input_dim
        self._pooling = pooling
        self._window_size = window_size

    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        if mask is None:
            mask = cast(torch.BoolTensor, inputs.new_ones(inputs.size()[:-1], dtype=torch.bool))
        mask = cast(torch.BoolTensor, mask.unsqueeze(-1))
        return masked_pool(inputs, mask, self._pooling, window_size=self._window_size)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


class TokenPooler(Seq2VecEncoder):
    def __init__(
        self,
        input_dim: int,
        positions: Union[int, Sequence[int]] = 0,
        output_dim: Optional[int] = None,
    ) -> None:
        if isinstance(positions, int):
            positions = [positions]

        super().__init__()
        self._input_dim = input_dim
        self._positions = positions
        self._output_dim = output_dim
        self._projection: Optional[torch.nn.Linear] = None
        if output_dim is not None:
            self._projection = torch.nn.Linear(len(positions) * input_dim, output_dim)

    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones_like(inputs[..., 0], dtype=torch.bool))

        lengths = mask.sum(dim=1).long()
        embeddings: List[torch.Tensor] = []
        for position in self._positions:
            positions = torch.full_like(lengths, position) if position >= 0 else lengths + position
            embeddings.append(inputs[torch.arange(inputs.size(0)), positions])

        output = cast(torch.FloatTensor, torch.cat(embeddings, dim=-1))
        if self._projection is not None:
            output = self._projection(output)

        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._output_dim is not None:
            return self._output_dim
        return len(self._positions) * self._input_dim


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
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)
        else:
            mask = cast(torch.BoolTensor, torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device).bool())

        inputs = torch.transpose(inputs, 1, 2)

        filter_outputs = []
        batch_size = inputs.shape[0]
        last_unmasked_inputs = mask.sum(dim=1).unsqueeze(dim=-1)  # Shape: (batch_size, 1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = inputs.shape[2] - convolution_layer.kernel_size[0] + 1

            activations = self._activation(convolution_layer(inputs))

            indices = (
                torch.arange(pool_length, device=activations.device).unsqueeze(0).expand(batch_size, pool_length)
            )  # Shape: (batch_size, pool_length)
            activations_mask = indices.ge(
                last_unmasked_inputs - convolution_layer.kernel_size[0] + 1
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


class ConcatSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        encoders: Sequence[Seq2VecEncoder],
        combination: Optional[str] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        if not len({encoder.get_input_dim() for encoder in encoders}) == 1:
            raise ValueError("All encoders must have the same input dimension")

        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)
        self._combination = combination
        self._output_dim = output_dim
        self._projection: Optional[torch.nn.Linear] = None
        if output_dim is not None:
            self._projection = torch.nn.Linear(
                sum(encoder.get_output_dim() for encoder in encoders),
                output_dim,
            )

    def get_input_dim(self) -> int:
        return cast(int, self._encoders[0].get_input_dim())

    def get_output_dim(self) -> int:
        if self._output_dim is not None:
            return self._output_dim
        return sum(encoder.get_output_dim() for encoder in self._encoders)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        tensors: List[torch.Tensor] = [encoder(inputs, mask) for encoder in self._encoders]
        if self._combination is not None:
            output = combine_tensors(self._combination, tensors)
        else:
            output = torch.cat(tensors, dim=-1)

        if self._projection is not None:
            output = self._projection(output)

        return cast(torch.FloatTensor, output)


class SelfAttentiveSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        input_dim: int,
        feedforward: Optional[FeedForward] = None,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim

        if feedforward is None:
            scorer: torch.nn.Module = torch.nn.Linear(input_dim, 1)
        else:
            scorer = torch.nn.Sequential(
                feedforward,
                torch.nn.Linear(feedforward.get_output_dim(), 1),  # type: ignore
            )
        self._global_attention = TimeDistributed(scorer)  # type: ignore

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, max_length, _embedding_dim = inputs.size()

        if mask is None:
            mask = cast(torch.BoolTensor, inputs.new_ones((batch_size, max_length), dtype=torch.bool))

        # Shape: (batch_size, max_length, 1)
        mask = cast(torch.BoolTensor, mask.unsqueeze(-1))

        # Shape: (batch_size, max_length, 1)
        global_attention_logits = self._global_attention(inputs)
        # Shape: (batch_size, max_length, 1)
        global_attention_weights = masked_softmax(global_attention_logits, mask, dim=1)

        # Shape: (batch_size, max_length, embedding_dim)
        weighted_embeddings = global_attention_weights * inputs

        # Shape: (batch_size, embedding_dim)
        output = masked_mean(weighted_embeddings, mask, dim=1)

        return cast(torch.FloatTensor, output)
