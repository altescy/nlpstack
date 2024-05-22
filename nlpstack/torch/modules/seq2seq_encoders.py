import math
from contextlib import suppress
from os import PathLike
from typing import Callable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast

import minato
import torch
import torch.nn.functional as F

from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.xlstm import XLSTM
from nlpstack.torch.util import add_positional_features, convert_to_toeplitz, fold, unfold


class Seq2SeqEncoder(torch.nn.Module):
    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class PassThroughSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        return inputs


class FeedForwardSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__()
        self._feedforward = feedforward

    def get_input_dim(self) -> int:
        return self._feedforward.get_input_dim()

    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        outputs = self._feedforward(inputs)
        return cast(torch.FloatTensor, outputs * mask.unsqueeze(dim=-1))


class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

        with suppress(AttributeError):
            if not self._module.batch_first:
                raise ValueError("PytorchSeq2SeqWrapper only supports batch_first=True")

        try:
            self._is_bidirectional = bool(self._module.bidirectional)
        except AttributeError:
            self._is_bidirectional = False

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        batch_size, seq_len, _ = inputs.size()
        sorted_seq_len = mask.sum(dim=1).detach().cpu()
        sorted_seq_len, sorted_idx = sorted_seq_len.sort(descending=True)
        _, unsorted_idx = sorted_idx.sort()
        sorted_inputs = inputs[sorted_idx]
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_seq_len, batch_first=True)
        packed_output, _ = self._module(packed_inputs)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[unsorted_idx]
        return cast(torch.FloatTensor, output)

    def get_input_dim(self) -> int:
        return cast(int, self._module.input_size)

    def get_output_dim(self) -> int:
        hidden_size = cast(int, self._module.hidden_size)
        return hidden_size * 2 if self._is_bidirectional else hidden_size


class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            torch.nn.LSTM(  # type: ignore[no-untyped-call]
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        )


class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            torch.nn.GRU(  # type: ignore[no-untyped-call]
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        )


class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__(
            torch.nn.RNN(  # type: ignore[no-untyped-call]
                input_dim,
                hidden_dim,
                num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
        )


class TransformerSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        num_attention_heads: int = 8,
        positional_encoding: Optional[Literal["sinusoidal", "embedding"]] = "sinusoidal",
        positional_embedding_size: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: Union[Literal["relu", "gelu"], Callable[[torch.Tensor], torch.Tensor]] = "relu",
        norm_first: bool = False,
    ) -> None:
        super().__init__()

        layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
        )
        self._transformer = torch.nn.TransformerEncoder(layer, num_layers)  # type: ignore[no-untyped-call]
        self._input_dim = input_dim

        # initialize parameters
        # We do this before the embeddings are initialized so we get the default initialization for the embeddings.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if positional_encoding is None:
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding = True
            self._positional_embedding = None
        elif positional_encoding == "embedding":
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = torch.nn.Embedding(positional_embedding_size, input_dim)
        else:
            raise ValueError("positional_encoding must be one of None, 'sinusoidal', or 'embedding'")

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        output = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        # For some reason the torch transformer expects the shape (sequence, batch, features), not the more
        # familiar (batch, sequence, features), so we have to fix it.
        output = cast(torch.FloatTensor, output.permute(1, 0, 2))
        # For some other reason, the torch transformer takes the mask backwards.
        mask = cast(torch.BoolTensor, ~mask)
        output = self._transformer(output, src_key_padding_mask=mask)
        output = cast(torch.FloatTensor, output.permute(1, 0, 2))

        return output


class ComposeSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, encoders: Sequence[Seq2SeqEncoder]) -> None:
        if not all(x.get_output_dim() == y.get_input_dim() for x, y in zip(encoders, encoders[1:])):
            dimcheck = ", ".join(f"({encoder.get_input_dim()} -> {encoder.get_output_dim()})" for encoder in encoders)
            raise ValueError(f"Encoders must have matching input and output dimensions, but found: {dimcheck}")

        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)

    def get_input_dim(self) -> int:
        return cast(int, self._encoders[0].get_input_dim())

    def get_output_dim(self) -> int:
        return cast(int, self._encoders[-1].get_output_dim())

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        output = inputs
        for encoder in self._encoders:
            output = encoder(output, mask)
        return output


class WindowConcatEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        window_size: Union[int, Tuple[int, int]],
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if not all(s >= 0 for s in window_size):
            raise ValueError("Window size must be greater than or equal to zero.")
        self._input_dim = input_dim
        self._window_size = window_size
        self._projection: Optional[torch.nn.Linear] = None
        if output_dim is not None:
            self._projection = torch.nn.Linear(
                (sum(window_size) + 1) * input_dim,
                output_dim,
            )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._projection is not None:
            return self._projection.out_features
        return (sum(self._window_size) + 1) * self._input_dim

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        batch_size, max_length, embedding_dim = inputs.size()
        inputs = cast(torch.FloatTensor, inputs * mask.float().unsqueeze(2))

        output = inputs
        lws, rws = self._window_size
        if lws > 0:
            pad = inputs.new_zeros((batch_size, lws, embedding_dim))
            x = torch.cat([pad, inputs], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(lws)], dim=2)
            output = cast(torch.FloatTensor, torch.cat([output, x], dim=2))
        if rws > 0:
            pad = inputs.new_zeros((batch_size, rws, embedding_dim))
            x = torch.cat([inputs, pad], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(1, rws + 1)], dim=2)
            output = cast(torch.FloatTensor, torch.cat([output, x], dim=2))

        if self._projection is not None:
            output = self._projection(output)

        return cast(torch.FloatTensor, output * mask.float().unsqueeze(2))


class ResidualSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        projection: bool = True,
    ) -> None:
        super().__init__()

        self._encoder = encoder
        self._projection: Optional[torch.nn.Module] = None
        if projection:
            self._projection = torch.nn.Linear(
                encoder.get_output_dim(),
                encoder.get_input_dim(),
            )
        else:
            if self._encoder.get_input_dim() != self._encoder.get_output_dim():
                raise ValueError(
                    "If not projecting, input and output dimensions must match, "
                    f"but found {self._encoder.get_input_dim()} and {self._encoder.get_output_dim()}."
                )

    def get_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.get_input_dim()

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # Shape: (batch_size, max_length, embedding_size)
        encodings = self._encoder(inputs, mask)
        if self._projection is not None:
            encodings = self._projection(encodings)

        return cast(torch.FloatTensor, inputs + encodings)


class MLPMixer(Seq2SeqEncoder):
    class SpatialLinear(torch.nn.Module):
        def __init__(
            self,
            spatial_dim: int,
            toeplitz: bool = False,
        ) -> None:
            super().__init__()
            self._toeplitz = toeplitz
            if self._toeplitz:
                weights = torch.randn(2 * spatial_dim - 1)
            else:
                weights = torch.randn(spatial_dim, spatial_dim)
            self._weights = torch.nn.Parameter(weights)
            self._biases = torch.nn.Parameter(torch.randn(spatial_dim))

        def _get_weights_and_biases(
            self,
            max_length: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self._toeplitz:
                weights = convert_to_toeplitz(self._weights)
            else:
                weights = self._weights
            weights = weights[:max_length, :max_length]
            biases = self._biases[:max_length]
            return weights, biases

        def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            batch_size, max_length, embedding_dim = inputs.size()

            # Shape: (batch_size * embedding_dim, max_length)
            inputs = inputs.transpose(1, 2).reshape(-1, max_length)
            if mask is not None:
                # Shape: (batch_size * embedding_dim, max_length)
                mask = mask.repeat_interleave(embedding_dim, dim=0)
                inputs *= mask

            # Shape: (max_length, max_length)
            # Shape: (max_length)
            weights, biases = self._get_weights_and_biases(max_length)

            # Shape: (batch_size * embedding_dim, max_length)
            output = inputs @ weights + biases
            # Shape: (batch_size, max_length, embedding_dim)
            output = output.reshape(batch_size, embedding_dim, max_length).transpose(1, 2)

            return output

    class MixerLayer(torch.nn.Module):
        def __init__(
            self,
            channel_dim: int,
            spatial_dim: int,
            toeplitz: bool = False,
        ) -> None:
            super().__init__()
            self._layer_norm_1 = torch.nn.LayerNorm(channel_dim)
            self._layer_norm_2 = torch.nn.LayerNorm(channel_dim)
            self._spatial_linear_1 = MLPMixer.SpatialLinear(spatial_dim, toeplitz)
            self._spatial_linear_2 = MLPMixer.SpatialLinear(spatial_dim, toeplitz)
            self._channel_linear_1 = torch.nn.Linear(channel_dim, channel_dim)
            self._channel_linear_2 = torch.nn.Linear(channel_dim, channel_dim)

        def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            res1 = inputs
            h = self._layer_norm_1(inputs)

            h = self._spatial_linear_1(h, mask)
            h = F.gelu(h)
            h = self._spatial_linear_2(h, mask)

            h = h + res1
            res2 = h

            h = self._layer_norm_2(h)

            h = self._channel_linear_1(h)
            h = F.gelu(h)
            h = self._channel_linear_2(h)

            h = h + res2
            return cast(torch.Tensor, h)

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        toeplitz: bool = False,
        max_length: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._max_length = max_length

        self._layers = torch.nn.ModuleList(
            [MLPMixer.MixerLayer(input_dim, max_length, toeplitz) for _ in range(num_layers)]
        )
        self._dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout) for _ in range(num_layers)])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self, inputs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        sequence_length = inputs.size(1)
        inputs = fold(inputs, self._max_length)
        if mask is not None:
            mask = fold(mask, self._max_length)
        h = inputs
        m = mask.float() if mask is not None else None
        for layer, dropout in zip(self._layers, self._dropouts):
            h = dropout(layer(h, m))
        output = unfold(h, sequence_length)
        return output


class HyperMixer(Seq2SeqEncoder):
    class TokenMixer(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self._feedforward = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, inputs: torch.FloatTensor, mask: torch.Tensor) -> torch.FloatTensor:
            inputs = cast(torch.FloatTensor, inputs * mask.unsqueeze(-1))

            # Shape: (batch_size, max_length, hidden_dim)
            W1 = self._compute_weights(inputs)
            # Shape: (batch_size, hidden_dim, max_length)
            W2 = W1.transpose(1, 2)

            # Shape: (batch_size, hidden_dim, embedding_dim)
            output = torch.bmm(W2, inputs)
            output = F.gelu(output)
            # Shape: (batch_size, max_length, embedding_dim)
            output = torch.bmm(W1, output)

            return cast(torch.FloatTensor, output)

        def _compute_weights(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
            output = add_positional_features(inputs)
            output = self._feedforward(output)
            return output

    class HyperMixerLayer(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self._layer_norm_1 = torch.nn.LayerNorm(input_dim)
            self._layer_norm_2 = torch.nn.LayerNorm(input_dim)
            self._token_mixer = HyperMixer.TokenMixer(input_dim, hidden_dim)
            self._feedforward = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, input_dim),
            )

        def forward(
            self,
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            res1 = inputs
            output = self._layer_norm_1(inputs)

            output = self._token_mixer(output, mask)
            output = output + res1

            res2 = output
            output = self._layer_norm_2(output)

            output = self._feedforward(output)
            output = output + res2

            return cast(torch.Tensor, output)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [HyperMixer.HyperMixerLayer(input_dim, hidden_dim) for _ in range(num_layers)]
        )
        self._dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout) for _ in range(num_layers)])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        h = inputs
        m = mask.float()
        for layer, dropout in zip(self._layers, self._dropouts):
            h = dropout(layer(h, m))
        return h


class GatedCnnSeq2SeqEncoder(Seq2SeqEncoder):
    """
    https://arxiv.org/abs/1612.08083
    https://arxiv.org/abs/1705.03122
    """

    class Layer(NamedTuple):
        kernel_size: int
        output_dim: int
        dilation: int = 1

    class ResidualBlock(torch.nn.Module):
        def __init__(
            self,
            input_dim: int,
            layers: Sequence["GatedCnnSeq2SeqEncoder.Layer"],
            direction: Literal["forward", "backward"],
            do_weight_norm: bool = True,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            self.dropout = dropout
            self._convolutions = torch.nn.ModuleList()
            last_dim = input_dim
            for k, layer in enumerate(layers):
                if layer.dilation == 1:
                    conv = torch.nn.Conv1d(
                        in_channels=last_dim,
                        out_channels=layer.output_dim * 2,
                        kernel_size=layer.kernel_size,
                        stride=1,
                        padding=layer[0] - 1,
                        bias=True,
                    )
                else:
                    assert layer.kernel_size == 2, "only support kernel = 2 for now"
                    conv = torch.nn.Conv1d(
                        in_channels=last_dim,
                        out_channels=layer.output_dim * 2,
                        kernel_size=layer.kernel_size,
                        stride=1,
                        padding=layer.dilation,
                        dilation=layer.dilation,
                        bias=True,
                    )

                if k == 0:
                    conv_dropout = dropout
                else:
                    conv_dropout = 0.0
                std = math.sqrt((4 * (1.0 - conv_dropout)) / (layer.kernel_size * last_dim))

                conv.weight.data.normal_(0, std=std)
                if conv.bias is not None:
                    conv.bias.data.zero_()

                if do_weight_norm:
                    conv = torch.nn.utils.weight_norm(conv, name="weight", dim=0)

                self._convolutions.append(conv)
                last_dim = layer.output_dim

            assert last_dim == input_dim

            if direction not in ("forward", "backward"):
                raise ValueError(f"invalid direction: {direction}")
            self._direction = direction

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output = inputs
            sequence_length = inputs.size(2)
            for k, convolution in enumerate(self._convolutions):
                if k == 0 and self.dropout > 0:
                    output = torch.nn.functional.dropout(output, self.dropout, self.training)

                conv_out = convolution(output)

                dims_to_remove = conv_out.size(2) - sequence_length
                if dims_to_remove > 0:
                    if self._direction == "forward":
                        conv_out = conv_out.narrow(2, 0, sequence_length)
                    else:
                        conv_out = conv_out.narrow(2, dims_to_remove, sequence_length)

                output = torch.nn.functional.glu(conv_out, dim=1)

            return (output + inputs) * math.sqrt(0.5)

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence["GatedCnnSeq2SeqEncoder.Layer"]],
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._forward_residual_blocks = torch.nn.ModuleList()
        self._backward_residual_blocks = torch.nn.ModuleList()
        self._input_dim = input_dim
        self._output_dim = output_dim or input_dim * 2

        for layer in layers:
            self._forward_residual_blocks.append(
                GatedCnnSeq2SeqEncoder.ResidualBlock(input_dim, layer, "forward", dropout=dropout)
            )
            self._backward_residual_blocks.append(
                GatedCnnSeq2SeqEncoder.ResidualBlock(input_dim, layer, "backward", dropout=dropout)
            )

        self._projection: Optional[torch.nn.Linear] = None
        if output_dim:
            self._projection = torch.nn.Linear(input_dim * 2, output_dim)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        transposed_embeddings = torch.transpose(inputs, 1, 2)
        mask_for_fill = ~mask.unsqueeze(1)

        outputs: List[torch.Tensor] = []
        for k, blocks in enumerate([self._forward_residual_blocks, self._backward_residual_blocks]):
            out = transposed_embeddings
            for block in blocks:
                out = block(out.masked_fill(mask_for_fill, 0.0))
            outputs.append(out)

        output = cast(torch.FloatTensor, torch.cat(outputs, dim=1).transpose(1, 2))
        if self._projection:
            output = self._projection(output)
        return output


class PretrainedTransformerSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        eval_mode: bool = False,
        train_parameters: bool = True,
        submodule: Optional[str] = None,
        load_weights: bool = True,
    ) -> None:
        from transformers import AutoConfig, AutoModel

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        super().__init__()
        if load_weights:
            self._model = AutoModel.from_pretrained(pretrained_model_name)
        else:
            self._model = AutoModel.from_config(AutoConfig.from_pretrained(pretrained_model_name))

        if submodule:
            self._model = getattr(self._model, submodule)

        self.eval_mode = eval_mode
        if eval_mode:
            self._model.eval()

        if not train_parameters:
            for param in self._model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "PretrainedTransformerSeq2SeqEncoder":
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def get_input_dim(self) -> int:
        return int(self._model.config.hidden_size)

    def get_output_dim(self) -> int:
        return int(self._model.config.hidden_size)

    def forward(self, inputs: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        output = self._model(inputs_embeds=inputs, attention_mask=mask)
        return cast(torch.FloatTensor, output.last_hidden_state)


class XLstmSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        layers: Sequence[Literal["s", "m"]],
        use_conv: bool = True,
        kernel_size: int = 4,
        projection_factors: Tuple[float, float] = (2.0, 4 / 3),
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._xlstm = XLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            use_conv=use_conv,
            kernel_size=kernel_size,
            projection_factors=projection_factors,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def get_input_dim(self) -> int:
        return self._xlstm.input_dim

    def get_output_dim(self) -> int:
        return self._xlstm.output_dim

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        output, _ = self._xlstm(inputs, mask=mask)
        return cast(torch.FloatTensor, output)
