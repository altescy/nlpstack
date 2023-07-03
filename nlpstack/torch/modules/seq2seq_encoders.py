from contextlib import suppress
from typing import Callable, Literal, Optional, Sequence, Tuple, Union, cast

import torch

from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.util import add_positional_features


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
        positional_encoding: Optional[Literal["sinusoidal", "embedding"]] = None,
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
