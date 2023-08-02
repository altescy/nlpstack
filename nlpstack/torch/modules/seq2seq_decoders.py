import dataclasses
from typing import Callable, Generic, Literal, Optional, Tuple, TypeVar, Union

import torch

from nlpstack.torch.modules.transformer import CausalTransformerDecoder, CausalTransformerDecoderLayer
from nlpstack.torch.util import add_positional_features

Seq2SeqDecoderState = TypeVar("Seq2SeqDecoderState")


class Seq2SeqDecoder(torch.nn.Module, Generic[Seq2SeqDecoderState]):
    State: Seq2SeqDecoderState

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        last_state: Optional[Seq2SeqDecoderState] = None,
    ) -> Tuple[torch.Tensor, Seq2SeqDecoderState]:
        raise NotImplementedError

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> Seq2SeqDecoderState:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class TransformerSeq2SeqDecoder(Seq2SeqDecoder["TransformerSeq2SeqDecoder.State"]):
    @dataclasses.dataclass
    class State:
        cache: Optional[torch.Tensor]

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
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._use_cross_attention = use_cross_attention
        self._decoder = CausalTransformerDecoder(  # type: ignore[no-untyped-call]
            CausalTransformerDecoderLayer(
                d_model=input_dim,
                nhead=num_attention_heads,
                dim_feedforward=feedforward_hidden_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                activation=activation,
                norm_first=norm_first,
                batch_first=True,
                use_cross_attention=use_cross_attention,
            ),
            num_layers=num_layers,
        )
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

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        last_state: Optional["TransformerSeq2SeqDecoder.State"] = None,
    ) -> Tuple[torch.Tensor, "TransformerSeq2SeqDecoder.State"]:
        if memory is not None and not self._use_cross_attention:
            raise ValueError("memory is given but use_cross_attention is False")

        output = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        output, cache = self._decoder(
            tgt=inputs,
            memory=memory,
            tgt_key_padding_mask=inputs_mask.float() if inputs_mask is not None else None,
            memory_key_padding_mask=memory_mask.float() if memory_mask is not None else None,
            cache=last_state.cache if last_state is not None else None,
        )
        return output, TransformerSeq2SeqDecoder.State(cache)

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> "TransformerSeq2SeqDecoder.State":
        return TransformerSeq2SeqDecoder.State(cache=None)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


class LstmSeq2SeqDecoder(Seq2SeqDecoder["LstmSeq2SeqDecoder.State"]):
    @dataclasses.dataclass
    class State:
        hidden: torch.Tensor
        cell: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._use_cross_attention = use_cross_attention
        self._decoder = torch.nn.LSTM(input_dim, hidden_dim, num_layers, bias=bias, dropout=dropout, batch_first=True)  # type: ignore[no-untyped-call]
        self._init_state_projection = torch.nn.LazyLinear(num_layers * hidden_dim) if use_cross_attention else None

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        last_state: Optional["LstmSeq2SeqDecoder.State"] = None,
    ) -> Tuple[torch.Tensor, "LstmSeq2SeqDecoder.State"]:
        last_state = last_state or self.get_initial_state(inputs, memory, inputs_mask, memory_mask)
        if self.training:
            output, (hidden, cell) = self._decoder(inputs, (last_state.hidden, last_state.cell))
            return output, LstmSeq2SeqDecoder.State(hidden, cell)
        outputs = []
        for last_embedding in inputs.unbind(dim=1):
            output, (hidden, cell) = self._decoder(last_embedding.unsqueeze(1), (last_state.hidden, last_state.cell))
            last_state = LstmSeq2SeqDecoder.State(hidden=hidden, cell=cell)
            outputs.append(output)
        return torch.cat(outputs, dim=1), last_state

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> "LstmSeq2SeqDecoder.State":
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        if memory is None:
            return LstmSeq2SeqDecoder.State(
                hidden=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
                cell=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
            )
        if self._use_cross_attention:
            raise ValueError("memory is given but use_cross_attention is False")
        assert self._init_state_projection is not None
        last_token_indices = (
            memory_mask.sum(dim=1) - 1 if memory_mask is not None else torch.full((batch_size,), sequence_length - 1)
        ).long()
        return LstmSeq2SeqDecoder.State(
            hidden=self._init_state_projection(memory[torch.arange(batch_size), last_token_indices])
            .view(batch_size, self._num_layers, self._hidden_dim)
            .transpose(0, 1),
            cell=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
        )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim
