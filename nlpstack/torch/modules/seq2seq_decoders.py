import dataclasses
from contextlib import suppress
from os import PathLike
from typing import Callable, Generic, Literal, Optional, Tuple, TypeVar, Union, cast

import minato
import torch

from nlpstack.torch.modules.transformer import CausalTransformerDecoder, CausalTransformerDecoderLayer
from nlpstack.torch.util import add_positional_features, masked_softmax, weighted_sum

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
        cache: Optional[torch.Tensor] = None

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
        return TransformerSeq2SeqDecoder.State()

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
        if self._use_cross_attention:
            if memory is None:
                raise ValueError("memory is required when use_cross_attention is True")
            memory_mask = cast(
                torch.BoolTensor, memory_mask if memory_mask is not None else memory.new_ones(memory.shape[:2]).bool()
            )
            attention_weight = masked_softmax(torch.bmm(inputs, memory.transpose(1, 2)), memory_mask)
            inputs = weighted_sum(memory, attention_weight)
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
        if not self._use_cross_attention:
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


class PretrainedTransformerSeq2SeqDecoder(Seq2SeqDecoder["PretrainedTransformerSeq2SeqDecoder.State"]):
    @dataclasses.dataclass
    class State:
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        eval_mode: bool = False,
        train_parameters: bool = True,
        submodule: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        super().__init__()
        self._model = AutoModel.from_pretrained(pretrained_model_name)
        if submodule:
            self._model = getattr(self._model, submodule)

        self.eval_mode = eval_mode
        if eval_mode:
            self._model.eval()

        if not train_parameters:
            for param in self._model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "PretrainedTransformerSeq2SeqDecoder":
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        last_state: Optional["PretrainedTransformerSeq2SeqDecoder.State"] = None,
    ) -> Tuple[torch.Tensor, "PretrainedTransformerSeq2SeqDecoder.State"]:
        last_state = last_state or self.get_initial_state(inputs, memory, inputs_mask, memory_mask)
        output = self._model(
            inputs_embeds=inputs,
            attention_mask=inputs_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=last_state.past_key_values,
            use_cache=not self.training,
        )
        return output.last_hidden_state, PretrainedTransformerSeq2SeqDecoder.State(output.past_key_values)

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> "PretrainedTransformerSeq2SeqDecoder.State":
        return PretrainedTransformerSeq2SeqDecoder.State()

    def get_input_dim(self) -> int:
        return int(self._model.config.hidden_size)

    def get_output_dim(self) -> int:
        return int(self._model.config.hidden_size)
