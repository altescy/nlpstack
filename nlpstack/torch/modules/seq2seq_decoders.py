"""
PyTorch modules for sequence-to-sequence decoders.
All decoders have the same interface and can be used in the same way.

Example:
    Decoding by putting batched inputs all together:

    >>> import torch
    >>> from nlpstack.torch.modules.seq2seq_decoders import LstmSeq2SeqDecoder
    >>> decoder = LstmSeq2SeqDecoder(input_dim=16, hidden_dim=16, num_layers=1)
    >>> inputs = torch.randn(2, 3, 16)
    >>> memory = torch.randn(2, 5, 16)
    >>> outputs, state = decoder(inputs, memory)

    Step-by-step decoding:

    >>> state = decoder.get_initial_state(inputs, memory)
    >>> output = inputs  # Shape: (batch_size, given_sequence_length, input_dim)
    >>> for _ in range(num_steps):
    ...     outputs, state = decoder(output, memory, last_state=state)
    ...     output = outputs[:, -1:]  # Shape: (batch_size, 1, input_dim)
"""


import dataclasses
from contextlib import suppress
from os import PathLike
from typing import Any, Callable, Generic, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

import minato
import torch

from nlpstack.torch.generation.beam_search import StepStateInterface
from nlpstack.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.torch.modules.transformer import CausalTransformerDecoder, CausalTransformerDecoderLayer
from nlpstack.torch.util import add_positional_features, masked_softmax, weighted_sum

Seq2SeqDecoderState = TypeVar("Seq2SeqDecoderState", bound=StepStateInterface)


class Seq2SeqDecoder(torch.nn.Module, Generic[Seq2SeqDecoderState]):
    """
    A base module for sequence-to-sequence decoders.

    Attributes:
        State: The type of the decoder state for step-by-step decoding.
    """

    State: Type[Seq2SeqDecoderState]

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        last_state: Optional[Seq2SeqDecoderState] = None,
    ) -> Tuple[torch.Tensor, Seq2SeqDecoderState]:
        """
        Args:
            inputs: A tensor of shape `(batch_size, input_length, input_dim)`.
            memory: A tensor of shape `(batch_size, memory_length, memory_dim)` representing the
                memory vectors such as the encoder outputs.
            inputs_mask: A tensor of shape `(batch_size, input_length)` representing the mask for
                the inputs.
            memory_mask: A tensor of shape `(batch_size, memory_length)` representing the mask for
                the memory.
            last_state: The last decoder state, which is used for step-by-step decoding. Defaults
                to `None`.
        """
        raise NotImplementedError

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> Seq2SeqDecoderState:
        raise NotImplementedError

    def can_take_memory(self) -> bool:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class TransformerSeq2SeqDecoder(Seq2SeqDecoder["TransformerSeq2SeqDecoder.State"]):
    """
    A sequence-to-sequence decoder based on the Transformer architecture.

    Args:
        input_dim: The dimension of the input vectors.
        num_layers: The number of layers.
        feedforward_hidden_dim: The hidden dimension of the feedforward layers. Defaults to `2048`.
        num_attention_heads: The number of attention heads. Defaults to `8`.
        positional_encoding: The type of positional encoding. Defaults to `"sinusoidal"`.
        positional_embedding_size: The size of the positional embeddings. Defaults to `512`.
        dropout: The dropout rate. Defaults to `0.1`.
        layer_norm_eps: The epsilon value for layer normalization. Defaults to `1e-5`.
        activation: The activation function. Defaults to `"relu"`.
        norm_first: Whether to apply layer normalization before the attention layer. Defaults to
            `False`.
        use_cross_attention: Whether to use cross attention. Please set `True` for encoder-decoder
            architectures. Defaults to `False`.
    """

    @dataclasses.dataclass
    class State:
        """
        The decoder state for step-by-step decoding.

        Parameters:
            cache: The cache for the decoder layers. A tensor of shape `(num_layers, batch_size,
                sequence_length, embedding_dim)`.
        """

        cache: Optional[torch.Tensor] = None

        def update(self, backpointer: torch.LongTensor) -> None:
            if self.cache is None:
                return
            batch_size, beam_size = backpointer.size()
            if self.cache.size(1) == batch_size:
                # Shape: (num_layers, batch_size * beam_size, sequence_length, embedding_dim)
                self.cache = self.cache.repeat_interleave(beam_size, dim=1)

            assert self.cache.size(1) == batch_size * beam_size

            # Shape: (batch_size * beam_size)
            flattened_backpointer = (
                backpointer + (beam_size * torch.arange(batch_size, device=backpointer.device))
            ).view(-1)
            # Shape: (num_layers, batch_size * beam_size, sequence_length, embedding_dim)
            self.cache = self.cache[:, flattened_backpointer, :, :]

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

    def can_take_memory(self) -> bool:
        return self._use_cross_attention

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim


class LstmSeq2SeqDecoder(Seq2SeqDecoder["LstmSeq2SeqDecoder.State"]):
    """
    A stacked LSTM decoder.

    Args:
        input_dim: The dimension of the inputs to the decoder.
        hidden_dim: The dimension of the outputs of the decoder.
        num_layers: The number of layers in the decoder.
        bias: Whether or not to include bias parameters in the LSTM. Defaults to `True`.
        dropout: The dropout probability. Defaults to `0.1`.
        use_cross_attention: Whether to use cross attention. Please set `True` for encoder-decoder
            architectures. Defaults to `False`.
        initial_state_encoder: An optional `Seq2VecEncoder` to use to initialize the hidden state
            of the decoder from memory vectors. If `None`, the initial hidden state is initialized
            to zero. Defaults to `None`.
    """

    @dataclasses.dataclass
    class State:
        """
        The decoder state for step-by-step decoding.

        Parameters:
            hidden: The hidden state of the decoder. A tensor of shape `(num_layers, batch_size, hidden_dim)`.
            cell: The cell state of the decoder. A tensor of shape `(num_layers, batch_size, hidden_dim)`.
        """

        hidden: torch.Tensor
        cell: torch.Tensor

        def update(self, backpointer: torch.LongTensor) -> None:
            batch_size, beam_size = backpointer.size()
            if self.hidden.size(1) == batch_size:
                self.hidden = self.hidden.repeat_interleave(beam_size, dim=1)
            if self.cell.size(1) == batch_size:
                self.cell = self.cell.repeat_interleave(beam_size, dim=1)

            assert self.hidden.size(1) == batch_size * beam_size
            assert self.cell.size(1) == batch_size * beam_size

            # Shape: (batch_size * beam_size,)
            flattened_backpointer = (
                backpointer + (beam_size * torch.arange(batch_size, device=backpointer.device).unsqueeze(1))
            ).view(-1)
            # Shape: (num_layers, batch_size * beam_size, hidden_dim)
            self.hidden = self.hidden[:, flattened_backpointer, :]
            # Shape: (num_layers, batch_size * beam_size, hidden_dim)
            self.cell = self.cell[:, flattened_backpointer, :]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        initial_state_encoder: Optional[Seq2VecEncoder] = None,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._use_cross_attention = use_cross_attention
        self._decoder = torch.nn.LSTM(input_dim, hidden_dim, num_layers, bias=bias, dropout=dropout, batch_first=True)  # type: ignore[no-untyped-call]
        self._initial_state_encoder = initial_state_encoder
        self._ininial_satte_projection = (
            None
            if initial_state_encoder is None
            else torch.nn.Linear(initial_state_encoder.get_output_dim(), hidden_dim * num_layers)
        )

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
        output, (hidden, cell) = self._decoder(inputs, (last_state.hidden, last_state.cell))
        return output, LstmSeq2SeqDecoder.State(hidden, cell)

    def get_initial_state(
        self,
        inputs: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        inputs_mask: Optional[torch.BoolTensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
    ) -> "LstmSeq2SeqDecoder.State":
        batch_size = inputs.size(0)
        if memory is None or self._initial_state_encoder is None:
            return LstmSeq2SeqDecoder.State(
                hidden=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
                cell=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
            )
        if memory is not None and not self._use_cross_attention:
            raise ValueError("memory is given but use_cross_attention is False")
        assert self._initial_state_encoder is not None and self._ininial_satte_projection is not None
        return LstmSeq2SeqDecoder.State(
            hidden=self._ininial_satte_projection(self._initial_state_encoder(memory, memory_mask)).view(
                self._num_layers, batch_size, self._hidden_dim
            ),
            cell=torch.zeros((self._num_layers, batch_size, self._hidden_dim), device=inputs.device),
        )

    def can_take_memory(self) -> bool:
        return self._use_cross_attention or self._initial_state_encoder is not None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim


class PretrainedTransformerSeq2SeqDecoder(Seq2SeqDecoder["PretrainedTransformerSeq2SeqDecoder.State"]):
    """
    A sequence decoder for pretrained transformer models.
    Note that this module requires transformers library to be installed.
    """

    @dataclasses.dataclass
    class State:
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None

        def update(self, backpointer: torch.LongTensor) -> None:
            raise NotImplementedError

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        eval_mode: bool = False,
        train_parameters: bool = True,
        submodule: Optional[str] = None,
        load_weights: bool = True,
        additional_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from transformers import AutoConfig, AutoModel

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        super().__init__()
        if load_weights:
            self._model = AutoModel.from_pretrained(pretrained_model_name, **(additional_config or {}))
        else:
            self._model = AutoModel.from_config(
                AutoConfig.from_pretrained(pretrained_model_name, **(additional_config or {}))
            )

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

    def can_take_memory(self) -> bool:
        memory_available_keywords = {"encdecatt", "encoder_att", "crossatt", "cross_att"}
        parmaeter_names = {name.lower() for name, _ in self.named_parameters()}
        return any(keyword in name for keyword in memory_available_keywords for name in parmaeter_names)

    def get_input_dim(self) -> int:
        return int(self._model.config.hidden_size)

    def get_output_dim(self) -> int:
        return int(self._model.config.hidden_size)
