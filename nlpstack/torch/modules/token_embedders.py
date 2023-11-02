from contextlib import suppress
from os import PathLike
from typing import Any, Literal, Optional, Union, cast

import minato
import numpy
import torch

from nlpstack.data import Vocabulary
from nlpstack.data.embeddings import WordEmbedding
from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.lazy import LazyEmbedding
from nlpstack.torch.modules.scalarmix import ScalarMix
from nlpstack.torch.modules.seq2vec_encoders import BagOfEmbeddings, Seq2VecEncoder
from nlpstack.torch.modules.time_distributed import TimeDistributed
from nlpstack.torch.util import batched_span_select, fold, unfold


class TokenEmbedder(torch.nn.Module):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get_weight(self) -> Optional[torch.FloatTensor]:
        return None


class PassThroughTokenEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor, **kwargs: Any) -> torch.FloatTensor:
        return cast(torch.FloatTensor, embeddings.float())

    def get_output_dim(self) -> int:
        return self._embedding_dim


class FeedForwardTokenEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int, feedforwrad: FeedForward) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._feedforward = feedforwrad

    def forward(self, embeddings: torch.Tensor, **kwargs: Any) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._feedforward(embeddings))

    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()


class Embedding(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        namespace: str = "tokens",
        pretrained_embedding: Optional[WordEmbedding] = None,
        extend_vocab: bool = False,
        freeze: bool = True,
    ) -> None:
        if extend_vocab and not pretrained_embedding:
            raise ValueError("extend_vocab is only available when pretrained_embedding is given")
        super().__init__()
        self._embedding = LazyEmbedding(
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            freeze=not freeze,
        )
        self._namespace = namespace
        self._pretrained_embedding = pretrained_embedding
        self._extend_vocab = extend_vocab

    def get_weight(self) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._embedding.weight)

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        weight: Optional[torch.Tensor] = None
        if self._pretrained_embedding is not None:
            if self._extend_vocab:
                self._pretrained_embedding.extend_vocab(vocab, self._namespace)
            all_tokens = set(vocab.get_token_to_index(self._namespace).keys())
            all_embeddings = numpy.asarray(
                [self._pretrained_embedding[token] for token in all_tokens if token in self._pretrained_embedding]
            )
            embeddings_mean = float(numpy.mean(all_embeddings))
            embeddings_std = float(numpy.std(all_embeddings))
            weight = torch.FloatTensor(vocab.get_vocab_size(self._namespace), self._embedding.embedding_dim)
            torch.nn.init.normal_(weight, embeddings_mean, embeddings_std)
            for token, index in vocab.get_token_to_index(self._namespace).items():
                if token in self._pretrained_embedding:
                    weight[index] = torch.FloatTensor(self._pretrained_embedding[token])
            self._pretrained_embedding = None
        self._embedding.initialize_parameters(
            num_embeddings=vocab.get_vocab_size(self._namespace),
            padding_idx=vocab.get_pad_index(self._namespace),
            weight=weight,
        )

    def forward(self, token_ids: torch.LongTensor, **kwargs: Any) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._embedding(token_ids))

    def get_output_dim(self) -> int:
        return self._embedding.embedding_dim


class TokenSubwordsEmbedder(TokenEmbedder):
    def __init__(
        self,
        embedder: TokenEmbedder,
        encoder: Optional[Seq2VecEncoder] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._embedder = TimeDistributed(embedder)
        self._encoder = TimeDistributed(encoder or BagOfEmbeddings(embedder.get_output_dim()))
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self._embedder.module.setup(*args, **kwargs)

    def forward(
        self,
        *args: Any,
        mask: Optional[torch.BoolTensor] = None,
        subword_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        embeddings = self._embedder(*args, mask=subword_mask, **kwargs)
        encodings = self._encoder(embeddings, mask=subword_mask)
        if self._dropout is not None:
            encodings = self._dropout(encodings)
        return cast(torch.FloatTensor, encodings)

    def get_output_dim(self) -> int:
        return self._encoder.module.get_output_dim()


class AggregativeTokenEmbedder(TokenEmbedder):
    def __init__(
        self,
        embedder: TokenEmbedder,
        encoder: Optional[Seq2VecEncoder] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoder = TimeDistributed(encoder or BagOfEmbeddings(embedder.get_output_dim()))
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        *args: Any,
        offsets: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
        subword_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        embeddings = self._embedder(*args, mask=subword_mask, **kwargs).contiguous()
        span_embeddings, span_mask = batched_span_select(embeddings, offsets)
        encodings = self._encoder(span_embeddings, mask=span_mask)
        if self._dropout is not None:
            encodings = self._dropout(encodings)
        return cast(torch.FloatTensor, encodings)


class PretrainedTransformerEmbedder(TokenEmbedder):
    class _Embedding(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            from transformers import PreTrainedModel

            super().__init__()

            assert isinstance(model, PreTrainedModel)
            self.embedding = model.get_input_embeddings()

        def forward(self, token_ids: torch.LongTensor) -> torch.FloatTensor:
            return cast(torch.FloatTensor, self.embedding(token_ids))

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        eval_mode: bool = False,
        layer_to_use: Literal["embeddings", "last", "all"] = "last",
        train_parameters: bool = True,
        submodule: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> None:
        from transformers import AutoModel

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model_name)
        scalar_mix: Optional[ScalarMix] = None
        output_dim: int = model.config.hidden_size

        if gradient_checkpointing is not None:
            model.config.update({"gradient_checkpointing": gradient_checkpointing})

        if submodule is not None:
            assert hasattr(model, submodule), f"Submodule {submodule} is not found in {pretrained_model_name}"
            model = getattr(model, submodule)

        self.eval_mode = eval_mode
        if eval_mode:
            model.eval()

        if not train_parameters:
            for param in model.parameters():
                param.requires_grad = False

        if layer_to_use == "all":
            scalar_mix = ScalarMix(model.config.num_hidden_layers)
            model.config.output_hidden_states = True
        elif layer_to_use == "embeddings":
            model = PretrainedTransformerEmbedder._Embedding(model)

        self._model = model
        self._scalar_mix = scalar_mix
        self._max_length = max_length
        self._output_dim = output_dim

    def train(self, mode: bool = True) -> "PretrainedTransformerEmbedder":
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self) -> int:
        return self._output_dim

    def get_weight(self) -> Optional[torch.FloatTensor]:
        if isinstance(self._model, PretrainedTransformerEmbedder._Embedding):
            return cast(torch.FloatTensor, self._model.embedding.weight)
        return cast(torch.FloatTensor, self._model.get_input_embeddings().weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        type_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                assert token_ids.shape == type_ids.shape

        sequence_length = token_ids.shape[1]

        if self._max_length is not None:
            token_ids = fold(token_ids, self._max_length)
            if mask is not None:
                mask = fold(mask, self._max_length)
            if type_ids is not None:
                type_ids = fold(type_ids, self._max_length)

        if isinstance(self._model, PretrainedTransformerEmbedder._Embedding):
            embeddings = self._model(token_ids)
        else:
            transofrmer_inputs = {
                "input_ids": token_ids,
                "attention_mask": mask.float() if mask is not None else None,
            }
            if type_ids is not None:
                transofrmer_inputs["token_type_ids"] = type_ids

            transformer_outputs = self._model(**transofrmer_inputs)

            if self._scalar_mix is not None:
                # The hidden states will also include the embedding layer, which we don't
                # include in the scalar mix. Hence the `[1:]` slicing.
                hidden_states = transformer_outputs.hidden_states[1:]
                embeddings = self._scalar_mix(hidden_states)
            else:
                embeddings = transformer_outputs.last_hidden_state

        if self._max_length is not None:
            embeddings = unfold(embeddings, sequence_length)

        return cast(torch.FloatTensor, embeddings)
