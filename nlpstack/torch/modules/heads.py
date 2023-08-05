from contextlib import suppress
from os import PathLike
from typing import Any, Optional, Union, cast

import minato
import numpy
import torch

from nlpstack.data import Vocabulary
from nlpstack.data.embeddings import WordEmbedding
from nlpstack.torch.modules.lazy import LazyLinearOutput


class Head(torch.nn.Module):
    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class ClassificationHead(Head):
    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        namespace: str = "labels",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim: Optional[int] = None
        self._projection = LazyLinearOutput(input_dim)
        self._namespace = namespace

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        self._projection.initialize_parameters(out_features=vocab.get_vocab_size(self._namespace))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Output dimension not set")
        return self._output_dim

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._projection(inputs))


class LanguageModelingHead(Head):
    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        namespace: str = "tokens",
        pretrained_embedding: Optional[WordEmbedding] = None,
        extend_vocab: bool = False,
    ) -> None:
        if extend_vocab and not pretrained_embedding:
            raise ValueError("extend_vocab is only available when pretraind_embedding is given")
        super().__init__()
        self._projection = LazyLinearOutput(input_dim, bias=bias)
        self._namespace = namespace
        self._pretrained_embedding = pretrained_embedding
        self._extend_vocab = extend_vocab

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
            weight = torch.FloatTensor(vocab.get_vocab_size(self._namespace), self._projection.in_features)
            torch.nn.init.normal_(weight, embeddings_mean, embeddings_std)
            for token, index in vocab.get_token_to_index(self._namespace).items():
                if token in self._pretrained_embedding:
                    weight[index] = torch.FloatTensor(self._pretrained_embedding[token])
            self._pretrained_embedding = None
        self._projection.initialize_parameters(out_features=vocab.get_vocab_size(self._namespace))
        if weight is not None:
            with torch.no_grad():
                self._projection.weight.copy_(weight)

    def get_input_dim(self) -> int:
        return self._projection.in_features

    def get_output_dim(self) -> int:
        return self._projection.out_features

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._projection(inputs))


class PretrainedTransformerHead(Head):
    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        train_parameters: bool = True,
    ) -> None:
        from transformers import AutoModelWithLMHead

        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)

        super().__init__()

        model = AutoModelWithLMHead.from_pretrained(pretrained_model_name)

        self._input_dim = int(model.config.hidden_size)
        self._output_dim = int(model.config.vocab_size)
        self._head = model.lm_head
        if not train_parameters:
            for parameter in self._head.parameters():
                parameter.requires_grad = False

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._head(inputs))
