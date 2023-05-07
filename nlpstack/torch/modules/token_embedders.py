from __future__ import annotations

from typing import Any, cast

import torch

from nlpstack.data import Vocabulary
from nlpstack.torch.modules.lazy import LazyEmbedding
from nlpstack.torch.modules.scalarmix import ScalarMix


class TokenEmbedder(torch.nn.Module):
    def get_output_dim(self) -> int:
        raise NotImplementedError


class PassThroughTokenEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    def forward(self, embeddings: torch.FloatTensor, **kwargs: Any) -> torch.FloatTensor:
        return embeddings

    def get_output_dim(self) -> int:
        return self._embedding_dim


class Embedding(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self._embedding = LazyEmbedding(
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self._namespace = namespace

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        self._embedding.initialize_parameters(
            num_embeddings=vocab.get_vocab_size(self._namespace),
            padding_idx=vocab.get_pad_index(self._namespace),
        )

    def forward(self, token_ids: torch.LongTensor, **kwargs: Any) -> torch.FloatTensor:
        return cast(torch.FloatTensor, self._embedding(token_ids))

    def get_output_dim(self) -> int:
        return self._embedding.embedding_dim


class PretrainedTransformerEmbedder(TokenEmbedder):
    def __init__(
        self,
        pretrained_model_name: str,
        eval_mode: bool = False,
        train_parameters: bool = True,
        last_layer_only: bool = True,
    ) -> None:
        from transformers import AutoModel

        super().__init__()
        self._model = AutoModel.from_pretrained(pretrained_model_name)
        self._scalaer_mix: ScalarMix | None = None

        self.eval_mode = eval_mode
        if eval_mode:
            self._model.eval()

        if not train_parameters:
            for param in self._model.parameters():
                param.requires_grad = False

        if not last_layer_only:
            self._scalaer_mix = ScalarMix(self._model.config.num_hidden_layers)
            self._model.config.output_hidden_states = True

    def train(self, mode: bool = True) -> "PretrainedTransformerEmbedder":
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self) -> int:
        return cast(int, self._model.config.hidden_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        type_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                assert token_ids.shape == type_ids.shape

        transofrmer_inputs = {
            "input_ids": token_ids,
            "attention_mask": mask.float() if mask is not None else None,
        }
        if type_ids is not None:
            transofrmer_inputs["token_type_ids"] = type_ids

        transformer_outputs = self._model(**transofrmer_inputs)

        if self._scalaer_mix is not None:
            # The hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_outputs.hidden_states[1:]
            embeddings = self._scalaer_mix(hidden_states)
        else:
            embeddings = transformer_outputs.last_hidden_state

        return cast(torch.FloatTensor, embeddings)
