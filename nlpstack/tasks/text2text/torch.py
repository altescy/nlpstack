import dataclasses
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

import numpy
import torch
import torch.nn.functional as F

from nlpstack.torch.generation import BeamSearch
from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.heads import LanguageModelingHead
from nlpstack.torch.modules.seq2seq_decoders import Seq2SeqDecoder, Seq2SeqDecoderState
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.util import get_mask_from_text, get_token_ids_from_text

from .datamodules import Text2TextDataModule
from .types import Text2TextInference


@dataclasses.dataclass
class TorchText2TextOutput:
    inference: Text2TextInference
    loss: Optional[torch.Tensor] = None


class TorchText2Text(TorchModel[Text2TextInference]):
    def __init__(
        self,
        source_embedder: Embedding,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        target_embedder: Optional[Embedding] = None,
        lmhead: Optional[LanguageModelingHead] = None,
        dropout: Optional[float] = None,
        ignore_padding_loss: bool = False,
        beam_search: Optional[BeamSearch] = None,
        source_namespace: str = "tokens",
        target_namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self._source_embedder = source_embedder
        self._target_embedder = target_embedder
        self._encoder = encoder
        self._decoder = decoder
        self._lmhead = lmhead
        self._dropout = None if dropout is None else torch.nn.Dropout(dropout)
        self._ignore_padding_loss = ignore_padding_loss
        self._beam_search = beam_search or BeamSearch()
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._target_bos_index: Optional[int] = None
        self._target_eos_index: Optional[int] = None
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def setup(
        self,
        *args: Any,
        datamodule: Text2TextDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        self._target_bos_index = datamodule.vocab.get_bos_index(self._target_namespace)
        if datamodule.vocab.has_eos_token(self._target_namespace):
            self._target_eos_index = datamodule.vocab.get_eos_index(self._target_namespace)
        else:
            self._target_eos_index = datamodule.vocab.get_pad_index(self._target_namespace)
        self._beam_search.setup(
            *args,
            datamodule=datamodule,
            vocab=datamodule.vocab,
            eos_index=self._target_eos_index,
            **kwargs,
        )

    def _compute_logits(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        if self._dropout:
            inputs = self._dropout(inputs)
        if self._lmhead:
            return cast(torch.FloatTensor, self._lmhead(inputs))
        return cast(
            torch.FloatTensor,
            F.linear(inputs, (self._target_embedder or self._source_embedder).weight),
        )

    def forward(  # type: ignore[override]
        self,
        source: Mapping[str, Mapping[str, torch.Tensor]],
        target: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> TorchText2TextOutput:
        if self._target_bos_index is None:
            raise RuntimeError("Target BOS index is not set")

        source_token_ids = get_token_ids_from_text(source)
        source_mask = get_mask_from_text(source)

        source_embeddings = self._source_embedder(source_token_ids)
        memory = self._encoder(source_embeddings, source_mask)

        batch_size = source_token_ids.size(0)

        loss: Optional[torch.FloatTensor] = None
        inference = Text2TextInference(
            pred_token_ids=self._target_bos_index * numpy.ones((batch_size, 1, 1), dtype=int),
            pred_mask=numpy.ones((batch_size, 1), dtype=bool),
            scores=numpy.zeros((batch_size, 1), dtype=float),
            metadata=metadata,
        )

        if target is not None:
            target_token_ids = get_token_ids_from_text(target)
            target_mask = get_mask_from_text(target)

            target_inputs = target_token_ids[:, :-1].contiguous()
            target_labels = target_token_ids[:, 1:].contiguous()
            target_inputs_mask = target_mask[:, :-1].contiguous()
            target_labels_mask = target_mask[:, 1:].contiguous()

            target_embeddings = (self._target_embedder or self._source_embedder)(target_inputs)
            output, _ = self._decoder(
                inputs=target_embeddings,
                memory=memory,
                inputs_mask=target_inputs_mask,
                memory_mask=source_mask,
            )
            logits = self._compute_logits(output)

            if self._ignore_padding_loss:
                target_labels = cast(torch.LongTensor, target_labels.masked_fill(~target_labels_mask, -100))

            loss = self._loss(logits.view(-1, logits.size(2)), target_labels.view(-1))
            perplexity = (
                F.log_softmax(logits, dim=2)
                .gather(dim=2, index=target_labels.masked_fill(~target_labels_mask, 0).unsqueeze(2))
                .squeeze(2)
                .masked_select(target_labels_mask)
                .mean()
                .neg()
                .exp()
                .item()
            )

            inference.pred_token_ids = logits.argmax(dim=2).unsqueeze(1).detach().cpu().numpy()
            inference.pred_mask = target_labels_mask.unsqueeze(1).detach().cpu().numpy()
            inference.gold_token_ids = target_labels.detach().cpu().numpy()
            inference.gold_mask = target_labels_mask.detach().cpu().numpy()
            inference.perplexity = perplexity

        if not self.training:

            @lru_cache
            def expand_memory(beam_size: int) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
                # Shape: (batch_size * beam_size, sequence_length, embedding_dim)
                expanded_memory = cast(torch.FloatTensor, memory.repeat_interleave(beam_size, dim=0))
                expanded_mask = cast(torch.BoolTensor, source_mask.repeat_interleave(beam_size, dim=0))
                return expanded_memory, expanded_mask

            def step(
                token_ids: torch.LongTensor, state: Seq2SeqDecoderState
            ) -> Tuple[torch.Tensor, Seq2SeqDecoderState]:
                batch_size, beam_size, sequence_length = token_ids.size()
                target_embeddings = (self._target_embedder or self._source_embedder)(
                    token_ids.view(batch_size * beam_size, sequence_length)
                )
                expanded_memory, expanded_mask = expand_memory(beam_size)
                decodings, state = self._decoder(
                    target_embeddings,
                    memory=expanded_memory,
                    memory_mask=expanded_mask,
                    last_state=state,
                )
                logits = self._compute_logits(decodings)
                log_probs = F.log_softmax(logits[:, -1, :], dim=1).view(batch_size, beam_size, -1)
                return log_probs, state

            target_token_ids = cast(
                torch.LongTensor,
                torch.full(
                    (batch_size, 1),
                    self._target_bos_index,
                    dtype=torch.long,
                    device=source_embeddings.device,
                ),
            )
            target_mask = cast(torch.BoolTensor, torch.ones_like(target_token_ids).bool())
            target_embeddings = (self._target_embedder or self._source_embedder)(target_token_ids)

            state = self._decoder.get_initial_state(target_embeddings, memory=memory, memory_mask=source_mask)
            pred_token_ids, pred_mask, scores = self._beam_search.search(
                target_token_ids, target_mask, state, step, **kwargs
            )

            inference.pred_token_ids = pred_token_ids.detach().cpu().numpy()
            inference.pred_mask = pred_mask.detach().cpu().numpy()
            inference.scores = scores.detach().cpu().numpy()

        return TorchText2TextOutput(inference=inference, loss=loss)
