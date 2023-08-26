import dataclasses
from typing import Any, Generic, Mapping, Optional, Sequence, Tuple, cast

import numpy
import torch
import torch.nn.functional as F

from nlpstack.torch.generation import BeamSearch
from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.heads import LanguageModelingHead
from nlpstack.torch.modules.seq2seq_decoders import Seq2SeqDecoder, Seq2SeqDecoderState
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.util import get_mask_from_text, get_token_ids_from_text

from .datamodules import CausalLanguageModelingDataModule
from .types import CausalLanguageModelingInference


@dataclasses.dataclass
class TorchCausalLanguageModelOutput:
    inference: CausalLanguageModelingInference
    loss: Optional[torch.FloatTensor] = None


class TorchCausalLanguageModel(TorchModel[CausalLanguageModelingInference], Generic[Seq2SeqDecoderState]):
    """
    A causal language model for PyTorch.

    Args:
        embedder: The token embedder.
        decoder: The sequence decoder.
        lmhead: The language modeling head. If `None`, the prediction head is tied with the token embedder.
        dropout: The dropout rate.
        ingore_padding_loss: Whether to ignore the padding tokens when computing the loss.
        beam_search: The beam search module.
        token_namespace: The namespace of the tokens.
    """

    def __init__(
        self,
        embedder: Embedding,
        decoder: Seq2SeqDecoder[Seq2SeqDecoderState],
        lmhead: Optional[LanguageModelingHead] = None,
        dropout: Optional[float] = None,
        ignore_padding_loss: bool = False,
        beam_search: Optional[BeamSearch] = None,
        token_namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._decoder = decoder
        self._lmhead = lmhead
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self._ignore_padding_loss = ignore_padding_loss
        self._beam_search = beam_search or BeamSearch()
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self._token_namespace = token_namespace
        self._eos_index: Optional[int] = None

    def setup(
        self,
        *args: Any,
        datamodule: CausalLanguageModelingDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        if datamodule.vocab.has_eos_token(self._token_namespace):
            self._eos_index = datamodule.vocab.get_eos_index(self._token_namespace)
        else:
            self._eos_index = datamodule.vocab.get_pad_index(self._token_namespace)
        self._beam_search.setup(
            *args,
            datamodule=datamodule,
            vocab=datamodule.vocab,
            tokenizer=datamodule.tokenizer,
            eos_index=self._eos_index,
            **kwargs,
        )

    def _compute_logits(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        if self._dropout:
            inputs = self._dropout(inputs)
        if self._lmhead is not None:
            return cast(torch.FloatTensor, self._lmhead(inputs))
        return cast(torch.FloatTensor, F.linear(inputs, self._embedder.weight))

    def _get_generated_sequences(
        self,
        given_token_ids: torch.LongTensor,
        pred_token_ids: torch.LongTensor,
        given_mask: torch.BoolTensor,
        pred_mask: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        device = given_token_ids.device
        batch_size, beam_size, total_length = pred_token_ids.size()
        extracted_token_ids = []
        extracted_mask = []
        for i in range(pred_token_ids.shape[0]):
            given_length = int(given_mask[i].sum().item())
            extracted_token_ids.append(pred_token_ids[i, :, given_length:])
            extracted_mask.append(pred_mask[i, :, given_length:])
        max_generated_length = max([x.size(1) for x in pred_token_ids])
        output_token_ids = torch.zeros((batch_size, beam_size, max_generated_length), dtype=torch.long, device=device)
        output_mask = torch.zeros((batch_size, beam_size, max_generated_length), dtype=torch.bool, device=device)
        for i in range(batch_size):
            extracted_length = extracted_token_ids[i].size(1)
            output_token_ids[i, :, :extracted_length] = extracted_token_ids[i]
            output_mask[i, :, :extracted_length] = extracted_mask[i]
        return cast(torch.LongTensor, output_token_ids), cast(torch.BoolTensor, output_mask)

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        labels: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        metadata: Optional[Sequence[Any]] = None,
        *,
        return_only_generated: bool = False,
        **kwargs: Any,
    ) -> TorchCausalLanguageModelOutput:
        token_ids = get_token_ids_from_text(text)
        mask = get_mask_from_text(text)

        embeddings = self._embedder(token_ids)

        loss: Optional[torch.FloatTensor] = None
        inference = CausalLanguageModelingInference(
            pred_token_ids=numpy.zeros((token_ids.size(0), 1), dtype=int),
            pred_mask=numpy.ones((token_ids.size(0), 1), dtype=bool),
            scores=numpy.zeros((token_ids.size(0), 1), dtype=float),
            metadata=metadata,
        )
        if labels is not None:
            encodings, _ = self._decoder(embeddings, inputs_mask=mask)
            logits = self._compute_logits(encodings)

            # compute loss and perplexity
            target = get_token_ids_from_text(labels)
            target_mask = get_mask_from_text(labels)

            if self._ignore_padding_loss:
                target = cast(torch.LongTensor, target.masked_fill(~target_mask, -100))

            loss = self._loss(logits.view(-1, logits.size(2)), target.contiguous().view(-1)) / logits.size(0)
            perplexity = (
                F.log_softmax(logits, dim=2)
                .gather(dim=2, index=target.masked_fill(~target_mask, 0).unsqueeze(2))
                .squeeze(2)
                .masked_select(target_mask)
                .mean()
                .neg()
                .exp()
                .item()
            )

            inference.pred_token_ids = logits.argmax(dim=-1).unsqueeze(1).detach().cpu().numpy()
            inference.pred_mask = target_mask.unsqueeze(1).detach().cpu().numpy()
            inference.gold_token_ids = target.detach().cpu().numpy()
            inference.gold_mask = target_mask.detach().cpu().numpy()
            inference.perplexity = perplexity

        if not self.training:

            def step(
                token_ids: torch.LongTensor, state: Seq2SeqDecoderState
            ) -> Tuple[torch.Tensor, Seq2SeqDecoderState]:
                batch_size, beam_size, sequence_length = token_ids.size()
                embeddings = self._embedder(token_ids.view(batch_size * beam_size, sequence_length))
                encodings, state = self._decoder(embeddings, last_state=state)
                logits = self._compute_logits(encodings)
                log_probs = F.log_softmax(logits[:, -1, :], dim=1).view(batch_size, beam_size, -1)
                return log_probs, state

            state = self._decoder.get_initial_state(embeddings, inputs_mask=mask)
            pred_token_ids, pred_mask, scores = self._beam_search.search(token_ids, mask, state, step, **kwargs)

            if return_only_generated:
                pred_token_ids, pred_mask = self._get_generated_sequences(token_ids, pred_token_ids, mask, pred_mask)

            inference.pred_token_ids = pred_token_ids.detach().cpu().numpy()
            inference.pred_mask = pred_mask.detach().cpu().numpy()
            inference.scores = scores.detach().cpu().numpy()

        return TorchCausalLanguageModelOutput(inference=inference, loss=loss)
