import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

import numpy
import torch
import torch.nn.functional as F

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.heads import LanguageModelingHead
from nlpstack.torch.modules.seq2seq_decoders import Seq2SeqDecoder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.util import get_mask_from_text, get_token_ids_from_text

from .datamodules import CausalLanguageModelingDataModule
from .types import CausalLanguageModelingInference


@dataclasses.dataclass
class TorchCausalLanguageModelOutput:
    inference: CausalLanguageModelingInference
    loss: Optional[torch.FloatTensor] = None


class TorchCausalLanguageModel(TorchModel[CausalLanguageModelingInference]):
    def __init__(
        self,
        embedder: Embedding,
        decoder: Seq2SeqDecoder,
        lmhead: Optional[LanguageModelingHead] = None,
        dropout: Optional[float] = None,
        ignore_padding_loss: bool = True,
        token_namespace: str = "tokens",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._decoder = decoder
        self._lmhead = lmhead
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self._ignore_padding_loss = ignore_padding_loss
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self._token_namespace = token_namespace
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_k = top_k
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
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        return_only_generated: bool = False,
        **kwargs: Any,
    ) -> TorchCausalLanguageModelOutput:
        max_new_tokens = max_new_tokens or self._max_new_tokens
        temperature = temperature or self._temperature
        top_k = top_k or self._top_k

        token_ids = get_token_ids_from_text(text)
        mask = get_mask_from_text(text)

        embeddings = self._embedder(token_ids)

        loss: Optional[torch.FloatTensor] = None
        inference = CausalLanguageModelingInference(
            pred_token_ids=numpy.zeros((token_ids.size(0), 1), dtype=int),
            pred_mask=numpy.ones((token_ids.size(0), 1), dtype=bool),
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
            # take decoding steps
            min_given_tokens = int(mask.sum(dim=1).min().item())
            max_given_tokens = int(mask.sum(dim=1).max().item())

            original_token_ids = token_ids
            token_ids = cast(torch.LongTensor, token_ids[:, :min_given_tokens])
            embeddings = embeddings[:, :min_given_tokens]

            last_state = self._decoder.get_initial_state(embeddings, inputs_mask=mask)
            is_done = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
            for timestep in range(min_given_tokens, min_given_tokens + max_new_tokens):
                output, last_state = self._decoder(embeddings, last_state=last_state)
                logits = self._compute_logits(output[:, -1:, :])
                if top_k:
                    v, _ = logits.topk(top_k, dim=2)
                    logits = cast(torch.FloatTensor, logits.masked_fill(logits < v[:, :, -1:], -float("inf")))
                new_token_id = (
                    torch.multinomial((logits[:, -1, :] / temperature).softmax(1), num_samples=1)
                    if temperature > 0
                    else logits.argmax(dim=2)
                )
                if timestep < max_given_tokens:
                    new_token_id = torch.where(
                        mask[:, timestep : timestep + 1],
                        original_token_ids[:, timestep : timestep + 1],
                        new_token_id,
                    )
                if self._eos_index is not None:
                    is_done |= new_token_id.squeeze(1) == self._eos_index
                    new_token_id = new_token_id.masked_fill(is_done.unsqueeze(1), self._eos_index)
                token_ids = cast(torch.LongTensor, torch.cat([token_ids, new_token_id], dim=1))
                if is_done.all():
                    break
                embeddings = self._embedder(new_token_id)

            pred_token_ids = cast(
                torch.LongTensor,
                token_ids.unsqueeze(1),
            )
            pred_mask = cast(
                torch.BoolTensor,
                torch.ones_like(pred_token_ids) if self._eos_index else pred_token_ids != self._eos_index,
            )
            if return_only_generated:
                pred_token_ids, pred_mask = self._get_generated_sequences(token_ids, pred_token_ids, mask, pred_mask)

            inference.pred_token_ids = pred_token_ids.detach().cpu().numpy()
            inference.pred_mask = pred_mask.detach().cpu().numpy()

        return TorchCausalLanguageModelOutput(inference=inference, loss=loss)
