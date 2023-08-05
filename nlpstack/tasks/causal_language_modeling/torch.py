import dataclasses
from typing import Any, Mapping, Optional, Sequence, cast

import numpy
import torch
import torch.nn.functional as F

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.lazy import LazyLinearOutput
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
        dropout: Optional[float] = None,
        tie_embeddings: bool = False,
        ignore_padding_loss: bool = True,
        token_namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._decoder = decoder
        self._head = LazyLinearOutput(decoder.get_output_dim()) if not tie_embeddings else None
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self._ignore_padding_loss = ignore_padding_loss
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
        if self._head is not None:
            vocab_size = datamodule.vocab.get_vocab_size(self._token_namespace)
            self._head.initialize_parameters(out_features=vocab_size)
        if datamodule.vocab.has_eos_token(self._token_namespace):
            self._eos_index = datamodule.vocab.get_eos_index(self._token_namespace)
        else:
            self._eos_index = datamodule.vocab.get_pad_index(self._token_namespace)

    def _compute_logits(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        if self._dropout:
            inputs = self._dropout(inputs)
        if self._head is not None:
            return cast(torch.FloatTensor, self._head(inputs))
        return cast(torch.FloatTensor, F.linear(inputs, self._embedder.weight))

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        labels: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        metadata: Optional[Sequence[Any]] = None,
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> TorchCausalLanguageModelOutput:
        token_ids = get_token_ids_from_text(text)
        mask = get_mask_from_text(text)

        embeddings = self._embedder(token_ids)

        loss: Optional[torch.FloatTensor] = None
        if labels is None:
            # take decoding steps
            last_state = self._decoder.get_initial_state(embeddings, inputs_mask=mask)
            is_done = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
            for _ in range(max_new_tokens):
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
                if self._eos_index is not None:
                    is_done |= new_token_id.squeeze(1) == self._eos_index
                    new_token_id = new_token_id.masked_fill(is_done.unsqueeze(1), self._eos_index)
                token_ids = cast(torch.LongTensor, torch.cat([token_ids, new_token_id], dim=1))
                if is_done.all():
                    break
                embeddings = self._embedder(new_token_id)
            pred_token_ids = token_ids.unsqueeze(1).detach().cpu().numpy()
            pred_mask = (
                numpy.ones(pred_token_ids.shape, dtype=bool) if self._eos_index else pred_token_ids != self._eos_index
            )
            inference = CausalLanguageModelingInference(
                pred_token_ids=pred_token_ids,
                pred_mask=pred_mask,
                metadata=metadata,
            )
        else:
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

            inference = CausalLanguageModelingInference(
                pred_token_ids=logits.argmax(dim=-1).unsqueeze(1).detach().cpu().numpy(),
                pred_mask=target_mask.unsqueeze(1).detach().cpu().numpy(),
                gold_token_ids=target.detach().cpu().numpy(),
                gold_mask=target_mask.detach().cpu().numpy(),
                perplexity=perplexity,
                metadata=metadata,
            )

        return TorchCausalLanguageModelOutput(inference=inference, loss=loss)
