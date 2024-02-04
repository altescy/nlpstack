import dataclasses
from typing import Any, Mapping, Optional, Sequence

import torch

from nlpstack.integrations.torch.model import TorchModel
from nlpstack.integrations.torch.modules.crf import CrfDecoder
from nlpstack.integrations.torch.modules.lazy import LazyLinearOutput
from nlpstack.integrations.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.integrations.torch.modules.text_embedders import TextEmbedder
from nlpstack.integrations.torch.util import get_mask_from_text

from .datamodules import SequenceLabelingDataModule
from .types import SequenceLabelingInference


@dataclasses.dataclass
class SequenceLabelerOutput:
    inference: SequenceLabelingInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchSequenceLabeler(TorchModel[SequenceLabelingInference]):
    """
    A neural sequence labeling model for PyTorch.

    Args:
        embedder: The text embedder.
        encoder: The sequence-to-sequence encoder for contextualizing the text representation.
        decoder: The CRF decoder. Defaults to `None`.
        dropout: The dropout rate. Defaults to `None`.
        top_k: The top-k parameter for the CRF decoder. If given, top-k predictions are returned. Defaults to `None`,
        label_namespace: The namespace of the labels. Defaults to `"labels"`.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2SeqEncoder,
        decoder: Optional[CrfDecoder] = None,
        dropout: Optional[float] = None,
        top_k: Optional[int] = None,
        label_namespace: str = "labels",
    ) -> None:
        if decoder is None and top_k is not None:
            raise ValueError("top_k is only supported when decoder is given.")
        if decoder is not None and top_k is None:
            top_k = 1

        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._decoder = decoder
        self._classifier = LazyLinearOutput(encoder.get_output_dim())
        self._dropout = torch.nn.Dropout(dropout) if dropout else None
        self._top_k = top_k
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self._label_namespace = label_namespace

    def setup(
        self,
        *args: Any,
        datamodule: SequenceLabelingDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        num_labels = datamodule.vocab.get_vocab_size(self._label_namespace)
        self._classifier.initialize_parameters(out_features=num_labels)
        if self._decoder is not None:
            self._decoder.setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)

    def forward(  # type: ignore[override]
        self,
        tokens: Mapping[str, Mapping[str, torch.Tensor]],
        labels: Optional[torch.LongTensor] = None,
        metadata: Optional[Sequence[Any]] = None,
        *,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> SequenceLabelerOutput:
        top_k = top_k or self._top_k
        mask = get_mask_from_text(tokens)

        embeddings = self._embedder(tokens)
        encodings = self._encoder(embeddings, mask)

        if self._dropout is not None:
            encodings = self._dropout(encodings)

        logits = self._classifier(encodings)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        inference = SequenceLabelingInference(
            probs=probs.detach().cpu().numpy(),
            mask=mask.detach().cpu().numpy(),
            metadata=metadata,
        )
        output = SequenceLabelerOutput(inference, logits)

        if labels is not None:
            inference.labels = labels.detach().cpu().numpy()
            if self._decoder is not None:
                log_likelihood = self._decoder(logits, labels, mask)
                output.loss = -log_likelihood
            else:
                flattened_logits = logits.view(-1, logits.size(-1))
                flattened_labels = labels.masked_fill(~mask, -100).long().view(-1)
                output.loss = self._loss(flattened_logits, flattened_labels) / logits.size(0)

        if self._decoder is not None and top_k is not None:
            output.inference.decodings = self._decoder.viterbi_decode(logits, mask, top_k=top_k)

        return output
