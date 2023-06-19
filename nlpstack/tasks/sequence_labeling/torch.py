import dataclasses
from typing import Any, Mapping, Optional, Sequence

import torch

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.crf import CrfDecoder
from nlpstack.torch.modules.lazy import LazyLinearOutput
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.util import get_mask_from_text

from .data import SequenceLabelingInference
from .datamodules import SequenceLabelingDataModule


@dataclasses.dataclass
class SequenceLabelerOutput:
    inference: SequenceLabelingInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchSequenceLabeler(TorchModel):
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

        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._decoder = decoder
        self._classifier = LazyLinearOutput(encoder.get_output_dim())
        self._dropout = torch.nn.Dropout(dropout) if dropout else None
        self._top_k = top_k
        self._loss = torch.nn.CrossEntropyLoss()
        self._label_namespace = label_namespace

    def setup(
        self,
        *args: Any,
        datamodule: SequenceLabelingDataModule,
        **kwargs: Any,
    ) -> None:
        num_labels = datamodule.vocab.get_vocab_size(self._label_namespace)
        self._classifier.initialize_parameters(out_features=num_labels)
        if self._decoder is not None:
            self._decoder.setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)

    def forward(  # type: ignore[override]
        self,
        tokens: Mapping[str, Mapping[str, torch.Tensor]],
        labels: Optional[torch.LongTensor] = None,
        metadata: Optional[Sequence[Any]] = None,
    ) -> SequenceLabelerOutput:
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

        if self._decoder is not None and self._top_k is not None:
            output.inference.decodings = self._decoder.viterbi_decode(logits, mask, top_k=self._top_k)

        if labels is not None:
            if self._decoder is not None:
                log_likelihood = self._decoder(logits, labels, mask)
                output.loss = -log_likelihood
            else:
                output.loss = self._loss(logits, labels.long())

        return output
