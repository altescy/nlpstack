from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from nlpstack.torch.metrics import Accuracy, Metric
from nlpstack.torch.models.model import Model
from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.util import get_mask_from_text


class TorchTextClassifier(Model):
    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Seq2SeqEncoder | None = None,
        feedforward: FeedForward | None = None,
        metrics: Sequence[Metric] | None = None,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(encoder.get_output_dim(), 2)

        self._contextualizer = contextualizer
        self._feedforward = feedforward

        self._loss = torch.nn.CrossEntropyLoss()

        self._metrics = metrics or [Accuracy()]

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        label: torch.LongTensor | None = None,
    ) -> dict[str, Any]:
        mask = get_mask_from_text(text)

        embeddings = self._embedder(text)

        if self._contextualizer is not None:
            embeddings = self._contextualizer(embeddings, mask=mask)

        encodings = self._encoder(embeddings, mask=mask)

        if self._feedforward is not None:
            encodings = self._feedforward(encodings)

        logits = self._classifier(encodings)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {
            "logits": logits,
            "probs": probs,
        }

        if label is not None:
            loss = self._loss(logits, label.long())
            output["loss"] = loss

            for metric in self._metrics:
                metric(probs, label)

        return output

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        metrics = {}
        for metric in self._metrics:
            metrics.update(metric.get_metrics(reset=reset))
        return metrics
