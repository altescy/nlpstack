from __future__ import annotations

from typing import Any, Mapping, Sequence, Union, cast

import torch

from nlpstack.data import Vocabulary
from nlpstack.torch.metrics import (
    Accuracy,
    AverageAccuracy,
    ClassificationMetric,
    MultilabelClassificationMetric,
    OverallAccuracy,
)
from nlpstack.torch.models.model import Model
from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.lazy import LazyLinearOutput
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.util import get_mask_from_text

ClassificationMetrics = Union[Sequence[ClassificationMetric], Sequence[MultilabelClassificationMetric]]


class TorchBasicClassifier(Model):
    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Seq2SeqEncoder | None = None,
        feedforward: FeedForward | None = None,
        metrics: ClassificationMetrics | None = None,
        dropout: float | None = None,
        multilabel: bool = False,
        label_namespace: str = "labels",
    ) -> None:
        if metrics:
            if multilabel:
                assert all(isinstance(metric, MultilabelClassificationMetric) for metric in metrics)
            else:
                assert all(isinstance(metric, ClassificationMetric) for metric in metrics)

        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._classifier = LazyLinearOutput(
            encoder.get_output_dim() if feedforward is None else feedforward.get_output_dim()
        )

        self._contextualizer = contextualizer
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None

        self._loss = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
        self._metrics = metrics or ([OverallAccuracy(), AverageAccuracy()] if multilabel else [Accuracy()])  # type: ignore[list-item]

        self._multilabel = multilabel
        self._label_namespace = label_namespace

    @property
    def multilabel(self) -> bool:
        return self._multilabel

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        super().setup(*args, vocab=vocab, **kwargs)
        num_labels = vocab.get_vocab_size(self._label_namespace)
        self._classifier.initialize_parameters(out_features=num_labels)

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

        if self._dropout is not None:
            encodings = self._dropout(encodings)

        logits = cast(torch.FloatTensor, self._classifier(encodings))
        if self._multilabel:
            probs = cast(torch.FloatTensor, torch.sigmoid(logits))
        else:
            probs = cast(torch.FloatTensor, torch.nn.functional.softmax(logits, dim=-1))

        output = {
            "logits": logits,
            "probs": probs,
        }

        if label is not None:
            label_for_task = label.float() if self._multilabel else label.long()
            loss = self._loss(logits, label_for_task)
            output["loss"] = loss

            for metric in self._metrics:
                metric(probs, label)

        return output

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        metrics = {}
        for metric in self._metrics:
            metrics.update(metric.get_metrics(reset=reset))
        return metrics
