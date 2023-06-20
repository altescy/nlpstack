from typing import Dict

import numpy

from nlpstack.evaluation import Metric

from .data import SequenceLabelingInference


class SequenceLabelingMetric(Metric[SequenceLabelingInference]):
    """Metric for sequence labeling tasks."""


class TokenBasedAccuracy(SequenceLabelingMetric):
    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, inference: SequenceLabelingInference) -> None:
        assert inference.labels is not None

        if inference.mask is None:
            mask = numpy.ones(inference.labels.shape, dtype=bool)
        else:
            mask = inference.mask.astype(bool)

        if inference.decodings is None:
            pred = inference.probs.argmax(axis=-1)[mask]
        else:
            pred = numpy.array([label_index for decodings in inference.decodings for label_index in decodings[0]])

        gold = inference.labels[mask]

        self._correct += (pred == gold).sum()
        self._total += len(gold)

    def compute(self) -> Dict[str, float]:
        return {"token_accuracy": self._correct / self._total if self._total > 0 else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
