from typing import Dict

from nlpstack.evaluation import Metric

from .data import MultilabelClassificationInference


class MultilabelClassificationMetric(Metric[MultilabelClassificationInference]):
    """Metric for multilabel clasificatino tasks."""


class MultilabelAccuracy(MultilabelClassificationMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._correct = 0
        self._total = 0

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        pred = inference.probs >= self._threshold
        gold = inference.labels.astype(bool)

        self._correct += (pred == gold).all(axis=1).sum()
        self._total += len(inference.labels)

    def compute(self) -> Dict[str, float]:
        return {"accuracy": self._correct / self._total if self._total > 0 else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
