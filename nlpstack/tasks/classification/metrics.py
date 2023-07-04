from typing import Dict, Literal, Optional, Sequence, Union

import numpy

from nlpstack.evaluation import Metric

from .types import ClassificationInference


class ClassificationMetric(Metric[ClassificationInference]):
    """Metric for classification tasks."""


class Accuracy(ClassificationMetric):
    def __init__(self, topk: int = 1) -> None:
        self.topk = topk
        self._correct = 0
        self._total = 0

    def update(self, inference: ClassificationInference) -> None:
        assert inference.labels is not None
        topk = inference.probs.argsort(axis=1)[:, -self.topk :]
        self._correct += (inference.labels[:, None] == topk).any(axis=1).sum()
        self._total += len(inference.labels)

    def compute(self) -> Dict[str, float]:
        return {"accuracy": self._correct / self._total if self._total else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


FBetaAverage = Literal["micro", "macro"]


class FBeta(ClassificationMetric):
    def __init__(
        self,
        beta: float = 1.0,
        average: Union[FBetaAverage, Sequence[FBetaAverage]] = "macro",
        topk: Optional[int] = None,
    ) -> None:
        self.beta = beta
        self.average = (average,) if isinstance(average, str) else average
        self.topk = topk
        self._true_positive: Optional[numpy.ndarray] = None
        self._false_positive: Optional[numpy.ndarray] = None
        self._false_negative: Optional[numpy.ndarray] = None

        if not set(self.average) <= {"micro", "macro"}:
            raise ValueError(f"Invalid average: {self.average}")

    def update(self, inference: ClassificationInference) -> None:
        assert inference.labels is not None
        if self.topk is None:
            prediction = inference.probs.argmax(axis=1, keepdims=True)  # Shape: (batch_size, 1)
        else:
            prediction = inference.probs.argsort(axis=1)[:, -self.topk :]  # Shape: (batch_size, topk)

        num_classes = inference.probs.shape[1]

        if self._true_positive is None:
            self._true_positive = numpy.zeros(num_classes, dtype=numpy.int64)
        if self._false_positive is None:
            self._false_positive = numpy.zeros(num_classes, dtype=int)
        if self._false_negative is None:
            self._false_negative = numpy.zeros(num_classes, dtype=int)

        for i in range(num_classes):
            self._true_positive[i] += ((inference.labels == i) & (prediction == i).any(axis=1)).sum()
            self._false_positive[i] += ((inference.labels != i) & (prediction == i).any(axis=1)).sum()
            self._false_negative[i] += ((inference.labels == i) & (prediction != i).all(axis=1)).sum()

    def compute(self) -> Dict[str, float]:
        if self._true_positive is None or self._false_positive is None or self._false_negative is None:
            return {"fbeta": 0.0, "precision": 0.0, "recall": 0.0}

        metrics: Dict[str, float] = {}
        if "macro" in self.average:
            precision = (self._true_positive / (self._true_positive + self._false_positive + 1e-13)).mean()
            recall = (self._true_positive / (self._true_positive + self._false_negative + 1e-13)).mean()
            fbeta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-13)
            metrics["macro_fbeta"] = fbeta
            metrics["macro_precision"] = precision
            metrics["macro_recall"] = recall
        if "micro" in self.average:
            precision = self._true_positive.sum() / (self._true_positive.sum() + self._false_positive.sum() + 1e-13)
            recall = self._true_positive.sum() / (self._true_positive.sum() + self._false_negative.sum() + 1e-13)
            fbeta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-13)
            metrics["micro_fbeta"] = fbeta
            metrics["micro_precision"] = precision
            metrics["micro_recall"] = recall

        return metrics

    def reset(self) -> None:
        self._true_positive = None
        self._false_positive = None
        self._false_negative = None
