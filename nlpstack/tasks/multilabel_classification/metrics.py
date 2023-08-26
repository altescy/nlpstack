from typing import Dict, Optional

import numpy

from nlpstack.evaluation import Metric

from .types import MultilabelClassificationInference


class MultilabelClassificationMetric(Metric[MultilabelClassificationInference]):
    """Metric for multilabel clasificatino tasks."""


class MultilabelAccuracy(MultilabelClassificationMetric):
    """
    The accuracy metric for multilabel classification.

    Args:
        threshold: The threshold for the prediction probabilities. If prediction probabilities are
            greater than or equal to the threshold, the label is predicted.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._threshold = threshold
        self._correct = 0
        self._total = 0

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        threshold = self._threshold or inference.threshold

        pred = inference.probs >= threshold
        gold = inference.labels.astype(bool)

        self._correct += (pred == gold).all(axis=1).sum()
        self._total += len(inference.labels)

    def compute(self) -> Dict[str, float]:
        return {"accuracy": self._correct / self._total if self._total > 0 else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class OverallAccuracy(MultilabelClassificationMetric):
    """
    The overall accuracy metric for multilabel classification.

    Args:
        threshold: The threshold for the prediction probabilities. If prediction probabilities are
            greater than or equal to the threshold, the label is predicted.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._threshold = threshold
        self._correct = 0
        self._total = 0

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        threshold = self._threshold or inference.threshold

        pred = inference.probs >= threshold
        gold = inference.labels.astype(bool)

        self._correct += (pred == gold).sum()
        self._total += gold.shape[0] * gold.shape[1]

    def compute(self) -> Dict[str, float]:
        return {"overall_accuracy": self._correct / self._total if self._total > 0 else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class AverageAccuracy(MultilabelClassificationMetric):
    """
    The average accuracy metric for multilabel classification.

    Args:
        threshold: The threshold for the prediction probabilities. If prediction probabilities are
            greater than or equal to the threshold, the label is predicted.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._threshold = threshold
        self._correct: Optional[numpy.ndarray] = None
        self._total = 0

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        threshold = self._threshold or inference.threshold

        pred = inference.probs >= threshold
        gold = inference.labels.astype(bool)

        if self._correct is None:
            self._correct = numpy.zeros(gold.shape[1], dtype=float)

        self._correct += (pred == gold).sum(axis=0)
        self._total += gold.shape[0]

    def compute(self) -> Dict[str, float]:
        if self._correct is None or self._total == 0:
            return {"average_accuracy": 0.0}
        return {"average_accuracy": float((self._correct / self._total).mean())}

    def reset(self) -> None:
        self._correct = None
        self._total = 0


class MicroMultilabelFBeta(MultilabelClassificationMetric):
    """
    The micro F-beta metric for multilabel classification.

    Args:
        beta: The beta parameter for F-beta.
        threshold: The threshold for the prediction probabilities. If prediction probabilities are
            greater than or equal to the threshold, the label is predicted.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: Optional[float] = None,
    ) -> None:
        self._beta = beta
        self._threshold = threshold
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        threshold = self._threshold or inference.threshold

        pred = inference.probs >= threshold
        gold = inference.labels.astype(bool)

        self._tp += (pred & gold).sum()
        self._fp += (pred & ~gold).sum()
        self._fn += (~pred & gold).sum()

    def compute(self) -> Dict[str, float]:
        precision = self._tp / (self._tp + self._fp) if self._tp + self._fp > 0 else 0.0
        recall = self._tp / (self._tp + self._fn) if self._tp + self._fn > 0 else 0.0
        fbeta = (
            (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall + 1e-13)
            if precision + recall > 0
            else 0.0
        )
        return {
            "micro_fbeta": fbeta,
            "micro_precision": precision,
            "micro_recall": recall,
        }

    def reset(self) -> None:
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0


class MacroMultilabelFBeta(MultilabelClassificationMetric):
    """
    The macro F-beta metric for multilabel classification.

    Args:
        beta: The beta parameter for F-beta.
        threshold: The threshold for the prediction probabilities. If prediction probabilities are
            greater than or equal to the threshold, the label is predicted.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: Optional[float] = None,
    ) -> None:
        self._beta = beta
        self._threshold = threshold

        self._tp: Optional[numpy.ndarray] = None
        self._fp: Optional[numpy.ndarray] = None
        self._fn: Optional[numpy.ndarray] = None

    def update(self, inference: MultilabelClassificationInference) -> None:
        assert inference.labels is not None
        assert inference.probs.shape == inference.labels.shape

        threshold = self._threshold or inference.threshold

        pred = inference.probs >= threshold
        gold = inference.labels.astype(bool)

        if self._tp is None:
            self._tp = numpy.zeros(gold.shape[1], dtype=float)
        if self._fp is None:
            self._fp = numpy.zeros(gold.shape[1], dtype=float)
        if self._fn is None:
            self._fn = numpy.zeros(gold.shape[1], dtype=float)

        self._tp += (pred & gold).sum(axis=0).astype(float)
        self._fp += (pred & ~gold).sum(axis=0).astype(float)
        self._fn += (~pred & gold).sum(axis=0).astype(float)

    def compute(self) -> Dict[str, float]:
        if self._tp is None or self._fp is None or self._fn is None:
            return {"macro_fbeta": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}
        precision = float((self._tp / (self._tp + self._fp + 1e-13)).mean())
        recall = float((self._tp / (self._tp + self._fn + 1e-13)).mean())
        fbeta = (
            (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {
            "macro_fbeta": fbeta,
            "macro_precision": precision,
            "macro_recall": recall,
        }

    def reset(self) -> None:
        self._tp = None
        self._fp = None
        self._fn = None
