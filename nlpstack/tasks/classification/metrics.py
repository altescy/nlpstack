from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy
import sklearn.metrics

from nlpstack.evaluation import Metric

from .datamodules import BasicClassificationDataModule
from .types import ClassificationInference


class ClassificationMetric(Metric[ClassificationInference]):
    """Metric for classification tasks."""


class Accuracy(ClassificationMetric):
    """
    Accuracy metric for classification tasks.

    Args:
        topk: The top-k accuracy. Defaults to `1`.
    """

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
    """
    F-beta score metric for classification tasks.

    Args:
        beta: The beta value. Defaults to `1.0`.
        average: The averaging method. Defaults to `"macro"`.
        topk: The top-k accuracy. Defaults to `None`.
    """

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


class PrecisionRecallAuc(ClassificationMetric):  # type: ignore[misc]
    """
    Precision-recall AUC metric for classification tasks.

    Args:
        positive_label: The positive label. Defaults to `"1"`.
        label_namespace: The label namespace. Defaults to `"labels"`.
    """

    def __init__(
        self,
        positive_label: str,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()
        self._positive_label = positive_label
        self._positive_label_index: Optional[int] = None
        self._label_namespace = label_namespace

        self._all_predictions: List[float] = []
        self._all_gold_labels: List[int] = []

    def setup(self, *args: Any, datamodule: BasicClassificationDataModule, **kwargs: Any) -> None:
        self._positive_label_index = datamodule.vocab.get_index_by_token(self._label_namespace, self._positive_label)

    def update(self, inference: ClassificationInference) -> None:
        assert inference.labels is not None
        assert self._positive_label_index is not None

        self._all_predictions.extend(inference.probs[:, self._positive_label_index].tolist())
        self._all_gold_labels.extend((inference.labels == self._positive_label_index).tolist())

    def compute(self) -> Dict[str, float]:
        if not self._all_gold_labels:
            return {"pr_auc": 0.0}
        precisions, recalls, _ = sklearn.metrics.precision_recall_curve(
            self._all_gold_labels,
            self._all_predictions,
        )
        auc = float(sklearn.metrics.auc(recalls, precisions))
        return {"pr_auc": auc}

    def reset(self) -> None:
        self._all_predictions = []
        self._all_gold_labels = []
