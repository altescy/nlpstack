from __future__ import annotations

from typing import cast

import torch

from nlpstack.torch.metrics.metric import ClassificationMetric, MultilabelClassificationMetric


class Accuracy(ClassificationMetric):
    def __init__(self, topk: int = 1) -> None:
        self._topk = topk
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, ) or (batch_size, num_classes)
        :param gold: (batch_size, )
        """
        pred, gold = self.detach_tensors(pred, gold)  # type: ignore[assignment]
        if self._topk == 1:
            if pred.dim() == 2:
                pred = cast(torch.LongTensor, pred.argmax(dim=-1))
            self._correct_count += (pred == gold).sum().item()
            self._total_count += gold.size(0)
        else:
            if pred.dim() == 2:
                pred = cast(torch.LongTensor, pred.topk(self._topk, dim=-1)[1])
            self._correct_count += (pred == gold.unsqueeze(-1)).sum().item()
            self._total_count += gold.size(0)

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        if self._total_count == 0:
            return {"accuracy": 0.0}
        metrics = {"accuracy": self._correct_count / self._total_count}
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0


class MultilabelAccuracy(MultilabelClassificationMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._num_correct = 0.0
        self._num_total = 0.0

    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, num_classes)
        :param gold: (batch_size, num_classes)
        """
        pred, gold = self.detach_tensors(pred, gold)  # type: ignore[assignment]
        if pred.dtype in (torch.float, torch.double):
            pred = cast(torch.LongTensor, (pred > self._threshold).long())

        self._num_correct += (pred == gold).all(dim=1).sum().item()
        self._num_total += pred.size(0)

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        if self._num_total == 0:
            return {"accuracy": 0.0}
        metrics = {"accuracy": self._num_correct / self._num_total}
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._num_correct = 0.0
        self._num_total = 0.0


class OverallAccuracy(MultilabelClassificationMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, num_classes)
        :param gold: (batch_size, num_classes)
        """
        pred, gold = self.detach_tensors(pred, gold)  # type: ignore[assignment]
        if pred.dtype in (torch.float, torch.double):
            pred = cast(torch.LongTensor, (pred > self._threshold).to(dtype=gold.dtype))
        self._correct_count += (pred == gold).sum().item()
        self._total_count += gold.size(0) * gold.size(1)

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        metrics = {"overall_accuracy": self._correct_count / self._total_count if self._total_count > 0 else 0.0}
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0


class AverageAccuracy(MultilabelClassificationMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._correct_count: torch.Tensor | None = None
        self._total_count = 0.0

    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, num_classes)
        :param gold: (batch_size, num_classes)
        """
        pred, gold = self.detach_tensors(pred, gold)  # type: ignore[assignment]
        if pred.dtype in (torch.float, torch.double):
            pred = cast(torch.LongTensor, (pred > self._threshold).to(dtype=gold.dtype))

        if self._correct_count is None:
            self._correct_count = torch.zeros(gold.size(1), dtype=torch.float, device=gold.device)

        self._correct_count += (pred == gold).sum(dim=0)
        self._total_count += gold.size(0)

    def get_metrics(self, reset: bool = True) -> dict[str, float]:
        if self._correct_count is None or self._total_count == 0:
            return {"average_accuracy": 0.0}
        accuracies = self._correct_count / self._total_count
        metrics = {
            "average_accuracy": accuracies.mean().item(),
        }
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._correct_count = None
        self._total_count = 0.0
