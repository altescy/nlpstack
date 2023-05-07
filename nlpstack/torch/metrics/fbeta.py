from __future__ import annotations

from typing import cast

import torch

from nlpstack.torch.metrics.metric import MultilabelClassificationMetric


class MicroMultilabelFBeta(MultilabelClassificationMetric):
    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        self._beta = beta
        self._threshold = threshold
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0

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

        self._tp += ((pred == 1) & (gold == 1)).sum().item()
        self._fp += ((pred == 1) & (gold == 0)).sum().item()
        self._fn += ((pred == 0) & (gold == 1)).sum().item()

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        precision = self._tp / (self._tp + self._fp) if self._tp + self._fp > 0 else 0.0
        recall = self._tp / (self._tp + self._fn) if self._tp + self._fn > 0 else 0.0
        fbeta = (
            (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall + 1e-13)
            if precision + recall > 0
            else 0.0
        )
        if reset:
            self.reset()
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
    def __init__(self, beta: float = 1.0, threshold: float = 0.5) -> None:
        self._beta = beta
        self._threshold = threshold

        self._tp: torch.Tensor | None = None
        self._fp: torch.Tensor | None = None
        self._fn: torch.Tensor | None = None

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

        if self._tp is None:
            self._tp = torch.zeros(gold.size(1), dtype=torch.float, device=gold.device)
        if self._fp is None:
            self._fp = torch.zeros(gold.size(1), dtype=torch.float, device=gold.device)
        if self._fn is None:
            self._fn = torch.zeros(gold.size(1), dtype=torch.float, device=gold.device)

        self._tp += ((pred == 1) & (gold == 1)).sum(dim=0).float()
        self._fp += ((pred == 1) & (gold == 0)).sum(dim=0).float()
        self._fn += ((pred == 0) & (gold == 1)).sum(dim=0).float()

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        if self._tp is None or self._fp is None or self._fn is None:
            return {"macro_fbeta": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}
        precision = (self._tp / (self._tp + self._fp + 1e-13)).mean().item()
        recall = (self._tp / (self._tp + self._fn + 1e-13)).mean().item()
        fbeta = (
            (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall)
            if precision + recall > 0
            else 0.0
        )
        if reset:
            self.reset()
        return {
            "macro_fbeta": fbeta,
            "macro_precision": precision,
            "macro_recall": recall,
        }

    def reset(self) -> None:
        self._tp = None
        self._fp = None
        self._fn = None
