from __future__ import annotations

from typing import cast

import torch

from nlp_learn.torch.metrics.metric import Metric


class Accuracy(Metric):
    def __init__(self) -> None:
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
        if pred.dim() == 2:
            pred = cast(torch.LongTensor, pred.argmax(dim=-1))
        self._correct_count += (pred == gold).sum().item()
        self._total_count += gold.size(0)

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        metrics = {"accuracy": self._correct_count / self._total_count}
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0
