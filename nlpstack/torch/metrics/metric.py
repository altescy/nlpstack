from __future__ import annotations

from typing import Any, Iterable

import torch


class Metric:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


class ClassificationMetric(Metric):
    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, ) or (batch_size, num_classes)
        :param gold: (batch_size, )
        """
        raise NotImplementedError


class MultilabelClassificationMetric(Metric):
    def __call__(  # type: ignore[override]
        self,
        pred: torch.LongTensor | torch.FloatTensor,
        gold: torch.LongTensor,
    ) -> None:
        """
        :param pred: (batch_size, num_classes)
        :param gold: (batch_size, num_classes)
        """
        raise NotImplementedError
