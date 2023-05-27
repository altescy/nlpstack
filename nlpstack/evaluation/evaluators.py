from __future__ import annotations

from typing import Any, Generic, Iterable, Mapping, TypeVar

from .metrics import Metric

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class Evaluator(Generic[Inference]):
    def evaluate(
        self,
        inferences: Iterable[Inference],
        **kwargs: Any,
    ) -> Mapping[str, float]:
        raise NotImplementedError


class SimpleEvaluator(Generic[Inference], Evaluator[Inference]):
    def __init__(self, metric: Metric[Inference]) -> None:
        self.metric = metric

    def evaluate(
        self,
        inferences: Iterable[Inference],
        **kwargs: Any,
    ) -> Mapping[str, float]:
        self.metric.reset()
        for inference in inferences:
            self.metric.update(inference)
        return self.metric.compute()
