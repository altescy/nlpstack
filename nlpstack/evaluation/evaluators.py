from typing import Any, Generic, Iterable, Mapping, TypeVar

from nlpstack.common import ProgressBar

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
        with ProgressBar(inferences, desc="Evaluating") as progressbar:
            for inference in progressbar:
                self.metric.update(inference)
                progressbar.set_postfix(**{key: f"{value:.2f}" for key, value in self.metric.compute().items()})
        return self.metric.compute()
