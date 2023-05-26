from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, Mapping, TypeVar

from nlpstack.data.datamodule import DataModule

from .metrics import Metric

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class Evaluator(Generic[Example, Prediction]):
    def evaluate(
        self,
        examples: Iterable[Example],
        predictions: Iterable[Prediction],
        **kwargs: Any,
    ) -> Mapping[str, float]:
        raise NotImplementedError


class SimpleEvaluator(
    Generic[Example, Inference, Prediction],
    Evaluator[Example, Prediction],
):
    def __init__(
        self,
        datamodule: DataModule[Example, Inference, Prediction],
        metric: Metric[Inference],
    ) -> None:
        self.datamodule = datamodule
        self.metric = metric

    def evaluate(
        self,
        examples: Iterable[Example],
        predictions: Iterable[Prediction],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Mapping[str, float]:
        self.metric.reset()

        def inference_iterator() -> Iterator[Inference]:
            example_batch: list[Example] = []
            prediction_batch: list[Prediction] = []
            for example, prediction in zip(examples, predictions):
                example_batch.append(example)
                prediction_batch.append(prediction)
                if len(example_batch) == batch_size:
                    yield self.datamodule.build_inference(example_batch, prediction_batch)
            if example_batch:
                yield self.datamodule.build_inference(example_batch, prediction_batch)

        for inference in inference_iterator():
            self.metric.update(inference)

        return self.metric.compute()
