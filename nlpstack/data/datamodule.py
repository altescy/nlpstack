from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

from collatable import Instance

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class DataModule(Generic[Example, Inference, Prediction]):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def build_instance(self, example: Example) -> Instance:
        raise NotImplementedError

    def build_predictions(self, inference: Inference) -> Iterator[Prediction]:
        raise NotImplementedError

    def build_inference(
        self,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
    ) -> Inference:
        raise NotImplementedError

    def read_dataset(
        self,
        dataset: Iterable[Example],
        is_training: bool = False,
        **kwargs: Any,
    ) -> Iterator[Instance]:
        raise NotImplementedError
