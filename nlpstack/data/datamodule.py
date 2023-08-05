from typing import Any, Generic, Iterable, Iterator, TypeVar

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

    def read_dataset(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Instance]:
        raise NotImplementedError
