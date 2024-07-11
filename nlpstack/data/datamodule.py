from typing import Any, Generic, Iterable, Iterator, TypeVar

from collatable import Instance

from nlpstack.common import wrap_iterator

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class DataModule(Generic[Example, Inference, Prediction]):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def preprocess(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Example]:
        return wrap_iterator(iter, dataset)

    def build_instance(self, example: Example) -> Instance:
        raise NotImplementedError

    def build_predictions(self, inference: Inference) -> Iterator[Prediction]:
        raise NotImplementedError

    def read_dataset(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Instance]:
        raise NotImplementedError
