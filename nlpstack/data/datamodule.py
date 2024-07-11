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

    def read_dataset(
        self,
        dataset: Iterable[Example],
        *,
        skip_preprocess: bool = False,
    ) -> Iterator[Instance]:
        """
        Read the dataset and return a generator of instances.

        Args:
            dataset: The dataset to read.
            skip_preprocess: Whether to skip the preprocessing step. If this is
            set to `True`, the `preprocess` method will not be called on the
            dataset. Defaults to `False`.

        Returns:
            A generator of instances.
        """

        if not skip_preprocess:
            dataset = self.preprocess(dataset)
        for example in dataset:
            yield self.build_instance(example)
