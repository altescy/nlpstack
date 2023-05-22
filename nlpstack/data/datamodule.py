from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, TypeVar

from collatable import Instance

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")


class DataModule(Generic[Example]):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def build_instance(self, example: Example) -> Instance:
        raise NotImplementedError

    def read_dataset(
        self,
        dataset: Iterable[Example],
        is_training: bool = False,
        **kwargs: Any,
    ) -> Iterator[Instance]:
        raise NotImplementedError
