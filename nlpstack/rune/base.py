from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, Mapping, Sequence, TypeVar

Self = TypeVar("Self", bound="Rune")

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")


class Rune(Generic[Example, Prediction]):
    def train(
        self: Self,
        train_dataset: Sequence[Example],
        valid_dataset: Sequence[Example] | None = None,
        resources: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def predict(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Prediction]:
        raise NotImplementedError

    def evaluate(self, dataset: Iterable[Example], **kwargs: Any) -> Mapping[str, float]:
        raise NotImplementedError
