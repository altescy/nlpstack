from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

from sklearn.base import BaseEstimator

from nlpstack.data import Dataset
from nlpstack.rune import Rune

Self = TypeVar("Self", bound="SklearnEstimatorForRune")

InputsX = TypeVar("InputsX")
InputsY = TypeVar("InputsY")
Outputs = TypeVar("Outputs")
Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class SklearnEstimatorForRune(
    BaseEstimator,  # type: ignore[misc]
    Generic[
        InputsX,
        InputsY,
        Outputs,
        Example,
        Prediction,
    ],
):
    primary_metric: str
    input_builder: Callable[[InputsX, Optional[InputsY]], Iterator[Example]]
    output_builder: Callable[[Iterator[Prediction]], Outputs]

    def __init__(self, rune: Rune[Example, Prediction]) -> None:
        self._rune = rune

    @property
    def rune(self) -> Rune[Example, Prediction]:
        return self._rune

    def fit(
        self: Self,
        X: InputsX,
        y: InputsY | None = None,
        *,
        X_valid: InputsX | None = None,
        y_valid: InputsY | None = None,
        resources: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        train_dataset = Dataset.from_iterable(self.input_builder(X, y))
        valid_dataset: Dataset[Example] | None = None
        if X_valid is not None and y_valid is not None:
            valid_dataset = Dataset.from_iterable(self.input_builder(X_valid, y_valid))
        self._rune.train(train_dataset, valid_dataset, resources, **kwargs)
        return self

    def predict(self, X: InputsX, **kwargs: Any) -> Outputs:
        return self.output_builder(self.generate_predictions(X, **kwargs))

    def score(self, X: InputsX, y: InputsY, *, metric: str | None = None, **kwargs: Any) -> float:
        return self.compute_metrics(X, y, **kwargs)[metric or self.primary_metric]

    def generate_predictions(self, X: InputsX, **kwargs: Any) -> Iterator[Prediction]:
        dataset = self.input_builder(X, None)
        yield from self._rune.predict(dataset, **kwargs)

    def compute_metrics(self, X: InputsX, y: InputsY, **kwargs: Any) -> dict[str, float]:
        dataset = self.input_builder(X, y)
        return self._rune.evaluate(dataset, **kwargs)
