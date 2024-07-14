from typing import Any, Callable, Generic, Iterator, Mapping, Optional, Sequence, TypeVar

from sklearn.base import BaseEstimator

from nlpstack.common import FileBackendSequence
from nlpstack.rune import Rune, SetupMode

Self = TypeVar("Self", bound="SklearnEstimatorForRune")

InputsX = TypeVar("InputsX")
InputsY = TypeVar("InputsY")
Outputs = TypeVar("Outputs")
Example = TypeVar("Example")
Prediction = TypeVar("Prediction")
SetupParams = TypeVar("SetupParams")
PredictionParams = TypeVar("PredictionParams")
EvaluationParams = TypeVar("EvaluationParams")


class SklearnEstimatorForRune(
    BaseEstimator,  # type: ignore[misc]
    Generic[
        InputsX,
        InputsY,
        Outputs,
        Example,
        Prediction,
        SetupParams,
        PredictionParams,
        EvaluationParams,
    ],
):
    primary_metric: str
    input_builder: Callable[[InputsX, Optional[InputsY]], Iterator[Example]]
    output_builder: Callable[[Iterator[Prediction]], Outputs]

    def __init__(
        self,
        rune: Rune[Example, Prediction, SetupParams, PredictionParams, EvaluationParams],
    ) -> None:
        self._rune = rune

    @property
    def rune(self) -> Rune[Example, Prediction, SetupParams, PredictionParams, EvaluationParams]:
        return self._rune

    def setup(self, mode: SetupMode, params: Optional[SetupParams] = None) -> None:
        self._rune.setup(mode, params)

    def fit(
        self: Self,
        X: InputsX,
        y: Optional[InputsY] = None,
        *,
        X_valid: Optional[InputsX] = None,
        y_valid: Optional[InputsY] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> Self:
        train_dataset: Sequence[Example] = FileBackendSequence.from_iterable(self.input_builder(X, y))
        valid_dataset: Optional[Sequence[Example]] = None
        if X_valid is not None and y_valid is not None:
            valid_dataset = FileBackendSequence.from_iterable(self.input_builder(X_valid, y_valid))
        self._rune.setup("training")
        self._rune.train(train_dataset, valid_dataset, resources)
        return self

    def predict(self, X: InputsX, **kwargs: Any) -> Outputs:
        return self.output_builder(self.generate_predictions(X, **kwargs))

    def score(self, X: InputsX, y: InputsY, *, metric: Optional[str] = None, **kwargs: Any) -> float:
        return self.compute_metrics(X, y, **kwargs)[metric or self.primary_metric]

    def generate_predictions(self, X: InputsX, params: Optional[PredictionParams] = None) -> Iterator[Prediction]:
        dataset = self.input_builder(X, None)
        self._rune.setup("prediction")
        yield from self._rune.predict(dataset, params)

    def compute_metrics(self, X: InputsX, y: InputsY, params: Optional[EvaluationParams] = None) -> Mapping[str, float]:
        dataset = self.input_builder(X, y)
        self._rune.setup("evaluation")
        return self._rune.evaluate(dataset, params)
