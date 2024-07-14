from typing import Any, Generic, Iterable, Iterator, Mapping, Optional, Sequence, Type, TypeVar

from .types import EvaluationParams as EvaluationParamsType
from .types import Example as ExampleType
from .types import Prediction as PredictionType
from .types import PredictionParams as PredictionParamsType
from .types import SetupMode
from .types import SetupParams as SetupParamsType

Self = TypeVar("Self", bound="Rune")


class Rune(
    Generic[
        ExampleType,
        PredictionType,
        SetupParamsType,
        PredictionParamsType,
        EvaluationParamsType,
    ]
):
    Example: Type[ExampleType]
    Prediction: Type[PredictionType]
    SetupParams: Type[SetupParamsType]
    PredictionParams: Type[PredictionParamsType]
    EvaluationParams: Type[EvaluationParamsType]

    def train(
        self: Self,
        train_dataset: Sequence[ExampleType],
        valid_dataset: Optional[Sequence[ExampleType]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def setup(self, mode: SetupMode, params: Optional[SetupParamsType] = None) -> None:
        """Setup the rune for training, prediction or evaluation.

        This method is called before `train`, `predict` or `evaluate` is called.
        Please note that this method should be used for environment-dependent
        settings which not affect the model itself. For example, if you want
        to use GPU for inference, you can set the device here.
        """
        pass

    def predict(
        self,
        dataset: Iterable[ExampleType],
        params: Optional[PredictionParamsType] = None,
    ) -> Iterator[PredictionType]:
        raise NotImplementedError

    def evaluate(
        self,
        dataset: Iterable[ExampleType],
        params: Optional[EvaluationParamsType] = None,
    ) -> Mapping[str, float]:
        raise NotImplementedError
