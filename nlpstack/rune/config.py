import dataclasses
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Optional, Type, cast

from nlpstack.common import FromJsonnet

from .base import Rune
from .types import EvaluationParams, Example, Prediction, PredictionParams, SetupParams


@dataclasses.dataclass
class RuneConfig(FromJsonnet, Generic[Example, Prediction]):
    model: Optional[Rune[Example, Prediction, Any, Any, Any]] = None
    reader: Optional[Callable[[str], Iterator[Example]]] = None
    writer: Optional[Callable[[str, Iterable[Prediction]], None]] = None
    setup: Optional[Mapping[str, Any]] = None
    predictor: Optional[Mapping[str, Any]] = None
    evaluator: Optional[Mapping[str, Any]] = None
    train_dataset_filename: Optional[str] = None
    valid_dataset_filename: Optional[str] = None
    test_dataset_filename: Optional[str] = None

    def get_setup_params(self, cls: Type[SetupParams]) -> Optional[SetupParams]:
        if self.setup is None:
            return None
        return cast(SetupParams, RuneConfig.__COLT_BUILDER__(self.setup, cls))

    def get_prediction_params(self, cls: Type[PredictionParams]) -> Optional[PredictionParams]:
        if self.predictor is None:
            return None
        return cast(PredictionParams, RuneConfig.__COLT_BUILDER__(self.predictor, cls))

    def get_evaluation_params(self, cls: Type[EvaluationParams]) -> Optional[EvaluationParams]:
        if self.evaluator is None:
            return None
        return cast(EvaluationParams, RuneConfig.__COLT_BUILDER__(self.evaluator, cls))
