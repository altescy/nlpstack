from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

from sklean.base import BaseEstimator

from nlpstack.data import Dataset, Instance
from nlpstack.data.datamodule import DataModule
from nlpstack.torch.models import Model
from nlpstack.torch.predictor import TorchPredictor
from nlpstack.torch.training import Trainer

Self = TypeVar("Self", bound="BaseEstimatorForTorch")

InputsX = TypeVar("InputsX")
InputsY = TypeVar("InputsY")
Outputs = TypeVar("Outputs")
Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class BaseEstimatorForTorch(
    BaseEstimator,
    Generic[
        InputsX,
        InputsY,
        Outputs,
        Example,
        Inference,
        Prediction,
    ],
):  # type: ignore[misc]
    def __init__(
        self,
        *,
        datamodule: DataModule[Example, Inference, Prediction],
        model: Model[Any, Inference],
        trainer: Trainer,
        input_builder: Callable[[InputsX, Optional[InputsY]], Iterator[Example]],
        output_builder: Callable[[Iterator[Prediction]], Outputs],
        predictor_factory: Callable[
            [DataModule[Example, Inference, Prediction], Model[Any, Inference]],
            TorchPredictor[Example, Inference, Prediction],
        ] = TorchPredictor,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.model = model
        self.trainer = trainer
        self.kwargs = kwargs

        self._predictor_factory = predictor_factory

        self._input_builder = input_builder
        self._output_builder = output_builder

    @cached_property
    def _predictor(self) -> TorchPredictor[Example, Inference, Prediction]:
        return self._predictor_factory(self.datamodule, self.model)

    def _read_dataset(
        self,
        X: InputsX,
        y: InputsY | None,
        *,
        is_training: bool = False,
    ) -> Iterator[Instance]:
        dataset = self._input_builder(X, y)
        yield from self.datamodule.read_dataset(dataset, is_training=is_training)

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
        self.datamodule.setup(**self.kwargs, **kwargs)
        self.model.setup(datamodule=self.datamodule, **self.kwargs, **kwargs)

        train_dataset = Dataset.from_iterable(self._read_dataset(X, y, is_training=True))
        valid_dataset: Dataset | None = None
        if X_valid is not None and y_valid is not None:
            valid_dataset = Dataset.from_iterable(self._read_dataset(X_valid, y_valid))

        self.trainer.train(
            model=self.model,
            train=train_dataset,
            valid=valid_dataset,
            resources=resources,
        )

        return self

    def predict(self, X: InputsX, **kwargs: Any) -> Outputs:
        dataset = self._build_examples(X, None)
        predictions = self._predictor.predict(dataset, **kwargs)
        return self._output_builder(predictions)
