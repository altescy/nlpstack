from __future__ import annotations

from collections import deque
from functools import cached_property
from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

from nlpstack.data import Dataset, Instance
from nlpstack.data.datamodule import DataModule
from nlpstack.torch.model import TorchModel
from nlpstack.torch.picklable import TorchPicklable
from nlpstack.torch.predictor import TorchPredictor
from nlpstack.torch.training import TorchTrainer

from .base import Rune

Self = TypeVar("Self", bound="RuneForTorch")

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class RuneForTorch(
    TorchPicklable,
    Generic[Example, Inference, Prediction],
    Rune[Example, Prediction],
):
    cuda_dependent_attributes = ["model"]

    def __init__(
        self,
        *,
        datamodule: DataModule[Example, Inference, Prediction],
        model: TorchModel[Any, Inference],
        trainer: TorchTrainer,
        predictor_factory: Callable[
            [DataModule[Example, Inference, Prediction], TorchModel[Any, Inference]],
            TorchPredictor[Example, Inference, Prediction],
        ] = TorchPredictor,
        **kwargs: Any,
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.trainer = trainer
        self.kwargs = kwargs

        self._predictor_factory = predictor_factory

    @cached_property
    def predictor(self) -> TorchPredictor[Example, Inference, Prediction]:
        return self._predictor_factory(self.datamodule, self.model)

    def train(
        self: Self,
        train_dataset: Sequence[Example],
        valid_dataset: Sequence[Example] | None = None,
        resources: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        self.datamodule.setup(**self.kwargs, **kwargs)

        train_instances = Dataset.from_iterable(self.datamodule.read_dataset(train_dataset, is_training=True))
        valid_instances: Dataset[Instance] | None = None
        if valid_dataset is not None:
            valid_instances = Dataset.from_iterable(self.datamodule.read_dataset(valid_dataset))

        self.model.setup(datamodule=self.datamodule, **self.kwargs, **kwargs)

        self.trainer.train(
            model=self.model,
            train=train_instances,
            valid=valid_instances,
            resources=resources,
        )

        return self

    def predict(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Prediction]:
        yield from self.predictor.predict(dataset, **kwargs)

    def evaluate(self, dataset: Iterable[Example], **kwargs: Any) -> dict[str, float]:
        self.model.get_metrics(reset=True)
        deque(self.predictor.predict(dataset, **kwargs), maxlen=0)
        return self.model.get_metrics()
