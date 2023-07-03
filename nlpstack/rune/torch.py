from functools import cached_property
from logging import getLogger
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

from nlpstack.common import FileBackendSequence
from nlpstack.data import Instance
from nlpstack.data.datamodule import DataModule
from nlpstack.evaluation import EmptyMetric, Evaluator, Metric, MultiMetrics, SimpleEvaluator
from nlpstack.torch.model import TorchModel
from nlpstack.torch.picklable import TorchPicklable
from nlpstack.torch.predictor import TorchPredictor
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.util import set_random_seed

from .base import Rune, SetupMode

logger = getLogger(__name__)

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
        model: TorchModel[Inference],
        trainer: TorchTrainer,
        metric: Optional[Union[Metric[Inference], Sequence[Metric[Inference]]]] = None,
        predictor_factory: Callable[
            [DataModule[Example, Inference, Prediction], TorchModel[Inference]],
            TorchPredictor[Example, Inference, Prediction],
        ] = TorchPredictor,
        evaluator_factory: Callable[
            [Metric[Inference]],
            Evaluator[Inference],
        ] = SimpleEvaluator,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.trainer = trainer
        self.kwargs = kwargs

        self.random_seed = random_seed

        self.metric: Metric[Inference]
        if metric is None:
            self.metric = EmptyMetric()
        elif isinstance(metric, Sequence):
            self.metric = MultiMetrics(metric)
        else:
            self.metric = metric

        self._predictor_factory = predictor_factory
        self._evaluator_factory = evaluator_factory

    @cached_property
    def predictor(self) -> TorchPredictor[Example, Inference, Prediction]:
        return self._predictor_factory(self.datamodule, self.model)

    @cached_property
    def evaluator(self) -> Evaluator[Inference]:
        return self._evaluator_factory(self.metric)

    def setup(self, mode: SetupMode, **kwargs: Any) -> None:
        if mode in ("prediction", "evaluation"):
            self.predictor.setup(**kwargs)

    def train(
        self: Self,
        train_dataset: Sequence[Example],
        valid_dataset: Optional[Sequence[Example]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        if self.random_seed is not None:
            set_random_seed(self.random_seed)

        logger.info("Setup datamodule...")
        self.datamodule.setup(**self.kwargs, **kwargs)

        logger.info("Reading training dataset...")
        train_instances: Sequence[Instance] = FileBackendSequence.from_iterable(
            self.datamodule.read_dataset(train_dataset, is_training=True)
        )
        valid_instances: Optional[Sequence[Instance]] = None
        if valid_dataset is not None:
            logger.info("Reading validation dataset...")
            valid_instances = FileBackendSequence.from_iterable(self.datamodule.read_dataset(valid_dataset))

        logger.info("Setup model...")
        self.model.setup(datamodule=self.datamodule, **self.kwargs, **kwargs)
        self.metric.setup(datamodule=self.datamodule, **self.kwargs, **kwargs)

        logger.info("Start training...")
        self.trainer.train(
            model=self.model,
            train=train_instances,
            valid=valid_instances,
            metric=self.metric,
            resources=resources,
        )

        return self

    def predict(
        self,
        dataset: Iterable[Example],
        **kwargs: Any,
    ) -> Iterator[Prediction]:
        yield from self.predictor.predict(dataset, **kwargs)

    def evaluate(
        self,
        dataset: Iterable[Example],
        **kwargs: Any,
    ) -> Mapping[str, float]:
        inferences = self.predictor.infer(dataset, **kwargs)
        return self.evaluator.evaluate(inferences, **kwargs)
