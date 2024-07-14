from functools import cached_property
from logging import getLogger
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, TypeVar, Union

from nlpstack.common import FileBackendSequence, ProgressBar
from nlpstack.data import Instance
from nlpstack.data.datamodule import DataModule
from nlpstack.evaluation import EmptyMetric, Evaluator, Metric, MultiMetrics, SimpleEvaluator
from nlpstack.integrations.torch.model import ModelInputs, TorchModel
from nlpstack.integrations.torch.picklable import TorchPicklable
from nlpstack.integrations.torch.predictor import TorchPredictor
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.util import set_random_seed
from nlpstack.rune.base import Rune, SetupMode

logger = getLogger(__name__)

Self = TypeVar("Self", bound="RuneForTorch")

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")
TorchModelParams = TypeVar("TorchModelParams")


class RuneForTorch(
    TorchPicklable,
    Generic[
        Example,
        Inference,
        Prediction,
        TorchModelParams,
    ],
    Rune[
        Example,
        Prediction,
        "RuneForTorch.SetupParams",
        TorchModelParams,
        TorchModelParams,
    ],
):
    cuda_dependent_attributes = ["model"]

    class SetupParams(NamedTuple):
        predictor: Optional[TorchPredictor.SetupParams] = None

    def __init__(
        self,
        *,
        datamodule: DataModule[Example, Inference, Prediction],
        model: TorchModel[Inference, ModelInputs, TorchModelParams],
        trainer: TorchTrainer,
        metric: Optional[Union[Metric[Inference], Sequence[Metric[Inference]]]] = None,
        predictor_factory: Callable[
            [
                DataModule[Example, Inference, Prediction],
                TorchModel[Inference, ModelInputs, TorchModelParams],
            ],
            TorchPredictor[Example, Inference, Prediction, TorchModelParams],
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
    def predictor(self) -> TorchPredictor[Example, Inference, Prediction, TorchModelParams]:
        return self._predictor_factory(self.datamodule, self.model)

    @cached_property
    def evaluator(self) -> Evaluator[Inference]:
        return self._evaluator_factory(self.metric)

    def setup(
        self,
        mode: SetupMode,
        params: Optional["RuneForTorch.SetupParams"] = None,
    ) -> None:
        params = params or RuneForTorch.SetupParams()
        if params.predictor is not None:
            self.predictor.setup(params.predictor)

    def train(
        self: Self,
        train_dataset: Sequence[Example],
        valid_dataset: Optional[Sequence[Example]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        if self.random_seed is not None:
            set_random_seed(self.random_seed)

        logger.info("Preprocessing training dataset...")
        train_dataset = FileBackendSequence.from_iterable(
            self.datamodule.preprocess(ProgressBar(train_dataset, desc="Preprocessing training dataset"))
        )
        if valid_dataset is not None:
            logger.info("Preprocessing validation dataset...")
            valid_dataset = FileBackendSequence.from_iterable(
                self.datamodule.preprocess(ProgressBar(valid_dataset, desc="Preprocessing validation dataset"))
            )

        logger.info("Setup datamodule...")
        self.datamodule.setup(dataset=train_dataset)

        logger.info("Setup model...")
        self.model.setup(datamodule=self.datamodule)

        logger.info("Setup metric...")
        self.metric.setup(datamodule=self.datamodule)

        logger.info("Reading training dataset...")
        train_instances: Sequence[Instance] = FileBackendSequence.from_iterable(
            self.datamodule.read_dataset(
                ProgressBar(train_dataset, desc="Reading training dataset"), skip_preprocess=True
            )
        )
        valid_instances: Optional[Sequence[Instance]] = None
        if valid_dataset is not None:
            logger.info("Reading validation dataset...")
            valid_instances = FileBackendSequence.from_iterable(
                self.datamodule.read_dataset(
                    ProgressBar(valid_dataset, desc="Reading validation dataset"), skip_preprocess=True
                )
            )

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
        params: Optional[TorchModelParams] = None,
    ) -> Iterator[Prediction]:
        yield from self.predictor.predict(dataset, params)

    def evaluate(
        self,
        dataset: Iterable[Example],
        params: Optional[TorchModelParams] = None,
    ) -> Mapping[str, float]:
        inferences = self.predictor.infer(dataset, params)
        return self.evaluator.evaluate(inferences)
