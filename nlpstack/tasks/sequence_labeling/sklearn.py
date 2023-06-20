import itertools
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.token_indexers import TokenIndexer
from nlpstack.data.tokenizers import Token
from nlpstack.sklearn.base import SklearnEstimatorForRune
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback

from .data import SequenceLabelingExample, SequenceLabelingPrediction
from .datamodules import SequenceLabelingDataModule
from .metrics import SequenceLabelingMetric
from .rune import BasicSequenceLabeler
from .torch import TorchSequenceLabeler

BasicInputsX = Sequence[Sequence[str]]
BasicInputsY = Sequence[Sequence[str]]
BasicOutputs = Sequence[Sequence[str]]


class SklearnBasicSequenceLabeler(
    SklearnEstimatorForRune[
        BasicInputsX,
        BasicInputsY,
        BasicOutputs,
        SequenceLabelingExample,
        SequenceLabelingPrediction,
    ]
):
    primary_metric = "token_accuracy"

    @staticmethod
    def input_builder(X: BasicInputsX, y: Optional[BasicInputsY]) -> Iterator[SequenceLabelingExample]:
        for tokens, labels in itertools.zip_longest(X, y or []):
            yield SequenceLabelingExample(
                text=[Token(surface) for surface in tokens],
                labels=labels,
            )

    @staticmethod
    def output_builder(predictions: Iterator[SequenceLabelingPrediction]) -> BasicOutputs:
        return [prediction.labels for prediction in predictions]

    def __init__(
        self,
        *,
        # data configuration
        min_df: Union[int, float, Mapping[str, Union[int, float]]] = 1,
        max_df: Union[int, float, Mapping[str, Union[int, float]]] = 1.0,
        pad_token: Union[str, Mapping[str, str]] = "@@PADDING@@",
        oov_token: Union[str, Mapping[str, str]] = "@@UNKNOWN@@",
        vocab: Optional[Vocabulary] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        datamodule: Optional[SequenceLabelingDataModule] = None,
        # model configuration
        classifier: Optional[TorchSequenceLabeler] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[SequenceLabelingMetric, Sequence[SequenceLabelingMetric]]] = None,
        **kwargs: Any,
    ) -> None:
        rune = BasicSequenceLabeler(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            vocab=vocab,
            token_indexers=token_indexers,
            classifier=classifier,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_callbacks=training_callbacks,
            trainer=trainer,
            metric=metric,
            **kwargs,
        )
        super().__init__(rune)
