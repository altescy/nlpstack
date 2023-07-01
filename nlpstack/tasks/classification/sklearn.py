import itertools
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.sklearn.base import SklearnEstimatorForRune
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback

from .data import ClassificationExample, ClassificationPrediction
from .datamodules import BasicClassificationDataModule
from .metrics import ClassificationMetric
from .rune import BasicClassifier as BasicClassifier
from .torch import TorchBasicClassifier

BasicInputsX = Sequence[str]
BasicInputsY = Sequence[str]
BasicOutputs = Sequence[ClassificationPrediction]


class SklearnBasicClassifier(
    SklearnEstimatorForRune[
        BasicInputsX,
        BasicInputsY,
        BasicOutputs,
        ClassificationExample,
        ClassificationPrediction,
    ]
):
    primary_metric = "accuracy"

    @staticmethod
    def input_builder(X: BasicInputsX, y: Optional[BasicInputsY]) -> Iterator[ClassificationExample]:
        for text, label in itertools.zip_longest(X, y or []):
            yield ClassificationExample(text=text, label=label)

    @staticmethod
    def output_builder(predictions: Iterator[ClassificationPrediction]) -> BasicOutputs:
        return list(predictions)

    def __init__(
        self,
        *,
        # data configuration
        min_df: Union[int, float, Mapping[str, Union[int, float]]] = 1,
        max_df: Union[int, float, Mapping[str, Union[int, float]]] = 1.0,
        pad_token: Union[str, Mapping[str, str]] = "@@PADDING@@",
        oov_token: Union[str, Mapping[str, str]] = "@@UNKNOWN@@",
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        datamodule: Optional[BasicClassificationDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        class_weights: Optional[Union[Literal["balanced"], Mapping[str, float]]] = None,
        classifier: Optional[TorchBasicClassifier] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[ClassificationMetric, Sequence[ClassificationMetric]]] = None,
        **kwargs: Any,
    ) -> None:
        rune = BasicClassifier(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            vocab=vocab,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            dropout=dropout,
            class_weights=class_weights,
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
