import itertools
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.sklearn.rune import SklearnEstimatorForRune
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback

from .datamodules import MultilabelClassificationDataModule
from .metrics import MultilabelClassificationMetric
from .rune import MultilabelClassifier
from .torch import TorchMultilabelClassifier
from .types import MultilabelClassificationExample, MultilabelClassificationPrediction

BasicInputsX = Sequence[str]
BasicInputsY = Sequence[Sequence[str]]
BasicOutputs = Sequence[Sequence[str]]


class SklearnMultilabelClassifier(
    SklearnEstimatorForRune[
        BasicInputsX,
        BasicInputsY,
        BasicOutputs,
        MultilabelClassificationExample,
        MultilabelClassificationPrediction,
    ]
):
    primary_metric = "accuracy"

    @staticmethod
    def input_builder(X: BasicInputsX, y: Optional[BasicInputsY]) -> Iterator[MultilabelClassificationExample]:
        for text, labels in itertools.zip_longest(X, y or []):
            yield MultilabelClassificationExample(text=text, labels=labels)

    @staticmethod
    def output_builder(predictions: Iterator[MultilabelClassificationPrediction]) -> BasicOutputs:
        return [pred.top_labels for pred in predictions]

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
        datamodule: Optional[MultilabelClassificationDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        class_weights: Optional[Union[Literal["balanced"], Mapping[str, float]]] = None,
        classifier: Optional[TorchMultilabelClassifier] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[MultilabelClassificationMetric, Sequence[MultilabelClassificationMetric]]] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        rune = MultilabelClassifier(
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
            random_seed=random_seed,
            **kwargs,
        )
        super().__init__(rune)
