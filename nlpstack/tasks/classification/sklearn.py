from __future__ import annotations

import itertools
from typing import Any, Iterator, Mapping, Sequence

from nlpstack.data import Vocabulary
from nlpstack.data.token_indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.sklearn.base import SklearnEstimatorForRune
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback

from .data import ClassificationExample, ClassificationPrediction
from .datamodules import BasicClassificationDataModule
from .models import ClassificationObjective, TorchBasicClassifier
from .rune import BasicClassifier as BasicClassifier

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
    def input_builder(X: BasicInputsX, y: BasicInputsY | None) -> Iterator[ClassificationExample]:
        for text, label in itertools.zip_longest(X, y or []):
            yield ClassificationExample(text=text, label=label)

    @staticmethod
    def output_builder(predictions: Iterator[ClassificationPrediction]) -> BasicOutputs:
        return list(predictions)

    def __init__(
        self,
        *,
        # data configuration
        min_df: int | float | Mapping[str, int | float] = 1,
        max_df: int | float | Mapping[str, int | float] = 1.0,
        pad_token: str | Mapping[str, str] = "@@PADDING@@",
        oov_token: str | Mapping[str, str] = "@@UNKNOWN@@",
        vocab: Vocabulary | None = None,
        tokenizer: Tokenizer | None = None,
        token_indexers: Mapping[str, TokenIndexer] | None = None,
        datamodule: BasicClassificationDataModule | None = None,
        # model configuration
        objective: ClassificationObjective = "multiclass",
        classifier: TorchBasicClassifier | None = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Sequence[Callback] | None = None,
        trainer: TorchTrainer | None = None,
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
            objective=objective,
            classifier=classifier,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_callbacks=training_callbacks,
            trainer=trainer,
            **kwargs,
        )
        super().__init__(rune)
