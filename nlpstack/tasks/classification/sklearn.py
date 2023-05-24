from __future__ import annotations

import itertools
import warnings
from typing import Any, Iterator, Mapping, Sequence

from nlpstack.data import DataLoader, Vocabulary
from nlpstack.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.sklearn.base import BaseEstimatorForTorch
from nlpstack.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory

from .data import ClassificationExample, ClassificationInference, ClassificationPrediction
from .datamodules import BasicClassificationDataModule
from .models import ClassificationObjective, TorchBasicClassifier

BasicInputsX = Sequence[str]
BasicInputsY = Sequence[str]
BasicOutputs = Sequence[ClassificationPrediction]


class BasicClassifier(
    BaseEstimatorForTorch[
        BasicInputsX,
        BasicInputsY,
        BasicOutputs,
        ClassificationExample,
        ClassificationInference,
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
        if datamodule is None:
            if vocab is None:
                default_token_namespace = "tokens"
                min_df = {default_token_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
                max_df = {default_token_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
                pad_token = {default_token_namespace: pad_token} if isinstance(pad_token, str) else pad_token
                oov_token = {default_token_namespace: oov_token} if isinstance(oov_token, str) else oov_token
                special_tokens: dict[str, set[str]] = {}
                for namespace, token in pad_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                for namespace, token in oov_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                vocab = Vocabulary(
                    min_df=min_df,
                    max_df=max_df,
                    pad_token=pad_token,
                    oov_token=oov_token,
                    special_tokens=special_tokens,
                )
            else:
                if (min_df, max_df, pad_token, oov_token) != (1, 1.0, "@@PADDING@@", "@@UNKNOWN@@"):
                    warnings.warn(
                        "Ignoring min_df, max_df, pad_token, and oov_token because vocab is given.",
                        UserWarning,
                    )

            if tokenizer is None:
                tokenizer = WhitespaceTokenizer()

            if token_indexers is None:
                token_indexers = {"tokens": SingleIdTokenIndexer()}

            datamodule = BasicClassificationDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if classifier is None:
            classifier = TorchBasicClassifier(
                embedder=TextEmbedder({"tokens": Embedding(64)}),
                encoder=BagOfEmbeddings(64),
                objective=objective,
            )
        else:
            if objective != classifier.objective:
                warnings.warn(
                    f"Ignoring objective={objective} because classifier (objective={classifier.objective}) is given.",
                    UserWarning,
                )

        if trainer is None:
            trainer = TorchTrainer(
                train_dataloader=DataLoader(batch_size=batch_size, shuffle=True),
                valid_dataloader=DataLoader(batch_size=batch_size, shuffle=False),
                max_epochs=max_epochs,
                optimizer_factory=AdamFactory(lr=learning_rate),
                callbacks=training_callbacks,
            )
        else:
            if (max_epochs, batch_size, learning_rate, training_callbacks) != (4, 32, 1e-3, None):
                warnings.warn(
                    "Ignoring max_epochs, batch_size, learning_rate, and training_callbacks because trainer is given.",
                    UserWarning,
                )

        super().__init__(
            datamodule=datamodule,
            model=classifier,
            trainer=trainer,
            input_builder=BasicClassifier.input_builder,
            output_builder=BasicClassifier.output_builder,
            **kwargs,
        )
