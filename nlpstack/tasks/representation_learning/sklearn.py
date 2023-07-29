from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.sklearn.rune import SklearnEstimatorForRune
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback

from .datamodules import RepresentationLearningDataModule
from .rune import UnsupervisedSimCSE
from .torch import TorchUnsupervisedSimCSE
from .types import RepresentationLearningExample, RepresentationLearningPrediction

InputX = Sequence[str]
InputY = Sequence[str]
Output = numpy.ndarray


class SklearnUnsupervisedSimCSE(
    SklearnEstimatorForRune[
        InputX,
        InputY,
        Output,
        RepresentationLearningExample,
        RepresentationLearningPrediction,
    ]
):
    @staticmethod
    def input_builder(X: InputX, y: Optional[InputY] = None) -> Iterator[RepresentationLearningExample]:
        for text in X:
            yield RepresentationLearningExample(text=text)

    @staticmethod
    def output_builder(predictions: Iterator[RepresentationLearningPrediction]) -> Output:
        return numpy.array([prediction.embedding for prediction in predictions])

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
        datamodule: Optional[RepresentationLearningDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        temperature: Optional[float] = None,
        use_feedforward_for_only_training: Optional[bool] = None,
        model: Optional[TorchUnsupervisedSimCSE] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        rune = UnsupervisedSimCSE(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            vocab=vocab,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            datamodule=datamodule,
            dropout=dropout,
            temperature=temperature,
            use_feedforward_for_only_training=use_feedforward_for_only_training,
            model=model,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_callbacks=training_callbacks,
            trainer=trainer,
            random_seed=random_seed,
            **kwargs,
        )
        super().__init__(rune)

    def transform(self, X: InputX, y: Optional[InputY] = None) -> Output:
        return self.predict(X)

    def fit_transform(self, X: InputX, y: Optional[InputY] = None) -> Output:
        self.fit(X, y)
        return self.transform(X, y)
