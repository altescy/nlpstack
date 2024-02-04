from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.evaluation import Metric
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback
from nlpstack.sklearn.rune import SklearnEstimatorForRune

from .datamodules import TopicModelingDataModule
from .rune import ProdLDA
from .torch import TorchProdLDA
from .types import TopicModelingExample, TopicModelingInference, TopicModelingPrediction

ProdLDAInputsX = Sequence[str]
ProdLDAInputsY = Sequence[str]
ProdLDAOutputs = Sequence[int]


class SklearnProdLDA(
    SklearnEstimatorForRune[
        ProdLDAInputsX,
        ProdLDAInputsY,
        ProdLDAOutputs,
        TopicModelingExample,
        TopicModelingPrediction,
    ]
):
    primary_metric = "perplexity"

    @staticmethod
    def input_builder(X: ProdLDAInputsX, y: Optional[ProdLDAInputsY]) -> Iterator[TopicModelingExample]:
        for text in X:
            yield TopicModelingExample(text=text)

    @staticmethod
    def output_builder(predictions: Iterator[TopicModelingPrediction]) -> ProdLDAOutputs:
        return [pred.topic for pred in predictions]

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
        datamodule: Optional[TopicModelingDataModule] = None,
        # model configuration
        num_topics: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        model: Optional[TorchProdLDA] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[Metric[TopicModelingInference], Sequence[Metric[TopicModelingInference]]]] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        rune = ProdLDA(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            vocab=vocab,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            num_topics=num_topics,
            hidden_dim=hidden_dim,
            dropout=dropout,
            model=model,
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
