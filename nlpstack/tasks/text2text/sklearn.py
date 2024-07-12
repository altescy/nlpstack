import itertools
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.integrations.sklearn.rune import SklearnEstimatorForRune
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback

from .datamodules import Text2TextDataModule
from .metrics import Text2TextMetric
from .rune import Text2Text
from .torch import TorchText2Text
from .types import Text2TextExample, Text2TextPrediction

Text2TextInputsX = Sequence[str]
Text2TextInputsY = Sequence[str]
Text2TextOutputs = Sequence[Sequence[str]]


class SklearnText2Text(
    SklearnEstimatorForRune[
        Text2TextInputsX,
        Text2TextInputsY,
        Text2TextOutputs,
        Text2TextExample,
        Text2TextPrediction,
    ]
):
    primary_metric = "perplexity"

    @staticmethod
    def input_builder(X: Text2TextInputsX, y: Optional[Text2TextInputsY]) -> Iterator[Text2TextExample]:
        for source, target in itertools.zip_longest(X, y or []):
            yield Text2TextExample(source, target)

    @staticmethod
    def output_builder(predictions: Iterator[Text2TextPrediction]) -> Text2TextOutputs:
        return [prediction.tokens for prediction in predictions]

    def __init__(
        self,
        *,
        # data configuration
        min_df: Union[int, float, Mapping[str, Union[int, float]]] = 1,
        max_df: Union[int, float, Mapping[str, Union[int, float]]] = 1.0,
        pad_token: Union[str, Mapping[str, str]] = "@@PADDING@@",
        oov_token: Union[str, Mapping[str, str]] = "@@UNKNOWN@@",
        bos_token: Union[str, Mapping[str, str]] = "@@BEGIN@@",
        eos_token: Union[str, Mapping[str, str]] = "@@END@@",
        vocab: Optional[Vocabulary] = None,
        source_tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        source_token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        target_token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        source_namespace: str = "tokens",
        target_namespace: str = "tokens",
        datamodule: Optional[Text2TextDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        ignore_padding_loss: bool = True,
        model: Optional[TorchText2Text] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[Text2TextMetric, Sequence[Text2TextMetric]]] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        rune = Text2Text(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            bos_token=bos_token,
            eos_token=eos_token,
            vocab=vocab,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_namespace=source_namespace,
            target_namespace=target_namespace,
            datamodule=datamodule,
            dropout=dropout,
            ignore_padding_loss=ignore_padding_loss,
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
        super().__init__(rune=rune)
