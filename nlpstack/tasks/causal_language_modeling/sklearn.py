from logging import getLogger
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from nlpstack.data import Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.integrations.sklearn.rune import SklearnEstimatorForRune
from nlpstack.integrations.torch.rune import RuneForTorch
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback

from .datamodules import CausalLanguageModelingDataModule
from .metrics import CausalLanguageModelingMetric
from .rune import CausalLanguageModel
from .torch import TorchCausalLanguageModel
from .types import CausalLanguageModelingExample, CausalLanguageModelingPrediction

logger = getLogger(__name__)


InputsX = Sequence[str]
InputsY = Sequence[str]
Outputs = Sequence[str]


class SklearnCausalLanguageModel(
    SklearnEstimatorForRune[
        InputsX,
        InputsY,
        Outputs,
        CausalLanguageModelingExample,
        CausalLanguageModelingPrediction,
        RuneForTorch.SetupParams,
        TorchCausalLanguageModel.Params,
        TorchCausalLanguageModel.Params,
    ]
):
    primary_metric = "perplexity"

    @staticmethod
    def input_builder(X: InputsX, y: Optional[InputsY]) -> Iterator[CausalLanguageModelingExample]:
        for text in X:
            yield CausalLanguageModelingExample(text)

    @staticmethod
    def output_builder(predictions: Iterator[CausalLanguageModelingPrediction]) -> Outputs:
        return [prediction.text for prediction in predictions]

    def __init__(
        self,
        *,
        # data configuration
        min_df: Union[int, float, Mapping[str, Union[int, float]]] = 1,
        max_df: Union[int, float, Mapping[str, Union[int, float]]] = 1.0,
        pad_token: Union[str, Mapping[str, str]] = "@@PADDING@@",
        oov_token: Union[str, Mapping[str, str]] = "@@UNKNOWN@@",
        bos_token: Union[str, Mapping[str, str]] = {},
        eos_token: Union[str, Mapping[str, str]] = {},
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        datamodule: Optional[CausalLanguageModelingDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        model: Optional[TorchCausalLanguageModel] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[CausalLanguageModelingMetric, Sequence[CausalLanguageModelingMetric]]] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        rune = CausalLanguageModel(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            bos_token=bos_token,
            eos_token=eos_token,
            vocab=vocab,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
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
        super().__init__(rune=rune)
