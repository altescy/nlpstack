import warnings
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import BasicBatchSampler, DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.integrations.torch.generation import BeamSearch
from nlpstack.integrations.torch.modules.seq2seq_decoders import LstmSeq2SeqDecoder
from nlpstack.integrations.torch.modules.token_embedders import Embedding
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback
from nlpstack.integrations.torch.training.optimizers import AdamFactory
from nlpstack.rune import RuneForTorch, SetupMode

from .datamodules import CausalLanguageModelingDataModule
from .metrics import CausalLanguageModelingMetric, Perplexity
from .torch import TorchCausalLanguageModel
from .types import CausalLanguageModelingExample, CausalLanguageModelingInference, CausalLanguageModelingPrediction

logger = getLogger(__name__)


class CausalLanguageModel(
    RuneForTorch[
        CausalLanguageModelingExample,
        CausalLanguageModelingInference,
        CausalLanguageModelingPrediction,
    ]
):
    """
    A causal language model.

    Args:
        min_df: The minimum document frequency of the tokens. If `float`, the minimum
            document frequency is the fraction of the total number of documents. Defaults to `1`.
        max_df: The maximum document frequency of the tokens. If `float`, the maximum
            document frequency is the fraction of the total number of documents. Defaults to `1.0`.
        pad_token: The padding token. You can specify a different padding token for each
            namespace by passing a mapping from namespace to padding token. Defaults to `"@@PADDING@@"`.
        oov_token: The out-of-vocabulary (OOV) token. You can specify a different OOV token
            for each namespace by passing a mapping from namespace to OOV token. Defaults to `"@@UNKNOWN@@"`.
        bos_token: The beginning-of-sentence (BOS) token. You can specify a different BOS token
            for each namespace by passing a mapping from namespace to BOS token. If BOS token
            is given, the BOS token will be prepended to the beginning of each sequence. Defaults to `{}`.
        eos_token: The end-of-sentence (EOS) token. You can specify a different EOS token
            for each namespace by passing a mapping from namespace to EOS token. If EOS token
            is given, the EOS token will be appended to the end of each sequence. Defaults to `{}`.
        vocab: The vocabulary. If given, the vocabulary-related arguments will be ignored, otherwise
            the vocabulary will be constructed from the data. Defaults to `None`.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens.
        datamodule: The data module. If given, the data module related arguments will be ignored,
            otherwise the data module will be constructed from the data. Defaults to `None`.
        dropout: The dropout rate.
        beam_search: The beam search module.
        model: The causal language model for PyTorch. If given, the model related arguments will be ignored,
            otherwise the model will be constructed from the data. Defaults to `None`.
        max_epochs: The maximum number of epochs to train the model. Defaults to `4`.
        batch_size: The batch size. Defaults to `32`.
        learning_rate: The learning rate. Defaults to `1e-3`.
        training_callbacks: The training callbacks for `TorchTrainer`. Defaults to `None`.
        trainer: The trainer for training the model. If given, the trainer related arguments will be ignored,
            otherwise the trainer will be constructed from the related arguments. Defaults to `None`.
        metric: The metric for evaluation. Defaults to `Perplexity()`.
        random_seed: The random seed. Defaults to `None`.
    """

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
        beam_search: Optional[BeamSearch] = None,
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
        if datamodule is None:
            if vocab is None:
                default_token_namespace = "tokens"
                min_df = {default_token_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
                max_df = {default_token_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
                pad_token = {default_token_namespace: pad_token} if isinstance(pad_token, str) else pad_token
                oov_token = {default_token_namespace: oov_token} if isinstance(oov_token, str) else oov_token
                bos_token = {default_token_namespace: bos_token} if isinstance(bos_token, str) else bos_token
                eos_token = {default_token_namespace: eos_token} if isinstance(eos_token, str) else eos_token
                special_tokens: Dict[str, Set[str]] = {}
                for namespace, token in pad_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                for namespace, token in oov_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                for namespace, token in bos_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                for namespace, token in eos_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                vocab = Vocabulary(
                    min_df=min_df,
                    max_df=max_df,
                    pad_token=pad_token,
                    oov_token=oov_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    special_tokens=special_tokens,
                )
            else:
                if (min_df, max_df, pad_token, oov_token, bos_token, eos_token) != (
                    1,
                    1.0,
                    "@@PADDING@@",
                    "@@UNKNOWN@@",
                    {},
                    {},
                ):
                    warnings.warn(
                        "Ignoring min_df, max_df, pad_token, oov_token, bos_token and eos_token because vocab is given.",
                        UserWarning,
                    )

            if tokenizer is None:
                tokenizer = WhitespaceTokenizer()

            if token_indexers is None:
                token_indexers = {"tokens": SingleIdTokenIndexer()}

            datamodule = CausalLanguageModelingDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if model is None:
            model = TorchCausalLanguageModel(
                embedder=Embedding(128),
                decoder=LstmSeq2SeqDecoder(128, 128, 2),
                dropout=dropout,
                beam_search=beam_search,
            )
        else:
            if (dropout, beam_search) != (None, None):
                warnings.warn("Ignoring dropout and beam_search because model is given.", UserWarning)

        if trainer is None:
            trainer = TorchTrainer(
                train_dataloader=DataLoader(BasicBatchSampler(batch_size=batch_size, shuffle=True)),
                valid_dataloader=DataLoader(BasicBatchSampler(batch_size=batch_size, shuffle=False)),
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

        if metric is None:
            metric = Perplexity()

        super().__init__(
            datamodule=datamodule,
            model=model,
            trainer=trainer,
            metric=metric,
            random_seed=random_seed,
            **kwargs,
        )

    def setup(self, mode: SetupMode, **kwargs: Any) -> None:
        super().setup(mode=mode, **kwargs)

        if mode == "training":
            self.datamodule.setup(generation_mode=False)
        elif mode == "evaluation":
            self.datamodule.setup(generation_mode=False)
        elif mode == "prediction":
            self.datamodule.setup(generation_mode=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
