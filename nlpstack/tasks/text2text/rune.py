import warnings
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import BasicBatchSampler, DataLoader, Vocabulary
from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Tokenizer
from nlpstack.integrations.torch.modules.seq2seq_decoders import LstmSeq2SeqDecoder
from nlpstack.integrations.torch.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from nlpstack.integrations.torch.modules.seq2vec_encoders import TokenPooler
from nlpstack.integrations.torch.modules.token_embedders import Embedding
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback
from nlpstack.integrations.torch.training.optimizers import AdamFactory
from nlpstack.rune import RuneForTorch

from .datamodules import Text2TextDataModule
from .metrics import Perplexity, Text2TextMetric
from .torch import TorchText2Text
from .types import Text2TextExample, Text2TextInference, Text2TextPrediction

logger = getLogger(__name__)


class Text2Text(
    RuneForTorch[
        Text2TextExample,
        Text2TextInference,
        Text2TextPrediction,
    ]
):
    """
    A text-to-text model.

    Args:
        min_df: The minimum document frequency of the tokens. If `float`, the minimum
            document frequency is the fraction of the total number of documents. Defaults to `1`.
        max_df: The maximum document frequency of the tokens. If `float`, the maximum
            document frequency is the fraction of the total number of documents. Defaults to `1.0`.
        pad_token: The padding token. You can specify a different padding token for each
            namespace by passing a mapping from namespace to padding token. Defaults to `"@@PADDING@@"`.
        oov_token: The out-of-vocabulary (OOV) token. You can specify a different OOV token
            for each namespace by passing a mapping from namespace to OOV token. Defaults to `"@@UNKNOWN@@"`.
        bos_token: The beginning-of-sentence (BOS) token. If given, the BOS token is added into the
            beginning of the text. Defaults to `"@@BEGIN@@"`.
        eos_token: The end-of-sentence (EOS) token. If given, the EOS token is added into the
            end of the text. Defaults to `"@@END@@"`.
        vocab: The vocabulary. If given, the vocabulary-related arguments will be ignored, otherwise
            the vocabulary will be constructed from the data. Defaults to `None`.
        source_tokenizer: The tokenizer for source text.
        target_tokenizer: The tokenizer for target text.
        source_token_indexers: The token indexers to index the source tokens.
        target_token_indexers: The token indexers to index the target tokens.
        source_namespace: The vocabulary namespace of source tokens.
        target_namespace: The vocabulary namespace of target tokens.
        datamodule: The data module. If given, the data module related arguments will be ignored,
        dropout: The dropout rate. Defaults to `None`.
        ignore_padding_loss: If `True`, the padding token loss is ignored. If `False`, EOS token is
            needed to stop generation. Defaults to `False`.
        model: The text-to-text model. If given, the model-related arguments will be ignored.
        max_epochs: The maximum number of epochs. Defaults to `4`.
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
        ignore_padding_loss: bool = False,
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
        if datamodule is None:
            if vocab is None:
                min_df = (
                    {source_namespace: min_df, target_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
                )
                max_df = (
                    {source_namespace: max_df, target_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
                )
                pad_token = (
                    {source_namespace: pad_token, target_namespace: pad_token}
                    if isinstance(pad_token, str)
                    else pad_token
                )
                oov_token = (
                    {source_namespace: oov_token, target_namespace: oov_token}
                    if isinstance(oov_token, str)
                    else oov_token
                )
                bos_token = (
                    {source_namespace: bos_token, target_namespace: bos_token}
                    if isinstance(bos_token, str)
                    else bos_token
                )
                eos_token = (
                    {source_namespace: eos_token, target_namespace: eos_token}
                    if isinstance(eos_token, str)
                    else eos_token
                )
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

            datamodule = Text2TextDataModule(
                vocab=vocab,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                source_token_indexers=source_token_indexers,
                target_token_indexers=target_token_indexers,
                source_namespace=source_namespace,
                target_namespace=target_namespace,
            )

        if model is None:
            model = TorchText2Text(
                source_embedder=Embedding(128),
                target_embedder=Embedding(128) if source_namespace != target_namespace else None,
                encoder=LstmSeq2SeqEncoder(128, 64, 1, bidirectional=True),
                decoder=LstmSeq2SeqDecoder(
                    128, 128, 1, use_cross_attention=True, initial_state_encoder=TokenPooler(128, (0, -1))
                ),
                dropout=dropout,
                ignore_padding_loss=ignore_padding_loss,
            )
        else:
            if (dropout, ignore_padding_loss) != (None, True):
                warnings.warn("Ignoring dropout because model is given.", UserWarning)

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
