import warnings
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.rune import RuneForTorch
from nlpstack.torch.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory

from .datamodules import SequenceLabelingDataModule
from .metrics import SequenceLabelingMetric, TokenBasedAccuracy
from .torch import TorchSequenceLabeler
from .types import SequenceLabelingExample, SequenceLabelingInference, SequenceLabelingPrediction

logger = getLogger(__name__)


class BasicSequenceLabeler(
    RuneForTorch[
        SequenceLabelingExample,
        SequenceLabelingInference,
        SequenceLabelingPrediction,
    ]
):
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
        datamodule: Optional[SequenceLabelingDataModule] = None,
        # model configuration
        sequence_labeler: Optional[TorchSequenceLabeler] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[SequenceLabelingMetric, Sequence[SequenceLabelingMetric]]] = None,
        **kwargs: Any,
    ) -> None:
        if datamodule is None:
            if tokenizer is None:
                tokenizer = WhitespaceTokenizer()

            if token_indexers is None:
                token_indexers = {"tokens": SingleIdTokenIndexer()}

            if vocab is None:
                token_namespaces = {
                    namespace
                    for namespace in (indexer.get_vocab_namespace() for indexer in token_indexers.values())
                    if namespace
                }
                min_df = (
                    {namespace: min_df for namespace in token_namespaces}
                    if isinstance(min_df, (int, float))
                    else min_df
                )
                max_df = (
                    {namespace: max_df for namespace in token_namespaces}
                    if isinstance(max_df, (int, float))
                    else max_df
                )
                pad_token = (
                    {namespace: pad_token for namespace in token_namespaces}
                    if isinstance(pad_token, str)
                    else pad_token
                )
                oov_token = (
                    {namespace: oov_token for namespace in token_namespaces}
                    if isinstance(oov_token, str)
                    else oov_token
                )
                special_tokens: Dict[str, Set[str]] = {}
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

            datamodule = SequenceLabelingDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if sequence_labeler is None:
            sequence_labeler = TorchSequenceLabeler(
                embedder=TextEmbedder({"tokens": Embedding(64)}),
                encoder=LstmSeq2SeqEncoder(64, 32, bidirectional=True),
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

        if metric is None:
            metric = TokenBasedAccuracy()

        super().__init__(
            datamodule=datamodule,
            model=sequence_labeler,
            trainer=trainer,
            metric=metric,
            **kwargs,
        )
