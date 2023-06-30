import warnings
from logging import getLogger
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import DataLoader, Vocabulary
from nlpstack.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.rune import RuneForTorch
from nlpstack.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory

from .data import MultilabelClassificationExample, MultilabelClassificationInference, MultilabelClassificationPrediction
from .datamodules import MultilabelClassificationDataModule
from .metrics import MultilabelAccuracy, MultilabelClassificationMetric
from .torch import TorchMultilabelClassifier

logger = getLogger(__name__)


class MultilabelClassifier(
    RuneForTorch[
        MultilabelClassificationExample,
        MultilabelClassificationInference,
        MultilabelClassificationPrediction,
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

            datamodule = MultilabelClassificationDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if classifier is None:
            classifier = TorchMultilabelClassifier(
                embedder=TextEmbedder({"tokens": Embedding(64)}),
                encoder=BagOfEmbeddings(64),
                dropout=dropout,
                class_weights=class_weights,
            )
        else:
            if (dropout, class_weights) != (None, None):
                warnings.warn("Ignoring dropout and class_weights because classifier is given.", UserWarning)

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
            metric = MultilabelAccuracy()

        super().__init__(
            datamodule=datamodule,
            model=classifier,
            trainer=trainer,
            metric=metric,
            **kwargs,
        )
