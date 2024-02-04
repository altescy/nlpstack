import warnings
from logging import getLogger
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import BasicBatchSampler, DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.integrations.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.integrations.torch.modules.text_embedders import TextEmbedder
from nlpstack.integrations.torch.modules.token_embedders import Embedding
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback
from nlpstack.integrations.torch.training.optimizers import AdamFactory
from nlpstack.rune import RuneForTorch

from .datamodules import MultilabelClassificationDataModule
from .metrics import MultilabelAccuracy, MultilabelClassificationMetric
from .torch import TorchMultilabelClassifier
from .types import (
    MultilabelClassificationExample,
    MultilabelClassificationInference,
    MultilabelClassificationPrediction,
)

logger = getLogger(__name__)


class MultilabelClassifier(
    RuneForTorch[
        MultilabelClassificationExample,
        MultilabelClassificationInference,
        MultilabelClassificationPrediction,
    ]
):
    """
    A basic neural multilabel text classification model.

    Args:
        min_df: The minimum document frequency of the tokens. If `float`, the minimum
            document frequency is the fraction of the total number of documents. Defaults to `1`.
        max_df: The maximum document frequency of the tokens. If `float`, the maximum
            document frequency is the fraction of the total number of documents. Defaults to `1.0`.
        pad_token: The padding token. You can specify a different padding token for each
            namespace by passing a mapping from namespace to padding token. Defaults to `"@@PADDING@@"`.
        oov_token: The out-of-vocabulary (OOV) token. You can specify a different OOV token
            for each namespace by passing a mapping from namespace to OOV token. Defaults to `"@@UNKNOWN@@"`.
        labels: The set of labels. If not given, the labels will be collected from the training dataset.
            Defaults to `None`.
        vocab: The vocabulary. If given, the vocabulary-related arguments will be ignored, otherwise
            the vocabulary will be constructed from the data. Defaults to `None`.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens.
        datamodule: The data module. If given, the data module related arguments will be ignored,
        dropout: The dropout rate. Defaults to `None`.
        class_weights: The class weights. If `"balanced"`, the class weights will be set to
            the inverse of the class frequencies. You can specify a different class weight for
            each class by passing a mapping from class to class weight. Defaults to `None`.
        classifier: The classifier. If given, the model related arguments will be ignored.
        max_epochs: The maximum number of epochs. Defaults to `4`.
        batch_size: The batch size. Defaults to `32`.
        learning_rate: The learning rate. Defaults to `1e-3`.
        training_callbacks: The training callbacks for `TorchTrainer`. Defaults to `None`.
        trainer: The trainer for training the model. If given, the trainer related arguments will be ignored,
            otherwise the trainer will be constructed from the related arguments. Defaults to `None`.
        metric: The metric for evaluation. Defaults to `MultilabelAccuracy()`.
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
        labels: Optional[Sequence[str]] = None,
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
        # other configuration
        random_seed: Optional[int] = None,
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
                labels=labels,
            )
        else:
            if (vocab, tokenizer, token_indexers, labels) != (None, None, None, None):
                warnings.warn(
                    "Ignoring vocab, tokenizer, token_indexers, and labels because datamodule is given.",
                    UserWarning,
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
            metric = MultilabelAccuracy()

        super().__init__(
            datamodule=datamodule,
            model=classifier,
            trainer=trainer,
            metric=metric,
            random_seed=random_seed,
            **kwargs,
        )
