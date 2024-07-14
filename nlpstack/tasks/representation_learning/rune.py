import warnings
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import BasicBatchSampler, DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.integrations.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.integrations.torch.modules.text_embedders import TextEmbedder
from nlpstack.integrations.torch.modules.token_embedders import Embedding
from nlpstack.integrations.torch.rune import RuneForTorch
from nlpstack.integrations.torch.training import TorchTrainer
from nlpstack.integrations.torch.training.callbacks import Callback
from nlpstack.integrations.torch.training.optimizers import AdamFactory

from .datamodules import RepresentationLearningDataModule
from .torch import TorchUnsupervisedSimCSE
from .types import RepresentationLearningExample, RepresentationLearningInference, RepresentationLearningPrediction

logger = getLogger(__name__)


class UnsupervisedSimCSE(
    RuneForTorch[
        RepresentationLearningExample,
        RepresentationLearningInference,
        RepresentationLearningPrediction,
        TorchUnsupervisedSimCSE.Params,
    ]
):
    """
    An unsupervised SimCSE model.

    Args:
        min_df: The minimum document frequency of the tokens. If `float`, the minimum
            document frequency is the fraction of the total number of documents. Defaults to `1`.
        max_df: The maximum document frequency of the tokens. If `float`, the maximum
            document frequency is the fraction of the total number of documents. Defaults to `1.0`.
        pad_token: The padding token. You can specify a different padding token for each
            namespace by passing a mapping from namespace to padding token. Defaults to `"@@PADDING@@"`.
        oov_token: The out-of-vocabulary (OOV) token. You can specify a different OOV token
            for each namespace by passing a mapping from namespace to OOV token. Defaults to `"@@UNKNOWN@@"`.
        vocab: The vocabulary. If given, the vocabulary-related arguments will be ignored, otherwise
            the vocabulary will be constructed from the data. Defaults to `None`.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens.
        datamodule: The data module. If given, the data module related arguments will be ignored,
        dropout: The dropout rate. Defaults to `None`.
        temperature: The temperature parameter for SimCSE training. Defaults to `0.05`.
        model: The unsupervised SimCSE model. If given, the model related arguments will be ignored.
        max_epochs: The maximum number of epochs. Defaults to `4`.
        batch_size: The batch size. Defaults to `32`.
        learning_rate: The learning rate. Defaults to `1e-3`.
        training_callbacks: The training callbacks for `TorchTrainer`. Defaults to `None`.
        trainer: The trainer for training the model. If given, the trainer related arguments will be ignored,
            otherwise the trainer will be constructed from the related arguments. Defaults to `None`.
        random_seed: The random seed. Defaults to `None`.
    """

    Example = RepresentationLearningExample
    Prediction = RepresentationLearningPrediction
    PredictionParams = TorchUnsupervisedSimCSE.Params
    EvaluationParams = TorchUnsupervisedSimCSE.Params

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
        if datamodule is None:
            if vocab is None:
                default_token_namespace = "tokens"
                min_df = {default_token_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
                max_df = {default_token_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
                pad_token = {default_token_namespace: pad_token} if isinstance(pad_token, str) else pad_token
                oov_token = {default_token_namespace: oov_token} if isinstance(oov_token, str) else oov_token
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

            if tokenizer is None:
                tokenizer = WhitespaceTokenizer()

            if token_indexers is None:
                token_indexers = {"tokens": SingleIdTokenIndexer()}

            datamodule = RepresentationLearningDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if model is None:
            classifier = TorchUnsupervisedSimCSE(
                embedder=TextEmbedder({"tokens": Embedding(64)}),
                encoder=BagOfEmbeddings(64),
                dropout=dropout,
                temperature=temperature,
                use_feedforward_for_only_training=use_feedforward_for_only_training,
            )
        else:
            if (dropout, temperature, use_feedforward_for_only_training) != (None, None, None):
                warnings.warn(
                    "Ignoring dropout and temperature because classifier is given.",
                    UserWarning,
                )

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

        super().__init__(
            datamodule=datamodule,
            model=classifier,
            trainer=trainer,
            random_seed=random_seed,
            **kwargs,
        )
