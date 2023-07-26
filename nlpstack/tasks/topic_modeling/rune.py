import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union, cast

import numpy

from nlpstack.data import DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.evaluation import Metric
from nlpstack.rune import RuneForTorch
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory

from .datamodules import TopicModelingDataModule
from .metrics import Perplexity
from .torch import TorchProdLDA
from .types import TopicModelingExample, TopicModelingInference, TopicModelingPrediction


class ProdLDA(
    RuneForTorch[
        TopicModelingExample,
        TopicModelingInference,
        TopicModelingPrediction,
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

            datamodule = TopicModelingDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if model is None:
            model = TorchProdLDA(
                num_topics=num_topics or 16,
                hidden_dim=hidden_dim or 128,
                dropout=dropout,
            )
        else:
            if (num_topics, hidden_dim, dropout) != (None, None, None):
                warnings.warn(
                    "Ignoring dropout and class_weights because classifier is given.",
                    UserWarning,
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
            metric = Perplexity()

        super().__init__(
            datamodule=datamodule,
            model=model,
            trainer=trainer,
            metric=metric,
            random_seed=random_seed,
            **kwargs,
        )

    def get_topics(self) -> numpy.ndarray:
        assert isinstance(self.model, TorchProdLDA)
        return cast(numpy.ndarray, self.model.get_topics().numpy())

    def get_topic_terms(self, topic_id: int, top_n: int = 10) -> List[str]:
        assert isinstance(self.model, TorchProdLDA)
        assert isinstance(self.datamodule, TopicModelingDataModule)
        topic_word_distribution = self.get_topics()[topic_id]
        top_word_indices = numpy.argsort(topic_word_distribution)[::-1][:top_n]
        return [
            self.datamodule.vocab.get_token_by_index(self.model._token_namespace, index) for index in top_word_indices
        ]
