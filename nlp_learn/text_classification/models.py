from __future__ import annotations

import functools
import itertools
from typing import Iterator, Mapping, Sequence, cast

import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from nlp_learn.data import DataLoader, Dataset, Instance, Vocabulary
from nlp_learn.data.fields import Field, LabelField, MappingField, TextField
from nlp_learn.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlp_learn.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nlp_learn.torch.models.text_classifier import TorchTextClassifier
from nlp_learn.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlp_learn.torch.modules.text_embedders import TextEmbedder
from nlp_learn.torch.modules.token_embedders import Embedding
from nlp_learn.torch.training import Trainer
from nlp_learn.torch.training.optimizers import AdamFactory
from nlp_learn.torch.util import move_to_device


class BasicNeuralTextClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    def __init__(
        self,
        *,
        classifier: TorchTextClassifier | None = None,
        tokenizer: Tokenizer | None = None,
        token_indexers: Mapping[str, TokenIndexer] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        pad_token: str = "@@PADDING@@",
        oov_token: str = "@@UNKNOWN@@",
        max_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        trainer: Trainer | None = None,
    ) -> None:
        super().__init__()
        self._classifier = classifier or TorchTextClassifier(
            embedder=TextEmbedder({"tokens": Embedding(64)}),
            encoder=BagOfEmbeddings(64),
        )
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._trainer = trainer or Trainer(
            train_dataloader=DataLoader(batch_size=batch_size, shuffle=True),
            valid_dataloader=DataLoader(batch_size=batch_size, shuffle=False),
            max_epochs=max_epochs,
            optimizer_factory=AdamFactory(lr=learning_rate),
        )
        self.token_namespace = "tokens"
        self.label_namespace = "labels"
        self.vocab = Vocabulary(
            min_df={self.token_namespace: min_df},
            max_df={self.token_namespace: max_df},
            pad_token={self.token_namespace: pad_token},
            oov_token={self.token_namespace: oov_token},
            special_tokens={self.token_namespace: {pad_token, oov_token}},
        )

    def _tokenize(self, documents: Sequence[str]) -> Sequence[list[Token]]:
        return Dataset.from_iterable(self._tokenizer.tokenize(document) for document in documents)

    def _build_vocab(self, tokenized_documents: Sequence[list[Token]], labels: Sequence[str]) -> None:
        def token_iterator() -> Iterator[list[str]]:
            for tokens in tokenized_documents:
                yield [token.surface for token in tokens]

        def label_iterator() -> Iterator[list[str]]:
            for label in labels:
                yield [label]

        self.vocab.build_vocab_from_documents(self.token_namespace, token_iterator())
        self.vocab.build_vocab_from_documents(self.label_namespace, label_iterator())

    def _text_to_instance(self, text: str | list[Token], label: str | None = None) -> Instance:
        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)
        fields: dict[str, Field] = {}
        fields["text"] = MappingField(
            {
                key: TextField(text, indexer=functools.partial(indexer, vocab=self.vocab))
                for key, indexer in self._token_indexers.items()
            }
        )
        if label is not None:
            fields["label"] = LabelField(
                label,
                vocab=self.vocab.get_token_to_index(self.label_namespace),
            )
        return Instance(**fields)

    def fit(self, X: Sequence[str], y: Sequence[str]) -> BasicNeuralTextClassifier:
        train_documents = X
        train_labels = y

        tokenized_documents = self._tokenize(train_documents)
        self._build_vocab(tokenized_documents, train_labels)

        self._classifier.setup(vocab=self.vocab)

        train_dataset = Dataset.from_iterable(
            itertools.starmap(self._text_to_instance, zip(tokenized_documents, train_labels))
        )

        self._trainer.train(self._classifier, train_dataset)
        return self

    def predict(
        self,
        X: Sequence[str],
        *,
        batch_size: int = 64,
        return_labels: bool = True,
    ) -> numpy.ndarray | list[str]:
        preds = cast(numpy.ndarray, self.predict_proba(X, batch_size=batch_size).argmax(axis=1))
        if return_labels:
            return [self.vocab.get_token_by_index(self.label_namespace, pred) for pred in preds.tolist()]
        return preds

    def predict_proba(
        self,
        X: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> numpy.ndarray:
        tokenized_documents = self._tokenize(X)
        dataset = Dataset.from_iterable(self._text_to_instance(text) for text in tokenized_documents)
        probs: list[numpy.ndarray] = []
        device = self._classifier.get_device()
        self._classifier.eval()
        with torch.no_grad():
            for batch in DataLoader(batch_size=batch_size)(dataset):
                batch = move_to_device(batch, device)
                output = self._classifier(**batch)
                probs.append(output["probs"].detach().numpy())
            return numpy.concatenate(probs)
