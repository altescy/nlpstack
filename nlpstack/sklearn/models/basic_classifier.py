from __future__ import annotations

import functools
import itertools
from typing import Iterator, Mapping, Sequence, cast

import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from nlpstack.data import DataLoader, Dataset, Instance, Vocabulary
from nlpstack.data.fields import Field, LabelField, MappingField, TensorField, TextField
from nlpstack.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nlpstack.torch.models import TorchBasicClassifier
from nlpstack.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.picklable import TorchPicklable
from nlpstack.torch.training import Trainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory
from nlpstack.torch.util import move_to_device


class BasicNeuralTextClassifier(TorchPicklable, BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    cuda_dependent_attributes = ["_classifier"]

    def __init__(
        self,
        *,
        classifier: TorchBasicClassifier | None = None,
        tokenizer: Tokenizer | None = None,
        token_indexers: Mapping[str, TokenIndexer] | None = None,
        multilabel: bool = False,
        min_df: int | float | Mapping[str, int | float] = 1,
        max_df: int | float | Mapping[str, int | float] = 1.0,
        pad_token: str | Mapping[str, str] = "@@PADDING@@",
        oov_token: str | Mapping[str, str] = "@@UNKNOWN@@",
        max_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        label_namespace: str = "labels",
        training_callbacks: Sequence[Callback] | None = None,
        trainer: Trainer | None = None,
        vocab: Vocabulary | None = None,
        do_not_build_vocab: bool = False,
    ) -> None:
        default_token_namespace = "tokens"
        min_df = {default_token_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
        max_df = {default_token_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
        pad_token = {default_token_namespace: pad_token} if isinstance(pad_token, str) else pad_token
        oov_token = {default_token_namespace: oov_token} if isinstance(oov_token, str) else oov_token
        special_tokens: dict[str, set[str]] = {}
        for namespace, token in pad_token.items():
            special_tokens.setdefault(namespace, set()).add(token)
        for namespace, token in oov_token.items():
            special_tokens.setdefault(namespace, set()).add(token)

        super().__init__()
        self._classifier = classifier or TorchBasicClassifier(
            embedder=TextEmbedder({"tokens": Embedding(64)}),
            encoder=BagOfEmbeddings(64),
            multilabel=multilabel,
        )
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._trainer = trainer or Trainer(
            train_dataloader=DataLoader(batch_size=batch_size, shuffle=True),
            valid_dataloader=DataLoader(batch_size=batch_size, shuffle=False),
            max_epochs=max_epochs,
            optimizer_factory=AdamFactory(lr=learning_rate),
            callbacks=training_callbacks,
        )
        self.label_namespace = label_namespace
        self.vocab = vocab or Vocabulary(
            min_df=min_df,
            max_df=max_df,
            pad_token=pad_token,
            oov_token=oov_token,
            special_tokens=special_tokens,
        )
        self._do_not_build_vocab = do_not_build_vocab

    def _tokenize(self, documents: Sequence[str]) -> Sequence[list[Token]]:
        return Dataset.from_iterable(self._tokenizer.tokenize(document) for document in documents)

    def _build_vocab(
        self, tokenized_documents: Sequence[list[Token]], labels: Sequence[str] | Sequence[Sequence[str]]
    ) -> None:
        def label_iterator() -> Iterator[list[str]]:
            for label in labels:
                if isinstance(label, str):
                    yield [label]
                else:
                    yield list(label)

        for token_indexer in self._token_indexers.values():
            token_indexer.build_vocab(self.vocab, tokenized_documents)

        self.vocab.build_vocab_from_documents(self.label_namespace, label_iterator())

    def _text_to_instance(self, text: str | list[Token], label: str | list[str] | None = None) -> Instance:
        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)
        fields: dict[str, Field] = {}
        fields["text"] = MappingField(
            {
                key: TextField(
                    text,
                    indexer=functools.partial(indexer, vocab=self.vocab),
                    padding_value=indexer.get_pad_index(self.vocab),
                )
                for key, indexer in self._token_indexers.items()
            }
        )
        if label is not None:
            if self._classifier.multilabel:
                assert isinstance(label, list)
                binary_label = numpy.zeros(self.vocab.get_vocab_size(self.label_namespace), dtype=int)
                for single_label in label:
                    binary_label[self.vocab.get_index_by_token(self.label_namespace, single_label)] = 1
                fields["label"] = TensorField(binary_label)
            else:
                assert isinstance(label, str)
                fields["label"] = LabelField(
                    label,
                    vocab=self.vocab.get_token_to_index(self.label_namespace),
                )
        return Instance(**fields)

    def fit(
        self,
        X: Sequence[str],
        y: Sequence[str] | Sequence[Sequence[str]],
        *,
        X_valid: Sequence[str] | None = None,
        y_valid: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> BasicNeuralTextClassifier:
        train_documents = X
        train_labels = y
        valid_documents = X_valid
        valid_labels = y_valid

        tokenized_documents = self._tokenize(train_documents)

        if not self._do_not_build_vocab:
            self._build_vocab(tokenized_documents, train_labels)

        self._classifier.setup(vocab=self.vocab)

        train_dataset = Dataset.from_iterable(
            itertools.starmap(self._text_to_instance, zip(tokenized_documents, train_labels))
        )
        valid_dataset: Dataset | None = None
        if valid_documents is not None and valid_labels is not None:
            tokenized_valid_documents = self._tokenize(valid_documents)
            valid_dataset = Dataset.from_iterable(
                itertools.starmap(self._text_to_instance, zip(tokenized_valid_documents, valid_labels))
            )

        self._trainer.train(self._classifier, train_dataset, valid_dataset)
        return self

    def _predict_multilabel(
        self,
        X: Sequence[str],
        *,
        batch_size: int,
        return_labels: bool,
        threshold: float,
    ) -> numpy.ndarray | list[list[str]]:
        preds = self.predict_proba(X, batch_size=batch_size)
        binary_labels = (preds >= threshold).astype(int)
        if return_labels:
            return [
                [self.vocab.get_token_by_index(self.label_namespace, index) for index, label in enumerate(row) if label]
                for row in binary_labels
            ]
        return binary_labels

    def _predict_singlelabel(
        self,
        X: Sequence[str],
        *,
        batch_size: int,
        return_labels: bool,
    ) -> numpy.ndarray | list[str]:
        preds = cast(numpy.ndarray, self.predict_proba(X, batch_size=batch_size).argmax(axis=1))
        if return_labels:
            return [self.vocab.get_token_by_index(self.label_namespace, pred) for pred in preds.tolist()]
        return preds

    def predict(
        self,
        X: Sequence[str],
        *,
        batch_size: int = 64,
        return_labels: bool = False,
        threshold: float = 0.5,
    ) -> numpy.ndarray | list[str] | list[list[str]]:
        if self._classifier.multilabel:
            return self._predict_multilabel(X, batch_size=batch_size, return_labels=return_labels, threshold=threshold)
        return self._predict_singlelabel(X, batch_size=batch_size, return_labels=return_labels)

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

    def score(
        self,
        X: Sequence[str],
        y: Sequence[str] | Sequence[Sequence[str]],
        *,
        metric: str | None = None,
        batch_size: int = 64,
    ) -> float:
        metric = metric or ("accuracy" if not self._classifier.multilabel else "overall_accuracy")
        return self.compute_scores(X, y, batch_size=batch_size)[metric]

    def compute_scores(
        self,
        X: Sequence[str],
        y: Sequence[str] | Sequence[Sequence[str]],
        *,
        batch_size: int = 64,
    ) -> dict[str, float]:
        device = self._classifier.get_device()

        tokenized_documents = self._tokenize(X)
        dataset = Dataset.from_iterable(itertools.starmap(self._text_to_instance, zip(tokenized_documents, y)))

        self._classifier.eval()
        self._classifier.get_metrics(reset=True)
        with torch.no_grad():
            for batch in DataLoader(batch_size=batch_size)(dataset):
                batch = move_to_device(batch, device)
                _ = self._classifier(**batch)

        return self._classifier.get_metrics(reset=True)
