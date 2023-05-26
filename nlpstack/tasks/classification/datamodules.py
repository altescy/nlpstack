from __future__ import annotations

import functools
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy

from nlpstack.data import DataModule, Dataset, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, LabelField, MappingField, MetadataField, TextField
from nlpstack.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .data import ClassificationExample, ClassificationInference, ClassificationPrediction


class BasicClassificationDataModule(
    DataModule[
        ClassificationExample,
        ClassificationInference,
        ClassificationPrediction,
    ]
):
    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Tokenizer | None = None,
        token_indexers: Mapping[str, TokenIndexer] | None = None,
        label_namespace: str = "labels",
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def label_namespace(self) -> str:
        return self._label_namespace

    def _tokenize(self, dataset: Iterable[ClassificationExample]) -> Sequence[ClassificationExample]:
        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[ClassificationExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield ClassificationExample(
                    text=tokenized_text,
                    label=example.label,
                )

        return Dataset.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Sequence[ClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[list[str]]:
            for example in dataset:
                label = example.label
                assert label is not None, "Dataset must have labels."
                yield [label]

        for token_indexer in self._token_indexers.values():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: ClassificationExample) -> Instance:
        text = example.text
        label = example.label
        metadata = example.metadata

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

        if metadata is not None:
            fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(
                label,
                vocab=self.vocab.get_token_to_index(self.label_namespace),
            )

        return Instance(**fields)

    def build_predictions(self, inference: ClassificationInference) -> Iterator[ClassificationPrediction]:
        sorted_indices = inference.probs.argsort(axis=1)[:, ::-1].tolist()
        sorted_probs = inference.probs.take(sorted_indices).tolist()
        for i, (top_indices, top_probs) in enumerate(zip(sorted_indices, sorted_probs)):
            yield ClassificationPrediction(
                top_probs=top_probs,
                top_labels=[self.vocab.get_token_by_index(self.label_namespace, index) for index in top_indices],
                metadata=inference.metadata[i] if inference.metadata is not None else None,
            )

    def build_inference(
        self,
        examples: Sequence[ClassificationExample],
        predictions: Sequence[ClassificationPrediction],
    ) -> ClassificationInference:
        probs: list[list[float]] = []
        labels: list[int] = []
        metadata: list[dict[str, Any]] = []
        for example, prediction in zip(examples, predictions):
            assert example.label is not None
            probs.append(prediction.top_probs)
            labels.append(self.vocab.get_index_by_token(self.label_namespace, example.label))
            metadata.append(example.metadata or {})

        return ClassificationInference(
            probs=numpy.array(probs),
            labels=numpy.array(labels),
            metadata=metadata,
        )

    def read_dataset(
        self,
        dataset: Iterable[ClassificationExample],
        is_training: bool = False,
        **kwargs: Any,
    ) -> Iterator[Instance]:
        if is_training:
            dataset = self._tokenize(dataset)
            self._build_vocab(dataset)

        for example in dataset:
            yield self.build_instance(example)
