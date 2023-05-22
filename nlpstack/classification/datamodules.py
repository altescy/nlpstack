from __future__ import annotations

import dataclasses
import functools
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy

from nlpstack.data import DataModule, Dataset, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, LabelField, MappingField, TensorField, TextField
from nlpstack.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer


@dataclasses.dataclass
class ClassificationExample:
    text: str | Sequence[Token]
    label: str | Sequence[str] | None = None


class BasicClassificationDataModule(DataModule[ClassificationExample]):
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

        def generator() -> Iterator[ClassificationExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield ClassificationExample(
                    text=tokenized_text,
                    label=example.label,
                )

        return Dataset.from_iterable(generator())

    def _build_vocab(self, dataset: Sequence[ClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[list[str]]:
            for example in dataset:
                label = example.label
                assert label is not None, "Dataset must have labels."
                if isinstance(label, str):
                    yield [label]
                else:
                    yield list(label)

        for token_indexer in self._token_indexers.values():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: ClassificationExample) -> Instance:
        text = example.text
        label = example.label

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
            if isinstance(label, str):
                fields["label"] = LabelField(
                    label,
                    vocab=self.vocab.get_token_to_index(self.label_namespace),
                )
            else:
                binary_label = numpy.zeros(self.vocab.get_vocab_size(self.label_namespace), dtype=int)
                for single_label in label:
                    binary_label[self.vocab.get_index_by_token(self.label_namespace, single_label)] = 1
                fields["label"] = TensorField(binary_label)
        return Instance(**fields)

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
